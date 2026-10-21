r"""Substrate or free-standing — the physics a 2D model is silent about (wave R4d).

A ``dim_r = 2`` pythtb model carries no out-of-plane coordinate, so it cannot
say what is on the other side of its layer, and the two answers are different
crystals.  **Free-standing** (suspended, or symmetrically encapsulated): the
full layer group applies, ``σ_h`` and everything composed with it included, and
the Hamiltonian selects among them — the detector by in-plane k-action, the
tensor test ``D(g) = O(g) ⊗ S(g)`` inside every ``σ_h`` fiber, each operation
held to its own spin lift -- on the whole solution space where a multiplet
wider than one orbital gives a fiber's two members one block -- and closure
resolution where the survivors do not compose.  **On a substrate**: one side
is different, every operation with ``R[2,2] = −1`` is absent by construction, and the group
is the side-preserving **subgroup** of that layer group — read on the same
point set, no projection, with each operation carrying its own lift in
pythtb's convention.

**The default for a ``dim_r = 2`` model is the substrate**, because it is the
reading that is always well defined: the free-standing lift of a
substrate-bound model need not be unique (a declared content can leave two
readings of the largest order, and the tie rule keeps one), and it is what the
old decorated cell always assumed.
A ``dim_r = 3`` slab is not silent — its geometry is the answer — so its default
is free-standing.

A substrate removes every operation that flips the layer's side and nothing
else, so the two readings are checked against each other on every 2D call by
that statement: the substrate group's operations, in both cosets, are exactly
the free-standing group's with ``R[2,2] = +1``.  A mismatch is the package
disagreeing with itself and raises ``InternalError`` rather than serving either,
and the other reading merely *refusing* is recorded as the outcome and is not a
mismatch; nor is a declaration in one environment's site irreps that the other
has no word for, which says which environment is meant and is not compared.
Every test below names its environment explicitly except the four that ask
what the default is, and every number is measured.
"""

import numpy as np
import pytest

from conftest import patch_everywhere
from pythtb import Lattice, TBModel
from pythtb.models import graphene, kane_mele

import tqc_models
from pythtqc import _sitechars
from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc import symmetry as tqs
from pythtqc.refusal import InternalError, Refusal
from pythtqc import _build
from pythtqc import _descent
from pythtqc import _layers
from pythtqc import _measure
from pythtqc import _realized
from pythtqc import _spacegroup
from pythtqc import _subgroups

FREE = dict(environment="free-standing")
SUB = dict(environment="substrate")

_ULP = np.finfo(float).eps          # one unit of double-precision roundoff, 2.2e-16
_FEW_ULP = 8 * _ULP                 # a few ulp: libm's sin(pi) residue, a canonical phase


def _names(group, inds=None):
    """The unitary operations of ``group`` by name, in the generator frame."""
    return sorted(_sitechars.operation_name(s, group.lattice)
                  for s in group.symmetries
                  if not s.time_reversal and (inds is None or int(s.ind) in set(inds)))


# ---------------------------------------------------------------------------
# the fixtures this module builds for itself
# ---------------------------------------------------------------------------

HEX = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def square_layer():
    """One spinless orbital at ``1a`` of the square lattice, flat."""
    m = TBModel(Lattice(np.eye(2), [[0.0, 0.0]], periodic_dirs=[0, 1]))
    m.set_onsite([0.0])
    m.set_hop(-1.0, 0, 0, [1, 0])
    m.set_hop(-1.0, 0, 0, [0, 1])
    return m


def square_s_pz(mix=0.0):
    """``s`` and ``p_z`` on one square site; ``mix`` is the parity-mixing term.

    The two orbitals are decoupled and differ in bandwidth unless ``mix`` is
    given, when ``⟨s|H|p_z⟩`` on the site couples them — the term a ``σ_h``
    would have to change the sign of.
    """
    m = TBModel(Lattice(np.eye(2), [[0.0, 0.0], [0.0, 0.0]], periodic_dirs=[0, 1]))
    m.set_onsite([0.5, -0.5])
    m.set_hop(-1.0, 0, 0, [1, 0])
    m.set_hop(-1.0, 0, 0, [0, 1])
    m.set_hop(-0.6, 1, 1, [1, 0])
    m.set_hop(-0.6, 1, 1, [0, 1])
    if mix:
        m.set_hop(mix, 0, 1, [0, 0])
    return m


def buckled(model2d, d=0.15, c=30.0):
    """The same layer with its two sublattices displaced to ``±d`` along ``z``.

    A genuine ``dim_r = 3``, ``dim_k = 2`` slab: the orbitals carry real
    out-of-plane coordinates, and ``H(k)`` is untouched by them — a ``dim_k = 2``
    Bloch phase never sees the non-periodic direction.
    """
    lat = np.zeros((3, 3))
    lat[:2, :2] = np.asarray(model2d.lat_vecs, dtype=float)
    lat[2] = [0.0, 0.0, c]
    orb = np.asarray(model2d.orb_vecs, dtype=float)
    orb3 = np.zeros((int(model2d.norb), 3))
    orb3[:, :2] = orb
    orb3[:, 2] = [(d if i % 2 == 0 else -d) / c for i in range(int(model2d.norb))]
    out = TBModel(Lattice(lat, orb3.tolist(), periodic_dirs=[0, 1]),
                  spinful=(int(model2d.nspin) == 2))
    out.set_onsite(model2d.onsite, mode="set")
    for h in model2d.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    list(h.get("lattice_vector", [0, 0])) + [0], mode="add")
    return out


def buckled_graphene(d=0.15, c=30.0):
    """A spinless buckled honeycomb: A at ``+d``, B at ``−d``, real NN hopping."""
    lat = np.zeros((3, 3))
    lat[:2, :2] = HEX
    lat[2] = [0.0, 0.0, c]
    m = TBModel(Lattice(lat, [[1 / 3, 1 / 3, d / c], [2 / 3, 2 / 3, -d / c]],
                        periodic_dirs=[0, 1]))
    m.set_onsite([0.0, 0.0])
    for R in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        m.set_hop(-1.0, 0, 1, R)
    return m


def rewritten(slab, third):
    """The same slab with its non-periodic lattice vector written as ``third``.

    Every orbital keeps its Cartesian position and every hopping its lattice
    vector, so the layer is the same layer in another cell: ``third`` need
    only leave the plane of the two periodic vectors, the first two.
    """
    lat = np.asarray(slab.lat_vecs, dtype=float).copy()
    cart = np.asarray(slab.orb_vecs, dtype=float) @ lat
    lat[2] = np.asarray(third, dtype=float)
    out = TBModel(Lattice(lat.tolist(), (cart @ np.linalg.inv(lat)).tolist(),
                          periodic_dirs=[0, 1]), spinful=(int(slab.nspin) == 2))
    out.set_onsite(slab.onsite, mode="set")
    for h in slab.hoppings:
        vec = h.get("lattice_vector")
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    [0, 0, 0] if vec is None else list(vec), mode="add")
    return out


def jittered(model, size, seed=1):
    """The same model with every orbital centre moved at random by about ``size``, reduced.

    The noise a fitted model's centres carry, every hopping and on-site term
    kept as it was.
    """
    rng = np.random.default_rng(seed)
    orb = np.asarray(model.orb_vecs, dtype=float)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          (orb + size * rng.normal(size=orb.shape)).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    for h in model.hoppings:
        vec = h.get("lattice_vector")
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    [0] * int(model.lattice.dim_r) if vec is None else list(vec),
                    mode="add")
    return out


def fcc_s(t=-1.0):
    """One ``s`` orbital on a face-centred cubic lattice, its twelve nearest neighbours."""
    m = TBModel(Lattice([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
                        [[0.0, 0.0, 0.0]], periodic_dirs=[0, 1, 2]))
    m.set_onsite([0.0])
    for R in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [0, 1, -1], [1, 0, -1]):
        m.set_hop(t, 0, 0, R)
    return m


def raw_axis(model):
    """The same Hamiltonian written for the **opposite** spin quantization axis.

    ``pythtqc.spin``'s convention is spin along ``+z`` in ``(↑, ↓)`` order; a
    source written along ``−z`` differs by ``W A W†`` with ``W = iσ_x`` on every
    block.  ``tqc_models.p41_flat_band`` applies that conversion to the SRSI
    paper's own blocks, so applying it again returns the paper's transcription
    as printed — the model whose site-swapping blocks carry the ``σ_h``
    partner's lift.  It is a unitary conjugation of ``H(k)``, so no band and no
    gap moves.
    """
    return spin_blocks_turned(model, 1j * np.array([[0, 1], [1, 0]], dtype=complex))


def spin_blocks_turned(model, w, lattice=None):
    """Every ``2x2`` block, on-site and hopping, conjugated ``W A W†``.

    With the lattice left as it is, the spin blocks are written in a frame
    turned against the lattice by ``W``; with ``lattice`` the same lattice
    turned by the rotation ``W`` lifts, the whole crystal is turned rigidly.
    """
    def turn(block):
        b = np.asarray(block, dtype=complex)
        return w @ b @ w.conj().T

    out = TBModel(model.lattice if lattice is None else lattice, spinful=True)
    out.set_onsite([turn(b) for b in np.asarray(model.onsite)], mode="set")
    for h in model.hoppings:
        vec = h.get("lattice_vector")
        out.set_hop(turn(h["amplitude"]), h["from_orbital"], h["to_orbital"],
                    [0] * int(model.lattice.dim_r) if vec is None else list(vec),
                    mode="add")
    return out


def aa_bilayer(d=0.15, c=30.0):
    """Two identical square layers at ``±d``, coupled vertically: ``σ_h`` is real."""
    lat = np.zeros((3, 3))
    lat[:2, :2] = np.eye(2)
    lat[2] = [0.0, 0.0, c]
    m = TBModel(Lattice(lat, [[0.0, 0.0, d / c], [0.0, 0.0, -d / c]],
                        periodic_dirs=[0, 1]))
    m.set_onsite([0.0, 0.0])
    for o in (0, 1):
        m.set_hop(-1.0, o, o, [1, 0, 0])
        m.set_hop(-1.0, o, o, [0, 1, 0])
    m.set_hop(-0.4, 0, 1, [0, 0, 0])
    return m


def conjugate_p4mm(t=1.0, delta=0.3):
    """The two-site square cell whose fourfold **fixes** the sites, not swaps them.

    Sites at ``(½,0)`` and ``(0,½)``.  ``C4`` about ``(0,½)`` carries each onto
    itself, so unequal on-site energies leave it alone while they break the
    ``C4`` about the origin, which exchanges them.  The four nearest-neighbour
    bonds are one orbit under it and carry equal amplitudes.
    """
    m = TBModel(Lattice(np.eye(2), [[0.5, 0.0], [0.0, 0.5]], periodic_dirs=[0, 1]))
    m.set_onsite([delta, -delta])
    for R in ([0, 0], [1, 0], [1, -1], [0, -1]):
        m.set_hop(-t, 0, 1, R)
    return m


def p_shell_honeycomb(t_sigma=1.0, t_pi=0.4, t_z=0.3):
    """A ``p`` shell at both honeycomb sites: ``p_z`` π bonds, and ``p_x, p_y`` σ and π ones.

    Orbitals ``(p_z, p_x, p_y)`` at each site, nearest neighbours only: ``t_z``
    between the two ``p_z``, and the Slater-Koster ``t_sigma n_a n_b + t_pi
    (δ_ab − n_a n_b)`` between the in-plane pairs, ``n`` the unit bond vector.
    ``p_z`` is odd under ``σ_h`` and the pair even, so nothing couples them and
    each site holds ``p_z`` and its doublet as two multiplets: ``A1 + E`` of
    ``3m`` on a substrate and ``A2'' + E'`` of ``-6m2`` free-standing, the
    model obeying ``P6/mmm1'`` whole.
    """
    A, B = np.array([1 / 3, 1 / 3]), np.array([2 / 3, 2 / 3])
    m = TBModel(Lattice(HEX.tolist(), [A.tolist()] * 3 + [B.tolist()] * 3,
                        periodic_dirs=[0, 1]))
    m.set_onsite([0.2, 0.0, 0.0, 0.2, 0.0, 0.0])
    for R in ([0, 0], [-1, 0], [0, -1]):
        d = (B + np.asarray(R, dtype=float) - A) @ HEX
        n = d / np.linalg.norm(d)
        t = t_sigma * np.outer(n, n) + t_pi * (np.eye(2) - np.outer(n, n))
        m.set_hop(t_z, 0, 3, R, mode="add")
        for a in range(2):
            for b in range(2):
                if abs(t[a, b]) > 1e-15:
                    m.set_hop(t[a, b], 1 + a, 4 + b, R, mode="add")
    return m


def pxpy_centred(t_sp=0.4, e_s=0.5, t_ss=0.0, s_first=False):
    """The ``p_x, p_y`` honeycomb with an ``s`` orbital at the hexagon centre.

    The centre is ``1a`` of ``P6/mmm``, and the ``s`` is σ-bonded to the six
    ``p`` doublets around it, ``t_sp (n · p)`` with ``n`` the unit vector from
    the centre to each site.  That bond is carried onto itself by every
    operation of the layer, so the model obeys ``P6/mmm1'`` whole, and the
    ``s`` gives the declared content a one-dimensional multiplet the doublets'
    characters are measured against.  ``t_ss`` hops the ``s`` to the six
    centres around it; with ``t_sp = 0`` it is then a band of its own that no
    hopping couples to the doublets.  ``s_first`` lists the ``s`` before the
    doublets, as orbital 0, rather than after them.
    """
    base = tqc_models.pxpy_honeycomb()
    lat = np.asarray(base.lat_vecs, dtype=float)
    pairs = np.asarray(base.orb_vecs).tolist()
    s, off = (0, 1) if s_first else (4, 0)
    m = TBModel(Lattice(lat.tolist(), [[0.0, 0.0]] + pairs if s_first else pairs + [[0.0, 0.0]],
                        periodic_dirs=[0, 1]))
    onsite = [0.0] * 5
    onsite[s] = e_s
    m.set_onsite(onsite, mode="set")
    for h in base.hoppings:
        m.set_hop(h["amplitude"], h["from_orbital"] + off, h["to_orbital"] + off,
                  list(h.get("lattice_vector", [0, 0])), mode="add")
    for site, orbs in (((2 / 3, 2 / 3), (0, 1)), ((1 / 3, 1 / 3), (2, 3))):
        for R in ([0, 0], [-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1]):
            d = (np.asarray(site) + np.asarray(R, dtype=float)) @ lat
            if abs(np.linalg.norm(d) - 1.0 / np.sqrt(3.0)) < 1e-9:
                for n, o in zip(d / np.linalg.norm(d), orbs):
                    if abs(n) > 1e-15 and t_sp:
                        m.set_hop(t_sp * n, s, o + off, R, mode="add")
    if t_ss:
        for R in ([1, 0], [0, 1], [1, -1]):
            m.set_hop(t_ss, s, s, R, mode="add")
    return m


# ---------------------------------------------------------------------------
# 1. the environment: the default, the declaration, and the grammar
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_flat_model_defaults_to_the_substrate_and_a_slab_to_free_standing():
    """``dim_r = 2`` → ``"substrate"``; ``dim_r = 3`` slab → ``"free-standing"``.

    The owner's ruling made into the API: a ``dim_r = 2`` model has no
    out-of-plane coordinate and is silent about its environment, so the reading
    that is *always* well defined is the default — the substrate.  Measured,
    Kane–Mele then reads ``P6mm1' #183.186``, 24 of 24, where wave R4b's default
    read ``#191.234``.  A ``dim_r = 3`` slab is not silent: its own ``z``
    coordinates are the answer, and its default is free-standing —
    ``P-3m11' #164.86``, 24 of 24, on a buckled honeycomb.  Both records say
    which reading they are and that nobody declared it.

    Orbital 0 declared ``A1`` of its ``3m`` site says a substrate is meant: the
    free-standing layer's site group ``-6m2`` has no ``A1``, so the free-standing
    layer is not read with the declaration, and the cross-check records the two
    as not compared rather than comparing the substrate's declared crystal with
    an undeclared one.
    """
    flat = tqg.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0), characters={0: "A1"})
    assert not isinstance(flat, Refusal), str(flat)
    assert flat.environment == "substrate"
    assert flat.environment_declared is False
    assert (flat.group.number_str, flat.group.size) == ("183.186", 24)
    assert len(flat.validated.indices) == 24
    w = flat.cross_check
    assert (w["outcome"], w["agrees"], w["free_standing"]) == ("not compared", None, None)
    assert w["substrate"]["number_str"] == "183.186"
    assert ("on a substrate (the default), not compared with the free-standing layer: the "
            "declaration is written in the substrate's site irreps") in repr(flat)

    slab = tqg.realized_group(buckled_graphene(), cross_check=False)
    assert not isinstance(slab, Refusal), str(slab)
    assert slab.environment == "free-standing"
    assert slab.environment_declared is False
    assert (slab.group.number_str, slab.group.size) == ("164.86", 24)


def test_what_the_call_refuses_before_anything_is_measured():
    """Mistakes in the call, and a model the package does not read, before any measurement.

    An unknown word is a grammar error, not a fact about a crystal.  So is
    asking a fully periodic model what is on the other side of its layer: it has
    no other side.  And a keyword the call does not take is Python's own
    ``TypeError``, however it came to be written.

    A model periodic in fewer than two directions is a molecule or a chain,
    whose symmetry is a point, rod or frieze group, and it is refused
    ``fewer_than_two_periodic_directions`` whatever the dimension of its cell:
    a chain written in one dimension, the same chain along either direction of
    a plane or along one of space, and one orbital, or two sites and a bond,
    with no periodic direction in a plane or in space.  Nothing is measured, so
    no group is named; the count of periodic directions is the value.
    """
    with pytest.raises(ValueError, match="environment must be"):
        tqg.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0), environment="2d")
    with pytest.raises(ValueError, match="no other side"):
        tqg.realized_group(tqc_models.i213_flicker2018(), **SUB)
    with pytest.raises(TypeError, match="unexpected keyword argument 'enviroment'"):
        tqg.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0), enviroment="substrate")

    def model(dim_r, periodic, sites=1):
        orbitals = [[0.0] * dim_r, [0.5] + [0.0] * (dim_r - 1)][:sites]
        m = TBModel(Lattice(np.eye(dim_r).tolist(), orbitals, periodic_dirs=list(periodic)))
        m.set_onsite([0.0, 0.3][:sites])
        if sites == 2:
            m.set_hop(-1.0, 0, 1)
        for d in periodic:
            step = [0] * dim_r
            step[d] = 1
            m.set_hop(-0.5, 0, 0, step)
        return m

    cases = [(1, (0,), 1), (2, (0,), 1), (2, (1,), 1), (3, (0,), 1),
             (2, (), 1), (3, (), 1), (2, (), 2)]
    for dim_r, periodic, sites in cases:
        out = tqg.realized_group(model(dim_r, periodic, sites))
        assert isinstance(out, Refusal), (dim_r, periodic, sites)
        assert out.gate == "fewer_than_two_periodic_directions"
        assert (out.value, out.threshold) == (float(len(periodic)), 2.0)
        assert out.context == {"periodic_dirs": periodic, "dim_r": dim_r}
        assert ("it is a chain or a ribbon, whose symmetry is a rod or frieze group"
                if periodic else
                "it is a molecule, whose symmetry is a point group") in out.reason


# ---------------------------------------------------------------------------
# 2. Kane-Mele: the whole layer group, when the mirror is real
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_free_standing_kane_mele_reads_its_whole_layer_group(km_inversion_free_standing):
    """``P6/mmm1' #191.234``, 48 of 48, from the cell as it stands.

    Declared free-standing, nothing is decorated away: the cell keeps ``σ_h``
    and the twelve operations composed with it, and the Hamiltonian is measured
    to obey all 48.  The site character is the physical orbital — ``p_z``
    resolves to ``A2''``, odd under the mirror — and it is a distinction this
    group can express, where a substrate reading cannot.  This is the answer
    wave R4b served by default; in R4c it is asked for, because a flat model
    cannot tell a reader which layer it is, and the record's repr says which
    reading it is and that it was declared.

    The detector leaves all 48 undetermined -- every class on a flat cell has a
    partner -- and the tensor test refutes none of them, because where the
    horizontal mirror is genuinely unbroken both members of each ``{g, σ_h·g}``
    are intertwiners and the ansatz member exists for both: nothing is left
    undetermined in the representation.  Time reversal is measured, not
    declared: the cell is asked as grey, and the Hamiltonian obeys every one of
    the 24 antiunitary operations, so nothing is said about it.
    """
    r = km_inversion_free_standing
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name, r.group.size) == ("191.234", "P6/mmm1'", 48)
    assert len(r.validated.indices) == 48
    assert r.group.provenance == "realized"
    assert r.environment == "free-standing" and r.environment_declared is True
    assert "; free-standing" in repr(r)
    assert "(the default)" not in repr(r), "this one was declared"
    assert r.descent is None, "the Hamiltonian obeys the whole cell group"
    assert r.decoration["n_atoms"] == 0 and r.decoration["dz"] is None
    assert "free-standing" in r.decoration["recipe"]
    # the declared p_z is A2'' of -6m2, returned on this spinful model as the corep -E2
    assert list(r.rep.characters.irreps) == ["-E2", "-E2"]
    assert len(r.validated.undetermined) == 48
    assert len(r.rep.indices) == 48, "every operation obeyed is represented, free-standing"
    # every antiunitary operation obeyed: the realization is grey and nothing moves
    assert r.time_reversal["outcome"] == "kept"
    assert r.time_reversal["n_broken"] == 0
    assert r.time_reversal["n_obeyed"] == r.time_reversal["n_antiunitary"] == 24
    assert r.beyond_grey is None
    assert "time reversal" not in repr(r), "nothing happened, so nothing is said"


# ---------------------------------------------------------------------------
# 3. the SRSI paper's P41' cell: the owner's case
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_chiral_pattern_lifts_to_its_own_group_and_a_declared_tie_is_the_rules(
        monkeypatch, p41_chiral_substrate):
    """``P41' #75.2``, 8 of 8, in both environments; a declaration's tie by the rule.

    On a substrate, the default for a flat model, the chiral model's fibers
    are collapsed by definition and the paper's own group is served.  The
    cross-check runs the free-standing reading anyway, and it lifts uniquely:
    each operation the detector leaves undetermined is held to its own spin
    lift, so of every ``σ_h`` fiber the member whose lift the Hamiltonian
    carries holds and the other is refuted by tensor structure -- ``S4`` and
    ``S4⁻¹`` among them, whose own lifts are not their partners' ``S(C4)``,
    ``S(C4⁻¹)``.  The free-standing group is ``P41'`` as well, the subgroup
    statement holds, the outcome is ``"agree"``, and nothing ties.  Its centres
    jittered by 1e-5 and its precision declared at 1e-4, the same cell reads
    the same group and labels: the centres are made exact before the tables'
    setting is read, where a setting search on the jittered centres themselves
    finds no match and refused ``operations_not_consistent``.

    A declaration's readings can tie, and the cross-check counts them.  The
    ``p_x, p_y`` honeycomb with an ``s`` at
    the hexagon centre, its sublattices declared ``E'`` and ``E''`` of
    ``-6m2``, is the cell ``P-6m21' #187.210`` before a hopping is read; the
    ``s`` is the one-dimensional multiplet the doublets' characters are read
    against, and ``E''`` holds against the hoppings neither ``σ_h`` nor ``S6``.
    Two readings of the largest order are left, the threefold with the three
    in-plane twofolds and the threefold with the three vertical mirrors, and
    the rule keeps the mirrors, whose six operations all keep the layer's
    side: ``P3m11' #156.50``.  That is what the substrate returns too, where
    ``σ_h`` is gone and the sublattices declared different are still different
    sites, so the two agree, ``tie`` naming the twofolds.  With no hopping
    between the ``s`` and the doublets the doublets are a coupled part of their
    own, read against their own blocks, and the same declaration states the
    same crystal.  The representation settles each part on its own, so that
    crystal is returned whatever the ``s``'s energy and hopping and whether the
    ``s`` is listed before the doublets or after them; undeclared, the model is
    ``P6/mmm1'`` whole, 48 of 48, the substrate agreeing.

    Had the rule kept the twofolds, the free-standing group would hold no
    mirror for the substrate's to match, and the statement would fail for a
    choice the measurement could not make, while it holds for the mirrors'
    reading: that is the outcome ``"ambiguous"``, never the ``InternalError`` a
    contradiction raises.  Reversing the rule's order shows it, the readings
    named and the pair the statement holds for, and leaves the chiral model,
    which has no tie for the rule to settle, as it was.
    """
    r = p41_chiral_substrate            # the session's shared record of that reading
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name, r.group.size) == ("75.2", "P41'", 8)
    assert len(r.validated.indices) == 8
    assert r.environment == "substrate"
    assert r.descent["from"]["number_str"] == "99.164"
    assert r.assignment.wyckoff_spglib == ("c",)
    assert {b["name"] for b in r.descent["dropped"]} >= {
        "m(x)", "m(y)", "m([110])", "m([1-10])",
        "m(x)'", "m(y)'", "m([110])'", "m([1-10])'"}

    w = r.cross_check
    assert w["outcome"] == "agree" and w["agrees"] is True
    assert w["free_standing"]["number_str"] == w["substrate"]["number_str"] == "75.2"
    assert w["refusal"] is None, "nothing over there refused"
    assert w["tie"] == (), "the free-standing lift is unique"
    assert "the free-standing group #75.2 agrees" in repr(r)

    # the same cell with every centre jittered by 1e-5, as a fitted model's can
    # be, its precision declared: the centres are made exact before the setting
    # is read, so the group is named, every label is the exact model's, and the
    # centres come back where the exact model has them
    exact = tqc_models.p41_flat_band(chiral=True)
    fitted = tqg.realized_group(jittered(exact, 1e-5), precision=1e-4)
    assert not isinstance(fitted, Refusal), str(fitted)
    assert (fitted.group.number_str, fitted.group.size) == ("75.2", 8)
    assert fitted.rep.characters.irreps == r.rep.characters.irreps
    assert (tqw.hsp_table(fitted.rep, bands=2).counts
            == tqw.hsp_table(r.rep, bands=2).counts)
    assert fitted.positions["frame"] == "model"
    assert float(np.abs(np.asarray(fitted.model.orb_vecs)[:, :2]
                        - np.asarray(exact.orb_vecs)).max()) <= _FEW_ULP
    assert fitted.cross_check["outcome"] == "agree"

    # the free-standing reading: every fiber's other member refuted by its own lift
    free = tqg.realized_group(tqc_models.p41_flat_band(chiral=True), cross_check=False,
                              **FREE)
    assert not isinstance(free, Refusal), str(free)
    assert (free.group.number_str, free.group.size) == ("75.2", 8)
    assert free.rep.characters.irreps == r.rep.characters.irreps
    step = free.descent["steps"][0]
    assert step["refuted_by"] == {"no symmetry matrix": 32, "tensor structure": 24}
    assert not step.get("tie")
    rotoinversions = {b["name"]: b["refuted_by"] for b in free.descent["dropped"]
                      if b["name"] in ("S4(z)", "S4^-1(z)")}
    assert rotoinversions == {"S4(z)": "tensor structure", "S4^-1(z)": "tensor structure"}

    # a declaration's readings: the vertical mirrors kept, and the substrate agrees
    declared = {(0, 1): "E'", (2, 3): "E''", 4: "s"}
    mirrors = ("E", "C3(z)", "C3^-1(z)", "m([010])", "m([100])", "m(y)")
    twofolds = ("E", "C3(z)", "C3^-1(z)", "C2([2-10])", "C2([1-20])", "C2(x)")
    hc = tqg.realized_group(pxpy_centred(), characters=declared, **FREE)
    assert not isinstance(hc, Refusal), str(hc)
    assert (hc.group.number_str, hc.group.name, hc.group.size) == ("156.50", "P3m11'", 12)
    assert hc.decoration["declared"]["to"]["number_str"] == "187.210"
    d = hc.descent
    assert d["from"]["number_str"] == "187.210" and d["dropped_by"]["declaration"] == 12
    assert d["steps"][0]["declaration"]["names"] == (mirrors, twofolds)
    assert "it holds 6 operations that keep the layer's side (R[2,2] = +1) where the " \
        "next holds 3" in d["note"]
    w = hc.cross_check
    assert w["outcome"] == "agree" and w["substrate"]["number_str"] == "156.50"
    assert {t["name"] for t in w["tie"]} == set(twofolds[3:])
    assert {t["environment"] for t in w["tie"]} == {"free-standing"}
    assert "the one the tie rule keeps" in w["note"]
    apart = tqg.realized_group(pxpy_centred(t_sp=0.0, t_ss=-0.4), characters=declared,
                               **FREE)
    assert not isinstance(apart, Refusal), str(apart)
    assert (apart.group.number_str, apart.descent["from"]["number_str"]) == \
        ("156.50", "187.210"), "the s decoupled from the doublets: the same descent"
    assert apart.descent["steps"][0]["declaration"]["names"] == (mirrors, twofolds)
    assert apart.rep.characters.irreps == hc.rep.characters.irreps
    assert apart.cross_check["outcome"] == "agree"
    for e_s, t_ss, first in ((0.5, -0.4, True), (-0.2, 0.25, False), (-0.2, 0.25, True)):
        words = {(1, 2): "E'", (3, 4): "E''"} if first else {(0, 1): "E'", (2, 3): "E''"}
        moved = tqg.realized_group(pxpy_centred(t_sp=0.0, e_s=e_s, t_ss=t_ss, s_first=first),
                                   typat=[1, 0, 0, 0, 0] if first else [0, 0, 0, 0, 1],
                                   characters=words, **FREE)
        assert not isinstance(moved, Refusal), f"{(e_s, t_ss, first)}: {moved}"
        assert (moved.group.number_str, moved.descent["from"]["number_str"]) == \
            ("156.50", "187.210"), (e_s, t_ss, first)
        assert moved.cross_check["outcome"] == "agree"
        assert moved.cross_check["substrate"]["number_str"] == "156.50"
        assert sorted(moved.rep.characters.irreps) == sorted(hc.rep.characters.irreps)
    bare = tqg.realized_group(pxpy_centred(t_sp=0.0, t_ss=-0.4), **FREE)
    assert not isinstance(bare, Refusal), str(bare)
    assert (bare.group.number_str, len(bare.validated.indices)) == ("191.234", 48)
    assert bare.cross_check["outcome"] == "agree"
    assert bare.cross_check["substrate"]["number_str"] == "183.186"

    # the other choice: the statement fails for it, and that is "ambiguous"
    ranked = _subgroups._tie_ranked
    patch_everywhere(monkeypatch, "_tie_ranked",
                        lambda group, subgroups: ranked(group, subgroups)[::-1])
    other = tqg.realized_group(tqc_models.p41_flat_band(chiral=True))
    assert not isinstance(other, Refusal), str(other)
    assert other.group.number_str == "75.2", "the substrate reading has no tie"
    w = other.cross_check
    assert (w["outcome"], w["free_standing"]["number_str"], w["tie"]) == ("agree", "75.2", ()), \
        "nor has the free-standing one, so the rule's order moves nothing"

    turned = tqg.realized_group(pxpy_centred(), characters=declared, **FREE)
    assert not isinstance(turned, Refusal), str(turned)
    assert (turned.group.number_str, turned.group.name) == ("149.22", "P3121'"), \
        "the twofolds' reading"
    w = turned.cross_check
    assert w["outcome"] == "ambiguous" and w["agrees"] is None
    assert w["substrate"]["number_str"] == "156.50"
    assert {t["name"] for t in w["tie"]} == set(mirrors[3:])
    assert ("{" + ", ".join(twofolds) + "} and {" + ", ".join(mirrors) + "}") in w["note"]
    assert ("does not hold for that one; it holds for {" + ", ".join(mirrors)
            + "} free-standing and P3m11' #156.50 on a substrate") in w["note"], w["note"]


@pytest.mark.slow
def test_a_refusal_over_there_is_refused_and_words_it_has_none_for_are_not_compared():
    """``"refused"`` and ``"not compared"``, and the substrate answer is still served.

    Constructed from the one knob that makes the two readings refuse
    differently: ``descend=False`` asks for the cell's upper bound, and
    Kane–Mele plus Rashba obeys the whole substrate cell (``P6mm1' #183.186``,
    24 of 24) while breaking 24 operations of the free-standing one — so that
    reading refuses ``cell_group_not_realized`` where this one is served.  The
    record carries the gate, and the outcome is ``"refused"`` rather than
    ``"ambiguous"``, which is reserved for a subgroup statement that fails for
    the candidate the tie rule keeps and holds for another of the candidates.  Allowed to descend, the free-standing
    reading loses the operations that turn the layer over, reaches ``3m`` site
    groups, reads the declared ``A1`` there, and the two agree.

    A declaration in site irreps the other environment has no word for is not
    compared at all.  Graphene's sublattices declared ``A1`` and ``A2`` of their
    ``3m`` sites on a substrate are ``P31' #143.2``, 6 of 6: each mirror that
    holds one bond keeps the ``A1`` orbital and reverses the ``A2`` one, so the
    hopping along the bond breaks it, and ``C3`` keeps both.  The free-standing
    layer's sites are ``-6m2``, which has no ``A1`` or ``A2``: the words say a
    substrate is meant, the free-standing layer is not read with them and its
    slot is ``None``, and nothing lifts them into its site irreps.  Asked
    free-standing, the same words are refused ``unknown_label``; and asked on a
    substrate, a free-standing layer's word, ``A2''`` of ``-6m2``, is refused
    the same way, naming the substrate's group: the substrate is the crystal
    asked about.
    """
    r = tqg.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), characters={0: "A1"},
                           descend=False, **SUB)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, len(r.validated.indices)) == ("183.186", 24)
    w = r.cross_check
    assert w["outcome"] == "refused" and w["agrees"] is None
    assert w["refusal"]["gate"] == "cell_group_not_realized"
    assert w["refusal"]["environment"] == "free-standing"
    assert set(w["refusal"]) == {"gate", "reason", "environment"}
    assert w["free_standing"] is None and w["substrate"]["number_str"] == "183.186"
    assert "not a mismatch" in w["note"]
    assert "the other environment refused" in repr(r)
    down = tqg.realized_group(kane_mele(0.0, -1.0, 0.3, 0.25), characters={0: "A1"}, **SUB)
    assert not isinstance(down, Refusal), str(down)
    w = down.cross_check
    assert (w["outcome"], w["free_standing"]["number_str"], w["substrate"]["number_str"]) \
        == ("agree", "183.186", "183.186"), "the free-standing reading reads A1 in 3m"

    words = {0: "A1", 1: "A2"}
    split = tqg.realized_group(graphene(0.0, -1.0), characters=words, **SUB)
    assert not isinstance(split, Refusal), str(split)
    assert (split.group.number_str, split.group.name, len(split.validated.indices)) == \
        ("143.2", "P31'", 6)
    assert split.descent["dropped_by"] == {"declaration": 6}
    w = split.cross_check
    assert (w["outcome"], w["agrees"], w["free_standing"], w["refusal"]) == \
        ("not compared", None, None, None)
    assert w["substrate"]["number_str"] == "143.2" and w["rows"] == ()
    assert "the declaration is written in the substrate's site irreps" in w["note"]
    assert ("on a substrate, not compared with the free-standing layer: the declaration "
            "is written in the substrate's site irreps") in repr(split)
    told = tqg.realized_group(graphene(0.0, -1.0), characters=words, **FREE)
    assert isinstance(told, Refusal) and told.gate == "unknown_label", str(told)
    assert told.context["group"].number_str == "191.234"
    other = tqg.realized_group(tqc_models.pxpy_honeycomb(), characters={0: "A2''"}, **SUB)
    assert isinstance(other, Refusal) and other.gate == "unknown_label", str(other)
    assert other.context["group"].number_str == "183.186"


# ---------------------------------------------------------------------------
# 4. closure resolution, on its own terms
# ---------------------------------------------------------------------------

def test_the_subgroup_enumeration_is_complete_on_a_group_it_can_be_checked_against():
    """Every subgroup of ``p4mm``'s eight unitary operations, by brute force.

    ``_subgroups_inside`` builds the join-closure of the cyclic subgroups, which
    is complete because a subgroup is the join of its own cyclic subgroups.
    Measured against an exhaustive search over all 256 subsets of the eight
    operations of ``P4mm1'``'s unitary part -- the square layer's cell on a
    substrate, one shadow under its site: the same 10 subgroups, exactly.
    """
    import itertools

    cell, typ = _layers._decorated_cell(_layers._slab_presentation(square_layer()), [0])
    group = _spacegroup._group_of(cell, typat=typ)
    assert not isinstance(group, Refusal), str(group)
    assert (group.number_str, group.name) == ("99.164", "P4mm1'")
    unit = [int(s.ind) for s in group.symmetries if not s.time_reversal]
    assert len(unit) == 8
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    ind_of = [int(s.ind) for s in group.symmetries]

    def closed(sub):
        sub = set(sub)
        return all(ind_of[int(prod[at[a], at[b]])] in sub for a in sub for b in sub)

    brute = {frozenset(c) for n in range(1, 9)
             for c in itertools.combinations(unit, n) if closed(c)}
    assert _subgroups._subgroups_inside(group, unit) == brute
    assert len(brute) == 10


# ---------------------------------------------------------------------------
# 5. parity and sigma_h on a free-standing spinless layer
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_flat_spinless_layer_keeps_sigma_h_and_the_declaration_names_it():
    """``P4/mmm1' #123.340``, 32 of 32 with ``p_z`` declared; the label is the declaration.

    A free-standing layer of odd orbitals is ``σ_h``-symmetric with character
    ``−1``: the group membership is the ``s`` layer's and only the labels flip.
    **The measurement is the label** — the irrep the served representation is
    identified as, ``A2u`` for the declared ``p_z`` where the undeclared layer
    reads ``A1g`` (the sibling test below), with the same 32 operations obeyed.
    The two characters below are the declaration echoed and not a second
    measurement: ``generator_character`` is ``_sitechars``' tabulated
    ``χ(σ_h) = χ(i) = −1`` for the orbital the caller named, and a flat model
    has nothing out of plane that could contradict it (which is what the
    sibling test states).  They are asserted so that the label and the table it
    came from stay the same statement.  Re-pinned in wave R4c only in the
    keyword: ``σ_h`` and inversion are what a free-standing layer has and a
    substrate-bound one does not, so this is asked of the free-standing reading
    by name.

    That sign is the one choice a flat layer read without spin leaves: each of
    the 32 operations shares its action on the orbitals and on in-plane ``k``
    with its product with ``m(z)``, and the mirror's sign on what the hoppings
    couple is the declared parity -- odd here -- or, with nothing declared, the
    even one of the lowest-angular-momentum rule.  Every product with the
    mirror carries its partner's matrix times that sign, inversion among them.
    """
    r = tqg.realized_group(square_layer(), characters={0: "pz"}, cross_check=False, **FREE)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.size) == ("123.340", 32)
    assert len(r.validated.indices) == 32
    assert list(r.rep.characters.irreps) == ["A2u"]
    chars = r.rep.characters.orbitals[0].generator_character
    assert complex(chars["m(z)"]).real == pytest.approx(-1.0, abs=1e-9)
    assert complex(chars["i"]).real == pytest.approx(-1.0, abs=1e-9)

    partners = _descent._mirror_partners(r.group, r.validated, r.assignment)
    assert len(partners) == 16 and len(set(partners) | set(partners.values())) == 32
    decls, _layout = _descent._declared_contents(r.model, r.group, r.assignment, r.validated,
                                            r.rep.multiplets, spin=None,
                                            characters={0: "pz"})
    assert _descent._mirror_sign(r.group, partners, decls) == {"sign": -1,
                                                          "determined_by": "declaration"}
    assert _descent._mirror_sign(r.group, partners) == {"sign": 1, "determined_by": "lowest-l"}
    for flipped, kept in partners.items():
        assert np.abs(r.rep.matrices[flipped] + r.rep.matrices[kept]).max() < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("mix, conserved", [(0.0, 1), (0.3, 0)])
def test_a_spinless_layers_sigma_h_is_never_broken_by_the_hoppings(mix, conserved):
    """32 of 32 with **and** without the parity-mixing term — and what it settles.

    A flat spinless model carries no out-of-plane structure at all, so ``σ_h``
    acts as the identity on everything it has and no Hamiltonian can refute it:
    the served group is ``P4/mmm1' #123.340``, 32 of 32, either way.  What the
    ``⟨s|H|p_z⟩`` term does is **measure** the relative ``σ_h`` character of the
    two orbitals, and what it measures is that they are equal.  Without it the
    two never meet: their relative parity is not measured at all, which the
    record says as a conserved operator — a phase on each orbital separately
    commutes with ``H(k)`` at every ``k``, so ``conserved["dim"]`` is ``1`` at
    ``mix = 0`` and ``0`` with the term.  The labels are equal either way,
    because where nothing measures a relative character the convention is the
    **least twist** and that is the trivial one; the contrast the physics
    carries is which of the two readings is a measurement.

    Breaking ``σ_h`` needs a spin structure (Kane–Mele plus Rashba), real
    out-of-plane geometry (the buckled layer below), or a substrate, which
    removes it by construction.  Asked free-standing, since that is the reading
    in which ``σ_h`` exists to be discussed at all; the substrate reading the
    cross-check runs beside it is the one-sided group, ``P4mm1' #99.164``, the
    in-plane image of ``#123.340``, and the parity-mixing term is invisible
    there, because the character it moves is a character of an operation that
    group does not have.

    Declared, ``s`` and ``p_z`` are ``A1g`` and ``A2u``, opposite under ``σ_h``,
    and their bands cross between ``GM`` and ``M``: the lower band alone fails
    the compatibility relations, so a decomposition refuses
    ``not_a_band_structure``, and the pair decomposes as the two orbitals at
    ``1a`` alone, the positions off the layer left out.
    """
    if not mix:
        told = tqg.realized_group(square_s_pz(), characters={0: "s", 1: "pz"},
                                  cross_check=False, **FREE)
        lower = tqe.decompose(tqw.hsp_table(told.rep, bands=1), told.group, rep=told.rep)
        assert isinstance(lower, Refusal) and lower.gate == "not_a_band_structure", str(lower)
        both = tqe.decompose(tqw.hsp_table(told.rep, bands=2), told.group, rep=told.rep)
        assert both.solution_strings() == (
            "1 x A1g↑G(1)@1a(4/mmm1',4/mmm) + 1 x A2u↑G(1)@1a(4/mmm1',4/mmm)",)
        assert sorted(both.restriction.outside_layer) == ["1b", "1d", "2e"]
    r = tqg.realized_group(square_s_pz(mix), **FREE)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.size) == ("123.340", 32)
    assert len(r.validated.indices) == 32 and r.descent is None
    assert list(r.rep.characters.irreps) == ["A1g", "A1g"]
    chars = [o.generator_character["m(z)"] for o in r.rep.characters.orbitals]
    assert abs(chars[0] - chars[1]) < 1e-9
    assert r.conserved["dim"] == conserved, (
        "the parity-mixing term is exactly what removes the conserved operator "
        "that left the relative parity open")
    w = r.cross_check
    assert w["outcome"] == "agree"
    assert w["free_standing"]["number_str"] == "123.340"
    assert w["substrate"]["number_str"] == "99.164" and w["substrate"]["size"] == 16


# ---------------------------------------------------------------------------
# 6. an undecorated dim_r = 3 slab: the geometry decides the fiber
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_buckled_honeycomb_is_read_from_its_real_geometry():
    """``P-3m11' #164.86``, 24 of 24 — the layer group of a buckled layer, in any cell.

    A ``dim_r = 3`` slab carries real ``z`` coordinates, so it is **not** silent
    about its environment and its default is free-standing: the cell is that
    geometry, the two sublattices at ``±d`` admit ``S3`` and the in-plane
    twofolds but not ``C6`` and not the vertical mirrors, and the Hamiltonian
    obeys all 24 of what is left.  Nothing descends, because nothing was allowed
    that the geometry does not have.  Measured on Kane–Mele on the buckled
    layer; the spinless layer reads the same group and count, since what
    buckling removes is a property of the point set, not of the spin structure.

    The geometry is where the orbitals are, not the cell they are written in.
    There is no translation along the non-periodic vector, so the same layer
    written with that vector tilted off the normal, every orbital where it was,
    reads the same group, operations and labels: the presentation puts the
    vector on the normal, and each orbital's in-plane reduced coordinates move
    by its reduced height times the tilt, ``±0.005 × (0.3, 0.2)`` here, a gauge
    on ``H(k)`` measured to rounding.  pythtb's ``cut_piece`` keeps a crystal's
    lattice vectors, so three ``{111}`` layers cut from a face-centred cubic
    crystal come with the vector tilted, and read ``P-3m11'`` as the same three
    layers written on the normal do, label for label; one layer, its orbital at
    the origin, moves nothing and is presented bit for bit.

    Decomposed over the layer's own band representations, those of the
    positions within it -- ``1b`` and ``3f`` lie outside the slab and are left
    out -- the valence pair has no atomic limit, with a proof, and the
    four bands are the orbitals' own spinor pair at ``2d``.  Its class is read
    on the decoupled stack, each orbital at its own height and the period the
    slab's ``0.3`` plus ``c``, the stack's Hamiltonian the layer's exactly at
    ``k_z = 0`` and up to the phases of the heights elsewhere: the valence pair
    reads ``(1, 0)`` in ``Z2 x Z4``, ``z2w,3 = 1`` and ``z4 = 2``, the flat
    layer's values on the names the two groups share, with stable real-space
    invariants that are stable; the four bands' are the orbitals' own.  The
    tilted presentation reads the same class.
    """
    r = tqg.realized_group(buckled(kane_mele(0.0, -1.0, 0.3, 0.0)), cross_check=False)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name, r.group.size) == ("164.86", "P-3m11'", 24)
    assert len(r.validated.indices) == 24 and r.descent is None
    assert r.environment == "free-standing" and r.environment_declared is False
    assert _names(r.group) == ["C2([010])", "C2([1-10])", "C2(x)", "C3(z)", "C3^-1(z)",
                               "E", "S3(z)", "S3^-1(z)", "i", "m([010])", "m([1-10])",
                               "m(x)"]
    assert r.presentation["tilted"] is None
    valence = tqe.decompose(tqw.hsp_table(r.rep, bands=2), r.group, rep=r.rep)
    assert (valence.verdict, valence.certificate) == ("none", "proof"), valence.summary()
    assert valence.signed_combinations["n_solutions"] == 0
    assert sorted(valence.restriction.outside_layer) == ["1b", "3f"]
    assert ("1b and 3f lie outside the slab, so their band representations are left "
            "out") in valence.summary()
    whole = tqe.decompose(tqw.hsp_table(r.rep, bands=4), r.group, rep=r.rep)
    assert whole.solution_strings() == ("1 x -E1↑G(4)@2d(3m1',3m)",)

    def indicator(realized, bands):
        return tqe.symmetry_indicator(tqw.hsp_table(realized.rep, bands=bands),
                                      realized.group, rep=realized.rep, bands=bands)

    # its class, on the decoupled stack at the orbitals' heights
    si = indicator(r, 2)
    assert (si.structure, si.values, si.canonical) == \
        ("Z2 x Z4", (1, 0), {"z2w,3": 1, "z4": 2}), si.summary()
    comp = si.completion
    assert (comp["outcome"], comp["stack"]["number_str"]) == ("completed", "164.86")
    assert (comp["stack_period"], comp["stack_extent"]) == (pytest.approx(8.3),
                                                            pytest.approx(0.3))
    identity = comp["gates"]["hamiltonian_identity"]
    assert identity["at_zero"] == 0.0 and identity["residual"] <= identity["threshold"]
    pair = tqr.srsi_of_bands(tqw.hsp_table(r.rep, bands=2), r.group, rep=r.rep, bands=2)
    assert pair.stable and [str(x) for x in pair.theta_Z] == \
        ["-1/2", "0", "0", "0", "0", "0", "0", "1/2", "0"]
    assert tqr.srsi_of_bands(tqw.hsp_table(r.rep, bands=4), r.group, rep=r.rep,
                             bands=4).theta_Z == tqr.srsi_of_model(r).theta_Z

    def labels(realized, bands):
        return tqw.hsp_table(realized.rep, bands=bands).counts

    layer = buckled(kane_mele(0.0, -1.0, 0.3, 0.0))
    lat = np.asarray(layer.lat_vecs, dtype=float)
    src = rewritten(layer, lat[2] + 0.3 * lat[0] + 0.2 * lat[1])
    t = tqg.realized_group(src, cross_check=False)
    assert not isinstance(t, Refusal), str(t)
    assert (t.group.number_str, t.group.size) == ("164.86", 24) and t.descent is None
    assert _names(t.group) == _names(r.group)
    assert labels(t, 2) == labels(r, 2)
    assert indicator(t, 2).values == (1, 0)
    assert t.source is src and np.abs(np.asarray(t.model.lat_vecs)[2, :2]).max() < 1e-12
    where = [np.asarray(m.orb_vecs) @ np.asarray(m.lat_vecs) for m in (t.model, src)]
    assert np.abs(where[0] - where[1]).max() < 1e-12, "every orbital where it was"
    moved = t.presentation["tilted"]
    assert moved["tilt"] == pytest.approx((0.3, 0.2), abs=1e-12)
    assert moved["n_moved"] == 2 and moved["max_move"] == pytest.approx(0.0015, abs=1e-12)
    assert max(moved["spectrum_residual"], t.presentation["residual"]) < 1e-12

    cut = fcc_s().cut_piece(3, 2)
    plane = np.asarray(cut.lat_vecs, dtype=float)[:2]
    normal = np.cross(plane[0], plane[1])
    hand = rewritten(cut, 30.0 * normal / np.linalg.norm(normal))
    a = tqg.realized_group(cut, cross_check=False)
    b = tqg.realized_group(hand, cross_check=False)
    assert not isinstance(a, Refusal) and not isinstance(b, Refusal), (str(a), str(b))
    assert (a.group.name, a.group.number_str) == (b.group.name, b.group.number_str) \
        == ("P-3m11'", "164.86")
    assert labels(a, 3) == labels(b, 3)
    assert list(a.rep.characters.irreps) == list(b.rep.characters.irreps)
    assert a.presentation["tilted"]["n_moved"] == 2 and b.presentation["tilted"] is None
    one = tqg.realized_group(fcc_s().cut_piece(1, 2), cross_check=False)
    assert (one.group.name, one.group.number_str) == ("P6/mmm1'", "191.234")
    assert one.presentation["tilted"]["n_moved"] == 0
    assert one.presentation["residual"] == 0.0, "nothing moved: bit for bit"


@pytest.mark.slow
def test_the_buckled_honeycomb_on_a_substrate_is_p3m1():
    """``P3m11' #156.50``, 12 of 12 — the side-preserving half of its ``p-3m1``.

    Wave R4d's changed reading, and the one that made the wave.  A buckled slab
    put on a substrate keeps its geometry: nothing is projected, the shadow goes
    under each site at that site's own height, and what the substrate removes is
    every operation with ``R[2,2] = −1`` — the ``S3``s, the inversion and the
    three in-plane twofolds of ``P-3m11' #164.86``.  Exactly the threefold and
    the three vertical mirrors survive, which is ``p3m1``, and the Hamiltonian
    obeys all 12 of them with no descent.

    Wave R4b read this cell by *projecting* it flat and serving each in-plane
    operation with whichever lift the model carried, which made it ``p6mm``
    (``#183.186``, six operations on their ``σ_h`` partner's lift ``S(σ_h)S(g)``)
    — and with spin–orbit coupling that reading refused ``substrate_spinor_action``,
    because ``(S(σ_h)S(C6))⁶ = +1`` where a sixfold of ``p6mm``'s double group
    needs ``−1``.  The subgroup reading never builds that mismatch: ``p3m1``'s
    double group is its own.  So Kane–Mele on the buckled layer reads the same
    group as the spinless layer does, without a refusal, where R4b's reading
    separated them.
    """
    r = tqg.realized_group(buckled(kane_mele(0.0, -1.0, 0.3, 0.0)), **SUB)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name, r.group.size) == ("156.50", "P3m11'", 12)
    assert len(r.validated.indices) == 12
    assert r.descent is None, "nothing was allowed that the geometry does not have"
    assert r.environment_declared is True
    assert _names(r.group) == ["C3(z)", "C3^-1(z)", "E", "m([010])", "m([1-10])", "m(x)"]
    assert r.assignment.site_orbitals == ((0,), (1,)), "the sublattices stay apart"
    assert r.assignment.wyckoff_spglib == ("c", "b")
    w = r.cross_check
    assert w["outcome"] == "agree" and w["free_standing"]["number_str"] == "164.86"
    assert w["missing"] == () and w["extra"] == ()
    assert w["flipping"] == () and w["unmapped"] == ()
    assert len(w["rows"]) == 12, "one row per substrate operation, both cosets"
    assert "preserves the layer's normal" in w["note"]


@pytest.mark.slow
def test_a_mirror_symmetric_bilayer_keeps_its_sigma_h():
    """``P4/mmm1' #123.340``, 32 of 32 — and on a substrate ``P4mm1' #99.164``.

    An AA bilayer at ``±d`` with a vertical bond is ``σ_h``-symmetric as a
    crystal, and free-standing — its ``dim_r = 3`` default — the undecorated
    cell reads it: the mirror exchanges the two layers, which is an operation of
    the point set and of the Hamiltonian.  Put on a substrate the mirror and
    everything composed with it goes and the group halves, but the **geometry
    stays**: the two layers remain two sites at ``±d``, each at ``1a`` of its own
    plane.  In wave R4b's projected reading they became one site — the number
    served was the same ``#99.164``, and that it is the same is the measurement
    that the subgroup reading did not move this fixture.

    Free-standing, the bilayer is decomposed over its own band
    representations, those of the positions within it: ``1b``, ``1d`` and
    ``2e`` lie outside the slab and are left out, so the bonding
    band is ``A1g`` at ``1a`` and the pair is ``A1g + A2u`` there, with no sum
    at ``1b``.  Its atomic-limit table names no other sum for the pair, and its
    SRSIs are evaluated at ``1a`` alone.  Its indicator group is trivial, the
    rows out of the plane measured on the decoupled stack at the two layers'
    heights.  On a substrate every position crosses the layer, and the pair is
    two ``A1`` at ``1a``.
    """
    free = tqg.realized_group(aa_bilayer())
    assert not isinstance(free, Refusal), str(free)
    assert (free.group.number_str, free.group.size) == ("123.340", 32)
    assert len(free.validated.indices) == 32 and free.descent is None
    assert free.environment == "free-standing"
    assert free.cross_check["outcome"] == "agree"
    bonding = tqe.decompose(tqw.hsp_table(free.rep, bands=1), free.group, rep=free.rep)
    assert bonding.solution_strings() == ("1 x A1g↑G(1)@1a(4/mmm1',4/mmm)",)
    pair = tqe.decompose(tqw.hsp_table(free.rep, bands=2), free.group, rep=free.rep)
    assert pair.solution_strings() == (
        "1 x A1g↑G(1)@1a(4/mmm1',4/mmm) + 1 x A2u↑G(1)@1a(4/mmm1',4/mmm)",)
    assert sorted(pair.restriction.outside_layer) == ["1b", "1d", "2e"]
    (orbit,) = free.rep.characters.band_representations
    assert orbit.alternatives == ()
    (srsi_orbit,) = tqr.srsi_of_model(free).orbits
    assert [loc["letter"] for loc in srsi_orbit.localizations] == ["a"]
    for bands in (1, 2):
        si = tqe.symmetry_indicator(tqw.hsp_table(free.rep, bands=bands), free.group,
                                    rep=free.rep, bands=bands)
        assert si.verdict == "trivial indicator group", si
        assert (si.completion["outcome"], si.completion["stack_extent"]) == \
            ("completed", pytest.approx(0.3))

    sub = tqg.realized_group(aa_bilayer(), **SUB)
    assert not isinstance(sub, Refusal), str(sub)
    assert (sub.group.number_str, sub.group.size) == ("99.164", 16)
    assert len(sub.validated.indices) == 16
    assert sub.assignment.site_orbitals == ((0,), (1,)), "the two layers stay two sites"
    assert sub.assignment.wyckoff_spglib == ("a", "a")
    assert sub.cross_check["outcome"] == "agree"
    assert sub.cross_check["free_standing"]["number_str"] == "123.340"
    on_sub = tqe.decompose(tqw.hsp_table(sub.rep, bands=2), sub.group, rep=sub.rep)
    assert on_sub.solution_strings() == ("2 x A1↑G(1)@1a(4mm1',4mm)",)
    assert on_sub.restriction.outside_layer == {}


# ---------------------------------------------------------------------------
# 7. which p4mm: the setting is measured, not chosen
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_translation_conjugate_p4mm_is_the_one_the_hamiltonian_realizes(p41_substrate):
    """The same number ``#99.164``, a different setting origin — measured.

    ``p4mm`` about the origin and ``p4mm`` about ``(0,½)`` are both groups on
    this model's lattice, and which one a Hamiltonian realizes is a measurement.
    With the fourfold arranged to **fix** the two sites rather than exchange
    them, the served fourfold carries the translation ``(½,½,0)`` where the
    paper's own two-site cell has ``(0,0,0)``; the two sites are then
    inequivalent (that is what selects this conjugate) and sit at ``1a`` and
    ``1b`` rather than one ``2c`` orbit.  Nothing about the coordinates was
    declared either way — only the environment, and both readings are the
    substrate's here.  The paper's cell with its centres jittered by 1e-5 and
    its precision declared reads the paper's conjugate too: the centres are made
    exact in the model's own frame before anything is named.
    """
    def _tau(group, name):
        sym = next(s for s in group.symmetries
                   if not s.time_reversal
                   and _sitechars.operation_name(s, group.lattice) == name)
        return tuple(round(float(x), 6) for x in np.asarray(sym.translation).reshape(-1))

    conj = tqg.realized_group(conjugate_p4mm(), cross_check=False, **SUB)
    assert not isinstance(conj, Refusal), str(conj)
    assert (conj.group.number_str, conj.group.size) == ("99.164", 16)
    assert len(conj.validated.indices) == 16
    assert _tau(conj.group, "C4(z)") == (0.5, 0.5, 0.0)
    assert conj.assignment.wyckoff_tables == ("b", "a")

    paper = p41_substrate
    assert not isinstance(paper, Refusal), str(paper)
    assert paper.group.number_str == conj.group.number_str
    assert _tau(paper.group, "C4(z)") == (0.0, 0.0, 0.0)
    assert paper.assignment.wyckoff_tables == ("c", "c")

    # the paper's cell with every centre jittered by 1e-5, its precision
    # declared, reads the same conjugate: the centres are made exact in the
    # model's own frame, and the fourfold passes through its origin
    fitted = tqg.realized_group(jittered(tqc_models.p41_flat_band(), 1e-5),
                                precision=1e-4, cross_check=False, **SUB)
    assert not isinstance(fitted, Refusal), str(fitted)
    assert fitted.group.number_str == paper.group.number_str
    assert _tau(fitted.group, "C4(z)") == (0.0, 0.0, 0.0)
    assert fitted.assignment.wyckoff_tables == ("c", "c")
    assert (tqw.hsp_table(fitted.rep, bands=2).counts
            == tqw.hsp_table(paper.rep, bands=2).counts)


# ---------------------------------------------------------------------------
# 8. the consistency gate raises on a mismatch
# ---------------------------------------------------------------------------

def test_the_record_names_the_side_flippers_and_the_error_reads_them():
    """The record and the mismatch it raises, on a substrate group that keeps ``σ_h``.

    The doctored case, rebuilt in wave R4d for the subgroup statement.  Handing
    ``_environment_record`` Kane–Mele's *free-standing* ``P6/mmm1' #191.234`` in
    the substrate slot is a claim no substrate can make: 24 of its 48 operations
    have ``R[2,2] = −1`` — the inversion, the horizontal mirror, the in-plane
    twofolds and the ``S_n``s — and a substrate removes every one of them by
    construction.  The record names them and ``_raise_environment_contradiction`` raises
    on them, since two selections of one Hamiltonian contradicting the
    subgroup statement is the package disagreeing with itself; that the
    *route* reaches it at all is measured below, through ``realized_group``
    itself.  The group is the cell's own, what the honeycomb point set allows
    before the Hamiltonian is asked.
    """
    group = _spacegroup._group_of(_layers._slab_presentation(tqc_models.kane_mele_inversion()))
    assert not isinstance(group, Refusal), str(group)
    assert (group.sg_number, group.number_str, group.size) == (191, "191.234", 48)
    record = _layers._environment_record(group, group)
    assert record["agrees"] is False and record["outcome"] is None
    assert len(record["flipping"]) == 24
    assert record["missing"] == () and record["extra"] == (), \
        "empty by construction: one group in both slots, so the two key sets agree"
    assert {r["name"] for r in record["flipping"]} >= {"i", "m(z)", "C2(x)", "S3(z)"}

    with pytest.raises(InternalError, match="flip the layer's side") as err:
        _layers._raise_environment_contradiction(record)
    assert "24 operations of the substrate group" in str(err.value)
    assert "191.234" in str(err.value)


@pytest.mark.slow
def test_realized_group_itself_raises_when_the_two_environments_disagree(monkeypatch):
    """The gate is *wired in*: the mismatch comes out of the route, raised.

    The test above calls the helpers by hand; this one makes ``realized_group``
    reach the mismatch, which is the claim that the gate is reached at all.  The
    route computes the other environment through ``_realized_selection``, so
    rebinding that module attribute replaces exactly that call and nothing else
    — measured by the counter: one selection call intercepted, no more.
    Kane–Mele's free-standing ``P6/mmm1' #191.234`` is then held against
    ``km_odd``'s substrate reading ``P3m11' #156.50``, a real group of a real
    model and not the side-preserving subgroup of this one: 12 operations that
    keep the layer's side are in the free-standing group and not in it, so they
    are ``extra``, no group is served and the package raises.  A tie does not
    excuse it: the declared honeycomb below, whose two free-standing readings
    tie, held against the undeclared honeycomb's ``P6mm1' #183.186``, a sixfold
    neither reading holds, raises too, both pairs of candidates held to the
    statement.

    Unpatched, the declared honeycomb is a crystal both environments read
    alike.  The ``p_x, p_y`` honeycomb's sublattices declared ``E'`` and ``E''``
    of ``-6m2`` are different sites, and the free-standing cell is
    ``P-6m21' #187.210``.  The hoppings carry the two pairs alike and the
    declarations state opposite ``σ_h`` parities, so the declared crystal
    descends: ``σ_h`` and ``S6`` are refuted, and of the two readings left, the
    threefold with the vertical mirrors and the threefold with the in-plane
    twofolds, the tie keeps the one whose operations all keep the layer's side,
    ``P3m11' #156.50``.  On a substrate ``σ_h`` is gone and both words restrict
    to one ``E`` of ``3m``, and the sites are still different, as two sites of
    different ``typat`` are: the substrate group is ``P3m11' #156.50`` as well,
    and the two agree.

    A word on one orbital of a doublet is the same kind of statement.
    ``{0: "A2''"}`` says a ``p_x, p_y`` site holds an ``A2''`` of ``-6m2``, odd
    under ``σ_h``, where the pair holds ``E'``: only on an ``mm2`` site group
    does one labelling of the pair contain it, and the free-standing crystal is
    ``Cmmm1' #65.482``, every operation it loses broken by the declaration.  On
    a substrate the same word is read where it was written and restricted to
    ``3m``, the ``A1`` it restricts to, which the pair contains on one mirror:
    ``Cmm21' #35.166``, the side-keeping part of the other, and the two agree.

    And one content written in different partitions at the two sites is one
    site: a ``p`` shell declared whole at one, ``p_z`` beside its doublet at
    the other, is ``A2'' + E'`` at both, and a site's whole content is what is
    compared.  Free-standing it is ``P6/mmm1' #191.234``, 48 of 48, every row
    the declaration's and the shell's word in the ``function`` column, each row
    its own multiplet's site irrep, as the shell declared orbital by orbital
    reads, the two environments agreeing; and on a substrate, written as
    ``A1 + E`` against ``A1`` and ``E``, it is ``P6mm1' #183.186``, those words
    being the substrate's site irreps, which the free-standing layer's ``-6m2``
    has none of, so they say a substrate is meant and the two are not
    compared.  The multiplets are the measured ones at both sites, the ``p_z``
    apart from its doublet, and the shell declared whole is one content over
    the two: over the ``p_z`` pair's coupled part of the crystal and the
    doublets' together.

    Undeclared, or declared orbital by orbital, the same shell is two coupled
    parts of the crystal, the ``p_z`` pair and the doublets, which no hopping
    connects.  Each is settled on its own, and the crystal is returned in both
    environments, the two agreeing: ``A1' + E'`` at each site free-standing,
    where the hoppings cannot tell the ``p_z`` from an ``s``, ``A2'' + E'`` as
    declared, and ``A1 + E`` on a substrate.  The doublets' part is assigned by
    the rule the reference orbital's part is, so the matrices returned are the
    same whichever part holds the reference orbital.
    """
    part = tqg.realized_group(tqc_models.pxpy_honeycomb(), characters={0: "A2''"}, **FREE)
    assert not isinstance(part, Refusal), str(part)
    assert (part.group.number_str, part.group.name, len(part.validated.indices)) == (
        "65.482", "Cmmm1'", 16)
    assert part.descent["steps"][0]["refuted_by"] == {"declaration": 32}
    w = part.cross_check
    assert (w["outcome"], w["substrate"]["number_str"]) == ("agree", "35.166"), \
        "the word restricted to the substrate's site group, and the two agree"

    for spelling, env, number, crossed, words in (
            ({(0, 1, 2): "p", 3: "pz", (4, 5): "px"}, FREE, "191.234",
             ("agree", "191.234", "183.186"), ["p"] * 3 + ["pz", "px", "px"]),
            ({(0, 1, 2): "A1 + E", 3: "A1", (4, 5): "E"}, SUB, "183.186",
             ("not compared", None, "183.186"), [""] * 6)):
        one = tqg.realized_group(p_shell_honeycomb(), characters=spelling, **env)
        assert not isinstance(one, Refusal), f"{spelling}: {one}"
        assert (one.group.number_str, len(one.validated.indices)) == (number, one.group.size)
        assert one.descent is None and (one.decoration or {}).get("declared") is None, \
            "one species: nothing excluded from the cell, nothing broken"
        assert one.rep.multiplets.by_site == (((0,), (1, 2)), ((3,), (4, 5)))
        rows = one.rep.characters.orbitals
        assert {o.determined_by for o in rows} == {"declaration"}
        assert [o.function for o in rows] == words, "a shell is a function word, a sum is not"
        assert one.rep.characters.irreps == (
            ("A2''", "E'", "E'") if env is FREE else ("A1", "E", "E")) * 2, \
            "each row its own multiplet's, as the shell declared orbital by orbital reads"
        w = one.cross_check
        assert (w["outcome"], (w["free_standing"] or {}).get("number_str"),
                w["substrate"]["number_str"]) == crossed

    for spelling, labels in ((None, ("A1'", "E'", "E'")),
                             (["pz", "px", "py"] * 2, ("A2''", "E'", "E'"))):
        free = tqg.realized_group(p_shell_honeycomb(), characters=spelling, **FREE)
        assert not isinstance(free, Refusal), f"{spelling}: {free}"
        assert (free.group.number_str, len(free.validated.indices)) == ("191.234", 48)
        assert free.rep.multiplets.by_site == (((0,), (1, 2)), ((3,), (4, 5)))
        assert free.rep.characters.irreps == labels * 2
        w = free.cross_check
        assert w["outcome"] == "agree" and w["substrate"]["number_str"] == "183.186"
        on = tqg.realized_group(p_shell_honeycomb(), characters=spelling, cross_check=False,
                                **SUB)
        assert not isinstance(on, Refusal), f"{spelling}: {on}"
        assert (on.group.number_str, len(on.validated.indices)) == ("183.186", 24)
        assert on.rep.characters.irreps == ("A1", "E", "E") * 2
        if spelling is None:
            undeclared = free
    # the reference orbital named, through the private entry that takes it
    other = _realized._realize(p_shell_honeycomb(), reference_orbital=1, cross_check=False,
                         **FREE)
    assert not isinstance(other, Refusal), str(other)
    assert max(float(np.abs(np.asarray(D) - np.asarray(undeclared.rep.matrices[i])).max())
               for i, D in other.rep.matrices.items()) < 1e-12, \
        "the same matrices whichever part holds the reference orbital"

    original = _realized._realized_selection
    calls = {"selection": 0}

    def wrapper(model, **kw):
        """Answer every selection the route asks for with km_odd's #156.50."""
        calls["selection"] += 1
        return original(tqc_models.kane_mele_z2("odd"), typat=[0, 1],
                        characters={0: "A1"}, cross_check=False, **SUB)

    patch_everywhere(monkeypatch, "_realized_selection", wrapper)
    with pytest.raises(InternalError, match="missing from the substrate group") as err:
        tqg.realized_group(kane_mele(0.0, -1.0, 0.3, 0.0), characters={0: "pz"},
                           **FREE)
    assert calls == {"selection": 1}, \
        "the other reading goes through the module attribute, exactly once"
    said = str(err.value)
    assert "12 operations of the free-standing group that preserve the layer's normal" in said
    assert "191.234" in said and "156.50" in said
    assert "Neither is returned" in said

    # ... and where the free-standing readings tie, a substrate group no pair of
    # the candidates explains: the declared honeycomb against the undeclared
    # one's P6mm1', whose sixfold neither reading holds -- raised all the same
    def planted(model, **kw):
        """Answer every selection with the undeclared honeycomb on a substrate."""
        calls["selection"] += 1
        return original(tqc_models.pxpy_honeycomb(), typat=[0] * 4, cross_check=False,
                        **SUB)

    patch_everywhere(monkeypatch, "_realized_selection", planted)
    with pytest.raises(InternalError, match="183.186") as err:
        tqg.realized_group(tqc_models.pxpy_honeycomb(), typat=[0] * 4,
                           characters={(0, 1): "E'", (2, 3): "E''"}, **FREE)
    assert calls == {"selection": 2}
    assert "156.50" in str(err.value) and "1 more pair" in str(err.value), str(err.value)


@pytest.mark.slow
def test_cross_check_false_skips_the_gate_and_records_nothing(monkeypatch):
    """One identification cheaper, and one red flag fewer — both counted.

    ``cross_check=False`` is the only way to get a 2D answer without the other
    environment being computed, and the record then carries ``None`` rather than
    a stale verdict.  The saving is the claim that needs a witness, since a
    route that computed the other reading and threw it away would carry ``None``
    just the same: the other reading is a ``_realized_selection`` call, so
    counting those calls counts the identifications.  Measured on the
    one-orbital square layer: **zero** with ``cross_check=False`` against
    **one** — the substrate reading — with the default, and the same
    ``#123.340``, 32 of 32, served either way.
    """
    original = _realized._realized_selection
    selections = []

    def counting(model, **kw):
        """Every call forwarded, and its environment recorded."""
        selections.append(str(kw.get("environment")))
        return original(model, **kw)

    patch_everywhere(monkeypatch, "_realized_selection", counting)
    r = tqg.realized_group(square_layer(), cross_check=False, **FREE)
    assert selections == [], "no second identification ran"
    checked = tqg.realized_group(square_layer(), **FREE)
    assert selections == ["substrate"], "the default runs it exactly once"

    assert not isinstance(r, Refusal), str(r)
    assert r.cross_check is None
    assert (r.group.number_str, len(r.validated.indices)) == ("123.340", 32)
    assert not isinstance(checked, Refusal), str(checked)
    assert checked.cross_check["outcome"] == "agree"
    assert checked.group.number_str == r.group.number_str


# ---------------------------------------------------------------------------
# 9. the closed representation gate
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_opposite_axis_transcription_reads_its_own_layer_group():
    """``P-4m21' #115.284``, 16 of 16 — the group the paper's raw blocks obey.

    The same Hamiltonian written for the opposite spin quantization axis is a
    unitary conjugation, so no band moves; what moves is *which* member of each
    ``σ_h`` fiber the operators are.  Conjugation by ``iσ_x`` carries ``S(C4)``
    to a multiple of ``S(S4)`` and a diagonal mirror's lift to an in-plane
    twofold's, so the layer group this transcription obeys is ``D2d`` where the
    converted one obeys ``C4v`` — isomorphic as abstract groups, and not the
    same operations.  The free-standing reading reads it off the geometry and
    the spin lifts, with the two fourfold operations left out by closure exactly
    as before.
    """
    r = tqg.realized_group(raw_axis(tqc_models.p41_flat_band()), **FREE)
    assert not isinstance(r, Refusal), str(r)
    assert (r.group.number_str, r.group.name, r.group.size) == ("115.284", "P-4m21'", 16)
    assert len(r.validated.indices) == 16
    assert _names(r.group) == ["C2([1-10])", "C2([110])", "C2(z)", "E",
                               "S4(z)", "S4^-1(z)", "m(x)", "m(y)"]
    assert r.descent["from"]["number_str"] == "123.340"
    assert r.cross_check["outcome"] == "agree"
    assert r.cross_check["substrate"]["number_str"] == "25.58", (
        "re-pinned in wave R4d: the substrate reading is the side-preserving "
        "subgroup of THIS group, and D2d keeps only {E, C2(z), m(x), m(y)}; "
        "R4b's projected reading served #99.164 there by giving four operations "
        "their sigma_h partner's lift")


@pytest.mark.slow
def test_the_opposite_axis_transcription_on_a_substrate_is_the_axial_mm2():
    """``Pmm21' #25.58``, 8 of 8 — the four operations it has with its own lift.

    The wave's second changed reading, and the owner's ruling made into a
    measurement: *the library shouldn't be flipping spin axes in the first
    place*.  Written for the opposite spin quantization axis, this Hamiltonian
    obeys ``S4``, ``S4⁻¹`` and the two in-plane twofolds where the converted one
    obeys ``C4``, ``C4⁻¹`` and the two diagonal mirrors — its free-standing group
    is ``P-4m21' #115.284``.  Every one of those four flips the layer's side, so
    a substrate breaks them and what is left is ``{E, C2(z), m(x), m(y)}``: the
    axial ``mm2``.  Wave R4b served ``P4mm1' #99.164`` here instead, by
    representing ``C4`` with its ``σ_h`` partner's lift ``S(σ_h)S(C4)`` — which
    is reading the model through a flipped axis, by measurement.

    The tensor test is asked on the ``g`` branch only in this reading, and that
    is the same rule: on the ``g⁻¹`` branch the candidate carries ``S(g)†``, and
    ``S(C4)† = −S(σ_h)S(C4)`` is that partner's lift exactly.  Measured on this
    cell, the ``g`` branch reads 1.0 for ``C4`` and ``C4⁻¹`` and ~7e-16 for the
    four it keeps.

    The **representation** of that group used to be refused ``rep_not_unique``,
    and wave R4e serves it: every element of ``mm2`` is an involution whose lift
    satisfies ``S(g)† ∝ S(g)``, so the two orientation branches ``D`` and ``D†``
    both close all three gates at machine zero (covariance 1.0e-16 each;
    internal factor 6.1e-17 against 1.2e-16; product table 0.0 against 4.9e-16),
    both label the two sites ``A1``, and nothing measures between them.  A tie
    no measurement can break is a convention, so ``+1`` is served and
    ``branch_tie`` says so; ``orientation_branch=±1`` still builds either, and
    on this model the two are the same matrices to 2.4e-16 — ``normalize_rep``'s
    canonical phase undoes the sign the adjoint put on each generator.

    And the same holds in three dimensions, where every operation is alone in
    its k-action class and is held to its own lift directly.  The spinful ``s``
    crystal ``R-3m1' #166.98`` (inversion, time reversal, a threefold axis)
    returns its group as written.  With every spin block turned by a generic
    ``U`` -- axis ``(0.3, -0.5, 0.8)``, 1.1 rad -- and the lattice left where it
    is, the same Hamiltonian as the crystal turned whole up to a rigid
    rotation, it keeps inversion and time reversal alone, whose lifts are the
    same in every frame: ``P-11' #2.5``, the other twenty operations refuted by
    tensor structure at scaled defects 0.51 to 0.87.  Turned whole, lattice and
    spin blocks together, it returns ``#166.98`` with the labels it has as
    written.  Before the lift was held per operation the turned crystal kept
    all 24 and its representation missed the covariance gate by 0.03, which
    was raised as the package disagreeing with itself.
    """
    raw = raw_axis(tqc_models.p41_flat_band())
    full = tqg.realized_group(raw, cross_check=False, **SUB)
    assert not isinstance(full, Refusal), str(full)
    g = full.group
    assert (g.number_str, g.name, g.size) == ("25.58", "Pmm21'", 8)
    assert len(full.validated.indices) == 8
    assert _names(g) == ["C2(z)", "E", "m(x)", "m(y)"]
    assert full.descent["from"]["number_str"] == "99.164"
    assert full.descent["steps"][0]["refuted_by"] == {"tensor structure": 24}
    assert full.rep._branch_tie is True, "re-pinned in wave R4e; it used to refuse"
    assert full.rep._choice["orientation"] == 1
    assert full.rep._search["n_survivors"] == 2, "what the GATES found"
    assert full.rep._search["n_covariant"] == 2
    assert "pinned by convention" in full.rep._search["branch_note"]

    built = {}
    for branch in (1, -1):
        one = _build._build_rep(full.model, g, full.assignment,
                             validated=full.validated, orientation_branch=branch)
        assert not isinstance(one, Refusal), str(one)
        assert one._choice["orientation"] == branch
        assert one._branch_tie is False, "a restricted search never asks the question"
        # the s orbital's A1 of mm2, returned on this spinful model as the corep -E
        assert list(one.characters.irreps) == ["-E", "-E"]
        built[branch] = one
    # the two branches are NOT compared bit for bit: each is built from its own
    # adjoint and then phased by `normalize_rep`, so what is left is libm's
    # sin(pi) residue in that canonical phase -- a few ulp, measured 2.4e-16
    # (1.1 ulp) here.
    assert max(float(np.abs(built[1].matrices[i] - built[-1].matrices[i]).max())
               for i in built[1].indices) <= _FEW_ULP, \
        "the two branches are the same representation after normalization"

    # in three dimensions: as written, spin blocks turned, and turned whole
    crystal = tqc_models.r3m_s_6c()
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    angle = 1.1
    n_sigma = (axis[0] * np.array([[0, 1], [1, 0]], dtype=complex)
               + axis[1] * np.array([[0, -1j], [1j, 0]], dtype=complex)
               + axis[2] * np.array([[1, 0], [0, -1]], dtype=complex))
    u = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * n_sigma
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    rot = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross
    whole = Lattice((np.asarray(crystal.lattice.lat_vecs, dtype=float) @ rot.T).tolist(),
                    np.asarray(crystal.lattice.orb_vecs, dtype=float).tolist(),
                    periodic_dirs=[0, 1, 2])

    written = tqg.realized_group(crystal)
    assert written and written.descent is None, str(written)
    assert (written.group.number_str, len(written.validated.indices)) == ("166.98", 24)
    assert written.rep.characters.irreps == ("-E1", "-E1")

    turned = tqg.realized_group(spin_blocks_turned(crystal, u))
    assert turned, str(turned)
    assert (turned.group.number_str, turned.group.name, turned.group.size) == (
        "2.5", "P-11'", 4)
    assert _names(turned.group) == ["E", "i"]
    assert turned.descent["from"]["number_str"] == "166.98"
    assert turned.descent["steps"][0]["refuted_by"] == {"tensor structure": 20}
    assert all(0.5 < b["residual"] < 0.9 for b in turned.descent["dropped"])

    rigid = tqg.realized_group(spin_blocks_turned(crystal, u, lattice=whole))
    assert rigid and rigid.descent is None, str(rigid)
    assert (rigid.group.number_str, len(rigid.validated.indices)) == ("166.98", 24)
    assert rigid.rep.characters.irreps == written.rep.characters.irreps
    assert (tqw.hsp_table(rigid.rep, bands=[0, 1]).counts
            == tqw.hsp_table(written.rep, bands=[0, 1]).counts)


@pytest.mark.slow
def test_the_closed_gate_raises_on_the_raw_blocks_on_the_tables_own_lift(p41_substrate):
    """The hole, and it is shut: the site-swapping operations have no matrix of their own.

    Handed the *converted* model's substrate group ``P4mm1' #99.164`` but the
    **paper's** raw blocks, the four site-swapping blocks carry
    ``S(σ_h)·S(g)``.  Before wave R4b that build was *served*, because the
    gate looked only at the channels an operation fixes and ``C4`` and the
    diagonal mirrors fix none; the product table closed too, at 5e-16, because
    ``C4v ≅ D2d``.  A faithful representation of one layer group under
    another's name is what the closed gate refused, at defect 1.0; each
    operation's space is now read in its own form before anything is built,
    so the build stops there, naming ``C4``, ``C4⁻¹``, the two diagonal
    mirrors and each of them with time reversal, by their spin lift.  The
    operations a representation is built for are the ones measured to hold
    in their own form, so it raises: no measurement of this model hands the
    build that group.

    Re-pinned in wave R4d only in where the group comes from: the raw model's
    own substrate reading is ``Pmm21' #25.58`` now, so the ``#99.164`` this
    tests against is taken from the converted model, which is the group R4b
    served the raw blocks in.
    """
    raw = raw_axis(tqc_models.p41_flat_band())
    host = p41_substrate                # the session's shared record of that reading
    assert not isinstance(host, Refusal), str(host)
    assert host.group.number_str == "99.164"
    m3 = _layers._slab_presentation(raw)
    assert not isinstance(m3, Refusal), str(m3)
    val = _measure._validate_group(m3, host.group, nk=6)
    with pytest.raises(InternalError) as err:
        _build._build_rep(m3, host.group, host.assignment, validated=val)
    said = str(err.value)
    assert "8 of the" in said, "the four site-swapping ones, and with T"
    assert said.count("its spin lift") == 8, said
    for name in ("C4(z)", "C4^-1(z)", "m([110])", "m([1-10])"):
        assert f"{name} (#" in said and f"{name}' (#" in said, name


# ---------------------------------------------------------------------------
# 10. the decoration record: the substrate's recipe and the caller's extras
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_decoration_record_names_the_base_and_the_extras_on_top():
    """The substrate's shadows and the caller's atom, both named in one record.

    ``decoration`` describes the whole addition, so it names both recipes when
    both ran.  On a substrate the shadow recipe runs whatever the caller adds:
    ``dz`` is the height those shadows went to, ``n_atoms`` counts one shadow
    under each of the two sites plus the caller's own atom, and ``recipe`` says
    the substrate and the extras on top of it.  The same atom read
    free-standing is the whole decoration — nothing was put under the layer, so
    no height was used and ``dz`` is ``None``.  Measured on the square cell
    whose fourfold fixes its two sites, with the extra atom on that axis, so
    both readings serve ``P4mm1' #99.164`` and what differs is the record.
    """
    cap = (([0.0, 0.5, tqg.DECORATION_DZ], "cap"),)

    on_top = tqg.realized_group(conjugate_p4mm(), decorations=cap, cross_check=False, **SUB)
    assert not isinstance(on_top, Refusal), str(on_top)
    assert (on_top.group.number_str, on_top.group.size) == ("99.164", 16)
    assert on_top.decoration["dz"] == tqg.DECORATION_DZ, "the shadow recipe ran"
    assert on_top.decoration["n_atoms"] == 3, "a shadow under each site, and the cap"
    assert on_top.decoration["typat"] == (1, 1, "cap")
    assert "shadow species under every site" in on_top.decoration["recipe"]
    assert on_top.decoration["recipe"].endswith("1 caller-supplied species on top")

    alone = tqg.realized_group(conjugate_p4mm(), decorations=cap, cross_check=False, **FREE)
    assert not isinstance(alone, Refusal), str(alone)
    assert (alone.group.number_str, alone.group.size) == ("99.164", 16)
    assert alone.decoration["dz"] is None, "nothing was put under the layer"
    assert alone.decoration["n_atoms"] == 1 and alone.decoration["typat"] == ("cap",)
    assert alone.decoration["recipe"] == "caller-supplied decoration"
