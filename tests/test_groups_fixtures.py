r"""Phase A1 regressions that run the ``tqc.groups`` adapter against the A0 goldens.

``test_groups.py`` exercises the adapter on inline toy models; this file runs it
on the three published fixtures of ``fixture_models.py`` — **lin3g**,
**Kane-Mele** (both Z2 regimes) and the **P4_232 / SG 208** model of
arXiv:1711.11045 — and locks what it measures there.

Every asserted number carries a comment saying where it comes from: a fixture
docstring, a literature fact, or a *measured lock* (a number this file measured
once and now pins so a regression shows up as a test failure rather than as a
quietly different answer).

Three route decisions, all forced by what the adapter actually supports
----------------------------------------------------------------------
* **Kane-Mele needs ``typat``.**  ``_group_of`` reads geometry only, and its
  documented default treats same-content sites as one species, so the bare
  honeycomb comes back as ``P6/mmm`` (191) — the *maximal* group the point set
  allows.  The fixture's recorded ``slab_sg_number = 187`` (``P-6m2``) is the
  answer for **inequivalent** sublattices, which is what ``delta != 0`` makes
  them, and this file reads it with ``typat=[0, 1]``.  Both answers are
  asserted: 191 is the cell's, 187 is the decorated cell's, and
  :func:`_validate_group` measures which of the 191 operations survive.

* **The bare P4_232 cell is a supercell of its own point set.**  Its two 2a
  orbital sites form a *body-centred* point set, so the cubic cell they live in
  is not primitive and ``_group_of`` serves the **lattice supergroup**:
  ``Im-3m1' #229.141``, 192 operations, ``lattice_index == 2``, the centring
  translation ``(1/2,1/2,1/2)`` among them.  That record *is* the "the cell
  reads Im-3m (229)" statement in the adapter's own voice, and no BCS table is
  served with it (``gate="lattice_not_descended"``) because the tables belong
  to the finer lattice.

* **So the P4_232 candidate operations come from a decorated cell.**  The paper
  itself says the crystal field of extra, non-orbital atoms is what brings the
  cell down to P4_232, and ``tqc_models.p4232_symmetry_cell()`` supplies such a
  decoration (the 24-fold orbit of a generic point).  For the *audit* we want a
  candidate set strictly larger than what the Hamiltonian obeys, so this file
  adds the inversion image of that orbit: the point set is then invariant under
  ``P4_232 + inversion``, which is **Pm-3n (SG 223)** — 48 operations, namely
  the 24 of SG 208 plus their 24 inversion-composed partners.  Those 48 are
  exactly the 48 point-group operations of the bcc cell the bare orbitals form,
  carrying the ``1/2 1/2 1/2`` translations that make half of them
  nonsymmorphic.  ``_validate_group`` never checks that the model it measures is
  the model the group was built from — it only consumes the operations' k-action
  — so the candidate group is built from the decorated cell and the covariance
  is measured on the golden's own Hamiltonian.

The slab caveat that shapes sections 3 and 6
--------------------------------------------
For a 2D model presented as a slab, an operation ``g`` and its ``sigma_h``-
composed partner ``sigma_h * g`` have the **same action on in-plane k** (the
mirror only flips ``k_z``, which is identically zero).  ``_validate_group``'s
detector sees nothing but the k-action, so it *cannot* break a
``sigma_h``-composed candidate: the two share one intertwiner problem and
therefore one verdict.  Both 2D goldens show it, and the tests below assert the
mechanism rather than leaving the survival counts mysterious.
"""

import warnings

import numpy as np
import pytest
from pythtb import Lattice, TBModel

import tqc_models as tqm   # loaded by conftest.py -- see its docstring

from pythtqc import groups as G
from pythtqc.refusal import Refusal
from pythtqc._detector import _find_intertwiners
from pythtqc import _measure
from pythtqc import _spacegroup
from pythtqc import _tables

# Third-party deprecations (spglib, irrep) are noise here; a deprecated pythtqc
# spelling is not, and the second mark keeps the suite's own guard in force
# (marks apply in order, and the last matching filter wins).
pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.filterwarnings("error:pythtqc:DeprecationWarning")]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def op_key(rotation, translation):
    """``(rotation, translation mod 1)`` as a hashable key.

    Operations must be compared by *content*, never by object identity or by
    ``.ind``: the candidate group here is built from a different cell than the
    one ``tqc_models.p4232_operations()`` describes, so the BCS indices need not
    line up even though the operations do.
    """
    R = np.rint(np.asarray(rotation)).astype(int)
    t = np.mod(np.asarray(translation, dtype=float), 1.0)
    t = np.where(np.abs(t - 1.0) < 1e-6, 0.0, t)
    return (tuple(R.ravel().tolist()), tuple(np.round(t, 6).tolist()))


def op_keys(symmetries):
    return {op_key(s.rotation, s.translation) for s in symmetries}


def k_action_matrix(group, op):
    """The operation's action on the model's reduced in-plane k, as a matrix."""
    act = group.k_action_of(op)
    assert act is not None, f"operation #{op.ind} has no in-plane k-action"
    return np.round(act(np.eye(group.dim_k)), 12)


def k_action_classes(group):
    """``{k-action: [ind, ...]}`` — operations the detector cannot tell apart."""
    classes = {}
    for s in group.symmetries:
        if s.time_reversal:
            continue
        classes.setdefault(tuple(k_action_matrix(group, s).ravel()), []).append(int(s.ind))
    return classes


def own_solve(model, group, ind, nk=8, tol=1e-8):
    """``_find_intertwiners`` on operation ``ind``'s **own** k-action.

    The independent oracle for the slab tests.  ``_validate_group`` solves a
    (k-action, antiunitarity) class once and writes *every* member's entry off
    that one :class:`~pythtqc.symmetry._IntertwinerSpace` (``solved_on`` names the
    member it was solved on), so a comparison between two members of a class is
    one measurement read twice.  This poses the member's problem again, on its
    own, and returns ``(dim, survives)`` — the nullspace dimension and the
    keep/break decision recomputed test-side from the same rule
    ``_validate_group`` applies (``dim > 0`` and the best residual inside ``tol``).
    """
    sym = next(s for s in group.symmetries if int(s.ind) == ind)
    space = _find_intertwiners(model, group.k_action_of(sym),
                               _measure._detection_grid(int(group.dim_k), nk),
                               antiunitary=bool(sym.time_reversal), tol=tol)
    best = float(np.min(space.residuals)) if space.dim else np.inf
    return int(space.dim), bool(space.dim and best <= tol)


def spin_mirror(model):
    """The spinful horizontal mirror ``sigma_h = -i s_z`` in pythtb's flat basis.

    Spin is the fast index (``state = orbital * nspin + spin``), so the operator
    is ``kron(I_norb, -i s_z)`` for any orbital content that is even under
    ``z -> -z``.  Its orbital part can only ever be a diagonal of +-1, and it is
    *never* a multiple of the identity, because ``-i s_z`` is traceless.
    """
    return np.kron(np.eye(model.norb), -1j * np.diag([1.0, -1.0]))


def p4232_candidate_cell(inversion, generic=(0.13, 0.27, 0.41)):
    """A pythtb model whose *cell* carries the P4_232 candidate operations.

    Built exactly like ``tqc_models.p4232_symmetry_cell()``: the two
    orbital-bearing 2a sites (two orbitals each, as in the golden) plus a
    crystal-field decoration made of the orbit of a generic point.  With
    ``inversion=False`` the decoration is that function's own 24-fold SG-208
    orbit and the cell reads **P4_232**; with ``inversion=True`` the inversion
    image of the orbit is added as well, the point set becomes invariant under
    ``P4_232 + inversion``, and the cell reads **Pm-3n (223)** — the 48-operation
    supergroup this file audits against.

    Only the geometry matters: ``_group_of`` never looks at a Hamiltonian.
    """
    cell = tqm.p4232_symmetry_cell(generic_site=generic)
    orbital_sites, deco = cell["positions"][:2], cell["positions"][2:]
    if inversion:
        image = np.mod(-deco, 1.0)
        image = np.where(np.abs(image - 1.0) < 1e-9, 0.0, image)
        deco = np.vstack([deco, image])
    # two orbitals on each 2a site, matching the golden's (d_z2, d_x2-y2) pair
    positions = np.vstack([np.repeat(orbital_sites, 2, axis=0), deco])
    m = TBModel(Lattice(np.eye(3), positions.tolist(), periodic_dirs=[0, 1, 2]),
                spinful=False)
    m.set_onsite([0.0] * positions.shape[0])
    typat = [0] * 4 + [1] * deco.shape[0]
    return m, typat, deco.shape[0]


# ---------------------------------------------------------------------------
# module-scoped adapter objects (_validate_group runs the detector per operation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lin3g_group(lin3g):
    """lin3g's slab cell group, grey (the adapter's ``include_TR=True`` default)."""
    g = _spacegroup._group_of(lin3g)
    assert not isinstance(g, Refusal), g
    return g


@pytest.fixture(scope="module")
def lin3g_validated(lin3g, lin3g_group):
    """What lin3g's Hamiltonian obeys, at the API's default grid and tolerance."""
    return _measure._validate_group(lin3g, lin3g_group)


@pytest.fixture(scope="module")
def km_odd_group(km_odd):
    """Kane-Mele's cell group with the *default* species labelling -> P6/mmm."""
    g = _spacegroup._group_of(km_odd, include_TR=False)
    assert not isinstance(g, Refusal), g
    return g


@pytest.fixture(scope="module")
def km_odd_validated(km_odd, km_odd_group):
    return _measure._validate_group(km_odd, km_odd_group)


@pytest.fixture(scope="module")
def p4232_candidate_group():
    """Pm-3n (223) with time reversal: the 48 candidate operations x TR."""
    model, typat, ndeco = p4232_candidate_cell(inversion=True)
    assert ndeco == 48, "the generic orbit and its inversion image must be disjoint"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = _spacegroup._group_of(model, typat=typat)
    assert not isinstance(g, Refusal), g
    return g


@pytest.fixture(scope="module")
def p4232_validated(p4232, p4232_candidate_group):
    """The audit: the golden's H(k) measured against the Pm-3n candidate set."""
    return _measure._validate_group(p4232, p4232_candidate_group)


# ---------------------------------------------------------------------------
# 1. the cell of the P4_232 golden, from lattice + orbital positions alone
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_p4232_bare_orbital_cell_is_body_centred_and_is_served_as_a_supergroup(p4232):
    """The bare 2a orbital point set is bcc, so its cubic cell is a supercell.

    Re-pinned in wave R4a, which retired the ``non_primitive_cell`` refusal this
    used to assert.  ``fixture_models`` records that this point set reads Im-3m
    (SG 229), and the adapter now says so *directly*: ``Im-3m1' #229.141``, 192
    operations = 48 rotations times the two cosets of the centring translation
    ``(1/2,1/2,1/2)`` -- 96 unitary, and the same again antiunitary -- with
    ``lattice_index == 2``.  No table is served off it —
    the tables belong to the finer (bcc) lattice — and it is
    ``realized_group`` that measures the translation and descends;
    ``tests/test_lattice_descent.py`` carries that.
    """
    r = _spacegroup._group_of(p4232)
    assert r, str(r)
    assert (r.number_str, r.name, r.size) == ("229.141", "Im-3m1'", 192)
    assert sum(1 for s in r.symmetries if not s.time_reversal) == 96
    assert r._lattice_index == 2
    assert r._fractional_translations == ((0.5, 0.5, 0.5),)  # the 1/2 1/2 1/2 centring
    assert r._tables_ready is False
    gate = G.wyckoff_positions_spglib(r)
    assert isinstance(gate, Refusal) and gate.gate == "lattice_not_descended"


# ---------------------------------------------------------------------------
# 2. the run01-style audit on a real golden: the cell allows Pm-3n, H obeys SG 208
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_p4232_hamiltonian_realizes_exactly_space_group_208(p4232_candidate_group,
                                                            p4232_validated):
    """48 candidate operations x TR go in; the 24 of SG 208 (x TR) come out.

    The candidate cell is Pm-3n (223) = ``P4_232 + inversion`` -- the 48
    point-group operations of the bcc cell the bare 2a orbitals form, with the
    ``1/2 1/2 1/2`` translations attached to half of them.  The paper's
    Hamiltonian breaks inversion (``t2 != 0`` makes the eight body-diagonal
    bonds inequivalent), so every inversion-composed candidate must die, and the
    survivors must be the 24 operations of SG 208 -- *matched by rotation and
    translation*, since the BCS indices of a Pm-3n cell have no reason to agree
    with anything in ``tqc_models``.

    The 48 broken operations are exactly the improper ones, each with an empty
    nullspace: ``refuted_by="no symmetry matrix"`` is the detector's *proof* (no
    matrix at all intertwines H, unitary or otherwise), not a tolerance verdict
    -- and the smallest singular value of each broken constraint sits seven
    orders of magnitude above the nullspace cut.
    """
    g, v = p4232_candidate_group, p4232_validated
    assert (g.sg_number, g.name, g.size) == (223, "Pm-3n1'", 96)   # 48 unitary + 48 antiunitary
    # record echoes, not behaviour: `n_kpts` is `8**dim_k` off `_detection_grid`
    # and `tol` is the argument copied back.  Pinned so a changed default shows up
    assert v.n_kpts == 512 and v.tol == 1e-8      # 8**3, the 3D defaults

    survivors = [s for s in g.symmetries if int(s.ind) in v.indices]
    unitary = [s for s in survivors if not s.time_reversal]
    antiunitary = [s for s in survivors if s.time_reversal]
    assert len(unitary) == 24                     # point group O -- the SG 208 content
    assert len(antiunitary) == 24                 # spinless real hoppings: TR survives too
    assert v.closed, "the surviving set must be a subgroup"

    # the survivors ARE SG 208, operation for operation (paper's generator closure)
    sg208 = {op_key(R, t) for R, t in tqm.p4232_operations()}
    assert op_keys(unitary) == sg208
    assert op_keys(antiunitary) == sg208           # TR times the same 24 rotations
    assert all(round(np.linalg.det(s.rotation)) == 1 for s in unitary), "O has no reflections"
    # the 4_2 screw: half the survivors carry the nonsymmorphic translation
    screws = [s for s in unitary if np.abs(np.mod(s.translation, 1.0) - 0.5).max() < 1e-6]
    assert len(screws) == 12
    # roundoff: measured worst 1.6e-15 on arm64/OpenBLAS 0.3.33 (3.5e-15 where the
    # lock was captured), 64x clear of the bound
    assert max(v.residuals[i] for i in v.indices) < 1e-13

    # every inversion-composed candidate dies, and with a proof
    assert len(v.discrepancies) == 48                            # 24 unitary + 24 antiunitary
    assert all(d["refuted_by"] == "no symmetry matrix" for d in v.discrepancies)
    assert all(round(np.linalg.det(g.symmetries[d["ind"] - 1].rotation)) == -1 for d in v.discrepancies)
    assert sum(1 for d in v.discrepancies if g.symmetries[d["ind"] - 1].time_reversal) == 24
    smallest = [v._records[d["ind"]]["smallest_singular_value"] for d in v.discrepancies]
    # re-measured 2026-09-09 on arm64/OpenBLAS 0.3.33: 0.7822 for all 48 (the
    # spread over them is 1e-15), against the 1e-8 nullspace cut
    assert min(smallest) > 1e-2
    assert all(v._nullspace_dims[d["ind"]] == 0 for d in v.discrepancies)


# ---------------------------------------------------------------------------
# 3. Kane-Mele survival
# ---------------------------------------------------------------------------

def test_kane_mele_hamiltonian_keeps_exactly_the_Pminus6m2_subgroup(km_odd, km_odd_group,
                                                                    km_odd_validated):
    """The staggering demotes the cell's P6/mmm to the P-6m2 the fixture records.

    Twelve of the 24 candidate operations survive, and they are *exactly* the
    twelve of the ``typat=[0, 1]`` group -- the operations that do not exchange
    the two sublattices.  The twelve that die are the sublattice-exchanging
    ones (C2z, C6, the three C2' axes, inversion, S3, and three of the six
    vertical mirrors), each with an empty nullspace: ``delta != 0`` is what
    breaks them.  With the two sublattices typed apart the cell itself reads
    that group, ``P-6m2`` (#187, 12 operations): the fixture's recorded
    ``slab_sg_number``.
    """
    g, v = km_odd_group, km_odd_validated
    assert (g.sg_number, g.size) == (191, 24)     # the cell's maximal group
    assert len(v.indices) == 12 and v.closed
    assert all(d["refuted_by"] == "no symmetry matrix" for d in v.discrepancies)
    # the broken constraints' smallest singular value: 0.0223 for all twelve
    # (spread below 1e-15), re-measured 2026-09-09 on arm64/OpenBLAS 0.3.33 --
    # the 0.42 the lock used to record is not reproducible here.  Not a roundoff
    # quantity; the bound only has to stay decades above the 1e-8 nullspace cut,
    # which is what makes "no symmetry matrix" a proof rather than a tolerance verdict
    assert min(v._records[d["ind"]]["smallest_singular_value"] for d in v.discrepancies) > 1e-3
    # a covariance residual at 1e-14 is roundoff, and this one travels: measured
    # 2.3e-14 on arm64/OpenBLAS 0.3.33 against 4.4e-15 where the lock was
    # captured, a factor 5 on the same code.  1e-13 left it 4.4x clear, which a
    # different BLAS could cross; 1e-12 is two decades of headroom
    assert max(v.residuals[i] for i in v.indices) < 1e-12

    # the survivors are the P-6m2 operations, matched by rotation and translation;
    # the cell typed apart reads that group, the number the fixture records
    staggered = _spacegroup._group_of(km_odd, include_TR=False, typat=[0, 1])
    assert (staggered.sg_number, staggered.name, staggered.size) == (187, "P-6m2", 12)
    assert tqm.KANE_MELE["odd"].slab_sg_number == 187
    survivors = [s for s in g.symmetries if int(s.ind) in v.indices]
    assert op_keys(survivors) == op_keys(staggered.symmetries)

    # C3 survives (both nontrivial powers), and it is what the honeycomb keeps
    rot = [np.rint(s.rotation).astype(int) for s in survivors]
    threefold = [R for R in rot
                 if not np.array_equal(R, np.eye(3, dtype=int))
                 and np.array_equal(R @ R @ R, np.eye(3, dtype=int))]
    assert len(threefold) == 2                    # C3 and C3^2 about z

    # mirrors: 3 of the 6 vertical mirrors survive, and sigma_h "survives" (see below)
    vertical = [s for s in g.symmetries if not s.time_reversal
                and round(np.linalg.det(s.rotation)) == -1 and round(s.rotation[2, 2]) == 1]
    assert len(vertical) == 6                     # P6/mmm has 6 vertical mirrors
    assert sum(1 for s in vertical if int(s.ind) in v.indices) == 3
    horizontal = [s for s in g.symmetries if not s.time_reversal
                  and np.array_equal(np.rint(s.rotation), np.diag([1, 1, -1]))]
    assert len(horizontal) == 1 and int(horizontal[0].ind) in v.indices


def test_kane_mele_sigma_h_survives_only_because_the_slab_hides_it(km_odd, km_odd_group,
                                                                   km_odd_validated):
    """The slab artefact -- and Rashba really does break the mirror.

    One expects the ``sigma_h``-type operations to die once ``rashba != 0``
    mixes the spin components, and physically they do: the covariance residual
    of the actual operator ``-i s_z`` is 1.31 on a Hamiltonian of size 2.2.  The
    detector nevertheless keeps them, because ``sigma_h`` leaves every in-plane k
    fixed and so shares its intertwiner problem with the identity.  Measured
    lock, with the mechanism asserted so the count cannot be misread as a
    statement about the mirror.

    The inheritance is read off ``solved_on`` and the answer re-measured by
    :func:`own_solve` -- and here that oracle discriminates, since six of the
    twelve classes survive and six die: the independent solve on the
    ``sigma_h``-composed member's own k-action reproduces both verdicts, so what
    the class inherited is what its own problem says.
    """
    g, v = km_odd_group, km_odd_validated
    classes = k_action_classes(g)
    assert len(classes) == 12                     # 24 unitary operations, 12 distinct k-actions
    own = {}
    for inds in classes.values():
        assert len(inds) == 2                     # {g, sigma_h * g}
        a, b = inds
        assert v._records[b]["solved_on"] == a == v._records[a]["solved_on"], \
            f"#{b} did not inherit #{a}'s solution space"
        # the inherited verdict, re-measured on sigma_h * g's own k-action
        own[b] = own_solve(km_odd, g, b)
        assert own[b][0] == v._nullspace_dims[b] == v._nullspace_dims[a]
        assert own[b][1] == (b in v.indices) == (a in v.indices)
    # so the decidable content is 6 of the 12 in-plane classes: the p3m1 wallpaper
    # group -- and the independent solves split 6/6 the same way
    assert sum(1 for inds in classes.values() if inds[0] in v.indices) == 6
    assert sum(1 for dim, survives in own.values() if survives) == 6
    assert tqm.KANE_MELE["odd"].group == "p3m1"

    k = np.array([[0.13, 0.29], [0.37, 0.11], [0.21, 0.47]])
    H = km_odd.hamiltonian(k_pts=k, flatten_spin_axis=True)
    U = spin_mirror(km_odd)
    assert np.abs(U @ H @ U.conj().T - H).max() > 0.5    # measured 1.31, |H|max = 2.22


# ---------------------------------------------------------------------------
# 4. _assign_orbitals: the goldens' orbitals on the groups' site orbits
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_lin3g_orbitals_sit_on_the_1a_and_2c_orbits_of_the_slab_group(lin3g, lin3g_group):
    """The Wieder quartet at 1a and the s pair at 2c, in the slab group's letters.

    The wallpaper letters of the SM (1a and 2c of p4mm) become **a** and **f**
    of the slab group P4/mmm: ``a = (0,0,0)`` and ``f = (0,1/2,0)``, the latter
    being the two-site orbit ``(1/2,0,0), (0,1/2,0)``.  spglib's independent
    per-site answer agrees, which is the cross-check ``OrbitalAssignment`` ships.
    """
    asg = _tables._assign_orbitals(lin3g, lin3g_group)
    assert not isinstance(asg, Refusal), asg
    assert asg.orbits == ((0,), (1, 2))          # site 0 alone; sites 1 and 2 one orbit
    assert asg.wyckoff_spglib == ("a", "f")     # measured lock; p4mm's 1a and 2c
    assert asg.wyckoff_spglib_dataset == ("a", "f", "f")
    assert asg.site_symmetry == ("4/mmm", "mmm", "mmm")
    # a computed site-to-Wyckoff distance: measured exactly 0.0 here because the
    # sites sit at 0 and 1/2, both exact in binary, but it is a residual -- the
    # honeycomb's thirds measure 1.1e-16 for the same quantity
    assert asg.max_distance < 1e-12
    # the 1a site is fixed by the whole group; each 2c site by half of it
    assert len(asg.site_group[0]) == 32 and len(asg.site_group[1]) == 16

    # the letters are the group's own Wyckoff table, not a typed constant
    letters = {w.letter: w for w in G.wyckoff_positions_spglib(lin3g_group)}
    assert letters["a"].n_free == 0 and np.allclose(letters["a"].offset, [0, 0, 0])
    assert letters["f"].n_free == 0 and np.allclose(letters["f"].offset, [0, 0.5, 0])

    # orbital <-> band bookkeeping (states, not bands): LIN3G's band sets
    nspin = lin3g.nspin
    on_1a = len(asg.site_orbitals[0]) * nspin
    on_2c = sum(len(asg.site_orbitals[s]) for s in asg.orbits[1]) * nspin
    bands = tqm.LIN3G.band_sets
    assert on_1a == len(bands["P2"]) + len(bands["Q2"]) == 4   # the 1a EBR, split P2 | Q2
    assert on_2c == len(bands["P4"]) == 4                      # the s@2c EBR, all occupied
    assert on_1a + on_2c == tqm.LIN3G.nbands == 8


@pytest.mark.slow
def test_p4232_orbitals_form_the_two_site_2a_orbit(p4232, p4232_candidate_group):
    """Both d orbitals of both sites sit on one orbit -- Wyckoff a, the paper's 2a.

    The assignment is run against the Pm-3n candidate cell, whose 2a orbit is
    the same ``(0,0,0), (1/2,1/2,1/2)`` pair; the golden's four bands are one
    EBR induced from it.  ``wyckoff_spglib_dataset`` is deliberately *not*
    asserted here: ``_assign_orbitals`` computes that cross-check by re-running
    spglib on the **model's own** cell, which for this golden is the bare bcc
    point set (Im-3m), so that field describes a different group from the one
    being assigned against.  ``site_symmetry`` is read off the operations of
    the group assigned against, and is asserted.
    """
    asg = _tables._assign_orbitals(p4232, p4232_candidate_group)
    assert not isinstance(asg, Refusal), asg
    assert asg.orbits == ((0, 1),)               # the two 2a sites, one orbit
    assert asg.site_orbitals == ((0, 1), (2, 3))  # (d_z2, d_x2-y2) on each site
    assert asg.wyckoff_spglib == ("a",)         # 2a
    assert np.allclose(asg.site_positions, [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert list(asg.orbit_of_orbital) == [0, 0, 0, 0]
    assert len(asg.site_group[0]) == 48        # 24 unitary (m-3, the 2a site group) x TR
    assert asg.site_symmetry == ("m-3", "m-3")
    unitary_ind = {int(s.ind) for s in p4232_candidate_group.u_symmetries}
    assert len(set(asg.site_group[0]) & unitary_ind) == 24
    # as for lin3g: measured exactly 0.0 (the 2a sites are at 0 and 1/2), asserted
    # as the residual it is
    assert asg.max_distance < 1e-12

    # the whole 4-band set is the one EBR induced from this orbit
    assert p4232.norb * p4232.nspin == tqm.P4232.nbands == 4
    assert len(tqm.P4232.band_sets["EBR4"]) == 4


# ---------------------------------------------------------------------------
# 5. k_grid membership on a golden
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_k_grid_membership_on_the_p4232_cubic_group(p4232_candidate_group):
    """The cubic points GM, X, M, R -- all on a 4x4x4 mesh, only GM on 3x3x3."""
    g = p4232_candidate_group
    hsp = G.high_symmetry_points(g)
    assert set(hsp) == {"GM", "X", "M", "R"}
    assert np.allclose(hsp["X"], [0.0, 0.5, 0.0])
    assert np.allclose(hsp["M"], [0.5, 0.5, 0.0])
    assert np.allclose(hsp["R"], [0.5, 0.5, 0.5])

    even, odd = G.k_grid(g, 4), G.k_grid(g, 3)
    assert even.divisions == (4, 4, 4) and len(even.points) == 64
    assert all(even.contains.values()) and even.missing == ()
    assert odd.contains["GM"] is True
    assert set(odd.missing) == {"X", "M", "R"}
    assert len(even.irreducible) == 10 and len(odd.irreducible) == 4   # measured locks
    _verify_membership(even)
    _verify_membership(odd)


def _verify_membership(grid):
    """The ``contains`` record must agree with a direct search of **the pythtb mesh**.

    ``k_grid`` returns a pythtb :class:`pythtb.Mesh` (A3r: the mesh currency is
    pythtb's, not a private point array), and ``grid.points`` is derived from
    it rather than stored alongside — so the membership ledger is checked
    against the mesh object a caller would actually hand to a ``WFArray``.
    """
    from pythtb import Mesh

    assert isinstance(grid.mesh, Mesh), type(grid.mesh)
    assert grid.mesh.filled and grid.mesh.is_grid
    dim_k = grid.mesh.dim_k
    # the record's points ARE the mesh's, in the mesh's own flat order
    assert np.array_equal(grid.points, np.asarray(grid.mesh.flat))
    assert np.array_equal(np.asarray(grid.mesh.get_k_points()).reshape(-1, dim_k),
                          grid.points)
    # `divisions` keeps the 3-vector including the padded directions (which
    # carry 1); the mesh has one axis per *periodic* direction, so the counts agree
    assert len(grid.mesh.shape_axes) == dim_k
    assert int(np.prod(grid.mesh.shape_axes)) == int(np.prod(grid.divisions))
    assert grid.points.shape == (int(np.prod(grid.divisions)), dim_k)

    for name, k in grid.hsp.items():
        delta = ((grid.points - k) + 0.5) % 1.0 - 0.5
        hit = bool(np.any(np.abs(delta).max(axis=1) < 1e-9))
        assert hit == grid.contains[name], (name, grid.divisions)
        assert (name in grid.missing) == (not hit)


# ---------------------------------------------------------------------------
# 6. k-action-class verdicts (the A1c annotation)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_slab_goldens_annotate_sigma_h_partners_as_shared(lin3g_validated,
                                                          km_odd_validated,
                                                          p4232_validated):
    """Slab survivors carry ``"undetermined"``; class structure is the σ_h pairing.

    For a slab group every (k-action, antiunitarity) class is the pair
    ``{g, σ_h·g}``, so *every* survivor's verdict is inherited from its class:
    lin3g's 32 tested ops fall into 16 two-member classes (8 unitary + 8
    antiunitary) and all 32 "survivors" are ``"undetermined"`` — the honest
    annotation for the mechanism section 3 proves (the physical σ_h is broken
    yet reads as surviving).  Kane–Mele: 24 tested ops in 12 two-member
    classes; the 12 survivors are all shared, and the 12 dead are all
    ``"broken"`` — a class failure breaks both members individually.

    The class structure is the discriminating half and is computed here from the
    served ``k_action_classes``.  The verdict rule itself is **bookkeeping**:
    ``verdicts`` is written from class size and membership in the kept set alone,
    and ``ValidatedGroup.undetermined`` is defined as the survivors carrying the
    ``"undetermined"`` label, so what is asserted of it is the words each kind of
    class carries, never a per-operation equality with the rule that wrote it.
    The sharing itself is witnessed by ``solved_on``, and that it inherits the
    *right* verdict is measured against an independent solve, both in section 3.

    One outcome word per operation, and the two views of ``indices`` they cut:
    ``"broken"`` for every member of a failed class, ``"undetermined"`` for a
    survivor that shares its class, ``"obeyed"`` for a survivor whose class is a
    singleton.  A slab carries the first two and never the third.  The negative
    control is a genuine 3D primitive group: rotations are faithful on 3D k in a
    primitive cell, so Pm-3n's 96 candidate operations (48 unitary + 48
    antiunitary) are 96 singleton classes, nothing is inherited, and every
    verdict is the operation's own -- ``"obeyed"`` for the 48 that survive,
    ``"broken"`` for the 48 the Hamiltonian refutes, ``undetermined`` empty.
    """
    for validated, n_ops, n_classes in ((lin3g_validated, 32, 16),
                                        (km_odd_validated, 24, 12)):
        classes = validated.k_action_classes
        assert sum(len(c) for c in classes) == n_ops
        assert len(classes) == n_classes
        assert all(len(c) == 2 for c in classes)
        # every survivor shares its class, so every survivor is "undetermined";
        # and nothing is lost on the way in: an `out_of_plane` discrepancy never
        # reaches `verdicts` at all (neither golden has one, and the guard that
        # used to hide it is gone)
        assert set(validated.undetermined) == set(validated.indices)
        assert set(validated.broken) <= set(validated.verdicts)

    # lin3g: everything survives, so `undetermined` == all 32 and `broken` is
    # empty -- a failure's word has no subject here
    assert len(lin3g_validated.indices) == 32
    assert lin3g_validated.broken == ()
    # Kane-Mele: 12 die
    km = km_odd_validated
    assert len(set(km.broken)) == 12
    # the words a slab carries: the survivors undetermined, the dead broken, and
    # `obeyed` and `undetermined` the two halves of `indices`
    assert set(km.verdicts.values()) == {"broken", "undetermined"}
    assert km.obeyed == ()
    assert km.undetermined == km.indices and len(km.undetermined) == 12

    # the negative control: singleton classes, every verdict the operation's own
    v = p4232_validated
    assert sum(len(c) for c in v.k_action_classes) == 96
    assert len(v.k_action_classes) == 96
    assert all(len(c) == 1 for c in v.k_action_classes)
    assert set(v.verdicts.values()) == {"obeyed", "broken"}
    assert v.obeyed == v.indices and len(v.obeyed) == 48
    assert v.undetermined == ()
