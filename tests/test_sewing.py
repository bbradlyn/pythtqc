r"""Phase A3 regressions: sewing matrices, characters and corep content.

What each test closes, and the number that closes it:

* **lin3g corep content at Γ / X / M**, for the occupied six and for the P₄, P₂
  and Q₂ blocks separately, against the content a reference implementation
  printed for the SM Sec. 3G measurement.  The reference printed
  *sewing angles* and its own two-way label ``Ebar1`` (angles ±π/4) /
  ``Ebar2`` (±3π/4); this suite reproduces the angle sets **exactly** and adds
  the ``irreptables`` corep labels, whose mapping onto the reference's naming
  is ``Ebar1 = -GM7/-M7`` and ``Ebar2 = -GM6/-M6`` at the shared phase anchor
  (the orbital-trivial ``s@1a`` reference orbital, which is what both the
  reference and ``_build_rep``'s default pin the representation against).
* **the ``V(G)`` fold** — the boundary factor of the reference's
  ``U~ = V(G) D_g``, here supplied by ``SpaceGroupRep.sewing_operator``.  The
  control replaces it by the bare ``D`` and shows exactly where that bites.
* **one diagonalization**: ``sewing_matrix`` has one ``states`` slot, so bras
  and kets from two different solves cannot enter one call, and states from
  somewhere else are measured against ``H(k)`` and refused.
* **``frontier_subspace``** against the reference's printed PsP frontier
  (1.000000 at Γ, 0.982512 at X, 0.977650 at M, ± paired at all three).
* **the two-tier threshold contract** against the same reference block
  thresholds (1.0e-14 at Γ — which is the flat rule, the conditioned floor there
  being 7.3e-15 — 1.8e-14 at X, 8.3e-14 at M) and a measured list of readouts on
  which the flat and conditioned rules disagree.

The eigenvector currency is pythtb's (A3r)
------------------------------------------
Every call here hands the module one of pythtb's own objects — a solved
``WFArray``, the ``(evals, evecs)`` pair ``TBModel.solve_ham`` returns, or the
model itself — because the package deliberately owns **no** eigenvector
container.  Three tests lock that seam rather than the physics on top of it:
``test_the_two_state_sources_agree`` (WFArray path ≡ model path, on content and
on sewing angles), ``test_a_feature_sector_wfarray_sews`` (phase B2's payoff:
the *same* sewing code on a ``P±`` sector ``WFArray`` from
``tqc_models.psp_sector_wfarrays``, whose fibers carry three states out of
eight and no energies at all), and the ``k``-provenance gates that replace what the
old package-local container used to enforce by holding onto its own ``k``.

Why the 2D goldens are presented in a mirror-broken slab cell
-------------------------------------------------------------
This is the phase's one architectural surprise and it is load-bearing, so it is
stated here rather than buried.

``groups._group_of`` answers "what does this cell allow?", and for a 2D model
padded into a slab cell with every orbital at ``z = 0`` the answer always
contains the horizontal mirror ``sigma_h``: a flat point set is ``sigma_h``
symmetric no matter what the Hamiltonian does.  lin3g's slab cell therefore
reads P4/mmm1' (#123.340) and Kane–Mele's reads P-6m21' (#187.210), and phase
A1's ``_validate_group`` cannot refute the ``sigma_h``-composed operations
either, because their action on *in-plane* k duplicates an in-plane partner's
(``test_groups_fixtures.py``: the mechanism on Kane–Mele in
``test_kane_mele_sigma_h_survives_only_because_the_slab_hides_it``, lin3g's
sixteen pairs in ``test_slab_goldens_annotate_sigma_h_partners_as_shared``).
Both Hamiltonians nevertheless **break** ``sigma_h``: their in-plane spin-orbit
terms (lin3g's ``vMz``, Kane–Mele's Rashba) are odd under it, which
``_build_rep`` detects — it leaves the ``sigma_h``-composed operations out of
the representation, and forcing them in raises.

The consequence for A3 is sharp: half of the unitary little group at every
high-symmetry point is then unrepresented, and a corep label of #123 or #187
describes a Hamiltonian that represents *all* of its little group.  A
representation of that group, which only a build by hand makes, is short of
its little group, and ``corep_content`` raises on it (locked below); the
labels these models actually carry are those of the group they really
realize — the p4mm lift, i.e. **P4mm (#99)** for lin3g and **P3m1 (#156)** for
Kane–Mele.  The way to obtain that group without touching the Hamiltonian is
the one the P4₂32 fixture already uses (and the one arXiv:1711.11045 itself
describes for its own model): decorate the *cell* with non-orbital atoms that
create a crystal field of the right symmetry.  Here two atoms of **different**
species at ``(0, 0, ±dz)`` do it — they break every operation with
``R[2,2] = -1`` and leave the slab centred on ``z = 0``, so the model's own
padded presentation is untouched.  ``groups._slab_presentation`` first
re-presents the 2D model in a 3D slab lattice with a *bit-identical*
Hamiltonian (it measures that itself, and refuses on any difference) so that
the decorated cell and the model share one padded lattice, which
``groups._assign_orbitals`` requires.
"""

import re

import numpy as np
import pytest
from pythtb import Lattice, Mesh, TBModel, WFArray

import pythtqc as tqc
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import symmetry as tqs
from pythtqc.refusal import InternalError, Refusal

import tqc_models

_ULP = float(np.finfo(float).eps)     # 2.22e-16; the unit of every roundoff bound below
_FEW_ULP = 8 * _ULP                   # one sandwich of O(1) unitaries, 1.78e-15


# ---------------------------------------------------------------------------
# scaffolding: one solve, in pythtb's currency
# ---------------------------------------------------------------------------

#: One ``solve_ham`` at one k -> ``(evals, evecs)``, the pair ``sewing_matrix``
#: takes.  This *is* the package's eigenvector currency: there is no ``tqc``
#: container to put it in, so a test that wants several readouts from one
#: diagonalization keeps the pair and hands it to each call — which is what
#: ``characters``/``corep_content`` do internally as well, through this same
#: function (``_resolve_states``' ``source=None`` branch).
solve_at = tqw.solve_at


def columns(evecs):
    """pythtb returns eigenvectors as **rows**; a sandwich ``V† O V`` wants columns."""
    return np.asarray(evecs).T


def hamiltonian_at(model, k, **params):
    """``H(k)``, recomputed on demand — the operator tier never stores it."""
    return np.asarray(model.hamiltonian(k_pts=np.atleast_2d(np.asarray(k, dtype=float)),
                                        flatten_spin_axis=True, **params))[0]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lin3g_slab(lin3g):
    """lin3g on its bare slab cell, P4/mmm1' (#123.340), represented by a build by hand."""
    group = tqg._group_of(lin3g)
    assert group, str(group)
    assignment = tqg._assign_orbitals(lin3g, group)
    validated = tqg._validate_group(lin3g, group, nk=6)
    rep = tqs._build_rep(lin3g, group, assignment, validated=validated)
    assert rep, str(rep)
    return lin3g, group, assignment, validated, rep


@pytest.fixture(scope="module")
def lin3g_wfarray(lin3g_undeclared):
    """lin3g solved into a pythtb ``WFArray`` on a 4x4 mesh that hits GM, M and X."""
    model = lin3g_undeclared.model
    mesh = Mesh(["k", "k"])
    mesh.build_grid((4, 4))
    wf = WFArray(model.lattice, mesh, spinful=True)
    wf.solve_model(model)
    return wf


# ---------------------------------------------------------------------------
# 0.  the slab cell's own group: labels that do not apply
# ---------------------------------------------------------------------------

def test_corep_content_raises_on_a_representation_short_of_its_little_group(lin3g_slab):
    """A representation that leaves out part of its own little group is a defect.

    Built by hand on P4/mmm1', the representation leaves out the
    sigma_h-composed half of the unitary little group at every high-symmetry
    point, which no call returns.  Reading its characters raises, naming the
    operations not carried, and so does reading coreps, in the table of every
    point as at each one.
    """
    model, group, _, _, rep = lin3g_slab
    hsp = tqg.high_symmetry_points(group)
    expected = {"GM": (5, 6, 7, 8, 9, 10, 11, 12), "M": (5, 6, 7, 8, 9, 10, 11, 12),
                "X": (5, 6, 9, 10)}
    for name, k in hsp.items():
        with pytest.raises(InternalError, match=re.escape(
                f"not carried: {list(expected[name])}")):
            tqw.characters(rep, k, bands=6)
        with pytest.raises(InternalError, match="not carried"):
            tqw.corep_content(rep, k, bands=6)

    with pytest.raises(InternalError, match="not carried"):
        tqw.hsp_table(rep, bands=6)


# ---------------------------------------------------------------------------
# 1.  the phase closer: lin3g content at Gamma / X / M
# ---------------------------------------------------------------------------

#: The reference's C4 (Gamma, M) and C2 (X) sewing angle sets, in units of
#: pi, of every band block of the coupled model.
OUT100_ANGLES = {
    ("GM", "P_4"): (-0.75, -0.25, 0.25, 0.75),
    ("GM", "P_2"): (-0.75, 0.75),
    ("GM", "P_6"): (-0.75, -0.75, -0.25, 0.25, 0.75, 0.75),
    ("GM", "Q_2"): (-0.25, 0.25),
    ("M", "P_4"): (-0.75, -0.25, 0.25, 0.75),
    ("M", "P_2"): (-0.25, 0.25),
    ("M", "P_6"): (-0.75, -0.25, -0.25, 0.25, 0.25, 0.75),
    ("M", "Q_2"): (-0.75, 0.75),
    ("X", "P_6"): (-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
    ("X", "P_2"): (-0.5, 0.5),
    ("X", "Q_2"): (-0.5, 0.5),
}

#: The measured corep content, block by block and point by point.  Read against
#: ``out100``'s two-way naming via ``Ebar1 = -GM7/-M7`` (angles ±pi/4) and
#: ``Ebar2 = -GM6/-M6`` (±3pi/4): out100's ``P_6`` = ``Ebar1 + 2xEbar2`` at
#: Gamma is ``-GM7 + 2x-GM6`` here, and ``2xEbar1 + Ebar2`` at M is
#: ``2x-M7 + -M6``.
LIN3G_CONTENT = {
    ("GM", "P_4"): ({"-GM6": 1, "-GM7": 1}, (0, 1)),
    ("GM", "P_2"): ({"-GM6": 1}, (0,)),
    ("GM", "P_6"): ({"-GM6": 2, "-GM7": 1}, (0, 1, 0)),
    ("GM", "Q_2"): ({"-GM7": 1}, (0,)),
    ("M", "P_4"): ({"-M6": 1, "-M7": 1}, (0, 1)),
    ("M", "P_2"): ({"-M7": 1}, (0,)),
    ("M", "P_6"): ({"-M6": 1, "-M7": 2}, (0, 1, 1)),
    ("M", "Q_2"): ({"-M6": 1}, (0,)),
    ("X", "P_4"): ({"-X5": 2}, (0, 0)),
    ("X", "P_2"): ({"-X5": 1}, (0,)),
    ("X", "P_6"): ({"-X5": 3}, (0, 0, 0)),
    ("X", "Q_2"): ({"-X5": 1}, (0,)),
}

LIN3G_BLOCKS = {"P_4": (0, 1, 2, 3), "P_2": (4, 5), "P_6": tuple(range(6)),
                "Q_2": (6, 7)}


def test_lin3g_corep_content_matches_the_reference_port(lin3g_undeclared):
    """PHASE CLOSER -- lin3g content at Gamma/X/M against out100 (= SM Sec. 3G).

    out100 measured, for the coupled model:
      ``P_6``  Gamma ``Ebar1 + 2xEbar2``, M ``2xEbar1 + Ebar2``, X ``+-i x3``;
      ``P_2``  Gamma ``Ebar2``,           M ``Ebar1``,           X ``+-i``;
      ``Q_2``  Gamma ``Ebar1``,           M ``Ebar2``,           X ``+-i``;
    and, on the decoupled control, ``P_4`` = ``Ebar1 + Ebar2`` at both Gamma and
    M -- the content of an EBR induced from a maximal Wyckoff position.  Every
    one of those is reproduced here as an ``irreptables`` label set, and the
    underlying sewing angle sets are identical to out100's printout, to 1e-9.
    Angles are what out100 printed and what survives any relabelling: the C4
    sewing eigenvalues of a spinful two-dimensional corep sit at ``exp(±i pi/4)``
    (``Ebar1``) or ``exp(±3i pi/4)`` (``Ebar2``), and the C2 ones at ``±i``.

    All four blocks are read off **one** ``solve_ham`` per k: the pair is
    handed to each call, which is the whole of the one-diagonalization
    discipline once the eigenvector currency is pythtb's.
    """
    model, group, _, _, rep = lin3g_undeclared
    hsp = tqg.high_symmetry_points(group)
    for point in ("GM", "X", "M"):
        k = hsp[point]
        evals, evecs = solve_at(model, k)
        H = hamiltonian_at(model, k)
        assert float(np.abs(H @ columns(evecs) - columns(evecs) * evals[None, :]).max()) \
            <= 1e-13
        assert tqw.multiplets(evals) == ((0, 1), (2, 3), (4, 5), (6, 7)), \
            "all eight lin3g bands are Kramers doublets at every TRIM"
        thr = tqw.two_tier_thresholds(H, tqw.multiplet_separation(evals))
        rot = np.array([[0, -1], [1, 0]]) if point in ("GM", "M") else -np.eye(2, dtype=int)
        ind = next(i for i in rep.little_group(k)
                   if not rep._operation(i).time_reversal
                   and np.array_equal(np.rint(rep._operation(i).rotation).astype(int)[:2, :2],
                                      rot))

        for block, bands in LIN3G_BLOCKS.items():
            out = tqw.corep_content(rep, k, states=(evals, evecs), bands=bands)
            assert out, (point, block, str(out))
            counts, partition = LIN3G_CONTENT[(point, block)]
            assert out.counts == counts, (point, block)
            assert out.partition == partition, (point, block)
            assert out.kpname == point
            assert max(out.table.unitarity.values()) <= out.thresholds.effective
            assert max(out.table.commutators.values()) <= tqw.FLAT_ZERO
            # the isolated frontier block, not an energy-ordered slice
            assert out.isolation > 0.5

            # the reference's printed sewing angles, from the same solve
            if (point, block) in OUT100_ANGLES:
                sw = tqw.sewing_matrix(rep, ind, k, states=(evals, evecs), bands=bands)
                assert isinstance(sw, tqw.SewingMatrix), str(sw)
                assert sw.unitarity <= thr.effective
                want = np.array(OUT100_ANGLES[(point, block)])
                assert np.abs(sw.angles / np.pi - want).max() < 1e-9, (point, block)


# ---------------------------------------------------------------------------
# 2.  the V(G) fold
# ---------------------------------------------------------------------------

def test_the_VG_fold_is_what_makes_the_boundary_points_identifiable(lin3g_undeclared):
    """The reference's ``U~ = V(G) D_g``, and exactly where dropping it bites.

    ``sewing_operator`` folds the Bloch operator back to k, which multiplies it
    by ``diag(exp(2 pi i G . tau))`` -- real ``diag(±1)`` here, and non-trivial
    precisely on the 2c orbitals at ``(1/2, 0)`` and ``(0, 1/2)``.  Measured:

    * at Gamma every wrap is zero and ``U = D`` exactly;
    * at X and at M the wrap is non-zero for most operations and flips a 2c sign,
      so ``max|U - D| = 2`` at both;
    * the *characters* nevertheless only change at **M** (by ±sqrt(2)), because
      every non-identity character at X vanishes anyway (measured: the fold moves
      them by 2.0e-15, inside this k's own conditioned threshold of 1.83e-14)
      -- and at M dropping the fold turns a clean label into an
      ``integer_multiplicity`` refusal with defect 0.5.

    So the fold is not decoration and it is not only a boundary-TRIM cosmetic:
    it is the difference between ``-M6/-M7`` and no answer at all.
    """
    model, group, _, _, rep = lin3g_undeclared
    hsp = tqg.high_symmetry_points(group)
    operator_shift, char_shift, verdicts, thresholds_at = {}, {}, {}, {}
    for name in ("GM", "X", "M"):
        k = hsp[name]
        evals, evecs = solve_at(model, k)
        V = columns(evecs)
        thresholds_at[name] = tqw.two_tier_thresholds(
            hamiltonian_at(model, k), tqw.multiplet_separation(evals))
        lg = tqg.little_group(group, k)
        rows = tqw.multiplets(evals)
        raw = np.zeros((len(rows), len(lg.indices)), dtype=complex)
        folded = np.zeros_like(raw)
        shift = 0.0
        for c, ind in enumerate(lg.indices):
            D = rep.matrices[int(ind)]
            U = rep.sewing_operator(int(ind), k)
            shift = max(shift, float(np.abs(U - D).max()))
            for r, m in enumerate(rows):
                Vm = V[:, list(m)]
                raw[r, c] = np.trace(Vm.conj().T @ D @ Vm)
                folded[r, c] = np.trace(Vm.conj().T @ U @ Vm)
        operator_shift[name] = shift
        char_shift[name] = float(np.abs(raw - folded).max())
        dims = [len(m) for m in rows]
        good = tqg.identify_from_characters(group, k, folded, multiplet_dims=dims)
        bad = tqg.identify_from_characters(group, k, raw, multiplet_dims=dims)
        assert good, (name, str(good))
        verdicts[name] = bad

    assert operator_shift["GM"] == 0.0
    assert operator_shift["X"] == pytest.approx(2.0)
    assert operator_shift["M"] == pytest.approx(2.0)

    assert char_shift["GM"] == 0.0
    assert char_shift["X"] <= thresholds_at["X"].effective, \
        f"X characters are zero with or without the fold: {char_shift['X']:.3e}"
    assert char_shift["M"] == pytest.approx(np.sqrt(2.0), abs=1e-9)

    assert verdicts["GM"] and verdicts["X"], "dropping the fold is invisible there"
    assert not verdicts["M"] and verdicts["M"].gate == "integer_multiplicity"
    assert verdicts["M"].value == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# 3.  one diagonalization: one states slot, and the k it came from
# ---------------------------------------------------------------------------

def test_sewing_has_one_states_slot_and_gates_its_k_provenance(lin3g_undeclared):
    """What replaces the old container's self-declared k: two measured gates.

    The signature has exactly one ``states`` slot, so bras and kets from two
    different solves cannot be expressed at all.  The remaining door -- handing
    in states that came from *somewhere else* -- is closed by measurement, not
    by bookkeeping: supplied ``(evals, evecs)`` are checked against ``H(k)``
    itself, and a ``WFArray`` that does not sample ``k`` refuses instead of
    quietly solving there.

    The second slot names an operation: an index into the representation's
    operation list -- the same number the operation carries as ``.ind`` -- and
    the record hands it back under that name.  Off the operation's little group
    there is no sewing operator, and that refusal is the result.
    """
    model, group, _, _, rep = lin3g_undeclared
    hsp = tqg.high_symmetry_points(group)
    kG, kM = hsp["GM"], hsp["M"]
    at_gamma = solve_at(model, kG)

    # the second slot is the operation, by position or by keyword, and required
    op = next(i for i in rep.little_group(kG)
              if not rep._operation(i).time_reversal
              and not np.array_equal(np.rint(rep._operation(i).rotation).astype(int),
                                     np.eye(3, dtype=int)))
    sw = tqw.sewing_matrix(rep, op, kG, states=at_gamma, bands=tuple(range(6)))
    assert isinstance(sw, tqw.SewingMatrix), str(sw)
    assert sw.operation == op
    assert int(rep._operation(sw.operation).ind) == op, \
        "the number is an index into rep's operation list, not a position in the block"
    by_keyword = tqw.sewing_matrix(rep, operation=op, k=kG, states=at_gamma,
                                   bands=tuple(range(6)))
    assert by_keyword.operation == sw.operation and by_keyword.bands == sw.bands
    assert np.array_equal(by_keyword.matrix, sw.matrix)
    with pytest.raises(TypeError, match="missing required argument 'operation'"):
        tqw.sewing_matrix(rep)

    # off the little group there is no sewing operator: the refusal is the result
    k_generic = np.array([0.31, 0.07])
    c4 = next(i for i in rep.indices
              if not rep._operation(i).time_reversal
              and np.array_equal(np.rint(rep._operation(i).rotation).astype(int)[:2, :2],
                                 [[0, -1], [1, 0]]))
    off_group = tqw.sewing_matrix(rep, c4, k_generic, states=solve_at(model, k_generic))
    assert not off_group and off_group.gate == "little_group"

    # a bare fiber of eigenvector rows is a legitimate source -- it is what a
    # WFArray carries -- but without energies the spectrum-tier gates cannot run
    bare = tqw.sewing_matrix(rep, rep.little_group(kG)[0], kG, states=at_gamma[1])
    assert isinstance(bare, tqw.SewingMatrix), str(bare)
    assert np.isnan(bare.residual)

    # Gamma's states, asked for at M: measured against H(M) and refused
    foreign = tqw.sewing_matrix(rep, rep.little_group(kM)[0], kM, states=at_gamma)
    assert not foreign and foreign.gate == "states_not_at_k"
    assert foreign.value > 1.0 and foreign.value / foreign.threshold > 1e6

    # a mesh that does not carry M refuses rather than re-solving there
    mesh = Mesh(["k", "k"])
    mesh.build_grid((3, 3))
    wf = WFArray(model.lattice, mesh, spinful=True)
    wf.solve_model(model)
    off = tqw.sewing_matrix(rep, rep.little_group(kM)[0], kM, states=wf)
    assert not off and off.gate == "k_not_on_mesh"
    assert off.value == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert tqw.characters(rep, kM, states=wf, bands=6).gate == "k_not_on_mesh"


def test_an_antiunitary_operation_sews_through_the_conjugating_branch(lin3g_undeclared):
    r"""Time reversal at M: the ket is conjugated, and the Kramers pairing proves it.

    An antiunitary operation acts as ``T u = U conj(u)``, so its sewing matrix is
    ``B = V_a^dag U conj(V_b)`` and its operator-tier check is
    ``U H(k)^* = H(k) U`` -- both branches live only in ``_sewing_core``, and
    nothing else in this file routes an antiunitary index through
    ``sewing_matrix``.  M is chosen because it is a zone-boundary TRIM: the fold
    ``G = Rk - k = (-1, -1)`` is nonzero there, so the ``V(G)`` factor is live in
    ``U`` as well.

    What closes it is a physical consequence of the conjugation, not a
    restatement of it.  For spinful time reversal the operator satisfies
    ``T conj(T) = -1`` (measured exactly: ``max|U conj(U) + 1| = 0``), and
    antiunitarity gives ``<T x | T y> = <y | x>``.  Put ``x = u_b``,
    ``y = T u_a``:

        ``<T u_b | T^2 u_a> = <T u_a | u_b>``,

    whose left side is ``-conj(B_ab)`` and whose right side is ``conj(B_ba)``.
    Hence ``B^T = -B``: **antisymmetric**, so its diagonal vanishes, its trace is
    zero, and on a single Kramers doublet the two surviving entries are
    unimodular (the doublet is closed under ``T``, so ``B`` is a 2x2 unitary
    antisymmetric matrix, ``B = beta * i sigma_y`` with ``|beta| = 1``).

    Measured at M on each of the three occupied Kramers doublets and on the
    occupied block as a whole: the matrix reproduces the hand sandwich to 0.0,
    the antisymmetry defect is at most 5.9e-17 (bounded here at 8 ulp of ``B``'s
    own O(1) scale), ``|B_01| - 1`` is at most 4.4e-16, and the commutator is
    4.9e-16 against the flat rule.  Deleting either conjugation in
    ``_sewing_core`` breaks every one of them: the first doublet's matrix moves
    by 0.989, the antisymmetry defect on the occupied block goes to 1.88, two of
    the three doublet blocks stop being unitary at all, and the commutator goes
    to 3.6.
    """
    model, group, _, _, rep = lin3g_undeclared
    k = tqg.high_symmetry_points(group)["M"]
    states = solve_at(model, k)
    evals, evecs = states
    V = columns(evecs)
    thresholds = tqw.two_tier_thresholds(hamiltonian_at(model, k),
                                         tqw.multiplet_separation(evals))

    ind_T = next(i for i in rep.little_group(k)
                 if rep._operation(i).time_reversal
                 and np.array_equal(np.rint(rep._operation(i).rotation).astype(int),
                                    np.eye(3, dtype=int)))
    assert ind_T == rep.antiunitary_operator[0], "the pure time reversal of the grey group"
    U = rep.sewing_operator(int(ind_T), k)
    assert not isinstance(U, Refusal), str(U)
    assert np.array_equal(rep.in_little_group(int(ind_T), k)[1], [-1, -1]), \
        "M is a boundary TRIM: the V(G) fold is live in U as well"
    # the premise of the antisymmetry derivation, measured on the operator: 0.0
    assert float(np.abs(U @ np.conj(U) + np.eye(U.shape[0])).max()) <= _FEW_ULP, \
        "spinful time reversal: T conj(T) = -1"

    assert tqw.multiplets(evals) == ((0, 1), (2, 3), (4, 5), (6, 7))
    for bands in ((0, 1), (2, 3), (4, 5), (0, 1, 2, 3, 4, 5)):
        sw = tqw.sewing_matrix(rep, ind_T, k, states=states, bands=bands)
        assert isinstance(sw, tqw.SewingMatrix), (bands, str(sw))
        assert sw.antiunitary is True, bands
        assert sw.bands == bands

        # (a) it is the conjugated sandwich, not the unitary one
        Vb = V[:, list(bands)]
        hand = Vb.conj().T @ (U @ np.conj(Vb))
        assert float(np.abs(sw.matrix - hand).max()) <= _FEW_ULP, bands

        # (b) B^T = -B, on a matrix whose own scale is 1
        assert float(np.abs(sw.matrix).max()) == pytest.approx(1.0, rel=1e-9, abs=0)
        assert float(np.abs(sw.matrix.T + sw.matrix).max()) <= _FEW_ULP, bands
        assert float(np.abs(np.diag(sw.matrix)).max()) <= _FEW_ULP, bands
        assert abs(sw.trace) <= _FEW_ULP, bands

        # (c) the block is closed under T, and the operator commutes with H(k)*
        assert sw.unitarity <= thresholds.effective, bands
        assert sw.commutator <= tqw.FLAT_ZERO, bands
        assert sw.residual <= 1e-13

    # on one Kramers doublet the two surviving entries are the pairing amplitude
    # itself, B = beta * i sigma_y: both unimodular, measured 1 + 4.4e-16
    doublet = tqw.sewing_matrix(rep, ind_T, k, states=states, bands=(0, 1)).matrix
    assert abs(doublet[0, 1]) == pytest.approx(1.0, rel=8 * _ULP, abs=0)
    assert abs(doublet[1, 0]) == pytest.approx(1.0, rel=8 * _ULP, abs=0)

    # the same operation is *not* a character column: corep tables carry none
    # for an antiunitary operation, and `characters` leaves it out
    table = tqw.characters(rep, k, states=states, bands=6)
    assert ind_T in table.antiunitary and ind_T not in table.columns


# ---------------------------------------------------------------------------
# 4.  labelling the isolated block, never an energy-ordered pair
# ---------------------------------------------------------------------------

def test_a_block_that_cuts_a_multiplet_is_refused_everywhere(lin3g_undeclared):
    """The occupancy boundary inside a Kramers doublet is refused, not labelled."""
    model, group, _, _, rep = lin3g_undeclared
    k = tqg.high_symmetry_points(group)["GM"]
    states = solve_at(model, k)
    ind = rep.little_group(k)[0]

    low = tqw.sewing_matrix(rep, ind, k, states=states, bands=(0,))
    assert not low and low.gate == "multiplet_split"
    tab = tqw.characters(rep, k, bands=5)
    assert not tab and tab.gate == "multiplet_split"
    out = tqw.corep_content(rep, k, bands=5)
    assert not out and out.gate == "multiplet_split"
    assert out.context["multiplet"] == (4, 5)

    ok = tqw.corep_content(rep, k, bands=6)         # the gauge-invariant block
    assert ok and ok.counts == {"-GM6": 2, "-GM7": 1}


# ---------------------------------------------------------------------------
# 5.  two-tier thresholds
# ---------------------------------------------------------------------------

def test_a_readout_where_the_two_rules_disagree_is_listed_not_reclassified(lin3g_undeclared):
    """The honest form of the result: the disagreement list.

    The classified numbers are the module's **own** sandwich, not a
    re-implementation of it: one ``sewing_matrix`` per little-group operation on
    the *whole* fiber (all eight bands, so the block is a union of complete
    multiplets and the call is legal), sliced into its off-diagonal
    multiplet-vs-multiplet blocks.  That routes the eight antiunitary operations
    at Gamma and M (four at X) through ``_sewing_core``'s conjugating branch,
    which is what makes their blocks vanish at all.

    Those inter-multiplet blocks ``<multiplet a| U |multiplet b>`` vanish by
    symmetry, and the first assertion is that the conditioned rule covers every
    one of them (measured maxima 1.5e-15 at Gamma, 2.2e-15 at X, 5.2e-14 at M,
    against effective thresholds of 1.0e-14 / 1.8e-14 / 8.3e-14).  At M they
    land *between* the two rules -- zero under the conditioned one and nonzero
    under the flat one -- which is where the conditioning bites; how many land
    there, and how far, is the machine's arithmetic (24 readouts near 3.6e-14 on
    the development machine, 28 between 4.3e-14 and 5.2e-14 on an M1 with
    OpenBLAS).  Whichever they are, they are listed; none is silently
    reclassified, and the flat rule is not widened to make them go away.

    The thresholds themselves are the reference's block thresholds,
    1.0e-14 / 1.8e-14 / 8.3e-14: ``nf = 8 eps ||H(k)||_2 / (smallest multiplet
    separation)``, computed here from pythtb's convention-I Hamiltonian and by
    the reference from its convention-II one.  The two are unitarily related,
    so both the spectral norm and the separations are convention-free and the
    numbers must agree.  At **Gamma** the reference's 1.0e-14 *is* the flat
    rule, not a reproduction of anything: the floor there is 7.31e-15 and the
    flat rule covers it, so the Gamma entry is asserted on the floor itself and
    the flat rule's dominance separately.  Every comparison passes ``abs=0``:
    ``approx``'s default **absolute** tolerance is 1e-12, four orders of
    magnitude *above* the thresholds compared.  Measured: 7.306e-15 /
    1.8306e-14 / 8.3008e-14.  The table ``characters`` builds at the same k was
    handed the same ``H`` and the same separation, and reports the same rule to
    1.3e-14 relative (it takes the norm on the Hermitized ``H(k)``).
    """
    model, group, _, _, rep = lin3g_undeclared
    hsp = tqg.high_symmetry_points(group)
    found = {}
    thresholds_at = {}
    reference = {"GM": 7.3e-15, "X": 1.8e-14, "M": 8.3e-14}
    for name, n_anti in (("GM", 8), ("X", 4), ("M", 8)):
        k = hsp[name]
        states = solve_at(model, k)
        evals = states[0]
        mult = tqw.multiplets(evals)
        thresholds = tqw.two_tier_thresholds(hamiltonian_at(model, k),
                                             tqw.multiplet_separation(evals))
        thresholds_at[name] = thresholds
        # the reference census: the floor at Gamma, the effective rule elsewhere
        assert thresholds.effective == max(thresholds.flat, thresholds.noise_floor)
        measured = thresholds.noise_floor if name == "GM" else thresholds.effective
        assert measured == pytest.approx(reference[name], rel=0.05, abs=0), name
        # the table built at that k was handed the same H and the same
        # separation, and so reports the same rule
        table_thr = tqw.characters(rep, k, bands=6).thresholds
        assert table_thr.flat == thresholds.flat
        assert table_thr.norm == pytest.approx(thresholds.norm, rel=1e-12, abs=0)
        assert table_thr.separation == pytest.approx(thresholds.separation, rel=1e-12,
                                                     abs=0)
        assert table_thr.effective == pytest.approx(thresholds.effective, rel=1e-12,
                                                    abs=0)
        readouts = {}
        antiunitary_columns = 0
        for ind in rep.little_group(k):
            sw = tqw.sewing_matrix(rep, ind, k, states=states)  # bands=None: the whole fiber
            assert isinstance(sw, tqw.SewingMatrix), (name, int(ind), str(sw))
            antiunitary_columns += bool(sw.antiunitary)
            for a, ma in enumerate(mult):
                for b, mb in enumerate(mult):
                    if a == b:
                        continue
                    readouts[f"#{ind}:{a}-{b}"] = float(
                        np.linalg.norm(sw.matrix[np.ix_(list(ma), list(mb))]))
        assert antiunitary_columns == n_anti, name
        # they all vanish by symmetry -- the conditioned rule covers the lot,
        # antiunitary operations included
        assert max(readouts.values()) <= thresholds.effective, name
        found[name] = tqw.disagreements(thresholds, readouts)

    assert found["GM"] == () and found["X"] == ()
    assert len(found["M"]) > 0, "M is where the conditioning bites"
    assert max(e["value"] for e in found["M"]) <= thresholds_at["M"].effective
    assert min(e["value"] for e in found["M"]) > thresholds_at["M"].flat
    # Gamma is the one point where the flat rule already dominates
    g_thr = tqw.characters(rep, hsp["GM"], bands=6).thresholds
    assert g_thr.noise_floor < g_thr.flat and g_thr.effective == g_thr.flat


def test_two_tier_helper_contract():
    """Unit contract of the two-tier rule, including the degenerate case.

    The floor is pinned as a *number* -- ``8 eps * 4.0 / 1.0 = 7.10543e-15`` to six
    digits -- rather than as the implementation's own expression, so that changing
    the documented constant 8 changes this test.  ``abs=0`` is not decoration:
    ``approx``'s default absolute tolerance is 1e-12, three orders of magnitude
    above the number being pinned, and with it the comparison passes for any
    floor the implementation cares to return.
    """
    thr = tqw.two_tier_thresholds(np.diag([1.0, -1.0]) * 4.0, 1.0)
    assert thr.norm == pytest.approx(4.0)
    assert thr.noise_floor == pytest.approx(7.10543e-15, rel=1e-6, abs=0)
    assert thr.noise_floor < thr.flat
    assert thr.effective == tqw.FLAT_ZERO           # the flat rule still dominates
    tight = tqw.two_tier_thresholds(10.0, 1e-6)
    assert tight.noise_floor > tqw.FLAT_ZERO and tight.effective == tight.noise_floor
    assert tight.zero(1e-10) and not tight.zero_flat(1e-10) and tight.disagree(1e-10)
    assert tqw.disagreements(tight, {"a": 1e-10, "b": 1e-16, "c": 1.0}) == (
        {"name": "a", "value": 1e-10, "zero_conditioned": True, "zero_flat": False},)
    degenerate = tqw.two_tier_thresholds(10.0, 0.0)
    assert degenerate.noise_floor == float("inf"), \
        "with degenerate multiplets nothing is conditioned"


def test_multiplet_helpers_are_the_analysis_over_pythtb_s_arrays():
    """Unit contract of the grouping the whole row structure rests on.

    ``multiplet_spread`` is asserted with ``abs=0``: ``approx``'s default absolute
    tolerance is itself 1e-12, which would swallow the planted splitting whole and
    pass on a hard-coded ``0.0``.  The planted 1e-12 is recovered to 2.2e-5
    relative (the stored double nearest ``-1.0 + 1e-12``).
    """
    w = np.array([-1.0, -1.0 + 1e-12, 0.5, 0.5, 0.5, 3.0])
    assert tqw.multiplets(w) == ((0, 1), (2, 3, 4), (5,))
    assert tqw.multiplet_separation(w) == pytest.approx(1.5)
    assert tqw.multiplet_spread(w) == pytest.approx(1e-12, rel=1e-3, abs=0)
    # a tolerance below the planted splitting resolves the pair
    assert tqw.multiplets(w, 1e-15) == ((0,), (1,), (2, 3, 4), (5,))
    # membership is tested against each cluster's *first* member, so a cluster
    # can never chain its way past tol
    assert tqw.multiplets(np.array([0.0, 0.6, 1.2]), 1.0) == ((0, 1), (2,))
    assert tqw.multiplet_separation(np.array([1.0, 1.0])) == float("inf")
    with pytest.raises(ValueError, match="ascending"):
        tqw.multiplets(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        tqw.multiplets(np.array([]))


# ---------------------------------------------------------------------------
# 6.  frontier_subspace
# ---------------------------------------------------------------------------

def test_frontier_subspace_reproduces_the_reference_PsP_frontier(lin3g_undeclared):
    """The reference's frontier: 1.000000 at Gamma, 0.982512 at X, 0.977650 at M, all paired.

    The spectrum handed in is the s_z feature spectrum ``P_6† s_z P_6`` -- built
    here from numpy alone, since :func:`frontier_subspace` is deliberately
    agnostic about what its input is (energies, feature eigenvalues, sewing
    ``-iS`` eigenvalues) and phase B is where the feature machinery lands.
    """
    model, group, _, _, rep = lin3g_undeclared
    hsp = tqg.high_symmetry_points(group)
    sz = np.kron(np.eye(model.norb), np.diag([1.0, -1.0]))
    want = {"GM": (1.000000, 6), "X": (0.982512, 2), "M": (0.977650, 2)}
    for name, (value, dim) in want.items():
        occ = columns(solve_at(model, hsp[name])[1])[:, :6]
        spec = np.linalg.eigvalsh(occ.conj().T @ sz @ occ)
        fr = tqw.frontier_subspace(spec)
        assert fr.value == pytest.approx(value, abs=1e-6), name
        assert len(fr) == dim, name
        assert fr.paired and not fr.at_zero, name
        assert len(fr.positive) == len(fr.negative) == dim // 2


def test_frontier_subspace_takes_a_block_not_a_split_pair():
    """A planted near-degeneracy is resolved as a block; the pairing flag is measured.

    ``modulus_tol`` is a window on ``|lambda|``, and it gates both memberships.
    The helper is agnostic about what the spectrum is -- energies shifted to the
    reference level, feature ``P O P`` eigenvalues, sewing ``-iS`` eigenvalues --
    so its tolerance is not an energy and is named for the quantity it is
    compared against.  One number decides two things about the same spectrum:
    an eigenvalue joins the block when its modulus is within it of the smallest
    present, and the frontier reads ``at_zero`` when that smallest modulus is
    itself within it of zero -- measured on a dimensionless ``s_z``-like
    spectrum, where an energy tolerance would be a category error.
    """
    spec = np.array([-1.0, -0.30000000005, 0.3, 0.9])
    fr = tqw.frontier_subspace(spec, modulus_tol=1e-9)
    assert fr.indices == (1, 2), "both near-degenerate states, never one of them"
    assert fr.value == pytest.approx(0.3)
    assert fr.paired and fr.positive == (2,) and fr.negative == (1,)

    tighter = tqw.frontier_subspace(spec, modulus_tol=1e-12)
    assert tighter.indices == (2,) and not tighter.paired, \
        "below the planted splitting the block collapses -- the tolerance is the statement"

    kramers = tqw.frontier_subspace(np.array([-1.0, -0.5, 0.5, 1.0]))
    assert kramers.paired and kramers.indices == (1, 2)

    at_zero = tqw.frontier_subspace(np.array([-1.0, 0.0, 0.0, 1.0]))
    assert at_zero.at_zero and not at_zero.paired, \
        "a frontier on zero has no resolved sign structure"

    single = tqw.frontier_subspace(np.array([0.7, 0.9]))
    assert single.indices == (0,) and not single.paired and not single.at_zero

    # the window is a modulus in the spectrum's own units, deciding the block ...
    feature = np.array([-1.0, -0.5, 0.5 + 1e-10, 1.0])
    wide = tqw.frontier_subspace(feature, modulus_tol=1e-9)
    assert wide.indices == (1, 2) and wide.paired and not wide.at_zero
    assert tqw.frontier_subspace(feature, modulus_tol=1e-12).indices == (1,), \
        "below the planted splitting the pair is no longer one block"
    # ... and the zero
    near_zero = np.array([-2.0, 1e-10, 2.0])
    assert tqw.frontier_subspace(near_zero, modulus_tol=1e-9).at_zero
    assert not tqw.frontier_subspace(near_zero, modulus_tol=1e-12).at_zero, \
        "the same number decides the block and the zero"
    assert wide.modulus_tol == 1e-9, "the record carries the window it was read at"

    with pytest.raises(ValueError):
        tqw.frontier_subspace([])


# ---------------------------------------------------------------------------
# 7.  the table phase A4 consumes
# ---------------------------------------------------------------------------

def test_hsp_table_shape(lin3g_undeclared):
    """The record A4's EBR machinery reads, and the flag it must gate on.

    The character table under it has columns in ``little_group(...).unitary``
    order, and an antiunitary operation never appears among them.  Both records
    belong to the model the representation was built on, and a foreign model
    is a ``ValueError``.
    """
    model, group, _, _, rep = lin3g_undeclared
    table = tqw.hsp_table(rep, bands=6)
    assert isinstance(table, tqw.HSPTable)
    assert table.complete and table.refusals == {} and table.skipped == {}
    assert set(table.points) == {"GM", "M", "X"}
    assert table.content == {"GM": {"-GM6": 2, "-GM7": 1},
                             "M": {"-M6": 1, "-M7": 2},
                             "X": {"-X5": 3}}
    assert table.partitions == {"GM": (0, 1, 0), "M": (0, 1, 1), "X": (0, 0, 0)}
    assert table.counts == {"-GM6": 2, "-GM7": 1, "-M6": 1, "-M7": 2, "-X5": 3}
    # the aggregated counts are exactly what decompose_ebrs takes
    assert set(table.counts) <= set(tqg.ebr_data(group)["labels_physics_convention"])

    # the character tables: the adapter's column order, antiunitary ops never
    for name, k in tqg.high_symmetry_points(group).items():
        lg = tqg.little_group(group, k)
        chars = tqw.characters(rep, k, bands=6)
        assert chars.columns == tuple(lg.indices)
        assert chars.columns == tuple(sorted(chars.columns))
        assert chars.antiunitary == tuple(sorted(int(s.ind) for s in lg.antiunitary))
        assert not set(chars.columns) & set(chars.antiunitary)
        assert chars.characters.shape == (len(chars.multiplets), len(chars.columns))
        assert np.allclose(chars.characters[:, 0], chars.dims), \
            "the identity column is the multiplet degeneracy"

    with pytest.raises(ValueError, match="rep.model"):
        tqw.hsp_table(rep, states=tqc_models.lin3g(), bands=6)
    with pytest.raises(ValueError, match="rep.model"):
        tqw.characters(rep, np.zeros(2), states=tqc_models.lin3g(), bands=6)


def test_hsp_table_skips_a_trivial_little_group():
    """A point with no unitary little group beyond the identity is *skipped*, not refused."""
    lat = Lattice([[1.0, 0.0, 0.0], [0.31, 1.13, 0.0], [0.17, 0.23, 1.7]],
                  [[0.0, 0.0, 0.0], [0.21, 0.37, 0.43]], periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=False)
    m.set_onsite([0.3, -0.2])
    m.set_hop(-0.4, 0, 0, [1, 0, 0])
    m.set_hop(-0.25, 0, 1, [0, 0, 0])
    m.set_hop(-0.15, 1, 1, [0, 0, 1])
    m.set_hop(0.07, 0, 1, [1, 1, 0])
    group = tqg._group_of(m, include_TR=False, typat=[0, 1])
    assert group and group.sg_number == 1
    assignment = tqg._assign_orbitals(m, group, typat=[0, 1])
    rep = tqs._build_rep(m, group, assignment,
                         validated=tqg._validate_group(m, group, nk=4), n_battery=3)
    assert rep, str(rep)
    table = tqw.hsp_table(rep, bands=1)
    assert table.complete and table.refusals == {} and table.content == {}
    assert set(table.skipped) == set(table.points)
    assert all("trivial little group" in why for why in table.skipped.values())


def test_the_degeneracy_tolerance_is_an_argument_of_the_table(lin3g_undeclared):
    """One solve, two groupings: the tolerance is a physics choice, stated per call.

    With no container to hold a "the" degeneracy tolerance, the threshold is an
    argument of the *table* and is recorded on it.  The same fiber at Gamma
    gives the four Kramers doublets at the default 1e-8 and, at a deliberately
    absurd 4.0, one four-fold row instead of two two-fold ones -- both
    internally consistent, and the ``spread``/``separation`` pair is what says
    which grouping was a safe choice (7.1e-15 against 3.98, versus 3.98 against
    8.05).

    The grouping, the gap between groups, the spread inside them and the
    threshold record built on the gap are four readings of that single choice,
    so they take it under one spelling, ``degeneracy_tol``, and the record
    echoes it back under that spelling too.
    """
    w = np.array([-1.0, -1.0 + 1e-12, 0.5, 0.5, 0.5, 3.0])
    assert tqw.multiplets(w, degeneracy_tol=1e-15) == ((0,), (1,), (2, 3, 4), (5,))
    assert tqw.multiplet_separation(w, degeneracy_tol=1e-15) == pytest.approx(
        1e-12, rel=1e-3, abs=0)
    assert tqw.multiplet_spread(w, degeneracy_tol=1e-15) == 0.0
    thr = tqw.two_tier_thresholds(4.0, 1.0, degeneracy_tol=1e-15)
    assert thr.degeneracy_tol == 1e-15, "the record carries the grouping it was built on"

    model, group, _, _, rep = lin3g_undeclared
    k = tqg.high_symmetry_points(group)["GM"]
    evals, evecs = solve_at(model, k)

    fine = tqw.characters(rep, k, states=(evals, evecs), bands=(0, 1, 2, 3))
    coarse = tqw.characters(rep, k, states=(evals, evecs), bands=(0, 1, 2, 3),
                            degeneracy_tol=4.0)
    assert fine.multiplets == ((0, 1), (2, 3)) and fine.dims == (2, 2)
    assert coarse.multiplets == ((0, 1, 2, 3),) and coarse.dims == (4,)
    assert coarse.degeneracy_tol == 4.0

    assert fine.spread == pytest.approx(0.0, abs=1e-13)
    assert fine.thresholds.separation == pytest.approx(3.9751, abs=1e-3)
    # the unsafe grouping, in the two numbers that say so: the within-multiplet
    # spread is now the *same size* as the separation, not orders below it
    assert coarse.spread == pytest.approx(3.9751, abs=1e-3)
    assert coarse.thresholds.separation == pytest.approx(8.0501, abs=1e-3)


def test_the_band_set_is_one_keyword_and_the_record_reports_its_bands(lin3g_undeclared):
    """``bands=`` selects the block; the record names the band indices it read.

    A count and the list it stands for are the same request, so the table
    records the block it measured -- ``(0, ..., 5)`` for six -- and a consumer
    comparing band sets has nothing to normalize.
    """
    model, group, _, _, rep = lin3g_undeclared
    k = tqg.high_symmetry_points(group)["GM"]

    want = tqw.hsp_table(rep, bands=6)
    assert want.bands == (0, 1, 2, 3, 4, 5)
    assert tqw.hsp_table(rep, bands=[5, 4, 3, 2, 1, 0]).bands == want.bands

    content = tqw.corep_content(rep, k, bands=6)
    assert content.counts == want.content["GM"]
    table = tqw.characters(rep, k, bands=6)
    assert table.bands == want.bands, "the same keyword on all three entry points"

    # the band set is not optional
    with pytest.raises(TypeError, match="missing required argument"):
        tqw.hsp_table(rep)


def test_the_label_tolerance_is_the_gate_the_labels_are_read_at(lin3g_undeclared):
    """``label_tol=`` is the dimensionless gate the adapter reads the labels at.

    It says two things about one character table, and both are measured here on
    the six-band block at Gamma -- three multiplets carrying two coreps.  The
    multiplicities must be integers within it, and here they miss integrality
    by a measured 1e-6 or so, two decades inside the default gate; set the gate
    below that defect and the same content refuses (``integer_multiplicity``),
    reporting the defect and the gate it was refused against.  And two rows
    share a partition class when their character vectors agree within it: at a
    gate far above the O(1) difference between these rows all three collapse
    into one class, while the labels -- rounded, never gated -- stay exactly
    what they were.  ``corep_content`` and the table over every high-symmetry
    point take the same number under the same spelling, and read the same
    labels at it.
    """
    _, group, _, _, rep = lin3g_undeclared
    k = tqg.high_symmetry_points(group)["GM"]

    fine = tqw.corep_content(rep, k, bands=6)
    assert fine.counts == {"-GM6": 2, "-GM7": 1}
    assert fine.partition == (0, 1, 0)

    defect = float(fine.content.integrality)
    assert 0.0 < defect < tqw.LABEL_TOL / 100.0, "integers, with decades to spare"

    refused = tqw.corep_content(rep, k, bands=6, label_tol=defect / 2.0)
    assert not refused and refused.gate == "integer_multiplicity"
    assert refused.threshold == defect / 2.0
    assert refused.value == pytest.approx(defect, rel=1e-9, abs=0)

    coarse = tqw.corep_content(rep, k, bands=6, label_tol=10.0)
    assert coarse.partition == (0, 0, 0), "every row agrees at a tolerance that big"
    assert coarse.counts == fine.counts
    # one keyword on both entry points
    table = tqw.hsp_table(rep, bands=6, label_tol=10.0)
    assert table.partitions["GM"] == coarse.partition
    assert table.content["GM"] == coarse.counts


# ---------------------------------------------------------------------------
# 8.  the A3r seam: pythtb's eigenvector currency
# ---------------------------------------------------------------------------

def test_the_two_state_sources_agree(lin3g_undeclared, lin3g_wfarray):
    """A solved ``WFArray`` and the model + k-list path give the same physics.

    The WFArray fiber at k and a fresh ``solve_ham`` at k are two routes to one
    diagonalization, so every gauge-invariant readout must coincide: the corep
    content and the phase-invariant partition exactly, the multiplet characters,
    the sewing eigen-angles and the two-tier thresholds to ~1e-12 -- the
    thresholds field by field, since they are floats off two routes' eigenvalues,
    and with ``abs=0`` because ``approx``'s default absolute tolerance of 1e-12 is
    far above a threshold's own magnitude.  (Measured here they agree to 0.0 --
    pythtb builds H(k) the same way on a mesh and at a single point -- but the
    contract asserted is the gauge-invariant one at a roundoff tolerance, because
    neither a multiplet's internal basis nor the last bit of a threshold is part
    of it.)
    """
    model, group, _, _, rep = lin3g_undeclared
    wf = lin3g_wfarray
    assert wf.nstates == model.nstate and tuple(wf.shape_mesh) == (4, 4)

    for name in ("GM", "M"):
        k = tqg.high_symmetry_points(group)[name]
        by_model = tqw.corep_content(rep, k, bands=6)
        by_wf = tqw.corep_content(rep, k, states=wf, bands=6)
        assert by_model and by_wf, (name, str(by_model), str(by_wf))
        assert by_wf.counts == by_model.counts, name
        assert by_wf.partition == by_model.partition, name
        assert by_wf.labels == by_model.labels, name
        assert by_wf.kpname == by_model.kpname == name
        assert by_wf.table.columns == by_model.table.columns
        assert np.abs(by_wf.table.characters - by_model.table.characters).max() < 1e-12
        for field in ("flat", "noise_floor", "effective", "norm", "separation",
                      "degeneracy_tol"):
            assert getattr(by_wf.table.thresholds, field) == pytest.approx(
                getattr(by_model.table.thresholds, field), rel=1e-12, abs=0), (name, field)

        ind = next(i for i in rep.little_group(k)
                   if not rep._operation(i).time_reversal
                   and np.array_equal(
                       np.rint(rep._operation(i).rotation).astype(int)[:2, :2],
                       [[0, -1], [1, 0]]))
        a = tqw.sewing_matrix(rep, ind, k, bands=tuple(range(6)))
        b = tqw.sewing_matrix(rep, ind, k, states=wf, bands=tuple(range(6)))
        assert np.abs(a.angles - b.angles).max() < 1e-12, name
        assert b.unitarity <= by_wf.thresholds.effective
        assert b.residual <= 1e-13

    # the whole table too, through the WFArray
    by_wf = tqw.hsp_table(rep, states=wf, bands=6)
    assert by_wf.complete, by_wf.refusals
    assert by_wf.content == tqw.hsp_table(rep, bands=6).content


def test_a_feature_sector_wfarray_sews(lin3g_undeclared):
    """Phase B2's payoff, smoke-tested: the *same* sewing code on a P+ sector.

    ``tqc_models.psp_sector_wfarrays`` builds, with pythtb alone, the ``P±``
    sector ``WFArray``\\ s whose fibers carry three states out of eight and
    **no energies at all** -- the ``nstates <= norb*nspin`` case the A3r API
    exists to support.  Nothing here
    is a corep claim; this is plumbing, and what it measures is which
    little-group operations the sector is even closed under:

    * the rotations commute with ``s_z``, so ``B`` is unitary on the P+ sector
      to 1.8e-15 -- asserted against the fiber's own roundoff scale,
      ``100 eps * nstates = 6.7e-14``, since this is an eigenvector-derived
      number on a three-state fiber and not an operator-tier one;
    * the mirrors of C4v reverse ``s_z`` and therefore map P+ onto P-, so ``B``
      is *not* unitary there (``max|B†B - 1| = 1.0``, i.e. the block is
      essentially annihilated) -- a reading that would be silently wrong if the
      unitarity number were not part of the record.

    The spectrum-tier gates cannot run on a source with no energies, and the
    record says so through ``residual = nan``.
    """
    model, group, _, _, rep = lin3g_undeclared
    mesh = Mesh(["k", "k"])
    mesh.build_grid((6, 6))
    wfa_plus, wfa_minus = tqc_models.psp_sector_wfarrays(model, mesh, list(range(6)),
                                                         n=(0, 0, 1))
    assert wfa_plus.nstates == 3 < model.nstate
    assert tuple(wfa_plus.wfs.shape) == (6, 6, 3, model.norb, model.nspin)

    k = tqg.high_symmetry_points(group)["GM"]
    c4 = next(i for i in rep.little_group(k)
              if not rep._operation(i).time_reversal
              and np.array_equal(np.rint(rep._operation(i).rotation).astype(int)[:2, :2],
                                 [[0, -1], [1, 0]]))
    sw = tqw.sewing_matrix(rep, c4, k, states=wfa_plus)
    assert isinstance(sw, tqw.SewingMatrix), str(sw)
    assert sw.matrix.shape == (3, 3) == (wfa_plus.nstates, wfa_plus.nstates)
    assert np.all(np.isfinite(sw.matrix))
    assert sw.bands == (0, 1, 2)
    assert np.isnan(sw.residual), "a sector WFArray carries states, not a spectrum"
    assert sw.commutator <= tqw.FLAT_ZERO
    fiber_roundoff = 100 * _ULP * sw.matrix.shape[0]      # 6.7e-14 for a 3-state fiber
    assert sw.unitarity <= fiber_roundoff, \
        f"the P+ sector is C4-closed: {sw.unitarity:.3e}"

    # which operations the sector is closed under, measured (no corep claim)
    closure = {}
    for ind in rep.little_group(k):
        if rep._operation(ind).time_reversal:
            continue
        closure[int(ind)] = tqw.sewing_matrix(rep, ind, k, states=wfa_plus).unitarity
    rotations = {i for i in closure
                 if abs(np.linalg.det(np.asarray(rep._operation(i).rotation)) - 1) < 1e-9}
    assert len(rotations) == 4 and len(closure) == 8
    assert max(closure[i] for i in rotations) <= fiber_roundoff
    assert min(closure[i] for i in closure if i not in rotations) == pytest.approx(1.0)

    # ... and the same on the P- sector, so the smoke test is not a one-sided accident
    minus = tqw.sewing_matrix(rep, c4, k, states=wfa_minus)
    assert minus.matrix.shape == (3, 3) and minus.unitarity <= fiber_roundoff


def test_module_surface():
    """The names the package serves: the sewing module's, the spinor declaration's
    and the symmetrizing pass's."""
    from pythtqc import spin as tqsp
    from pythtqc import symmetrization as tqsym

    for name in ("sewing_matrix", "characters", "corep_content", "hsp_table",
                 "frontier_subspace", "two_tier_thresholds", "disagreements",
                 "multiplets", "multiplet_separation", "multiplet_spread",
                 "FLAT_ZERO", "DEGENERACY_TOL"):
        assert name in tqw.__all__, name
    # `tqc.sewing` is the module (exactly as `tqc.groups` is), and `sewing_matrix`
    # is re-exported at package level
    assert tqc.sewing is tqw
    for name in ("sewing_matrix", "characters", "corep_content", "hsp_table",
                 "frontier_subspace", "multiplets", "Frontier",
                 "SewingMatrix", "CharacterTable", "CorepContent", "HSPTable"):
        assert getattr(tqc, name) is getattr(tqw, name), name
        assert name in tqc.__all__, name
    # the threshold contract and the multiplet measures are the module's own,
    # reached through `pythtqc.sewing`: the top level carries what the documents
    # teach
    for name in ("two_tier_thresholds", "disagreements", "multiplet_separation",
                 "multiplet_spread", "FLAT_ZERO", "DEGENERACY_TOL", "Thresholds"):
        assert not hasattr(tqc, name) and name not in tqc.__all__, name
    assert tqw.FLAT_ZERO == 1e-14 and tqw.DEGENERACY_TOL == 1e-8

    # the spinor-basis declaration: `tqc.spin_basis`, `tqc.SpinBasis`, the spin
    # operator it builds, `tqc.spinless_presentation`, and the module itself
    for name in ("SpinBasis", "spin_basis", "spin_operator", "spinless_presentation"):
        assert hasattr(tqc, name) and name in tqc.__all__, name
        assert getattr(tqc, name) is getattr(tqsp, name)
    assert "spin" in tqc.__all__ and tqc.spin is tqsp

    # `tqc.symmetrize` is the projection itself, not the module holding it
    assert tqc.symmetrize is tqc.symmetrization.symmetrize is tqsym.symmetrize
    assert callable(tqc.utils.save_figure)
    for name in ("symmetrization", "utils", "symmetrize"):
        assert name in tqc.__all__, name
