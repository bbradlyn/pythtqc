"""Smoke anchors for the golden fixtures: each builder is the published model.

Every fixture is checked against an oracle that is not this package -- a
paper's closed form, or an independent code's golden -- and the builders are
checked for being deterministic and independent, which the session cache and
the tests that mutate a model both rely on:

* **lin3g** -- a numerical cross-check against the pythtb-1.8 golden in
  ``tests/data``.
* **P4_232** -- the pythtb build reproduces the paper's analytic H(k)
  matrix-for-matrix, and the paper's degeneracies come out: doubled everywhere
  without ``H_k^1``, and with it doubled only where symmetry enforces it.
* **inversion-symmetric Kane-Mele** -- the declaration a free-standing layer of
  it takes: ``A1`` names no irrep of its ``-6m2`` site.
* **the hexagonal fragile golden** (Po/Watanabe/Vishwanath) -- checked against
  the published closed-form Bloch Hamiltonian, since its bond geometry is
  printed in no paper, plus every spectrum and gap the papers state.
* **the spinless p_x,p_y honeycomb** (Cano *et al.*, the P4_232 paper's other
  model) -- reproduced against Eqs. (2)-(4) matrix-for-matrix *in the paper's
  own conjugate Fourier convention*, with the gap of its ``tci`` regime.
* **SG 199 (I2_1 3)** (Flicker *et al.*, App. F.1) -- reproduced against
  Eqs. (96)-(98) including the embedding matrix V(u,k).
* **the screw-group and re-gluing atomic controls** -- their literal ``{R|t}``
  are ``irreptables``' own shipped operations, in order.

Grids are deliberately coarse: these are fixture smoke tests, not convergence
studies.
"""

import numpy as np
import pytest

from pythtqc import groups as tqg

import tqc_models as tqm   # loaded by conftest.py -- see its docstring

RNG_SEED = 20260818


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_kpts(dim_k, n=20, seed=RNG_SEED):
    return np.random.default_rng(seed).random((n, dim_k))


# ---------------------------------------------------------------------------
# the builders
# ---------------------------------------------------------------------------

def test_builders_are_deterministic_and_independent():
    """Two calls give identical Hamiltonians, and mutating one does not touch the other."""
    a, b = tqm.lin3g(), tqm.lin3g()
    k = _random_kpts(2, n=5)
    # *identical*, and meant literally: one deterministic builder called twice in
    # one process, so bit-equality is the claim.  `assert_allclose(atol=0.0)`
    # would have kept its default rtol=1e-7, i.e. 1.2e-6 on these entries.
    assert float(np.abs(a.hamiltonian(k, flatten_spin_axis=True)
                        - b.hamiltonian(k, flatten_spin_axis=True)).max()) == 0.0
    a.set_onsite([1.0, 0.0, 0.0, 0.0], mode="add")
    assert not np.allclose(a.hamiltonian(k, flatten_spin_axis=True),
                           b.hamiltonian(k, flatten_spin_axis=True))


# ---------------------------------------------------------------------------
# lin3g
# ---------------------------------------------------------------------------

def test_lin3g_matches_the_pythtb18_golden():
    """The local builder is the published model: energies match pythtb 1.8 to < 1e-9.

    The golden at ``tests/data/lin3g_pythtb18.npz`` was written by the
    validation script the builder was lifted from, under pythtb 1.8 and the
    original spin-resolved library -- a different code, so it is an oracle for
    the copy in ``fixture_models.py`` rather than a second copy of it.
    Measured max|delta| = 1.2e-14 on the development machine, 2.1e-14 on an
    M1 with OpenBLAS.
    """
    golden = tqm.lin3g_golden()
    e = tqm.lin3g().solve_ham(golden["k_test"])
    delta = float(np.max(np.abs(np.sort(e, 1) - np.sort(golden["e_test"], 1))))
    assert delta < 1e-9, f"lin3g vs pythtb-1.8: {delta:.3e}"


# ---------------------------------------------------------------------------
# P4_232 (SG 208) -- arXiv:1711.11045
# ---------------------------------------------------------------------------

def test_p4232_reproduces_the_papers_hamiltonian(p4232):
    """The pythtb build equals Eqs. (5)-(7) of arXiv:1711.11045, matrix-for-matrix.

    Not merely isospectral: the paper's Bloch phases carry the orbital offsets
    (convention I), the same convention pythtb's ``hamiltonian`` uses, so the
    two 4x4 matrices must agree entry by entry.
    """
    k = _random_kpts(3, n=20)
    H = p4232.hamiltonian(k)
    ref = np.array([tqm.p4232_bloch_hamiltonian(2 * np.pi * kk) for kk in k])
    residual = float(np.abs(H - ref).max())
    assert residual < 1e-12, f"pythtb build differs from paper Eqs. (5)-(7): {residual:.3e}"


def test_p4232_bare_Hk_is_doubly_degenerate_and_gapped(p4232):
    """Paper: "The band structure is doubly-degenerate and gapped when t1,2,3 != 0."

    With ``t4 = t5 = t6 = 0`` every k must carry two two-fold levels, and the
    half-filling gap must stay open.

    Paper Fig. 3a: ``H_k + H_k^1`` loses the doubling almost everywhere, and
    keeps it at Gamma, X, M and R -- two two-fold levels at each, 2 + 2 and
    never 4 + 0, because they are the two-dimensional little-group irreps
    induced by the ``(d_z2, d_x2-y2)`` site irrep.  That is symmetry-enforced,
    not accidental, so it holds at parameters the paper never used.
    """
    m = tqm.p4232_cano2018(t4=0.0, t5=0.0, t6=0.0)
    E = m.solve_ham(m.k_uniform_mesh([11, 11, 11]))
    np.testing.assert_allclose(E[:, 1], E[:, 0], atol=1e-12)
    np.testing.assert_allclose(E[:, 3], E[:, 2], atol=1e-12)
    assert float(E[:, 2].min() - E[:, 1].max()) > 0.4

    # the full model: the extra degeneracy is gone almost everywhere (it
    # survives only on the symmetry-protected lines and planes)
    E = p4232.solve_ham(p4232.k_uniform_mesh([11, 11, 11]))
    split = E[:, 1] - E[:, 0]
    assert np.mean(split < 1e-8) < 0.10
    np.testing.assert_allclose(float(np.median(split)), 0.1659, atol=1e-3)
    # particle-hole partner: the top pair splits by the same amount
    np.testing.assert_allclose(E[:, 3] - E[:, 2], split, atol=1e-10)

    # ... and at the high-symmetry points the 2 + 2 is enforced, at random parameters
    hsp = {"Gamma": [0.0, 0.0, 0.0], "X": [0.0, 0.5, 0.0],
           "M": [0.5, 0.5, 0.0], "R": [0.5, 0.5, 0.5]}
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(4):
        pars = dict(zip(("t1", "t2", "t3", "t4", "t5", "t6"), rng.uniform(-0.4, 0.4, 6)))
        m = tqm.p4232_cano2018(**pars)
        for k in hsp.values():
            E = m.solve_ham(np.array([k, k]))[0]
            np.testing.assert_allclose([E[1], E[3]], [E[0], E[2]], atol=1e-12)


# ---------------------------------------------------------------------------
# inversion-symmetric Kane-Mele
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_substrate_word_is_refused_on_the_free_standing_layer_with_the_words_its_site_takes():
    """``A1`` declared on the free-standing layer of inversion-symmetric Kane-Mele.

    ``A1`` is a label of the substrate site group ``3m``; the free-standing site
    group is ``-6m2``, whose labels are ``A1'``, ``A2'``, ``A1''``, ``A2''``,
    ``E'``, ``E''``, so ``A1`` is refused there as ``unknown_label``, and the
    refusal lists the words that site takes, the double-valued coreps they lift
    to included.  The two groups are pinned by the baseline's
    ``layer/kane_mele`` records.
    """
    model = tqm.kane_mele_inversion()
    wrong = tqg.realized_group(model, environment="free-standing",
                               characters={0: "A1"})
    assert not wrong and wrong.gate == "unknown_label"
    # the words a declaration may be written in at this spinful -6m2 site: the
    # orbital irreps, and the double-valued coreps they lift to
    assert set(wrong.context["irreps"]) == {"A1'", "A2'", "A1''", "A2''", "E'", "E''",
                                            "-E1", "-E2", "-E1 + -E3", "-E2 + -E3"}


# ---------------------------------------------------------------------------
# the hexagonal fragile golden
# ---------------------------------------------------------------------------

def test_pwv_reproduces_the_published_closed_form_bloch_hamiltonian():
    """The fixture against its own oracle -- the check the figure-read needs.

    The bond *geometry* of this model is printed in none of the three papers
    that use it: it exists only in Fig. 10 of arXiv:1807.09729 / Fig. 1(c) of
    arXiv:1709.06551.  Po *et al.* do print the **closed-form Bloch
    Hamiltonian** (their App. A), which needs no figure at all -- so the
    symmetrised bond construction is checked against it here rather than taken
    on trust, for the four-band ``H_0`` and for the six-band interpolation.

    And every number the papers state about the model, measured:
    ``E(GM) = (-1.2, -1.2, 0.8, 0.8)``, ``E(M) = (-0.6536 x2, 0.787 x2)`` and
    ``E(K) = (-1.066 x2, 0.6062, 0.7258)`` are Bradlyn *et al.*'s Fig. 2(a);
    the ``0.39t`` gap and the ``~2t`` valence bandwidth are Po *et al.*'s
    text.  Note which gap is which: the paper's 0.39 is the **indirect** gap;
    the direct gap is 0.5604.
    """
    for six, mu in ((False, 0.0), (True, 0.0), (True, 0.3)):
        model = tqm.pwv_fragile(mu, six_band=six)
        assert model.nstate == (6 if six else 4)
        k = _random_kpts(2, n=6)
        E = np.asarray(model.solve_ham(k))
        for i, kk in enumerate(k):
            ref = np.linalg.eigvalsh(tqm.pwv_bloch_hamiltonian(kk, mu, six_band=six))
            np.testing.assert_allclose(np.sort(E[i]), np.sort(ref), atol=1e-13,
                                       err_msg=f"six={six} mu={mu} k={kk}")

    model = tqm.pwv_fragile()
    for name, k, want in (("GM", [0.0, 0.0], [-1.2, -1.2, 0.8, 0.8]),
                          ("K", [1 / 3, 1 / 3], [-1.066, -1.066, 0.6062, 0.7258]),
                          ("M", [0.5, 0.0], [-0.6536, -0.6536, 0.787, 0.787])):
        E = np.asarray(model.solve_ham(np.array([k, k])))[0]
        np.testing.assert_allclose(E, want, atol=1e-4, err_msg=name)
    k = np.stack(np.meshgrid(*[np.linspace(0, 1, 121, endpoint=False)] * 2,
                             indexing="ij"), -1).reshape(-1, 2)
    E = np.asarray(model.solve_ham(k))
    assert float(E[:, 2].min() - E[:, 1].max()) == pytest.approx(0.390557, abs=1e-5)
    assert float(np.min(E[:, 2] - E[:, 1])) == pytest.approx(0.560386, abs=1e-5)
    assert float(E[:, 1].max() - E[:, 0].min()) == pytest.approx(2.209161, abs=1e-5)


# ---------------------------------------------------------------------------
# the spinless p_x, p_y honeycomb
# ---------------------------------------------------------------------------

def _pxpy_kcart(k_red):
    """Reduced k of the builder's cell -> the cartesian k Eqs. (3)-(4) use."""
    return np.linalg.solve(tqm.PXPY_LATTICE, 2 * np.pi * np.asarray(k_red, float))


def test_pxpy_reproduces_the_papers_hamiltonian_in_its_own_convention():
    """Eqs. (2)-(4) matrix-for-matrix -- conjugated, and that is the whole story.

    The supplement's Fourier transform is ``c_k = sum_R e^{+i k.(R+r)} c_R``,
    which puts ``e^{-i k.delta}`` in Eq. (3); pythtb's convention I puts
    ``e^{+i k.(R + r_j - r_i)}``.  The two differ by complex conjugation, i.e.
    by ``k -> -k``, and nothing else -- so the check is ``H_pythtb(k) ==
    conj(H_paper(k))`` **matrix-for-matrix**, not merely spectrally.  If only
    the spectra agreed, the sign of the ``sigma_x`` term (the
    ``delta_2 <-> delta_3`` labelling of Fig. 1a) would go unverified.

    The ``tci`` regime is gapped as Fig. 1c says, and its *direct* gap is an
    exact number: it sits at ``K``, where the supplement's closed forms give
    ``(3/8)(-4(t_pi-t_sigma) + sqrt3 x) + 3 sqrt3 x/8 = 0.4794228634``, and a
    90x90 grid through ``K`` reproduces it to 1e-12.  The *indirect* gap is
    not: the valence maximum sits at no special point and creeps down with
    refinement (0.47575 at 90 divisions, 0.47515 at 120), so it is bounded
    rather than pinned.
    """
    for regime in ("tci", "semimetal"):
        model = tqm.pxpy_honeycomb(regime=regime)
        p = tqm.PXPY_PARAMS[regime]
        k = _random_kpts(2, n=8, seed=RNG_SEED + 3) - 0.5
        H = model.hamiltonian(k, flatten_spin_axis=True)
        ref = np.array([tqm.pxpy_bloch_hamiltonian(_pxpy_kcart(kk), **p) for kk in k])
        assert np.abs(H - ref.conj()).max() < 1e-14, regime
        # ... and the conjugation really is k -> -k, not a transposition typo
        back = np.array([tqm.pxpy_bloch_hamiltonian(-_pxpy_kcart(kk), **p) for kk in k])
        assert np.abs(ref.conj() - back).max() < 1e-14, regime

    ts, tp, x = (tqm.PXPY_PARAMS["tci"][n] for n in ("t_sigma", "t_pi", "x"))
    exact = 3 / 8 * (-4 * (tp - ts) + np.sqrt(3) * x) + 3 * np.sqrt(3) * x / 8
    grid = np.stack(np.meshgrid(*[np.linspace(0, 1, 90, endpoint=False)] * 2,
                                indexing="ij"), -1).reshape(-1, 2)
    E = np.asarray(tqm.pxpy_honeycomb().solve_ham(grid))
    assert float(np.min(E[:, 2] - E[:, 1])) == pytest.approx(exact, abs=1e-12)
    assert 0.47 < float(E[:, 2].min() - E[:, 1].max()) < exact


# ---------------------------------------------------------------------------
# the SG-199 golden
# ---------------------------------------------------------------------------

def _i213_kcart(k_red):
    return np.linalg.solve(tqm.I213_LATTICE, 2 * np.pi * np.asarray(k_red, float))


def test_i213_reproduces_the_appendix_bloch_hamiltonian():
    """Eqs. (96)-(98) matrix-for-matrix, the embedding matrix ``V`` included.

    Appendix F's whole point is that ``V(u,k)`` -- the orbital positions in the
    Bloch phase -- is physical even though the spectrum ignores it.  pythtb's
    convention I carries exactly that, so the comparison is direct and the
    ``u``-dependence is under test rather than divided out.
    """
    for u in (tqm.I213_U, 0.2, -0.3):
        model = tqm.i213_flicker2018(u=u)
        k = _random_kpts(3, n=6, seed=RNG_SEED + 5) - 0.5
        H = model.hamiltonian(k, flatten_spin_axis=True)
        ref = np.array([tqm.i213_bloch_hamiltonian(_i213_kcart(kk), u=u) for kk in k])
        assert np.abs(H - ref).max() < 1e-14, u
    # the spectrum, and only the spectrum, is u-independent (App. F, para. 1)
    k = _random_kpts(3, n=5, seed=RNG_SEED + 6)
    np.testing.assert_allclose(np.asarray(tqm.i213_flicker2018(u=0.12).solve_ham(k)),
                               np.asarray(tqm.i213_flicker2018(u=0.31).solve_ham(k)),
                               atol=1e-12)
    Ha = np.asarray(tqm.i213_flicker2018(u=0.12).hamiltonian(k))
    Hb = np.asarray(tqm.i213_flicker2018(u=0.31).hamiltonian(k))
    assert np.abs(Ha - Hb).max() > 0.1, "the embedding must move H(k) itself"


# ---------------------------------------------------------------------------
# the screw-group and re-gluing atomic controls (#91, #152, #80, #171)
# ---------------------------------------------------------------------------
#
# These are the only fixtures built here rather than ported, so the smoke
# suite checks their *inputs* as well as their outputs: that the hard-coded
# {R|t} really are irreptables' own shipped operations for the group.

def test_screw_fixture_operations_are_irreptables_own():
    """The literals in ``fixture_models`` are the shipped ``{R|t}``, in order.

    Everything downstream -- the orbits, the symmetrized hoppings, the
    operation *indices* the oracle quotes -- is built from these eight (six)
    tuples, so they are checked against the source of truth rather than
    trusted.  ``irreptables`` prints the translations to five decimals, hence
    the 1e-5 tolerance on ``t``.

    ``tables_status()`` is called first on purpose, and not only as an
    assertion: ``irreptables.irreps`` imports ``irrep.utility``, so importing
    it *before* ``irrep.spacegroup_irreps`` leaves ``irreptables.irreps``
    half-built when ``irrep`` runs its own ``from irreptables.irreps import
    IrrepTable``, and ``irrep`` then falls back to ``DummyMethodTables``
    silently for the rest of the process.  Touching ``irrep`` first is what the
    library itself does -- ``_group_of`` calls ``tables_status`` before anything
    reaches ``irreptables`` -- and it is why this hazard has never bitten it.
    """
    assert tqg.tables_status()["tables"], "irrep must load the real tables first"
    from irreptables.irreps import IrrepTable

    for number_str, ops in (("91", tqm.P4122_OPS), ("152", tqm.P3121_OPS),
                            ("80", tqm.I41_OPS), ("171", tqm.P62_OPS)):
        table = IrrepTable(number_str, False, magnetic=False)
        assert len(table.symmetries) == len(ops), number_str
        for i, ((R, t), sym) in enumerate(zip(ops, table.symmetries), start=1):
            np.testing.assert_array_equal(np.asarray(R, dtype=int).reshape(3, 3),
                                          np.asarray(sym.R, dtype=int),
                                          err_msg=f"#{number_str} op {i}")
            np.testing.assert_allclose(np.asarray(t, dtype=float),
                                       np.asarray(sym.t, dtype=float), atol=1e-5,
                                       err_msg=f"#{number_str} op {i}")
