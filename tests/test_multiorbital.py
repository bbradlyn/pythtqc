r"""Phase A3b: multi-orbital rep hardening -- three programs, three pathologies.

The suite before this file has exactly one multi-orbital site in it -- the
``2a`` doublet of the P4_232 golden, on a hand-assembled 3D bundle.  This file is about what happens when that stops
being a special case: two new fixtures whose orbital structure is the *point*,
and the lin3g declaration machinery carried the rest of the way down the
pipeline.  Three programs, chosen so that no two of them break the same thing.

(a) **lin3g's declared content, end to end.**  Declaring ``s`` at 1a orbital
    0 and ``d_x2-y2`` at orbital 1 leaves the *matrices* where they were, and
    declaring both ``s`` refuses with the measured relative character.  What
    is carried through the rest of the pipeline here is the point of the whole
    gauge apparatus: a *deliberate* ``B1`` twist moves every corep label and
    every EBR site-irrep name while leaving ``decompose``'s **Wyckoff
    positions** exactly where they were.  Labels rest on a labelling;
    positions do not.  ``EBRDecomposition.site_content`` says so in prose; this asserts it
    in numbers.

(b) **The spinless ``p_x, p_y`` honeycomb of arXiv:1711.11045** -- the same
    paper as the P4_232 golden, its *other* model.  Two firsts:

    * a **two-dimensional channel** runs, for the first time, through the
      whole 2D pipeline -- ``realized_group`` -> ``corep_content`` ->
      ``decompose`` -- rather than on a hand-assembled 3D bundle as the P4_232
      golden does.  C_3 rotates ``(p_x, p_y)`` into each other by 120 degrees,
      which is no signed permutation of orbitals, so one channel per orbital
      is refused at covariance 0.329 and the layout glues the pair.
    * the ``sigma_h``-composed operations are **genuine**.  ``p_x`` and ``p_y``
      are in-plane and hence even under the horizontal mirror, so unlike every
      other 2D fixture in this suite the model realizes the *full* slab group
      ``P6/mmm1' (#191.234)`` -- 48 of 48 operations represented, none left
      undetermined.  The detector cannot say so (the ``sigma_h``-composed
      operations share their k-action with an in-plane partner, so all 48
      verdicts read ``"undetermined"``); the **representation layer** can, and
      the three gates are what close it -- covariance, product table and
      canonical norms all of order 1e-15 to 1e-14 here, roundoff the test
      bounds rather than digits it pins.  What the model still cannot supply
      is the *z-parity* of its own orbitals -- a 2D pythtb model has no z
      structure -- so that stays a declaration, and the record shows exactly
      that: ``E'`` narrows the admissible global characters to the four
      ``sigma_h``-even ones and ``E''`` flips every parity label wholesale.

(c) **The SG-199 (``I2_1 3``) model of arXiv:1806.09642 App. F.1** -- chosen for
    being pathological three independent ways at once: a body-centred Bravais
    lattice (so the conventional cell must be refused), a nonsymmorphic group
    **none of whose rotations is intrinsically a screw**, and a Wyckoff orbit
    whose four sites permute *between unit cells*.  It also turned up a second
    instance of the ``Fd-3m`` W-point class: at ``P`` and ``-P`` the raw
    tabulated characters disagree with the measured ones by a sign on exactly
    the fractionally-translated operations, which used to produce half-integer
    multiplicities and a refusal.  That sign was BCS's ``e^{+ik·t}`` convention
    for the lattice translations meeting pythtb's physical ``e^{-ik·t}``; the
    adapter now conjugates the factor system at the boundary and both points
    identify exactly, as ``P1 + P3`` and ``PA1 + PA3`` -- the content the EBR
    row predicts, with the two *distinct* (chiral) points keeping their own
    labels.
"""

import dataclasses
import re

import numpy as np
import pytest

from pythtqc import _sitechars as tqc_sitechars
from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import spin as tqsp
from pythtqc import symmetry as tqs

import tqc_models as tqm

GATE = tqs.EXACT_TOL          # the exact floor of _build_rep's covariance gate
_ULP = np.finfo(float).eps          # one unit of double-precision roundoff, 2.2e-16

# lin3g band blocks, from LIN3G.band_sets
P4, P2, P6, Q2 = (0, 1, 2, 3), (4, 5), 6, (6, 7)


#: One ``solve_ham`` at one k -> the ``(evals, evecs)`` pair sewing takes.  The
#: library's own, so a test and a caller feed the readouts the same object.
_solve_at = tqw.solve_at


def _same_matrices(a, b, atol=1e-14):
    return max(float(np.abs(a.matrices[i] - b.matrices[i]).max()) for i in a.indices) <= atol


# ===========================================================================
# (a)  lin3g: the declared pipeline, end to end
# ===========================================================================

@pytest.fixture(scope="module")
def lin3g_twisted(lin3g):
    """lin3g on a substrate with orbital 0 *declared* to be the ``B1`` partner."""
    out = tqg.realized_group(lin3g, typat=[0, 0, 1, 1], cross_check=False,
                             characters={0: "B1"})
    assert out, str(out)
    return out


@pytest.mark.slow
def test_the_b1_twist_moves_every_label_and_no_wyckoff_position(
        lin3g_undeclared, lin3g_twisted, lin3g_declared_cross_checked):
    """The A3a gauge measurement, carried into the EBR layer.

    Twisting by the one-dimensional character ``B1`` sends every corep label
    ``rho -> rho (x) B1`` and every EBR ``sigma@q -> (sigma (x) B1)@q``.  The
    *site irrep names* in ``decompose``'s answers therefore swap ``-E1``
    against ``-E2`` wholesale -- and the **Wyckoff positions do not move**,
    which is precisely the claim ``EBRDecomposition.site_content`` makes in
    prose, and the reason a row resting on the lowest-``l`` rule is quoted by
    its positions rather than its names.  Measured on all four lin3g band blocks at once.

    The same twist declared in function words, ``d_x2-y2`` wherever lin3g's own
    content has ``s`` and ``s`` where it has ``d_x2-y2``, is another labelling of
    the same hoppings, each row echoing the word declared for it; and the site
    content of a build whose every row is declared carries its site irreps and
    says that every row was declared.

    Only the product table is gated below.  ``P4mm1' (#99.164)`` is symmorphic --
    measured, 0 of its 256 product-table entries carry a nonzero translation
    difference -- so ``factor_residual``'s translation term vanishes identically
    and the two numbers are the same measurement written in another basis
    (bit-identical here, 6.674e-16 both).  The nonsymmorphic factor system is
    asserted where it is substantive, on ``#199`` below.
    """
    _, group, _, _, rep = lin3g_undeclared
    twisted = lin3g_twisted.rep
    assert twisted.characters.irreps == ("-E2", "-E1", "-E", "-E")
    assert max(twisted.residuals.values()) <= GATE
    assert twisted.product_residual < 1e-13

    swap = {"-E1": "-E2", "-E2": "-E1"}

    def relabelled(terms):
        return {frozenset((m, swap.get(n.split("↑")[0], n.split("↑")[0]), wy)
                          for m, n, wy in solution) for solution in terms}

    def bare(terms):
        return {frozenset((m, n.split("↑")[0], wy) for m, n, wy in solution)
                for solution in terms}

    # the solutions come with the fewest elementary band representations first,
    # then in the tables' order
    moves = {}
    for bands, positions in ((P6, (("1b", "2c"), ("1a", "1a", "1b"),
                                   ("1b", "1b", "1b"))),
                             (list(P4), (("2c",), ("1a", "1a"), ("1b", "1b"))),
                             (list(P2), (("1b",),)),
                             (list(Q2), (("1b",),))):
        a = tqe.decompose(tqw.hsp_table(rep, bands=bands), group, rep=rep, bands=bands)
        b = tqe.decompose(tqw.hsp_table(twisted, bands=bands), group, rep=twisted,
                          bands=bands)
        assert a.decomposable and b.decomposable
        assert a.positions == b.positions == positions, (
            "a global character cannot move a Wyckoff position")
        assert relabelled(a.terms) == bare(b.terms), (
            "... and it must relabel every site irrep it touches")
        assert "lowest-l" in a.site_content["determined_by"]
        assert "lowest-l" not in b.site_content["determined_by"]
        moves[tuple(bands) if isinstance(bands, list) else bands] = a.terms != b.terms

    assert moves[P6] and moves[P2] and moves[Q2], "the names must actually move"
    assert not moves[P4], (
        "P4's three solutions are each invariant under E1 <-> E2 (every one of "
        "them carries the pair, or carries neither), so this is the control that "
        "the relabelling is a relabelling and not a re-solve")

    # the standing example, spelled out: P2 and Q2 exchange their EBR rows
    p2 = tqe.decompose(tqw.hsp_table(rep, bands=list(P2)), group, rep=rep,
                       bands=list(P2))
    q2 = tqe.decompose(tqw.hsp_table(rep, bands=list(Q2)), group, rep=rep,
                       bands=list(Q2))
    p2t = tqe.decompose(tqw.hsp_table(twisted, bands=list(P2)), group, rep=twisted,
                        bands=list(P2))
    assert p2.solution_strings()[0].startswith("1 x -E2")
    assert q2.solution_strings()[0].startswith("1 x -E1")
    assert p2t.solution_strings()[0] == q2.solution_strings()[0]

    # the twist in function words, and the site content of a build whose every
    # row is declared
    model, group, assignment, validated, declared = lin3g_declared_cross_checked
    alt = tqs._build_rep(model, group, assignment, validated=validated,
                         characters=["dx2-y2", "s", "dx2-y2", "dx2-y2"])
    assert alt, str(alt)
    assert alt.characters.irreps == twisted.characters.irreps, "the same twist"
    assert [o.function for o in declared.characters.orbitals] == ["s", "dx2-y2", "s", "s"]
    assert [o.function for o in alt.characters.orbitals] == \
        ["dx2-y2", "s", "dx2-y2", "dx2-y2"], \
        "the declared content is another labelling of the same hoppings"
    d = tqe.decompose(tqw.hsp_table(declared, bands=P6), group, rep=declared)
    assert d.site_content["irreps"] == declared.characters.irreps
    assert d.site_content["determined_by"] == ("declaration",) * 4, \
        "this build's rows were all declared, and the site content says so"
    assert set(d.site_content) == {"number_str", "irreps", "determined_by", "residual"}


# ===========================================================================
# (b)  the spinless p_x, p_y honeycomb -- 2D multiplets and a real sigma_h
# ===========================================================================

@pytest.fixture(scope="module")
def pxpy_bare():
    """The full slab group ``P6/mmm1' (#191.234)``: no decoration, sigma_h kept.

    ``environment="free-standing"`` is the statement that the horizontal mirror
    is a real symmetry of *these* orbitals: the layer is suspended, the cell is
    the point set as given with ``sigma_h`` and everything composed with it
    kept, and the Hamiltonian selects.  Since wave R4c it has to be said, a
    ``dim_r = 2`` model being read on a substrate by default -- and a substrate
    would remove exactly the mirror this fixture is about.  Measured:
    ``#191.234``, 48 of 48.
    """
    out = tqg.realized_group(tqm.pxpy_honeycomb(), typat=[0, 0, 0, 0],
                             environment="free-standing", cross_check=False,
                             characters={(0, 1): "E'"})
    assert out, str(out)
    return out


@pytest.fixture(scope="module")
def pxpy_183():
    """The paper's own group ``P6mm1' (#183.186)``: the layer on a substrate.

    Restricting to the ``sigma_h``-free subgroup is legitimate (it is the
    in-plane image) and it is what arXiv:1711.11045's Sec. S1B analyses, so this
    is the reading whose labels can be compared with the paper's sentences.
    ``environment="substrate"`` is how that restriction is asked for, and since
    wave R4c it is also a ``dim_r = 2`` model's default: the model is pressed
    onto its own plane (``H(k)`` bit-identical), one shadow species goes under
    every site, and every ``sigma_h`` fiber is collapsed by definition, leaving
    the 24 in-plane operations -- measured 24 of 24, ``#183.186``, Wyckoff
    ``b``, all four states ``E``.

    The declaration moved with the reading, and only the declaration.
    ``E`` is a label of the in-plane site group ``3m``; free-standing the
    site group is ``-6m2``, which has ``E'`` and ``E''`` and no bare ``E``, and
    that is what ``pxpy_bare`` declares instead.  The refusal that names the six
    ``-6m2`` labels is itself pinned, in
    ``test_pxpy_declares_its_two_dimensional_multiplet_jointly``.
    """
    out = tqg.realized_group(tqm.pxpy_honeycomb(), typat=[0, 0, 0, 0],
                             environment="substrate", cross_check=False,
                             characters={(0, 1): "E"})
    assert out, str(out)
    assert out.environment == "substrate"
    assert len(out.validated.indices) == 24 == out.group.size
    return out


@pytest.mark.slow
def test_pxpy_one_channel_per_orbital_refuses_because_C3_mixes_px_into_py(pxpy_bare):
    """The layout a threefold axis forbids, stated instead of measured.

    ``(p_x, p_y)`` is the two-dimensional ``E'`` of the site's ``-6m2``, and a
    three-fold rotation sends ``p_x -> -p_x/2 + sqrt3 p_y/2``.  No signed
    permutation of two orbitals does that, so on a layout that gives each
    orbital its own channel -- which is what ``multiplets=`` states here, and what
    the measurement refuses to return for this model -- every candidate dies at
    the *covariance* gate, not at the internal factor and not at the product
    table, and by a wide margin (0.329 against a gate of 8.6e-14).

    This is also the suite's honest exercise of the **read-off's fallback**
    (``test_symmetry.py`` section 11): an ansatz that cannot realize the
    operation set at all still produces a clean-looking read, and what refuses
    it is the covariance gate.  The read-off then hands the build back to the
    enumeration -- ``strategy == "enumerate"``, with the reason recorded -- so
    the refusal the caller sees is the enumeration's: it carries the whole
    candidate count and says what the read-off read, and the build hands its
    kill record back beside it.
    """
    model, group, assignment, validated, _ = pxpy_bare
    singletons = [[(int(o),) for o in orbs] for orbs in assignment.site_orbitals]
    record = {}
    out = tqs._build_rep(model, group, assignment, validated=validated,
                         multiplets=singletons, include_vacuum_ops=True, n_battery=4,
                         _search_record=record)
    assert not out and out.gate == "rep_not_covariant"
    assert out.value == pytest.approx(0.3294812449, abs=1e-8)
    search = record["search"]
    assert out.context["n_candidates"] == search["n_candidates"]
    assert search["n_survivors"] == 0 and search["n_covariant"] == 0
    # the per-candidate rows, with their own count: `all(())` is True, so the
    # kill record the docstring promises is a claim only once the record is
    # known to be the whole enumeration (measured 8 distinct channel groupings)
    assert len(search["candidates"]) == search["n_distinct"] == 8
    assert all(c["killed_by"] == "covariance" for c in search["candidates"])
    assert "what the measured multiplets read off the solution spaces" in out.reason

    assert search["strategy"] == "enumerate" and search["requested"] == "readoff"
    assert "do not give one representation" in search["fallback_reason"]
    assert "(branch +1: generator #" in search["fallback_reason"], "what was read"
    assert f"The enumeration ran because {search['fallback_reason']}." in out.reason
    assert search["n_candidates"] == search["n_enumeration"]
    enumerated_record = {}
    enumerated = tqs._build_rep(model, group, assignment, validated=validated,
                                multiplets=singletons, include_vacuum_ops=True,
                                n_battery=4, strategy="enumerate",
                                _search_record=enumerated_record)
    assert not enumerated and enumerated.gate == out.gate
    assert enumerated.value == out.value

    def _rows(rec):                    # nan != nan, so compare the fields that ran
        return [(c["orientation"], c["combo"], c["n_merged"], c["covariance"],
                 c["killed_by"]) for c in rec["candidates"]]

    assert _rows(enumerated_record["search"]) == _rows(search)


@pytest.mark.slow
def test_pxpy_sigma_h_acts_as_a_scalar_and_that_scalar_is_the_declaration(pxpy_bare):
    """How the ``sigma_h`` half was validated, stated as a measurement.

    A 2D pythtb model carries no out-of-plane orbital structure, so the only
    thing ``sigma_h`` *can* be on this basis is a scalar ``eta`` with
    ``|eta| = 1`` -- and the representation measures exactly that, with
    ``D(sigma_h g) = eta D(g)`` for all 48 operations to 1.2e-15.  ``eta`` is
    not fixed by the Hamiltonian: it is the global one-dimensional character,
    i.e. the ``E'`` versus ``E''`` declaration, and **the scalar must be the
    declared one** -- ``+1`` for ``E'`` (the in-plane doublet, which is what
    ``p_x, p_y`` are; ``chi_E'(sigma_h) = +2``) and ``-1`` for ``E''``.  That
    comparison is the measurement this test is named for, and until 2026-09-09
    it was not made: the build served ``E'`` with ``eta = -1`` -- the
    ``sigma_h``-ODD induced representation, E1g + E2u at Gamma where in-plane p
    orbitals at ``2c`` induce E1u + E2g -- because a gauge offset the build
    then applied (meaningless for a two-dimensional site multiplet)
    hid the raw sign from the declaration, and the two label sets pinned below
    were the inverted ones.  A measured channel's build carries no offset, the
    declaration twists the matrices, and the labels are the textbook ones:
    ``E'`` gives ``GM6-`` + ``GM5+`` (E1u + E2g), ``E''`` the ``+``/``-`` swap.

    The detector alone cannot say the ``sigma_h`` half is genuine: a
    ``sigma_h``-composed operation's action on in-plane k duplicates an
    in-plane partner's, so every one of the 48 verdicts reads
    ``"undetermined"``, a statement about the *question*.  The representation
    answers it, with all 48 represented and none left undetermined.

    The same pairs listed in turned bases read one valence content, and the
    honeycomb with site group ``3`` (``tqc_models.p31m_pxpy_honeycomb``) reads
    one lowest band at ``K`` and ``M`` in every listing: there the orbitals
    cannot tell the two labellings apart, and the band labels decide.  So do
    they for ``(p_x, p_y)`` at the two sites ``(0, 0, ±0.2)`` of ``P321``
    (``tqc_models.p321_pxpy_pair``), where the labelling returned is the one
    whose labels come first in the tables' order and its table says so; with
    the spin and an on-site ``L_z S_z`` no band label depends on the choice, and
    nothing is said.
    """
    model, group, assignment, validated, rep = pxpy_bare
    assert set(validated.verdicts.values()) == {"undetermined"}, (
        "the detector never resolves a sigma_h-composed operation on its own")
    assert len(rep.indices) == 48
    frame = {name: ind for ind, name in rep.characters.frame.items()}
    mz = frame["m(z)"]
    D = rep.matrices[mz]
    eta = complex(D[0, 0])
    assert abs(abs(eta) - 1.0) < 1e-12
    assert np.abs(D - eta * np.eye(4)).max() < 1e-12, "sigma_h can only be a scalar here"

    prod = group.get_product_table(get_diff=False)
    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    worst, pairs = 0.0, 0
    for ind in rep.indices:
        c = int(group.symmetries[int(prod[pos[mz], pos[ind]])].ind)
        assert c in rep.matrices
        worst = max(worst, float(np.abs(rep.matrices[c] - eta * rep.matrices[ind]).max()))
        pairs += 1
    assert pairs == 48 and worst < 1e-12

    assert rep.characters.irreps == ("E'",) * 4
    table = rep.characters.site_tables[0]
    assert table.order == 12
    assert set(table.labels) == {"A1'", "A2'", "A1''", "A2''", "E'", "E''"}
    i_ep, i_epp = table.labels.index("E'"), table.labels.index("E''")
    assert complex(table.character_on(i_ep, mz)).real == pytest.approx(2.0, abs=1e-9)
    assert complex(table.character_on(i_epp, mz)).real == pytest.approx(-2.0, abs=1e-9)
    # the scalar IS the declaration: E' is sigma_h-even, so eta = chi_E'(sigma_h) / 2
    assert eta == pytest.approx(complex(table.character_on(i_ep, mz)) / 2, abs=1e-9), \
        "the served sigma_h must carry the declared parity"
    # and the record says what fixed it: the declared pair, and the other site's
    # pair read relative to it by the hoppings
    assert [o.determined_by for o in rep.characters.orbitals] == \
        ["declaration"] * 2 + ["hamiltonian"] * 2

    other = tqs._build_rep(model, group, assignment, validated=validated,
                           include_vacuum_ops=True, n_battery=4,
                           characters={(0, 1): "E''"})
    assert other, str(other)
    assert other.characters.irreps == ("E''",) * 4
    eta_other = complex(np.asarray(other.matrices[mz])[0, 0])
    assert eta_other == pytest.approx(complex(table.character_on(i_epp, mz)) / 2, abs=1e-9)
    assert eta_other == pytest.approx(-eta, abs=1e-9), "E'' is the other parity"

    hsp = tqg.high_symmetry_points(group)
    # the valence pair.  sigma_h-even (E'), with the sign a (p_x, p_y) pair takes
    # between the sublattices: E2g at Gamma, the Gamma5 the paper's own P6mm
    # gives the same two bands (pxpy_183 below), of the E1u + E2g the in-plane p
    # pairs carry together.  E'' takes the sign of the lowest-l functions
    # carrying it, z (p_x, p_y) = (d_zx, d_yz), on both sublattices
    even = {"GM": {"GM5+": 1}, "K": {"K5": 1}, "M": {"M4-": 1, "M3-": 1}}
    odd = {"GM": {"GM5-": 1}, "K": {"K6": 1}, "M": {"M3+": 1, "M4+": 1}}
    for name in even:
        states = _solve_at(model, hsp[name])
        a = tqw.corep_content(rep, hsp[name], states=states, bands=2)
        b = tqw.corep_content(other, hsp[name], states=states, bands=2)
        assert a and b
        assert a.counts == even[name] and b.counts == odd[name], name
        # the valence pair is ONE multiplet at GM and at K, so ``(0,)`` is the
        # only partition either build could report there, and at M it is the two
        # bands' own classes ``(0, 1)``: nothing here has a partition to move
        assert a.partition == b.partition == {"GM": (0,), "K": (0,),
                                              "M": (0, 1)}[name], name

    # Nothing declared, and every pair written in a basis turned inside itself by
    # one angle, the same at both sites -- an orientation of the Wannier pairs.
    # The lowest-l rule reads characters, which no such turn moves, and between
    # the two sublattices the functions the pairs read as, each paired with its
    # image, which a turn common to both leaves alone:
    # the valence pair is the paper's set at every angle.  The site irreps leave
    # four labellings open, and the pair's labels -- two of the four bands --
    # rest on the one returned, which its content says; all four bands do not.
    for turn in (0.37, 1.3):
        turned = tqg.realized_group(tqm.pxpy_honeycomb(turn=turn), typat=[0, 0, 0, 0],
                                    environment="free-standing", cross_check=False)
        assert turned, str(turned)
        assert turned.rep.characters.irreps == ("E'",) * 4
        valence = tqw.hsp_table(turned.rep, bands=2)
        assert valence and valence.content == even, turn
        assert [key for choice in turned.rep.characters._labellings["choices"]
                for key, _labels in choice["others"]] == ["orbitals", "orbitals", "bands"]
        assert "rest on a one-dimensional character the hoppings leave open" \
            in valence.summary()
        assert not tqw.hsp_table(turned.rep, bands=4).note

    # The honeycomb with site group 3 -- the mirrors through the hexagon centre
    # exchanging the sublattices, P31m1' on a substrate and P-62m1' free-standing
    # -- leaves two labellings under which the pairs read as the same
    # functions, and the band labels decide: the lowest band reads one set of
    # labels at K and M however each sublattice's pair is listed.
    for environment, lowest in (("substrate", {"K": {"K1": 1}, "M": {"M2": 1}}),
                                ("free-standing", {"K": {"K1": 1}, "M": {"M4": 1}})):
        for turn_a, turn_b in ((0.0, 0.0), (np.pi / 2, np.pi / 2), (0.0, np.pi),
                               (0.3, -0.5)):
            site3 = tqg.realized_group(tqm.p31m_pxpy_honeycomb(turn_a, turn_b),
                                       typat=[0, 0, 0, 0], environment=environment,
                                       cross_check=False)
            assert site3, str(site3)
            assert [key for choice in site3.rep.characters._labellings["choices"]
                    for key, _labels in choice["others"]] == ["bands"]
            band = tqw.hsp_table(site3.rep, bands=1)
            assert {p: band.content[p] for p in lowest} == lowest, \
                (environment, turn_a, turn_b, band.content)

    # ... and in three dimensions: (p_x, p_y) at the two sites (0, 0, ±0.2) of
    # P321, each pair 1E2E of its site group 3, the twofolds exchanging the
    # sites carrying one pair onto the other with either sign -- two labellings
    # that label the lower pair of bands M1 twice or M2 twice.  The one returned
    # is the one whose labels come first in the tables' order, so every listing
    # -- as built, both pairs turned alike by 90 degrees (which is the other
    # labelling's frame exactly), one pair negated -- reads the lower pair alike,
    # and its table says which step returned it.  With the spin and an on-site
    # L_z S_z no band label depends on the choice, which the group's own order
    # then makes, and nothing is said.
    listings = ((0.0, 0.0), (np.pi / 2, np.pi / 2), (0.0, np.pi))
    for soc, bands, decided, turns in ((None, 2, ["bands"], listings),
                                       (0.35, 4, [], listings[:2])):
        read = set()
        for turn_a, turn_b in turns:
            got = tqg.realized_group(tqm.p321_pxpy_pair(turn_a, turn_b, soc=soc),
                                     typat=[0] * 4)
            assert got, str(got)
            assert got.group.number_str == "150.26"
            table = got.rep.characters
            assert [key for choice in table._labellings["choices"]
                    for key, _labels in choice["others"]] == decided, \
                "the orbitals read alike under both: the band labels decide, or none rests on it"
            lower = tqw.hsp_table(got.rep, bands=bands)
            assert lower and lower.complete
            read.add(tuple(sorted((p, tuple(sorted(c.items())))
                                  for p, c in lower.content.items())))
            if soc is None:
                assert table.irreps == ("1E2E",) * 4
                assert lower.content["M"] == {"M1": 2}
                assert lower.note.endswith(
                    "the lowest-angular-momentum rule chose the one whose band labels "
                    "come first in the tables' order")
            else:
                assert not lower.note, "no band label depends on the choice"
        assert len(read) == 1, (soc, read)


@pytest.mark.slow
def test_pxpy_declares_its_two_dimensional_multiplet_jointly(pxpy_bare):
    """Declarations on a genuine 2D multiplet, three spellings.

    The label ``E'`` on one orbital, the physics shorthand ``py`` for the pair
    jointly, and ``px`` orbital by orbital all say the same thing and give the
    same matrices: the orbit either shorthand generates at this site *is* the
    pair.  The refusals are the interesting half: a bare ``E`` is *not* a label
    of ``-6m2`` (it has ``E'`` and ``E''`` and the message lists them),
    declaring the pair ``s`` jointly is grammar, not a measurement, and ``E'``
    at one site with ``E''`` at the other is a crystal with less symmetry,
    measured against the pairs' own blocks although no multiplet here is one
    orbital -- while ``E''`` at either site alone states that both pairs carry
    it: one crystal, whichever site it is written at.
    """
    model, group, assignment, validated, rep = pxpy_bare

    def rebuild(spec):
        return tqs._build_rep(model, group, assignment, validated=validated,
                              include_vacuum_ops=True, n_battery=4,
                              characters=spec)

    # ``px`` and ``py`` are the same declaration -- the multiplet, not one of
    # its members -- so every spelling here must serve the anchor's own
    # matrices; comparing two of them with each other afterwards would only
    # restate this loop by transitivity
    for spec in ({(0, 1): "py"}, {0: "E'"}, ["px", "px", "px", "px"]):
        out = rebuild(spec)
        assert out, f"{spec}: {out}"
        assert _same_matrices(rep, out), spec
        assert out.characters.irreps == ("E'",) * 4
        if isinstance(spec, list):
            assert [o.determined_by for o in out.characters.orbitals] == \
                ["declaration"] * 4

    bare_e = rebuild({(0, 1): "E"})
    assert not bare_e and bare_e.gate == "unknown_label"
    assert set(bare_e.context["irreps"]) == {"A1'", "A2'", "A1''", "A2''", "E'", "E''"}

    # A one-dimensional label on a JOINT key is grammar, not a measurement: the
    # joint form names the content its orbitals hold together, and ``A1'`` is
    # not what two orbitals hold.  The message says which spelling means two
    # of them, and nothing about the model is consulted to say it.
    as_s = rebuild({(0, 1): "s"})
    assert not as_s and as_s.gate == "ambiguous_multiplet"
    assert "the joint form names the content its orbitals hold together" in as_s.reason
    assert "1-dimensional against the 2 orbitals named" in as_s.reason
    assert as_s.context["orbitals"] == (0, 1) and as_s.context["dim"] == 1

    # ``E'`` at one site of the orbit and ``E''`` at the other: no labelling of
    # these hoppings carries both.  In a crystal of multiplets the verdict reads
    # a declared character against one pair's own blocks, settled into a
    # representation, and the two parities differ on m(z) and the S6 pair,
    # which the declaration therefore breaks.  Handed the fixed P6/mmm1', the
    # build names what the declared content is covariant on: four readings of
    # order twelve, each holding the threefold and none of those three, E' and
    # E'' being one E on each of their site groups there, and the first of them
    # by the tie rule.
    split = rebuild({(0, 1): "E'", (2, 3): "E''"})
    assert not split and split.gate == "declaration_breaks_group", str(split)
    names = {int(i): n for i, n in split.context["frame"].items()}
    readings = [{names[i] for i in r} for r in split.context["tied"]]
    assert len(readings) == 4 and all(len(r) == 12 for r in readings)
    assert all({"C3(z)", "C3^-1(z)"} <= r and not {"m(z)", "S6(z)", "S6^-1(z)"} & r
               for r in readings)
    assert {names[i] for i in split.context["subgroup"]} == \
        readings[0] | {name + "'" for name in readings[0]}
    parity = {b["name"] for b in split.context["broken"]
              if "the declarations ((0, 1) " in b["note"]}
    assert parity == {"m(z)", "S6(z)", "S6^-1(z)"}
    assert (split.context["measured_irrep"], split.context["relative_character"]) == \
        ("E'", "A2''")
    # ``E''`` at one site alone: the hoppings carry the two pairs alike, so a
    # declaration on either states what the other is -- the reference pair's
    # no more than the other's -- and both read ``E''`` with nothing broken
    alone = rebuild({(2, 3): "E''"})
    mirror = rebuild({(0, 1): "E''"})
    assert alone and mirror, (str(alone), str(mirror))
    assert alone.characters.irreps == mirror.characters.irreps == ("E''",) * 4
    assert [o.determined_by for o in alone.characters.orbitals] == \
        ["hamiltonian"] * 2 + ["declaration"] * 2
    assert _same_matrices(alone, mirror), "one crystal, whichever site was declared"
    whole = tqg.realized_group(tqm.pxpy_honeycomb(), typat=[0, 0, 0, 0],
                               environment="free-standing", cross_check=False,
                               characters={(2, 3): "E''"})
    assert whole, str(whole)
    assert (whole.group.number_str, len(whole.validated.indices)) == ("191.234", 48)
    assert whole.descent is None and whole.rep.characters.irreps == ("E''",) * 4

    # ... and `realized_group`, which lowers the cell where the two sites were
    # declared different content and then descends: the threefold with the
    # in-plane twofolds, or with the vertical mirrors, each reading keeping one
    # E per site, the tie rule keeping the first
    down = tqg.realized_group(tqm.pxpy_honeycomb(), typat=[0, 0, 0, 0],
                              environment="free-standing", cross_check=False,
                              characters={(0, 1): "E'", (2, 3): "E''"})
    assert down, str(down)
    d = down.descent
    assert d["from"]["number_str"] == "187.210" and down.group.size == 12
    assert d["steps"][0]["refuted_by"] == {"declaration": 12}
    note = d["steps"][0]["declaration"]["note"]
    assert "{E, C3(z), C3^-1(z), C2([2-10]), C2([1-20]), C2(x)}" in note
    assert "{E, C3(z), C3^-1(z), m([010]), m([100]), m(y)}" in note
    assert {"m(z)", "S6(z)", "S6^-1(z)", "m(z)'", "S6(z)'", "S6^-1(z)'"} <= \
        {b["name"] for b in d["dropped"]}
    assert [o.determined_by for o in down.rep.characters.orbitals] == ["declaration"] * 4
    assert down.rep.characters.irreps == ("E",) * 4


@pytest.mark.slow
def test_pxpy_each_branch_of_the_disconnected_ebr_is_a_difference_not_a_sum(pxpy_183):
    """The disconnected-EBR verdict, and the paper's own arithmetic for it.

    The four bands together are the paper's ``E^G(4)@2b``, exactly and
    uniquely.  Neither half of it is a nonnegative-integer sum of EBRs -- both
    are ``"none"`` with a CP-SAT infeasibility certificate.  The signed probe
    then finds what the supplement says it should: the valence content is
    exactly ``[A1^G(3)@3c] - [A1^G(1)@1a]`` (and its ``A2`` partner), a
    *difference* of EBRs.  That is why the supplement's own sentence is "a
    classification scheme that only looks at the irreps at high-symmetry
    points will classify our valence bands as trivial" while the Wilson loop
    winds: the obstruction lies beyond the symmetry indicators, which is
    exactly what this layer's verdict wording reserves room for.
    """
    _, group, _, _, rep = pxpy_183
    four = tqe.decompose(tqw.hsp_table(rep, bands=4), group, rep=rep, bands=4)
    assert four.decomposable and four.unique
    assert four.solution_strings() == ("1 x E↑G(4)@2b(3m1',3m)",)
    assert four.completeness["claim"] == "exact"

    rows = tqe.ebr_rows(group)
    points = set(tqg.high_symmetry_points(group))

    def in_plane(key):
        return {lab: v for lab, v in rows[key].items()
                if v and tqe.kpoint_of_label(lab) in points}

    a1_3c = next(k for k in rows if k.startswith("A1↑G(3)@3c"))
    a1_1a = next(k for k in rows if k.startswith("A1↑G(1)@1a"))
    valence = tqw.hsp_table(rep, bands=2)
    assert valence.counts == {"GM5": 1, "K3": 1, "M3": 1, "M4": 1}
    diff = {lab: in_plane(a1_3c).get(lab, 0) - in_plane(a1_1a).get(lab, 0)
            for lab in set(in_plane(a1_3c)) | set(in_plane(a1_1a))}
    assert {lab: v for lab, v in diff.items() if v} == valence.counts, (
        "the supplement's own subtraction, row by row")

    for bands, counts in ((2, {"GM5": 1, "K3": 1, "M3": 1, "M4": 1}),
                          ((2, 3), {"GM6": 1, "K1": 1, "K2": 1, "M1": 1, "M2": 1})):
        table = tqw.hsp_table(rep, bands=bands)
        assert table.counts == counts
        d = tqe.decompose(table, group, rep=rep, bands=bands)
        assert d.verdict == "none" and d.solutions == ()
        assert d.certificate == "proof"
        assert d.signed_combinations and d.signed_combinations["n_solutions"] > 0

    d = tqe.decompose(valence, group, rep=rep, bands=2)
    shortest = [terms for terms in d.signed_combinations["terms"] if len(terms) == 2]
    named = {tuple(sorted((m, name, wy) for m, name, wy in terms)) for terms in shortest}
    assert named == {
        ((-1, "A1↑G(1)", "1a(6mm1',6mm)"), (1, "A1↑G(3)", "3c(mm21',mm2)")),
        ((-1, "A2↑G(1)", "1a(6mm1',6mm)"), (1, "A2↑G(3)", "3c(mm21',mm2)")),
    }


@pytest.mark.slow
def test_pxpy_the_191_presentation_refines_the_labels_and_decomposes_in_the_layer(
        pxpy_bare):
    """Keeping ``sigma_h`` buys sharper labels, and the decomposition reads the layer's own.

    In ``P6/mmm`` the corep labels carry the parity the ``P6mm`` ones cannot
    (``GM5+``, the E2g of the lower in-plane p pair, where ``P6mm`` says
    ``GM5``), which is more information, and the group's EBR basis now has
    ``A``/``H``/``L`` rows that are **not** copies of in-plane ones over every
    band representation, so the in-plane restriction of the whole basis is
    not exact.  The layer's own band representations are those of the
    positions within it -- ``1b``, ``2d`` and ``3g`` lie off the layer -- and
    on them every such row repeats an in-plane one: the four
    bands decompose, exactly, as the orbitals' own ``E'`` at ``2c``.
    """
    _model, group, _, _, rep = pxpy_bare
    restriction = tqe.in_plane_restriction(group)
    assert not restriction and restriction.gate == "restriction_not_exact"
    out = tqe.decompose(tqw.hsp_table(rep, bands=4), group, rep=rep)
    assert out, str(out)
    assert out.solution_strings() == ("1 x E'↑G(4)@2c(-6m21',-6m2)",)
    assert out.completeness["claim"] == "exact"
    assert sorted(out.restriction.outside_layer) == ["1b", "2d", "3g"]


@pytest.mark.slow
def test_pxpy_declared_content_is_the_hand_derived_induced_character(pxpy_bare):
    r"""The declared ``E'`` against the band representation it must induce.

    The gap the 2026-09-09 sign inversion lived in: nothing in this file
    compared a *declared* site irrep with its own induced character, and that
    comparison is four lines with the oracle the screw fixtures already own
    (``_induced_character`` and ``_decompose_against``, further down).

    For one Wyckoff orbit whose sites share a stabilizer -- measured below --
    the induced character factors,

    .. math:: \chi^{\mathrm{Ind}}(g) = \chi_\sigma(R_g)
              \sum_{j:\, g q_j = q_j} e^{-2\pi i k \cdot m_j},

    because the site-group element that ``g`` *is* at each site it fixes has
    the same rotation part ``R_g``, and ``chi_sigma`` depends on nothing else.
    The sum is exactly what ``_induced_character`` computes (its site irrep is
    the trivial one), and the prefactor is read from the site-symmetry table at
    the **declared** label ``E'`` -- so the oracle contains no matrix the build
    produced, only the group's site permutations and the served character
    table.

    At ``GM`` every phase is 1 and the answer is the textbook graphene one: the
    in-plane ``p`` pair at ``2c`` of ``P6/mmm`` induces E1u + E2g, which is
    ``GM6-`` + ``GM5+`` in this table's spelling.  ``K`` is where the phases
    earn their keep -- ``chi = 4`` on ``E`` and ``m(z)``, ``+1`` on the
    three-folds and the ``S6``, zero elsewhere -- and at ``M`` all but ``E``
    and ``m(z)`` cancel between the two sites.

    The negative half is the other parity: ``Ind(E'')`` shares not one label
    with the served content at any of the three points, so a build that served
    the declaration's matrices untwisted -- which is what used to happen --
    cannot pass this.
    """
    model, group, assignment, _, rep = pxpy_bare
    hsp = tqg.high_symmetry_points(group)
    sites = np.asarray(assignment.site_positions, float)
    assert sites.shape == (2, 3)
    assert rep.characters.irreps == ("E'",) * 4, "the build serves the declaration"

    # `_induced_character` keys its answer by position in `ops`, counting from
    # 1, which for this group is the unitary half's own index order
    unitary = [s for s in group.symmetries if not s.time_reversal]
    assert [int(s.ind) for s in unitary] == list(range(1, 25))
    ops = [(s.rotation, s.translation) for s in unitary]

    # the two sites of 2c share one stabilizer, so every operation that fixes a
    # site fixes both and the site character comes out of the sum as a factor
    stab = set(assignment.site_group[0])
    assert stab == set(assignment.site_group[1])

    table = rep.characters.site_tables[0]

    def induced(label, name):
        i = table.labels.index(label)
        # the model is 2D: the non-periodic direction carries no k, and the
        # honeycomb sites sit at z = 0, so m_j has no z component either
        phases = _induced_character(sites, ops, np.append(hsp[name], 0.0))
        tab = tqg.corep_table(group, hsp[name])
        assert tab, str(tab)
        chi = {}
        for ind in sorted(next(iter(tab.values()))):
            if ind in stab:
                chi[ind] = complex(table.character_on(i, ind)) * phases[ind]
            else:
                assert abs(phases[ind]) < 1e-12, ind   # it fixes no site at all
                chi[ind] = 0j
        return _decompose_against(tab, chi)

    want = {"GM": {"GM6-": 1, "GM5+": 1}, "K": {"K5": 1, "K4": 1, "K1": 1},
            "M": {"M2+": 1, "M1+": 1, "M4-": 1, "M3-": 1}}
    for name, counts in want.items():
        served = tqw.corep_content(rep, hsp[name], states=_solve_at(model, hsp[name]),
                                   bands=4)
        assert served, f"{name}: {served}"
        # the oracle first: `want` below restates the same answer as a literal,
        # so it is the convenience and this line is the subject
        assert induced("E'", name) == served.counts, name
        assert served.counts == counts, name
        flipped = induced("E''", name)
        assert len(flipped) == len(counts), (name, flipped)
        assert set(flipped) & set(served.counts) == set(), (name, flipped)


# ===========================================================================
# (c)  SG 199 -- nonsymmorphic without a screw, body-centred, permuting sites
# ===========================================================================

@pytest.fixture(scope="module")
def i213_bundle():
    """``(model, group, assignment, validated, rep)`` for the primitive bcc build."""
    model = tqm.i213_flicker2018()
    group = tqg._group_of(model)
    assert group and group.sg_number == 199, str(group)
    assignment = tqg._assign_orbitals(model, group)
    assert assignment, str(assignment)
    validated = tqg._validate_group(model, group, nk=4)
    rep = tqs._build_rep(model, group, assignment, validated=validated, n_battery=5)
    assert rep, str(rep)
    return model, group, assignment, validated, rep


def test_i213_site_permutations_follow_the_measured_action(i213_bundle):
    """The second and third pathologies: sites permute *between cells*, measured.

    ``_assign_orbitals`` computes the permutation and the lattice shift from the
    positions; the *detector* computes a unitary from the Hamiltonian alone,
    with no ansatz about what the operation does to orbitals.  The two agree:
    every detector unitary has its support exactly on the permutation
    ``_assign_orbitals`` predicted, worst off-support-to-on-support ratio
    1.2e-15 over all 24 operations.  That is what makes the built rep's signed
    permutations a measurement rather than a table lookup -- and it matters
    here because 36 of the 96 (operation, site) images sit in a *different*
    unit cell.
    """
    _, group, assignment, validated, rep = i213_bundle
    perms = np.asarray(assignment.permutations, int)
    shifts = np.asarray(assignment.lattice_shift, int)
    assert perms.shape == (24, 4) and shifts.shape == (24, 4, 3)
    assert int(np.count_nonzero(np.any(shifts != 0, axis=-1))) == 36
    for p in range(24):
        assert sorted(perms[p].tolist()) == [0, 1, 2, 3], "every image is a permutation"

    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    worst = 0.0
    for ind, U in validated._intertwiners.items():
        support = np.zeros((4, 4), bool)
        support[perms[pos[int(ind)]], np.arange(4)] = True
        U = np.asarray(U)
        on = float(np.abs(U[support]).min())
        worst = max(worst, float(np.abs(U[~support]).max()) / max(on, 1e-300))
    assert len(validated._intertwiners) == 24
    assert worst < 1e-12, "the Hamiltonian's own intertwiners confirm the permutation"

    for ind in rep.indices:
        support = np.zeros((4, 4), bool)
        support[perms[pos[int(ind)]], np.arange(4)] = True
        D = rep.matrices[int(ind)]
        assert np.abs(D[~support]).max() == 0.0
        assert np.abs(np.abs(D[support]) - 1.0).max() < 1e-12

    # the site symmetry group is exactly the three operations that fix site 0
    # without a shift, i.e. the C_3 about <111> through it
    fixes = [int(group.symmetries[p].ind) for p in range(24)
             if perms[p][0] == 0 and not np.any(shifts[p][0])]
    # EQUALITY, not inclusion: ``fixes`` is defined by the strictly stronger
    # predicate (it adds the zero-shift condition), so one direction holds for
    # any model and the claim the comment makes is the other one -- that no
    # shifted operation is counted as a site symmetry of this site
    assert set(assignment.site_group[0]) == set(fixes)
    assert len([i for i in fixes if not group.symmetries[pos[i]].time_reversal]) == 3


def test_i213_rep_reproduces_the_nonsymmorphic_factor_system(i213_bundle):
    """The Bloch operators carry ``omega = spf * exp(-2 pi i (g1 g2 k).Delta)``.

    The *k-independent* matrices reproducing the product table says nothing
    about the translation part of the factor system, ``Delta = t1 + A1 t2 -
    t3``; the Bloch operators must carry it too, and on a nonsymmorphic group
    it is not vacuous.  Checked on **every** one of SG 199's 216 pairs with a
    nonzero translation difference.
    """
    _, group, _, _, rep = i213_bundle
    prod, tdiff, spf = group.get_product_table(get_diff=True)
    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    per = list(group.periodic_dirs)
    k = np.array([[0.13, -0.27, 0.41]])
    worst, checked = 0.0, 0
    for a in rep.indices:
        for b in rep.indices:
            pa, pb = pos[a], pos[b]
            if not np.any(tdiff[pa, pb] != 0):
                continue
            c = int(group.symmetries[int(prod[pa, pb])].ind)
            kb = rep.transform_k(b, k)
            Ua, Ub, Uc = (rep.bloch_operator(a, kb), rep.bloch_operator(b, k),
                          rep.bloch_operator(c, k))
            lhs = Ua @ (np.conj(Ub) if rep._operation(a).time_reversal else Ub)
            delta = (np.asarray(rep._operation(a).translation)
                     + np.asarray(rep._operation(a).rotation, float)
                     @ np.asarray(rep._operation(b).translation)
                     - np.asarray(rep._operation(c).translation))
            omega = int(spf[pa, pb]) * np.exp(-2j * np.pi
                                              * (rep.transform_k(c, k) @ delta[per]))
            worst = max(worst, float(np.abs(lhs - omega[:, None, None] * Uc).max()))
            checked += 1
    assert checked == 216
    assert worst < 1e-12


@pytest.mark.slow
def test_i213_operation_naming_in_the_T_family_is_sane(i213_bundle):
    """``_sitechars`` naming on a tetrahedral group with a body-centred basis.

    The names are read off the *measured* Cartesian matrices and the axes are
    reported as **lattice** directions, which for a primitive bcc basis is a
    trap worth documenting rather than fixing: ``C3([100])`` is the three-fold
    about ``R_1 = (a/2)(-1,1,1)``, a body diagonal of the conventional cube,
    not about ``x``.  Asserted here by rotating the named lattice direction and
    checking it is fixed -- which is the only thing the name is claiming.
    """
    _, group, _, _, rep = i213_bundle
    frame = rep.characters.frame
    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    lattice = np.asarray(group.lattice, float)
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    names = {frame[i] for i in frame if not group.symmetries[pos[i]].time_reversal}
    assert names == {"E", "C2(x)", "C2(y)", "C2(z)",
                     "C3([111])", "C3^-1([111])", "C3([100])", "C3^-1([100])",
                     "C3([010])", "C3^-1([010])", "C3([001])", "C3^-1([001])"}
    assert all(n.endswith("'") for i, n in frame.items()
               if group.symmetries[pos[i]].time_reversal)
    assert len([n for n in frame.values() if n.endswith("'")]) == 12, (
        "half the grey group is antiunitary and every one of them must be "
        "named -- without this count the line above is vacuous on an all-"
        "unitary frame")

    def _uvw(spec):               # "[1-11]" -> [1, -1, 1], one digit per index
        return [int(t) for t in re.findall(r"-?\d", spec)]

    body_diagonals = 0
    for ind, name in frame.items():
        if name.endswith("'"):
            continue
        head, _, tail = name.partition("(")
        if not tail:
            continue
        spec = tail.rstrip(")")
        R = np.asarray(group.symmetries[pos[ind]].rotation_cart, float)
        if spec in axes:
            v = np.asarray(axes[spec], float)
        else:
            v = np.asarray(_uvw(spec), float) @ lattice
        np.testing.assert_allclose(R @ v, v, atol=1e-9, err_msg=name)
        if head.startswith("C3"):
            cart = v / np.abs(v).max()
            assert sorted(np.abs(np.round(cart, 9))) == [1.0, 1.0, 1.0], (
                f"{name}: a three-fold axis of T must be a cube body diagonal")
            body_diagonals += 1
    assert body_diagonals == 8


@pytest.mark.slow
def test_i213_declares_the_trivial_site_irrep_the_appendix_uses(i213_bundle):
    """App. F.1: "all symmetries in the stabilizer group ... are represented trivially".

    The site group is the order-three ``.3.``, whose irreps the tables spell
    ``A1`` and the time-reversal-glued ``1E2E``; ``s`` resolves to ``A1`` and
    every one of the
    three site-symmetry characters measures ``+1`` on the nose.  The negative
    controls are what the pathology buys: ``1E2E`` on a one-orbital site is a
    *grammar* error (``ambiguous_multiplet``, not a mismatch), a ``4mm`` label
    is not an irrep of ``3`` at all, and the ``p`` shorthands span
    ``1 x 1E2E + 1 x A1`` -- three dimensions, the whole p shell -- because at a
    three-fold site whose axis is a body diagonal no Cartesian ``p`` function
    is an eigenfunction.
    """
    model, group, assignment, validated, rep = i213_bundle

    def rebuild(spec):
        return tqs._build_rep(model, group, assignment, validated=validated,
                              n_battery=5, characters=spec)

    table = rep.characters.site_tables[0]
    assert table.order == 3
    assert set(table.labels) == {"A1", "1E2E"}
    assert dict(zip(table.labels, table.dims)) == {"A1": 1, "1E2E": 2}
    assert table.orthogonality < 1e-10
    assert rep.characters.irreps == ("A1",) * 4
    axes = set()
    for row in rep.characters.orbitals:
        assert row.irrep == "A1" and row.dim == 1
        assert len(row.site_group) == 3           # the represented unitary '.3.'
        names = set(row.generator_character)
        assert "E" in names and len(names) == 3
        spun = {n.partition("(")[2].rstrip(")") for n in names if n.startswith("C3")}
        assert len(spun) == 1, f"one axis per site, got {spun}"
        axes |= spun
        for value in row.character.values():
            assert abs(complex(value) - 1.0) < 1e-9
    assert axes == {"[111]", "[100]", "[010]", "[001]"}, (
        "the four sites of the orbit sit on the four different <111> axes -- in "
        "the primitive bcc basis those read [111], [100], [010], [001]")

    for spec in ({0: "s"}, ["s", "s", "s", "s"], {0: "A1"}, {3: "s"}):
        out = rebuild(spec)
        assert out, f"{spec}: {out}"
        assert _same_matrices(rep, out)
        assert out.characters.irreps == ("A1",) * 4
        assert all(o.determined_by != "lowest-l" for o in out.characters.orbitals), \
            "a declaration on this site group leaves no row to the lowest-l rule"

    doublet = rebuild({0: "1E2E"})
    assert not doublet and doublet.gate == "ambiguous_multiplet"
    assert doublet.context["dim"] == 2 and doublet.context["orbitals"] == (0,)

    wrong = rebuild({0: "B1"})
    assert not wrong and wrong.gate == "unknown_label"
    assert set(wrong.context["irreps"]) == {"A1", "1E2E"}

    shell = rebuild({0: "pz"})
    assert not shell and shell.gate == "unknown_label"
    assert shell.context["decomposition"] == "1x1E2E + 1xA1"

    # ... and the model's own dz2 span at this site IS the glued doublet
    _cons, label, dim, _note = tqc_sitechars.resolve_spec(group, table, "dz2",
                                                          "orbital 0")
    assert (label, dim) == ("1E2E", 2)


@pytest.mark.slow
def test_i213_P_and_PA_identify_and_keep_their_chirality(i213_bundle):
    """The second measured instance of the ``Fd-3m`` W-point class -- resolved.

    At ``P = (1/4,1/4,1/4)`` the little group is all twelve unitary operations
    and its factor system is the nontrivial class of the Schur multiplier of
    ``A_4``, so ``irreptables`` lists three **two**-dimensional projective
    irreps (``P1``, ``P2``, ``P3``) and the spectrum is two doublets.  This
    point used to refuse with ``integer_multiplicity`` and a defect of exactly
    0.5: the measured characters were the tabulated ``P1`` and ``P3`` *in
    absolute value* but differed from them by ``-1`` on precisely the
    operations carrying a half fractional translation, so the inner product
    split the difference.  That factor of ``-1`` was BCS's convention for the
    lattice translations (``e^{+ik·t}``, the historical artifact) meeting
    pythtb's physical ``e^{-ik·t}``; conjugating the factor system at the
    boundary removes it, and the transported characters are the measured ones
    -- defect at ``irreptables``' printed precision.

    ``P`` and ``PA = -P`` are **distinct** points -- SG 199 is chiral, so no
    operation of the group maps one to the other -- and the fix must not
    exchange them: ``P`` carries ``P1 + P3`` and ``PA`` carries ``PA1 + PA3``.
    Flipping the sign of ``k`` to repair the multiplicities would have passed
    every count and swapped these two label sets.  Stability is checked the way
    the ``Fd-3m`` case was diagnosed: the identification survives a rigid
    half-lattice shift of the cell origin.
    """
    model, group, _, _, rep = i213_bundle
    hsp = tqg.high_symmetry_points(group)
    want = {"P": {"P1": 1, "P3": 1}, "PA": {"PA1": 1, "PA3": 1}}

    k = hsp["P"]
    tab = tqg.corep_table(group, k)
    assert set(tab) == {"P1", "P2", "P3"}
    inds = sorted(next(iter(tab.values())))
    A = np.array([[tab[lab][i] for i in inds] for lab in tab], dtype=complex)
    # three orthonormal two-dimensional projective irreps.  Both 1e-05-scale
    # tolerances below are ``irreptables``' own printed precision in its data
    # file and not roundoff -- measured 1.209e-05 here against 1e-4, and
    # 1.047e-05 in the character comparison further down against 2e-5 -- which
    # is why the moduli, where the printed digits cancel, are checked at
    # 4.4e-16 instead
    np.testing.assert_allclose(A[:, 0].real, [2.0, 2.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(A @ A.conj().T / 12.0, np.eye(3), atol=1e-4)

    # the transported characters ARE the measured ones, not their absolute values
    ch = tqw.characters(rep, k, states=_solve_at(model, k), bands=4)
    assert ch.multiplets == ((0, 1), (2, 3)) and ch.dims == (2, 2)
    meas = np.asarray(ch.characters)
    for row, label in ((0, "P1"), (1, "P3")):
        ref = np.array([tab[label][i] for i in ch.columns], dtype=complex)
        np.testing.assert_allclose(meas[row], ref, atol=2e-5, err_msg=label)

    from pythtb import Lattice, TBModel

    # the same crystal with its origin moved by half a lattice vector
    q = tqm.i213_sites(0.12) + np.asarray((0.5, 0.0, 0.0), float)
    alt = TBModel(Lattice(tqm.I213_LATTICE.tolist(), q.tolist(),
                          periodic_dirs=[0, 1, 2]), spinful=False)
    for i, j, R in tqm.I213_BONDS:
        alt.set_hop(1.0, i, j, list(R), mode="add")
    g2 = tqg._group_of(alt)
    assert g2 and g2.number_str == "199.13"
    a2 = tqg._assign_orbitals(alt, g2)
    v2 = tqg._validate_group(alt, g2, nk=4)
    r2 = tqs._build_rep(alt, g2, a2, validated=v2, n_battery=5)
    assert r2, str(r2)
    p2 = tqg.high_symmetry_points(g2)
    assert tqw.corep_content(r2, p2["GM"], states=_solve_at(alt, p2["GM"]), bands=4).counts \
        == {"GM1": 1, "GM4": 1}
    for name, counts in want.items():
        got = tqw.corep_content(r2, p2[name], states=_solve_at(alt, p2[name]), bands=4)
        assert got, f"{name}: {got}"
        assert got.counts == counts, name


# ===========================================================================
# (d)  the screw-group atomic controls: the completion of the convention
#      transport, measured from Hamiltonians (2026-08-21)
# ===========================================================================
#
# `groups._transported_table` used to keep the tabulated point-group content
# and re-attach our phase everywhere.  At a k-point whose tabulated factor
# system is complex -- a 3_1/4_1/6_1-type screw at the zone boundary -- that is
# not a representation of the physical factor system at all, and identification
# there produced the 0.5/0.5 multiplicities that used to be blamed on the
# tables.  The completion serves the WHOLE-CHARACTER CONJUGATE at those points,
# with each label keeping its BCS name, and these two fixtures are what makes
# that testable from a Hamiltonian: deep atomic limits whose content at every
# k-point is closed-form.

@pytest.fixture(scope="module")
def p4122_bundle():
    """``(model, group, assignment, validated, rep)`` for the #91 atomic control."""
    return _screw_bundle(tqm.p4122_atomic(), 91)


@pytest.fixture(scope="module")
def p3121_bundle():
    """``(model, group, assignment, validated, rep)`` for the #152 (quartz) control."""
    return _screw_bundle(tqm.p3121_atomic(), 152)


@pytest.fixture(scope="module")
def p3121_unitary_bundle():
    """The same crystal on the **unitary** tables (``include_TR=False``).

    The hand-derived oracle below indexes operations the way ``irreptables``
    does, which is the unitary group's own order; the grey table interleaves
    the antiunitary half.
    """
    return _screw_bundle(tqm.p3121_atomic(), 152, include_TR=False)


def _screw_bundle(model, sg_number, include_TR=True):
    group = tqg._group_of(model, include_TR=include_TR)
    assert group and group.sg_number == sg_number, str(group)
    assignment = tqg._assign_orbitals(model, group)
    assert assignment, str(assignment)
    validated = tqg._validate_group(model, group, nk=4)
    rep = tqs._build_rep(model, group, assignment, validated=validated, n_battery=5)
    assert rep, str(rep)
    return model, group, assignment, validated, rep


def _induced_character(sites, ops, k, s=-1):
    r"""``χ^{(s)}({R|t}) = Σ_{j: g q_j = q_j} e^{s·2πi k·m_j}`` for a trivial site irrep.

    The textbook induced character of a band representation, with
    ``m_j = R τ_j + t − τ_j`` the lattice vector the operation carries the site
    by.  ``s = −1`` is the physical convention (:mod:`tqc.sewing`'s), ``s = +1``
    BCS's.  ``χ_ρ ≡ 1`` because the site irrep is the trivial one of the
    order-2 site group, so the whole thing is a sum of unimodular phases and
    the answer can be written down by hand.
    """
    chi = {}
    for ind, (R, t) in enumerate(ops, start=1):
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float)
        total = 0j
        for tau in sites:
            m = R @ tau + t - tau
            if np.abs(m - np.rint(m)).max() < 1e-9:
                total += np.exp(s * 2j * np.pi * np.asarray(k, dtype=float)
                                @ np.rint(m))
        chi[ind] = total
    return chi


def _decompose_against(table, chi):
    """Multiplicities of ``chi`` in the served corep table, as exact integers."""
    labels = sorted(table)
    inds = sorted(next(iter(table.values())))
    assert sorted(chi) == inds, (sorted(chi), inds)
    A = np.array([[complex(table[lab][i]) for i in inds] for lab in labels]).T
    y = np.array([complex(chi[i]) for i in inds])
    n, *_ = np.linalg.lstsq(A, y, rcond=None)
    assert np.abs(A @ n - y).max() < 1e-3, "the served table does not span the content"
    assert np.abs(n - np.round(n.real)).max() < 1e-3, ("non-integer", labels, n)
    return {lab: int(round(v.real)) for lab, v in zip(labels, n) if round(v.real)}


@pytest.mark.slow
def test_p3121_screw_point_content_is_the_hand_derived_induced_character(
        p3121_unitary_bundle):
    r"""Quartz's group at ``A``: closed-form arithmetic against the served table.

    The occupied bands are ``s`` at the 3a orbit ``(x,0,1/3)``, whose three
    sites are ``τ₀ = (x,0,1/3)``, ``τ₁ = (0,x,2/3)`` and ``τ₂ = (−x,−x,0)``.  At
    ``A = (0,0,1/2)`` the little group is all six operations, and the induced
    character is a two-line hand calculation:

    * ``E`` fixes all three sites with ``m = 0``           -> ``χ = 3``;
    * the two 3-fold screws fix none                       -> ``χ = 0``;
    * ``{2_x,x,0 | 0}`` (index 4) fixes only ``τ₂``, ``m = 0``   -> ``χ = 1``;
    * ``{2_x,0,1/3 | (0,0,2/3)}`` (index 5) fixes only ``τ₀``, ``m = 0``
      -> ``χ = 1``;
    * ``{2_0,y,1/6 | (0,0,1/3)}`` (index 6) fixes only ``τ₁`` and carries it by
      ``m = (0,0,−1)``, so it contributes ``e^{-2πi k·m} = e^{iπ} = −1``.

    So ``χ = (3, 0, 0, 1, 1, −1)``, and against the six-element little group the
    multiplicities come out ``n_A1 = (3 + 1 + 1 + 1)/6 = 1``,
    ``n_A2 = (3 − 1 − 1 − 1)/6 = 0``, ``n_A3 = (3·2)/6 = 1``: **A1 + A3**, which
    is exactly what the Hamiltonian measures and exactly the ``A^G(3)@3a`` row.

    ``A`` is a TRIM, so this induced character is the *same* in both
    conventions -- what the completion fixes is the table it is decomposed
    against, and with the tabulated one it splits as 1/2 A1 + 1/2 A2 + A3.
    """
    model, group, _, _, rep = p3121_unitary_bundle
    hsp = tqg.high_symmetry_points(group)
    sites = tqm.p3121_sites()[0]
    np.testing.assert_allclose(hsp["A"], [0.0, 0.0, 0.5], atol=1e-12)

    chi = _induced_character(sites, tqm.P3121_OPS, hsp["A"])
    np.testing.assert_allclose([chi[i] for i in range(1, 7)],
                               [3, 0, 0, 1, 1, -1], atol=1e-12)

    table = tqg.corep_table(group, hsp["A"])
    assert table, str(table)
    assert _decompose_against(table, chi) == {"A1": 1, "A3": 1}

    measured = tqw.corep_content(rep, hsp["A"], states=_solve_at(model, hsp["A"]),
                                 bands=list(range(3)))
    assert measured, str(measured)
    assert measured.counts == {"A1": 1, "A3": 1}

    # and the *other* orbit, whose row is the discriminating one
    chi_b = _induced_character(tqm.p3121_sites()[1], tqm.P3121_OPS, hsp["A"])
    assert _decompose_against(table, chi_b) == {"A2": 1, "A3": 1}
    other = tqw.corep_content(rep, hsp["A"], states=_solve_at(model, hsp["A"]),
                              bands=list(range(3, 6)))
    assert other and other.counts == {"A2": 1, "A3": 1}


@pytest.mark.slow
@pytest.mark.parametrize("fixture,occ,rows,pin", [
    ("p4122_bundle", 4, ("A↑G(4)@4a", "A↑G(4)@4b"),
     ("A", {"A1": ([2, 0, 2 ** 0.5, -2 ** 0.5, 0, 0, 0, 0], 1, 1),
            "A2": ([2, 0, -2 ** 0.5, 2 ** 0.5, 0, 0, 0, 0], 1, 1)})),
    ("p3121_bundle", 3, ("A↑G(3)@3a", "A↑G(3)@3b"),
     ("A", {"A1": ([1, -1, 1, 1, 1, -1], 1, 0),
            "A2": ([1, -1, 1, -1, -1, 1], 0, 1)})),
])
def test_screw_fixture_content_is_its_own_EBR_row_at_every_point(
        request, fixture, occ, rows, pin):
    """The oracle, at **every** k-point including the previously-refused ones.

    An ``s`` orbital on a maximal Wyckoff orbit induces exactly one EBR, so the
    occupied block must reproduce that row entry for entry and the empty block
    the other orbit's -- and ``decompose`` must return ``1 x row @ its own
    site`` with the claim ``exact``.  Nothing here is typed: the target is
    ``irreptables``' own row, transported.

    What that comparison certifies is the **multiplicity vector and the Wyckoff
    position**, not the names: ``ebr_rows`` takes its labels from
    ``_ebr_transport`` and ``hsp_table`` takes its from ``corep_table``, and both
    read the one map ``_table_conjugation`` returns, so a consistent relabelling
    of the served naming moves the row and the measurement together and cancels
    out.  The names are pinned separately, by the one thing that is outside the
    package: two labels per fixture tied to ``irreptables``' own printed
    character vector at ``A``, typed below to the five decimals BCS prints
    (hence ``atol=2e-3``).  For #152 those two are also the discriminating pair
    -- the hand derivation in
    ``test_p3121_screw_point_content_is_the_hand_derived_induced_character``
    puts ``A1`` on the 3a orbit and ``A2`` on 3b, so the multiplicities below
    differ between the blocks; for #91 ``A1`` and ``A2`` both appear once in
    both rows, and there it is the character vectors alone that hold the names
    down.
    """
    model, group, _, _, rep = request.getfixturevalue(fixture)
    all_rows = tqe.ebr_rows(group)

    point, pins = pin
    served = tqg.corep_table(group, tqg.high_symmetry_points(group)[point])
    assert served, (point, str(served))
    inds = sorted(next(iter(served.values())))
    for label, (chi_lit, _m0, _m1) in pins.items():
        assert label in served, (point, label, sorted(served))
        np.testing.assert_allclose([complex(served[label][i]) for i in inds],
                                   chi_lit, atol=2e-3)

    for slot, (bands, key_prefix) in enumerate(((list(range(occ)), rows[0]),
                                                (list(range(occ, model.norb)), rows[1]))):
        table = tqw.hsp_table(rep, bands=bands)
        assert table.complete is True and table.refusals == {}, table.refusals
        key = next(k for k in all_rows if k.startswith(key_prefix))
        assert {lab: v for lab, v in all_rows[key].items() if v} == table.counts, key
        for label, spec in pins.items():
            assert table.counts.get(label, 0) == spec[1 + slot], (key, label, table.counts)

        got = tqe.decompose(table, group, rep=rep, bands=bands)
        assert got.decomposable and got.unique, str(got)
        assert got.solution_strings() == (f"1 x {key}",)
        assert got.positions == ((key_prefix.split("@")[1],),)
        assert got.completeness["claim"] == "exact"
        # all four fixtures of this section are 3D and tabulated at every point,
        # so the specific statement is that the restriction never fired; what
        # `exact`/`orphans` can say belongs to the slab tests that can move them
        assert got.restriction.applied is False, got.restriction


@pytest.mark.slow
@pytest.mark.parametrize("fixture,occ,points", [
    ("p3121_bundle", 3, ("A", "H", "HA", "L")),
])
def test_screw_points_refuse_without_the_whole_character_conjugation(
        request, monkeypatch, fixture, occ, points):
    """The completion, measured: switch it off and every screw point refuses.

    With ``_transported_table`` forced back to "keep the tabulated content,
    flip the phase" at every k-point, each of these points comes back as
    ``gate="integer_multiplicity"`` with a defect of exactly **0.5** -- the
    Fd-3m ``W`` / SG 199 ``P`` signature -- and ``decompose`` refuses with
    ``incomplete_content``.  Every *other* point of the same model is untouched,
    which is the invariance statement: the completion only reaches the k-points
    the transport record names.
    """
    model, group, _, _, rep = request.getfixturevalue(fixture)
    hsp = tqg.high_symmetry_points(group)
    assert set(points) <= set(hsp)
    bands = list(range(occ))
    before = {name: tqw.corep_content(rep, hsp[name], states=_solve_at(model, hsp[name]),
                                      bands=bands).counts for name in hsp}

    monkeypatch.setattr(tqg, "_conjugated_kpoints", lambda *a, **k: frozenset())
    for name in hsp:
        out = tqw.corep_content(rep, hsp[name], states=_solve_at(model, hsp[name]),
                                bands=bands)
        if name in points:
            assert not out and out.gate == "integer_multiplicity", (name, str(out))
            # exactly 1/2, up to irreptables' five printed decimals
            assert float(out.value) == pytest.approx(0.5, abs=1e-4), name
        else:
            assert out and out.counts == before[name], (
                f"{name} moved, and only the conjugated points may")

    table = tqw.hsp_table(rep, bands=bands)
    assert table.complete is False and set(table.refusals) == set(points)
    got = tqe.decompose(table, group, rep=rep, bands=bands)
    assert not got and got.gate == "incomplete_content", str(got)


# ===========================================================================
# (e)  the RE-GLUING points: the time-reversal gluing itself is
#      convention-sensitive, measured from Hamiltonians (2026-08-21)
# ===========================================================================
#
# At a k-point where the pairing antiunitary is not pure time reversal -- or
# where it is, but the little group's own fractional translations put `k.m`
# outside (1/2)Z -- every term of the Wigner-Herring sum
# `I_s(D) = (1/|H|) sum_a chi_D(a^2)` carries `e^{s.2pi.i k.m}`, so the
# indicator, and with it the question of WHICH unitary irrep doubles, is
# convention-sensitive.  The physical corep list is then a different partition
# of the same unitary irreps than BCS's, and at SG 80's P the two partitions do
# not even have the same dimensions.  These two fixtures are deep spinful
# atomic limits whose content at those points is closed form.

@pytest.fixture(scope="module")
def i41_bundle():
    """``(model, group, assignment, validated, rep)`` for the #80 re-gluing anchor."""
    return _screw_bundle(tqm.i41_atomic(), 80)


@pytest.fixture(scope="module")
def p62_bundle():
    """``(model, group, assignment, validated, rep)`` for the #171 companion."""
    return _screw_bundle(tqm.p62_atomic(), 171)


def _spinful_induced_character(sites, ops, su2, k, s=-1):
    r"""The induced character of a **spinful** ``s`` orbital on one Wyckoff orbit.

    The hoppings of these fixtures are spin-independent, so the band
    representation is the spinless one tensored with spin 1/2 and the character
    factorises: ``chi(g) = tr(S_R) * chi_spinless(g)`` with ``S_R`` the
    operation's own SU(2) matrix (taken from ``irreptables``, not
    reconstructed).  Since ``tr(S) = 2cos(theta/2)``, a two-fold rotation
    contributes zero and kills the whole term whatever the orbit does.
    """
    chi = _induced_character(sites, ops, k, s=s)
    return {ind: chi[ind] * complex(np.trace(np.asarray(su2[ind - 1], dtype=complex)))
            for ind in chi}


def _table_su2(number_str):
    """``[S]`` per unitary symmetry of the double-valued table, in its own order."""
    assert tqg.tables_status()["tables"]
    from irreptables.irreps import IrrepTable
    table = IrrepTable(number_str, True, magnetic=False)
    return [np.asarray(s.S, dtype=complex) for s in table.symmetries]


@pytest.mark.slow
@pytest.mark.parametrize("fixture,occ,row,letter,pin", [
    ("p62_bundle", 6, "-1E2E\u2191G(6)@3a", "3a",
     ("H", {"-H4": ([1, -1, 1], 2), "-H5H6": ([2, 1, -1], 2)})),
])
def test_reglue_atomic_content_is_its_own_EBR_row_at_every_point(
        request, fixture, occ, row, letter, pin):
    """The oracle at **every** k-point, re-glued ones included.

    A spinful ``s`` orbital on a maximal Wyckoff orbit induces exactly one grey
    elementary band representation, so the occupied block must reproduce that
    row entry for entry -- in the *transported* corep basis -- and
    ``decompose`` must return it with the claim ``exact``.

    As in the screw section above, that comparison certifies the multiplicity
    vector and the Wyckoff position and **not** the names: row and measurement
    read the same transported naming map, so a consistent relabelling cancels
    out of it.  The names are pinned by the served character vectors at ``H``,
    where the ``3a`` orbit's induced character is ``(6, 0, 0)`` on the unitary
    little group ``{E, 3+, 3-}`` -- the two threefold screws fix no site -- and
    the served table ``{-H4: (1, -1, 1), -H5H6: (2, 1, -1)}`` takes it as
    ``2 x -H4 + 2 x -H5H6``: those two names must carry those two vectors *and*
    those two multiplicities in the row.

    #171's ``3a`` and ``3b`` carry identical EBR vectors, so its decomposition
    is exact but not unique; that is a property of the group, not a weakness of
    the measurement, and it is asserted rather than papered over.
    """
    model, group, _, _, rep = request.getfixturevalue(fixture)
    all_rows = tqe.ebr_rows(group)
    table = tqw.hsp_table(rep, bands=list(range(occ)))
    assert table.complete is True and table.refusals == {}, table.refusals
    key = next(k for k in all_rows if k.startswith(row))
    assert {lab: v for lab, v in all_rows[key].items() if v} == table.counts, key

    point, pins = pin
    served = tqg.corep_table(group, tqg.high_symmetry_points(group)[point])
    assert served, (point, str(served))
    inds = sorted(next(iter(served.values())))
    for label, (chi_lit, mult) in pins.items():
        assert label in served, (point, label, sorted(served))
        np.testing.assert_allclose([complex(served[label][i]) for i in inds],
                                   chi_lit, atol=2e-3)
        assert table.counts.get(label, 0) == mult, (label, table.counts)

    got = tqe.decompose(table, group, rep=rep, bands=occ)
    assert got.decomposable, str(got)
    assert f"1 x {key}" in got.solution_strings(), got.solution_strings()
    assert (letter,) in got.positions, got.positions
    assert got.completeness["claim"] == "exact"
    # a 3D model tabulated at every point: the restriction never fires (see the
    # screw test above; `exact`/`orphans` are the slab tests' statement)
    assert got.restriction.applied is False, got.restriction
    assert got.unique is False, "#171's 3a and 3b have identical EBR vectors"


@pytest.mark.slow
def test_the_reglued_labels_and_the_EBR_transport_need_the_lift(
        monkeypatch, i41_bundle, p62_bundle, i41_soc_bundle):
    """Switch the re-gluing off: SG 80 still fails hard, SG 171 no longer needs it.

    * **SG 80** fails hard, and this half is unchanged by the 2026-08-24 naming
      directive: with the corep naming decided by plain conjugation of whole
      character vectors, ``P``'s shipped list is not conjugation-closed (its
      dimensions are ``2+1`` and the physical ones ``1+2``), the table is
      declared corrupt, and ``ebr_data`` -- hence ``decompose`` -- refuses the
      whole group with ``gate="ebr_convention"``.  That refusal is exactly the
      "data defect" the point used to be filed under, and it is why ``P`` is
      one of the 59 ``divergent`` entries: no naming rule can reach it.
    * **SG 171 no longer needs the lift.**  Before the directive it was the
      *quiet* failure -- the served rows at ``H`` were the same characters under
      BCS's names, so nothing refused while the labels contradicted the unitary
      table's spelling.  Now the character-matched naming reaches BCS's names on
      its own, the lift agrees with it, and switching the lift off changes
      nothing at all.  That is asserted rather than dropped, because it is the
      measurement that says which of the two mechanisms is load-bearing where.

    Every other k-point of both models is untouched, which is the invariance
    half: the re-gluing reaches only the points the transport record names.  That
    is a **before/after snapshot** of the measured content at every point of
    both models, taken outside the monkeypatch and compared inside it -- the
    same shape ``test_screw_points_refuse_without_the_whole_character_conjugation``
    uses, and the only shape that can catch a re-gluing that reached too far.

    In the atomic limit the lift switched off produces no refusal at all, just
    the silent ``{-P3P3: 1, -P4: 2}`` on one unresolved level.  With spin-orbit
    coupling resolving ``P``'s valence level into ``[1, 2, 1]``, two of the
    three are arithmetically impossible under BCS's gluing: its one-dimensional
    corep has ``chi(2_z) = +i`` and the measured singlets ``-i``, so each
    refuses ``integer_multiplicity`` at a defect of exactly **0.5**, and the
    doublet is *mislabelled rather than refused* -- ``(2, +2i)`` is two copies
    of BCS's ``-P4``, an accidental degeneracy of two one-dimensional coreps,
    which is precisely what an enforced pair is not.
    """
    _soc_model, soc_group, _sa, _sv, soc_rep = i41_soc_bundle
    P, states, _w, _mult = _P_valence(i41_soc_bundle)
    want = ((i41_bundle, 4, ("P",), ("-P3P3", "-P4")),
            (p62_bundle, 6, (), ()))
    for bundle, _occ, points, changed in want:
        transport = tqg.ebr_data(bundle[1])["convention"]
        assert tuple(transport["reglued_points"]) == points, transport
        # the positive counterpart of #171's negative above: SG 80's P really is
        # one of the 59 ``divergent`` entries, and these are the labels that move
        assert tuple(transport["divergent_points"]) == points, transport
        assert tuple(transport["changed_labels"]) == changed, transport

    # the snapshot: every point of both models, in the unpatched state
    before = {}
    for bundle, occ, _p, _c in want:
        b_model, b_group, _ba, _bv, b_rep = bundle
        b_hsp = tqg.high_symmetry_points(b_group)
        before[b_group.number_str] = {
            name: tqw.corep_content(b_rep, b_hsp[name],
                                    states=_solve_at(b_model, b_hsp[name]),
                                    bands=list(range(occ))).counts
            for name in b_hsp}
    assert set(before["80.30"]) == {"GM", "M", "N", "P", "X"}, sorted(before["80.30"])
    assert set(before["171.122"]) == {"A", "GM", "H", "K", "L", "M"}, \
        sorted(before["171.122"])
    # ... and every served label is named after its own k-point, at the untouched
    # points too.  A snapshot on its own cannot say that: it is taken and compared
    # in the same naming, so a *globally* wrong name at those points cancels out
    # of it, while this rule is what such a name breaks.
    for tag in before:
        for name, counts in before[tag].items():
            assert counts, (tag, name, counts)
            assert all(lab.lstrip("-").startswith(name) for lab in counts), \
                (tag, name, sorted(counts))

    def clear():
        tqg._unitary_conjugation.cache_clear()
        tqg._transport_naming.cache_clear()
        tqg._ebr_transport.cache_clear()

    p62_before = dict(tqg.corep_table(p62_bundle[1],
                                      tqg.high_symmetry_points(p62_bundle[1])["H"]))

    monkeypatch.setattr(tqg, "_lifted_conjugation", lambda table: {})
    clear()
    try:
        model, group, _, _, rep = i41_bundle
        hsp = tqg.high_symmetry_points(group)
        assert sorted(tqg.corep_table(group, hsp["P"])) == ["-P3P3", "-P4"]
        data = tqg.ebr_data(group)
        assert not data and data.gate == "ebr_convention", str(data)
        table = tqw.hsp_table(rep, bands=list(range(4)))
        assert table.complete is True, "identification itself does not refuse"
        assert table.counts["-P4"] == 2 and table.counts["-P3P3"] == 1
        got = tqe.decompose(table, group, rep=rep, bands=4)
        assert not got, str(got)
        for name in hsp:       # only P may have moved
            out = tqw.corep_content(rep, hsp[name], states=_solve_at(model, hsp[name]),
                                    bands=list(range(4)))
            assert out, (name, str(out))
            if name == "P":
                assert out.counts == {"-P3P3": 1, "-P4": 2}, out.counts
                assert out.counts != before["80.30"][name], "P has to move"
            else:
                assert out.counts == before["80.30"][name], (
                    f"{name} moved, and only the re-glued point may")

        # the levels spin-orbit coupling resolves at P, under BCS's gluing
        assert sorted(tqg.corep_table(soc_group, P)) == ["-P3P3", "-P4"]
        for block in ((0,), (3,)):
            out = tqw.corep_content(soc_rep, P, states=states, bands=list(block))
            assert not out and out.gate == "integer_multiplicity", (block, str(out))
            assert float(out.value) == pytest.approx(0.5, abs=1e-4), block
            assert tuple(out.context["labels"]) == ("-P3P3", "-P4")
            np.testing.assert_allclose(np.asarray(out.context["multiplicities"]).ravel(),
                                       [0.5, 0.0], atol=1e-4)
        doublet = tqw.corep_content(soc_rep, P, states=states, bands=[1, 2])
        assert doublet, str(doublet)
        assert doublet.counts == {"-P4": 2}, "mislabelled as two lone coreps"
        whole = tqw.corep_content(soc_rep, P, states=states, bands=[0, 1, 2, 3])
        assert not whole and whole.gate == "integer_multiplicity", str(whole)
        assert list(whole.context["rows"]) == [0, 2]
        soc_table = tqw.hsp_table(soc_rep, bands=[0, 1, 2, 3])
        assert soc_table.complete is False and set(soc_table.refusals) == {"P"}
        assert set(soc_table.content) == {"GM", "M", "N", "X"}, "only P moves"

        model, group, _, _, rep = p62_bundle
        hsp = tqg.high_symmetry_points(group)
        after = tqg.corep_table(group, hsp["H"])
        assert sorted(after) == ["-H4", "-H5H6"], sorted(after)
        assert sorted(after) == sorted(p62_before), "the lift is not load-bearing here"
        for name in sorted(after):
            for ind in after[name]:
                assert abs(after[name][ind] - p62_before[name][ind]) < 1e-9
        counts = tqw.corep_content(rep, hsp["H"], states=_solve_at(model, hsp["H"]),
                                   bands=list(range(6))).counts
        assert counts == {"-H4": 2, "-H5H6": 2}, counts
        for name in hsp:       # nothing at all may move in #171
            got = tqw.corep_content(rep, hsp[name], states=_solve_at(model, hsp[name]),
                                    bands=list(range(6)))
            assert got, (name, str(got))
            assert got.counts == before["171.122"][name], (
                f"{name} moved, and in #171 the lift is not load-bearing anywhere")
    finally:
        clear()


# ===========================================================================
# (f)  the SOC-doublet demonstration at the re-glued point (2026-08-24):
#      WHICH irrep doubles is a fact about a spectrum, not a convention
# ===========================================================================
#
# The atomic fixtures above cannot finish the argument.  Their hoppings are
# spin-independent, so ``[H(k), s_i] = 0`` for every component, every level is
# SU(2)-degenerate, and at ``P`` the four valence states are ONE level: the
# content arrives as a single row, and BOTH corep lists span it -- ``2*(-P3) +
# -P4P4`` and ``-P3P3 + 2*(-P4)`` are four units of dimension either way.
#
# ``i41_soc("generic")`` adds a group-averaged spin-orbit bond set.  It is
# still EXACTLY ``I4_11'``-symmetric (8 of 8 operations validate on the default
# 8**3 mesh) and it splits ``P``'s valence level into ``[1, 2, 1]``.
# Degeneracies are convention-independent physical facts, so the middle level
# is a *symmetry-enforced* pair -- the unitary little group ``{E, 2_z}`` has
# nothing but one-dimensional irreps, so only the antiunitary ``C4.T`` coset
# can force it -- and the two singlets are genuinely alone.  ``[1, 2, 1]`` by
# itself is still arithmetically consistent with both conventions (physical:
# ``2*(-P3) + (-P4P4)``; BCS: ``(-P3P3) + 2*(-P4)``).  What decides is WHICH
# irrep the doublet carries, and that the sewing characters measure:
# ``chi(2_z) = +2i`` on the doublet, ``-i`` on each singlet, read in
# irreptables' own SU(2) lift.  ``examples/tqc_sg80_soc_doublet.py`` prints the
# levels and their labels; what the lift decides is measured below.

@pytest.fixture(scope="module")
def i41_soc_bundle():
    """``(model, group, assignment, validated, rep)`` for ``i41_soc`` at lam=0.4."""
    return _screw_bundle(tqm.i41_soc("generic", lam=0.4), 80)


def _P_valence(bundle):
    """``(P, states, evals[:4], multiplets)`` from ONE diagonalization at ``P``."""
    model, group, _, _, _ = bundle
    P = tqg.high_symmetry_points(group)["P"]
    states = _solve_at(model, P)
    w = np.asarray(states[0], dtype=float).reshape(-1)[:4]
    return P, states, w, tqw.multiplets(w)


# ---------------------------------------------------------------------------
# (f2) the same crystal on the UNITARY tables (2026-08-25): the gauge the
#      grey build gets from time reversal, and the unitary one from the tables
# ---------------------------------------------------------------------------
#
# Building #80 with ``include_TR=False`` used to land on the rep twisted by the
# linear character ``lambda(E, 2_z, 4+, 4-) = (1, -1, +i, -i)`` of its Z4 --
# measured as the exact ratio of the two builds' matrices -- and so read P's
# levels the other way round (singlets ``-P4``, doublet ``2 x -P3``).  Nothing
# downstream could catch it: a linear-character twist preserves the entire
# factor system, so covariance, the internal factor, the product table and
# ``hsp_table``'s completeness are all blind to it.  Owner's ruling
# (2026-08-25): the tables fix that gauge, so the unitary build is anchored to
# the group's own ``spinor_rotation`` (``test_symmetry.py`` section 10 measures
# the anchor itself), and what the two builds must now agree on is *everything*.


@pytest.fixture(scope="module")
def i41_soc_unitary_bundle(i41_soc_bundle):
    """The same model and the same parameters, on ``#80``'s unitary tables."""
    return _screw_bundle(i41_soc_bundle[0], 80, include_TR=False)


def test_the_unitary_build_of_i41_soc_is_the_grey_build(i41_soc_bundle,
                                                        i41_soc_unitary_bundle):
    """Bit for bit on the four unitary operations -- not even a global phase apart.

    Both builds normalize the same four-fold screw against the same reference
    orbital; the grey one gets the reality of the phase from ``T`` (central, so
    ``lambda = conj(lambda)``) and the unitary one from the tables (an SU(2)
    lift is pseudo-real).  Two different arguments for one and the same
    condition, so the answers coincide exactly rather than up to anything:
    measured ``max |D_unitary(g) - D_grey(g)| = 0.0`` over ``{E, 2_z, 4+, 4-}``,
    where the twisted build gave the character ``(1, -1, +i, -i)``.
    """
    _model, grey_group, _a, _v, grey = i41_soc_bundle
    _model_u, u_group, _au, _vu, urep = i41_soc_unitary_bundle
    assert grey_group.number_str == "80.30" and u_group.number_str == "80"
    assert urep.antiunitary_operator[0] is None and grey.antiunitary_operator[0] is not None

    def key(op):
        return (tuple(np.asarray(op.rotation, dtype=int).ravel()),
                tuple(np.round(np.asarray(op.translation) % 1.0, 6)))

    grey_of = {key(grey._operation(i)): i for i in grey.indices
               if not grey._operation(i).time_reversal}
    assert len(grey_of) == len(urep.indices) == 4
    worst = 0.0
    for i in urep.indices:
        j = grey_of[key(urep._operation(i))]
        worst = max(worst, float(np.abs(np.asarray(urep.matrices[i])
                                        - np.asarray(grey.matrices[j])).max()))
    assert worst == 0.0, "the two gauges are the same gauge"

    # the record says which condition did it in each build.  `anchor_used is
    # False` on the unitary side would only restate the premise -- it is assigned
    # inside `if T is not None`, and `urep.antiunitary_operator[0] is None` is asserted
    # above -- so what stands in its place is the two builds' `accepted` sets:
    # two different arguments narrowing the same generator to the same pair.
    assert len(urep._generators) == len(grey._generators) == 1, (urep._generators,
                                                                 grey._generators)
    for ind in urep._generators:
        rec = urep._normalization[int(ind)]
        assert rec.table_anchored is True
        g_rec = grey._normalization[grey_of[key(urep._operation(ind))]]
        assert len(rec.accepted) == len(g_rec.accepted) == 2, (rec.accepted,
                                                              g_rec.accepted)
        np.testing.assert_allclose(np.asarray(rec.accepted, dtype=complex),
                                   np.asarray(g_rec.accepted, dtype=complex),
                                   atol=4 * _ULP)
    for ind in grey._generators:
        assert grey._normalization[int(ind)].table_anchored is False, \
            "the grey path never sees the table anchor -- it has T"
        assert grey._normalization[int(ind)].anchor_used is True


def _conjugate_labels(table):
    """``{label: label*}`` at one k-point, from the served characters alone.

    The complex-conjugate corep is the one whose whole character vector is the
    complex conjugate of this one's -- a statement about numbers, so the map is
    independent of every naming rule the package applies.  ``2e-3`` is
    ``irreptables``' five printed decimals, as everywhere else in this file.
    """
    inds = sorted(next(iter(table.values())))
    vecs = {lab: np.array([complex(table[lab][i]) for i in inds]) for lab in table}
    out = {}
    for lab, v in vecs.items():
        hit = [m for m, w in vecs.items() if np.abs(w - v.conj()).max() < 2e-3]
        assert len(hit) == 1, (lab, hit)
        out[lab] = hit[0]
    return out


def _spelled_out(glued, alphabet):
    """The unitary labels a grey corep label spells: ``-GM5GM7 -> (-GM5, -GM7)``.

    The spelling invariant written as a *rule* rather than as pinned strings: a
    grey label is either one of the unitary labels of the same k-point or the
    concatenation of two of them, the second re-wearing the first's ``-`` prefix
    (so ``-P4P4 -> (-P4, -P4)`` and ``-N2N2 -> (-N2, -N2)``).  ``alphabet`` is
    the served *unitary* table's own label set at that point.
    """
    if glued in alphabet:
        return (glued,)
    pre = "-" if glued.startswith("-") else ""
    body = glued[len(pre):]
    for j in range(1, len(body)):
        pair = (pre + body[:j], pre + body[j:])
        if pair[0] in alphabet and pair[1] in alphabet:
            return pair
    raise AssertionError(f"{glued} spells nothing in {sorted(alphabet)}")


@pytest.mark.slow
def test_the_unitary_build_names_P_and_GM_the_way_the_grey_one_does(
        i41_soc_bundle, i41_soc_unitary_bundle):
    r"""The labels, per level, against the SERVED tables of each group.

    ``P``, from one diagonalization: ``chi = (1, -i)`` on each singlet and
    ``(2, +2i)`` on the doublet, **identical** in the two builds, so the
    singlets are ``-P3`` in both spellings and the doublet is the grey
    ``-P4P4`` = ``2 x`` the unitary ``-P4``, entry for entry.  At ``GM`` the
    same content unglues the other way: the grey ``{-GM5GM7, -GM6GM8}``
    against the unitary ``{-GM5, -GM6, -GM7, -GM8}``, one each, and the glued
    names spell exactly the unitary ones.  The twisted build agreed with
    neither: its ``P`` singlets read ``-P4`` and its doublet ``2 x -P3``.

    The spelling is checked as a **rule** and not only at ``GM``: every grey
    label at every point both presentations reach is split against the served
    unitary label set and the multiplicities have to add up, which is a statement
    a string-parsing bug can fail even with the four literal contents below
    pinned.  And one level further down, the same gluing is checked on the
    **site** irreps: ``#80``'s ``4a`` carries the complex-conjugate pair
    ``-1E``/``-2E``, which time reversal glues into the single grey row
    ``-1E2E↑G(4)@4a`` and the unitary tables keep apart.  The two unitary rows
    must be each other's complex conjugate label by label -- ``Ind(rho*) =
    Ind(rho)*`` -- with the conjugation read off the served character vectors
    rather than from the package's own site-name rule, and their sum must be the
    grey row spelled out.  This is the only place in this file that reaches an
    *unglued* complex site irrep at all: every site irrep the grey fixtures
    occupy is self-conjugate (``A``/``B`` of the order-2 site groups of #91 and
    #152, the already-glued ``-1E2E`` of #80 and #171), so without the unitary
    row pair here nothing would notice if that rule regressed.
    """
    model, grey_group, _a, _v, grey = i41_soc_bundle
    _mu, u_group, _au, _vu, urep = i41_soc_unitary_bundle
    P, states, _w, mult = _P_valence(i41_soc_bundle)
    assert mult == ((0,), (1, 2), (3,)), mult

    want = ({"-P3": 1}, {"-P4": 2}, {"-P3": 1})     # the unitary spelling of (f)'s answer
    for block, counts in zip(mult, want):
        u_out = tqw.corep_content(urep, P, states=states, bands=list(block))
        g_out = tqw.corep_content(grey, P, states=states, bands=list(block))
        assert u_out and g_out, (block, str(u_out), str(g_out))
        assert u_out.counts == counts, (block, u_out.counts)
        np.testing.assert_allclose(np.asarray(u_out.table.characters).ravel(),
                                   np.asarray(g_out.table.characters).ravel(), atol=1e-9)
    assert tqw.corep_content(grey, P, states=states, bands=[1, 2]).counts == {"-P4P4": 1}

    u_table = tqw.hsp_table(urep, bands=[0, 1, 2, 3])
    g_table = tqw.hsp_table(grey, bands=[0, 1, 2, 3])
    assert u_table.complete is True and u_table.refusals == {}, u_table.refusals
    assert u_table.content["P"] == {"-P3": 2, "-P4": 2}
    assert g_table.content["P"] == {"-P3": 2, "-P4P4": 1}
    assert u_table.content["GM"] == {"-GM5": 1, "-GM6": 1, "-GM7": 1, "-GM8": 1}
    assert g_table.content["GM"] == {"-GM5GM7": 1, "-GM6GM8": 1}

    # the spelling invariant as a rule, at every point both presentations reach.
    # `N` is grey-only here (the unitary build reads no content there, and its
    # `-N2` row is the orphan `decompose` drops), so the rule runs at GM/M/P/X.
    u_hsp = tqg.high_symmetry_points(u_group)
    shared = sorted(set(g_table.content) & set(u_table.content))
    assert shared == ["GM", "M", "P", "X"], (shared, sorted(g_table.content))
    for point in shared:
        alphabet = set(tqg.corep_table(u_group, u_hsp[point]))
        spelled = {}
        for glued, count in g_table.content[point].items():
            for lab in _spelled_out(glued, alphabet):
                spelled[lab] = spelled.get(lab, 0) + count
        assert spelled == u_table.content[point], (point, spelled,
                                                   u_table.content[point])

    # one level down: the SITE irreps.  `-1E`/`-2E` at 4a are a complex-conjugate
    # pair, glued by T into the one grey row and kept apart by the unitary tables.
    u_rows = tqe.ebr_rows(u_group)
    assert u_rows, str(u_rows)

    def _row(rows, prefix):
        key = next(k for k in rows if k.startswith(prefix))
        return {lab: v for lab, v in rows[key].items() if v}

    r1, r2 = _row(u_rows, "-1E↑G(2)@4a"), _row(u_rows, "-2E↑G(2)@4a")
    assert r1 != r2, "the two halves of -1E2E are different EBRs on the unitary tables"
    conjugated = {}
    for point, k in u_hsp.items():
        cmap = _conjugate_labels(tqg.corep_table(u_group, k))
        for lab, v in r1.items():
            if lab in cmap:
                conjugated[cmap[lab]] = conjugated.get(cmap[lab], 0) + v
    assert conjugated == r2, (conjugated, r2)

    # ... and the grey row is their sum, spelled out by the same rule
    g_row = _row(tqe.ebr_rows(grey_group), "-1E2E↑G(4)@4a")
    g_hsp = tqg.high_symmetry_points(grey_group)
    assert set(g_hsp) == set(u_hsp), (sorted(g_hsp), sorted(u_hsp))
    spelled = {}
    for point in u_hsp:
        alphabet = set(tqg.corep_table(u_group, u_hsp[point]))
        for glued, count in g_row.items():
            if glued in tqg.corep_table(grey_group, g_hsp[point]):
                for lab in _spelled_out(glued, alphabet):
                    spelled[lab] = spelled.get(lab, 0) + count
    assert spelled == {lab: r1.get(lab, 0) + r2.get(lab, 0)
                       for lab in set(r1) | set(r2)}, (spelled, r1, r2)


# ===========================================================================
# (f3) the anchor realizes the DECLARATION protocol (2026-08-25): the built
#      rep is Ind(declared content) for EVERY declaration, not the default one
# ===========================================================================
#
# Section (f2) measures one declaration -- the default -- on one crystal.  The
# principle the anchor has to realize is larger: **the representation is fixed
# by the declaration of what the orbitals are.**  ``characters=`` pins the twist
# on each declared orbital's site group directly, and Frobenius carries it to
# the whole group, ``lam (x) Ind(rho) = Ind(lam|_site (x) rho)``.  So the claim
# under test is that the table anchor plus the declaration machinery TOGETHER
# are "the built rep is the induced representation of the declared content" --
# and the three seams where that could fail are:
#
#   * a NON-DEFAULT declaration.  ``P3_121``'s order-2 site groups carry a
#     genuine ``B``, so declaring it is a real second data point rather than a
#     re-spelling of the trivial one, and its induced character is a two-line
#     closed form at every one of the six high-symmetry points.
#   * an operation NO occupied site group contains.  ``#80``'s four-fold screw
#     fixes no site of either orbit, so no declaration can ever mention it and
#     no CHARACTER can see it either (``chi(4+) = 0`` identically, at every k).
#     The matrices can, and that is where the anchor does its work.
#   * COMPOSITION.  A declared twist is applied after normalization and
#     re-measured; the anchor must neither fight it nor be applied twice.


def _induced_character_rho(sites, ops, k, site_char=None, s=-1):
    r"""``chi(g) = sum_{j: g q_j = q_j} chi_rho(g|_j) e^{s 2 pi i k.m_j}``.

    :func:`_induced_character` with a **site irrep** rather than the trivial
    one: ``site_char(ind)`` is the character of the site group's element that
    fixes the site, by operation index.  For an order-2 site group the whole
    thing is two values, which is why ``P3_121`` can carry this test in closed
    form.
    """
    chi = {}
    for ind, (R, t) in enumerate(ops, start=1):
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float)
        total = 0j
        for tau in sites:
            m = R @ tau + t - tau
            if np.abs(m - np.rint(m)).max() < 1e-9:
                w = 1.0 if site_char is None else complex(site_char(ind))
                total += w * np.exp(s * 2j * np.pi * np.asarray(k, dtype=float)
                                    @ np.rint(m))
        chi[ind] = total
    return chi


def _induced_matrices(model, group, assignment, rep):
    r"""``O(g) (x) S(g)`` -- the induced rep of the DEFAULT declaration, assembled.

    The k-independent half of a band representation induced from the trivial
    orbital site irrep tensored with spin 1/2: the **measured** site permutation
    (``_assign_orbitals``' action on the model's own orbital positions, read
    through ``symmetry._orbital_permutation``) tensored with the operation's own
    ``sym.spinor_rotation``.  Nothing here is fitted to the build; both factors
    come from outside it.
    """
    nspin, norb = int(model.nspin), int(model.norb)
    syms = list(group.symmetries)
    pos = {int(s.ind): p for p, s in enumerate(syms)}
    out = {}
    for ind in rep.indices:
        p = pos[int(ind)]
        cperm = tqs._orbital_permutation(assignment, p, model)
        S = (np.asarray(syms[p].spinor_rotation, dtype=complex) if nspin == 2
             else np.eye(1, dtype=complex))
        D = np.zeros((norb * nspin,) * 2, dtype=complex)
        for c in range(norb):
            img = int(cperm[c])
            D[np.ix_(np.arange(img * nspin, (img + 1) * nspin),
                     np.arange(c * nspin, (c + 1) * nspin))] = S
        out[int(ind)] = D
    return out


@pytest.fixture(scope="module")
def p3121_declared_rep(p3121_unitary_bundle):
    """The ``#152`` unitary-only build with its ``3a`` reference orbital declared ``B``."""
    model, group, assignment, validated, _default = p3121_unitary_bundle
    declared = tqs._build_rep(model, group, assignment, validated=validated,
                              n_battery=5, characters={0: "B"})
    assert declared, str(declared)
    return declared


@pytest.mark.slow
def test_a_declared_B_site_irrep_on_the_unitary_tables_is_its_own_induced_rep(
        p3121_unitary_bundle, p3121_bundle, p3121_declared_rep):
    r"""The non-default declaration, closed form, at every high-symmetry point.

    ``P3_121``'s two Wyckoff orbits sit on order-2 site groups -- ``3a`` on
    ``{E, 2_x}`` -- whose irreps are ``A`` and ``B``.  ``B`` is a genuine
    second declaration, not a re-spelling of the trivial one, and it is legal:
    the global character ``A2`` of the group restricts to ``B`` on every one of
    those site groups, so the twist exists and the declaration is *pinned* by
    it (``admissible = ('A2',)``).

    The induced character is then two lines.  ``chi_Ind(g) = sum_{j: g q_j = q_j}
    chi_rho(g|_j) e^{-2 pi i k.m_j}``, ``chi_rho = 1`` on ``E`` and ``-1`` on the
    two-fold that fixes the site, so ``B``'s induced character is ``A``'s with
    the sign of the three two-folds reversed -- which is exactly ``A2 (x)
    Ind(A)``, Frobenius written out.  Decomposed against each point's SERVED
    table it gives, and the Hamiltonian measures on the 3a block:

    ======  ==============================  ==============================
    point   ``Ind(A@3a)`` (default)         ``Ind(B@3a)`` (declared)
    ======  ==============================  ==============================
    GM      ``GM1 + GM3``                   ``GM2 + GM3``
    A       ``A1 + A3``                     ``A2 + A3``
    K       ``K1 + K3``                     ``K2 + K3``
    H       ``H1 + H3``                     ``H2 + H3``
    M       ``2 M1 + M2``                   ``M1 + 2 M2``
    L       ``L1 + 2 L2``                   ``2 L1 + L2``
    ======  ==============================  ==============================

    All six agree, and the two that are NOT related by a relabelling of one
    pair (``M`` and ``L``, where the multiplicities themselves move) are the
    ones that make it a measurement rather than a naming convention.

    The unitary build is where this has to be checked, because it is the one
    with no ``T`` to anchor the phase: the table anchor is what keeps the
    ``lam`` on the three-fold generator real (``table_anchored`` is ``True``
    there and ``False`` in the grey build, which has ``T``), and the declaration
    then rides on top of it.  Both builds' matrices agree bit for bit, declared
    or not: ``max |D_u - D_grey| = 0.0``.
    """
    model, u_group, _a, _v, u_default = p3121_unitary_bundle
    u_declared = p3121_declared_rep
    _m, grey_group, g_a, g_v, grey_default = p3121_bundle
    assert u_group.number_str == "152" and grey_group.number_str == "152.34"

    # the site group really does carry a B, and the A2 labelling is what supplies it
    tab = u_default.characters.site_tables[0]
    assert set(tab.labels) == {"A", "B"} and int(tab.order) == 2
    assert u_declared.characters.irreps == ("B",) * 6
    assert [o.determined_by for o in u_declared.characters.orbitals] == \
        ["declaration"] + ["hamiltonian"] * 5, \
        "one declaration fixes the labelling, and the hoppings carry it to every row"
    assert all(o.determined_by == "lowest-l" for o in u_default.characters.orbitals), \
        "nothing was declared, so every row rests on the lowest-l rule"

    # the anchor is load-bearing here and only here
    anchored = {int(i): u_default._normalization[int(i)].table_anchored
                for i in u_default._generators}
    assert anchored == {2: True, 4: False}, anchored
    assert all(grey_default._normalization[int(i)].table_anchored is False
               for i in grey_default._generators)
    assert all(u_declared._normalization[int(i)].table_anchored is anchored[int(i)]
               for i in u_declared._generators), "the declaration does not move the anchor"

    hsp = tqg.high_symmetry_points(u_group)
    sites = tqm.p3121_sites()[0]
    B_of = lambda ind: 1.0 if ind == 1 else -1.0        # noqa: E731 - a two-value table

    want = {"GM": ({"GM1": 1, "GM3": 1}, {"GM2": 1, "GM3": 1}),
            "A": ({"A1": 1, "A3": 1}, {"A2": 1, "A3": 1}),
            "K": ({"K1": 1, "K3": 1}, {"K2": 1, "K3": 1}),
            "H": ({"H1": 1, "H3": 1}, {"H2": 1, "H3": 1}),
            "M": ({"M1": 2, "M2": 1}, {"M1": 1, "M2": 2}),
            "L": ({"L1": 1, "L2": 2}, {"L1": 2, "L2": 1})}
    for point, (a_counts, b_counts) in want.items():
        k = hsp[point]
        table = tqg.corep_table(u_group, k)
        assert table, (point, str(table))
        little = sorted(next(iter(table.values())))
        chi_a = _induced_character_rho(sites, tqm.P3121_OPS, k)
        chi_b = _induced_character_rho(sites, tqm.P3121_OPS, k, site_char=B_of)
        assert _decompose_against(table, {i: chi_a[i] for i in little}) == a_counts, point
        assert _decompose_against(table, {i: chi_b[i] for i in little}) == b_counts, point

        states = _solve_at(model, k)
        got_a = tqw.corep_content(u_default, k, states=states, bands=[0, 1, 2])
        got_b = tqw.corep_content(u_declared, k, states=states, bands=[0, 1, 2])
        assert got_a and got_b, (point, str(got_a), str(got_b))
        assert got_a.counts == a_counts, (point, got_a.counts)
        assert got_b.counts == b_counts, (point, got_b.counts)

    # and the declared build is the grey declared build, bit for bit
    grey_declared = tqs._build_rep(_m, grey_group, g_a, validated=g_v, n_battery=5,
                                   characters={0: "B"})
    assert grey_declared, str(grey_declared)

    def _echo(rep):
        return (rep.characters.irreps,
                tuple((o.function, o.determined_by) for o in rep.characters.orbitals))

    assert _echo(u_declared) == _echo(grey_declared), "the echo record, entry for entry"
    assert _echo(u_default) == _echo(grey_default)
    assert _echo(u_declared)[0] == ("B",) * 6

    def _key(op):
        return (tuple(np.asarray(op.rotation, dtype=int).ravel()),
                tuple(np.round(np.asarray(op.translation) % 1.0, 6)))

    for urep, grep in ((u_default, grey_default), (u_declared, grey_declared)):
        grey_of = {_key(grep._operation(i)): i for i in grep.indices
                   if not grep._operation(i).time_reversal}
        worst = max(float(np.abs(np.asarray(urep.matrices[i])
                                 - np.asarray(grep.matrices[grey_of[_key(urep._operation(i))]])
                                 ).max()) for i in urep.indices)
        assert worst == 0.0, "declared or not, the two anchors are the same anchor"


@pytest.mark.slow
@pytest.mark.parametrize("fixture", ["i41_soc_bundle"])
def test_the_operation_no_site_group_contains_is_pinned_by_induction(request, fixture):
    r"""Frobenius closure on ``#80``'s four-fold screw, which no declaration mentions.

    Neither occupied site group of ``i41`` contains ``4+``: the ``4a`` orbit's
    is ``{E, 2_z}`` and the general ``8b``'s is trivial (measured, from
    ``_assign_orbitals``).  So a declaration can say nothing about ``4+``
    directly, and what fixes ``D(4+)`` has to be induction -- which is exactly
    the claim the table anchor is asked to realize on the unitary tables, where
    there is no ``T``.

    Two measurements, and the order matters.

    **The character route is vacuous here, and that is a fact worth stating.**
    The ``4_1`` screw fixes no site of either orbit, so ``chi_Ind(4+) = 0``
    identically -- at *every* k -- and ``tr S(2_z) = 0`` kills the ``2_z`` term
    besides.  The closed-form induced characters are ``(12, 0, 0, 0)`` at both
    ``GM`` and ``M`` for the whole 12-band content (``4`` from ``4a``, ``8``
    from ``8b``), the built rep reproduces them, and a twist by ANY linear
    character of ``#80``'s ``Z4`` would reproduce them too.  Whole-block
    characters cannot close this seam.

    **The matrices can.**  The induced representation of the declared content is
    ``O(g) (x) S(g)`` -- the measured site permutation tensored with the tables'
    own spinor lift -- and the built rep equals it on all four operations,
    ``4+`` included, at **one or two ulp of an O(1) matrix entry**, in the
    unitary and the grey build alike.  The residual is the accumulated error of
    ``_propagate``'s BFS products of 12x12 complex matrices -- ``4+`` is the
    generator and is assembled directly, ``2_z`` and ``4-`` are propagated -- so
    it moves with the summation order a given BLAS picks, and the bound is
    spelled in ulps rather than as a decimal for exactly that reason.
    The search's own record says the same thing from the other side: the winner
    is ``orientation = +1`` with every free class sign ``+1``, i.e. the bare
    ansatz, and the anchor's job was to leave the phase at ``lam = 1`` rather
    than at the ``i`` the positivity tie used to hand it.
    """
    bundle = request.getfixturevalue(fixture)
    model = bundle[0]
    u_group = tqg._group_of(model, include_TR=False)
    assert u_group and u_group.number_str == "80", str(u_group)
    u_assign = tqg._assign_orbitals(model, u_group)
    urep = tqs._build_rep(model, u_group, u_assign, n_battery=5,
                          validated=tqg._validate_group(model, u_group, nk=4))
    assert urep and urep.antiunitary_operator[0] is None, str(urep)

    # no occupied site group contains the four-fold
    site_groups = {s: tuple(int(i) for i in u_assign.site_group[s])
                   for s in range(len(u_assign.site_orbitals))}
    assert site_groups == {0: (1, 2), 1: (1, 2), 2: (1,), 3: (1,), 4: (1,), 5: (1,)}
    four = int(urep._generators[0])
    assert four == 3 and not any(four in g for g in site_groups.values())

    # (1) the character route, closed form -- and blind
    hsp = tqg.high_symmetry_points(u_group)
    su2 = _table_su2("80")
    orbit_a, orbit_b = tqm.i41_sites()
    for point in ("GM", "M"):
        k = hsp[point]
        chi = {i: (_spinful_induced_character(orbit_a, tqm.I41_PRIM_OPS, su2, k)[i]
                   + _spinful_induced_character(orbit_b, tqm.I41_PRIM_OPS, su2, k)[i])
               for i in range(1, 5)}
        np.testing.assert_allclose([chi[i] for i in range(1, 5)], [12, 0, 0, 0],
                                   atol=1e-9)
        out = tqw.corep_content(urep, k, states=_solve_at(model, k), bands=list(range(12)))
        assert out, (point, str(out))
        measured = np.asarray(out.table.characters).sum(axis=0)
        np.testing.assert_allclose(measured, [chi[i] for i in out.table.columns],
                                   atol=1e-8)
        assert abs(chi[four]) < 1e-12, "the four-fold's character is zero, at every k"

    # (2) the matrix route -- the induced rep of the declared content, on all four
    induced = _induced_matrices(model, u_group, u_assign, urep)
    worst = {int(i): float(np.abs(np.asarray(urep.matrices[int(i)]) - induced[int(i)]).max())
             for i in urep.indices}
    assert max(worst.values()) <= 16 * _ULP, worst
    assert worst[four] <= 16 * _ULP, "including the one no site group contains"
    assert urep._choice["orientation"] == 1
    assert all(abs(v - 1.0) < 1e-12 for v in urep._choice["signs"].values()), \
        "the winner IS the bare ansatz, so O (x) S is not fitted to it"
    assert urep._normalization[four].table_anchored is True
    assert urep._normalization[four].factor == pytest.approx(1.0, abs=1e-12)

    # the grey build induces the same matrices, which is what makes the two agree
    grey = bundle[4]
    grey_induced = _induced_matrices(model, bundle[1], bundle[2], grey)
    assert max(float(np.abs(np.asarray(grey.matrices[int(i)]) - grey_induced[int(i)]).max())
               for i in grey.indices
               if not grey._operation(int(i)).time_reversal) <= 16 * _ULP


@pytest.mark.slow
def test_the_twist_no_site_group_separates_is_a_genuine_equivalence(i41_soc_bundle):
    r"""The other half of the principle, and the reason ``#80`` reports ``pinned=False``.

    The owner's statement of the protocol has two branches: a character twist
    trivial on every occupied site group either **contradicts** the induced
    characters where they do not vanish, or is a **genuine equivalence** where
    they do.  This is the equivalence; what a complex twist serves is measured
    in ``test_declared_content.py``.

    ``Z4``'s two REAL characters are ``A`` (all ``+1``) and
    ``B = (1, +1, -1, -1)`` on ``(E, 2_z, 4+, 4-)`` -- note ``B(2_z) = +1``,
    because ``2_z = (4+)^2``.  ``B`` is therefore trivial on ``{E, 2_z}`` and on
    the trivial ``8b`` group alike, so no declaration can separate it from
    ``A``; and since ``chi_Ind(4+/-) = 0`` at every k, ``B (x) Ind`` is
    character-identical to ``Ind`` everywhere.  The record says exactly that:
    both labellings give every row the same site irrep, so every row is
    ``hamiltonian``.  And the equivalence is *measured*, not argued: the
    ``B``-twisted rep is built by hand and its content is identical at every
    high-symmetry point, on the valence block and on all twelve bands, and per
    level at ``P`` (``-P3``, ``2 x -P4``, ``-P3``), at a product-table residual
    of 4.5e-16.  That is the sharp contrast with the COMPLEX twist
    ``lam(4+) = i``, which is equally multiplicative and exchanges every ``P``
    label -- the difference being that ``chi_Ind`` does not vanish on
    ``2_z = (4+)^2``, where ``B`` is ``+1`` and ``lam`` is ``-1``.
    """
    model = i41_soc_bundle[0]
    u_group = tqg._group_of(model, include_TR=False)
    u_assign = tqg._assign_orbitals(model, u_group)
    u_valid = tqg._validate_group(model, u_group, nk=4)
    default = tqs._build_rep(model, u_group, u_assign, validated=u_valid, n_battery=5)
    assert default, str(default)
    assert all(o.determined_by == "hamiltonian" for o in default.characters.orbitals), \
        "B is trivial on every occupied site group, so both labellings give each row " \
        "the same site irrep"
    assert default.characters.irreps == ("-1E + -2E", "-1E + -2E") + ("2x-A",) * 4

    # the equivalence, measured: twist by B and nothing anywhere moves
    def _B(op):                       # B(g) = -1 exactly on the two four-folds
        return -1.0 if abs(float(np.trace(np.asarray(op.rotation, dtype=float))) - 1.0) \
            < 1e-9 else 1.0

    chi_B = {int(i): _B(default._operation(int(i))) for i in default.indices}
    assert sorted(chi_B.values()) == [-1.0, -1.0, 1.0, 1.0]
    twisted = dataclasses.replace(
        default, matrices={i: chi_B[int(i)] * np.asarray(default.matrices[i])
                           for i in default.indices})
    pos = {int(s.ind): p for p, s in enumerate(u_group.symmetries)}
    prod, _tdiff, spf = u_group.get_product_table(get_diff=True)
    assert float(tqs._product_table_residual(
        {pos[int(i)]: twisted.matrices[i] for i in twisted.indices},
        [pos[int(i)] for i in twisted.indices], u_group.symmetries, prod, spf)) < 1e-13

    for bands in ([0, 1, 2, 3], list(range(12))):
        ta = tqw.hsp_table(default, bands=bands)
        tb = tqw.hsp_table(twisted, bands=bands)
        assert ta.complete is tb.complete is True
        assert ta.content == tb.content, (bands, ta.content, tb.content)
    P, states, _w, mult = _P_valence(i41_soc_bundle)
    for block in mult:
        a_out = tqw.corep_content(default, P, states=states, bands=list(block))
        b_out = tqw.corep_content(twisted, P, states=states, bands=list(block))
        assert a_out and b_out and a_out.counts == b_out.counts, (block, a_out.counts)


@pytest.fixture(scope="module")
def lin3g_realized_unitary(lin3g):
    """``(model, group, assignment, validated)`` for lin3g's realized ``P4mm`` (#99).

    ``realized_group`` is grey by construction, so the unitary presentation of
    the *realized* group is assembled from the same public steps with
    ``include_TR=False``: slab presentation, the two-species decoration that
    removes ``sigma_h``, ``_group_of``, ``_assign_orbitals``, ``_validate_group``.
    Eight unitary operations, none undetermined -- so unlike the ``#123`` slab
    group the little groups are complete and the content is readable.
    """
    m3 = tqg._slab_presentation(lin3g)
    assert m3 is not None and not isinstance(m3, tqg.Refusal), str(m3)
    cell, cell_typat = tqg._decorated_cell(m3, [0, 0, 1, 1])
    group = tqg._group_of(cell, typat=cell_typat, spinor=True, include_TR=False)
    assert group and group.number_str == "99", str(group)
    assignment = tqg._assign_orbitals(m3, group, typat=[0, 0, 1, 1])
    assert assignment, str(assignment)
    return m3, group, assignment, tqg._validate_group(m3, group, nk=6)


@pytest.mark.slow
def test_the_declared_twist_and_the_table_anchor_compose_exactly_once(
        lin3g_realized_unitary, lin3g_twisted):
    r"""Composition order, on the crystal where the anchor is genuinely load-bearing.

    lin3g's four-fold generator is the operation whose positivity tie the anchor
    exists to break, and ``B1`` -- declaring orbital 0 to be the ``d_x2-y2``
    partner -- is ``-1`` on exactly that generator.  So this is the case where
    the two could fight, and the measurement is that they do not:

    * the declared build is the default build times the character, exactly:
      ``max |D_B1(g) - B1(g) D_default(g)| = 0.0`` over all eight unitary
      operations of the realized ``P4mm``;
    * the generators' normalization records are *identical* in the two builds
      (``factor`` and ``table_anchored`` both), because the anchor runs on the
      search matrices and the twist is applied afterwards and re-measured -- so
      it is applied once, not squared into ``B1 (x) B1 = A1``;
    * no ``InternalError`` is raised on either side, and the declared build's
      own reference block is a real ``-1`` multiple of ``S(4+)`` at defect 0 up
      to roundoff (bit-exact on the development machine, one ulp on an M1),
      which the anchor **accepts**: its reality half is the tables', and
      re-normalizing the declared matrix hands back ``lam = -1`` -- its
      positivity half would put the reference orbital back on ``A1``, which is
      why the build applies the twist *after* normalization and never before;
    * the content moves the way ``B1`` says it must and no other way:
      ``-GM6 <-> -GM7``, ``-M6 <-> -M7``, ``-X5`` fixed (``B1`` restricted to
      ``X``'s little group is trivial), on all four lin3g band blocks.  Section
      (a) makes the same measurement in the grey presentation and carries it
      into ``decompose``; this is the statement that the tables' anchor buys the
      identical answer with no ``T`` in the group.

    Everything above is on the UNITARY tables.  The grey build of the same
    declaration is bit-identical (``0.0``), which is the section (f2) statement
    carried past the default declaration.

    And the two error classes do not blur on a group with no ``T``: a
    declaration the Hamiltonian contradicts -- both ``1a`` orbitals ``s``, where
    the hoppings measure the second ``B1`` relative to the first -- is a
    measurement about the model and comes back a ``Refusal`` naming the
    subgroup it holds on, where a gauge the tables fix and this package failed
    to pin would raise.
    """
    model, u_group, u_assign, u_valid = lin3g_realized_unitary
    base = tqs._build_rep(model, u_group, u_assign, validated=u_valid, n_battery=5)
    twisted = tqs._build_rep(model, u_group, u_assign, validated=u_valid, n_battery=5,
                             characters={0: "B1"})
    assert base and twisted, (str(base), str(twisted))
    assert len(base.indices) == 8 and base.antiunitary_operator[0] is None
    assert len(twisted.indices) == 8

    assert twisted.characters.irreps == ("-E2", "-E1", "-E", "-E"), (
        "the 2c rows are named by their point group here: this group carries no "
        "antiunitary operation, and its own EBR entries list nothing at that letter, "
        "every double-valued site irrep there inducing a composite band representation")
    assert [o.named_by for o in twisted.characters.orbitals] == \
        ["tables", "tables", "point group", "point group"]
    # ... which the atomic-limit table writes out, in the elementary band
    # representations at 1a, on the closure of the mirror plane through 2c
    doublet, pair = twisted.characters.band_representations
    assert (pair.orbitals, pair.wyckoff) == ((2, 3), "2c")
    assert [(m, name, position.split("(")[0]) for m, name, position in pair.terms] == \
        [(1, "-E1↑G(2)", "1a"), (1, "-E2↑G(2)", "1a")]
    summary = twisted.characters.summary()
    assert pair.sentence() in summary and doublet.sentence() not in summary
    # B1, by name: -1 on the fourfold and on the two diagonal mirrors
    frame = twisted.characters.frame
    b1 = {"E": 1.0, "C2(z)": 1.0, "C4(z)": -1.0, "C4^-1(z)": -1.0,
          "m(y)": 1.0, "m(x)": 1.0, "m([110])": -1.0, "m([1-10])": -1.0}
    chi = {int(i): b1[frame[int(i)]] for i in base.indices}

    worst = max(float(np.abs(np.asarray(twisted.matrices[i])
                             - complex(chi[int(i)]) * np.asarray(base.matrices[i])).max())
                for i in base.indices)
    assert worst == 0.0, "applied once, exactly, and after normalization"
    for gen in base._generators:
        before, after = base._normalization[int(gen)], twisted._normalization[int(gen)]
        assert after.factor == pytest.approx(before.factor, abs=1e-15), gen
        assert after.table_anchored is before.table_anchored, gen
    assert base.product_residual < 1e-13 and twisted.product_residual < 1e-13

    four = next(int(i) for i in base._generators if base._normalization[int(i)].order == 4)
    S = np.asarray(base._operation(four).spinor_rotation, dtype=complex)
    for rep, want in ((base, 1.0), (twisted, -1.0)):
        D = np.asarray(rep.matrices[four])
        img = int(np.abs(D[:, 0]).argmax()) // 2
        rho, res = tqs._proportionality(D[np.ix_(np.arange(img * 2, img * 2 + 2),
                                                 np.arange(2))], S)
        assert res <= 4 * _ULP and abs(complex(rho).imag) <= 4 * _ULP, (res, rho)
        assert rho.real == pytest.approx(want, abs=1e-12)

        again = tqs.normalize_rep(D, rep._operation(four), u_group,
                                  anchor=tqs.RepAnchor(rows=(2 * img, 2 * img + 1),
                                                       cols=(0, 1), internal=S))
        assert again, str(again)
        assert again.table_anchored is True
        assert len(again.candidates) == 4 and len(again.accepted) == 2
        assert all(abs(complex(lam).imag) < 1e-12 for lam in again.accepted)
        assert again.factor == pytest.approx(want, abs=1e-12), (
            "reality is the tables'; the sign is the reference orbital's, and a "
            "declaration is exactly a statement about the reference orbital")

    clash = tqs._build_rep(model, u_group, u_assign, validated=u_valid, n_battery=5,
                           characters={0: "s", 1: "s"})
    assert not clash and clash.gate == "declaration_breaks_group", str(clash)
    assert isinstance(clash, tqg.Refusal), \
        "a measurement about the model is a result, not an exception"
    assert clash.context["orbitals"] == (1,)
    assert clash.context["relative_character"] == "B1"
    assert {b["name"] for b in clash.context["broken"]} == {
        "C4(z)", "C4^-1(z)", "m([110])", "m([1-10])"}, \
        "the four operations B1 is odd under, and the four that leave the subgroup"
    assert len(clash.context["subgroup"]) == 4, "E, C2(z), m(y), m(x)"

    # the content, on all four band blocks
    swap = {"-GM6": "-GM7", "-GM7": "-GM6", "-M6": "-M7", "-M7": "-M6", "-X5": "-X5"}
    for bands in (P6, list(P4), list(P2), list(Q2)):
        ta = tqw.hsp_table(base, bands=bands)
        tb = tqw.hsp_table(twisted, bands=bands)
        assert ta.complete is tb.complete is True
        assert set(ta.content) == set(tb.content) == {"GM", "M", "X"}
        for point in ta.content:
            assert {swap[n]: c for n, c in ta.content[point].items()} == tb.content[point], \
                (bands, point, ta.content[point], tb.content[point])
    assert tqw.hsp_table(base, bands=P6).content["GM"] == {"-GM6": 2, "-GM7": 1}
    assert tqw.hsp_table(twisted, bands=P6).content["GM"] == {"-GM7": 2, "-GM6": 1}

    # and the grey build of the same declaration -- section (a)'s own fixture
    grey = lin3g_twisted.rep
    assert [(o.function, o.determined_by) for o in grey.characters.orbitals] == \
        [(o.function, o.determined_by) for o in twisted.characters.orbitals], \
        "the echo record is the same in both presentations, row for row"
    assert grey.characters.irreps == twisted.characters.irreps
    assert [o.named_by for o in grey.characters.orbitals[2:]] == ["tables"] * 2, (
        "the one thing that does differ: a site irrep is named by the entries the "
        "group's own EBR table carries, and the grey group and its unitary subgroup "
        "are two groups with two tables")
    assert [(m, name, position.split("(")[0]) for m, name, position
            in grey.characters.band_representations[1].terms] == [(1, "-E↑G(4)", "2c")]

    def _key(op):
        return (tuple(np.asarray(op.rotation, dtype=int).ravel()),
                tuple(np.round(np.asarray(op.translation) % 1.0, 6)))

    grey_of = {_key(grey._operation(i)): i for i in grey.indices
               if not grey._operation(i).time_reversal}
    assert max(float(np.abs(np.asarray(twisted.matrices[i])
                            - np.asarray(grey.matrices[grey_of[_key(twisted._operation(i))]])
                            ).max()) for i in twisted.indices) == 0.0


# ===========================================================================
# (g)  the measured channel layout: what the Hamiltonian glues at every site
# ===========================================================================
#
# A channel is the internal space one factor of the ansatz
# ``D(g) = O(g) (x) S(g)`` acts on, and which orbitals share one is a
# **measurement**: the finest partition of every site's orbitals for which
# every validated operation's solution space still holds a nonsingular
# intertwiner in block form.  One channel per orbital where nothing mixes, the
# multiplet where the Hamiltonian glues one -- and the served matrices carry
# the tables' own lift on the first and the model's own measured block on the
# second.  Three models say what that means, each pathological in its own way:
#
#   * a square lattice with one ``p_x, p_y`` pair at a single ``4mm`` site and
#     Slater-Koster hoppings, ``t_sigma`` along the bond and ``t_pi`` across
#     it. Its two orbitals never hop into each other, so the commutant of
#     ``H(k)`` is a torus and every operation's solution space is
#     two-dimensional: the doublet is measured against that freedom rather than
#     read off one member of it.
#   * two decoupled degenerate orbitals at one such site, which is the same
#     freedom without the mixing: a diagonal member exists, so the finest
#     partition is one channel per orbital and the pair is NOT a doublet --
#     and beside them the orbits whose sites write copies of one site irrep in
#     bases turned from site to site, where the copies are one channel.
#   * two such sublattices in one cell with no hopping between them, where the
#     commutant additionally exchanges the two and no representation of the
#     cell's group is unique.


def _square_pxpy(sublattices=1):
    """A ``p_x, p_y`` pair at every ``4mm`` site of a square lattice, spinless.

    Slater-Koster nearest-neighbour hoppings with ``t_sigma = 1.0`` along the
    bond and ``t_pi = 0.3`` across it: along ``x`` the ``p_x`` orbital hops by
    ``t_sigma`` and ``p_y`` by ``t_pi``, and along ``y`` the other way about.
    Nothing couples ``p_x`` to ``p_y``, and with ``sublattices=2`` nothing
    couples the ``(0, 0)`` site to the ``(1/2, 1/2)`` one either.
    """
    from pythtb import Lattice, TBModel

    centres = [[0.0, 0.0]] * 2 + ([[0.5, 0.5]] * 2 if sublattices > 1 else [])
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], centres, periodic_dirs=[0, 1]),
                spinful=False)
    for s in range(sublattices):
        o = 2 * s
        m.set_hop(1.0, o, o, [1, 0], mode="add")
        m.set_hop(0.3, o + 1, o + 1, [1, 0], mode="add")
        m.set_hop(0.3, o, o, [0, 1], mode="add")
        m.set_hop(1.0, o + 1, o + 1, [0, 1], mode="add")
    return m


@pytest.mark.slow
def test_the_square_pxpy_doublet_is_measured_against_a_torus_of_a_commutant():
    """The owner's example: ``P4mm1' #99.164`` with ``E`` at its ``1a`` site.

    ``p_x`` and ``p_y`` never hop into each other here, so the commutant of
    ``H(k)`` contains a phase on each of them and every operation's solution
    space is two-dimensional: covariance alone does not say which member the
    representation is.  The two orbitals are two sectors the fourfold axis
    exchanges, and every representation the hoppings leave open is induced
    from a character of their stabilizer: the doublet, where ``C4`` is the
    rotation by a quarter turn on the pair (character 0) and the mirrors its
    diagonal reflections (character 0), which is ``E``; and ``A1 + B1``, where
    ``C4`` is the bare exchange of the two orbitals, with the same covariance
    and the same product table.  Which one the pair is, the Hamiltonian cannot
    say, and the lowest-``l`` rule reads it ``p``: ``E``, returned in the
    tables' own matrices of ``E`` on ``(p_x, p_y)``, ``C4(z)`` carrying ``p_x``
    onto ``+p_y``.
    """
    out = tqg.realized_group(_square_pxpy(), typat=[0, 0], cross_check=False)
    assert out, str(out)
    assert out.group.number_str == "99.164" and out.group.size == 16
    assert len(out.validated.indices) == 16 and out.descent is None

    multiplets = out.rep.multiplets
    assert multiplets.by_site == (((0, 1),),) and multiplets.dims == ((2,),)
    assert multiplets.accidental == ()

    rec = out.rep.characters
    assert rec.irreps == ("E", "E")
    assert [o.determined_by for o in rec.orbitals] == ["lowest-l"] * 2
    assert [o.dim for o in rec.orbitals] == [2, 2]
    frame = {name: ind for ind, name in rec.frame.items()}
    chi = rec.orbitals[0].character
    assert complex(chi[frame["C4(z)"]]).real == pytest.approx(0.0, abs=1e-9)
    assert complex(chi[frame["m(x)"]]).real == pytest.approx(0.0, abs=1e-9)
    assert max(out.rep.residuals.values()) <= GATE
    assert out.rep.product_residual < 1e-13
    # every operation, the antiunitary ones too, acts on the pair as its rotation
    # acts on (x, y): the tables' E on (p_x, p_y), whatever the reading drew
    syms = {int(s.ind): s for s in out.group.symmetries}
    for ind, M in out.rep.matrices.items():
        assert np.allclose(M, np.asarray(syms[int(ind)].rotation_cart)[:2, :2], atol=1e-12), ind

    # ... and `characters={(0, 1): "E"}` is a check on that measurement, not an
    # anchor: the declaration names the channel the layout measured, so no
    # multiplet has to be guessed at, and `E (x) chi` is `E` again for every
    # real character of `4mm`, so the served matrices are these ones entry for
    # entry.
    declared = tqg.realized_group(_square_pxpy(), typat=[0, 0], cross_check=False,
                                  characters={(0, 1): "E"})
    assert declared, str(declared)
    assert declared.group.number_str == "99.164" and declared.descent is None
    assert declared.rep.characters.irreps == ("E", "E")
    assert [o.determined_by for o in declared.rep.characters.orbitals] == \
        ["declaration"] * 2
    assert _same_matrices(out.rep, declared.rep, atol=0.0)


def _decoupled_pair():
    """Two orbitals at one ``4mm`` site with identical isotropic hoppings, coupled to nothing."""
    from pythtb import Lattice, TBModel

    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]] * 2,
                        periodic_dirs=[0, 1]), spinful=False)
    for o in (0, 1):
        m.set_hop(1.0, o, o, [1, 0], mode="add")
        m.set_hop(1.0, o, o, [0, 1], mode="add")
    return m


#: The multiplets of the ``8c`` models with the ``s`` at ``2a`` first: the ``d``
#: shell turned inside its isotypic components, and ``p`` before an aligned ``d``.
_THE_8C_D_TURNED = (((0,),),) + tuple(
    ((5 * s + 1, 5 * s + 2, 5 * s + 3, 5 * s + 4), (5 * s + 5,)) for s in range(4))
_THE_8C_PD = (((0,),),) + tuple(
    ((8 * s + 1, 8 * s + 2), (8 * s + 3,), (8 * s + 4, 8 * s + 7), (8 * s + 5, 8 * s + 6),
     (8 * s + 8,)) for s in range(4))


def _spinor_import(model):
    """A spinful model as a ``wannier90`` spinor import delivers it: the spin in the orbital index."""
    flat = tqsp.spinless_presentation(model, order="interleaved")
    assert isinstance(flat, tuple), str(flat)
    return flat[0]


def _held(out):
    """``{site: {irrep: copies}}`` -- what each site holds, read off the rows' named sums.

    A row prints its multiplet's content, whose dimension is the row's ``dim``,
    and carries one orbital's states of it, so each row adds the terms of its
    label weighted by ``nspin / dim`` and a multiplet adds its content once.
    """
    per_row = int(out.rep.model.nspin)
    held = {}
    for o in out.rep.characters.orbitals:
        for term, k in tqs._label_terms(o.irrep):
            here = held.setdefault(o.site, {})
            here[term] = here.get(term, 0.0) + k * per_row / o.dim
    return {s: {term: round(n, 9) for term, n in terms.items()} for s, terms in held.items()}


_R3M_TURN = 0.71      # radians, about a direction of no symmetry

#: Per case: the model, the call's keywords, the group, the multiplets, what
#: each site holds -- ``{site: {irrep: copies}}``, read off the rows' named
#: sums -- the rows' labels in model order, where the rows' labels are pinned at
#: all, and, for a shell written in a basis turned inside its site, the same
#: crystal written aligned.  The ``8c`` models are read on a ``4 x 4 x 4`` mesh:
#: their group, multiplets and rows are the ones the default ``6 x 6 x 6`` reads.
_BOUND_AT_A_SITE = {
    "two decoupled orbitals": (_decoupled_pair, {"typat": [0, 0], "cross_check": False},
                               "99.164", (((0, 1),),), {0: {"A1": 2}}, ("A1", "A1"), None),
    "P2/m, one site's pair turned": (lambda: tqm.p2m_mirror_pair(angle=0.37),
                                     {}, "10.43", (((0, 1),), ((2, 3),)),
                                     {0: {"A'": 2}, 1: {"A'": 2}}, None, None),
    "P2/m, p_y and p_x listed in the other order at one site": (
        lambda: tqm.p2m_mirror_pair(functions=("px", "py"), listed_at_b=("py", "px")),
        {}, "10.43", (((0,), (1,)), ((2,), (3,))),
        {0: {"A'": 1, "A''": 1}, 1: {"A'": 1, "A''": 1}}, ("A'", "A''", "A''", "A'"), None),
    "P2/m, one site's p_x written as i times itself": (
        lambda: tqm.p2m_mirror_pair(phases={2: 1j}), {}, "10.43",
        (((0,), (1,)), ((2,), (3,))), {0: {"A'": 2}, 1: {"A'": 2}}, ("A'",) * 4, None),
    "Im-3m 8c, d turned at every site, read from an 8c orbital": (
        lambda: tqm.im3m_8c(("d",), rotated=True, centre=True),
        {"nk": 4, "reference_orbital": 1},
        "229.141", _THE_8C_D_TURNED,
        {0: {"A1g": 1}, **{s: {"Eg": 2, "A1g": 1} for s in range(1, 5)}}, None, None),
    "Im-3m 8c, p and d in each site's own frame": (
        lambda: tqm.im3m_8c(("p", "d"), centre=True), {"nk": 4}, "229.141",
        _THE_8C_PD,
        {0: {"A1g": 1}, **{s: {"Eu": 1, "A2u": 1, "Eg": 2, "A1g": 1} for s in range(1, 5)}},
        None, None),
    "R3m 3a, a p shell turned inside its site": (
        lambda: tqm.r3m_p_shell(angle=_R3M_TURN), {"decorations": tqm.R3M_SHADOW}, "160.66",
        (((0, 1, 2),),), {0: {"A1": 1, "E": 1}}, ("A1 + E",) * 3, tqm.r3m_p_shell),
    "P4/mmm 1a, a p shell listed p_z first, turned inside its site": (
        lambda: tqm.p4mmm_p_shell(angle=_R3M_TURN), {}, "123.340",
        (((0, 1, 2),),), {0: {"A2u": 1, "Eu": 1}}, ("A2u + Eu",) * 3, tqm.p4mmm_p_shell),
    "R3m 3a, the p shell with spin-orbit coupling turned inside its site": (
        lambda: tqm.r3m_p_shell(angle=_R3M_TURN, spinful=True),
        {"decorations": tqm.R3M_SHADOW}, "160.66", (((0, 1, 2),),),
        {0: {"-E1": 2, "-1E2E": 1}}, ("2x-E1 + -1E2E",) * 3,
        lambda: tqm.r3m_p_shell(spinful=True)),
    "R3m 3a, the same as a wannier90 spinor import, read with spin=": (
        lambda: _spinor_import(tqm.r3m_p_shell(angle=_R3M_TURN, spinful=True)),
        {"decorations": tqm.R3M_SHADOW, "spin": "interleaved"}, "160.66",
        (((0, 1, 2, 3, 4, 5),),), {0: {"-E1": 2, "-1E2E": 1}}, ("2x-E1 + -1E2E",) * 6,
        lambda: _spinor_import(tqm.r3m_p_shell(spinful=True))),
    "R-3m 6c, a spinful s pair": (
        tqm.r3m_s_6c, {}, "166.98", (((0,),), ((1,),)), {0: {"-E1": 1}, 1: {"-E1": 1}},
        ("-E1", "-E1"), None),
    "R-3m 6c, the p shell with spin-orbit coupling turned inside each site": (
        lambda: tqm.r3m_p_shell(angle=_R3M_TURN, spinful=True, pair=True), {}, "166.98",
        (((0, 1, 2),), ((3, 4, 5),)), {s: {"-E1": 2, "-1E2E": 1} for s in (0, 1)},
        ("2x-E1 + -1E2E",) * 6, lambda: tqm.r3m_p_shell(spinful=True, pair=True)),
    "P-43m 3c, a spinful p shell, one site's shell turned, the other two aligned": (
        lambda: tqm.p43m_3c_p_shell(turned=True), {"decorations": tqm.P43M_SHADOW}, "215.71",
        (((0, 1, 2),), ((3, 4, 5),), ((6, 7, 8),)),
        {s: {"-E2": 2, "-E1": 1} for s in range(3)}, ("2x-E2 + -E1",) * 9,
        tqm.p43m_3c_p_shell),
}

#: The band representation each case's ``6c`` orbit induces, where it is pinned:
#: its terms, each at the position it is written at.
_INDUCED = {
    "R-3m 6c, a spinful s pair": [
        ((1, "-E1g↑G(2)", "3a"), (1, "-E1u↑G(2)", "3a"))],
    "R-3m 6c, the p shell with spin-orbit coupling turned inside each site": [
        ((1, "-1Eg2Eg↑G(2)", "3a"), (1, "-1Eu2Eu↑G(2)", "3a"), (2, "-E1g↑G(2)", "3a"),
         (2, "-E1u↑G(2)", "3a"))],
}

#: The lowest two bands at Γ where the functions the orbitals read as settle
#: them: the ``p`` shell's, odd under the inversion carrying one ``6c`` site
#: onto the other, turned and aligned alike.
_LOWEST_PAIR_AT_GAMMA = {
    "R-3m 6c, the p shell with spin-orbit coupling turned inside each site": {"-GM8": 1},
}

#: The same crystal with each site's axis uncoupled from its pair, where the
#: lowest six bands' labels rest on a one-dimensional character the hoppings
#: leave open: per reading, the coupled part the note names and what chose.
_LABELS_REST_ON = {
    "P-43m 3c, a spinful p shell, one site's shell turned, the other two aligned": (
        (lambda: tqm.p43m_3c_p_shell(decoupled=True), "orbitals 1, 2, 3, 5, 6 and 7", "bands"),
        (lambda: tqm.p43m_3c_p_shell(decoupled=True, turned=True, turned_site=1),
         "every orbital", "orbitals")),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(_BOUND_AT_A_SITE))
def test_a_site_s_operations_bind_its_orbitals_and_copies_of_one_irrep_are_one(case):
    """The finest partition the site's own operations allow; copies of one irrep together.

    Two orbitals at one ``4mm`` site with identical isotropic hoppings, coupled
    to nothing, are identical copies: the matrices the Hamiltonian conserves
    exchange them, so they are one multiplet in whatever basis they are
    written, though a fourfold rotation has a diagonal member in their space.
    What the two carry together is not fixed by the hoppings, and the
    lowest-angular-momentum rule reads the copies copy by copy: ``A1`` on each
    row, the site holding two copies of it, and not the doublet ``E`` the pair
    could carry together.

    ``P2/m``'s ``2m`` orbit carries ``p_x`` and ``p_z`` at each site, both ``A'``
    of the site's mirror.  With one site's pair turned inside itself, the
    twofold carries each orbital onto a combination of the two at the next
    site: the mirror still leaves them apart, and the merge the twofold asks
    for is made because the two are copies of one irrep -- one multiplet per
    site, each of its rows ``A'`` with the characters of ``A'``, and each site
    holding two copies.  Which orbital answers to which across the orbit is
    what the operations carry, never the order the sites list them in:
    ``(p_x, p_y)`` at one site and ``(p_y, p_x)`` at the other are each orbital
    its own multiplet, ``A'`` and ``A''`` on their own rows.  And a basis
    function's phase is a gauge: one site's ``p_x`` written as ``i`` times the
    image of the other's puts ``±i`` on the twofold and the inversion that
    exchange them, and time reversal carries ``-1`` on it against its partner;
    the group, the multiplets and the rows are the real basis's.

    ``Im-3m``'s ``8c`` (an ``s`` at ``2a`` beside it) carries ``d`` functions in
    each site's own frame, its ``z`` along the site's threefold, which is
    ``A1g`` and two copies of ``Eg`` of its ``-3m`` -- and ``p``, ``A2u + Eu``,
    where the ``p`` along each site's own axis is carried onto plus or minus
    the one at the next site, which the model's own transport says.  Aligned,
    each copy of ``Eg`` is a multiplet; written in a basis turned inside each
    isotypic component, differently at every site, the two copies are one
    multiplet whose four rows each read ``Eg``.  Either way the crystal is
    ``Im-3m1'``, and no operation is refuted.  With an ``8c`` orbital as the
    reference, the lowest-``l`` rule takes the ``d`` shell whole, at the lowest
    single shell of real harmonics that holds all of its site's content --
    ``l = 2``, ``A1g + 2Eg`` -- and not orbital by orbital, which would take a
    ``p`` shell and a second ``p`` pair, ``A2u + 2Eu``; the ``s`` at ``2a`` then
    follows as ``A1g``.

    Which function each orbital is, is a gauge: a Wannier basis is any unitary
    mixture of the functions at a site.  A ``p`` shell at the ``3m`` site of
    ``R3m``, its orbitals turned inside the site by a rotation of no symmetry,
    has no subset of orbitals the site group carries onto itself, so the site's
    own operations bind the whole shell -- two different irreps -- into one
    multiplet.  Its rows are named from its characters alone, as the sum
    ``A1 + E``, and with the spin as ``2x-E1 + -1E2E``: the ``-E1`` of ``p_z``
    and the one inside the pair's ``-1E2E + -E1`` counted together, every term
    named by the tables.  The aligned shell spreads the same irreps over two
    multiplets and carries the same bands; no function is imposed on the turned
    orbitals or the aligned ones, and a ``wannier90`` spinor import of the
    shell, read with ``spin=``, returns the same.  At the centrosymmetric
    ``4/mmm`` site of ``P4/mmm``, listed ``p_z`` first as a ``wannier90``
    projection lists it, the aligned shell's first orbital would read ``s`` on
    its own; its site's content is one ``p`` shell, ``A2u + Eu``, aligned and
    turned alike, and so are the parities of its bands.

    The same shell at both sites of ``6c`` of ``R-3m`` -- a position the
    tables list no site irrep for -- reads the same content, named by the site
    point group itself: its double-valued names are the tables' own at the
    origin of a symmorphic group of ``3m``.  So does a spinful ``s`` pair
    there, ``-E1`` on each row.  Turned or aligned, the pair labels every
    subset of its bands alike: paired with their images at the other site, the
    orbitals read as the ``p`` shell's functions, whether ``p_z`` is measured
    apart from its pair or not, and the lowest two bands read ``-GM8``, odd.
    Every orbit's band representation is on the record: at a position the
    tables name its rows at, their own elementary ones there, and at ``6c`` a
    sum written first at ``3a``, the first position with elementary band
    representations on its closure -- the ``p`` shell's ``-E1`` pair and
    ``-1E2E`` go up to both parities of ``-3m`` there -- which the table prints
    under its rows.

    At the ``3c`` orbit of ``P-43m`` -- three sites of site group ``-42m``, each
    with its ``-4`` axis along its own cube axis -- a spinful ``p`` shell is
    ``-E2`` (the ``p`` along the axis) and ``-E1 + -E2`` (the other two).  One
    site's shell turned and the other two aligned, the operations exchanging
    the sites carry a turned shell onto aligned ones, so every site's shell is
    one multiplet, each row ``2x-E2 + -E1``; the crystal, its site content and
    the labels of every subset of its bands are the aligned crystal's.  With
    each site's axis uncoupled from its pair, the hoppings leave the pairs'
    orbit a one-dimensional character of its own, and the lowest six bands
    read differently under another: the table says their labels rest on it,
    naming the pairs' orbitals and the band labels as what chose, and with site
    1's shell turned, every orbital and the functions the orbitals read as.
    """
    build, kwargs, number, by_site, content, labels, aligned = _BOUND_AT_A_SITE[case]
    # a case that names the orbital the rule reads first runs the private entry,
    # which takes it; every other case is the public call
    realize = tqg._realize if "reference_orbital" in kwargs else tqg.realized_group
    out = realize(build(), **kwargs)
    assert out, str(out)
    assert out.group.number_str == number and out.descent is None
    assert len(out.validated.indices) == out.group.size
    multiplets = out.rep.multiplets
    assert multiplets.by_site == by_site
    # a multiplet's dimension counts its spinors where the spin is declared in pairs
    pair = 1 if out.rep.spin is None else 2
    assert multiplets.dims == tuple(tuple(len(ch) // pair for ch in site) for site in by_site)
    assert multiplets.accidental == ()
    rows = out.rep.characters.orbitals
    for o in rows:
        # every row carries its own irrep's characters, a copy's and not the
        # multiplet's -- the orbital factor's, half the states' where they carry spin
        assert complex(o.generator_character["E"]).real == \
            pytest.approx(o.dim / (2 if out.group.spinor else 1), abs=1e-9)
    if content is not None:
        held = _held(out)
        assert held == content
        turned = [o for o in rows if len(o.multiplet) > o.dim]
        assert all(o.function == "" for o in turned), \
            "nothing was declared for a copy, so no function is returned for it"
    if labels is not None:
        assert out.rep.characters.irreps == labels
    # every orbit's band representation is on the record: its rows' own
    # elementary ones at its own position where the tables name its rows there,
    # and otherwise a sum written elsewhere, printed under the table
    table = out.rep.characters
    assert sorted(a for orbit in table.band_representations for a in orbit.orbitals) == \
        [o.orbital for o in rows]
    for orbit in table.band_representations:
        own = all(rows[a].named_by == "tables" for a in orbit.orbitals)
        assert all(rows[a].named_by == ("tables" if own else "point group")
                   for a in orbit.orbitals)
        assert own == all(position.split("(")[0] == orbit.wyckoff
                          for _m, _name, position in orbit.terms)
        assert (orbit.sentence() in table.summary()) is not own
    if case in _INDUCED:
        assert [tuple((m, name, position.split("(")[0]) for m, name, position in orbit.terms)
                for orbit in table.band_representations] == _INDUCED[case]
    if aligned is not None:
        # orientation is a gauge: the turned shell and the aligned one are one
        # crystal, with one site content and one band content
        assert all(o.function == "" for o in rows), \
            "named from the characters, and no function imposed on a turned orbital"
        twin = realize(aligned(), **kwargs)
        assert twin, str(twin)
        assert twin.group.number_str == number and twin.descent is None
        assert len(twin.rep.multiplets.by_site[0]) == 2, "p_z apart from the (p_x, p_y) pair"
        assert [(o.named_by, o.function) for o in twin.rep.characters.orbitals] == \
            [(o.named_by, "") for o in rows], "nor on an aligned one: nothing was declared"
        assert twin.rep.characters.band_representations == table.band_representations
        assert _held(twin) == held
        # one band content, on all the bands and on every subset of them: a
        # subset's labels rest on the functions the orbitals read as between
        # the sites, the shell's however it is split into multiplets
        nstate = int(out.rep.model.nstate)
        for bands in sorted(set(range(2, nstate, 2)) | {nstate}):
            here = tqw.hsp_table(out.rep, bands=bands)
            there = tqw.hsp_table(twin.rep, bands=bands)
            assert here and there and here.complete == there.complete, bands
            assert here.content == there.content, bands
            if bands == nstate:
                assert here.complete
            if bands == 2 and case in _LOWEST_PAIR_AT_GAMMA:
                assert here.content["GM"] == _LOWEST_PAIR_AT_GAMMA[case]
    # ... and where the hoppings leave a coupled part a one-dimensional character
    # of its own, a subset of the bands that reads differently under another
    # says so, naming the part and what chose: the pairs' orbit aligned, every
    # orbital with a site's shell turned
    for apart, part, key in _LABELS_REST_ON.get(case, ()):
        note = tqw.hsp_table(realize(apart(), **kwargs).rep, bands=6).note
        assert f"leave open on the coupled part holding {part}: another choice" in note \
            and note.endswith(tqw._DECIDED_BY[key]), (part, note)
    # the covariance gate is the larger of the exact floor and the rounding the
    # representation's own arithmetic carries, |G| N eps max|H| on its battery:
    # the floor for the 4mm pair, the P2/m cases and the shells of R3m, R-3m and
    # P-43m, and far above it on the 96 operations of the 8c models, whose
    # measured blocks are carried through the group's words -- and every case
    # closes it
    rep = out.rep
    scale = float(np.abs(np.asarray(rep.model.hamiltonian(
        k_pts=rep._battery, flatten_spin_axis=True))).max())
    gate = tqs.covariance_gate(len(rep.indices), int(rep.model.nstate), scale)
    assert rep._search["gate"] == gate and (gate > GATE) == (len(rep.indices) == 96)
    assert max(rep.residuals.values()) <= gate
    assert rep.product_residual < 1e-13


def _spin_lift_defect(rep):
    r"""``max ‖D(g) (1 ⊗ S(g))† − O(g) ⊗ 1‖`` over a spinful record's unitary operations.

    ``S(g)`` is the operation's own ``sym.spinor_rotation`` and ``O(g)`` half
    the spin trace of ``D(g) (1 ⊗ S(g))†``, in pythtb's ``(orbital, spin)``
    order: zero to rounding where every matrix is an orbital factor times the
    operation's own spin lift.
    """
    syms = {int(s.ind): s for s in rep.group.symmetries}
    worst = 0.0
    for ind, D in rep.matrices.items():
        if syms[int(ind)].time_reversal:
            continue
        n = np.asarray(D).shape[0] // 2
        S = np.asarray(syms[int(ind)].spinor_rotation, dtype=complex)
        W = np.einsum("asbu,tu->asbt", np.asarray(D).reshape(n, 2, n, 2), S.conj())
        O = 0.5 * np.einsum("asbs->ab", W)
        worst = max(worst, float(np.abs(W - np.einsum("ab,st->asbt", O, np.eye(2))).max()))
    return worst


@pytest.mark.slow
def test_two_decoupled_sublattices_are_measured_and_the_convention_names_the_rest():
    """A commutant that exchanges two sites: a verdict, never a crash.

    Two ``p_x, p_y`` sites in one cell with no hopping between them have a
    commutant that contains the exchange of the two, so a solution space holds
    members that carry one site's orbitals onto the other's -- which no
    ``O (x) S`` is, whatever the partition, and which the layout therefore
    measures nothing from: the question is asked inside the site blocks the
    operation's own permutation allows, and the answer there is one
    two-dimensional channel per site, exactly as for the single site.

    What no measurement can then settle is the *relative* character of the two
    sublattices.  Given the same species, the cell's point set repeats under the
    translation between the two sites and the Hamiltonian obeys it, so the model
    is a supercell of its own primitive cell and no group is served on the cell
    as handed in.  Declared as two species, the translation is not an operation
    of the cell at all, and P1c serves the model rather than refusing
    ``rep_not_unique``: a channel class the hoppings leave free takes the
    trivial relative character wherever it sits, on this site or the next, so
    both sublattices read ``E`` and what is decoupled is said on the record --
    ``conserved`` counts the seven matrices that commute with ``H`` at every k
    beyond the identity.  Nothing couples a pair's ``p_x`` to its ``p_y``
    either, so each pair is carried as faithfully by ``A1 + B1`` as by the
    doublet, and ``E`` is the lowest-``l`` rule's choice, which every row says.

    A spin the Hamiltonian conserves on a site of two orbitals is the same
    record from the spin's side.  The spinful ``(p_x, p_y)`` honeycomb conserves
    every spin rotation without spin-orbit coupling, and ``S_z`` with an on-site
    ``λ L_z S_z``, so each operation's solution space holds its matrix times
    every rotation conserved.  The matrix returned is the one whose spin factor
    is the operation's own lift, ``D(g) = O(g) ⊗ S(g)`` with ``O(g)`` the
    orbital factor the hoppings fix -- as Kane-Mele's one orbital a site
    carries the tables' lift -- and the rotations are said on the record, three
    and one: ``P6mm1'`` on a substrate and ``P6/mmm1'`` free-standing, the two
    readings agreeing, each pair ``-1E2E + -E1`` at ``3m`` and ``-E2 + -E3`` at
    ``-6m2``, and the spin never chosen.  With each pair written in its own
    turned basis, what a change of basis cannot move stays -- the group, the
    rows, the band representation -- and the band labels read alike wherever
    no label rests on the labelling returned: the two pairs are turned by more
    than a right angle relative to one another, past which the orbitals as
    listed read as the other labelling's letter, and the table says where.
    """
    model = _square_pxpy(sublattices=2)
    same = tqg.realized_group(model, typat=[0, 0, 0, 0])
    assert not same and same.gate == "supercell"
    assert "obeys the pure translation" in same.reason.lower()

    split = tqg.realized_group(model, typat=[0, 0, 1, 1], environment="substrate",
                               cross_check=False)
    assert split, str(split)
    assert split.group.number_str == "99.164" and split.descent is None
    assert len(split.validated.indices) == split.group.size == 16
    assert split.rep.multiplets.by_site == (((0, 1),), ((2, 3),))
    assert split.rep.characters.irreps == ("E", "E", "E", "E")
    assert all(o.determined_by == "lowest-l" for o in split.rep.characters.orbitals)
    assert split.conserved["dim"] == 7, "the decoupling, stated rather than refused"
    assert "additional commuting matrices: 7" in repr(split)
    assert max(split.rep.residuals.values()) < 1e-13
    assert split.rep.product_residual < 1e-13

    # a conserved spin on a site of two orbitals: the operation's own lift
    for soc, rotations in ((None, 3), ("LzSz", 1)):
        for environment, number, rows, how in (
                ("substrate", "183.186", "-1E2E + -E1", "hamiltonian"),
                ("free-standing", "191.234", "-E2 + -E3", "lowest-l")):
            read = []
            for turn_a, turn_b in ((0.0, 0.0), (0.7, -1.3)):
                spun = tqg.realized_group(
                    tqm.pxpy_honeycomb_spinful(soc, turn_a=turn_a, turn_b=turn_b),
                    environment=environment)
                assert spun, str(spun)
                assert spun.group.number_str == number
                assert len(spun.validated.indices) == spun.group.size
                assert spun.cross_check["outcome"] == "agree"
                assert spun.conserved["dim"] == rotations
                assert [(s["copies"], s["dim"]) for s in spun.conserved["sectors"]] == \
                    [(1, 4)], "a conserved spin rotation is no sector: the orbitals' one"
                assert f"additional commuting matrices: {rotations}" in repr(spun)
                assert _spin_lift_defect(spun.rep) < 1e-12, "D(g) = O(g) (x) S(g)"
                table = spun.rep.characters
                assert table.irreps == (rows,) * 4 and table.multiplicity_spaces == ()
                assert all(o.determined_by == how for o in table.orbitals)
                read.append(([b.spelled() for b in table.band_representations],
                             {n: tqw.hsp_table(spun.rep, bands=n) for n in (2, 4, 6)}))
            (terms, bands), (terms_turned, bands_turned) = read
            assert terms_turned == terms
            for n in (2, 4, 6):
                assert bands_turned[n].content == bands[n].content or all(
                    "rest on a one-dimensional character the hoppings leave open" in t.note
                    for t in (bands[n], bands_turned[n])), (soc, environment, n)
            if soc and environment == "substrate":
                assert not bands[6].note and bands[6].content == {
                    "GM": {"-GM7": 1, "-GM8": 1, "-GM9": 1},
                    "K": {"-K4": 1, "-K5": 1, "-K6": 2}, "M": {"-M5": 3}}


@pytest.mark.slow
def test_a_stated_layout_that_is_not_a_partition_is_a_grammar_error(monkeypatch):
    """A channel is a set of orbitals at ONE site, and every orbital is in one.

    Everything that can go wrong in a *stated* layout is grammar -- an orbital
    named twice or not at all, a channel spread over two sites, an empty
    channel, a list of channels where one list per site goes -- so it raises
    rather than refusing, and before anything is measured: none of it is a
    measurement about the Hamiltonian, and ``realized_group`` reads it before
    any group is identified, the sites off the centres alone.  Which site index
    holds which channel is not the declaration's business: the channels are
    read as a flat set and re-keyed onto the assignment's own sites, which is
    what carries one layout to another presentation of a model.
    """
    model = _square_pxpy(sublattices=2)
    group = tqg._group_of(model, typat=[0, 0, 1, 1])
    assert group, str(group)
    assignment = tqg._assign_orbitals(model, group, typat=[0, 0, 1, 1])
    assert assignment.site_orbitals == ((0, 1), (2, 3))
    validated = tqg._validate_group(model, group)

    malformed = (([[(0, 1)]], "partitions every orbital"),
                 ([[(0,)], [(1, 2, 3)]], "spreads over sites"),
                 ([[(0, 1)], [(2, 2)]], "partitions every orbital"),
                 ([[(0, 1)], [(2, 7)]], "out of range"),
                 ([[(0, 1)], [()], [(2, 3)]], "no empty multiplet"),
                 ([(0, 1), (2, 3)], "one list per site, [[(0, 1)], [(2, 3)]] states"),
                 ("nonsense", "given the str 'nonsense'"),
                 ([[(0, 1)], [(2, 3.0)]], "by the float 3.0"),
                 ([[(0, 1)], [3]], "holds the int 3 where a multiplet goes"))
    for bad, says in malformed:
        with pytest.raises(ValueError, match=re.escape(says)):
            tqs._build_rep(model, group, assignment, validated=validated, multiplets=bad)

    def measured(*args, **kwargs):
        raise AssertionError("the stated layout's grammar is read before any group")

    monkeypatch.setattr(tqg, "_group_of", measured)
    for bad, says in malformed:
        with pytest.raises(ValueError, match=re.escape(says)):
            tqg.realized_group(model, typat=[0, 0, 1, 1], multiplets=bad)
