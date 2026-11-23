"""Phase A4: ``tqc.ebr`` — EBR decompositions, honest "none"s, and the counting.

Every asserted number is commented with where it comes from: a paper, an
``irreptables`` row, or a measurement made here.

The verdicts this file exists to lock
-------------------------------------
* **lin3g P6 decomposes** — measured in-plane as ``-E2^G(2)@1b + -E^G(4)@2c``.
  The raw ``groups.decompose_ebrs`` refuses the same content
  (``incomplete_content``), because the ``A``/``R``/``Z`` rows of the slab
  group are unreachable and a vector that says nothing about them is a vector
  of a different band set.  That contrast is asserted below: it is the whole
  reason the in-plane restriction is load-bearing rather than cosmetic.
* **lin3g P2 is exactly the ``-E2^G(2)@1b`` row** — eigenvalue-compatible with
  an obstructed atomic limit at ``1b`` while being fragile: a decomposition is
  necessary for an atomic limit and never sufficient.
* **The hexagonal fragile golden's conduction pair is a genuine "none"** — the
  first *physical* none in the fixture set, CP-SAT infeasible over a bound that
  is *proved* exhaustive, and it comes with the signed difference
  ``[-E1^G(4)@2b] - [-E1^G(2)@1a]`` that is what "symmetry-indicated fragile"
  means.

Kane-Mele's two regimes decomposing at different Wyckoff positions, and
``P4_232``'s four bands as the ``2a`` row and its two branches admitting no
EBR sum, are what ``examples/tqc_kane_mele_realized_group.py`` and
``examples/tqc_multifold_connectivity.py`` print, and their goldens pin them.
"""

import numpy as np
import pytest
from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc import symmetry as tqs
from pythtqc.refusal import MagneticScopeWarning, Refusal
from pythtqc import _build
from pythtqc import _layers
from pythtqc import _measure
from pythtqc import _smith
from pythtqc import _spacegroup
from pythtqc import _srsitables
from pythtqc import _tables

import tqc_models as tqm
from conftest import FKM_BOND, fkm_regime, grey_numbers, km_z2_regime, patch_everywhere

# lin3g band blocks, from LIN3G.band_sets
P4, P2, P6, Q2 = (0, 1, 2, 3), (4, 5), 6, (6, 7)


def _labels(d):
    """Each candidate sum of a decomposition, as it prints."""
    return [s.label for s in d.candidate_sums]


def _positions(d):
    """Each candidate sum's Wyckoff positions, one per elementary band representation, sorted."""
    return tuple(tuple(sorted(t.wyckoff for t in s for _ in range(int(t.multiplicity))))
                 for s in d.candidate_sums)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def km_odd_realized(realized):
    """Kane-Mele's Z2-odd #156 regime in the group it realizes: P3m11' (#156.50)."""
    out = realized(km_z2_regime("odd"), typat=[0, 1], cross_check=False,
                   characters={0: "A1"})
    assert out, str(out)
    return out


@pytest.fixture(scope="module")
def pwv_realized(realized):
    """The hexagonal fragile golden in its realized group P6mm1' (#183.186)."""
    out = realized(tqm.pwv_fragile, cross_check=False, characters={0: "A1"})
    assert out, str(out)
    return out


@pytest.fixture(scope="module")
def km191_realized(realized):
    """Kane-Mele at ``delta = rashba = 0`` in its **in-plane** group P6mm1' (#183.186).

    ``environment="substrate"`` is the reading asked for, because the two tests
    this fixture serves start from the in-plane group: one reads its indicator
    group, the other hands its content to the decoupled stack.  A substrate
    makes one side of the layer different, so the ``sigma_h`` fibers are
    collapsed by definition and the 24 in-plane operations are what is served --
    the ``#183.186`` those two statements are made in.  That is also a
    ``dim_r = 2`` model's **default**, and it is named here anyway.  The other
    environment is measured beside it, and the orbital is declared ``s``, a
    word both environments read, so the two are compared: the free-standing
    layer's group is ``P6/mmm1' #191.234``, 48 of 48 -- the baseline's
    ``layer/kane_mele|free-standing`` record -- and the indicator names it
    beside the substrate's class without measuring it again.  (``A1``, the
    substrate's own word, is not a label of the free-standing ``-6m2`` site
    group, and a declaration in it is not compared at all.)

    Built with the **default** search space.  This fixture used to carry
    ``orientation_branch=+1``, because covariance and the product table are
    both blind to the ``g -> g^-1`` branch here; the internal-factor gate added
    with A3b's wave pins it by measurement instead (the losing branch carries
    ``S(C3)^dag`` where the group says ``S(C3)``), so the restriction is no
    longer needed.  ``test_symmetry.py`` owns the regression that both branches
    survive the other two gates, on the detector's own solution spaces; here
    each operation's space is read in its own form, whose members carry
    ``S(C3)`` alone, so the losing branch is not built at all, and what is
    asserted is only that removing the restriction did not change this
    fixture's answer -- the search closes on exactly one representation.
    """
    out = realized(tqm.kane_mele_inversion, environment="substrate",
                   characters={0: "s"})
    assert out, str(out)
    assert out.rep._search["n_survivors"] == 1
    return out


def _row(group, wyckoff, irrep=None):
    """The EBR row at ``wyckoff`` (of ``irrep``), from irreptables."""
    rows = tqe.ebr_rows(group)
    assert rows, str(rows)
    keys = [k for k in rows if k[1] == wyckoff and (irrep is None or k[0] == irrep)]
    assert len(keys) == 1, f"{wyckoff}/{irrep} matched {keys}"
    return keys[0], rows[keys[0]]


def _in_plane(group, row):
    points = set(tqg.high_symmetry_points(group))
    return {lab: v for lab, v in row.items() if tqe.kpoint_of_label(lab) in points}


# ---------------------------------------------------------------------------
# 1.  the in-plane restriction: verified per group, not assumed
# ---------------------------------------------------------------------------

def test_the_in_plane_restriction_is_verified_row_by_row_in_each_group(
        lin3g_declared_cross_checked, km_odd_realized):
    """SG 99 and SG 156: every dropped row is a label-mapped copy of a retained one.

    Measured (2026-08-19), and the reason the restricted decomposition is
    reported as *exact* rather than merely necessary: ``-A6 = -M6``,
    ``-A7 = -M7``, ``-R5 = -X5``, ``-Z6 = -GM6``, ``-Z7 = -GM7`` as vectors over
    the five EBRs.  The rank is unchanged by the restriction, which is the same
    statement in linear-algebra form.  ``Restriction.exact`` *is* ``orphans ==
    ()`` (``exact = not missing``, ``orphans = tuple(missing)``), so only one of
    the two is asserted here and below.

    SG 156 is the same verification on a hexagonal group with paired coreps:
    ``-A4A5 = -GM4GM5``, ``-A6 = -GM6``, ``-H4 = -K4``, ``-H5 = -K5``,
    ``-H6 = -K6``, ``-L3L4 = -M3M4`` (measured).  Two groups, two lattices: the
    verification is a per-group measurement, never a rule about slabs.
    """
    group = lin3g_declared_cross_checked.group
    r = tqe.in_plane_restriction(group)
    assert r, str(r)
    assert r.dropped and r.orphans == ()
    assert r.kpoints == ("GM", "M", "X")
    assert r.retained == ("-GM6", "-GM7", "-M6", "-M7", "-X5")
    assert r.dropped == ("-A6", "-A7", "-R5", "-Z6", "-Z7")
    assert r.copies == {"-A6": ("-M6",), "-A7": ("-M7",), "-R5": ("-X5",),
                        "-Z6": ("-GM6",), "-Z7": ("-GM7",)}
    assert r._rank == r._full_rank == 3      # 5 EBRs, 2 relations (2c = 1a pair = 1b pair)

    r = tqe.in_plane_restriction(km_odd_realized.group)
    assert r, str(r)
    assert r.dropped and r.orphans == ()
    assert r.kpoints == ("GM", "K", "M")
    assert r.copies == {"-A4A5": ("-GM4GM5",), "-A6": ("-GM6",), "-H4": ("-K4",),
                        "-H5": ("-K5",), "-H6": ("-K6",), "-L3L4": ("-M3M4",)}
    assert r._rank == r._full_rank == 4


# ---------------------------------------------------------------------------
# 2.  lin3g -- the phase closers
# ---------------------------------------------------------------------------

def test_lin3g_P6_decomposes_and_the_raw_passthrough_refuses(lin3g_declared_cross_checked):
    """The in-plane restriction, against the vector it restricts.

    The occupied six bands decompose in the in-plane basis as
    ``-E^G(4)@2c + -E2^G(2)@1b``.  The *same* measured counts handed to
    ``groups.decompose_ebrs`` are refused (``incomplete_content``): the EBR
    basis of the slab space group is written over ``A``, ``R`` and ``Z`` as
    well, a monolayer's zone has no ``k`` for any of them, and a vector that
    says nothing about them is a vector of a different band set.  The refusal
    names all six points and the three that are missing, which is what the
    restricted route above exists to supply.
    """
    record = lin3g_declared_cross_checked
    model, group, rep = record.model, record.group, record.rep
    table = tqw.hsp_table(record, bands=P6)
    assert table.complete, table.refusals
    # measured content, phase A3 (docs/guide/atomic-limits.md)
    assert table.symmetry_data == {"-GM6": 2, "-GM7": 1, "-M6": 1, "-M7": 2, "-X5": 3}

    d = tqe.decompose(table)
    assert d.decomposable and d.candidate_sums
    # a success rests on the solutions it exhibits, so there is nothing to
    # certify: the field is None rather than a word-shaped empty string, and the
    # report carries no certificate line at all
    assert d.certificate is None
    assert "(.certificate" not in d.summary()
    # the measured solution, in the basis irreptables stores
    assert "-E2 at 1b (4mm1') + -E at 2c (mm21')" in _labels(d)
    # ... and in the MAXIMAL 1a/1b basis, using 2c = 1a-pair = 1b-pair
    assert "-E1 at 1a (4mm1') + -E2 at 1a (4mm1') + -E2 at 1b (4mm1')" in _labels(d)
    assert len(d.candidate_sums) == 3
    assert sorted(_positions(d)) == sorted([("1b", "2c"), ("1b", "1b", "1b"),
                                            ("1a", "1a", "1b")])
    # a cap equal to the number of sums lists them all, and says so
    for cap, listed, cut in ((3, 3, False), (2, 2, True)):
        e = tqe.decompose(table, cap=cap).numerical.enumeration
        assert (e.n_solutions, e.truncated) == (listed, cut), cap
    # a cap below 1 asks for nothing, and is an error in the call
    with pytest.raises(ValueError, match=r"^decompose\(\): cap=0; a cap is a positive "
                                         "whole number"):
        tqe.decompose(table, cap=0)

    # the contrast: the thin passthrough, on the same numbers
    raw = tqg.decompose_ebrs(group, table.symmetry_data)
    assert isinstance(raw, Refusal) and not raw
    assert raw.gate == "incomplete_content"
    assert raw.context["missing_points"] == ("A", "R", "Z")
    assert set(raw.context["points"]) == {"GM", "M", "X", "A", "R", "Z"}


def test_lin3g_P2_is_exactly_the_1b_row_directly_and_as_a_difference(
        lin3g_declared_cross_checked):
    """P2 = [P6] - [P4], and both routes give the single ``-E2^G(2)@1b`` row.

    The statement, verified against ``irreptables``' own SG-99 spinful
    data: P2's vector *is* that row exactly, so P2 is eigenvalue-compatible with
    an obstructed atomic limit at ``1b`` — and it is nevertheless fragile, which
    HSP content cannot see: a decomposition is necessary for an atomic limit
    and never sufficient, and the record holds no Wilson-loop slot that would
    close it.

    ``P4``, the block subtracted, is itself the in-plane ``-E^G(4)@2c`` row of
    the tables, read band by band off the sewing matrices.  And the difference
    taken the other way round, ``[P4] - [P6]``, is not a band set, and saying so
    is the answer.
    """
    record = lin3g_declared_cross_checked
    model, group, rep = record.model, record.group, record.rep
    t6 = tqw.hsp_table(record, bands=P6)
    t4 = tqw.hsp_table(record, bands=list(P4))
    t2 = tqw.hsp_table(record, bands=list(P2))
    assert t2.symmetry_data == {"-GM6": 1, "-M7": 1, "-X5": 1}       # measured, phase A3
    assert t4.symmetry_data == _in_plane(group, _row(group, "2c")[1]), "P4 is the 2c row"

    diff = tqe.subtract(t6, t4)
    assert diff == t2.symmetry_data, "P2 = [P6] - [P4]"
    bad = tqe.subtract(t4, t6)
    assert isinstance(bad, Refusal) and bad.gate == "negative_multiplicity"
    assert list(bad.context["negative"].items()) == [("-GM6", -1), ("-M7", -1), ("-X5", -1)]
    # a content is listed point by point, GM first, and by number at each point
    assert list(tqe.subtract({"-X5": 1, "-GM10": 1, "-GM2": 2, "-GM11GM9": 1},
                             {"-GM2": 1})) == ["-GM2", "-GM10", "-GM11GM9", "-X5"]

    key, row = _row(group, "1b", "-E2")
    for d in (tqe.decompose(t2), tqe.decompose(diff, record)):
        assert d.decomposable and len(d.candidate_sums) == 1
        assert [tuple(s) for s in d.candidate_sums] == [(tqg.Term(1, "-E2", "1b", "4mm1'", "4mm"),)]
        assert _positions(d) == (("1b",),)
    # ... and the slot a Wilson-loop measurement would fill is not there to fill
    with pytest.raises(TypeError):
        tqe.decompose(t2, winding_certificate={})
    assert {lab: v for lab, v in t2.symmetry_data.items()} == _in_plane(group, row)


def test_lin3g_P2_and_Q2_sum_to_the_1a_pair_but_neither_half_is_one(
        lin3g_declared_cross_checked):
    """The fragile signature in decomposition form.

    P2 + Q2 is the full Wieder quartet: it decomposes as the two ``1a`` EBRs
    (and equivalently as the single ``2c`` row, and as the two ``1b`` rows).
    Neither half alone is a ``1a`` EBR — each is a ``1b`` one, which is the
    measured detail worth being precise about.
    """
    record = lin3g_declared_cross_checked
    model, group, rep = record.model, record.group, record.rep
    p2 = tqw.hsp_table(record, bands=list(P2)).symmetry_data
    q2 = tqw.hsp_table(record, bands=list(Q2)).symmetry_data
    assert q2 == {"-GM7": 1, "-M6": 1, "-X5": 1}              # measured, phase A3
    total = {lab: p2.get(lab, 0) + q2.get(lab, 0) for lab in set(p2) | set(q2)}

    d = tqe.decompose(total, record)
    assert d.decomposable
    assert "-E1 at 1a (4mm1') + -E2 at 1a (4mm1')" in _labels(d)
    assert ("1a", "1a") in _positions(d)

    a_rows = [_in_plane(group, v) for k, v in tqe.ebr_rows(group).items() if k[1] == "1a"]
    assert p2 not in a_rows and q2 not in a_rows, "neither half alone is a 1a EBR"
    for half in (p2, q2):
        got = tqe.decompose(half, record)
        assert _positions(got) == (("1b",),), "each half is a 1b EBR"


def _general_position_bands(ops, lattice, spinful, zeeman=0.0):
    """``s`` orbitals on the orbit of a generic point under ``ops``: the realized record and its band count."""
    from pythtqc import models as tqcm

    orbit = tqcm.orbit_of(ops, (0.137, 0.271, 0.413))
    n = len(orbit)
    seeds = ([(0, j, (0, 0, 0), -0.3 - 0.05 * j) for j in range(1, n)]
             + [(0, 0, L, -0.2) for L in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
             + [(0, j, (1, 0, 0), -0.07 - 0.01 * j) for j in range(1, n)])
    m = tqcm.atomic_model(lattice, ops, [orbit], [0.0] * n, seeds, spinful=spinful)
    for i in range(n if zeeman else 0):
        m.set_onsite([0.0, 0.0, 0.0, zeeman], ind_i=i, mode="add")
    return tqg.realized_group(m, typat=[0] * n), n * (2 if spinful else 1)


def test_lin3g_regular_completion_to_sixteen_bands(lin3g_declared_cross_checked):
    """One regular copy is the bands of orbitals at the general position; lin3g's is sixteen bands.

    Orbitals on the orbit of a generic point of ``I4_1`` carry a whole number
    of copies of what ``regular_content`` returns, read by ``hsp_table`` and not
    through its formula: one copy spinless in ``I4_11'``, one copy of eight
    bands spinful, the Kramers pair at each of the four sites, and two copies
    of four spinful in ``I4_1`` with a Zeeman field.  Completing all their
    bands leaves nothing missing.

    In ``P4mm1'``, ``|U| = 8``, a spinful copy is ``2|U| = 16`` bands, and its
    content is ``{-GM6: 4, -GM7: 4, -M6: 4, -M7: 4, -X5: 8}``.  lin3g's six
    lowest bands complete to it.

    And the anchor ``A = 3E1bar@1a + 2E2bar@1a + Ebar@2c``, measured: the
    padding that carries a *single* ``E2bar^G(2)@1a`` up to 16 bands, one
    regular copy of ``P4mm1'``.  It is quoted in both bases: as written above
    (with ``2c``) and, since the ``2c`` row's content is a pair's at ``1a`` or
    at ``1b``, in the ``1a``/``1b`` basis as ``4E1bar@1a + 3E2bar@1a``.
    """
    from pythtqc import models as tqcm

    for spinful, zeeman, name, copies in ((False, 0.0, "I4_11'", 1), (True, 0.0, "I4_11'", 1),
                                          (True, 0.4, "I4_1", 2)):
        if zeeman:
            with pytest.warns(MagneticScopeWarning):
                r, n = _general_position_bands(tqcm.I41_PRIM_OPS, tqcm.I41_LATTICE,
                                               spinful, zeeman)
        else:
            r, n = _general_position_bands(tqcm.I41_PRIM_OPS, tqcm.I41_LATTICE, spinful)
        assert r and r.group.name == name and len(r.group._u_symmetries) == 4, str(r)
        table = tqw.hsp_table(r, bands=n)
        assert table.complete
        one = tqe.regular_content(r.group)
        assert table.symmetry_data == {k: copies * v for k, v in one.items()}, \
            "the orbit's bands, read, are whole copies"
        rc = tqe.regular_completion(table, bands=n)
        assert rc and (rc.copies, rc.bands_per_copy) == (copies, n // copies), str(rc)
        assert rc.regular == one and rc.padding == {}

    record = lin3g_declared_cross_checked
    model, group, rep = record.model, record.group, record.rep
    reg = tqe.regular_content(group)
    assert reg == {"-GM6": 4, "-GM7": 4, "-M6": 4, "-M7": 4, "-X5": 8}

    rc = tqe.regular_completion(tqw.hsp_table(record, bands=P6), bands=16)
    assert rc, str(rc)
    assert (rc.copies, rc.bands_per_copy) == (1, 16)
    assert rc.regular == reg
    assert rc.padding == {"-GM6": 2, "-GM7": 3, "-M6": 3, "-M7": 2, "-X5": 5}

    _, e2_1a = _row(group, "1a", "-E2")
    seed = _in_plane(group, e2_1a)
    assert seed == {"-GM6": 1, "-M6": 1, "-X5": 1}

    rc = tqe.regular_completion(seed, record, bands=16)
    assert rc, str(rc)
    assert (rc.copies, rc.bands_per_copy) == (1, 16)
    assert {k: rc.copies * v for k, v in rc.regular.items()} == {
        "-GM6": 4, "-GM7": 4, "-M6": 4, "-M7": 4, "-X5": 8}
    assert rc.padding == {"-GM6": 3, "-GM7": 4, "-M6": 3, "-M7": 4, "-X5": 7}
    got = _labels(rc.decomposition)
    assert "3x-E1 at 1a (4mm1') + 2x-E2 at 1a (4mm1') + -E at 2c (mm21')" in got, \
        "the anchor A, as written"
    assert "4x-E1 at 1a (4mm1') + 3x-E2 at 1a (4mm1')" in got, \
        "the same A in the maximal basis"


# ---------------------------------------------------------------------------
# 4.  Kane-Mele #156: the deep atomic limit decides the rows' convention
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_kane_mele_deep_atomic_limit_lands_on_the_occupied_site():
    """The physical control on the EBR rows' convention.

    Drive the staggered potential to ``delta = 50`` with everything else at the
    fixture's values and the valence pair becomes an atomic limit in the most
    literal sense: **99.96 % of its weight sits on one orbital**, the one the
    assignment puts at ITA ``1b``.  Whatever else a decomposition may or may not
    certify, a band set localized to that accuracy on that site must decompose
    *there* — so this measurement, and not a table, is what fixes the
    convention: rows read in the other one would put the EBR at ``1c``, a site
    carrying 0.04 % of the weight, and fail the assertion.

    The measured content ``{-GM6, -K4, -K5, -M3M4}`` is the trivial regime's,
    whose letter is therefore ``1b``.
    """
    from pythtb.models import kane_mele

    model = kane_mele(50.0, -1.0, 0.3, 0.25)          # _KM_ARGS["even"] with delta -> 50
    _, vec = model.solve_ham(k_pts=model.k_uniform_mesh([12, 12]),
                             return_eigvecs=True, flatten_spin_axis=False)
    weight = (np.abs(vec[:, :2]) ** 2).sum(axis=(0, 1, -1))
    weight = weight / weight.sum()
    assert weight[1] > 0.999, f"the deep limit is not localized enough: {weight}"

    out = tqg.realized_group(model, typat=[0, 1], cross_check=False,
                             characters={0: "A1"})
    assert out, str(out)
    group, assignment, rep = out.group, out._assignment, out.rep
    assert group.number_str == "156.50"
    # orbital 1 -- the one carrying the weight -- is the 1b site
    assert assignment.wyckoff_spglib[assignment.orbit_of_orbital[1]] == "b"

    table = tqw.hsp_table(out, bands=2)
    assert table.complete and table.symmetry_data == {"-GM6": 1, "-K4": 1, "-K5": 1,
                                               "-M3M4": 1}
    d = tqe.decompose(table)
    assert d.decomposable and len(d.candidate_sums) == 1
    assert list(d.symmetry_data) == ["-GM6", "-K4", "-K5", "-M3M4"]
    assert _positions(d) == (("1b",),), "the weight is at 1b; so is the EBR"


# ---------------------------------------------------------------------------
# 5.  the hexagonal fragile golden -- the first physical "none"
# ---------------------------------------------------------------------------

def test_pwv_valence_is_the_1a_EBR_and_the_conduction_pair_is_a_none(pwv_realized):
    """Po/Watanabe/Vishwanath's fragile model, decided here.

    The paper's own statement — 'the valence bands of our model are trivial,
    whereas the conduction bands feature fragile topology' — comes out of the
    pipeline:

    * valence ``{-GM9, -K6, -M5}`` decomposes **uniquely** as ``-E1^G(2)@1a``,
      the paper's ``(E1bar ^ G)_1a`` at the empty hexagon centre;
    * conduction ``{-GM8, -K4, -K5, -M5}`` admits **no** nonnegative-integer
      decomposition (CP-SAT infeasible over a proved-exhaustive bound), while
      the signed probe finds ``[-E1^G(4)@2b] - [-E1^G(2)@1a]``.  A group of
      bands whose irreps are a *difference* of EBRs is exactly what these papers
      call symmetry-indicated fragile.
    * the four bands together are the single ``-E1^G(4)@2b`` row — the honeycomb
      EBR that splits.

    The bound is proved exhaustive: the EBR matrix is nonnegative and every
    column has a positive entry, so ``M x = y`` forces ``x_j <= max(y) = 1``
    term by term, and ``INFEASIBLE`` at that bound is a proof -- which the
    report prints.  ``P6mm1'`` has no symmetry indicator at all, so this
    obstruction is invisible to indicators by construction, which is what
    fragile means, and the record holds no divisor rather than pretending a
    zero class is an answer.

    Note the orientation: in **this** model it is the conduction pair that is
    fragile, not the valence pair.  See the fixture's provenance note.
    """
    model, group, rep = pwv_realized.model, pwv_realized.group, pwv_realized.rep
    assignment, validated = pwv_realized._assignment, pwv_realized.numerical.validated
    assert group.number_str == "183.186"          # P6mm1', the realized group
    assert assignment.wyckoff_spglib == ("b",)   # the honeycomb sites, Wyckoff 2b
    assert validated.discrepancies == () and len(validated._indices) == 24
    assert len(rep._indices) == 24

    valence = tqw.hsp_table(pwv_realized, bands=2)
    assert valence.complete
    assert valence.symmetry_data == {"-GM9": 1, "-K6": 1, "-M5": 1}
    dv = tqe.decompose(valence)
    assert dv.decomposable and len(dv.candidate_sums) == 1
    assert _positions(dv) == (("1a",),)
    assert [tuple(s) for s in dv.candidate_sums] == [(tqg.Term(1, "-E1", "1a", "6mm1'", "6mm"),)]

    conduction = tqw.hsp_table(pwv_realized, bands=(2, 3))
    assert conduction.complete
    assert conduction.symmetry_data == {"-GM8": 1, "-K4": 1, "-K5": 1, "-M5": 1}
    dc = tqe.decompose(conduction)
    assert not dc.decomposable and dc.candidate_sums == ()
    assert dc.numerical.bound == 1 and dc.numerical.bound_is_rigorous
    assert dc.certificate == "proof"
    assert ("proved: no elementary band representation can occur more than once"
            in dc.summary()), "and a vector no sum carries prints it"
    assert dc.numerical.restriction.dropped and dc.numerical.restriction.exact
    assert dc.numerical.completeness.claim == "exact"
    signed = {tuple(sorted((int(t.multiplicity), t.irrep, t.wyckoff) for t in s))
              for s in dc.signed_combinations.sums}
    assert ((-1, "-E1", "1a"), (1, "-E1", "2b")) in signed, "the fragile difference [2b] - [1a]"
    si = tqe.symmetry_indicator(conduction)
    assert si and si.divisors == () and si.indicator_group == "trivial"

    four = tqw.hsp_table(pwv_realized, bands=4)
    d4 = tqe.decompose(four)
    assert d4.decomposable and len(d4.candidate_sums) == 1
    assert [tuple(s) for s in d4.candidate_sums] == [(tqg.Term(1, "-E1", "2b", "3m1'", "3m"),)]


# ---------------------------------------------------------------------------
# 6.  refusals: the gates that keep a wrong answer from being confident
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_deliberately_dropped_point_is_refused_and_the_override_downgrades_the_claim(
        fkm_undistorted):
    """The ``allow_incomplete`` override mechanism itself, on a synthetic hole.

    No fixture in the suite has a *table*-caused refusal any more, so the
    override is exercised by removing a point's measurement by hand: Fu-Kane-
    Mele's four-band table on the undistorted diamond, ``s`` declared, with
    ``W`` moved from ``records`` to ``refusals``.

    The whole table is the reference: a 3D crystal samples every tabulated
    point, so its in-plane restriction drops nothing, and the four bands
    decompose with the claim **exact**.

    Without the override ``decompose`` refuses (``incomplete_content``) — a
    symmetry data vector missing a constraint decomposes too easily.  With it, the
    ``W`` **rows** are dropped along with the measurement (keeping the equation
    while dropping the measurement would force those multiplicities to zero,
    which over-constrains rather than relaxes), and because ``-W3W4``/``-W5W6``
    are not copies of any retained row the claim is downgraded to a necessary
    condition — the same answer, held less strongly.  (``-W7`` is dropped too
    but *is* a copy of a retained row, so it is not an orphan; that is the
    distinction ``Restriction.orphans`` exists to make.)
    """
    import dataclasses

    group, rep = fkm_undistorted.group, fkm_undistorted.rep
    assert group.number_str == "227.129"
    full = tqw.hsp_table(fkm_undistorted, bands=4)
    assert full.complete and "W" in full.multiplets
    restriction = tqe.in_plane_restriction(group)
    assert restriction and restriction.dropped == ()
    assert tqe.decompose(full).numerical.completeness.claim == "exact"

    dropped = Refusal("integer_multiplicity", "synthetic: W removed by hand",
                      value=0.5, threshold=1e-3, context={"kpname": "W"})
    holed = dataclasses.replace(
        full,
        multiplets={k: v for k, v in full.multiplets.items() if k != "W"},
        _records={k: v for k, v in full._records.items() if k != "W"},
        refusals={"W": dropped},
        symmetry_data={lab: v for lab, v in full.symmetry_data.items()
                       if not lab.startswith("-W")},
        complete=False)
    assert holed.symmetry_data == {"-GM6": 1, "-GM9": 1, "-L8": 1, "-L9": 1, "-X5": 1}

    gated = tqe.decompose(holed)
    assert isinstance(gated, Refusal) and gated.gate == "incomplete_content"
    assert gated.context["refusals"] == {"W": "integer_multiplicity"}
    for out in (tqe.subtract(full, holed), tqe.regular_completion(holed, bands=48)):
        assert isinstance(out, Refusal) and out.gate == "incomplete_content"
        assert out.context["refusals"] == {"W": "integer_multiplicity"}

    got = tqe.decompose(holed, allow_incomplete=True)
    restricted = got.numerical.restriction
    assert got.numerical.completeness.claim == "necessary-condition"
    assert set(restricted.orphans) == {"-W3W4", "-W5W6"}
    assert "-W7" in restricted.dropped and restricted.copies["-W7"], (
        "dropped, but a copy of a retained row, so not an independent constraint")
    assert not restricted.exact and not restricted._require_exact
    assert "W" not in restricted.kpoints
    assert "-E1 at 8a (-43m1')" in _labels(got)
    assert len(got.candidate_sums) == 2, (
        "and this is what the missing constraint costs: without W the 8b row "
        "becomes admissible too, so the answer stops being unique -- exactly "
        "the sense in which a necessary condition is weaker than an exact one")
    assert "-E1 at 8b (-43m1')" in _labels(got)


def test_a_label_the_group_does_not_carry_is_refused(lin3g_declared_cross_checked):
    """``create_symmetry_vector`` would drop it silently; this does not."""
    out = tqe.decompose({"-GM6": 1, "-NOSUCH3": 2}, lin3g_declared_cross_checked)
    assert isinstance(out, Refusal) and out.gate == "unknown_label"
    assert out.context["unknown"] == ("-NOSUCH3",)
    assert out.context["group"] is lin3g_declared_cross_checked.group


def test_regular_completion_refuses_a_band_count_that_is_not_a_regular_multiple(
        lin3g_declared_cross_checked):
    """One regular copy is induced from the general position; other counts are refused.

    One copy of ``P4mm1'``, spinful, is 16 bands, so 6 and the model's own 8
    are refused.  On a triclinic ``s`` orbital, ``|U| = 2``: one copy of the
    spinful ``P-11'`` is 4 bands, the Kramers pair at the general position, and
    its corep ``-GM2GM2``, one irrep taken twice, occurs once; the spinless
    ``P-11'`` and the spinful ``P-1`` with time reversal broken by a Zeeman
    field hold one state there, 2 bands.  ``regular_content`` agrees with the
    completion in each.
    """
    record = lin3g_declared_cross_checked
    model, group, rep = record.model, record.group, record.rep
    for n in (6, 8):
        out = tqe.regular_completion(tqw.hsp_table(record, bands=P6), bands=n)
        assert isinstance(out, Refusal) and out.gate == "not_a_regular_multiple"
        assert out.context["bands_per_copy"] == 16

    # ... and a band set that is not contained in the regular target propagates
    # the difference refusal rather than inventing negative padding
    too_much = {lab: 9 for lab in ("-GM6", "-GM7", "-M6", "-M7", "-X5")}
    out2 = tqe.regular_completion(too_much, record, bands=16)
    assert isinstance(out2, Refusal) and out2.gate == "negative_multiplicity"
    assert "Pass a larger multiple of 16 to bands=." in out2.reason
    assert set(out2.context) == {"negative", "a", "b"}
    assert out2.value == min(out2.context["negative"].values())
    typo = tqe.regular_completion({"-GM6": 1, "-NOSUCH3": 1}, record, bands=16)
    assert typo.gate == "negative_multiplicity" and "none of ['-NOSUCH3']" in typo.reason

    from pythtb import Lattice, TBModel

    def triclinic(spinful, zeeman=0.0):
        lat = Lattice([[1.0, 0.0, 0.0], [0.23, 1.17, 0.0], [0.31, 0.19, 1.41]],
                      [[0, 0, 0]], periodic_dirs=[0, 1, 2])
        m = TBModel(lat, spinful=spinful)
        m.set_onsite([[0.3, 0.0, 0.0, zeeman]] if zeeman else [0.3])
        for t, R in [(-1.0, [1, 0, 0]), (-0.7, [0, 1, 0]), (-0.45, [0, 0, 1]),
                     (-0.2, [1, 1, 0]), (-0.13, [0, 1, 1]), (-0.08, [1, 0, 1])]:
            m.set_hop(t, 0, 0, R)
        return m

    with pytest.warns(MagneticScopeWarning):
        zeeman = tqg.realized_group(triclinic(True, 0.2))
    for r, name, low, per_copy, other in (
            (tqg.realized_group(triclinic(True)), "P-11'", 2, 4, "-AuAu at 1a (-11')"),
            (tqg.realized_group(triclinic(False)), "P-11'", 1, 2, "Au at 1a (-11')"),
            (zeeman, "P-1", 1, 2, "-Au at 1a (-1)")):
        assert r and r.group.name == name and len(r.group._u_symmetries) == 2, str(r)
        table = tqw.hsp_table(r, bands=low)
        out = tqe.regular_completion(table, bands=per_copy // 2)
        assert isinstance(out, Refusal) and out.gate == "not_a_regular_multiple"
        assert out.context["bands_per_copy"] == per_copy
        rc = tqe.regular_completion(table, bands=per_copy)
        assert rc and (rc.copies, rc.bands_per_copy) == (1, per_copy), str(rc)
        assert rc.regular == tqe.regular_content(r.group)
        assert set(rc.regular.values()) == {1}, "every corep once, at every point"
        assert len(rc.padding) == len(table.kpoints) and set(rc.padding.values()) == {1}
        assert [s.label for s in rc.decomposition.candidate_sums] == [other], \
            "the missing bands are the other parity's orbital at the same site"


def test_content_of_takes_a_table_or_a_mapping_and_refuses_nonsense(
        lin3g_declared_cross_checked):
    """One content plumbing, two input shapes."""
    record = lin3g_declared_cross_checked
    model, group, rep = record.model, record.group, record.rep
    table = tqw.hsp_table(record, bands=P6)
    assert tqe.content_of(table) == tqe.content_of(table.symmetry_data) == table.symmetry_data
    assert tqe.content_of({"-GM6": 2, "-GM7": 0}) == {"-GM6": 2}
    assert tqe.content_of({"-GM6": -1}).gate == "negative_multiplicity"
    assert tqe.content_of({"-GM6": 0.5}).gate == "non_integer_multiplicity"
    with pytest.raises(TypeError):
        tqe.content_of(["-GM6"])


def test_kpoint_of_label_parses_every_label_of_every_fixture_group(
        lin3g_declared_cross_checked, km_odd_realized, pwv_realized):
    """The label -> k-point rule is structural, and it is checked against the tables.

    ``'-GM4GM5' -> 'GM'``, ``'GM2-' -> 'GM'``, ``'-M3M4' -> 'M'``, ``'X1' -> 'X'``.
    Every EBR-basis label of every fixture group must parse to a k-point the
    group's own tables carry.
    """
    assert tqe.kpoint_of_label("-GM4GM5") == "GM"
    assert tqe.kpoint_of_label("GM2-") == "GM"
    assert tqe.kpoint_of_label("-M3M4") == "M"
    assert tqe.kpoint_of_label("X1") == "X"

    groups = [lin3g_declared_cross_checked.group, km_odd_realized.group,
              pwv_realized.group]
    for group in groups:
        data = tqg.ebr_data(group)
        assert data, str(data)
        in_plane = set(tqg.high_symmetry_points(group))
        names = {tqe.kpoint_of_label(lab) for lab in data.corep_names}
        assert names >= in_plane, (group.number_str, names, in_plane)


# ---------------------------------------------------------------------------
# 7.  symmetry indicators: the Smith-form class in {BS}/{AI}
# ---------------------------------------------------------------------------
#
# Scope, once, for everything below: this package validates magnetic types I
# and II only -- the 230 space groups as unitary groups, and the 230 grey
# (type-II) groups ``include_TR=True`` produces.  For types III and IV
# ``irreptables`` ships the EBRs and the unitary part's characters at the
# maximal k-points but not the corep type at a k-point, and no named formula
# exists for a spinless one; ``symmetry_indicator`` refuses on them, and that
# refusal is asserted below.


@pytest.fixture(scope="module")
def fkm_strong(realized):
    """Fu-Kane-Mele's strong distorted phase in ``R-3m1' #166.98``.

    Declared with the bond-centre species (``FKM_BOND``), which names the group
    outright from the cell.  ``examples/tqc_fkm_diamond.py`` reads the same
    phase without the species, by descent from ``Fd-3m1' #227.129``.
    """
    out = realized(fkm_regime("strong"), decorations=FKM_BOND, characters={0: "s"})
    assert out, str(out)
    assert out.group.number_str == "166.98"
    return out


def test_the_shipped_smith_basis_is_the_convention_and_a_de_novo_snf_verifies_it(
        fkm_strong, monkeypatch):
    """Pinned, then checked: an independent integer Smith form of the same matrix.

    ``irreptables`` ships ``u``, ``v``, ``R`` with each group's EBR block and
    ``groups.ebr_data`` transports them with the rows; that transported ``u`` is
    the convention this package's indicator is written in.  Every call
    recomputes the Smith form from the matrix alone, in exact integers, and
    checks (i) that the shipped factors really satisfy ``u M v = R`` on the
    transported matrix, (ii) that the elementary divisors agree, and (iii) that
    the order of the class -- which no basis choice can move -- agrees.  What
    the two bases are *allowed* to disagree about is the individual components,
    that freedom being an automorphism of the indicator group; the change of
    basis is therefore recorded rather than assumed away, and what is asserted
    below is that the record *is* such an automorphism and carries the one
    reading onto the other -- never which of the many valid Smith bases this
    package's ``_snf`` elimination lands in, which is its pivot rule's business
    and no statement about the indicator.  (Observed: the identity.)

    Checks (i)-(iii) are **not** the record's ``agrees`` /
    ``shipped_is_smith`` / ``denovo_is_smith`` keys, which ``_verify_smith``
    writes as unconditional literals: a disagreement raises
    :class:`~..refusal.InternalError` instead of landing in the dict, so
    those keys cannot be False once the call has returned.  What is asserted
    here is the de-novo work itself (``divisors_denovo``) and, below, the raise
    -- provoked on a doctored Smith factor that is arranged to pass check (i)
    and fail check (ii), so it is the independent Smith computation and not the
    ``u M v = R`` identity that catches it.
    """
    group = fkm_strong.group
    si = tqe.symmetry_indicator(tqw.hsp_table(fkm_strong, bands=2))
    v = si.numerical.verification
    assert v.divisors_denovo == (2, 4)
    # What the two bases are allowed to disagree about is checked as the
    # freedom it is, not pinned to the basis `_snf`'s pivot rule happens to
    # reach: the recorded change of basis must be an AUTOMORPHISM of the
    # indicator group Z_2 + Z_4 -- a bijection of its eight elements -- and it
    # must carry the shipped basis's reading of this content onto the de-novo
    # one.  Observed here it is the identity (`change_of_basis_is_identity` is
    # True and `values_denovo == si.values`); with `_snf` pivoting on the
    # LARGEST entry instead of the smallest it is ((1, 2), (0, 1)) and both of
    # those readings change while nothing about the class does.
    div, cob = v.divisors_denovo, v.change_of_basis

    def _act(y):
        return tuple(sum(cob[i][j] * y[j] for j in range(len(y))) % div[i]
                     for i in range(len(div)))

    elements = [(a, b) for a in range(div[0]) for b in range(div[1])]
    assert {_act(y) for y in elements} == set(elements), "an automorphism, not a map"
    assert _act(si.values) == v.values_denovo, "the same class, in the other basis"

    # Pm-3m1' with spin-orbit coupling, Z4 x Z8: the two bases differ, and an
    # entry from Z4 into Z8 is fixed modulo 8, the divisor of its row.  The
    # vector whose class is (1, 0) in the shipped basis is read in both.
    data = _tables._ebr_data(_srsitables._TableKey("221.93", True))
    u, sv, d, m = _smith._smith_system(data, "221.93")
    slots = tuple(i for i, x in enumerate(d) if x > 1)
    y = [row[slots[0]] for row in _smith._unimodular_inverse(u)]
    got = _smith._verify_smith("221.93", m, u, sv, d, y, slots, (4, 8), (1, 0), 4)
    div, cob = got["divisors_denovo"], got["change_of_basis"]
    assert div == (4, 8) and not got["change_of_basis_is_identity"]
    assert _act((1, 0)) == got["values_denovo"]

    # the hand-rolled Smith normal form is exact and unimodular, on a matrix
    # whose divisors are not all 1
    matrix = [[int(x) for x in row] for row in np.asarray(tqg.ebr_data(group).matrix)]
    u0, d0, v0 = _smith._snf(matrix)
    assert _smith._matmul(_smith._matmul(u0, matrix), v0) == d0
    assert abs(round(np.linalg.det(np.array(u0, dtype=float)))) == 1
    assert abs(round(np.linalg.det(np.array(v0, dtype=float)))) == 1
    assert sorted(d0[i][i] for i in range(len(d0)) if d0[i][i] > 1) == [2, 4]

    # ... and the check that carries (ii) is the one that can fail: double row 0
    # of the shipped `u` AND the matching entry of `R`.  That keeps `u M v = R`
    # exactly -- check (i) passes -- and keeps every factor integer, but the
    # shipped divisors are now (2, 2, 4) where the matrix's own are (2, 4), so
    # the de-novo Smith form is what refutes it.  A convention the tables fix
    # leaves no gauge for a caller to supply, so this raises rather than
    # refusing.
    from pythtqc import InternalError

    real = _tables._ebr_data(group)
    u_bad, r_bad = np.array(real["smith"][0], dtype=int), np.array(real["smith"][1],
                                                                  dtype=int)
    u_bad[0] *= 2
    r_bad[0, 0] *= 2
    doctored = dict(real, smith=(u_bad, r_bad, real["smith"][2]))
    assert (_smith._matmul(_smith._matmul([[int(x) for x in row] for row in u_bad], matrix),
                        [[int(x) for x in row]
                         for row in np.asarray(real["smith"][2], dtype=int)])
            == [[int(x) for x in row] for row in r_bad]), "check (i) still passes"
    patch_everywhere(monkeypatch, "_ebr_data", lambda g: doctored)
    with pytest.raises(InternalError) as excinfo:
        tqe.symmetry_indicator(tqw.hsp_table(fkm_strong, bands=2))
    assert "elementary divisors" in str(excinfo.value)


@pytest.mark.data
def test_the_indicator_sweep_over_every_group_this_package_validates():
    """A5.1's closing regressions, swept over the whole validated universe.

    The universe is the 230 space groups as unitary groups and the 230 grey
    (type-II) groups, both valuednesses: 920 systems.  Three statements are
    checked on every one of them, from the tables alone.

    **Every EBR indicates zero.**  An EBR *is* an atomic limit, so its column
    must sit at the identity of ``{BS}/{AI}``.  A column that did not would mean
    the shipped Smith factors and the shipped EBR matrix are not the same
    object.

    **An independent Smith computation reproduces the shipped divisors.**  The
    elementary divisors are a property of the matrix, so a hand-rolled exact
    integer Smith normal form of it has to agree with what ``irreptables``
    ships -- for the transported matrix as well as the raw one.

    **The transport moves no indicator value.**  The indicator GROUP is
    transport-invariant because a row permutation is a Smith equivalence.  The
    VALUE could in principle shift by an automorphism of that group -- ``y' =
    u.y`` is a linear form on labelled coordinates and the transport relabels
    them.  It does not, and the reason is structural rather than lucky:
    ``groups.ebr_data`` rewrites ``u`` by exactly the permutation ``P`` that
    relabels the content vector, so ``u' y_phys = (u P)(P y_BCS) = u y_BCS``
    identically.  Checked on every basis vector and four seeded probes per
    system, in the 379 systems that have something to shift.
    """
    from irreptables.ebrs import load_ebr_data

    rng = np.random.default_rng(20260828)
    systems = nontrivial = permuted = columns = 0
    for number_str, magnetic in _in_scope_keys():
        for spinor in (False, True):
            raw = load_ebr_data(number_str, spinor)
            if any("wyckoff_position" not in e for e in raw["ebrs"]):
                continue           # ten such files, every one type III
            systems += 1
            labels, _conj, _partner, _moved, _new, perm = _tables._ebr_transport(
                number_str, spinor, magnetic)
            r = np.asarray(raw["smith_form"]["r"], dtype=int)
            d = ([int(x) if i == 0 else 0 for i, x in enumerate(r)] if r.ndim == 1 else
                 [int(r[i, i]) for i in range(min(r.shape))])
            u_bcs = np.asarray(raw["smith_form"]["u"], dtype=int)
            matrix = np.rint(np.array([e["vector"] for e in raw["ebrs"]],
                                      dtype=float)).astype(int).T

            # (i) every EBR column maps to the identity of the quotient
            on_ebrs = u_bcs @ matrix
            for col in range(matrix.shape[1]):
                columns += 1
                for i, di in enumerate(d):
                    v = int(on_ebrs[i, col])
                    assert (v == 0 if di == 0 else v % di == 0), (
                        number_str, spinor, col, i, v, di)

            # (ii) a de-novo Smith form agrees, raw and transported
            divisors = sorted(x for x in d if x > 1)
            phys = _tables._to_physical_convention(raw, number_str, spinor, magnetic)
            transported = np.rint(np.array([e["vector"] for e in phys["ebrs"]],
                                           dtype=float)).astype(int).T
            for m in (matrix, transported):
                _u0, d0, _v0 = _smith._snf([[int(x) for x in row] for row in m])
                got = sorted(d0[i][i] for i in range(min(len(d0), len(d0[0])))
                             if d0[i][i] > 1)
                assert got == divisors, (number_str, spinor, got, divisors)

            if not divisors:
                continue
            nontrivial += 1
            if list(perm) != list(range(len(labels))):
                permuted += 1

            # (iii) the raw-basis and transported-basis values are the same
            u_phys = np.asarray(phys["smith_form"]["u"], dtype=int)
            n = len(labels)
            probes = [np.eye(n, dtype=int)[j] for j in range(n)]
            probes += [rng.integers(0, 4, size=n) for _ in range(4)]
            for b_phys in probes:
                b_bcs = np.array([b_phys[perm[j]] for j in range(n)], dtype=int)
                here, there = u_phys @ b_phys, u_bcs @ b_bcs
                for i, di in enumerate(d):
                    if di == 0:
                        assert here[i] == there[i], (number_str, spinor, i)
                    elif di > 1:
                        assert (here[i] - there[i]) % di == 0, (number_str, spinor, i)
    # measured 2026-08-28: 920 systems in scope, 379 with a nontrivial indicator
    # group, and in 193 of those the transport really does permute the rows --
    # so the agreement above is not the vacuous statement that nothing moved
    assert (systems, nontrivial) == (920, 379), (systems, nontrivial)
    assert permuted == 193, permuted
    assert columns == 10398, columns          # every EBR of every group in scope


def _in_scope_keys():
    """``(number_str, magnetic)`` for every group this package validates.

    The 230 space groups as unitary groups (bare ``number_str``), and the 230
    grey (type-II) groups.  A grey group is identified by the operation list --
    it contains time reversal itself, up to a lattice vector -- and not by
    parsing its name: ``conftest.grey_numbers()`` reads that off the operations
    each correpresentation table lists.
    """
    import os

    from irreptables import ebrs as _ebrs

    grey = set(grey_numbers().values())

    root = os.path.join(os.path.dirname(_ebrs.__file__), "data", "ebrs")
    keys = sorted(f[: -len("_ebrs.json")] for f in os.listdir(root))
    return ([(k, False) for k in keys if "." not in k]
            + [(k, True) for k in keys if k in grey])


def test_kane_mele_z2_is_not_eigenvalue_indicated_in_its_realized_group(km191_realized):
    """Kane-Mele at ``delta = rashba = 0``: the indicator group is *trivial* there.

    ``delta = rashba = 0`` makes the two sublattices equivalent, and the
    wallpaper collapse of the cell keeps ``C2z`` -- a **rotation**, whose action
    on in-plane ``k`` is the one inversion would have, which is why the two are
    easy to confuse and not a reason to conflate them.  The group the
    Hamiltonian realizes in that reading is ``P6mm1' #183.186``, 24 of 24,
    which contains no inversion at all and whose EBR matrix has no elementary
    divisor above 1.  So the QSH ``Z2`` of this model is not an eigenvalue
    indicator in *this* group -- the same verdict as the ``#156`` pair, reached
    from the tables rather than assumed.  Genuine inversion needs ``sigma_h``,
    and the layer-group tests below are where it is measured.

    Asked of the symmetry data vector written out by hand, this is the whole
    answer: the in-plane group is the group the measurement returned, and every
    row out of its plane repeats one in it.  Asked of the *model* instead, the
    free-standing reading finds the mirror in the point set and answers
    ``P6/mmm1' #191.234`` directly; that is a different question, and the
    baseline's ``layer/kane_mele|free-standing`` record pins it.

    The out-of-plane rows are recovered rather than dropped: each of
    ``A``/``H``/``L`` duplicates an in-plane row of the EBR matrix, so its
    multiplicity is determined by the ones the model measured.  That is the
    in-plane restriction's own exactness argument, read in the other direction,
    and the record names every row it filled and where from.  A full EBR row of
    the group, handed in with its out-of-plane entries, indicates zero -- an EBR
    is an atomic limit -- and the entries it supplied are read as they are,
    never filled over.
    """
    group = km191_realized.group
    assert group.number_str == "183.186"

    si = tqe.symmetry_indicator(dict(tqw.hsp_table(km191_realized, bands=2).symmetry_data),
                                km191_realized)
    assert si, str(si)
    assert si.indicator_group == "trivial"
    assert si.divisors == () and si.values == () and si.order == 1
    assert si.indicators is None
    assert si.numerical.completion is None, "nothing was re-presented, and the record says so"

    assert si.numerical.reconstructed == {"-A7": "-GM7", "-A8": "-GM8", "-A9": "-GM9",
                                          "-H4": "-K4", "-H5": "-K4", "-H6": "-K6",
                                          "-L5": "-M5"}
    assert si.numerical.restriction.dropped and si.numerical.restriction.exact
    # the measured multiplicities the reconstruction above copied from (the
    # equalities themselves are the map already pinned in `reconstructed`)
    filled = si.numerical.completed_symmetry_data
    assert filled["-GM9"] == 1 and filled["-M5"] == 1

    rows = tqe.ebr_rows(group)
    row = rows[sorted(rows)[0]]
    supplied = set(row) & set(si.numerical.restriction.dropped)
    assert supplied, "the row carries its out-of-plane entries"
    whole = tqe.symmetry_indicator(row, km191_realized)
    assert whole and not any(whole.values) and whole.order == 1
    assert not supplied & set(whole.numerical.reconstructed), "a supplied value is left alone"


def test_a_layers_vector_written_in_its_plane_is_completed_on_its_own_band_representations(
        realized):
    """A layer's symmetry data vector written in its plane is completed, and gets a class.

    ``P6/mmm1'`` is where the Kane-Mele ``Z2`` *could* be indicated --
    ``X_BS = Z6 x Z12``, and ``z4``, the Fu-Kane parity index, is one of its
    named forms.  That group's ``A``/``H``/``L`` rows are not duplicates of
    in-plane ones over every band representation, which is why the restriction
    of the whole basis refuses.  Over the layer's own band representations,
    those of the Wyckoff positions within it, every row out of the plane
    repeats one in it, so a vector written in the plane is completed there and
    its class read on the whole basis.  Every elementary band representation
    is an atomic limit, so each row, written in the plane or written whole,
    has class zero; and a whole row of a position off the layer, read as
    written, is decomposed over the layer's own band representations and
    found not decomposable rather than refused (``-E3`` at ``2d`` excepted,
    which the layer's own ``-E3`` at ``2c`` reproduces).
    """
    free = realized(tqm.kane_mele_inversion, environment="free-standing",
                    characters={0: "s"}, cross_check=False)
    group = free.group
    assert group.number_str == "191.234"
    hsp = tqg.high_symmetry_points(group)
    assert sorted(hsp) == ["GM", "K", "M"]
    rows = tqe.ebr_rows(group)
    assert list(rows["-E1", "2c"]) == [
        "-GM9", "-GM11", "-A9", "-A11", "-H7", "-H8", "-K7", "-K8", "-L5", "-L6", "-M5", "-M6"]
    # the same rows make the restriction of the whole basis refuse
    whole = tqe.in_plane_restriction(group)
    assert whole.gate == "restriction_not_exact"
    # listed by point and, at each, by ascending number, as every content is
    assert "['-A7', '-A8', '-A9', '-A10', '-A11', '-A12', '-H8'" in whole.reason, whole.reason
    assert str(list(whole.context["missing"])) in whole.reason, "one refusal, one order"
    assert whole.context["group"] is group

    assert len(rows) == 22
    for (irrep, wyckoff), row in rows.items():
        in_plane = {lab: m for lab, m in row.items() if tqe.kpoint_of_label(lab) in hsp}
        plane = tqe.symmetry_indicator(in_plane, free)
        assert plane and plane.values == (0, 0), (irrep, wyckoff, plane)
        assert plane.numerical.completion is None and plane.numerical.reconstructed
        assert plane.symmetry_data == in_plane
        whole = tqe.symmetry_indicator(dict(row), free)
        assert whole and whole.values == (0, 0), (irrep, wyckoff, whole)
        if wyckoff in ("1b", "2d", "3g"):
            d = tqe.decompose(dict(row), free)
            assert d and d.numerical.completeness.claim == "exact", (irrep, wyckoff, d)
            assert d.decomposable == ((irrep, wyckoff) == ("-E3", "2d")), (irrep, wyckoff)


# ---------------------------------------------------------------------------
# 7b.  the LAYER group: real inversion at #191.234, and the class it certifies
# ---------------------------------------------------------------------------
#
# The test above is the refusal on in-plane content alone.  This block is what a
# monolayer can say anyway.  There is no inversion in a 2D wallpaper group --
# `C2z` is a rotation that merely shares inversion's action on in-plane `k` --
# but a flat layer whose `sigma_h` is unbroken has genuine inversion
# `P = sigma_h * C2z` as a LAYER-group operation, and #191.234 is the space
# group that carries it.  The presentation used here is the decoupled STACK:
# the same layer repeated along `z` with no interlayer hopping, a genuinely 3D
# crystal with the same `H` on every plane, which can be asked for content at
# `k_z = 1/2`.


def _stacked(model, c=8.0):
    """The monolayer repeated along ``z`` with no interlayer hopping.

    The same crystal ``groups.stack_presentation`` builds, constructed
    independently here: ``H(k)`` gains a third argument it does not depend on.
    Kept as the tests' own hand-rolled version on purpose, so the library's
    presentation is checked against a construction that shares none of its
    code (``examples/tqc_kane_mele_three_presentations.py`` asks the library
    call).
    """
    from pythtb import Lattice, TBModel

    lat = np.zeros((3, 3))
    lat[:2, :2] = np.asarray(model.lat_vecs, dtype=float)
    lat[2] = [0.0, 0.0, float(c)]
    orb = np.zeros((int(model.norb), 3))
    orb[:, :2] = np.asarray(model.orb_vecs, dtype=float)
    out = TBModel(Lattice(lat, orb.tolist(), periodic_dirs=[0, 1, 2]),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    for h in model.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    list(h.get("lattice_vector", [0, 0])) + [0], mode="add")
    return out


#: The layer-group anchor, measured 2026-08-28.  ``X_BS(#191.234) = Z6 x Z12``;
#: the Kane-Mele valence pair at ``delta = rashba = 0``, six-point content, sits
#: at ``(1, 0)`` of order 6, with ``z2w,3 = 1`` and ``z4 = 2``: every component
#: an invariant of the layer's decoupled stack, fixed by the layer's own content.
#: ``KM_LAYER_INDICATORS`` in the paper's names, ``KM_LAYER_NAMED`` in irreptables'.
KM_LAYER_CLASS = (1, 0)
KM_LAYER_INDICATORS = {"z2w,3": 1, "z4": 2, "z3m,pi": 1, "z3m,0": 1,
                       "z6m,0": 5, "z6m,pi": 5, "z12": 10}
KM_LAYER_NAMED = {"z2w,3": 1, "z4": 2, "z3m,π": 1, "z3m,0": 1,
                  "z6m,0": 5, "z6m,π": 5, "z12": 10}


@pytest.mark.slow
def test_the_layer_class_moves_only_by_the_declaration_it_states():
    """The declared functions move the labels they can -- and invert the class.

    The site character is a **declaration**: ``s`` (= ``A1'``, even under
    ``sigma_h``) and ``A2'`` (odd under it, even under the twofold axes — that
    is how this position's own site-symmetry listing spells that row) are both
    labellings of these orbitals, and the hoppings cannot choose between them.
    The lowest-``l`` functions that carry ``A2'`` are
    ``z x (x^2 - 3y^2)``: odd under ``sigma_h`` and ``C2z``, even under
    inversion.  Declaring it moves the labels at ``GM``, ``A``, ``H`` and
    ``K`` and sends the class to its inverse in ``Z6 x Z12``; at ``M`` and
    ``L``, whose two double-valued coreps only inversion tells apart, the
    labels stay.  What does not move is everything of order at most 2 -- the
    order itself, ``z2w,3`` and ``z4`` -- which is why the physics statement
    ("a nonzero class of order 6, ``z2w,3 = 1``") survives the declaration
    and the individual components need it stated.
    """
    stack = tqg.realized_group(_stacked(tqm.kane_mele_inversion()),
                               characters={0: "s"})
    assert stack and stack.group.number_str == "191.234", str(stack)
    even = tqe.symmetry_indicator(tqw.hsp_table(stack, bands=2))
    assert even.site_content is not None
    assert even.site_content.determined_by[0] == "declaration", \
        "the even labelling is the one this record states"

    odd_rep = _build._build_rep(stack.model, stack.group, stack._assignment,
                             validated=stack.numerical.validated, characters={0: "A2'"})
    assert odd_rep, str(odd_rep)
    odd = tqe.symmetry_indicator(tqw._hsp_table_rep(odd_rep, bands=2))
    assert odd, str(odd)
    assert set(odd.symmetry_data) & set(even.symmetry_data) == {"-M6", "-L6"}, \
        "at M and L only inversion tells the coreps apart, and the functions are even"
    assert odd.values == (5, 0), "the inverse class in Z6 x Z12"
    assert all((a + b) % d == 0 for a, b, d in zip(odd.values, even.values,
                                                   even.divisors))
    assert odd.order == even.order == 6
    assert odd.indicators["z2w,3"] == even.indicators["z2w,3"] == 1
    assert odd.indicators["z4"] == even.indicators["z4"] == 2
    assert any(odd.values) and any(even.values)


# ---------------------------------------------------------------------------
# 7c.  the same physics from a `lat.shape == (2, 2)` model, with no user-side
#      stack: the indicator builds one itself and says so
# ---------------------------------------------------------------------------
#
# The block above hands the six-point content in, measured on a stack the test
# built.  These two do not: they hand in a layer and its representation, and
# the indicator measures the rows out of the layer's plane on the decoupled
# stack of that same layer -- `groups.stack_presentation`, built internally at
# the one step whose arithmetic needs a three-dimensional EBR basis.  Two
# presentations of one physics; the class must not know the difference.


def _declared_slab(model, c=30.0):
    """The layer re-declared by hand as a 3D slab: ``dim_r = 3``, ``z`` not periodic.

    Deliberately hand-built rather than taken from ``groups._slab_presentation``
    -- this is the presentation a physicist *declares*, and the point of the
    test is that declaring it changes no number.
    """
    from pythtb import Lattice, TBModel

    lat = np.zeros((3, 3))
    lat[:2, :2] = np.asarray(model.lat_vecs, dtype=float)
    lat[2] = [0.0, 0.0, float(c)]
    orb = np.zeros((int(model.norb), 3))
    orb[:, :2] = np.asarray(model.orb_vecs, dtype=float)
    out = TBModel(Lattice(lat, orb.tolist(), periodic_dirs=[0, 1]),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    for h in model.hoppings:
        out.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                    list(h.get("lattice_vector", [0, 0])) + [0], mode="add")
    return out


@pytest.mark.slow
def test_a_pure_2D_layer_reaches_the_layer_class_by_measuring_its_own_stack(
        km191_realized, realized):
    """``lat.shape == (2, 2)`` in, the layer-group class out, asked free-standing.

    The journey with no user-side trick in it, taken from the **pure-2D**
    reading: a ``dim_r = 2`` model is read on a substrate by default, which
    presses it onto its own plane (``H(k)`` bit-identical) and collapses the
    ``sigma_h`` fibers by definition, so ``realized_group`` answers in the
    one-sided group ``P6mm1' #183.186``, 24 of 24, and ``hsp_table`` serves
    ``GM``/``K``/``M`` in that group's own language.  That collapse is not a
    device but a physical claim: a substrate makes one side of the layer
    different, and the operations with ``R[2,2] = -1`` are then absent rather
    than merely unrefutable.  (Declared **free-standing**, the same model keeps
    the fibers and measures them, and reads ``P6/mmm1' #191.234``, 48 of 48.)

    **Which question asks it.**  The layer-group class is a statement about a
    layer with vacuum on both sides, the free-standing question, and a
    substrate record does not ask it: the stack obeys exactly what the layer
    obeys, every row of ``#183.186`` out of the plane repeats an in-plane one,
    and so no stack is built.  Measured below, the substrate record's own class
    is ``X_BS(#183.186)``'s trivial one, the honest one for a layer lying on
    something, and the free-standing group it names beside it is the one
    ``realized_group`` measured in the other environment, never built again.

    The free-standing record asks it.  Ten of the eleven ``A``/``H``/``L`` rows
    of ``#191.234`` repeat no in-plane row, so the indicator measures them on
    the layer's decoupled stack: repeated along ``z`` with no interlayer
    hopping, a genuine 3D crystal with the same ``H`` on every plane, read in
    the layer's own group with the layer's own matrices.  The record says what
    tied the stack to the layer -- its Hamiltonian at every ``k_z``, and each
    fiber at ``k_z = ½`` against its partner in the plane -- and the class that
    comes back is the layer-group class, ``(1, 0)`` of order 6 in
    ``Z6 x Z12``, ``z2w,3 = 1``, ``z4 = 2``, with the 2D reading beside it,
    because the model is still one layer.

    In the paper's names it is ``z3m,pi`` where ``irreptables`` writes
    ``z3m,π``, and the spelling map carries the one onto the other exactly.  ``Z6 x Z12`` is the largest indicator group in the tables, 72
    classes, so it is where the determinism of those names has the most to do.
    """
    assert np.asarray(tqm.kane_mele_inversion().lat_vecs).shape == (2, 2), \
        "a genuinely 2D model"

    layer = km191_realized
    assert layer.environment.kind == "substrate"
    assert layer.group.number_str == "183.186" and layer.group._dim_k == 2
    assert len(layer.numerical.validated._indices) == 24 == layer.group._size
    content = tqw.hsp_table(layer, bands=2)
    assert content.complete and sorted(content.kpoints) == ["GM", "K", "M"]
    assert dict(content.symmetry_data) == {"-GM9": 1, "-K6": 1, "-M5": 1}, "#183's own language"

    # on a substrate that is the whole answer: no stack is built, and the
    # free-standing layer's group is the one realized_group measured beside it,
    # named and not used, since a sigma_h-based indicator does not apply here
    free = tqg.NamedGroup("P6/mmm1'", "191.234", 48)
    assert layer.environment.cross_check.outcome == "agree"
    assert layer.environment.cross_check.free_standing == free
    on_substrate = tqe.symmetry_indicator(content)
    assert on_substrate, str(on_substrate)
    assert on_substrate.group.number_str == "183.186" and on_substrate.values == ()
    assert on_substrate.numerical.completion is None, \
        "every row out of the plane repeats one in it"
    assert on_substrate.numerical.free_standing_group == free
    assert "is P6/mmm1' #191.234, larger than this one" in on_substrate.numerical.summary()

    # the free-standing question, asked of the free-standing record: the
    # layer's own group #191.234, its rows at k_z = 1/2 measured on the stack
    free = realized(tqm.kane_mele_inversion, environment="free-standing",
                    characters={0: "s"}, cross_check=False)
    assert free, str(free)
    assert free.group.number_str == "191.234" and len(free.numerical.validated._indices) == 48
    free_content = tqw.hsp_table(free, bands=2)
    assert dict(free_content.symmetry_data) == {"-GM9": 1, "-K8": 1, "-M6": 1}
    si = tqe.symmetry_indicator(free_content)
    assert si, str(si)
    assert si.group.number_str == "191.234" and si.indicator_group == "Z6 x Z12"
    assert si.values == KM_LAYER_CLASS and si.order == 6
    assert si.indicators == KM_LAYER_INDICATORS
    # irreptables' own named forms read the same class, and the spelling map
    # carries the paper's names onto theirs
    data = _tables._ebr_data(free.group)
    named, _touched, _note, _coeffs = tqe._named_forms(
        free.group, data, np.asarray(data["matrix"], dtype=int), list(si._y), "II")
    assert named == KM_LAYER_NAMED
    assert {_srsitables.to_irreptables_name(k): v for k, v in si.indicators.items()} == \
        named, "the spelling map applied to real data"
    tex = {f.name: f.name_tex for f in tqr.srsi_tables(191, spinor=True)._indicator_rows}
    assert tex["z3m,pi"] == "z_{3m,\\pi}"
    assert "all 72 elements of Z6xZ12 have different canonical values" in \
        si.numerical.indicators_note
    assert tqr.canonical_determinism(tqr.srsi_tables(191, spinor=True))["n_classes"] == 72

    comp = si.numerical.completion
    assert comp.outcome == "completed" and comp.stopped_at is None
    assert comp.environment == "free-standing"
    assert comp.stack.dim_k == 3 and comp.stack.periodic_dirs == (0, 1, 2)
    assert comp.stack_period == tqg.STACK_C
    gates = comp.gates
    assert gates.hamiltonian_identity.residual == 0.0
    assert gates.hamiltonian_identity.k_z == (0.0, 0.25, 0.5)
    # the coreps at A are those at GM, H = K, L = M -- checked, not assumed
    fibers = {(f.kpoint, f.in_plane_kpoint) for f in gates.fibers}
    assert fibers == {("A", "GM"), ("H", "K"), ("L", "M")}
    for f in gates.fibers:
        assert f.partition_equal and f.multiplicities_equal
        assert f.energy_defect == 0.0 and f.character_defect < 1e-12

    # the 2D reading is returned beside the class: the model is one layer, and
    # the points are split by whether their k lies in the layer's plane
    td = si.numerical.two_dimensional
    assert td and td.dim_k == 2 and td.periodic_dirs == (0, 1)
    assert td.vacuum_dirs == (2,) and td.out_of_plane_kpoints == ("A", "H", "L")
    assert td.presentation == "decoupled-STACK"
    assert td.support["z2w,3"] == "out-of-plane only"
    assert td.support["z4"] == "mixed"
    # the class read as the decoupled stack's, every component alike
    for phrase in ("the class of the layer's decoupled stack",
                   "no hopping between copies",
                   "every component is an invariant of that stack"):
        assert phrase in td.note and phrase in si.numerical.summary(), phrase
    assert not {"weak", "strong"} & set(td.note.lower().replace(",", " ").split())
    assert "decoupled stack" in si.numerical.summary()
    assert "#191.234" in repr(si) and "completed on the stack" in repr(si)


@pytest.mark.slow
def test_the_declared_slab_reaches_the_same_class_through_the_layer_route(monkeypatch):
    """The same layer *declared* as a slab: same cell group, same class, same numbers.

    ``dim_r = 3`` with a non-periodic third vector and every site at ``z = 0``
    is what a physicist writes when the model is a monolayer in a vacuum
    supercell.  ``_group_of`` reads ``P6/mmm1' #191.234`` straight off that cell
    -- no decoration removes the mirror -- and a ``dim_r = 3`` slab's own
    default is the free-standing reading, which represents the
    ``sigma_h``-composed operations rather than leaving them undetermined; the
    covariance gate then decides them.

    That presentation has ``dim_k = 2``, so ten of the eleven ``A``/``H``/``L``
    rows are orphans and the class is not reachable from the plane.  The stack
    supplies them, read in the layer's *own* group: the stack obeys exactly what
    the layer obeys, so its group is the one handed in, carried onto the
    stack's cell by each operation's Cartesian action, and its matrices are the
    layer's -- nothing is identified or validated again, and the group the
    stack's cell allows is never asked.  What ties the stack to the layer is
    measured, and each is an identity: the layer's matrices on the stack's
    ``H(k, k_z)`` to rounding, the labels of the in-plane content exactly, the
    44 characters to machine precision (measured 0.0 at the shipped ``STACK_C``
    and 1.5e-15 when that arbitrary ``c`` is moved, the two presentations being
    different arithmetic rather than one computation) -- and the class is the
    one the pure-2D journey reaches.  A failed identity is the package
    disagreeing with itself, and raises.  A decomposition needs no stack: over
    the layer's own band representations, the positions off the layer left
    out, it finds no atomic limit, with a proof, as the class says.
    """
    hand = _declared_slab(tqm.kane_mele_inversion())
    assert hand.lattice.dim_r == 3 and tuple(hand.periodic_dirs) == (0, 1)
    assert np.abs(np.asarray(hand.orb_vecs)[:, 2]).max() == 0.0

    from_cell = _spacegroup._group_of(hand)
    assert from_cell.number_str == "191.234", "the cell says it before anything is built"
    assert from_cell._dim_k == 2 and from_cell._vacuum_dirs == (2,)

    slab = tqg.realized_group(hand, cross_check=False, characters={0: "s"})
    assert slab, str(slab)
    assert slab.group.number_str == "191.234" and len(slab.numerical.validated._indices) == 48
    assert len(slab.rep._indices) == 48

    content = tqw.hsp_table(slab, bands=2)
    assert content.complete and dict(content.symmetry_data) == {"-GM9": 1, "-K8": 1, "-M6": 1}
    assert tqe.in_plane_restriction(slab.group, require_exact=False).orphans, \
        "10 orphan rows"

    si = tqe.symmetry_indicator(content)
    assert si, str(si)
    assert si.group.number_str == "191.234" and si.values == KM_LAYER_CLASS
    assert si.order == 6 and si.indicators == KM_LAYER_INDICATORS
    comp = si.numerical.completion
    assert comp.outcome == "completed"
    assert comp.route == "decoupled stack", "which stack the rows came from"
    assert comp.environment == "free-standing"
    # the record says how the class was reached
    assert "completed on the stack" in repr(si)
    assert "in the layer's own group P6/mmm1' #191.234" in si.numerical.summary()
    assert comp.input.group.number_str == comp.stack.group.number_str == "191.234"
    represented = comp.gates.representation
    assert represented.n_operations == 48
    assert represented.covariance <= represented.threshold < 1e-12
    assert comp.gates.characters.labels_equal is True
    # a c=30 slab against a c=STACK_C stack: the same 44 characters computed in
    # two presentations, so roundoff and not an identity
    assert comp.gates.characters.worst < 1e-12
    assert comp.gates.characters.n_comparisons == 44
    assert len(comp.input.orphans) == 10
    assert si.numerical.two_dimensional.out_of_plane_kpoints == ("A", "H", "L")
    # ... and a decomposition is read in the plane, over the layer's own band
    # representations: it finds no atomic limit, with a proof, as the class says
    d = tqe.decompose(content)
    assert (d.decomposable, d.certificate) == (False, "proof"), d.summary()
    assert d.signed_combinations.sums == ()
    assert sorted(d.numerical.restriction.outside_layer) == ["1b", "2d", "3g"]
    assert ("1b, 2d and 3g lie off the layer, so their band representations are left "
            "out") in d.numerical.summary()

    # the vector of the plane written out by hand, beside the measurement it came
    # from, is completed on the layer's own band representations: the same class
    # and the same completed vector as the stack's, with nothing stacked
    vector = tqe.symmetry_indicator(dict(content.symmetry_data), slab)
    assert vector, str(vector)
    for name in ("values", "divisors", "order", "indicators", "symmetry_data"):
        assert getattr(vector, name) == getattr(si, name), name
    assert (vector.numerical.completed_symmetry_data
            == si.numerical.completed_symmetry_data)
    assert vector.numerical.completion is None and si.numerical.completion is not None
    assert {"-A9": "-GM9", "-H8": "-K8", "-L6": "-M6"}.items() <= \
        vector.numerical.reconstructed.items()
    assert si.numerical.reconstructed == {}
    assert (tqr.srsi_of_bands(dict(content.symmetry_data), slab).theta_Z
            == tqr.srsi_of_bands(content).theta_Z)
    # the vector naming every point is read as written, to the same class
    every = tqe.symmetry_indicator(si.numerical.completed_symmetry_data, slab)
    assert every.values == si.values and not every.numerical.reconstructed

    # ... and when the stack cannot be built for the layer, the rows are still
    # missing: the refusal comes back carrying where it stopped, never a class
    # read off a truncated vector.  A stack whose Hamiltonian is not the layer's
    # is not a verdict at all but the package disagreeing with itself, and it
    # raises.  (Both provoked: a hopping along the normal is no layer's, and the
    # stack's own Hamiltonian is moved off the layer's as it is evaluated.)
    from pythtqc import InternalError
    from pythtqc import _bloch

    with monkeypatch.context() as mp:
        patch_everywhere(mp, "stack_presentation",
                   lambda *a, **k: Refusal("hopping_across_the_stack", "synthetic",
                                           value=1.0))
        stopped = tqe.symmetry_indicator(content)
    evaluator = _bloch.evaluator

    def moved_off(model, *args, **kwargs):
        # the stack is the one model here periodic in three directions
        bloch = evaluator(model, *args, **kwargs)
        if len(model.periodic_dirs) < 3:
            return bloch
        return lambda k: np.asarray(bloch(k)) + 0.5

    with monkeypatch.context() as mp:
        mp.setattr(_bloch, "evaluator", moved_off)
        with pytest.raises(InternalError, match="the decoupled stack is not the layer"):
            tqe.symmetry_indicator(content)
    assert isinstance(stopped, Refusal) and stopped.gate == "indicator_not_evaluable"
    assert stopped.context["group"] is content.group
    assert stopped.context["stack"]["stopped_at"] == "stack_presentation"
    assert stopped.context["stack"]["outcome"] == "stopped"
    # a stack never built has no period and no extent
    assert stopped.context["stack"]["stack_period"] is None
    assert stopped.context["stack"]["stack_extent"] is None
    # what was found, in its own words; where it stopped stays on the record
    assert "stopped: the stack is not built for this layer; " \
        "synthetic" in stopped.reason
    assert "stopped at" not in stopped.reason
    # the little-group representations named in ascending order at each point
    assert ("['-A7', '-A8', '-A9', '-A10', '-A11', '-A12', '-H8', '-H9', '-L5', '-L6'] "
            "of P6/mmm1' #191.234") in stopped.reason
    # a representation with no matrix for one operation, and for three
    rep, full = content._rep, tuple(content._rep._indices)
    for dropped, words in ((1, "operation 48 of"), (3, "operations 46-48 of")):
        object.__setattr__(rep, "_indices", full[:-dropped])
        try:
            short = tqe.symmetry_indicator(content)
        finally:
            object.__setattr__(rep, "_indices", full)
        assert short.context["stack"]["stopped_at"] == "representation"
        assert f"has no matrix for {words} P6/mmm1' #191.234." in short.reason
    # the restriction refusing once the group's tables were read is a defect,
    # and it names the group
    with monkeypatch.context() as mp:
        mp.setattr(tqe, "_restriction",
                   lambda *a, **k: Refusal("ebr_tables", "synthetic", value=1.0))
        with pytest.raises(InternalError,
                           match=r"^P6/mmm1' #191\.234: in_plane_restriction refused"):
            tqe.symmetry_indicator(content)


@pytest.mark.slow
def test_the_bare_call_serves_the_class_under_the_named_s_convention():
    """No ``characters=`` at all: a full class, and the convention it rests on.

    A pythtb orbital carries no symmetry character: the hoppings fix how the
    orbitals transform relative to one another, not what they are, and
    *somebody* has to choose the labelling.  The choice here is a named
    convention -- **the lowest-l rule**: each site's content is put at the
    lowest angular momentum it allows, then the letter the orbitals read as
    and the band labels decide, and only last the group's order of operations
    (``symmetry._distance_from_trivial``), so an undeclared orbital reads as
    s-like where nothing refutes it.  It is
    deterministic and it is stated rather than hidden -- every row whose site
    irrep that choice decides reads ``lowest-l``.

    What this pins is the contract in each environment.  The convention settled on
    a substrate is the one *of the in-plane group*: ``6mm``'s four real linear
    characters are the freedom, ``A1`` is the least-twisted of them, and the
    site group has no operation that moves ``z`` for a declaration to disagree
    about -- so there the Hamiltonian decides and nothing needs explaining.  A
    **free-standing** layer has a larger site group (``-6m2``, whose
    ``sigma_h`` *does* move ``z``), so there the labelling is the convention's,
    named on the record, and the class of the layer group -- the free-standing
    question -- completes on the layer's own stack under it.
    """
    from pythtb.models import kane_mele

    model = kane_mele(0.0, -1.0, 0.3, 0.0)
    bare = tqg.realized_group(model, environment="substrate", cross_check=False)
    assert bare, str(bare)                                    # no characters= at all
    assert bare.group.number_str == "183.186"

    # (1) the contract: conventional, deterministic, trivial-preferring -- and
    # here the choice decides nothing: every one-dimensional character of 6mm
    # restricts to A1 or A2 of the 3m site, and both lift to the one -E1
    rec = bare.rep._characters
    assert rec.irreps == ("-E1", "-E1"), "both orbitals read s-like"
    assert [o.determined_by for o in rec.orbitals] == ["hamiltonian"] * 2, \
        "s and p_z cannot be told apart by a group with no operation moving z"
    assert tqs._LOWEST_L_LEGEND not in rec.summary(), "nothing to explain"

    # (2) the served class of the FREE-STANDING layer, with nothing declared
    free = tqg.realized_group(model, environment="free-standing", cross_check=False)
    assert free, str(free)
    assert free.group.number_str == "191.234"
    assert all(o.determined_by == "lowest-l"
               for o in free.rep._characters.orbitals), \
        "sigma_h tells s from p_z, and nothing was declared"
    assert tqs._LOWEST_L_LEGEND in free.rep._characters.summary(), "the convention is named"
    assert free.rep._characters.irreps == ("-E1", "-E1"), \
        "the lowest-l rule's labelling, lifted: the spinor doublet of -6m2"
    si = tqe.symmetry_indicator(tqw.hsp_table(free, bands=2))
    assert si, str(si)
    assert si.group.number_str == "191.234" and si.numerical.completion.outcome == "completed"
    assert si.site_content is not None
    assert si.site_content.irreps == ("-E1", "-E1"), (
        "the record a class is read from carries the labels it was read under, "
        "and the stack the completion read carried the flat model's own matrices "
        "and site content")


#: The Qi-Wu-Zhang two-band square-lattice model, ``H(k) = sin kx sx +
#: sin ky sy + (m + cos kx + cos ky) sz``.  Spinless, no time reversal (the
#: Chern number breaks it), two orbitals on one site at the origin.  Local to
#: this file: it is one anchor's model, not a fixture the suite reuses.
_PAULI = {"x": np.array([[0, 1], [1, 0]], dtype=complex),
          "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "z": np.array([[1, 0], [0, -1]], dtype=complex)}

#: A **chiral** decoration: one C4 orbit of a generic in-plane point, repeated
#: above and below the plane as two different species.  C4 survives; every
#: vertical mirror is broken because the orbit is not mirror-symmetric, and
#: ``sigma_h`` (with ``S4`` and inversion) because it would have to exchange
#: the species.  What is left is the group the Chern Hamiltonian obeys.
_PINWHEEL = ((0.3, 0.1), (-0.1, 0.3), (-0.3, -0.1), (0.1, -0.3))
_CHIRAL_CELL = tuple([([x, y, 0.05], "top") for x, y in _PINWHEEL]
                     + [([x, y, -0.05], "bot") for x, y in _PINWHEEL])


def _qwz(m):
    """The QWZ model at mass ``m``; the lower band carries ``C = +-1`` for ``|m| < 2``."""
    from pythtb import Lattice, TBModel

    out = TBModel(Lattice(np.eye(2), [[0.0, 0.0], [0.0, 0.0]], periodic_dirs=[0, 1]),
                  spinful=False)
    out.set_onsite([float(m), -float(m)], mode="set")
    for R, T in (([1, 0], 0.5 * (_PAULI["z"] - 1j * _PAULI["x"])),
                 ([0, 1], 0.5 * (_PAULI["z"] - 1j * _PAULI["y"]))):
        for i in range(2):
            for j in range(2):
                if T[i, j] != 0:
                    out.set_hop(complex(T[i, j]), i, j, R, mode="add")
    return out


def _chern(model, n=16, band=0):
    """One band's Chern number by plaquette Berry flux (Fukui-Hatsugai).

    Written out here rather than delegated: the point of the cross-check is
    that the number comes from the *wavefunctions* by a route with no corep,
    no character table and no EBR row anywhere in it.
    """
    ks = np.linspace(0.0, 1.0, n, endpoint=False)
    grid = np.array([[kx, ky] for kx in ks for ky in ks])
    _e, vecs = model.solve_ham(k_pts=grid, return_eigvecs=True)
    v = np.asarray(vecs).reshape(n, n, 2, 2)[:, :, band, :]
    total = 0.0
    for i in range(n):
        for j in range(n):
            a, b = v[i, j], v[(i + 1) % n, j]
            c, d = v[(i + 1) % n, (j + 1) % n], v[i, (j + 1) % n]
            total += np.angle(np.vdot(a, b) * np.vdot(b, c)
                              * np.vdot(c, d) * np.vdot(d, a))
    return total / (2 * np.pi)


def _qwz_realized(m):
    """The QWZ model in the group it realizes -- one call, no ``include_TR``.

    Time reversal is not declared anywhere here.  ``realized_group`` asks the
    cell as a grey group, the detector refutes every antiunitary operation of
    it -- a Chern number is what breaks time reversal -- and the cell is
    re-identified as the unitary group it is.  The two orbitals sit on one site
    and carry *different* one-dimensional irreps of ``C4`` (s-like and
    ``(px - i py)``-like), so the relative factor between their channels is a
    fourth root of unity and not a sign, which the build reads off the
    Hamiltonian as the root it is.
    """
    return tqg.realized_group(_qwz(m), typat=[0, 0], decorations=_CHIRAL_CELL)


@pytest.mark.slow
def test_a_layer_whose_own_group_can_answer_never_builds_the_stack():
    """The other side of the trigger: a layer that needs no stack does not get one.

    The two anchors above are layers whose own presentation cannot deliver a
    class -- rows it has no coordinate for, or a group with no indicator -- so
    the rows are measured on the decoupled stack.  This one is the control, and
    it is a different physics: a **spinless C4 Chern insulator**, the QWZ model
    at ``m = 1``, with no time reversal to have.

    Everything it needs is already in the plane.  Its cell is ``P4/mmm #123``,
    of which the Hamiltonian obeys eight operations: the four vertical-mirror
    classes are refuted by the detector outright -- a chirality is visible in
    ``k`` -- while the four rotation classes survive with the verdict their
    ``sigma_h`` partners share.  A chiral decoration (one ``C4`` orbit of a
    generic point, two species above and below) asks the cell the question the
    Hamiltonian answers, and **one call** then answers it: the cell is asked as
    the grey group ``P41' #75.2``, every one of its four antiunitary operations
    is refuted -- a Chern number is what breaks time reversal -- and the cell is
    re-identified as ``P4 #75``, unitary, all four operations obeyed.  Time
    reversal is nowhere declared; it is measured, and the descent is on the
    record.

    ``#75``'s ``A``/``R``/``Z`` rows are duplicates of ``M``/``X``/``GM`` ones,
    so the symmetry data vector completes by arithmetic; and ``X_BS(#75) = Z4`` is
    not trivial, so there is a class to read.  Neither reason to re-present is
    present, and **the stack is never built** -- ``completion is None``, the
    class is served in the group that was handed in, and it is nonzero.

    The cross-check is the physics, by a route with no corep in it: the lower
    band's Chern number from the Berry flux.  Read over the mass, the
    correspondence the group's own ``Z4`` shows is ``class = -C mod 4``:
    ``C = +1`` gives ``(3,)``, ``C = -1`` gives ``(1,)``, and both trivial
    masses give ``(0,)``.  That relation is *measured* on this family, not
    imported: what is asserted is the correspondence, and that the class has
    order 4 exactly where the Chern number is nonzero.
    """
    realized = _qwz_realized(1.0)
    assert realized, str(realized)
    model, group, rep = realized.model, realized.group, realized.rep
    _asg, validated = realized._assignment, realized.numerical.validated

    # (1)-(2) the cell allows more than the Hamiltonian obeys, and which half
    # is refuted is the physics: mirrors die, rotations live
    bare = _spacegroup._group_of(_qwz(1.0), include_TR=False, spinor=False)
    assert bare.number_str == "123" and bare.name == "P4/mmm" and bare._size == 16
    bare_v = _measure._validate_group(_layers._slab_presentation(_qwz(1.0)), bare, nk=8)
    assert len(bare_v._indices) == 8 and len(bare_v.broken) == 8 and bare_v.closed
    assert len(bare_v.undetermined) == 8, \
        "every survivor shares its k-action with a partner"
    assert all(bare_v._verdicts[i] == "broken" for i in bare_v.broken), (
        "a chirality is in-plane detectable, so the mirrors are refuted outright")

    # (3) the group the Hamiltonian realizes, reached in one call: the grey
    # group was asked, every antiunitary operation of it was refuted, and the
    # cell was re-identified as P4 -- unitary, four of four
    assert group.number_str == "75" and group.name == "P4" and group._size == 4
    assert tqe._magnetic_type(group) == "I" and group.magnetic_type == "I"
    assert len(validated._indices) == 4 and validated.broken == ()
    assert len(rep._indices) == 4 and rep._search["n_survivors"] == 1
    tr = realized.time_reversal
    assert tr.outcome == "measured broken"
    assert tr.grey_number_str == "75.2" and tr.grey_name == "P41'"
    assert (tr.n_antiunitary, tr.n_broken, tr.n_obeyed) == (4, 4, 0)
    assert tr.undetermined == (), "no antiunitary class hid behind a shared verdict"
    assert realized.beyond_grey is None, "nothing antiunitary survived to warn about"
    assert "time reversal measured broken, descended from P41' #75.2" in repr(realized)
    # ... and _build_rep called again on that group reads the fourth root of unity
    # between the two channels off the solution spaces, whatever `phases=`
    # offers, and builds the representation realized_group returned
    plain = _build._build_rep(_layers._slab_presentation(model), group, _asg,
                           validated=validated)
    assert plain, str(plain)
    assert plain._search["strategy"] == "readoff", plain._search["fallback_reason"]
    assert max(float(np.abs(np.asarray(plain.matrices[i]) - np.asarray(rep.matrices[i])).max())
               for i in rep._indices) == 0.0

    # (4) nothing is missing: every out-of-plane row duplicates an in-plane one
    restriction = tqe.in_plane_restriction(group, require_exact=False)
    assert restriction.dropped and restriction.exact and restriction.orphans == ()
    assert {tqe.kpoint_of_label(l) for l in restriction.dropped} == {"A", "R", "Z"}
    assert all(restriction.copies[l] for l in restriction.dropped)

    # (5) ... and this group has an indicator to read
    assert _smith._divisors_of(_tables._ebr_data(group), group.number_str) == (4,)

    # (6) so the class is served in the group that was handed in, and no stack
    # was ever built
    content = tqw.hsp_table(realized, bands=1)
    assert content.complete and sorted(content.kpoints) == ["GM", "M", "X"]
    si = tqe.symmetry_indicator(content)
    assert si, str(si)
    assert si.numerical.completion is None, \
        "the layer's own group returned the class; nothing was stacked"
    assert si.group.number_str == "75" and si.indicator_group == "Z4" and si.divisors == (4,)
    assert si.values == (3,) and si.order == 4 and any(si.values)
    assert (si.numerical.reconstructed
            and set(si.numerical.reconstructed) == set(restriction.dropped))

    # (7) the physics, by a route with no corep in it -- and the correspondence
    # the Z4 shows over the mass, measured rather than imported
    seen = {}
    for mass in (1.0, -1.0, 3.0):
        got = realized if mass == 1.0 else _qwz_realized(mass)    # m = 1 is read above
        assert got, str(got)
        mod, grp, rp = got.model, got.group, got.rep
        cls = tqe.symmetry_indicator(tqw.hsp_table(got, bands=1)).values
        seen[mass] = (cls, _chern(_qwz(mass)))
    assert seen[1.0][0] == (3,) and round(seen[1.0][1]) == 1
    assert seen[-1.0][0] == (1,) and round(seen[-1.0][1]) == -1
    assert seen[3.0][0] == (0,) and round(seen[3.0][1]) == 0
    for cls, c in seen.values():
        assert cls[0] == (-round(c)) % 4, "class = -C mod 4, on this family"
    assert abs(seen[1.0][1] - 1.0) < 1e-9 and abs(seen[3.0][1]) < 1e-9

    # (8) the same in P3, whose Z3 needs KA: Haldane's honeycomb with an offset,
    # read at K's partner -k, which the group carries no operation onto K, and
    # at M, whose little group is the identity alone and whose one irrep a band
    # set carries once per band.  The lowest band is no atomic limit, the class
    # is -C mod 3 in both fluxes, and the pair is the two sublattices' s
    # orbitals, exactly
    from pythtb.models import haldane

    for phi, c in ((0.5, 1), (-0.5, -1)):
        hal = haldane(0.3, -1.0, 0.15, phi)
        got = tqg.realized_group(hal, cross_check=False)
        assert got, str(got)
        assert (got.group.number_str, got.group.name) == ("143", "P3")
        lowest = tqw.hsp_table(got, bands=1)
        assert lowest.complete and list(lowest.kpoints) == ["GM", "K", "KA", "M"]
        assert lowest.symmetry_data == ({"GM1": 1, "K2": 1, "KA3": 1, "M1": 1} if c > 0
                                        else {"GM1": 1, "K3": 1, "KA2": 1, "M1": 1})
        none = tqe.decompose(lowest)
        assert not none.decomposable and none.numerical.restriction.exact
        again = tqe.decompose(dict(lowest.symmetry_data), got)
        assert ((again.decomposable, again.symmetry_data, again.numerical.restriction.retained)
                == (none.decomposable, none.symmetry_data,
                    none.numerical.restriction.retained)), \
            "the table's own vector, handed on as a mapping, reads as the table"
        si = tqe.symmetry_indicator(lowest)
        assert round(_chern(hal)) == c and si.numerical.completion is None
        assert (si.indicator_group, si.values) == ("Z3", ((-c) % 3,))
        assert tqe.symmetry_indicator(dict(lowest.symmetry_data), got).values == si.values
        rc = tqe.regular_completion(lowest, bands=3)
        assert rc and rc.padding["M1"] == 2 and rc.decomposition.candidate_sums == (), \
            "the two bands that complete a Chern band are no atomic limit either"
        off = tqe.regular_content(got.group, kpoints=["L"])
        assert isinstance(off, Refusal) and off.gate == "out_of_plane"
        pair = tqe.decompose(tqw.hsp_table(got, bands=2))
        assert pair.numerical.restriction.exact and _labels(pair) == [
            "A1 at 1b (3) + A1 at 1c (3)"]


@pytest.mark.slow
def test_the_indicator_refuses_on_physics_and_raises_on_a_broken_convention(
        fkm_strong, monkeypatch):
    """The standing law, on one function: measurement refuses, convention raises.

    A content that is not a band structure of the group, a content that is
    missing a point, a label the basis does not carry, a magnetic type outside
    the validated universe -- all of those are *measurements* about the input,
    and all of them come back as falsy :class:`Refusal` results the caller can
    read and act on.

    A shipped Smith form that is not a Smith form of the matrix it came with is
    a different animal: the tables fix that convention outright, so there is no
    gauge left for a caller to supply and nothing honest to hand back.  It
    raises :class:`~..refusal.InternalError`.  Provoked here by corrupting
    the transported ``u`` -- doubling one of its rows keeps it integer and keeps
    its shape, and destroys nothing except the identity ``u M v = R``.
    """
    import dataclasses

    from pythtqc import InternalError
    from pythtb.models import kane_mele

    model, group, rep = fkm_strong.model, fkm_strong.group, fkm_strong.rep
    table = tqw.hsp_table(fkm_strong, bands=2)

    # --- refusals: every one of these is a measurement ---------------------
    holed = dataclasses.replace(
        table, complete=False,
        refusals={"T": Refusal("integer_multiplicity", "synthetic", value=0.5)})
    out = tqe.symmetry_indicator(holed)
    assert isinstance(out, Refusal) and out.gate == "incomplete_content"
    assert "no little-group representations at ['T']" in out.reason \
        and "`refusals`" in out.reason
    assert out.context["completion"] == "stable-rsi"
    assert "A5" not in out.reason, "the refusal names physics, not a build plan"

    out = tqe.symmetry_indicator({"-GM9": 1}, fkm_strong)
    assert isinstance(out, Refusal) and out.gate == "not_a_band_structure"
    assert out.context["violations"] and out.value > 0
    # each violation is a Smith coordinate and its value, no row's label
    y_prime = out.context["y_prime"]
    assert all(len(pair) == 2 and y_prime[pair[0]] == pair[1] != 0
               for pair in out.context["violations"])
    # ... the check a decomposition runs before it solves, the same one
    told = tqe.decompose({"-GM9": 1}, fkm_strong)
    assert isinstance(told, Refusal) and told.gate == "not_a_band_structure"
    assert told.context["violations"] == out.context["violations"]
    # the group is known, so the refusal names it and carries it
    assert told.context["group"] is group and out.context["group"] is group
    assert f"violates {group.name} #{group.number_str}'s compatibility" in out.reason

    out = tqe.symmetry_indicator({"-NOSUCH3": 1}, fkm_strong)
    assert isinstance(out, Refusal) and out.gate == "unknown_label"
    assert out.context["group"] is group

    # The magnetic scope gate is applied at the CONSTRUCTOR (groups._group_of,
    # tests/test_groups.py), so an out-of-scope group cannot reach here by
    # the shipped route at all.  Lift it, build the group anyway, and the
    # indicator's body, handed that group, refuses it on its own: one gate
    # name, one detector, two places.
    km_afm = dict(magmom=[[0, 0, 1.0], [0, 0, -1.0]])
    refused = _spacegroup._group_of(kane_mele(0.0, -1.0, 0.3, 0.0), **km_afm)
    assert isinstance(refused, Refusal)
    assert refused.gate == "magnetic_type_out_of_scope"
    with monkeypatch.context() as mp:
        patch_everywhere(mp, "_magnetic_scope_refusal", lambda _sg: None)
        magnetic = _spacegroup._group_of(kane_mele(0.0, -1.0, 0.3, 0.0), **km_afm)
    assert magnetic and magnetic.number_str == "191.236", str(magnetic)
    assert tqe._magnetic_type(magnetic) == "III"
    out = tqe._symmetry_indicator({}, magnetic)
    assert isinstance(out, Refusal) and out.gate == "magnetic_type_out_of_scope"
    assert out.context["magnetic_type"] == "III"
    # ... and the two types that ARE in scope are told apart by the operations
    assert tqe._magnetic_type(group) == "II"
    assert tqe._magnetic_type(_spacegroup._group_of(kane_mele(0.0, -1.0, 0.3, 0.0),
                                            include_TR=False)) == "I"

    # --- and the convention failure raises ---------------------------------
    real = _tables._ebr_data(group)
    broken = dict(real)
    u, r, v = real["smith"]
    bad = np.array(u, dtype=int)
    bad[0] *= 2
    broken["smith"] = (bad, r, v)
    patch_everywhere(monkeypatch, "_ebr_data", lambda g: broken)
    with pytest.raises(InternalError) as excinfo:
        tqe.symmetry_indicator(table)
    assert "Smith" in str(excinfo.value)


def test_named_indicator_forms_are_transported_before_they_are_read():
    """``irreptables``' named formulas are keyed by BCS labels; ours are not.

    The coefficients live on the BCS convention's labels, so a form has to be
    re-keyed by the transport's row permutation before it can be read on a
    served symmetry data vector.  In scope that changes exactly one form: the ``z2``
    of grey ``220.90`` (``I-43d1'``), whose coefficients sit on the ``P``/``PA``
    labels the transport swaps.

    The re-keying is asked of ``_named_forms`` rather than performed here: the
    module reports which forms the permutation moved in ``touched`` and returns
    the re-keyed coefficient vector it actually evaluated, and those are what
    is read below.  No fixture in this suite is grey ``#220.90`` -- ``I-43d1'``
    is a body-centred cubic group no model here realizes -- so the group record
    is stood up from the three fields the table route reads off it, and the
    tables themselves supply everything else.
    """
    import json
    import os
    import types

    from irreptables import symmetry_indicators as _si

    root = os.path.join(os.path.dirname(_si.__file__), "data", "symmetry_indicators")
    grey = json.load(open(os.path.join(root, "double_indicators_magnetic.json"),
                          encoding="utf-8"))

    # the one in-scope form the transport moves, asked of the module
    g220 = types.SimpleNamespace(number_str="220.90", spinor=True, _magnetic=True)
    data = _tables._ebr_data(g220)
    assert isinstance(data, dict), str(data)
    matrix = [[int(x) for x in row]
              for row in np.asarray(data["matrix"], dtype=int)]
    y = [row[0] for row in matrix]                  # an EBR column: any form is 0 on it
    named, touched, note, coeffs = tqe._named_forms(g220, data, matrix, y, "II")
    assert touched == ("z2",), touched
    assert named == {"z2": 0}, "and it vanishes on an EBR, as an indicator must"
    assert "re-keyed by the row permutation of the change of convention" in note

    # ... and the coefficients it evaluated are not the shipped ones: they differ
    # exactly on the P/PA labels the transport swaps (measured 2026-09-09; the
    # eight entries there change sign)
    labels_bcs = list(data["convention"]["labels_bcs"])
    shipped = [float(grey["220.90"]["indicators"]["z2"]["factors"].get(l, 0.0))
               for l in labels_bcs]
    served = list(coeffs["z2"])
    assert len(served) == len(shipped) == 17
    moved_labels = {labels_bcs[i] for i in range(len(shipped))
                    if served[i] != shipped[i]}
    assert len(moved_labels) == 8
    assert {tqe.kpoint_of_label(l) for l in moved_labels} == {"P", "PA"}


def test_a_call_handed_what_it_does_not_take_raises_saying_what_to_pass(
        lin3g_declared_cross_checked, p4232_descended):
    """A table is read alone, and a vector written out by hand with what realized_group returned.

    A table carries what it was read from, its group, its site irreps and, for
    a layer, the representation its decoupled stack is built from, so anything
    handed beside it is a mistake; a vector written out by hand carries
    nothing, so it is read in the measurement handed second.  Each mistake
    raises ``TypeError`` naming what to pass.  Two tables of two groups are no
    one alphabet to subtract in.  The lookups take the group, and a name the
    group does not tabulate raises listing the names.
    """
    record = lin3g_declared_cross_checked
    table = tqw.hsp_table(record, bands=P6)
    vector = dict(table.symmetry_data)
    for call in (tqe.decompose, tqe.symmetry_indicator, tqr.srsi_of_bands,
                 tqe.regular_completion):
        name = call.__name__
        kw = {"bands": 8} if call is tqe.regular_completion else {}
        with pytest.raises(TypeError, match=rf"^{name}\(\): a table is passed with realized= "
                                            rf"beside it; pass the table alone: {name}\(t\)\."):
            call(table, record, **kw)
        with pytest.raises(TypeError, match=rf"second: {name}\(symmetry_data, r\)\."):
            call(vector, **kw)
        for second in (record.group, record.rep):
            with pytest.raises(TypeError, match=r"pass what realized_group returns second") as exc:
                call(vector, second, **kw)
            assert f"realized= is of type {type(second).__name__};" in str(exc.value)
        with pytest.raises(TypeError, match=r"symmetry_data= is of type list; pass the table "
                                            r"hsp_table returns"):
            call(["-GM6"], record, **kw)
        with pytest.raises(TypeError, match="unexpected keyword argument 'points'"):
            call(table, points=("GM", "M", "X"), **kw)
    # handed what each takes, the table alone and its vector beside r read alike
    assert tqe.decompose(table).candidate_sums == tqe.decompose(vector, record).candidate_sums
    assert tqe.symmetry_indicator(vector, record).values == tqe.symmetry_indicator(table).values
    with pytest.raises(TypeError, match=r"^regular_completion\(\): no bands= is passed"):
        tqe.regular_completion(table)
    with pytest.raises(ValueError) as exc:
        tqe.subtract(table, tqw.hsp_table(p4232_descended, bands=2))
    assert "#99.164" in str(exc.value) and "#208.45" in str(exc.value), \
        "the message names both groups"
    for args, name in (((["-GM6"], vector), "symmetry_data_a"),
                       ((vector, ["-GM6"]), "symmetry_data_b")):
        with pytest.raises(TypeError, match=rf"^subtract\(\): {name}= is of type list"):
            tqe.subtract(*args)
    # a layer's table keeps the representation its stack is built from; a
    # crystal's holds no reference to the model
    assert table._rep is record.rep
    assert tqw.hsp_table(p4232_descended, bands=2)._rep is None
    for call in (tqe.in_plane_restriction, tqe.regular_content):
        for wrong in (record, record.rep):
            with pytest.raises(TypeError, match=r"r\.group"):
                call(wrong)
        with pytest.raises(ValueError, match=r"has no k-point named 'Q'; pass one of GM, M, X"):
            call(record.group, kpoints=("GM", "Q"))


def test_module_surface():
    """The names phase A5 and the docs will import."""
    import pythtqc as tqc

    # the three content helpers are the module's own and stay in `pythtqc.ebr`,
    # as the stack's period stays in `pythtqc.groups`: the top level carries
    # what the documents teach
    module_only = {"regular_content", "kpoint_of_label", "content_of", "STACK_C"}
    for name in ("decompose", "subtract", "regular_completion",
                 "regular_content", "in_plane_restriction", "ebr_rows",
                 "kpoint_of_label", "content_of", "symmetry_indicator",
                 "EBRDecomposition", "Restriction", "RegularCompletion",
                 "SymmetryIndicator"):
        assert name in tqe.__all__, name
        if name in module_only:
            assert not hasattr(tqc, name) and name not in tqc.__all__, name
            continue
        assert getattr(tqc, name) is getattr(tqe, name), name
        assert name in tqc.__all__, name
    assert tqc.ebr is tqe
    for name in ("realized_group", "stack_presentation", "RealizedGroup", "STACK_C"):
        assert name in tqg.__all__, name
        if name in module_only:
            assert not hasattr(tqc, name) and name not in tqc.__all__, name
            continue
        assert name in tqc.__all__, name
        assert getattr(tqc, name) is getattr(tqg, name), name
    # the records a decomposition and an indicator hold are the module's own
    for name in ("EBRDecompositionNumerical", "SignedCombinations", "SiteContent",
                 "SymmetryIndicatorNumerical", "StackCompletion", "TwoDimensionalReading"):
        assert name in tqe.__all__ and not hasattr(tqc, name), name
    for name in ("completeness_check", "CompletenessReport", "WannierCaveat", "CAVEAT_TEXT"):
        assert not hasattr(tqe, name) and not hasattr(tqc, name), name
    for name in ("_slab_presentation", "_decorated_cell"):
        assert name not in tqg.__all__ and not hasattr(tqc, name), name
        assert callable(getattr(_layers, name)), name
    # the convention statement travels on every record
    assert "e^(-ik.t)" in tqe.INDICATOR_CONVENTION
    assert "Wyckoff positions never move" in tqe.INDICATOR_CONVENTION
