"""Phase A2 regressions: the representation constructor of ``tqc.symmetry``.

What each test is for, and the number that closes it:

* **rep pinning, unique of 32 at covariance <= 5e-14** — the phase closer.
  A reference implementation machine-pinned ``(rowphase, dC4, 2cC4, dMx, 2cMx) =
  (-1,-1,+1,+1)`` out of 32 candidates in the *strictly-periodic* convention;
  this suite reproduces the same enumeration in **pythtb's** convention
  (the logic and the numbers port, the phases are re-derived) and
  gets the same four orbital signs with the same uniqueness.
* **the uniqueness gate is joint** — covariance, the group's own internal
  factor, and its multiplication table, with the per-candidate record of which
  one fired.  The anchor above is untouched by the two new gates (they kill
  nothing there); the fixture that *needs* them is inversion-symmetric
  Kane–Mele in ``P6mm1'``, whose two orientation branches are both covariant
  **and** both multiplicative and are separated only by the internal factor.
  Both halves of that are measured here, including the negative one — the
  losing branch is built explicitly and its product-table residual read.
* **the built reps reproduce ``get_product_table(get_diff=True)``** — spinor
  factors included — on lin3g's spinful ``P4/mmm`` slab group, against an
  independent loop; the nonsymmorphic factor system of the Bloch operators is
  ``test_multiorbital.py``'s, on SG 199.
* **the antiunitary anchor** — a ``(-i)x`` operator is planted and caught, and
  the corep relabelling it would have caused is displayed alongside.
* **the Bloch covariance battery** on a multi-orbit model whose orbitals sit at
  half lattice vectors, which is where the site-permutation keying of gotcha #3
  bites and where single-orbit models are silent.
* **the canonical norm and the anchor as refusal gates**, each on a matrix
  built to fail it.
"""

import time

import numpy as np
import pytest

from conftest import patch_everywhere
from pythtb import Lattice, TBModel

from pythtqc import _sitechars
from pythtqc import groups as tqg
from pythtqc import spin as tqspin
from pythtqc import ebr as tqe
from pythtqc import sewing as tqw
from pythtqc import symmetry as tqs
from pythtqc import InternalError
from pythtqc.refusal import Refusal
from pythtqc import _build
from pythtqc import _layers
from pythtqc import _matrices
from pythtqc import _measure
from pythtqc import _spacegroup
from pythtqc import _tables

import tqc_models

GATE = tqs.EXACT_TOL         # the exact floor of _build_rep's covariance gate
_ULP = np.finfo(float).eps         # one unit of double-precision roundoff, 2.2e-16
_FEW_ULP = 8 * _ULP                # a few ulp: the rounding of an order-one matrix product


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------

def _nspin(rep):
    return int(rep.model.nspin)


def _rep_bundle(model, group=None, **kwargs):
    """``(group, assignment, validated, rep)`` for one model, built once."""
    group = _spacegroup._group_of(model) if group is None else group
    assert group, str(group)
    assignment = _tables._assign_orbitals(model, group)
    assert assignment, str(assignment)
    validated = _measure._validate_group(model, group, nk=kwargs.pop("nk", 6))
    rep = _build._build_rep(model, group, assignment, validated=validated, **kwargs)
    return group, assignment, validated, rep


@pytest.fixture(scope="module")
def lin3g_rep(lin3g):
    return _rep_bundle(lin3g)


@pytest.fixture(scope="module")
def lin3g_rep_enum(lin3g_rep):
    """The same model built by the **exhaustive** search, on the same measurement.

    ``strategy="readoff"`` is the default and reads the discrete choices off the
    detector's solution space, so it builds one candidate per orientation branch
    and its kill record has one row per *read*.  The anchors below regress the
    enumeration itself — ``1 of 32``, its 31 covariance kills, the 512-candidate
    fourth-root run — so they name that strategy rather than being weakened to
    whatever the default happens to produce.  Section 11 is where the two are
    measured against each other.
    """
    group, assignment, validated, rep = lin3g_rep
    out = _build._build_rep(rep.model, group, assignment, validated=validated,
                         strategy="enumerate")
    assert out, str(out)
    return group, assignment, validated, out


def inline_2c_model():
    """Spinless p4mm: an s orbital at 1a and an s pair at 2c = (1/2,0), (0,1/2).

    The point of this fixture is the **2c orbit**: C4 exchanges its two sites
    with an integer lattice shift ``(-1, 0)``, so every row phase of the Bloch
    operator has to follow the row's *preimage* through the operation.  A
    single-orbit model (every orbital at the origin, or one site per cell)
    cannot expose that bug, which is exactly why gotcha #3 survived so long.
    """
    lat = Lattice([[1, 0], [0, 1]], [[0, 0], [0.5, 0.0], [0.0, 0.5]], periodic_dirs=[0, 1])
    m = TBModel(lat, spinful=False)
    m.set_onsite([0.4, -0.2, -0.2], mode="add")
    for R in ([1, 0], [0, 1]):                       # 1a-1a nearest neighbours
        m.set_hop(0.35, 0, 0, R, mode="add")
    for R in ([0, 0], [-1, 0]):                      # 1a - 2c(1/2,0)
        m.set_hop(0.5, 0, 1, R, mode="add")
    for R in ([0, 0], [0, -1]):                      # 1a - 2c(0,1/2)
        m.set_hop(0.5, 0, 2, R, mode="add")
    for R in ([0, 0], [1, 0], [0, -1], [1, -1]):     # the four 2c-2c bonds
        m.set_hop(0.17, 1, 2, R, mode="add")
    return m


@pytest.fixture(scope="module")
def inline_rep():
    model = inline_2c_model()
    return (model,) + _rep_bundle(model)


# ---------------------------------------------------------------------------
# 1.  the phase closer: rep pinning, unique of 32
# ---------------------------------------------------------------------------

def test_lin3g_rep_is_pinned_unique_of_32_by_covariance(lin3g_rep_enum):
    """The reference's pinning in pythtb's convention: 1 of 32 at covariance <= 5e-14.

    The reference enumerated ``2**5`` candidates — one global convention branch
    (its row-phase sign) times four orbital signs: the ``d`` orbital at 1a and
    the 2c block, each under ``C4`` and under a mirror, with the ``s`` orbital
    at 1a pinned to ``+1`` as the orbital-trivial reference.  Here the four
    orbital signs are the same four (two channel classes beyond the reference,
    two generators) and the convention branch is pythtb's analogue: ``D``
    versus ``D†``, i.e. representing ``g`` versus ``g**-1`` while testing
    against ``g``'s own k-action.  The row-phase *sign* itself is not a
    candidate in this convention because it is invisible on this model: every
    orbital sits at a half lattice vector, so every wrap phase is real.

    Uniqueness is asserted on **three** gates -- covariance, the group's own
    internal factor and its multiplication table -- and each candidate says
    which killed it.  On lin3g the covariance battery does all the work, so the
    record is one row per distinct candidate, a residual per gate that actually
    ran, ``nan`` for the gates that were never reached, and ``killed_by`` naming
    exactly one of them.

    Built with ``strategy="enumerate"``: this is the enumeration's own anchor, and
    section 11 measures that the default read-off reproduces every number in it.
    """
    group, assignment, validated, rep = lin3g_rep_enum
    assert rep, str(rep)

    assert rep._search["strategy"] == "enumerate"
    assert rep._search["n_candidates"] == 32
    assert rep._search["n_distinct"] == 32
    assert rep._search["n_survivors"] == 1
    assert rep._search["best"] <= GATE
    # the survivor is isolated by ~14 orders of magnitude, not by a hair
    assert rep._search["runner_up"] > 1e-3

    assert max(rep.residuals.values()) <= GATE

    # the four machine-pinned orbital signs, against out101 [0b]'s (-1,-1,+1,+1)
    c4, mirror = rep._generators
    classes = rep._classes
    d_class = int(classes[1])          # the d orbital at 1a
    c_class = int(classes[2])          # the 2c pair
    assert len({int(classes[0]), d_class, c_class}) == 3, \
        "three distinct channel classes: the four pinned signs rest on that"
    signs = rep._choice["signs"]
    assert signs[(c4, d_class)] == pytest.approx(-1)
    assert signs[(c4, c_class)] == pytest.approx(-1)
    assert signs[(mirror, d_class)] == pytest.approx(+1)
    assert signs[(mirror, c_class)] == pytest.approx(+1)
    assert rep._choice["orientation"] == 1

    # the represented set is p4mm's double-group lift plus time reversal
    assert len(rep.indices) == 16
    assert len(rep.antiunitary) == 8

    # the joint gate: covariance alone pins this model, the other two gates are
    # pure confirmation, and the survivor closes all three
    s = rep._search
    assert (s["n_distinct"], s["n_covariant"], s["n_consistent"], s["n_survivors"]) \
        == (32, 1, 1, 1)
    assert s["killed_by"] == {"covariance": 31, "internal_factor": 0, "product_table": 0}
    assert s["internal"] < 1e-12 and s["product"] < 1e-13
    rows = s["candidates"]
    assert len(rows) == 32 and sum(r["killed_by"] is None for r in rows) == 1
    winner = next(r for r in rows if r["killed_by"] is None)
    assert winner["orientation"] == rep._choice["orientation"] == 1
    assert winner["combo"] == (-1, -1, 1, 1), "out101 [0b]'s four orbital signs, again"
    for r in rows:
        if r["killed_by"] is None:
            continue
        assert r["killed_by"] == "covariance" and r["covariance"] > 1e-3
        assert np.isnan(r["internal"]) and np.isnan(r["product"]), \
            "a gate that never ran reports nan, not a number nobody measured"


def _decoupled_two_class_layer(e1=0.45, t0=0.30, t1=-0.19, d0=0.05, d1=-0.03):
    """Two spinless orbitals on a square layer, at ``1a`` and ``1b``, with **nothing between them**.

    ``4mm`` fixes both positions, so each orbital is its own channel class, and
    the Hamiltonian is block diagonal in those two classes at every ``k`` --
    different on-site energies and different hoppings, no inter-orbital term.
    A *relative* phase between the two classes is therefore a symmetry of ``H``
    that the covariance battery cannot see, which is what makes this the
    fixture that reaches the third gate (see the test below).  ``typat``
    separates the two sites: undecorated, ``(0,0)`` and ``(1/2,1/2)`` are one
    body-centred point set and the two orbitals would land in one class.
    """
    lattice = Lattice(np.array([[1.0, 0.0], [0.0, 1.0]]),
                      [[0.0, 0.0], [0.5, 0.5]], periodic_dirs=[0, 1])
    model = TBModel(lattice, spinful=False)
    model.set_onsite([0.0, e1], mode="set")
    for orb, t, d in ((0, t0, d0), (1, t1, d1)):
        model.set_hop(t, orb, orb, [1, 0])
        model.set_hop(t, orb, orb, [0, 1])
        model.set_hop(d, orb, orb, [1, 1])
        model.set_hop(d, orb, orb, [1, -1])
    return model


@pytest.mark.slow
def test_the_product_table_gate_kills_what_covariance_cannot_see():
    """The third gate, on the one fixture where it is the gate that decides.

    On every model above the covariance battery does all the killing, because
    the phases the enumeration ranges over are *relative* phases between channel
    classes (``_build_rep._build``: ``sgn[c] = combo[...]`` on the free classes
    only) and a relative phase inside a coupled Hamiltonian moves ``U H U†``.
    Decouple the classes and it does not: two spinless orbitals at ``1a`` and
    ``1b`` of a square layer with no hopping between them, whose ``H(k)`` is
    block diagonal in the two classes at every ``k``.  Every one of the 16
    distinct fourth-root candidates then closes covariance (measured 1.1e-16 on
    all of them) and the internal-factor gate is vacuous besides (``S`` is
    ``1x1`` on a spinless model with no declared spinor basis, so every block is
    a multiple of it by arithmetic, at exactly ``0.0``).

    What is left is the multiplication table, and it keeps exactly the four real
    linear characters of ``4mm``: that group's abelianization is ``Z2 x Z2``, so
    no fourth root is a character of it, and ``killed_by`` reads
    ``{covariance: 0, internal_factor: 0, product_table: 12}``.  Of the 12, four
    reach the table and miss it by exactly **2.0** -- the worst pair is
    ``C2(z) o m([110])`` against ``m([1-10])``, whose class-1 entries come out
    ``+1`` and ``-1`` -- and eight are refused by the canonical-norm step before
    the table is measured (``product`` is ``inf``), which ``killed_by`` files
    under the table too, since the two gates before it passed.

    Four survivors is four genuine representations differing by a character, so
    the build refuses ``rep_not_unique`` rather than serving one, and the search
    record is handed back beside the refusal.  That is the honest outcome here:
    what this test is for is the gate, not a served representation.
    """
    model = _decoupled_two_class_layer()
    group = _spacegroup._group_of(model, typat=[0, 1])
    assert group and group.number_str == "123.340", str(group)
    assignment = _tables._assign_orbitals(model, group)
    assert assignment and assignment.orbits == ((0,), (1,)), str(assignment)
    validated = _measure._validate_group(model, group, nk=6)
    assert validated, str(validated)
    assert set(validated._nullspace_dims.values()) == {2}, \
        "the decoupling IS the second dimension: the relative phase commutes with H"

    record = {}
    out = _build._build_rep(model, group, assignment, validated=validated,
                         phases=(1, -1, 1j, -1j), strategy="enumerate",
                         _search_record=record)
    assert not out and out.gate == "rep_not_unique", str(out)
    assert out.value == 4.0 and out.context["n_candidates"] == 32
    s = record["search"]
    assert (s["n_candidates"], s["n_distinct"]) == (32, 16), \
        "the two orientation branches build the same matrices on a spinless model"
    assert (s["n_covariant"], s["n_consistent"], s["n_survivors"]) == (16, 16, 4)
    assert s["killed_by"] == {"covariance": 0, "internal_factor": 0, "product_table": 12}, \
        "the third gate is the one that decides here"

    rows = s["candidates"]
    assert len(rows) == 16
    assert max(r["covariance"] for r in rows) <= _FEW_ULP, \
        "covariance is blind to a relative phase between decoupled classes"
    assert max(r["internal"] for r in rows) <= _FEW_ULP, \
        "and a 1x1 internal factor divides every block (measured exactly 0.0)"

    killed = [r for r in rows if r["killed_by"] == "product_table"]
    assert len(killed) == 12
    measured = [r for r in killed if np.isfinite(r["product"])]
    assert len(measured) == 4, "four candidates reach the table itself"
    assert {tuple(complex(z) for z in r["combo"]) for r in measured} == {
        (1j, 1 + 0j), (1j, -1 + 0j), (-1j, 1 + 0j), (-1j, -1 + 0j)}, \
        "a fourth root on the four-fold, a real sign on the mirror"
    for r in measured:
        assert r["product"] == pytest.approx(2.0, abs=1e-12), r["combo"]
    assert all(r["product"] == float("inf") for r in killed if r not in measured), \
        "the other eight never reach it: the canonical norm refuses first"

    survivors = {(r["orientation"], tuple(complex(z) for z in r["combo"]))
                 for r in rows if r["killed_by"] is None}
    assert survivors == {(1, (a, b)) for a in (1 + 0j, -1 + 0j) for b in (1 + 0j, -1 + 0j)}, \
        "the four real characters of 4mm, and nothing else"


# ---------------------------------------------------------------------------
# 2.  the built reps reproduce the group's own product table
# ---------------------------------------------------------------------------

def test_lin3g_reps_reproduce_the_product_table_with_spinor_factors(lin3g_rep):
    """Spinful group #1: ``D(g1) o D(g2) = spf(g1,g2) D(g1 g2)`` to machine precision."""
    group, assignment, validated, rep = lin3g_rep
    prod, tdiff, spf = group.get_product_table(get_diff=True)
    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}

    assert set(np.unique(spf)) == {-1, 1}, "a spinful group must have both spinor factors"
    worst, seen = 0.0, set()
    for a in rep.indices:
        for b in rep.indices:
            pa, pb = pos[a], pos[b]
            c = int(group.symmetries[int(prod[pa, pb])].ind)
            assert c in rep.matrices, "the represented set is not closed"
            lhs = _matrices._compose(rep.matrices[a], bool(rep._operation(a).time_reversal),
                               rep.matrices[b])
            worst = max(worst, float(np.abs(lhs - int(spf[pa, pb]) * rep.matrices[c]).max()))
            seen.add(int(spf[pa, pb]))
    assert seen == {-1, 1}, "the spinor factors were never exercised"
    assert worst < 1e-13
    # the object's own number, against the loop above -- which is the only place
    # in this file where ``product_residual`` meets an independently computed
    # value rather than another call of the routine that filled it.  Two routes
    # over the same pairs of the same matrices in one process, so the honest
    # tolerance is exact: measured 6.67378471839441e-16 on both sides.
    assert rep.product_residual == pytest.approx(worst, abs=0.0)

    # canonical norms and the anchor, per operation, after normalization
    assert max(rep._norm_residuals.values()) < 1e-13
    assert max(rep._anchor_residuals.values()) < 1e-13


# ---------------------------------------------------------------------------
# 2b.  the orientation branch: the case covariance cannot decide
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def km191_bundle():
    """Inversion-symmetric Kane-Mele (``delta = rashba = 0``) in ``P6mm1' #183.186``.

    The fixture whose orientation branch used to need a workaround.  Built
    through ``groups.realized_group``'s own pieces so this test can re-run
    ``_build_rep`` with different gate settings on the *same* group.
    """
    model = tqc_models.kane_mele_inversion()
    m3 = _layers._slab_presentation(model)
    cell, cell_typat = _layers._decorated_cell(m3, [0, 0])
    group = _spacegroup._group_of(cell, typat=cell_typat, spinor=True)
    assert group and group.number_str == "183.186"
    assignment = _tables._assign_orbitals(m3, group, typat=[0, 0])
    validated = _measure._validate_group(m3, group, nk=6)
    return m3, group, assignment, validated


def test_kane_mele_at_191_pins_with_default_settings(km191_bundle):
    """The regression: both branches are asked, one is read, and the record says why.

    ``kane_mele(delta=0, rashba=0)`` is the case that forced the joint gate.
    It conserves ``s_z``, so the detector's solution space for ``C₃`` holds its
    own matrix times every rotation about ``z``, ``S(C₃)†`` among them, and
    both orientation branches read off that space are covariant; the build
    used to need ``orientation_branch=+1``, which does not decide anything, it
    just stops asking.  Each operation's space is read in its own form, whose
    members carry ``S(g)`` alone, so the ``g⁻¹`` branch, which reads a member
    as ``D(g)†``, has no member to be read: its block is not a multiple of the
    lift, the defect the internal-factor gate measured when the branch was
    built, and the answer is the one the workaround took.

    Why the product table is **not** what pins this, measured rather than
    assumed: the losing branch, asked for alone with the internal-factor gate
    opened, closes the covariance gate *and* reproduces the multiplication
    table with its spinor factors -- ``g -> D(g⁻¹)†`` is a homomorphism
    whatever ``D`` is.  What it does not carry is the internal factor of the
    operations it claims to represent: on a channel ``C₃`` fixes its block is
    ``S(C₃)†``, and the site character read off it is no character at all
    (``chi(C3) = -0.5``).
    """
    model, group, assignment, validated = km191_bundle
    rep = _build._build_rep(model, group, assignment, validated=validated,
                         characters={0: "A1"})
    assert rep, str(rep)
    s = rep._search
    branches = s["readoff"]["branches"]
    assert branches[1]["ok"] and not branches[-1]["ok"], "both are asked, one is read"
    assert "is not a multiple of S(g)" in branches[-1]["reason"]
    assert (s["n_candidates"], s["n_survivors"]) == (1, 1)
    assert s["killed_by"] == {"covariance": 0, "internal_factor": 0, "product_table": 0}

    assert rep._choice["orientation"] == 1
    assert rep._branch_tie is False, "measured by the internal factor, not conventional"
    assert rep._search["branch_note"] is None
    assert max(rep.residuals.values()) <= GATE
    assert rep.product_residual < 1e-13 and rep.factor_residual < 1e-13
    # the s orbitals' A1 of the site group, returned on this spinful model as the corep -E1
    assert rep.characters.irreps == ("-E1", "-E1") and rep.characters.residual < 1e-12

    # ... and it is bit-for-bit what the old workaround produced
    old = _build._build_rep(model, group, assignment, validated=validated,
                         characters={0: "A1"}, orientation_branch=+1)
    assert old, str(old)
    assert max(float(np.abs(old.matrices[i] - rep.matrices[i]).max())
               for i in rep.indices) == 0.0

    # the losing branch alone, the internal-factor gate opened
    loser = _build._build_rep(model, group, assignment, validated=validated,
                           orientation_branch=-1, internal_tol=np.inf)
    assert loser, str(loser)
    assert loser._choice["orientation"] == -1
    assert max(loser.residuals.values()) <= GATE, "exactly as covariant as the winner"
    assert loser.product_residual < 1e-13, "and exactly as multiplicative"
    assert loser.factor_residual < 1e-13
    assert max(loser._norm_residuals.values()) < 1e-13, "the canonical norms pass too"
    assert max(loser._anchor_residuals.values()) < 1e-13, "and so does the antiunitary anchor"

    assert loser._search["internal"] > 0.5, "the one measurement that separates them"
    assert loser.characters.irreps == ("", ""), \
        "no row of the site table matches what these blocks measure"
    assert abs(loser.characters.orbitals[0].generator_character["C3(z)"] + 0.5) < 1e-9

    # with the gate at its default, that branch alone is a refusal, not a rep
    record = {}
    refused = _build._build_rep(model, group, assignment, validated=validated,
                             orientation_branch=-1, _search_record=record)
    assert not refused and refused.gate == "rep_internal_factor"
    assert refused.value > 0.5 and refused.threshold == 1e-9
    assert record["search"]["killed_by"]["internal_factor"] == 1


# ---------------------------------------------------------------------------
# 2c.  the orientation branch where NOTHING can decide (wave R4e, 2026-09-06)
# ---------------------------------------------------------------------------
#
# Section 2b is the case a measurement settles: `S(C3)^dag` is not a multiple of
# `S(C3)`, so the internal-factor gate kills the `g^-1` branch at defect 0.866.
# For a group of INVOLUTIONS only -- `mm2`, `P-1`, a spinful `2/m` -- every lift
# obeys `S(g)^dag = -S(g)`, that gate's free per-block coefficient absorbs the
# sign, and covariance and the product table never saw the branch at all.  There
# is nothing left to measure, and `_build_rep` pins the branch by convention
# instead of refusing (`branch_tie`).
#
# The owner set two consistency criteria for that convention (2026-09-06), and
# both are measured below.  **Continuity**: weakly break the `C4` of a `4mm`
# model and the child's representation must deterministically match the parent's
# on the `mm2` that survives.  **Convention**: a pure EBR model in `mm2` must
# match the induced representation the tables define.
#
# They agree with each other, and the answer to both is the same one: the branch
# is not what is at stake.  On every tie here the two branches build the
# IDENTICAL matrices (worst 3.7e-16) -- `normalize_rep`'s canonical phase undoes
# the sign the adjoint put on each generator -- so the convention moves nothing.
# What separates a child from its parent, and a build from the induced
# representation, is exactly one REAL LINEAR CHARACTER of the group: the single
# global one-dimensional character the hoppings cannot supply, which `_build_rep`
# documents as a declaration and every undeclared row's `status` names.  Both
# tests measure that character, check it is one (multiplicative on the group's
# own product table), and check that once it is divided out the two objects
# agree to machine precision.  The spinless control below has no such freedom
# (`S = [[1]]`, the branches merge outright) and the child's representation is
# the parent's restriction at exactly 0.0.

def _op_key(sym):
    """An operation's identity across two different groups on the same lattice."""
    return (tuple(np.asarray(sym.rotation, dtype=int).ravel().tolist()),
            tuple(np.round(np.asarray(sym.translation, dtype=float) % 1.0, 9).tolist()),
            bool(sym.time_reversal))


def _snap(measured):
    """``({ind: ±1.0}, defect)`` — the measured ratios read as the signs they are."""
    snapped = {ind: float(np.sign(complex(v).real)) for ind, v in measured.items()}
    off = max((abs(complex(v) - snapped[ind]) for ind, v in measured.items()), default=0.0)
    return snapped, float(off)


def _relative_character(child_rep, child_group, parent_rep, parent_group):
    """``({ind: ±1}, block defect, snapping defect)`` — ``D_child = chi·D_parent``."""
    parent = {_op_key(s): int(s.ind) for s in parent_group.symmetries}
    rho, defect = {}, 0.0
    for s in child_group.symmetries:
        ind = int(s.ind)
        if ind not in child_rep.matrices:
            continue
        pind = parent[_op_key(s)]
        r, res = _matrices._proportionality(np.asarray(child_rep.matrices[ind]),
                                      np.asarray(parent_rep.matrices[pind]))
        rho[ind], defect = complex(r), max(defect, float(res))
    snapped, off = _snap(rho)
    return snapped, defect, off


def _is_linear_character(rho, group):
    """``chi(g)chi(h) = chi(gh)`` on every composable pair, and every value ±1."""
    if any(abs(abs(v) - 1.0) > 1e-9 for v in rho.values()):
        return False
    prod = group.get_product_table(get_diff=False)
    pos = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    for a, va in rho.items():
        for b, vb in rho.items():
            c = int(group.symmetries[int(prod[pos[a], pos[b]])].ind)
            if c in rho and abs(va * vb - rho[c]) > 1e-9:
                return False
    return True


def _deviation(child_rep, child_group, parent_rep, parent_group, rho):
    """``max_g ‖D_child(g) − rho(g) D_parent(g)‖``, the sign included."""
    parent = {_op_key(s): int(s.ind) for s in parent_group.symmetries}
    worst = 0.0
    for s in child_group.symmetries:
        ind = int(s.ind)
        if ind not in child_rep.matrices:
            continue
        A = np.asarray(child_rep.matrices[ind])
        B = np.asarray(parent_rep.matrices[parent[_op_key(s)]])
        worst = max(worst, float(np.abs(A - complex(rho[ind]) * B).max()))
    return worst


def _axially_split(delta):
    """The paper's ``P4mm1'`` flat-band model with ``±delta`` on its two sites.

    ``C4`` and the diagonal mirrors exchange ``(1/2, 0)`` with ``(0, 1/2)``, so
    an on-site energy difference breaks exactly those four (and their
    time-reversed copies) and keeps ``E``, ``C2(z)``, ``m(x)``, ``m(y)``, which
    fix both sites: the axial ``mm2``.
    """
    model = tqc_models.p41_flat_band()
    model.set_onsite([+float(delta) * np.eye(2), -float(delta) * np.eye(2)], mode="add")
    return model


@pytest.mark.slow
def test_a_weakly_broken_4mm_model_restricts_to_its_parent_on_the_pinned_branch(
        p41_substrate):
    """The owner's continuity criterion, measured — and it does not depend on δ.

    Take ``M`` = the paper's flat-band model in ``P4mm1' #99.164`` on a
    substrate (16 of 16; its ``C4`` has ``S(C4)† ≠ ±S(C4)``, so its own branch
    is *measured* and ``branch_tie`` is ``False``).  Add ``±δ`` to the two
    sites' on-site energies: ``H_M'`` differs from ``H_M`` by exactly ``δ``,
    ``C4`` and the diagonal mirrors are broken by it, and what is left is the
    axial ``mm2`` ``Pmm21' #25.58``, 8 of 8.  With ``δ = 1e-6``, a hundred times
    the declared detector tolerance ``1e-8``, the operations are **broken**
    rather than near, so nothing is symmetrized.

    ``realized_group(M')`` serves that ``mm2`` on the **default** branch, ``+1``,
    with ``branch_tie`` — every element is an involution and no gate can choose.
    The child's representation is then the parent's restriction **on the nose**:
    the two agree at roundoff -- 2.54e-16 on the development machine, 3.22e-16
    on an M1 with OpenBLAS -- the whole ``mm2`` coset included.  Until
    2026-09-10 they differed by exactly one real linear character of ``mm2``,
    ``λ = (E: +1, C2(z): −1, m(y): +1, m(x): −1)`` and the same on the four
    antiunitary operations: the reference-gauge offset was recorded and not
    applied.  Now both builds multiply their offsets into the matrices and
    ``λ`` is the identity.

    The deviation being roundoff is what makes the "continuity constant" of
    ``max‖D_M'(g) − λ(g)D_M(g)‖ ≤ C·δ`` zero: the representation is discrete
    signs times exact table lifts and does not vary with ``δ`` at all.  And it
    is not the orientation branch: ``orientation_branch=-1`` on the child gives
    the same matrices to roundoff (2.45e-16).
    """
    # `realized_group(tqc_models.p41_flat_band(), environment="substrate")`,
    # the session's shared record of that call
    parent = p41_substrate
    assert parent, str(parent)
    assert (parent.group.number_str, len(parent.validated.indices)) == ("99.164", 16)
    assert parent.rep._branch_tie is False, "a fourfold's lift settles its own branch"

    child = tqg.realized_group(_axially_split(1e-6), environment="substrate",
                               cross_check=False)
    assert child, str(child)
    assert (child.group.number_str, child.group.name) == ("25.58", "Pmm21'")
    assert len(child.validated.indices) == 8 and child.symmetrization is None
    assert child.rep._choice["orientation"] == 1
    assert child.rep._branch_tie is True
    assert child.rep._search["n_survivors"] == 2

    rho, defect, off = _relative_character(child.rep, child.group,
                                           parent.rep, parent.group)
    assert defect <= 1e-15, "every block is a multiple of the parent's, not merely close"
    assert off <= 1e-15, "and the multiple is a sign, to machine precision"
    assert _is_linear_character(rho, child.group), "and the signs are a character"
    named = {_sitechars.operation_name(s, child.group.lattice)
             + ("'" if s.time_reversal else ""): rho[int(s.ind)]
             for s in child.group.symmetries if int(s.ind) in rho}
    assert named == {"E": 1.0, "C2(z)": 1.0, "m(y)": 1.0, "m(x)": 1.0,
                     "E'": 1.0, "C2(z)'": 1.0, "m(y)'": 1.0, "m(x)'": 1.0}, \
        "the child IS the parent's restriction: both are the induced representation " \
        "of the content they return"
    assert all(o.determined_by != "declaration"
               for rep in (parent.rep, child.rep) for o in rep.characters.orbitals), \
        "nothing was declared, so neither record claims a declaration"

    worst = _deviation(child.rep, child.group, parent.rep, parent.group, rho)
    assert worst <= _FEW_ULP, f"roundoff, not a function of delta: got {worst:.3g}"

    # the branch is not what any of this is about
    other = _build._build_rep(child.rep.model, child.group, child.assignment,
                           validated=child.validated, orientation_branch=-1)
    assert other, str(other)
    assert other._choice["orientation"] == -1 and other._branch_tie is False
    assert max(float(np.abs(child.rep.matrices[i] - other.matrices[i]).max())
               for i in child.rep.indices) <= _FEW_ULP


def _square_s_layer(tx=0.30, ty=0.30, t2=0.07):
    """A spinless ``s`` orbital on a square layer; ``tx != ty`` breaks ``4mm`` to ``mm2``."""
    lattice = Lattice(np.array([[1.0, 0.0], [0.0, 1.0]]), [[0.0, 0.0]],
                      periodic_dirs=[0, 1])
    model = TBModel(lattice, spinful=False)
    model.set_onsite([0.0], mode="set")
    model.set_hop(tx, 0, 0, [1, 0])
    model.set_hop(ty, 0, 0, [0, 1])
    model.set_hop(t2, 0, 0, [1, 1])
    model.set_hop(t2, 0, 0, [1, -1])
    return model


@pytest.mark.slow
def test_the_spinless_4mm_model_never_tied_and_restricts_exactly():
    """The control: with no SU(2) factor there is no branch and no character.

    The same descent on a **spinless** square layer — one ``s`` orbital, an
    anisotropic nearest-neighbour hopping breaking ``4mm`` to ``mm2``.  Here
    ``S(g)`` is ``[[1]]``, so ``D`` and ``D†`` are the *same matrix* on every
    generator and the deduplication merges the two branches outright: one
    distinct candidate of the two built, ``branch_tie`` ``False``, and nothing
    about this model changed in wave R4e.

    And because there is no SU(2) sign for ``normalize_rep`` to choose, there is
    no global character between parent and child either: the child's
    representation is the parent's restriction at exactly **0.0**, on every
    operation of both cosets.  That is the isolation of the finding above — the
    character the spinful case carries is the sign of the spinor lift, not
    anything about the orientation branch.
    """
    parent = tqg.realized_group(_square_s_layer(), environment="substrate",
                                cross_check=False)
    assert parent, str(parent)
    assert (parent.group.number_str, len(parent.validated.indices)) == ("99.164", 16)
    assert parent.rep._branch_tie is False
    assert (parent.rep._search["n_candidates"], parent.rep._search["n_distinct"]) == (2, 1), \
        "both branches were built and they are the same matrices"

    child = tqg.realized_group(_square_s_layer(ty=0.31), environment="substrate",
                               cross_check=False)
    assert child, str(child)
    assert (child.group.number_str, len(child.validated.indices)) == ("25.58", 8)
    assert child.rep._branch_tie is False
    assert (child.rep._search["n_candidates"], child.rep._search["n_distinct"]) == (2, 1)

    rho, defect, off = _relative_character(child.rep, child.group,
                                           parent.rep, parent.group)
    assert defect <= 1e-15 and off <= 1e-15
    assert set(rho.values()) == {1.0}, "no character at all"
    assert _deviation(child.rep, child.group, parent.rep, parent.group, rho) == 0.0


# --- the induced representation, built from the tables and nothing else ------

def _induced_rep(model, group, assignment, indices):
    """``{ind: P(g) ⊗ S(g)}`` — the EBR the site content induces, from the tables.

    ``P(g)`` is the site permutation ``_assign_orbitals`` measured (one orbital
    per site here, so a plain permutation matrix — every operation of these
    models carries a site onto another with no lattice translation, so nothing
    convention-I has to appear in a ``k``-independent matrix), and ``S(g)`` is
    the operation's own ``sym.spinor_rotation`` in the pinned spinor branch,
    with the tables' antiunitary lift ``iσ_y conj(·)`` = ``spinor_rotation_TR``
    for the primed coset.  This is written here rather than taken from the
    package so that it is an **independent** statement of the convention: the
    read-off never enters it.

    Independent in the spinor lift and in the scalar, **not** in ``P(g)``: the
    site permutation is read from the same ``assignment.permutations`` the
    ansatz of ``_build_rep`` is assembled from, so it is a shared premise and a
    permutation-convention error in ``_assign_orbitals`` would reproduce itself
    on both sides.  It is also unfalsifiable on these fixtures, whose ``mm2``
    permutations are all involutions -- inverting one changes nothing.  The
    non-involutive case is covered by
    ``test_bloch_covariance_battery_on_a_permuted_wyckoff_orbit``, where the
    permutation is measured against the Hamiltonian rather than assumed.
    """
    perms = np.asarray(assignment.permutations, dtype=int)
    norb = int(model.norb)
    out = {}
    for pos, sym in enumerate(group.symmetries):
        ind = int(sym.ind)
        if ind not in indices:
            continue
        S = np.asarray(sym.spinor_rotation_TR if sym.time_reversal
                       else sym.spinor_rotation, dtype=complex)
        P = np.zeros((norb, norb), dtype=complex)
        for a in range(norb):
            P[int(perms[pos, a]), a] = 1.0
        out[ind] = np.kron(P, S)
    return out


def _ebr_1a_mm2(tx=0.30, ty=0.17, soc=0.0):
    """One spinful ``s`` orbital at the origin of a rectangular layer: ``mm2``, ``t_x ≠ t_y``."""
    sigma = [np.eye(2, dtype=complex),
             np.array([[0, 1], [1, 0]], dtype=complex),
             np.array([[0, -1j], [1j, 0]], dtype=complex)]
    lattice = Lattice(np.array([[1.0, 0.0], [0.0, 1.37]]), [[0.0, 0.0]],
                      periodic_dirs=[0, 1])
    model = TBModel(lattice, spinful=True)
    model.set_onsite([0.0 * sigma[0]], mode="set")
    model.set_hop(tx * sigma[0], 0, 0, [1, 0])
    model.set_hop(ty * sigma[0], 0, 0, [0, 1])
    if soc:
        # the mm2-symmetric Rashba pattern: i*lambda*(sigma_y, -sigma_x) on the
        # two axes, which keeps E, C2(z), m(x), m(y) and makes H complex
        model.set_hop(1j * soc * sigma[2], 0, 0, [1, 0])
        model.set_hop(-1j * soc * sigma[1], 0, 0, [0, 1])
    return model


def _ebr_2fold_mm2(u=0.27, tx=0.30, ty=0.17, tp=0.11, soc=0.0):
    """Two spinful ``s`` orbitals at ``(±u, 0)`` — a two-fold orbit of the same ``mm2``.

    ``C2(z)`` and ``m(x)`` exchange the two sites and ``m(y)`` fixes them, so the
    induced representation carries a genuine site permutation.
    """
    sigma = [np.eye(2, dtype=complex),
             np.array([[0, 1], [1, 0]], dtype=complex),
             np.array([[0, -1j], [1j, 0]], dtype=complex)]
    lattice = Lattice(np.array([[1.0, 0.0], [0.0, 1.37]]),
                      [[u, 0.0], [-u % 1.0, 0.0]], periodic_dirs=[0, 1])
    model = TBModel(lattice, spinful=True)
    model.set_onsite([0.0 * sigma[0], 0.0 * sigma[0]], mode="set")
    model.set_hop(tp * sigma[0], 0, 1, [0, 0])
    for orb in (0, 1):
        model.set_hop(tx * sigma[0], orb, orb, [1, 0])
        model.set_hop(ty * sigma[0], orb, orb, [0, 1])
        if soc:
            model.set_hop(1j * soc * sigma[2], orb, orb, [1, 0])
            model.set_hop(-1j * soc * sigma[1], orb, orb, [0, 1])
    return model


@pytest.mark.slow
@pytest.mark.parametrize("orbit,soc,environment", [("1a", 0.0, "substrate"),
                                                   ("2-fold", 0.05, "free-standing")])
def test_a_pure_ebr_is_the_induced_representation_in_the_tables_convention(
        orbit, soc, environment):
    """The owner's convention criterion, measured against the tables' own induction.

    Two pure EBR models of the axial ``mm2`` ``Pmm21' #25.58``: one spinful ``s``
    orbital at the origin (site symmetry ``mm2``, so the induced representation
    is ``S(g)`` itself), read on a substrate, and two at the two-fold position
    ``(±u, 0)`` (site symmetry ``{E, m(y)}``, so the induction carries a site
    permutation) with an ``mm2``-symmetric Rashba term, read free-standing, so
    that the answer is not an accident of a real Hamiltonian.
    ``D_ind(g) = P(g) ⊗ S(g)`` is built here from ``_assign_orbitals``' measured
    permutation and the operations' own ``sym.spinor_rotation``, with no
    read-off in it.

    Both cases are a **branch tie**: the group is all involutions, both branches
    close all three gates, and ``_build_rep`` serves ``+1`` by convention.
    Measured:

    * every block of the served ``D(g)`` is a multiple of ``D_ind(g)``, at
      proportionality defect **6.1e-17** — so the two differ by at most a
      scalar per operation;
    * that scalar is the **identity** on every operation, in both cases:
      the built representation **is** the induced one, at **2.4e-16** (one
      orbital) and **3.1e-16** (two);
    * the other branch gives the same matrices to 2.4e-16 / 3.7e-16, so the
      agreement is not the branch.

    Those digits are observations.  Each is the roundoff of an order-one matrix
    product whose exact value is ``0``, so each is asserted at a few ulp and
    nothing is lost by the room: were the served matrices off by a character,
    or by the orientation branch, the same differences would be ``O(1)``.

    **Until 2026-09-10 the scalar was not the identity** but the real linear
    character ``λ = (E: +1, C2(z): −1, m(y): −1, m(x): +1)``: the raw build's
    reference orbital measured ``A2`` against the tables' lift — its spin block
    on ``C2(z)`` and ``m(y)`` was *minus* ``S(g)`` — and the gauge offset that
    undoes it was recorded and not applied, so the record read ``A1`` over
    matrices that were the induced representation of ``A2``.  The owner's
    ruling: the served matrices are the induced representation in the tables'
    convention (spin up along +z, right-handed frame), so the offset is
    multiplied into the matrices.  The record still names it — ``B2``, ``mm2``'s
    name for that character, on the ``1a`` model, whose reference site group is
    the whole group — and declaring that label serves ``λ·D_ind``, the other
    sign, which is the witness that the declaration moves the matrices and the
    default no longer does.

    The global one-dimensional character is the one thing the hoppings cannot
    supply — every row of every one of these builds reads the convention, the
    four ``mm2`` characters standing behind it; at the two-fold position the
    site group is order 2 and a declaration can only fix the character on it,
    which is why the rows away from it stay conventional.

    The band representation the orbits induce is the tables' doublet at
    ``1a``, once for the one orbital and twice for the pair.  The same doublet
    at ``1b``, ``1c`` or ``1d`` has the same symmetry content and other stable
    real-space invariants, so the one orbital's table names no other sum; twice
    over the four positions' invariants agree, and the pair's table names the
    other three as not distinguished from it.
    """
    model = (_ebr_1a_mm2(soc=soc) if orbit == "1a" else _ebr_2fold_mm2(soc=soc))
    got = tqg.realized_group(model, environment=environment, cross_check=False)
    assert got, str(got)
    assert (got.group.number_str, got.group.name) == ("25.58", "Pmm21'")
    assert len(got.validated.indices) == 8

    rep = got.rep
    assert rep._choice["orientation"] == 1
    assert rep._branch_tie is True, "every element is an involution; no gate can choose"
    assert rep._search["n_survivors"] == 2
    assert "pinned by convention" in rep._search["branch_note"]
    assert all(o.determined_by != "declaration" for o in rep.characters.orbitals), \
        "the labelling is a declaration's to fix and none was made"
    (induces,) = rep.characters.band_representations
    assert [(m, position.split("(")[0]) for m, _name, position in induces.terms] == \
        [(1 if orbit == "1a" else 2, "1a")]
    assert [[position.split("(")[0] for _m, _name, position in terms]
            for terms in induces.alternatives] == \
        ([] if orbit == "1a" else [["1b"], ["1c"], ["1d"]])

    induced = _induced_rep(rep.model, got.group, got.assignment, set(rep.indices))
    measured, defect = {}, 0.0
    for ind, Dind in induced.items():
        r, res = _matrices._proportionality(np.asarray(rep.matrices[ind]), Dind)
        measured[ind], defect = complex(r), max(defect, float(res))
    assert defect <= 1e-15, f"a multiple of the induced block, not merely close: {defect:.3g}"
    lam, off = _snap(measured)
    assert off <= 1e-15, "and the multiple is a sign, to machine precision"
    assert _is_linear_character(lam, got.group)

    named = {_sitechars.operation_name(s, got.group.lattice): lam[int(s.ind)]
             for s in got.group.symmetries
             if int(s.ind) in lam and not s.time_reversal}
    expect = {"E": 1.0, "C2(z)": 1.0, "m(y)": 1.0, "m(x)": 1.0}
    assert named == expect, "the served representation IS the induced one, no character"
    # and the primed coset agrees -- counted first, or the loop would pass
    # vacuously if the antiunitary operations ever dropped out of ``lam``
    primed = [s for s in got.group.symmetries if int(s.ind) in lam and s.time_reversal]
    assert len(primed) == len(named) == len(expect), \
        "the primed coset reached lam: its 4 operations"
    for s in primed:
        assert lam[int(s.ind)] == expect[_sitechars.operation_name(s, got.group.lattice)]

    worst = max(float(np.abs(np.asarray(rep.matrices[ind]) - lam[ind] * Dind).max())
                for ind, Dind in induced.items())
    assert worst <= _FEW_ULP, \
        f"measured 2.4e-16 (1a) / 3.1e-16 (2-fold), got {worst:.3g}"

    other = _build._build_rep(rep.model, got.group, got.assignment,
                           validated=got.validated, orientation_branch=-1)
    assert other, str(other)
    shared = [i for i in rep.indices if i in other.matrices]
    assert max(float(np.abs(rep.matrices[i] - other.matrices[i]).max())
               for i in shared) <= _FEW_ULP, \
        "the branch is not what lambda is (measured 2.4e-16 / 3.7e-16)"

    if orbit == "1a":
        # the reference orbital's site group IS this group, so declaring B2 --
        # mm2's name for (C2(z): -1, m(y): -1, m(x): +1), the sign the raw spin
        # block carried on C2(z) and m(y) against the tables' lift -- serves
        # Ind(B2), the old matrices, which is what shows the default moved and
        # the declaration still does
        b2 = {"E": 1.0, "C2(z)": -1.0, "m(y)": -1.0, "m(x)": 1.0}
        b2_of = {int(s.ind): b2[_sitechars.operation_name(s, got.group.lattice)]
                 for s in got.group.symmetries if int(s.ind) in induced}
        declared = _build._build_rep(
            rep.model, got.group, got.assignment, validated=got.validated,
            characters={0: "B2"})
        assert declared, str(declared)
        assert [o.determined_by for o in declared.characters.orbitals][0] == "declaration"
        assert max(float(np.abs(np.asarray(declared.matrices[ind])
                                - b2_of[ind] * Dind).max())
                   for ind, Dind in induced.items()) <= _FEW_ULP, \
            "declaring B2 serves the B2-induced representation"


@pytest.mark.slow
@pytest.mark.parametrize("name,build,number,n_unitary,n_site,rows,other,other_rows", [
    ("P4/mmm 2g", tqc_models.p4mmm_s_2g, "123.340", 16, 8,
     ("-E1g↑G(2)@1a", "-E1u↑G(2)@1a"), "B1", ("-E2g↑G(2)@1a", "-E2u↑G(2)@1a")),
])
def test_a_spinful_s_orbital_off_the_mirror_planes_is_the_tables_induced_representation(
        name, build, number, n_unitary, n_site, rows, other, other_rows):
    """The lift question where it matters most: mirrors and inversion as transport.

    In a centrosymmetric group with mirrors, an orbital placed where no mirror
    passes has a site group without them, so the orbital type it is declared
    as cannot say how it transforms under the mirror or the inversion -- those
    operations carry the site to another site.  For a spinor that sign is
    nonetheless fixed, by how the spin is chosen to transform under inversion:
    it does not (``S(I·R) = S(R)``), which makes a spinful ``s`` orbital the
    *gerade* spinor at an inversion centre and the ``Ē1`` (not ``Ē2``) doublet
    at a ``4mm`` site.  Owner's point of 2026-09-10, and the reason the
    lowest-``l`` rule reads, between the sites an operation exchanges, the
    functions of the point group the orbitals read as when each is paired with
    its image (the labelling's frame key): only then is the default the
    induced representation ``P(g) ⊗ S(g)`` in the tables' convention on the
    nose.

    A group-averaged model (``fixture_models.p4mmm_s_2g``):

    * the served block of the reference orbital equals the tables' own lift
      ``sym.spinor_rotation`` on every unitary operation -- site and transport
      alike (8 of 16 operations move the site) -- at 16 ulp (measured 3.1e-16);
    * the content of all bands is the tables' EBR rows
      ``-E1g↑G(2)@1a + -E1u↑G(2)@1a`` for the ``2g`` orbit of ``P4/mmm``
      (``2g`` is not maximal; the ``1b`` pair is the same band representation,
      and the content does not move between ``z = 0.21`` and ``0.38``);
    * the *other* spinor is one declaration away and is the other rows, which
      is what shows the default is a convention and the declaration moves it:
      at ``2g`` it is ``B1``, the ``d_x²−y²``-type orbital, because ``Ē2`` of
      ``4mm`` differs from ``Ē1`` on the fourfold (``χ(C4) = ∓√2``) and not on
      the mirrors, where every spinor character vanishes -- declaring ``A2``
      (odd under the mirrors only) leaves the spinor ``Ē1`` and the content
      alone.

    The build's raw offset is ``A2g``, as on every layer: it is ``−1`` only on
    the twofolds and mirrors, where ``4/mmm``'s spinor characters vanish, so
    here the ruling moves the matrices (the mirror blocks) and no label -- the
    mutation "offset left out of the matrices" fails this test at the block
    comparison and nowhere in its content.  The mutation "``spin_bonds`` lifts
    an improper operation by its full matrix" cannot even build the model
    (``spinor_of`` refuses an improper matrix).
    """
    model = build()
    got = tqg.realized_group(model)
    assert got, str(got)
    assert got.group.number_str == number
    rep = got.rep
    assert all(o.determined_by == "lowest-l" for o in rep.characters.orbitals), \
        "the other spinor fits the hoppings as well; the s labelling is the lowest-l rule's"
    assert len(set(rep.characters.irreps)) == 1, "one orbit, one site label"

    # the served block against the tables' lift, transport included
    ref = int(rep._reference["orbital"])
    perm = np.asarray(got.assignment.permutations, dtype=int)
    site_orbs = got.assignment.site_orbitals
    ref_site = int(np.asarray(got.assignment.site_of_orbital)[ref])
    checked, moved, worst = 0, 0, 0.0
    for pos, s in enumerate(got.group.symmetries):
        ind = int(s.ind)
        if s.time_reversal or ind not in rep.matrices:
            continue
        img_site = int(perm[pos, ref_site])
        img = int(site_orbs[img_site][0])
        block = np.asarray(rep.matrices[ind])[np.ix_([2 * img, 2 * img + 1], [2 * ref, 2 * ref + 1])]
        worst = max(worst, float(np.abs(block - np.asarray(s.spinor_rotation)).max()))
        checked += 1
        moved += int(img_site != ref_site)
    assert (checked, moved) == (n_unitary, n_unitary - n_site)
    assert worst <= 16 * np.finfo(float).eps, f"{name}: served block is not the tables' lift: {worst:.3g}"

    # the content is the tables' own induction
    ebr = tqe.ebr_rows(got.group)

    def total(names):
        out = {}
        for n in names:
            key = next(k for k in ebr if k.startswith(n))
            for lab, v in ebr[key].items():
                if int(v):
                    out[str(lab)] = out.get(str(lab), 0) + int(v)
        return out

    table = tqw.hsp_table(rep, bands=int(model.nstate))
    assert table.complete and table.refusals == {}, name
    counts = {str(k): int(v) for k, v in table.counts.items()}
    assert counts == total(rows), name
    assert counts != total(other_rows), "the two spinor rows differ, so the check has teeth"

    # ... and the other spinor is the declared odd / A2 orbital
    declared = _build._build_rep(model, got.group, got.assignment, validated=got.validated,
                              characters={0: other})
    assert declared, str(declared)
    # the declaration fixes the character on the SITE group only; the group is
    # larger, so several global characters restrict to it, and the row it is
    # made about is the declaration's while the rest are measured against it
    assert declared.characters.orbitals[0].determined_by == "declaration"
    t2 = tqw.hsp_table(declared, bands=int(model.nstate))
    assert {str(k): int(v) for k, v in t2.counts.items()} == total(other_rows), name


# ---------------------------------------------------------------------------
# 3.  the antiunitary anchor
# ---------------------------------------------------------------------------

def test_antiunitary_anchor_catches_a_planted_minus_i(lin3g_rep):
    """Plant ``(-i) x D(C4)`` and watch every gate except the anchor let it through.

    This is gotcha #2 in the act.  The planted operator
    * is exactly as covariant as the true one (the equation is homogeneous),
    * satisfies the canonical norm ``D**4 = -1`` exactly, because ``(-i)**4 = 1``,
    * and rotates **every** sewing eigenvalue by ``-pi/2``, which relabels every
      corep at once — here ``E1b`` (angles +-pi/4) and ``E2b`` (+-3pi/4) trade
      places on all four multiplets simultaneously, self-consistently.
    Only ``T conj(D) T† = D`` sees it.  And ``T``'s own phase is a convention:
    its canonical norm ``T**2 = -1`` constrains none, and the record says so.
    """
    group, assignment, validated, rep = lin3g_rep
    model = rep.model
    c4 = next(ind for ind in rep.indices
              if rep._operation(ind).angle_str.strip() == "1/2 pi"
              and not rep._operation(ind).time_reversal
              and not rep._operation(ind).inversion)
    tr = next(ind for ind in rep.antiunitary
              if np.all(np.asarray(rep._operation(ind).rotation) == np.eye(3, dtype=int)))
    D_true = rep.matrices[c4]
    T = rep.matrices[tr]
    planted = -1j * D_true

    # (a) the canonical norm cannot see it
    assert np.abs(np.linalg.matrix_power(planted, 4) + np.eye(model.nstate)).max() < 1e-13
    # (b) neither can the covariance battery
    k = np.array([[0.13, -0.29], [0.41, 0.07]])
    H = np.asarray(model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    Hr = np.asarray(model.hamiltonian(k_pts=rep.transform_k(c4, k), flatten_spin_axis=True))
    assert np.abs(planted @ H @ planted.conj().T - Hr).max() <= GATE
    # (c) the anchor does
    assert np.abs(T @ np.conj(planted) @ T.conj().T - planted).max() > 1.0

    anchor = tqs.RepAnchor(antiunitary=T, rows=(0, 1), cols=(0, 1))
    rec = tqs.normalize_rep(planted, rep._operation(c4), group, anchor=anchor)
    assert rec, str(rec)
    assert rec.anchor_used
    assert rec.factor == pytest.approx(1j, abs=1e-12)
    assert np.abs(rec.matrix - D_true).max() < 1e-12
    assert rec.norm_residual < 1e-13
    assert rec.anchor_residual < 1e-13
    # the canonical norm allowed all four fourth roots; the anchor kept the two real
    # multiples of the true phase, and positivity picked between them
    assert len(rec.candidates) == 4
    assert len(rec.accepted) == 2

    # the corep relabelling the plant would have caused, displayed
    Hg = np.asarray(model.hamiltonian(k_pts=np.zeros((1, 2)), flatten_spin_axis=True))[0]
    _w, V = np.linalg.eigh(Hg)
    for lo in (0, 2, 4, 6):
        blk = V[:, lo:lo + 2]
        true_ang = np.sort(np.angle(np.linalg.eigvals(blk.conj().T @ D_true @ blk)) / np.pi)
        bad_ang = np.sort(np.angle(np.linalg.eigvals(blk.conj().T @ planted @ blk)) / np.pi)
        # every true multiplet is a Kramers pair at +-pi/4 (E1b) or +-3pi/4 (E2b)
        assert abs(true_ang.sum()) < 1e-9
        assert (np.abs(np.abs(true_ang) - 0.25).max() < 1e-9
                or np.abs(np.abs(true_ang) - 0.75).max() < 1e-9)
        # the plant rotates every eigenvalue by -pi/2 ...
        shifted = np.sort(((true_ang - 0.5 + 1.0) % 2.0) - 1.0)
        assert np.allclose(shifted, bad_ang, atol=1e-9)
        # ... which destroys the +- pairing itself: the multiplet no longer reads as
        # any corep at all, and it does so on all four multiplets at once.
        assert abs(bad_ang.sum()) > 0.1

    # the time-reversal operator's own phase is phase-blind
    t_rec = tqs.normalize_rep(rep.matrices[rep.antiunitary[0]],
                              rep._operation(rep.antiunitary[0]), group)
    assert t_rec, str(t_rec)
    assert t_rec.order == 2
    assert t_rec.required == -1                  # T**2 = -1 for a spinful group
    assert t_rec.norm_informative is False       # ... and it constrains no phase


def test_normalize_rep_refuses_a_matrix_that_represents_nothing(lin3g_rep):
    """A matrix whose ``D**n`` is not a multiple of the identity is refused, not rescaled.

    Note how hard this is to trigger by accident, which is the lesson: rescaling
    one row by ``i`` leaves ``D**4 = -1`` intact because ``i**4 = 1``.  The
    perturbation has to be a phase that is *not* a fourth root of unity.
    """
    group, assignment, validated, rep = lin3g_rep
    c4 = rep._generators[0]
    benign = rep.matrices[c4].copy()
    benign[0, :] *= 1j
    assert np.abs(np.linalg.matrix_power(benign, 4)
                  + np.eye(rep.model.nstate)).max() < 1e-13     # invisible

    junk = rep.matrices[c4].copy()
    junk[0, :] *= np.exp(1j * np.pi / 3.0)
    out = tqs.normalize_rep(junk, rep._operation(c4), group)
    assert not out
    assert out.gate == "canonical_norm"
    assert out.context["order"] == 4
    assert out.context["required"] == -1


def test_normalize_rep_refuses_when_the_anchor_disagrees(lin3g_rep):
    """An operator no ``lambda`` can reconcile with the anchor is refused.

    ``D(C4)`` with its two spin blocks made equal — ``e^{-i pi/4} O (x) 1``
    instead of ``O (x) e^{-i pi/4 s_z}`` — still satisfies the canonical norm
    ``D**4 = -1`` exactly, and is still unitary.  The anchor demands
    ``lambda**2 = i``, i.e. ``lambda = +-e^{i pi/4}``, which is not a fourth
    root of unity: no phase satisfies both, and that is the verdict.
    """
    group, assignment, validated, rep = lin3g_rep
    c4 = rep._generators[0]
    tr = rep.antiunitary[0]
    T = rep.matrices[tr]
    twist = np.kron(np.eye(rep.model.norb), np.diag([1.0, -1j]))
    spoiled = rep.matrices[c4] @ twist
    assert np.abs(spoiled.conj().T @ spoiled - np.eye(rep.model.nstate)).max() < 1e-12
    assert np.abs(np.linalg.matrix_power(spoiled, 4)
                  + np.eye(rep.model.nstate)).max() < 1e-13    # the norm is satisfied

    anchor = tqs.RepAnchor(antiunitary=T, rows=(0, 1), cols=(0, 1))
    out = tqs.normalize_rep(spoiled, rep._operation(c4), group, anchor=anchor)
    assert not out
    assert out.gate == "antiunitary_anchor"
    assert len(out.context["candidates"]) == 4


# ---------------------------------------------------------------------------
# 5.  the Bloch operator and its covariance battery
# ---------------------------------------------------------------------------

def test_bloch_covariance_battery_on_a_permuted_wyckoff_orbit(inline_rep):
    """Gotcha #3's pythtb-convention case: a 2c orbit exchanged by C4.

    Both forms of the identity are gated: at the **unwrapped** image (where
    convention I hides the positions entirely) and at the image folded back
    into the cell (where they reappear as ``diag(e^{2 pi i n.tau})``).
    """
    model, group, assignment, validated, rep = inline_rep
    assert rep, str(rep)
    assert rep._search["n_survivors"] == 1
    # one orbital per site, so there is nothing at any site to bind: every
    # multiplet is one orbital, and every internal factor is the tables' lift
    assert rep.multiplets.determined_by == "hamiltonian"
    assert rep.multiplets.by_site == (((0,),), ((1,),), ((2,),))
    assert rep.multiplets.dims == ((1,), (1,), (1,))

    # the orbit really is permuted, with a nonzero integer lattice shift
    pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    c4 = rep._generators[0]
    perm = assignment.permutations[pos[c4]]
    assert perm.tolist() != list(range(len(perm))), "C4 must move the 2c sites"
    assert np.abs(assignment.lattice_shift[pos[c4]]).max() == 1

    rng = np.random.default_rng(11)
    k = rng.uniform(-0.5, 0.5, size=(8, 2))
    H = np.asarray(model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    worst_unwrapped = worst_wrapped = 0.0
    for ind in rep.indices:
        op = rep._operation(ind)
        kp = rep.transform_k(ind, k)
        # fold the image back by a *random nonzero* reciprocal-lattice vector, so the
        # row phases are genuinely exercised (in convention I, H(k+n) != H(k))
        n = rng.integers(1, 3, size=kp.shape) * rng.choice([-1, 1], size=kp.shape)
        kw = kp - n
        assert np.abs(np.asarray(model.hamiltonian(k_pts=kw, flatten_spin_axis=True))
                      - np.asarray(model.hamiltonian(k_pts=kp,
                                                     flatten_spin_axis=True))).max() > 1e-3
        Hp = np.asarray(model.hamiltonian(k_pts=kp, flatten_spin_axis=True))
        Hw = np.asarray(model.hamiltonian(k_pts=kw, flatten_spin_axis=True))
        src = H.conj() if op.time_reversal else H
        U = rep.bloch_operator(ind, k)
        worst_unwrapped = max(worst_unwrapped,
                              float(np.abs(U @ src @ np.conj(U.transpose(0, 2, 1)) - Hp).max()))
        Uw = rep.bloch_operator(ind, k, k_out=kw)
        worst_wrapped = max(worst_wrapped,
                            float(np.abs(Uw @ src @ np.conj(Uw.transpose(0, 2, 1)) - Hw).max()))
    assert worst_unwrapped <= GATE
    assert worst_wrapped <= GATE
    assert rep.product_residual < 1e-13


def test_bloch_operator_row_phases_follow_the_preimage(inline_rep):
    """The self-check fires when the matrix and the site permutation disagree.

    And a wrapped image must differ from the rotated ``k`` by an integer
    reciprocal-lattice vector, or the Bloch operator refuses to be built.
    """
    model, group, assignment, validated, rep = inline_rep
    c4 = rep._generators[0]
    D = rep.matrices[c4].copy()
    D[[1, 2], :] = D[[2, 1], :]              # keep it unitary, break the permutation
    with pytest.raises(ValueError, match="integer-shift identity|preimages"):
        tqs.bloch_operator(model, group, assignment, rep._operation(c4), D,
                           np.array([0.2, 0.3]))
    with pytest.raises(ValueError, match="integer reciprocal-lattice"):
        rep.bloch_operator(c4, np.array([0.2, 0.3]), k_out=np.array([0.1, 0.1]))


def test_embedding_phase_sign_is_pinned_at_third_lattice_vectors(km_odd):
    """Kane-Mele's orbitals sit at thirds, and there the sign is measurable.

    The correct sign closes the wrapped covariance at machine precision; the
    opposite one misses by O(1).  This is the position-general content of the
    Bloch operator, and it is a **gate**, never a chooser: pythtb fixes the
    convention and ``groups`` supplies the operation's action on positions.
    """
    group, assignment, validated, rep = _rep_bundle(km_odd)
    assert rep, str(rep)
    tau = np.asarray(km_odd.orb_vecs)
    assert np.abs(3 * tau - np.round(3 * tau)).max() < 1e-9
    assert np.abs(2 * tau - np.round(2 * tau)).max() > 0.1

    rng = np.random.default_rng(2)
    k = rng.uniform(-0.5, 0.5, size=(6, 2))
    H = np.asarray(km_odd.hamiltonian(k_pts=k, flatten_spin_axis=True))
    tau_state = np.repeat(tau, km_odd.nspin, axis=0)
    good = bad = 0.0
    for ind in rep.indices:
        op = rep._operation(ind)
        kp = rep.transform_k(ind, k)
        n = rng.integers(1, 3, size=kp.shape) * rng.choice([-1, 1], size=kp.shape)
        kw = kp - n
        Hw = np.asarray(km_odd.hamiltonian(k_pts=kw, flatten_spin_axis=True))
        src = H.conj() if op.time_reversal else H
        U = rep.bloch_operator(ind, k, k_out=kw)
        good = max(good, float(np.abs(U @ src @ np.conj(U.transpose(0, 2, 1)) - Hw).max()))
        flipped = np.exp(-2j * np.pi * (n @ tau_state.T))[:, :, None] * rep.matrices[ind][None]
        bad = max(bad, float(np.abs(flipped @ src @ np.conj(flipped.transpose(0, 2, 1))
                                    - Hw).max()))
    assert good <= GATE
    assert bad > 1e-2


# ---------------------------------------------------------------------------
# 6.  the surface phase A3 consumes
# ---------------------------------------------------------------------------

def test_sewing_operators_commute_with_H_at_high_symmetry_points(lin3g_rep):
    """What A3 needs: ``U(g;k) H(k) U† = H(k)`` for every little-group operation.

    At ``M``, where every operation is in the little group and the images wrap
    by an integer reciprocal-lattice vector ``in_little_group`` returns.
    """
    group, assignment, validated, rep = lin3g_rep
    model = rep.model
    name, k = "M", [0.5, 0.5]
    little = rep.little_group(k)
    assert little, f"{name} has an empty little group"
    reference = tqg.little_group(group, k)
    assert reference, str(reference)
    ref_inds = {int(o.ind) for o in reference.unitary} | {int(o.ind) for o in
                                                          reference.antiunitary}
    assert set(little) <= ref_inds, "our little group must be inside the adapter's"

    Hk = np.asarray(model.hamiltonian(k_pts=np.array([k]), flatten_spin_axis=True))[0]
    for ind in little:
        U = rep.sewing_operator(ind, k)
        assert not isinstance(U, Refusal), str(U)
        src = Hk.conj() if rep._operation(ind).time_reversal else Hk
        assert np.abs(U @ src @ U.conj().T - Hk).max() <= GATE
        assert np.abs(U.conj().T @ U - np.eye(model.nstate)).max() < 1e-12

    c4 = rep._generators[0]
    ok, G = rep.in_little_group(c4, k)
    assert ok
    assert G.dtype.kind == "i"
    assert np.abs(rep.transform_k(c4, np.array(k))[0] - np.array(k) - G).max() < 1e-12


def test_the_representation_commutes_with_H_at_k_points_the_build_never_saw(lin3g_rep):
    """Covariance at three k-points the build never used, through ``rep.covariance``.

    A representation that only closed the battery of k-points it was gated on
    would show up here.
    """
    group, assignment, validated, rep = lin3g_rep
    fresh = rep.covariance(k_pts=np.array([[0.11, 0.37], [-0.23, 0.41], [0.05, -0.19]]))
    assert max(fresh.values()) <= GATE


# ---------------------------------------------------------------------------
# 7.  what a representation prints
# ---------------------------------------------------------------------------

def test_repr_smoke(lin3g_rep, lin3g_rep_enum):
    """The repr names the group, the multiplets and the residuals.

    How the one survivor was found is the search record's own business
    (``search["strategy"]``), not the line a representation prints: the
    read-off and the enumeration return one representation, and print it alike.
    The group's symbol fixes how many operations it has, so no count is printed.
    """
    group, assignment, validated, rep = lin3g_rep
    text = repr(rep)
    assert text.startswith(f"SpaceGroupRep({group.name} #{group.number_str}, ")
    assert " ops" not in text and "antiunitary" not in text
    assert " multiplets" in text and "covariance<=" in text and "product=" in text
    assert "1 of" not in text and "read off" not in text
    assert "NormalizedRep" in repr(rep._normalization[rep._generators[0]])
    assert repr(lin3g_rep_enum[3]) == text


# ---------------------------------------------------------------------------
# 10.  the table anchor: what fixes the gauge when the group has no T
# ---------------------------------------------------------------------------
#
# Everything above rests on the antiunitary anchor for the last root of unity.
# A group built with ``include_TR=False`` has no such operation, and what is
# left over is then not a sign but a **linear character** of the group: on #80's
# Z4 the canonical norm allows lam in {1, i, -1, -i} on the four-fold generator,
# all four propagate to exact representations of the group (the twist preserves
# the entire factor system), and the positivity convention picks between them by
# a tie in Re broken by Im.  The tables close it, because the internal factor the
# ansatz is built from is an SU(2) lift and SU(2) is pseudo-real.  Owner's ruling
# (2026-08-25): "The ambiguity is fixed by the convention used to construct the
# representations, and this is FIXED by the tables / BCS / Herring.  There should
# be no refusal.  If we get a refusal, it means there is a bug in our application
# of the convention.  It should ERROR."


@pytest.fixture(scope="module")
def i41_soc_unitary():
    """``(group, assignment, validated, rep)`` for ``i41_soc`` on the UNITARY tables."""
    model = tqc_models.i41_soc("generic", lam=0.4)
    group = _spacegroup._group_of(model, include_TR=False)
    assert group and group.number_str == "80", str(group)
    return _rep_bundle(model, group=group, nk=4, n_battery=5)


def _reference_block(rep, ind):
    """``(rows, cols)`` of the reference orbital's block under one operation."""
    n = _nspin(rep)
    img = int(np.abs(np.asarray(rep.matrices[ind])[:, 0]).argmax()) // n
    return tuple(range(img * n, (img + 1) * n)), tuple(range(n))


def test_the_positivity_tie_is_what_left_the_unitary_gauge_free(i41_soc_unitary):
    """The mechanism, measured on the one operation that carries it.

    ``normalize_rep`` picks the residual root of unity by *positivity* on the
    reference block's principal entry: ``Re > 0``, ties broken by ``Im > 0``.
    For a spinful four-fold the principal entry is ``S(4+)[0,0] = e^{-i pi/4}``,
    so ``lam = 1`` and ``lam = i`` give ``e^{-i pi/4}`` and ``e^{+i pi/4}`` —
    **the same** ``Re = 0.7071`` — and the tie-break hands it to ``i``.  That is
    the whole of the bug: not a wrong measurement anywhere, one coin flip in a
    convention, and it twists the whole representation by the linear character
    ``lambda(E, 2_z, 4+, 4-) = (1, -1, +i, -i)`` of #80's Z4.

    With time reversal in the group the same call keeps only the *real*
    multiples and the tie never arises; here the group ships no ``T``, and
    ``RepAnchor.internal`` — the operation's own ``sym.spinor_rotation`` —
    is what says the ratio must be real.  Both branches are run below on the
    identical matrix, so the before and the after are one measurement.

    **Where no table lift stands behind the reference** the anchor has nothing
    to read: a multi-dimensional multiplet's block is measured, so positivity
    alone picks among all ``n`` roots, and nothing makes its pick a
    representation of the group.  Measured on the spinful ``s`` pair at ``6c``
    of ``R-3m``, flattened as a ``wannier90`` spinor import delivers it and
    asked for its unitary single-valued group, where each spinor is one
    measured doublet: on ``S3``, the rotoinversion of order six, positivity
    picks a root that is not a sign (measured ``e^{i pi/3}``), a cube root of
    unity away from the two roots that multiply, so ``(S3 C2')^2 = E`` misses by
    ``|1 - e^{2 pi i/3}| = sqrt(3)`` -- the defect ``rep_not_multiplicative``
    was returned at.  The multiplication table
    chooses among the six roots instead (``product_anchored``), the twofold
    keeps positivity's sign, and the representation returned is the
    single-valued one the spinor class of ``-3m`` allows: every spinor ``E``
    of ``3m``, ``Ē1`` rephased by ``-1`` on the threefold, its site blocks
    closing as a single-valued representation.
    """
    group, _assignment, _validated, rep = i41_soc_unitary
    gen = rep._generators[0]
    op = rep._operation(gen)
    rows, cols = _reference_block(rep, gen)
    S = np.asarray(op.spinor_rotation, dtype=complex)
    entry = np.asarray(rep.matrices[gen])[rows[0], cols[0]]
    assert entry == pytest.approx(np.exp(-1j * np.pi / 4), abs=1e-12)
    assert (1 * entry).real == pytest.approx((1j * entry).real, abs=1e-12), "the tie itself"
    assert (1j * entry).imag > 0 > (1 * entry).imag, "and which way it broke"

    loose = tqs.normalize_rep(rep.matrices[gen], op, group,
                              anchor=tqs.RepAnchor(rows=rows, cols=cols))
    assert loose, str(loose)
    assert loose.factor == pytest.approx(1j, abs=1e-12), "the unpinned convention's answer"
    assert len(loose.candidates) == 4 and len(loose.accepted) == 4
    assert loose.anchor_used is False and loose.table_anchored is False

    tight = tqs.normalize_rep(rep.matrices[gen], op, group,
                              anchor=tqs.RepAnchor(rows=rows, cols=cols, internal=S))
    assert tight, str(tight)
    assert tight.factor == pytest.approx(1.0, abs=1e-12), "the tables' answer"
    assert tight.table_anchored is True and tight.anchor_used is False
    assert len(tight.candidates) == 4 and len(tight.accepted) == 2, \
        "the anchor keeps the real pair; the reference orbital picks inside it"
    assert all(abs(complex(lam).imag) < 1e-12 for lam in tight.accepted)
    # (this last line re-reads ``normalize_rep``'s own arithmetic rather than the
    #  build's: ``tight.factor`` was just asserted to be 1 and ``tight.matrix`` is
    #  ``factor * polar(input)`` on an already-unitary input, so it says
    #  ``polar(D) == D``.  The load-bearing pair is ``loose.factor``/``tight.factor``.)
    assert np.abs(tight.matrix - rep.matrices[gen]).max() < 1e-13, \
        "the build already landed there, which is the point of the anchor"

    # where no table lift stands behind the reference: the table itself chooses
    flat, _basis = tqspin.spinless_presentation(tqc_models.r3m_s_6c())
    trig = _spacegroup._group_of(flat, spinor=False, include_TR=False)
    assert (trig.number_str, trig.size, trig.spinor) == ("166", 12, False), str(trig)
    _trig, _tassign, tvalid, srep = _rep_bundle(flat, group=trig, nk=4)
    assert len(tvalid.indices) == 12 and srep, str(srep)
    assert srep.antiunitary_operator[0] is None and srep.multiplets.dims == ((2,), (2,)), \
        "each spinor is one measured doublet, and the group ships no T"
    six = next(int(i) for i in srep._generators if srep._normalization[int(i)].order == 6)
    two = next(int(i) for i in srep._generators if srep._normalization[int(i)].order == 2)
    assert srep._normalization[six].product_anchored is True
    assert srep._normalization[six].table_anchored is False
    assert len(srep._normalization[six].accepted) == 6, "nothing narrowed the sixth roots"
    assert srep._normalization[two].product_anchored is False, "positivity's sign multiplies"

    D6, D2 = np.asarray(srep.matrices[six]), np.asarray(srep.matrices[two])
    eye = np.eye(D6.shape[0])
    assert np.abs(D6 @ D2 @ D6 @ D2 - eye).max() < 1e-12, "(S3 C2')^2 = E"
    states, ref = srep.multiplets._chan_states, int(srep._reference["multiplet"])
    img = next(c for c in range(len(states))
               if np.abs(D6[np.ix_(states[c], states[ref])]).max() > 0.5)
    unpinned = tqs.normalize_rep(D6, srep._operation(six), trig, anchor=tqs.RepAnchor(
        rows=tuple(states[img]), cols=tuple(states[ref])))
    lam = complex(unpinned.factor)
    assert unpinned.table_anchored is False and len(unpinned.accepted) == 6
    assert abs(lam ** 6 - 1.0) < 1e-12 and abs(lam ** 2 - 1.0) == pytest.approx(
        np.sqrt(3.0), abs=1e-12), "positivity's own pick on the same matrix: not a sign"
    assert np.abs((lam * D6) @ D2 @ (lam * D6) @ D2 - eye).max() == pytest.approx(
        np.sqrt(3.0), abs=1e-9), "and what it misses the table by"

    assert srep.product_residual < 1e-12
    assert max(srep.residuals.values()) <= srep._search["gate"]
    assert srep.characters.irreps == ("E",) * 4
    assert {(o.wyckoff, o.site_symmetry) for o in srep.characters.orbitals} == {("6c", "3m")}
    assert srep.characters.residual < 1e-12, "the site blocks close single-valued"


def test_the_table_anchor_errors_rather_than_refusing(monkeypatch):
    """The owner's ruling in code: a gauge the tables fix cannot come back as a result.

    The forcing is the one the ruling names — make the matrices the search
    builds disagree with the ``spinor_rotation`` the group ships — and it has to
    be a corruption **every gate is blind to**, or the gate catches it first and
    the anchor is never reached.  A global unitary conjugation
    ``S(g) -> V S(g) V†`` is exactly that, and it is blind for a reason in each
    case rather than by luck:

    * covariance — ``i41_atomic``'s hoppings are spin-independent, so
      ``[H(k), s_i] = 0`` and ``O (x) (V S V†)`` is exactly as covariant as
      ``O (x) S``;
    * the internal-factor gate — it measures each block's *proportionality* to
      the lift the build was handed, which is the conjugated one on **every**
      operation, so every block is proportional to it exactly;
    * the canonical norm — ``(V S(4+) V†)**4 = V S(4+)**4 V† = -1``;
    * the product table — conjugation is an isomorphism, so the factor system
      is untouched.

    What is not blind is the table anchor, and deliberately so: it reads
    ``sym.spinor_rotation`` off the **shipped datum** rather than through the
    build's own lift, so it is literally "what the tables say".  With
    ``V`` the Hadamard the reference block comes out ``(1 - i·s_x)/sqrt2`` where
    the tables ship ``diag(e^-iπ/4, e^+iπ/4)``: not a multiple of it at all
    (best ratio 0.5, defect 0.707 against 1e-09), so no phase can put the
    matrices in the tables' lift.

    What comes back is an ``InternalError`` and **not** a ``Refusal``: a
    refusal is a measurement about the model that a caller can act on, and
    there is nothing here for a caller to supply.

    Corrupting **one** operation instead is now caught a gate earlier, and that
    is the R4b closure rather than a loss.  ``_internal_defect`` measures the
    ansatz on every block of every kept unitary operation, not only on the
    channels an operation fixes, and ``I4_1``'s two screws fix none — their site
    permutations are ``(0 1)(2 4 3 5)`` and its inverse, with no fixed point —
    so on the fixed channels alone there was nothing to measure on either of
    them, which is the hole this forcing used to walk through.  Corrupt the
    generator's lift alone and its powers inherit the corruption while their own
    lifts do not: the gate reads 1.0 against 1e-09 and refuses
    ``rep_internal_factor``.  That is a Refusal and belongs to the model —
    matrices that commute with ``H`` but are not the matrices of *these*
    operations — which is exactly the distinction the ruling draws.

    The same statement at the level of the measurement, on the branch no public
    path reaches: a reference block proportional to ``S`` by a phase the
    canonical norm's candidates cannot make real is unpinnable in the same way,
    and says so with the phases it was offered.
    """
    model = tqc_models.i41_atomic()
    group = _spacegroup._group_of(model, include_TR=False)
    assert group and group.number_str == "80", str(group)
    assignment = _tables._assign_orbitals(model, group)
    validated = _measure._validate_group(model, group, nk=4)

    true_internal = _matrices._internal_from_table
    gen = 3                                    # the four-fold screw, #80's only generator

    # the two screws fix no site, which is why the fixed-channel gate was blind
    perms = np.asarray(assignment.permutations)
    assert not any(int(perms[p][s]) == s for p in (2, 3) for s in range(perms.shape[1]))

    # 1.  one operation's lift corrupted: the R4b gate catches it, and refuses
    def one_operation(sym):
        S = true_internal(sym)
        return S @ np.diag([1.0, -1.0]).astype(complex) if int(sym.ind) == gen else S

    with pytest.MonkeyPatch.context() as mp:
        patch_everywhere(mp, "_internal_from_table", one_operation)
        record = {}
        caught = _build._build_rep(model, group, assignment, validated=validated,
                                n_battery=5, _search_record=record)
    assert isinstance(caught, Refusal) and not caught
    assert caught.gate == "rep_internal_factor"
    assert caught.value == pytest.approx(1.0, abs=1e-12) and caught.threshold == 1e-9
    assert record["search"]["killed_by"]["internal_factor"] == 1

    # 2.  the whole lift conjugated: blind to every gate, and the anchor ERRORS
    V = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)

    def conjugated(sym):
        return V @ true_internal(sym) @ V.conj().T

    patch_everywhere(monkeypatch, "_internal_from_table", conjugated)
    with pytest.raises(InternalError) as excinfo:
        _build._build_rep(model, group, assignment, validated=validated, n_battery=5)

    message = str(excinfo.value)
    assert not isinstance(excinfo.value, Refusal), "the ruling: it must ERROR"
    assert isinstance(excinfo.value, RuntimeError)
    assert f"operation #{gen}" in message
    assert "the tables fix this gauge completely" in message
    assert "bug in this package's application of that convention" in message
    assert "not a multiple of the operation's own sym.spinor_rotation" in message

    # 3.  the helper-level branch: a ratio no offered phase can make real
    S = np.diag([-1j, 1j])
    ok = _matrices._table_anchored((1.0 + 0j, 1j, -1.0 + 0j, -1j), S, S,
                             op_ind=3, order=4, zeta=-1)
    assert ok == (1.0 + 0j, -1.0 + 0j), "the real pair, in the order it was offered"

    with pytest.raises(InternalError) as excinfo:
        _matrices._table_anchored((np.exp(1j * np.pi / 3),), S, S, op_ind=3, order=4, zeta=-1)
    assert "real multiple" in str(excinfo.value)
    assert "one-dimensional character of the group" in str(excinfo.value)

    with pytest.raises(InternalError) as excinfo:
        _matrices._table_anchored((1.0 + 0j,), np.diag([-1j, -1j]), S,
                            op_ind=3, order=4, zeta=-1)
    assert "not a multiple of the operation's own sym.spinor_rotation" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 10b.  the anchor against the DECLARATION protocol (2026-08-25)
# ---------------------------------------------------------------------------
#
# Section 10 measures the anchor on the DEFAULT declaration.  The owner's
# principle is larger than that: **the representation is fixed by the
# declaration of what the orbitals are**, the ``characters=`` machinery is where
# that declaration is written, and everything else follows through Frobenius
# (``lam (x) Ind(rho) = Ind(lam|_site (x) rho)``).  So the anchor is only right
# if it *realizes* that protocol -- for every declaration, not the default one.
# That a declared twist is applied once, after the anchor, and that a
# declaration the Hamiltonian contradicts is a Refusal where an unpinnable gauge
# is an InternalError, are ``test_multiorbital.py`` section (f3)'s.  The seam at
# this file's altitude is the one path the anchor does not take: it SKIPS a
# vanishing reference block, and nothing may leak in there.


@pytest.mark.slow
def test_the_skipped_reference_block_is_a_branch_covariance_has_already_killed():
    """The one path the anchor does **not** take, and the proof nothing leaks through it.

    ``normalize_rep`` applies the table anchor only where the reference block is
    nonzero.  A candidate whose block vanishes is the ``g^-1`` branch of an
    operation that *moves* the reference channel: ``P6_2``'s three-fold cycles
    the ``3a`` channels ``0 -> 1 -> 2``, so ``D(g)`` has its reference block at
    ``[1, 0]`` and ``D(g)^dagger`` has ``0`` there -- measured **0.0** here, and
    asserted at ``normalize_rep``'s own zero-block floor (it skips the anchor at
    ``|block|.max() <= max(atol, 1e-10)``), because the polar SVD does not
    preserve structural zeros: 22 of this very matrix's 306 structurally-zero
    entries come back nonzero, the largest at 5.0e-16.

    Handed that matrix, ``normalize_rep`` skips the anchor and the leak is real
    and visible: all six sixth-roots survive, positivity falls back to ``D``'s
    own principal entry, and the phase it returns is ``e^{i pi/3}`` -- a complex
    number, exactly the class of twist the anchor exists to remove.

    It never reaches the anchor in a real build.  Both ``g^-1`` candidates are
    killed by **covariance** at ``0.0996`` -- the first of the three gates, two
    gates before ``normalize_rep`` runs at all -- and the same is true on every
    fixture in this suite (``killed_by`` is ``covariance`` everywhere except
    inversion-symmetric Kane-Mele, where it is ``internal_factor``, still ahead
    of normalization).  The surviving branch's reference block is the *moved*
    one, and the anchor fires on it (``table_anchored``).
    """
    model = tqc_models.p62_atomic()
    group = _spacegroup._group_of(model, include_TR=False)
    assert group and group.number_str == "171", str(group)
    assignment = _tables._assign_orbitals(model, group)
    validated = _measure._validate_group(model, group, nk=4)
    # `strategy="enumerate"`: what is regressed here is the enumeration's kill
    # record -- that the `g^-1` candidates it proposes die on covariance, two
    # gates before `normalize_rep` runs.  The read-off never proposes them on
    # this model (the reference channel moves, so the `g^-1` pattern has nothing
    # in the reference block to read a phase off), which is a different
    # statement that no test pins.
    rep = _build._build_rep(model, group, assignment, validated=validated, n_battery=5,
                         strategy="enumerate")
    assert rep and rep.antiunitary_operator[0] is None, str(rep)

    gen = int(rep._generators[0])
    p = next(i for i, s in enumerate(group.symmetries) if int(s.ind) == gen)
    cperm = _matrices._orbital_permutation(assignment, p, model)
    ref = int(rep._reference["multiplet"])
    img = int(cperm[ref])
    assert (ref, img) == (0, 1) and int(cperm[img]) == 2, "the reference channel moves"
    rows = tuple(range(img * 2, (img + 1) * 2))
    cols = tuple(range(ref * 2, (ref + 1) * 2))

    D = np.asarray(rep.matrices[gen])
    S = np.asarray(rep._operation(gen).spinor_rotation, dtype=complex)
    kept = D[np.ix_(np.asarray(rows), np.asarray(cols))]
    assert float(np.abs(kept).max()) > 0.5, "the surviving branch's block is the moved one"
    assert rep._normalization[gen].table_anchored is True, "and the anchor fired on it"

    skipped = D.conj().T[np.ix_(np.asarray(rows), np.asarray(cols))]
    assert float(np.abs(skipped).max()) < 1e-10, \
        "the g^-1 branch's block, zero to the anchor's own zero-block floor"
    loose = tqs.normalize_rep(D.conj().T, rep._operation(gen), group,
                              anchor=tqs.RepAnchor(rows=rows, cols=cols, internal=S))
    assert loose, str(loose)
    assert loose.table_anchored is False, "no block, no anchor"
    assert len(loose.candidates) == 6 and len(loose.accepted) == 6
    assert complex(loose.factor) == pytest.approx(np.exp(1j * np.pi / 3), abs=1e-9), \
        "and this is the complex phase that would have come back"

    minus = [c for c in rep._search["candidates"] if c["orientation"] == -1]
    assert minus and all(c["killed_by"] == "covariance" for c in minus)
    assert all(c["covariance"] == pytest.approx(0.0996, abs=5e-4) for c in minus), \
        "two gates before normalize_rep ever sees them"


# ---------------------------------------------------------------------------
# 10c.  where the anchor's premise is FALSE: the model and the tables disagree
#       about valuedness (two owner rulings, 2026-09-01)
# ---------------------------------------------------------------------------
#
# The anchor of section 10 above used to fire on both mismatches and blame
# *this package*: `_table_anchored` raises InternalError saying "a bug in this
# package's application of that convention", and that verdict rests on a premise
# -- the internal factor is a pseudo-real SU(2) lift written in the tables the
# group ships -- which either mismatch falsifies.  No real phase exists, and
# nothing is a package bug.  Both are refusals now, for different reasons:
#
# * **spinless model, DOUBLE-valued tables** -- a real thing to want, and a
#   docket item of its own rather than a magnetic-program rider.  It says the
#   orbitals carry half-integer spin themselves: a spinor Wannier basis, which
#   is how pythtb's `wannier90` importers deliver a spinor calculation (always a
#   spinless model with its spin partners in a canonical order -- non-magnetic, groups
#   already in scope), or one without local time-reversal symmetry, which is the
#   magnetic case and the only half that waits on the magnetic program.  What is
#   missing either way is a double-valued *site* declaration.
# * **spinful model, SINGLE-valued tables** -- not a deferred feature but an
#   invalid ask.  The owner: "nspin==2 should always invoke spinor=True at this
#   level (and at any level), theres no case where that can break."  Refused at
#   `_group_of` and again here.

@pytest.fixture(scope="module")
def spinless_asked_double_valued():
    """``(model, group, assignment, validated)``: the reproduction, up to ``_build_rep``.

    Haldane at ``phi = 0.5`` -- two spinless ``s``-like orbitals on a honeycomb
    -- asked for the **double-valued** tables of the cell it sits in.
    """
    from pythtb.models import haldane

    model = haldane(0.0, -1.0, 0.15, 0.5)
    assert int(model.nspin) == 1 and not model.spinful, "the model has no spin axis"
    group = _spacegroup._group_of(model, spinor=True, include_TR=False)
    assert group, str(group)
    assignment = _tables._assign_orbitals(model, group)
    assert assignment, str(assignment)
    return model, group, assignment, _measure._validate_group(model, group)


@pytest.mark.slow
def test_a_spinless_model_asked_with_double_valued_tables_refuses(
        spinless_asked_double_valued):
    """The reproduction, and the ruling in code: a falsy ``Refusal``, never an error.

    Before the gate this call reached ``_table_anchored`` and raised
    ``InternalError``.  The six phases that message listed for the six-fold
    -- the sixth roots of ``-1``, ``e^{i pi (2k+1)/6}`` -- are exactly the
    **double-valued characters of C6**, i.e. precisely the set a site
    declaration would have picked one from.  The crash message was the
    specification of the missing feature, not a bug report: nothing pinned the
    phase because nothing had been declared, and there is no way to declare it.

    The reason names what it waits on, and since the spinor-basis declaration
    landed (2026-09-02) the non-magnetic half is no longer waiting: a
    ``wannier90`` spinor import arrives exactly like this, and ``spin=`` is
    what it is missing.  Haldane's two orbitals are **not** such a basis --
    they are two genuinely spin-free s orbitals on a honeycomb -- so this
    refusal is still the right answer for this model, and the message now says
    which ask would not be refused.
    """
    model, group, assignment, validated = spinless_asked_double_valued
    assert len(validated.indices) == 12 and len(validated.broken) == 12, \
        "the Haldane mass and flux refute half of P6/mmm; the group still stands"

    try:
        out = _build._build_rep(model, group, assignment, validated=validated)
    except InternalError as exc:            # pragma: no cover - the defect
        pytest.fail("the anchor's premise is false here, so its verdict -- 'a bug in "
                    f"this package' -- is wrong; it must refuse instead: {exc}")

    assert isinstance(out, Refusal) and not out, str(out)
    assert out.gate == "rep_double_valued_spinless"
    assert "half-integer local spin" in out.reason
    assert "wannier90" in out.reason, "the non-magnetic use case is named"
    assert "`spin=`" in out.reason, "and the route that now exists for it"
    assert 'order="interleaved"' in out.reason and "`pairs=`" in out.reason
    assert "an unpaired spinor orbital" in out.reason, "what is still unsupported"
    assert "double-group site declarations" in out.reason
    assert out.context["nspin"] == 1 and out.context["spinful"] is False
    assert out.context["number_str"] == "191" and out.context["spinor"] is True

    # structural, so neither the stated multiplets nor time reversal changes the
    # verdict: on the grey group the antiunitary anchor would otherwise answer
    # `antiunitary_anchor`, a true measurement of a false setup.
    stated = _build._build_rep(model, group, assignment, validated=validated,
                            multiplets=[[(0,)], [(1,)]])
    assert stated.gate == "rep_double_valued_spinless", str(stated)
    grey = _spacegroup._group_of(model, spinor=True, include_TR=True)
    assert grey and grey.number_str == "191.234", str(grey)
    with_tr = _build._build_rep(model, grey, _tables._assign_orbitals(model, grey))
    assert with_tr.gate == "rep_double_valued_spinless", str(with_tr)


def test_a_spinful_model_cannot_be_asked_for_single_valued_tables():
    """The owner's second ruling: ``nspin == 2`` invokes the double group, always.

    *"nspin==2 should always invoke spinor=True at this level (and at any
    level), theres no case where that can break."*  So this is an invalid ask,
    not a deferred feature, and the constructor is where it dies -- both
    presentations of it, unitary and grey.

    ``_build_rep`` enforces the same rule on a group that arrived another way:
    the model's own **spinless** decorated cell is a legitimate ``spinor=False``
    ask, and the spinful model handed in beside it raises ``ValueError``, a
    malformed argument, rather than being blamed on this package by the
    anchor.
    """
    from pythtb.models import kane_mele

    model = kane_mele(1.0, 1.0, 0.6, 0.0)
    assert int(model.nspin) == 2 and model.spinful

    for include_TR in (False, True):
        out = _spacegroup._group_of(model, spinor=False, include_TR=include_TR)
        assert isinstance(out, Refusal) and not out, str(out)
        assert out.gate == "single_valued_spinful"
        assert "no single-valued question to put to it" in out.reason
        assert out.context["nspin"] == 2 and out.context["spinor"] is False

    # the second enforcement point, reached without the constructor's help
    m3 = _layers._slab_presentation(model)
    cell, cell_typat = _layers._decorated_cell(m3, [0, 0])
    assert int(cell.nspin) == 1, "_decorated_cell builds a spinless geometry cell"

    group = _spacegroup._group_of(cell, typat=cell_typat, spinor=False, include_TR=False)
    assert group and group.spinor is False, str(group)

    assignment = _tables._assign_orbitals(m3, group, typat=[0, 0])
    validated = _measure._validate_group(m3, group, nk=6)
    with pytest.raises(ValueError, match="the model is spinful"):
        _build._build_rep(m3, group, assignment, validated=validated, n_battery=3)


# ---------------------------------------------------------------------------
# 11.  the read-off: the discrete choices measured off the detector's own space
#      (performance phase, wave P2, 2026-09-02)
# ---------------------------------------------------------------------------
#
# `_build_rep` used to pin its ansatz `D(g) = O(g) (x) S(g)` by ENUMERATION --
# one phase from `phases` per (generator, channel class) except the reference
# one, times the orientation branch, each candidate through the three gates.
# That is 2-512 candidates on the fixtures above and `2**46` on a
# wannier90-scale model, which is the cliff this section is about.
#
# The detector has already solved `X H(k) = H(Rk) X` for every operation and
# `ValidatedGroup._bases` carries the whole solution space; a realizable ansatz
# is an ELEMENT of that space, so the choices are read off it instead.  Where
# `dim >= 2` -- a conserved operator -- the member the detector kept is
# generally NOT of ansatz form, and the read-off projects the space onto ansatz
# form before it reads.
#
# What must be true, and is what this section measures:
#
#   * the read-off and the enumeration return the SAME representation -- the
#     same discrete choices, bit-identical matrices, the same gate residuals;
#   * the read-off never ACCEPTS anything: its candidate goes through the same
#     three gates and the same uniqueness bookkeeping, and a read the gates do
#     not confirm falls back to the enumeration, recorded with its reason;
#   * an enumeration too large to run is a refusal (`rep_search_too_large`),
#     never an hour of arithmetic nobody asked for.


def _readoff_pair(model, *, group=None, group_kw=None, typat=None, nk=6, **build_kw):
    """``(validated, readoff_rep, enumerated_rep)`` for one model, one measurement."""
    group = _spacegroup._group_of(model, **(group_kw or {})) if group is None else group
    assert group, str(group)
    assignment = _tables._assign_orbitals(model, group, typat=typat)
    assert assignment, str(assignment)
    validated = _measure._validate_group(model, group, nk=nk)
    common = dict(validated=validated, **build_kw)
    return (validated,
            _build._build_rep(model, group, assignment, strategy="readoff", **common),
            _build._build_rep(model, group, assignment, strategy="enumerate", **common))


def _lin3g_spinor_basis():
    """lin3g re-presented as the spinless model wannier90 would deliver.

    A channel is then a **declared** ``(up, dn)`` pair rather than a native spin
    axis, and the read-off has to run on that: same ansatz, same SU(2) internal
    factor, different channel bookkeeping.
    """
    flat, basis = tqspin.spinless_presentation(tqc_models.lin3g(), order="interleaved")
    return dict(model=flat, group_kw={"spinor": True}, spin=basis)


#: ``_build_rep``'s default phase set — the two signs its enumeration tries on
#: every fixture below, and the roots of unity their read ratios are read as.
_OFFERED = (1 + 0j, -1 + 0j)

#: The fixtures the read-off is measured against the enumeration on: spinful
#: and spinless, one channel per orbital (``lin3g``), a degenerate solution space
#: (``km_rashba0``, whose conserved ``s_z`` makes every space two-dimensional), a
#: measured two-dimensional channel whose internal factor is the reference
#: channel's own block rather than the group's table (``p4232_doublet``), and a
#: declared spinor basis (``lin3g_spinor_basis``).
_READOFF_FIXTURES = ("km_rashba0", "lin3g", "lin3g_spinor_basis", "p4232_doublet")


@pytest.fixture(scope="module")
def readoff_pairs(lin3g_rep, lin3g_rep_enum, p4232_decorated):
    """``{name: (validated, readoff_rep, enumerated_rep)}``, one measurement each.

    lin3g's read-off and enumeration are this module's own two builds of it, and
    the ``P4_232`` doublet's read-off is the session's shared build of the
    decorated cell (``_validate_group(nk=5)``, ``_build_rep(n_battery=4)``): the
    same calls ``_readoff_pair`` makes.
    """
    _group, _assignment, validated, readoff = lin3g_rep
    model, group, assignment, p_validated, p_readoff = p4232_decorated
    p_enumerated = _build._build_rep(model, group, assignment, validated=p_validated,
                                  n_battery=4, strategy="enumerate")
    return {"lin3g": (validated, readoff, lin3g_rep_enum[3]),
            "km_rashba0": _readoff_pair(tqc_models.kane_mele_inversion()),
            "p4232_doublet": (p_validated, p_readoff, p_enumerated),
            "lin3g_spinor_basis": _readoff_pair(**_lin3g_spinor_basis())}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_READOFF_FIXTURES))
def test_the_readoff_returns_the_enumerations_representation(readoff_pairs, name):
    """Same discrete choices, same matrices to ``0.0``, same gate residuals.

    This is the whole claim of the wave: ``strategy="readoff"`` is a cheaper way
    to *find* the candidate the enumeration would have found, not a different
    answer -- including on a degenerate solution space, where the read-off
    projects the space onto ansatz form before it reads.

    And every read ratio is within **roundoff** of a sign, per (generator,
    class).  ``search["ratios"]`` is the block ratio against ``S(g)`` divided by
    the reference class's, and ``search["defects"]`` is how far that block was
    from being a multiple of ``S(g)`` at all.  The read-off reads a ratio
    within ``1e-6`` of a root of unity as that root; what that code does not
    guarantee, and is asserted here, is how far the ratio sat from it: every
    ratio sits at roundoff from its sign, and every defect under the
    internal-factor gate -- the condition the read-off defers on.
    """
    _validated, ro, en = readoff_pairs[name]
    assert ro, str(ro)
    assert en, str(en)

    assert ro._search["strategy"] == "readoff", ro._search["fallback_reason"]
    assert en._search["strategy"] == "enumerate"
    assert 1 <= ro._search["n_candidates"] <= 2, "one candidate per orientation branch"
    assert en._search["n_candidates"] == en._search["n_enumeration"]

    assert ro._choice == en._choice, "the same signs and the same orientation branch"
    assert ro._generators == en._generators and tuple(ro._classes) == tuple(en._classes)
    worst = max(float(np.abs(np.asarray(ro.matrices[i]) - np.asarray(en.matrices[i])).max())
                for i in en.indices)
    assert worst == 0.0, f"bit-identical matrices, got {worst:.3g}"

    for ind in en.indices:
        assert ro.residuals[ind] == en.residuals[ind]
    assert ro.product_residual == en.product_residual
    assert ro._search["internal"] == en._search["internal"]
    assert ro._search["product"] == en._search["product"]
    assert ro._search["n_survivors"] == en._search["n_survivors"] == 1

    ref_cls = int(ro._reference["orbit"])
    ratios, defects = ro._search["ratios"], ro._search["defects"]
    assert ratios and set(ratios) == set(defects)

    n_cls = int(np.asarray(ro._classes).max()) + 1
    assert set(ratios) == {(int(g), c) for g in ro._generators for c in range(n_cls)}

    for (g_ind, cls), rho in ratios.items():
        assert defects[(g_ind, cls)] <= 1e-9, "the internal-factor gate's own number"
        snapped = min(_OFFERED, key=lambda w: abs(rho - w))
        assert abs(rho - snapped) <= 1e-12, \
            f"{(g_ind, cls)}: read {rho!r}, snapped to {snapped}"
        if cls == ref_cls:
            assert snapped == 1, "the reference class supplies the phase, so it reads +1"
    assert max(defects.values()) < 1e-12, f"machine level, got {max(defects.values()):.2g}"
    assert ro._search["ansatz_defect"] < 1e-11


@pytest.mark.slow
def test_an_enumeration_larger_than_max_candidates_is_refused_not_run(lin3g_rep):
    """``rep_search_too_large``: a resource ceiling, named and measured.

    Refusing is the point -- an enumeration of ``2**26`` candidates at a few
    milliseconds each is a day of arithmetic, and returning it as "still
    working" is not a result.  The ceiling only bites when an enumeration is
    actually needed: the same call on the default read-off, which never
    enumerates, builds.  Where nothing can be read -- a validated group kept
    without its solution spaces, the same operations surviving -- the build
    defers to the enumeration, which builds the representation the read-off
    reads, bit for bit, and records why it ran; past the ceiling, the
    ceiling's refusal says what the read-off could not use.
    """
    group, assignment, validated, rep = lin3g_rep

    out = _build._build_rep(rep.model, group, assignment, validated=validated,
                         strategy="enumerate", max_candidates=8)
    assert not out and out.gate == "rep_search_too_large", str(out)
    assert out.value == 32 and out.threshold == 8
    ctx = out.context
    assert (ctx["n_generators"], ctx["n_classes"], ctx["n_free"]) == (2, 3, 4)
    assert ctx["n_candidates"] == 32 == 2 * 2 ** ctx["n_free"], \
        "two orientation branches, and a sign on each free choice"
    assert ctx["fallback_reason"] is None, "nothing deferred; the enumeration was asked for"
    assert "32 candidates" in out.reason.replace(",", "")

    # the read-off does not enumerate, so the same ceiling is no obstacle
    fine = _build._build_rep(rep.model, group, assignment, validated=validated,
                          max_candidates=1)
    assert fine, str(fine)
    assert fine._search["strategy"] == "readoff" and fine._choice == rep._choice
    assert fine._search["n_enumeration"] == 32, "what the enumeration would have run"

    # nothing to read defers to the enumeration, which builds what the read-off
    # reads and records why it ran
    lean = _measure._validate_group(rep.model, group, nk=6, keep_intertwiners=False)
    assert lean._intertwiners == {} and lean._bases == {}
    assert set(rep.indices) <= set(lean.indices), "the same operations survive"
    deferred = _build._build_rep(rep.model, group, assignment, validated=lean)
    assert deferred, str(deferred)
    s = deferred._search
    assert s["strategy"] == "enumerate" and s["requested"] == "readoff"
    assert s["n_candidates"] == s["n_enumeration"] == 32
    assert "no measured intertwiner" in s["fallback_reason"]
    assert s["readoff"]["branches"][1]["ok"] is False
    assert s["ratios"] == {} and np.isnan(s["ansatz_defect"]), \
        "no read was accepted, so nothing read stands for this build"
    assert deferred._choice == rep._choice
    assert max(float(np.abs(np.asarray(deferred.matrices[i])
                            - np.asarray(rep.matrices[i])).max())
               for i in rep.indices) == 0.0, "the representation the read-off reads"

    # ... and when the read-off defers, the ceiling is what it defers into
    stuck = _build._build_rep(rep.model, group, assignment, validated=lean,
                           max_candidates=1)
    assert not stuck and stuck.gate == "rep_search_too_large", str(stuck)
    assert stuck.value == 32 and stuck.threshold == 1
    assert "no measured intertwiner" in stuck.context["fallback_reason"]
    assert "The read-off deferred to it because" in stuck.reason


# ---------------------------------------------------------------------------
# 11b.  the wannier90-scale anchor: a hundred orbitals, twenty-seven classes
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
    """A Fu-Kane-Mele supercell with its folding degeneracy lifted, and its typing.

    The same construction ``test_detector.py`` measures the eigen-sift route on:
    a random **inversion-symmetric** on-site potential breaks the small cell's
    translations and the star degeneracy while leaving inversion exact.  The
    sites are then typed by inversion orbit -- the bare supercell's *positions*
    still carry the small translations, so what ``_group_of`` reads off it is the
    **lattice supergroup** (``Fd-3m1' #227.129`` at ``lattice_index`` 8 and 27;
    before wave R4a it was refused ``non_primitive_cell``), and the typing says
    the truth about this Hamiltonian rather than asserting a symmetry -- as well
    as being much the cheaper route: 0.2 s against 20.3 s at n = 108, since the
    supergroup carries 2592 operations.
    """
    model = tqc_models.fu_kane_mele_z2("strong").make_supercell(sc, to_home=True)
    perm = _inversion_permutation(model)
    rng = np.random.default_rng(seed)
    e = rng.normal(size=model.norb)
    e = 0.5 * (e + e[perm])                       # exactly symmetric under inversion
    for i in range(model.norb):
        model.set_onsite(lift * float(e[i]), i, mode="add")
    return model, [int(min(i, int(perm[i]))) for i in range(model.norb)]


@pytest.fixture(scope="module")
def sc108_bundle():
    """A degeneracy-lifted 3x3x3 Fu-Kane-Mele supercell (n = 108), group and all.

    ``orientation_branch=+1`` is used with it below, and that is a statement
    about the convention rather than about this wave: the only generator is
    inversion, a Hermitian involution, so its two branches build the identical
    unitary matrices and differ only in the sign of ``D(T)`` -- a twist all
    three gates are blind to.  Since wave R4e the default build settles that tie
    itself (``branch_tie``, +1 by convention), so the restriction is no longer
    needed here; it is kept because the test below is about the **read-off
    against the enumeration** and a restricted search is the cleanest way to
    hold the candidate count fixed while that is measured.
    """
    model, typat = _lifted_supercell([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    # The bare cell's own reading -- the index-27 lattice supergroup, 2592
    # operations -- is not measured here: it costs 20.3 s, and what this fixture
    # is for is the read-off at n = 108, which needs the typed group.  The same
    # reading on the 2x2x2 and 3x1x1 cells is `tests/test_lattice_descent.py`'s.
    group = _spacegroup._group_of(model, typat=typat)
    assert group and group.number_str == "2.5", str(group)
    assignment = _tables._assign_orbitals(model, group, typat=typat)
    assert assignment, str(assignment)
    validated = _measure._validate_group(model, group, nk=3)
    assert len(validated.indices) == 4, str(validated)
    return model, group, assignment, validated


@pytest.mark.slow
def test_the_readoff_builds_what_the_enumeration_must_refuse(sc108_bundle):
    """n = 108, 27 channel classes: 2**26 candidates by enumeration, one by read-off.

    The cliff this section is about, at wannier90 scale and without wannier90
    data.  A 3x3x3 Fu-Kane-Mele supercell in ``P-11'`` has 54 orbitals in 27
    inversion orbits, so the enumeration's choice set is
    ``2**(1 generator x 26 free classes) = 67,108,864`` -- refused in
    milliseconds rather than run for a day.  The read-off reads all 26 phases
    off one measured intertwiner and builds in a fraction of a second, at
    machine-level covariance.

    Those 54 orbitals sit one to a site, so there is nothing at any site to
    glue: the layout is the one channel per orbital there is to have, and every
    internal factor is the tables' own lift.
    """
    model, group, assignment, validated = sc108_bundle
    assert model.nstate == 108 and model.norb == 54

    common = dict(validated=validated, orientation_branch=+1)
    refused = _build._build_rep(model, group, assignment, strategy="enumerate", **common)
    assert not refused and refused.gate == "rep_search_too_large", str(refused)
    assert refused.value == 2 ** 26 == 67_108_864 and refused.threshold == 2 ** 20
    assert (refused.context["n_generators"], refused.context["n_classes"]) == (1, 27)

    started = time.perf_counter()
    rep = _build._build_rep(model, group, assignment, strategy="readoff", **common)
    seconds = time.perf_counter() - started
    assert rep, str(rep)
    assert rep._search["strategy"] == "readoff"
    assert rep._search["n_candidates"] == 1 and rep._search["n_enumeration"] == 2 ** 26
    assert len(rep._choice["signs"]) == 26
    assert rep.multiplets.determined_by == "hamiltonian" and all(
        d == 1 for dims in rep.multiplets.dims for d in dims), \
        "one orbital per site: one multiplet per orbital, and no block to measure"
    # ``<= GATE`` would restate ``assert rep`` (the winner closed that gate to be
    # served at all), so the covariance is pinned where it landed instead
    assert max(rep.residuals.values()) < 1e-14, \
        f"measured 5.5e-16; got {max(rep.residuals.values()):.3g}"
    assert rep.product_residual < 1e-12
    assert rep._search["ansatz_defect"] < 1e-10
    assert max(rep._search["defects"].values()) < 1e-11
    print(f"\nn = 108: read-off build {seconds:.2f} s "
          f"(measured 0.24 s), covariance {max(rep.residuals.values()):.1e}")
    assert seconds < 30.0, "measured 0.24 s; this bound is for a regression to minutes"
