r"""Wave R1: **realized means realized** — the unitary descent in ``realized_group``.

A cell group is an upper bound.  Before this wave ``realized_group`` refused
(``cell_group_not_realized``) whenever the Hamiltonian broke part of it and left
the caller to describe a better cell; now it **lowers** the cell to the
operations that were measured to hold and serves that group, so nothing
downstream — corep labels, EBR rows, indicators, SRSIs — is ever read in a group
the model does not have.  ``descend=False`` restores the refusal, to the letter.

What the mechanism is, and what it is not:

* the kept set is **known** before the descent starts (``validated.indices``
  minus the antiunitary operations), so nothing here searches for a group;
* a cell carrying exactly that set is built — the parent cell's own points plus
  the orbit of **one generic point** under the kept set, as one extra species —
  because only the identification step (spglib + ``irrep`` on a point set) can
  *name* the subgroup, hand back its setting transformation and stamp the
  ``.ind`` order every table lookup keys on;
* a generic point has trivial stabilizer, so an operation preserving its orbit
  belongs to the kept set — and the identification is **verified** against that
  set operation by operation anyway, never trusted: a cell read as another group
  is rebuilt from the next of a few generic points, and the package raises
  ``InternalError`` when none of them is read as the kept set;
* the model is never touched.  What comes back is the original presentation and
  the group it realizes; the orbit lives on ``.cell``, which is geometry.

The tests below are the three claims that makes: the two measured descents
(Fu–Kane–Mele's bond distortion in 3D, the P41' lattice's broken mirrors in 2D)
**agree with the decoration route** that says the same thing before any
Hamiltonian is measured; every fixture whose operations are all obeyed comes
back untouched; and the verification raises where it must.

Wave R2 adds the second measurement the descent reads, and with it the
horizontal mirror.  The detector answers about a **k-action class**: on a slab
every class is the pair ``{g, σ_h·g}``, the two pose the identical intertwiner
problem, the in-plane partner satisfies it, and both members are marked
``"undetermined"``.  No presentation repairs that,
because a layer's ``H(k)`` does not depend on ``k_z``.  What separates the pair
is the **tensor structure** of the representation, ``D(g) = O(g) ⊗ S(g)``: the
site permutation is the identity on a flat layer either way, and ``S(σ_h·g)`` is
a different SU(2) element from ``S(g)``.  So the operations the detector leaves
undetermined get the ansatz test ``_build_rep``'s read-off already applies to a
generator, per operation, and the ones it refutes leave the kept set exactly as
a detector discrepancy does (``refuted_by == "tensor structure"``).  Kane–Mele at
``rashba = 0.25`` read ``environment="free-standing"`` therefore **descends**
``P6/mmm1' #191.234 → P6mm1' #183.186``, naming all 24 operations with
``R[2,2] = −1`` — the last section of this file.

Read free-standing, the cell is the slab point set *as given*, with no
decoration at all, so it keeps ``σ_h`` and everything composed with it.  The
Hamiltonian then does all the selecting, and every 2D fixture whose realized
group is smaller than its geometry's reaches that group by a **descent** where
it used to reach it by construction: ``lin3g`` ``#123.340 → #99.164``,
Kane–Mele ``#187.210 → #156.50``, Po–Watanabe–Vishwanath
``#191.234 → #183.186``, each of them a free-standing record of the baseline
gate, and the flat Rashba slab of section 5 here.
The served group, its size, the Wyckoff letters, the corep labels and the SRSIs
are unchanged in every case; what is new is that ``.descent`` is a record rather
than ``None`` and the provenance stamp reads ``"descended"``.

Two of the wave's own gates bear on the models here.  The tensor test holds
every operation the detector leaves undetermined to its own spin lift, so of
each ``σ_h`` fiber the Hamiltonian breaks one member survives: the chiral
``P41'`` lattice of section 2 read free-standing keeps ``(E, C4, C2, C4⁻¹)``
and refutes ``S4`` and ``S4⁻¹`` by tensor structure, whose partners' lifts
``S(C4)† = −S(σ_h)·S(C4)`` would have carried them on the other branch; and
where a site's orbitals are one multiplet wider than one orbital, whose block
the two members share, the lift is asked of the whole solution space, which
separates them there too (the last section).  Where the survivors still do
not close — an operation on the wrong side of a declared tolerance — the
unique largest subgroup inside them
is served with the rest recorded ``refuted_by == "closure"``, and where **two**
subgroups tie at the largest order the tie rule keeps the first — on a layer
the most operations that keep its side, then the most rotations, then the
highest-order rotation, then its axis — with the operations only the other
carries marked ``refuted_by == "tie"``.
That model is asked here the question it answers without the rule —
``environment="substrate"``, which since wave R4c is a ``dim_r = 2`` model's
own default and the reading ``examples/tqc_srsi_p4_flatband.py`` uses — and the
paper's own group, corep contents and SRSIs come back exactly as this file has
always pinned them.
"""

import numpy as np
import pytest
from pythtb import Lattice, TBModel
from pythtb.models import kane_mele

from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc import symmetry as tqs
from pythtqc.refusal import InternalError, Refusal
from pythtqc import _descent
from pythtqc import _layers
from pythtqc import _measure
from pythtqc import _multiplets
from pythtqc import _spacegroup
from pythtqc import _tables

import tqc_models as tqm                                            # noqa: F401
from conftest import FKM_BOND, fkm_regime, patch_everywhere


def anisotropic_stack(tx=-1.0, ty=-0.6, tz=-0.4):
    """The same square layer stacked on a tetragonal lattice: a 3D crystal.

    The cell allows ``P4/mmm1' #123.340``, 32 operations; the Hamiltonian obeys
    the 16 of ``Pmmm1' #47.250``.  Every broken operation is ``"no symmetry
    matrix"``.
    """
    m = TBModel(Lattice(np.diag([1.0, 1.0, 1.6]), [[0.0, 0.0, 0.0]],
                        periodic_dirs=[0, 1, 2]), spinful=False)
    m.set_onsite([0.0])
    for t, R in ((tx, [1, 0, 0]), (ty, [0, 1, 0]), (tz, [0, 0, 1])):
        m.set_hop(t, 0, 0, R)
    return m


#: A chiral crystal field: one C4 orbit of a generic in-plane point, above and
#: below the plane as two species.  It keeps the fourfold axis and breaks every
#: vertical mirror, as the next-nearest hoppings of ``p41_flat_band(chiral=True)``
#: do (0.12 and 0.05 there; ``examples/tqc_srsi_p4_flatband.py`` breaks the same
#: mirrors in its hoppings, at 0.03 and 0.0125).
_PINWHEEL = ((0.3, 0.1), (-0.1, 0.3), (-0.3, -0.1), (0.1, -0.3))
CHIRAL_CELL = tuple([([x, y, 0.05], "top") for x, y in _PINWHEEL]
                    + [([x, y, -0.05], "bot") for x, y in _PINWHEEL])


# ---------------------------------------------------------------------------
# 1.  the 3D descent: a bond distortion no symmetry finder can see
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fkm_phases(realized):
    """The strong and weak distorted Fu-Kane-Mele regimes, descended and decorated.

    The decorated half is the session's shared record of the bond-centre
    decoration, the call the invariant modules make.
    """
    out = {}
    for name in ("strong", "weak"):
        down = tqg.realized_group(fkm_regime(name)(), nk=4, characters={0: "s"})
        told = realized(fkm_regime(name), decorations=FKM_BOND, characters={0: "s"})
        assert down and told, str(down if not down else told)
        out[name] = (down, told)
    return out


@pytest.mark.slow
def test_the_distorted_diamond_descends_to_the_group_it_obeys(fkm_phases):
    """``#227.129 -> #166.98``, 72 of 96 broken, every one of them a *proof*.

    A bond distortion moves no atom, so ``_group_of`` reads the full diamond
    group whatever the hoppings do.  The Hamiltonian obeys 12 of its 48 unitary
    operations (and 12 of its 48 antiunitary ones), and the cell built from
    their generic orbit is identified as ``R-3m1' #166.98`` — the same group one
    species at the bond midpoint gives from the geometry alone.  The broken
    operations are named in the generator frame ``characters=`` uses, a
    fourfold screw or a glide with its translation in the cell:
    ``{C4(z)|1/4,1/4,1/4}``.
    """
    for name, (down, _told) in fkm_phases.items():
        d = down.descent
        assert d is not None, name
        assert d["from"] == {"number_str": "227.129", "name": "Fd-3m1'", "size": 96}
        assert d["to"] == {"number_str": "166.98", "name": "R-3m1'", "size": 24}
        assert d["iterations"] == 1 and d["dropped_by"] == {"hoppings": 72}
        assert len(d["dropped"]) == 72 and len(d["cell"]["orbit"]) == 12
        assert {b["refuted_by"] for b in d["dropped"]} == {"no symmetry matrix"}
        assert {b["name"] for b in d["dropped"]} >= {
            "{C4(z)|1/4,1/4,1/4}", "{m(x)|1/4,1/4,1/4}", "S4(z)", "{C4(z)|1/4,1/4,1/4}'"}
        assert d["cell"]["generic_point"] == _descent._DESCENT_POINT   # all three directions
        assert len(down.validated.indices) == down.group.size == 24
        assert not down.validated.broken
        assert down.group.provenance == down.rep.group.provenance == "descended"
        assert ("descended from #227.129 (72 of 96 coset representatives of the lattice "
                "translations broken") in repr(down)
        # the orbit is on the CELL; the model is the one that was handed in
        assert down.model is down.source and down.cell is not down.model
        assert int(down.cell.norb) == 14 and list(down.cell_typat[:2]) == [0, 0]
        assert set(down.cell_typat[2:]) == {"descent"}
        assert down.decoration is None, "the descent's own species are not a decoration"


@pytest.mark.slow
def test_the_descent_and_the_decoration_agree_on_the_strong_and_weak_phases(fkm_phases):
    """Two independent routes to the same group, content and class.

    The decoration says *what* the distortion is (a crystal field at the
    strengthened bond's midpoint) from the geometry alone; the descent says
    *that* it is there, by refuting 72 operations on the Hamiltonian.  They are
    built by different code from different inputs, so their agreeing is a
    measurement.  Asserted on the strong and weak regimes, whose classes in
    ``X_BS(#166.98) = Z2 x Z4`` are two different nontrivial ones; the trivial
    regime's class is the baseline gate's.

    The decorated record says what was added: one caller-supplied species at
    the bond midpoint, on top of the model's own orbitals, with no
    re-presentation of the model and no height of the substrate recipe, which
    never ran.
    """
    classes = {}
    for name, (down, told) in fkm_phases.items():
        assert told.descent is None, f"{name}: the decorated cell is already right"
        assert told.group.provenance == "realized"
        assert (told.group.number_str, told.group.sg_number) == ("166.98", 166)
        assert len(told.validated.indices) == told.group.size == 24
        assert told.model is told.source, "no re-presentation: only the cell changed"
        assert told.cell is not told.model and told.cell.norb == told.model.norb + 1
        assert told.cell_typat == [0, 0, "bond"] and told.typat == [0, 0]
        assert told.decoration["typat"] == ("bond",) and told.decoration["n_atoms"] == 1
        assert told.decoration["dz"] is None, "the substrate recipe never ran"
        assert told.decoration["recipe"] == "caller-supplied decoration"
        assert told.presentation["c"] is None
        assert (down.group.number_str, down.group.name, down.group.size) == \
               (told.group.number_str, told.group.name, told.group.size)
        assert down.assignment.wyckoff_spglib == told.assignment.wyckoff_spglib
        assert down.assignment.wyckoff_tables == told.assignment.wyckoff_tables
        assert tuple(down.validated.indices) == tuple(told.validated.indices)
        assert down.rep.characters.irreps == told.rep.characters.irreps

        a = tqw.hsp_table(down.rep, bands=2)
        b = tqw.hsp_table(told.rep, bands=2)
        assert a.complete and b.complete, name
        assert a.counts == b.counts, name
        assert {p: dict(v) for p, v in a.content.items()} == \
               {p: dict(v) for p, v in b.content.items()}, name
        sa = tqe.symmetry_indicator(a, down.group, rep=down.rep, bands=2)
        sb = tqe.symmetry_indicator(b, told.group, rep=told.rep, bands=2)
        assert sa and sb, str(sa if not sa else sb)
        assert sa.values == sb.values and sa.order == sb.order
        classes[name] = sa.values
    assert classes == {"strong": (0, 1), "weak": (1, 0)}


@pytest.mark.slow
def test_descend_false_restores_the_refusal_the_descent_replaced():
    """The upper bound is still askable, and it is the old record verbatim.

    On a 3D crystal whose hoppings break the fourfold axis: the cell allows
    ``P4/mmm1' #123.340``, the Hamiltonian obeys 16 of its 32 operations, and
    every broken one is a proof.  The context carries the audit the refusal
    was made on, and the message names the routes to the group the Hamiltonian
    has -- ``typat=``, ``decorations=`` for a distortion no symmetry finder can
    see, and the descent itself -- rather than a tolerance to loosen.
    """
    r = tqg.realized_group(anisotropic_stack(), descend=False)
    assert not r and isinstance(r, Refusal)
    assert r.gate == "cell_group_not_realized"
    assert (r.value, r.threshold) == (16.0, 0.0)
    assert r.context["n_obeyed"] == 16 and r.context["size"] == 32
    assert r.context["refuted_by"] == {"no symmetry matrix": 16}
    assert r.context["number_str"] == "123.340"
    assert len(r.context["broken"]) == 16
    assert isinstance(r.context["validated"], tqg.ValidatedGroup)
    assert r.context["validated"].broken == r.context["broken"]
    assert isinstance(r.context["group"], tqg.GroupData)
    assert "typat=" in r.reason and "decorations=" in r.reason
    assert "no symmetry finder can see it" in r.reason
    assert "unmeasured upper bound" in r.reason
    assert "descend=True" in r.reason


# ---------------------------------------------------------------------------
# 2.  the 2D descent: the paper's P41' lattice, with its mirrors broken
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def p41_routes(p41_chiral_substrate):
    """The chiral P41' lattice collapsed onto its plane, and the paper's own model decorated.

    The chiral model is read **on a substrate**, which is a ``dim_r = 2``
    model's default since wave R4c and the one of the two readings this model
    answers without the tie rule: read free-standing its survivors hold the
    fourfold rotations and the fourfold rotoinversions alike, and the rule keeps
    the rotations.  The decorated route needs no such choice — the pinwheel says
    which crystal this is, and the free-standing reading of *that* cell is
    unambiguous and needs no descent at all.
    """
    chiral = p41_chiral_substrate       # the session's shared record of that reading
    told = tqg.realized_group(tqm.p41_flat_band(), decorations=CHIRAL_CELL,
                              cross_check=False)
    assert chiral and told, str(chiral if not chiral else told)
    return chiral, told


@pytest.mark.slow
def test_the_chiral_pattern_descends_the_square_lattice_to_p4(p41_routes):
    """``#99.164 -> #75.2``: the four vertical mirrors refuted, and ``tau`` with them.

    The paper's model as transcribed obeys the vertical mirrors of the square
    lattice too — 16 of 16 of ``P4mm1'``.  One C4- and time-reversal-symmetric
    next-nearest pattern with unequal amplitudes on the two diagonals breaks
    exactly those, and the descent reads them off by name: ``m(x)``, ``m(y)``,
    ``m([110])``, ``m([1-10])`` and their primed partners, every one
    ``"no symmetry matrix"``.  Time reversal is untouched, and the result is grey.

    Two things about the **parent** cell moved in wave R4b, and neither moves
    the answer.  The reading asked for is the substrate one, since the
    free-standing reading of this model is the one question it cannot answer.
    And the substrate's own decoration goes over *every* projected
    site rather than once at the origin, so the flat point set — two sites at
    ``(1/2, 0)`` and ``(0, 1/2)`` — keeps its accidental repeat under
    ``tau = (1/2, 1/2, 0)`` and the cell allows the lattice supergroup, 32
    operations rather than 16.  The Hamiltonian breaks ``tau`` (measured
    covariance residual 1.0 against ``tol = 1e-8``) and 11 coset partners with
    it, so the 24 refutations are those 8 mirrors once per coset — 16 of them —
    and the 8 rotations that make up the rest of the ``tau`` coset.  What is
    served is the same ``P41' #75.2``, 8 of 8, on the model's own lattice.

    A decoration with the same chirality (``CHIRAL_CELL``) reaches the same
    group with no descent at all, and the same model SRSIs: those are read off
    the **orbitals** -- two Kramers pairs at ``2c`` -- so breaking the mirrors
    moves neither them nor the site the pairs sit on.  The five maximal keys of
    ``#75.2`` are the columns the SRSI criterion works in, and their band
    dimensions bound its enumeration.
    """
    chiral, told = p41_routes
    d = chiral.descent
    assert chiral.environment == "substrate"
    assert d["from"] == {"number_str": "99.164", "name": "P4mm1'", "size": 32}
    assert d["to"] == {"number_str": "75.2", "name": "P41'", "size": 8}
    assert d["iterations"] == 1 and d["dropped_by"] == {"hoppings": 24} and len(d["dropped"]) == 24
    assert d["steps"][0]["refuted_by"] == {"no symmetry matrix": 16, "tensor structure": 8}
    mirrors = [b for b in d["dropped"] if b["refuted_by"] == "no symmetry matrix"]
    assert len(mirrors) == 16, "the four mirrors and their primed partners, twice"
    plain = {"m(x)", "m(y)", "m([110])", "m([1-10])"}
    with_tau = {"{" + m + "|1/2,1/2,0}" for m in plain}
    assert {b["name"] for b in mirrors} == {
        n + p for n in plain | with_tau for p in ("", "'")}, "each mirror, and with tau"
    # the other eight carry the rotations of the identity coset with tau on them
    assert {b["name"] for b in d["dropped"] if b["refuted_by"] == "tensor structure"} == {
        "{" + r + "|1/2,1/2,0}" + p for r in ("E", "C4(z)", "C2(z)", "C4^-1(z)")
        for p in ("", "'")}

    step = d["lattice"]
    assert step["index"] == 2 and step["realized"] == ()
    assert step["fractional_translations"] == ((0.5, 0.5, 0.0),)
    assert step["coset_partners_refuted"] == 11
    (tau,) = step["operations"]
    assert tau["translation"] == (0.5, 0.5, 0.0)
    assert tau["refuted_by"] == "tensor structure"
    assert tau["residual"] == pytest.approx(1.0, abs=1e-9)

    assert len(chiral.validated.indices) == chiral.group.size == 8
    assert chiral.group.provenance == "descended"
    assert chiral.time_reversal["outcome"] == "kept"
    assert chiral.time_reversal["n_obeyed"] == chiral.time_reversal["n_antiunitary"] == 4
    assert chiral.assignment.wyckoff_spglib == ("c",)

    # the generic point is placed OFF the plane, inside the decorated cell's own
    # extent, so the slab's vacuum padding is the parent cell's bit for bit.
    # `descent["cell"]["generic_point"]` is in the PADDED frame, and DECORATION_DZ is a
    # fraction of the CELL's third vector, so the depth is read where the two
    # agree -- off the descent species' own position in `cell.orb_vecs`.
    assert d["cell"]["generic_point"][:2] == (0.1234, 0.2718)
    depth, = {round(float(v[2]), 12)
              for t, v in zip(chiral.cell_typat,
                              np.asarray(chiral.cell.orb_vecs, dtype=float))
              if str(t) == "descent"}
    assert -tqg.DECORATION_DZ < depth < 0.0, \
        "between the substrate's shadow species at -dz and the sites at 0"
    assert len(d["cell"]["orbit"]) == 4, "one point per kept unitary operation"
    assert repr(chiral.rep).startswith("SpaceGroupRep(P41' #75.2, ")

    # the decoration route: the same group, no descent, and the same model SRSIs
    assert told.group.number_str == "75.2"
    assert told.descent is None and told.group.provenance == "realized"
    assert told.assignment.wyckoff_spglib == chiral.assignment.wyckoff_spglib == ("c",)
    m, told_m = tqr.srsi_of_model(chiral), tqr.srsi_of_model(told)
    assert m and told_m, str(m if not m else told_m)
    assert tuple(told_m.theta_Z) == tuple(m.theta_Z)
    assert told_m.theta_Zn == m.theta_Zn

    # the five maximal keys of #75.2 are the columns the criterion works in,
    # and their band dimensions are what bounds its enumeration
    tables = tqr.srsi_tables(chiral.group)
    assert tables, str(tables)
    assert [key.label for key in tables.maximal_keys] == [
        "(-1E12E1)_a", "(-1E22E2)_a", "(-1E12E1)_b", "(-1E22E2)_b", "(-1E2E)_c"]
    assert [key.dim for key in tables.maximal_keys] == [2, 2, 2, 2, 4]


# ---------------------------------------------------------------------------
# 3.  the fixtures whose operations are all obeyed: nothing moved
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("build,served", [
    pytest.param(tqm.i213_flicker2018, ("199.13", 24), id="i213"),
    pytest.param(tqm.i41_soc, ("80.30", 8), id="i41_soc"),
    pytest.param(tqm.p62_atomic, ("171.122", 12), id="p62"),
    pytest.param(tqm.p3121_atomic, ("152.34", 12), id="p3121"),
])
def test_a_fixture_that_obeys_its_cell_group_is_untouched(realized, build, served):
    """``descent is None``, and the record is the one this suite has always pinned.

    The descent exists for the models whose Hamiltonians break something; every
    fixture that obeys its cell group must reach exactly the record it reached
    before, which is what makes the wave a strict addition.  The pinned fields
    are the ones a descent would have moved: the group and its stamp, the
    survivor list, the cell (which a descent replaces) and the decoration
    record (which a descent's own species must stay out of).

    These four are fully periodic, so they are also where the layer's
    questions do not arise: a crystal periodic in three directions has no layer
    and no other side of one, so the record's ``environment`` is ``None`` (not a
    default word -- the question does not apply), nothing is cross-checked, no
    cell is decorated and ``decoration`` is ``None`` with it, and the repr names
    no reading.
    """
    # the session's shared records, with nothing declared
    r = realized(build)
    assert r, str(r)
    assert r.descent is None
    assert (r.group.number_str, r.group.size) == served
    assert r.group.provenance == r.rep.group.provenance == "realized"
    assert r.group._lattice_index == 1 and r.group._tables_ready
    assert len(r.validated.indices) == r.group.size
    assert not r.validated.broken
    assert "descended from #" not in repr(r)
    # the cell IS the model's own point set: nothing was added to it, by a
    # decoration or by a descent, and a fully periodic model records no
    # decoration at all
    assert r.cell is r.model, "the model's own point set was the cell"
    assert r.decoration is None
    assert "descent" not in {str(t) for t in r.cell_typat}
    # ... and no environment, no cross-check, and no reading in the repr
    assert r.environment is None and r.environment_declared is False
    assert r.cross_check is None and r.rep.environment is None
    assert "free-standing" not in repr(r) and "on a substrate" not in repr(r)


def _flat_rashba_slab(own_c=5.0):
    """Kane-Mele with a staggered potential and Rashba, as a flat ``dim_r = 3`` slab.

    ``kane_mele(1.0, -1.0, 0.3, 0.25)`` with a real out-of-plane lattice vector
    of length ``own_c`` and every orbital at ``z = 0``: nothing gives the cell an
    extent to put a descent's generic point inside, so its orbit sits off the
    plane.
    """
    layer = kane_mele(1.0, -1.0, 0.3, 0.25)
    lat = np.zeros((3, 3))
    lat[:2, :2] = np.asarray(layer.lat_vecs, dtype=float)
    lat[2, 2] = float(own_c)
    orb = np.zeros((int(layer.norb), 3))
    orb[:, :2] = np.asarray(layer.orb_vecs, dtype=float)
    out = TBModel(Lattice(lat.tolist(), orb.tolist(), periodic_dirs=[0, 1]), spinful=True)
    onsite = np.asarray(layer.onsite)
    for i in range(int(layer.norb)):
        out.set_onsite(onsite[i], i)
    for h in layer.hoppings:
        out.set_hop(np.asarray(h["amplitude"]), int(h["from_orbital"]),
                    int(h["to_orbital"]), list(h.get("lattice_vector", [0, 0])) + [0])
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,build,kw,parent,served,refuted_by,closure_names,labels,letters", [
        (f"rashba_slab_c{c:g}", _flat_rashba_slab, dict(c=c),
         ("191.234", "P6/mmm1'", 48), ("156.50", "P3m11'", 12),
         {"no symmetry matrix": 24, "tensor structure": 12}, set(), ("-E1", "-E1"), ("c", "b"))
        for c in (5.0, 30.0)
    ])
def test_a_flat_fixture_reaches_the_same_group_by_the_layer_descent(
        name, build, kw, parent, served, refuted_by, closure_names,
        labels, letters):
    """A flat slab, whose group is now measured rather than decorated for.

    Before wave R4b a flat model reached its group by construction: the cell
    was decorated so that ``sigma_h`` and everything composed with it left the
    geometry before the Hamiltonian was consulted.  The **free-standing**
    reading — declared here, since wave R4c makes the substrate a flat model's
    default — removes nothing, so all of that arrives at the detector and the
    tensor test, and what comes back is a **descent**: the ``R[2,2] = -1``
    operations refuted one by one, ``refuted_by == "tensor structure"``, plus
    the ones the staggered potential breaks outright.  The group, its size,
    the corep labels and the Wyckoff letters are the ones the decoration used
    to supply; the goldens' own descents are records of the baseline gate.

    The step's ``refuted_by`` histogram settles the whole of the dropped list's
    bookkeeping — ``n_dropped`` is ``len(dropped)`` and the histogram is taken
    *over* ``dropped`` — so only it is pinned, together with the one thing it does
    not carry: what the operations left out to close a group are **called**.

    The two cases are one flat slab presented at two vacuum lengths ``c``.  Its
    descent's generic orbit sits off the plane, where it widens the cell's
    extent, and the cell is padded with the model's own vacuum whatever it
    holds, so the group identified from the model is returned at every ``c``
    and the record is the same both times.
    """
    r = tqg.realized_group(build(), environment="free-standing", **kw)
    assert r, str(r)
    d = r.descent
    assert d is not None, name
    assert (d["from"]["number_str"], d["from"]["name"], d["from"]["size"]) == parent
    assert (d["to"]["number_str"], d["to"]["name"], d["to"]["size"]) == served
    assert d["iterations"] == 1
    assert d["steps"][0]["refuted_by"] == refuted_by
    # the operations left out to close a group, by NAME, which the histogram
    # above does not give
    assert {b["name"] for b in d["dropped"]
            if b["refuted_by"] == "closure"} == closure_names

    # the served answer, unchanged: this is what the decoration used to supply
    assert (r.group.number_str, r.group.name, r.group.size) == served
    assert r.group.provenance == r.rep.group.provenance == "descended"
    assert len(r.validated.indices) == r.group.size
    assert not r.validated.broken and len(r.rep.indices) == r.group.size
    assert r.rep.characters.irreps == labels
    assert r.assignment.wyckoff_spglib == letters
    assert f"descended from #{parent[0]}" in repr(r)

    # the generic orbit that carried the descent lives on the CELL and nowhere
    # else: the caller decorated with nothing, and the model was not touched
    assert r.decoration["n_atoms"] == 0
    assert {str(t) for t in r.cell_typat[len(r.typat):]} == {"descent"}
    assert int(r.cell.norb) == r.model.norb + len(d["cell"]["orbit"])
    assert len(d["cell"]["orbit"]) == r.group.size // 2, "one point per kept unitary operation"

    # ... and the two readings of the model agree on which group it is
    assert r.cross_check["agrees"] is True
    assert r.cross_check["outcome"] == "agree"
    assert r.cross_check["free_standing"]["number_str"] == served[0]
    assert r.cross_check["substrate"]["number_str"] == served[0]


# ---------------------------------------------------------------------------
# 4.  the verification gate
# ---------------------------------------------------------------------------

def _cheap_parent():
    """The anisotropic stack, its cell group and the detector's verdicts: 8 unitary kept."""
    model = anisotropic_stack()
    group = _spacegroup._group_of(model)
    validated = _measure._validate_group(model, group, nk=4, tol=1e-8)
    return model, group, validated


@pytest.mark.slow
def test_an_identification_that_is_not_the_kept_set_raises():
    """The verification step, provoked: a cell that cannot carry the kept set.

    ``_descend_once`` never trusts the construction — it matches the identified
    unitary operations against the kept ones, rotation by rotation and
    translation modulo the lattice, and a cell read as any other group is built
    again from the next generic point.  Handed a cell whose own points have no
    symmetry at all, the orbit of the generic point is all there is, whichever
    point it is, and every identification comes back ``P11' #1.2``: seven of
    the eight kept operations of ``Pmmm1'`` are simply not in it.  With the
    points exhausted the package raises rather than serving a group that has
    moved — the kept set is a group and a generic orbit under it has it for a
    stabilizer, so this is the package disagreeing with itself.
    """
    model, group, validated = _cheap_parent()
    lat = np.asarray(model.lat_vecs, dtype=float)
    lopsided = TBModel(Lattice(lat, [[0.0, 0.0, 0.0], [0.31, 0.17, 0.43]],
                               periodic_dirs=[0, 1, 2]), spinful=False)

    with pytest.raises(InternalError, match="are not that set") as err:
        _descent._descend_once(lopsided, [0, 0], group, validated, spinor=False,
                          include_TR=True, symprec=1e-5, step=1)
    said = str(err.value)
    assert "#123.340" in said and "#1.2" in said
    assert "7 kept operations" in said and "are not among" not in said
    assert f"last of {len(_descent._DESCENT_POINTS)} generic points" in said


def test_the_operation_matcher_pairs_one_to_one_in_both_directions():
    """The matcher the descent's verification reads, on its own.

    A one-to-one pairing is what makes "the identification IS the kept set" a
    statement rather than a count: an identification that is still too large
    leaves *found* operations unmatched, and one that has moved leaves *kept*
    ones unmatched.  Both directions are asserted, on real operations.

    The matcher's rule on translations is "**modulo the lattice**, to
    ``symprec``", the finder's distance, so the difference is read as the
    length it is on the cell; both halves are probed at the boundary rather than
    with a shift an exact comparison would refuse just as well.  On this cell,
    ``(1, 1, 1.6)``, moving every reduced component of a translation by ``s`` is
    a displacement of length ``2.135·s``: a shift of exactly ``1.0`` is the
    *same* operation and must pair; ``6e-6``, a length of ``1.28e-5`` at
    ``symprec = 1e-5``, is a different one and must not; ``4e-6``, a length of
    ``8.5e-6``, is the same one again and must.  A matcher comparing
    translations outright would fail the first and the third, one ignoring
    ``symprec`` would fail the second, and one reading the reduced difference
    against ``symprec`` as though it were a length would fail the second too.
    """
    group = _spacegroup._group_of(anisotropic_stack())
    assert not isinstance(group, Refusal), str(group)
    unitary = [s for s in group.symmetries if not s.time_reversal]
    four = unitary[:4]
    lattice = group.lattice

    pairs, missing, extra = _descent._match_operations(four, four, 1e-5, lattice)
    assert len(pairs) == 4 and not missing and not extra
    assert [a for a, _b in pairs] == [int(s.ind) for s in four]

    pairs, missing, extra = _descent._match_operations(four, four[:2], 1e-5, lattice)
    assert len(pairs) == 2 and missing == tuple(int(s.ind) for s in four[2:])
    assert extra == ()

    pairs, missing, extra = _descent._match_operations(four[:2], four, 1e-5, lattice)
    assert len(pairs) == 2 and missing == ()
    assert extra == tuple(int(s.ind) for s in four[2:])

    # a translation moved by more than symprec is a different operation
    moved = [_TranslatedOp(s, 0.5) for s in four[:1]]
    pairs, missing, extra = _descent._match_operations(moved, four, 1e-5, lattice)
    assert not pairs and len(missing) == 1 and len(extra) == 4

    # ... and one moved by a whole lattice vector is the SAME operation
    same = [_TranslatedOp(s, 1.0) for s in four[:1]]
    pairs, missing, extra = _descent._match_operations(same, four, 1e-5, lattice)
    assert len(pairs) == 1 and missing == () and len(extra) == 3

    # the gate itself: symprec = 1e-5, straddled as a length
    just_out = [_TranslatedOp(s, 6e-6) for s in four[:1]]
    pairs, missing, extra = _descent._match_operations(just_out, four, 1e-5, lattice)
    assert not pairs and len(missing) == 1 and len(extra) == 4

    just_in = [_TranslatedOp(s, 4e-6) for s in four[:1]]
    pairs, missing, extra = _descent._match_operations(just_in, four, 1e-5, lattice)
    assert len(pairs) == 1 and missing == () and len(extra) == 3


class _TranslatedOp:
    """One operation with its translation moved — a different operation."""

    def __init__(self, sym, shift):
        self.ind = int(sym.ind)
        self.rotation = np.asarray(sym.rotation)
        self.translation = np.asarray(sym.translation, dtype=float) + float(shift)
        self.time_reversal = bool(sym.time_reversal)


# ---------------------------------------------------------------------------
# 5.  wave R2: the mirror the detector cannot refute, refuted by tensor structure
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def slab_mirror():
    """Kane-Mele at ``rashba = 0.25``, asked with ``sigma_h`` kept.

    The cell keeps the mirror and the Hamiltonian refutes it — the
    **free-standing** reading, declared, since wave R4c reads a flat model on a
    substrate unless told otherwise.
    """
    rashba = kane_mele(0.0, -1.0, 0.3, 0.25)
    down = tqg.realized_group(rashba, environment="free-standing",
                              characters={0: "pz"}, cross_check=False)
    assert down, str(down)
    return rashba, down


@pytest.mark.slow
def test_the_slab_mirror_is_refuted_by_tensor_structure_and_the_cell_descends(
        slab_mirror):
    """``#191.234 -> #183.186``: the 24 ``R[2,2] = -1`` operations, named.

    The detector kept all 48 and left them undetermined, so the descent asks the
    **representation's** question of each one: does the solution space of that
    operation's class contain a member of the form ``O(g) (x) S(g)``?  On a flat
    layer ``O`` is the same permutation for ``g`` and ``sigma_h*g``, and ``S``
    is not — so a Rashba term, which is odd under ``S(sigma_h)``, refutes
    exactly the 24 operations with ``R[2,2] = -1`` and nothing else.  Each is
    held to its own lift, and on it the scaled defect is 1.0 for every one --
    no member of the solution space has that tensor form at all -- against a
    gate of 1e-6; the operations kept read 1.3e-15.

    The same selection where a site's orbitals are one multiplet wider than one
    orbital, whose block is read off the solution space both members of a pair
    share and so tells them apart by nothing, is made by the spin lift alone:
    each member held to its own lift on that whole space, with any orbital
    factor.  The spinful ``(p_x, p_y)`` honeycomb, each site's pair one
    multiplet, descends the same way with a nearest-neighbour Rashba term, the
    same 24 refuted.  With a nearest-neighbour ``i lam (sigma . d)`` instead
    every rotation holds, the in-plane twofolds among them, and every improper
    operation is refuted: ``P6221' #177.150``, whose side-preserving half is
    the ``P61' #168.110`` the substrate reads.  Every refuted member misses its
    own lift outright (1.0), one member of every pair holding, and with each
    site's pair written in a basis turned by its own angle every group, refuted
    operation and label is the same.
    """
    _rashba, down = slab_mirror
    d = down.descent
    assert d is not None
    assert d["from"] == {"number_str": "191.234", "name": "P6/mmm1'", "size": 48}
    assert d["to"] == {"number_str": "183.186", "name": "P6mm1'", "size": 24}
    assert d["iterations"] == 1 and d["dropped_by"] == {"hoppings": 24} and len(d["dropped"]) == 24
    assert d["steps"][0]["refuted_by"] == {"tensor structure": 24}
    assert {b["refuted_by"] for b in d["dropped"]} == {"tensor structure"}
    assert {b["name"] for b in d["dropped"]} == {
        "C2(x)", "C2(y)", "C2([010])", "C2([110])", "C2([1-10])", "C2([2-10])",
        "i", "m(z)", "S3(z)", "S3^-1(z)", "S6(z)", "S6^-1(z)",
        "C2(x)'", "C2(y)'", "C2([010])'", "C2([110])'", "C2([1-10])'", "C2([2-10])'",
        "i'", "m(z)'", "S3(z)'", "S3^-1(z)'", "S6(z)'", "S6^-1(z)'"}
    assert min(b["residual"] for b in d["dropped"]) == pytest.approx(1.0, abs=1e-9)
    assert max(b["residual"] for b in d["dropped"]) == pytest.approx(1.0, abs=1e-9)
    # five decades above the gate those defects are measured against, so every
    # one of the 24 is refuted outright and none of them is a near miss
    assert min(b["residual"] for b in d["dropped"]) > 1e4 * tqs.ANSATZ_GATE

    # the broken set IS the sigma_h family, read off the rotations themselves
    m3 = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.25))
    # a third real direction and still two periodic ones: the layer gets a
    # coordinate to be mirrored in, and no third momentum
    assert m3.lattice.dim_r == 3 and tuple(m3.periodic_dirs) == (0, 1)
    cell, cell_typat = _layers._decorated_cell(m3, [0, 0], environment="free-standing")
    parent = _spacegroup._group_of(cell, typat=cell_typat, spinor=True)
    rot = {int(s.ind): int(np.rint(np.asarray(s.rotation, dtype=float)[2, 2]))
           for s in parent.symmetries}
    assert {rot[b["ind"]] for b in d["dropped"]} == {-1}
    assert {rot[i] for i in d["steps"][0]["kept"]} == {1}
    assert len(d["cell"]["orbit"]) == 12, "one point per kept unitary operation"

    # ... and only that group is served
    assert len(down.validated.indices) == down.group.size == 24
    assert not down.validated.broken and len(down.rep.indices) == 24
    # one intertwiner per operation, against the two the Rashba-free layer has:
    # the second solution there is s_z, and the Rashba term is what takes it
    # away (the commutant the tqc_kane_mele_realized_group golden prints: "the
    # identity and 1 more" at rashba = 0, s_z, and "0 more" at rashba = 0.25)
    assert set(down.validated._nullspace_dims.values()) == {1}
    assert len(down.validated._nullspace_dims) == 24
    assert down.group.provenance == down.rep.group.provenance == "descended"
    assert ("descended from #191.234 (24 of 48 coset representatives of the lattice "
            "translations broken") in repr(down)
    assert int(down.cell.norb) == 14 and set(down.cell_typat[2:]) == {"descent"}
    assert down.decoration["n_atoms"] == 0, "the caller decorated with nothing"

    # ... and where each site's pair is one multiplet: the spin lift on the
    # solution space the pair shares, in whatever basis each site's pair is written
    improper = {"i", "m(z)", "S3(z)", "S3^-1(z)", "S6(z)", "S6^-1(z)"}
    vertical = {"m(x)", "m(y)", "m([010])", "m([100])", "m([1-20])", "m([2-10])"}
    in_plane = {"C2(x)", "C2(y)", "C2([010])", "C2([100])", "C2([1-20])", "C2([2-10])"}
    for soc, want, refuted, substrate in (
            ("sigma.d", ("177.150", "P6221'"), improper | vertical, "168.110"),
            ("rashba", ("183.186", "P6mm1'"), improper | in_plane, "183.186")):
        read = {}
        for turn_a, turn_b in ((0.0, 0.0), (0.7, -1.3)):
            r = tqg.realized_group(tqm.pxpy_honeycomb_spinful(soc, turn_a=turn_a, turn_b=turn_b),
                                   environment="free-standing")
            assert r, str(r)
            assert r.rep.multiplets.dims == ((2,), (2,)), "each site's pair is one multiplet"
            d = r.descent
            assert (r.group.number_str, r.group.name, r.group.size) == (*want, 24)
            assert d["from"]["number_str"] == "191.234" and d["iterations"] == 1
            assert d["steps"][0]["refuted_by"] == {"tensor structure": 24}
            assert {b["name"] for b in d["dropped"]} == refuted | {n + "'" for n in refuted}
            assert min(b["residual"] for b in d["dropped"]) == pytest.approx(1.0, abs=1e-9)
            assert (r.cross_check["outcome"], r.cross_check["substrate"]["number_str"]) == (
                "agree", substrate)
            read[turn_a] = (r.group.number_str, sorted(b["name"] for b in d["dropped"]),
                            r.rep.characters.irreps)
        assert read[0.0] == read[0.7], "the basis of each site's pair moves nothing"
        assert read[0.0][2] == ("-1E2E + -E1",) * 4


@pytest.mark.slow
def test_descend_false_refuses_and_names_the_tensor_structure(slab_mirror):
    """The upper bound is still askable, and the refusal says which measurement made it."""
    rashba, _down = slab_mirror
    r = tqg.realized_group(rashba, environment="free-standing",
                           characters={0: "pz"}, descend=False)
    assert not r and isinstance(r, Refusal)
    assert r.gate == "cell_group_not_realized"
    assert (r.value, r.threshold) == (24.0, 0.0)
    assert r.context["n_obeyed"] == 24 and r.context["size"] == 48
    assert r.context["refuted_by"] == {"tensor structure": 24}
    assert len(r.context["tensor_structure_refuted"]) == 24
    assert r.context["number_str"] == "191.234"
    assert "broken by tensor structure" in r.reason
    assert "O(g) (x) S(g)" in r.reason
    assert "descend=True" in r.reason


@pytest.mark.slow
def test_the_ansatz_defects_of_an_exact_model_are_at_machine_level(realized):
    """The ceiling: every operation an exact fixture obeys reads ``0``, not ``1``.

    The gate is 1e-6 on a **scaled** defect that lives in ``[0, 1]`` — the
    fraction of the closest-to-ansatz member of the detector's solution space
    that no ``O(g) (x) S(g)`` can carry.  Asserted here at 1e-10 over every
    operation of an exact 3D fixture and over all 48 shared operations of the
    ``sigma_h``-keeping slab: measured 2.85e-15 and 7.27e-15.  A ceiling is what
    makes the 0.5 and 1.0 of the broken case a separation rather than a
    threshold.

    Both branches of the read-off are measured for every operation, and the
    defect on the record is the **better** of the two.  Which one wins is not
    pinned and deliberately so: on an exact model the two branches read 1e-16
    against 1e-16, so the winner is a roundoff coin-flip and pinning it would
    make this file machine-dependent.  What is asserted instead is the relation
    that a record naming the wrong branch would break —
    ``min(branches.values()) == scaled``.
    """
    cases = []
    r = realized(tqm.i213_flicker2018)          # the session's shared record
    assert r, str(r)
    assert not r.validated.undetermined, "a 3D primitive group has singleton classes"
    cases.append((r.model, r.group, r.assignment, r.validated, set(r.validated.indices)))

    m3 = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))
    cell, cell_typat = _layers._decorated_cell(m3, [0, 0], environment="free-standing")
    group = _spacegroup._group_of(cell, typat=cell_typat, spinor=True)
    asg = _tables._assign_orbitals(m3, group, typat=[0, 0])
    val = _measure._validate_group(m3, group, nk=6, tol=1e-8)
    assert len(val.undetermined) == 48
    cases.append((m3, group, asg, val, set(val.undetermined)))

    for model, group, asg, val, inds in cases:
        positions = [i for i, s in enumerate(group.symmetries) if int(s.ind) in inds]
        got = _multiplets._ansatz_defects(model, group, asg, val, positions)
        assert not isinstance(got, Refusal), str(got)
        assert set(got) == inds
        assert all(rec["realizable"] for rec in got.values())
        assert max(rec["scaled"] for rec in got.values()) < 1e-10
        # the branch the read-off would take is on the record, and both are
        # measured -- and the defect recorded IS the better of the two
        assert all(set(rec["branches"]) == {1, -1} for rec in got.values())
        assert all(min(rec["branches"].values()) == rec["scaled"]
                   for rec in got.values())


@pytest.mark.slow
@pytest.mark.parametrize("build,kw,chain,name,number_str", [
    (lambda: tqm.p41_flat_band(chiral=True),
     dict(environment="substrate", cross_check=False), "#99.164 -> #75.2",
     "P41'", "75.2"),
    (tqm.i213_flicker2018, {}, None, "I2_131'", "199.13"),
], ids=["descends", "realizes_its_cell_group"])
def test_the_consistency_guard_surfaces_a_descent_that_did_not_clear_the_gate(
        monkeypatch, build, kw, chain, name, number_str):
    """``rep_not_covariant`` left standing beyond ``tol`` is a bug, and it raises.

    Three independent measurements say which operations are obeyed — the
    detector's per-k-action-class verdict, the ansatz test on the ones it leaves
    undetermined, and ``_build_rep``'s own covariance gate — so the third has to
    close on the **first** attempt, whatever settled the group: a descent, or
    the cell's own group realized outright.  Disagreeing is a defect in
    ``realized_group`` and not a tolerance to loosen, so the package raises
    ``InternalError`` naming the group and the residual rather than trying a
    second descent, with the descent chain in the message only where one ran.
    No Hamiltonian measured here produces it, so the branch is provoked on two
    fixtures: the chiral flat-band layer on a substrate, which descends
    ``#99.164 -> #75.2`` by tensor structure before a representation is ever
    asked for, and the primitive ``I2_13`` crystal, which obeys its cell's
    group outright and never descends.

    One case is carved out of the guard: a covariance residual inside the
    detector's ``tol`` — the near-symmetric signal, which ``symmetrize=`` is the
    answer to — would otherwise be hidden behind this message for every 2D
    caller, since a descent runs on every 2D model in layer mode. Provoked at
    1e-9 against the default ``tol`` of 1e-8, the guard stands aside and
    ``rep_not_covariant`` comes back verbatim, naming the group identified and
    carrying it in ``context["group"]``; provoked at 1.42 it fires, because
    three measurements of the same fact disagreeing by 1.42 is a defect and not
    a tolerance.
    """
    import pythtqc.symmetry as _sym
    from pythtqc import _build

    real_build_rep = _build._build_rep

    def refusing_at(value):
        def refusing(*args, **kwargs):
            out = real_build_rep(*args, **kwargs)
            if isinstance(out, Refusal):
                return out
            return Refusal("rep_not_covariant",
                           "provoked: no candidate closes the covariance gate.",
                           value=value, threshold=tqs.EXACT_TOL,
                           context={"n_candidates": 4})
        return refusing

    patch_everywhere(monkeypatch, "_build_rep", refusing_at(1.42))
    with pytest.raises(InternalError,
                       match=r"still does not carry H\(k\) onto itself within the "
                             "tolerance") as err:
        tqg.realized_group(build(), **kw)
    said = str(err.value)
    assert f"{name} #{number_str}" in said
    assert ("by the descent" in said) == (chain is not None)
    if chain is not None:
        assert f"by the descent {chain}" in said
    assert "reaches 1.42 against 5e-14" in said
    assert "a defect to report, not a tolerance to loosen" in said

    # inside the declared tolerance the guard is not what the caller is told
    patch_everywhere(monkeypatch, "_build_rep", refusing_at(1e-9))
    near = tqg.realized_group(build(), **kw)
    assert not near and isinstance(near, Refusal)
    assert near.gate == "rep_not_covariant"
    assert (near.value, near.threshold) == (1e-9, tqs.EXACT_TOL)
    assert "a defect to report" not in near.reason
    assert f"The group identified is {name} #{number_str}" in near.reason
    assert near.context["group"].name == name
    assert near.context["group"].number_str == number_str
