r"""Wave S1b: a **model's** stable real-space invariants, and a band subset's.

Where ``test_srsi_tables.py`` says the shipped tables are the paper's, this file
says the two routes onto a model are right.  Both of them turn on one thing that
is easy to get wrong and impossible to notice afterwards — **which setting a
Wyckoff letter is read in** — so that is where it starts.

1. **The letter in the tables' setting** (``groups.wyckoff_sites_tables``).  A
   Wyckoff letter is a table key, and it is setting-dependent:
   ``wyckoff_positions_spglib()`` (``irrep`` -> ``pyxtal`` -> spglib's
   transformation) and spglib's own per-site letters both live in spglib's
   standardization, which is *not* the setting
   ``irreptables``' rows and the SRSI tables are written in.  Route G differs
   from spglib's per-site answer on **every one** of the four fixtures here, and
   from ``wyckoff_positions_spglib()``'s on the diamond; the disagreements are
   asserted as **measurements**, not reconciled.  The setting itself is audited
   over all 230 space groups.
2. **A model -> its SRSIs** (``srsi.srsi_of_model``).  Per site orbit: the
   block's content measured off the representation at the points the model can
   sample, the orbit's letter measured off the geometry, the localization solved
   as a nonnegative-integer decomposition at a letter on the closure of the
   orbit's variety that the tables list EBRs at, and the SRSIs read off that.
   The distorted Fu-Kane-Mele orbit sits on a line the tables list no EBR at,
   and every localization of it carries the same SRSIs, which is the adiabatic
   shift measured rather than assumed.
3. **A band subset -> its Z-SRSIs** (``srsi.srsi_of_bands``), reproducing the
   fixture table of wave S1 through the public entry point.
4. **The identity that ties the two**: a model's ``theta_Z`` equals the band
   route's on *all* of its bands, on lin3g and on the odd Kane-Mele regime.
"""

import math
import re

import numpy as np
import pytest

from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc import _spacegroup
from pythtqc import _tables

import tqc_models as tqm                                            # noqa: F401
from conftest import FKM_BOND, fkm_regime, km_z2_regime

# lin3g band blocks, from LIN3G.band_sets (the same names test_ebr.py uses)
P4, P2, P6 = (0, 1, 2, 3), (4, 5), 6

_POSITION = re.compile(r"^(\d+)([a-zA-Z])\((.*)\)$")


class _Key:
    """The three fields ``_tables._ebr_data`` reads: a group record with no Hamiltonian."""

    def __init__(self, number_str, spinor, magnetic=True):
        self.number_str = number_str
        self.spinor = bool(spinor)
        self._magnetic = bool(magnetic)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fkm_strong(realized):
    """Fu-Kane-Mele's strong distorted phase in ``R-3m1' #166.98``."""
    out = realized(fkm_regime("strong"), decorations=FKM_BOND, characters={0: "s"})
    assert out, str(out)
    assert out.group.number_str == "166.98"
    return out


def _p43m_3c():
    """Every ``p`` orbital at ``3c`` of ``P-43m``, spinless: a composite multiplet beside another on one site."""
    from pythtqc import models
    return models._cubic_3c_p_model([(s, a) for s in range(3) for a in range(3)],
                                    models._cubic_rotations(False), 7)


@pytest.fixture(scope="module")
def p43m_composite(realized):
    """The ``3c`` model of ``P-43m1' #215.71``: two Wyckoff orbits on one orbit of sites, one of them composite."""
    out = realized(_p43m_3c)
    assert out, str(out)
    assert out.group.number_str == "215.71"
    return out


@pytest.fixture(scope="module")
def km_realized(realized):
    """Kane-Mele's two regimes in ``P3m11' #156.50``."""
    out = {}
    for regime in ("odd", "even"):
        got = realized(km_z2_regime(regime), typat=[0, 1], cross_check=False,
                       characters={0: "A1"})
        assert got, f"{regime}: {got}"
        assert got.group.number_str == "156.50"
        out[regime] = got
    return out


# ---------------------------------------------------------------------------
# 1.  the Wyckoff letter, in the setting the tables are written in
# ---------------------------------------------------------------------------

#: The measured table: what each fixture's orbits read in the **tables'**
#: setting, against what ``_assign_orbitals``' two answers in spglib's
#: standardization say.  ``wyckoff_spglib`` is per orbit and the other two per
#: site.
WYCKOFF_LETTERS = {
    "FKM diamond": (("a", "a"), ("b",), ("b", "b")),
    "lin3g": (("a", "c", "c"), ("a", "c"), ("a", "f", "f")),
    "Kane-Mele odd": (("c", "b"), ("c", "b"), ("e", "c")),
    "FKM strong": (("c", "c"), ("c",), ("b", "b")),
}


@pytest.mark.slow
def test_wyckoff_tables_reads_the_letters_the_tables_are_keyed_by(
        fkm_undistorted, lin3g_declared_cross_checked, km_realized, fkm_strong):
    """Route G on the four fixtures, and the two answers that disagree with it.

    The letter that matters for a table lookup is the one in the **reference
    (conventional) cell** ``irrep``'s own ``refUC``/``shiftUC`` define — the cell
    the k-labels and the EBR rows are already transported into — and it is read
    off ``pyxtal``'s Wyckoff list of the space-group number there.

    Every one of the four disagrees with spglib's per-site answer, and the
    diamond disagrees with ``wyckoff_positions_spglib()``'s as well; the
    disagreements are pinned rather than reconciled: the diamond's orbit is
    ``8a`` here and ``8b`` in both of them;
    lin3g's second orbit is ``2c`` here and spglib's ``f``; Kane-Mele's two
    sublattices are ``1c``/``1b`` here and spglib's ``e``/``c``; the distorted
    Fu-Kane-Mele orbit is ``6c`` here and spglib's ``b``.  Only the first column
    is ever used to key a table.

    And the diamond's two letters are not two names for one point set.
    ``Fd-3m`` has two origin choices, and in the one ``pyxtal`` and the tables
    use the model's two carbon sites transport to exactly ``(1/8,1/8,1/8)`` and
    ``(7/8,7/8,7/8)``: on ``8a``'s variety, and a quarter of a cell away from
    ``8b``'s.  The disagreement is about which cell the letter is read in, not an
    ambiguity within one.
    """
    fixtures = {"FKM diamond": fkm_undistorted, "lin3g": lin3g_declared_cross_checked,
                "Kane-Mele odd": km_realized["odd"], "FKM strong": fkm_strong}
    for name, realized in fixtures.items():
        group, assignment = realized.group, realized._assignment
        want_tables, want_spglib, want_dataset = WYCKOFF_LETTERS[name]
        assert assignment.wyckoff_tables == want_tables, name
        assert assignment.wyckoff_spglib == want_spglib, name
        assert assignment.wyckoff_spglib_dataset == want_dataset, name
        # ... and the record itself, on the group's own sites
        sites = tqg.wyckoff_sites_tables(group)
        # a gate that is 1e5 tighter than `wyckoff_sites_tables`'s own
        # WYCKOFF_REFERENCE_TOL: machine-exact, not merely within tolerance
        assert sites and all(site._residual < 1e-9 for site in sites), name

    # the diamond's sites in the tables' setting, and what the other letter
    # would cost: `8b` is a different orbit here, not the same one relabelled
    # (the gate is WYCKOFF_REFERENCE_TOL = 1e-4)
    group = fkm_undistorted.group
    references = [site._reference for site in tqg.wyckoff_sites_tables(group)]
    positions = sorted(tuple(np.round(reference, 6)) for reference in references)
    assert positions == [(0.125, 0.125, 0.125), (0.875, 0.875, 0.875)]
    arms = {position["letter"]: position["arms"]
            for position in _spacegroup._pyxtal_wyckoff(int(group._sg_number))[0]}
    assert _spacegroup._variety_residual(arms["a"], references).max() < 1e-9
    assert _spacegroup._variety_residual(arms["b"], references) == pytest.approx(
        [0.25, 0.25], abs=1e-12)


def test_the_distorted_orbit_sits_on_the_6c_line_between_3a_and_3b(fkm_strong):
    """``6c`` of ``R-3m``, the line ``(0,0,z)``, with its two ends measured.

    The orbit has no EBR column of its own — the tables list none at ``6c`` —
    and the positions with EBRs on the closure of its variety are ``3a`` at
    ``z = 0`` and ``3b`` at ``z = 1/2``.  That is a *geometric* fact, read off
    ``pyxtal``'s varieties, and it is what licenses the adiabatic shift the SRSI
    evaluation makes.

    The atomic-limit table says the same.  Its one row, the orbit, is the spinor
    doublet ``-E1`` of the ``3m`` site, named by the point group because the tables list
    no site irrep at ``6c``; and the orbit's band representation, written at
    ``3a``, the first letter of the closure, is the localization the SRSIs are
    evaluated on there, printed under the table because it sits at another
    position than the orbit's own.  The same pair at ``3b``, the other end of the
    line, carries the same symmetry content and the same SRSIs, so the table
    names it as the sum nothing it reads distinguishes from the one at ``3a``.
    """
    group, assignment = fkm_strong.group, fkm_strong._assignment
    sites = tqg.wyckoff_sites_tables(group)
    orbit_sites = [site for site in sites if site.wyckoff not in site.slides_onto]
    assert len(orbit_sites) == 2
    for site in orbit_sites:
        assert (site.wyckoff, site.n_free) == ("6c", 1)
        assert site.slides_onto == ("3a", "3b")
        assert site._varieties == ("c", "h", "i")
    assert sorted(round(float(site._reference[2]), 6) for site in orbit_sites) == \
        [0.125, 0.875]
    # the decorating species sits at 3a, which is its own closure
    decoration = [site for site in sites if site.wyckoff in site.slides_onto]
    assert [(site.wyckoff, site.slides_onto) for site in decoration] == [("3a", ("3a",))]
    assert assignment.wyckoff_tables == ("c", "c")

    table = fkm_strong.wyckoff_orbits
    (orbit,) = table
    assert (orbit.wyckoff, orbit.unitary_site_symmetry, orbit.orbitals, orbit.label,
            orbit.numerical.named_by) == ("6c", "3m", (0, 1), "-E1", "point group")
    at_a = [loc for loc in
            tqr.srsi_of_model(fkm_strong).wyckoff_orbits[0].numerical.localizations
            if loc.wyckoff == "3a"]
    assert [(t.multiplicity, t.irrep, t.wyckoff) for t in orbit.band_representation] == \
        [(t.multiplicity, t.irrep, t.wyckoff) for t in at_a[0]] == \
        [(1, "-E1g", "3a"), (1, "-E1u", "3a")]
    assert [[(t.multiplicity, t.irrep, t.wyckoff) for t in alternative]
            for alternative in orbit.numerical.alternatives] == \
        [[(1, "-E1g", "3b"), (1, "-E1u", "3b")]]
    assert ("not told apart from -E1g at 3b (-3m1') + -E1u at 3b (-3m1') by any stable "
            "real-space invariant") in table.summary()


@pytest.mark.data
def test_pyxtal_is_in_the_tables_setting_for_every_one_of_the_230_space_groups():
    """**The audit**, swept: zero offenders, and the sweep is not vacuous.

    Route G rests entirely on ``pyxtal``'s Wyckoff list being written in the
    same setting as ``irreptables``' operations, and that is measured rather
    than assumed: for every maximal position the group's EBR list names, the
    letter must exist in ``pyxtal``'s list with the same conventional
    multiplicity, and the **stabilizer** of ``pyxtal``'s own representative
    point under the tabulated reference-cell operations (plus the conventional
    cell's centering translations, themselves read off ``pyxtal``'s general
    position) must have order ``|P| n_centering / M``.

    The two sides come from independent sources, so an origin choice or an axis
    convention that differed between them would collapse the stabilizer.
    Measured 2026-09-03 over all 230 space groups: **0 offenders**, 837 maximal
    positions checked.  Both valuednesses induce from the same positions, so the
    sweep runs once per group.

    (The reference-cell translations are snapped to small-denominator rationals
    first, the way ``_table_kpoints`` snaps k-points: ``irreptables`` prints
    ``1/3`` as ``0.33333``, which is 3.3e-6 off and enough to lose a stabilizer
    at a 1e-6 gate.  Without the snap, SGs 151-154 read a stabilizer of 1 where
    the orbit-stabilizer theorem demands 2.)
    """
    from irreptables.irreps import IrrepTable

    offenders, checked, groups = [], 0, 0
    for sg in range(1, 231):
        data = _tables._ebr_data(_Key(str(sg), False, magnetic=False))
        assert data, (sg, data)
        positions = []
        for _name, position in data["ebrs"]:
            match = _POSITION.match(str(position))
            positions.append((int(match.group(1)), match.group(2), match.group(3)))
        table = IrrepTable(str(sg), False, magnetic=False)
        ops = [(np.rint(np.asarray(sym.R, dtype=float)).astype(int),
                _spacegroup._snap_rational(np.asarray(sym.t, dtype=float)))
               for sym in table.symmetries if not bool(sym.time_reversal)]
        audit = _spacegroup._wyckoff_setting_audit(sg, ops, positions)
        if not audit["ok"]:
            offenders.append((sg, audit["offenders"]))
        checked += audit["checked"]
        groups += 1
    assert groups == 230
    assert checked == 837, checked
    assert offenders == [], offenders


def test_a_drifted_center_reads_as_the_general_position_rather_than_refusing():
    """Measured, and worth stating: the ``wyckoff_unresolved`` gate cannot fire.

    The **general** position's variety is the whole cell, so in exact arithmetic
    every point lies on *some* variety and a centre that has drifted off ``8a``
    reads as ``192i`` — the honest geometric answer, not a failure.  What that
    costs is nothing here and everything downstream: the general position's
    closure carries all four letters the tables list EBRs at, and
    ``srsi_of_model`` then reports
    whichever one the block's content actually decomposes at, or refuses
    ``wyckoff_orbit_not_localizable``.  The gate stays as a guard against a ``pyxtal``
    list with no general position; this test is what says it is a guard.

    ``site._residual`` is **not** what says the centre drifted: the general
    position's projector is the zero matrix, so the residual is 0.0 for any
    point whatsoever.  What the drift cost is measured against ``8a``'s own
    arms, where the point is 0.18 away in reduced coordinates -- 1.8e3 times the
    1e-4 gate, and the same number reads 0.0 for a centre that is on ``8a``.

    The group is the diamond cell's, ``Fd-3m1' #227.129``, read off the geometry:
    a Wyckoff letter is a property of the cell's point set, which is all this
    asks, and no Hamiltonian has to be measured to ask it.
    """
    group = _spacegroup._group_of(tqm.fu_kane_mele_z2("uniform"))
    assert group and group.number_str == "227.129", str(group)
    out = tqg.wyckoff_sites_tables(group, [[0.03, 0.11, 0.27]])
    assert out and len(out) == 1, str(out)
    site = out[0]
    assert (site.wyckoff, site.n_free) == ("192i", 3)
    assert site.slides_onto == ("8a", "8b", "16c", "16d"), (
        "every position the tables list EBRs at is on the closure of the general one, "
        "and the general one is not among them")
    assert "wyckoff_unresolved" in tqg.wyckoff_sites_tables.__doc__
    # what drifting cost: the point is no longer on the maximal variety
    arms = {position["letter"]: position["arms"]
            for position in _spacegroup._pyxtal_wyckoff(int(group._sg_number))[0]}
    drifted = float(_spacegroup._variety_residual(arms["a"], [site._reference])[0])
    assert drifted > tqg.WYCKOFF_REFERENCE_TOL, drifted
    assert drifted == pytest.approx(0.18, abs=1e-12)
    on_site = _spacegroup._variety_residual(
        arms["a"], [s._reference for s in tqg.wyckoff_sites_tables(group)])
    assert on_site.max() < 1e-9, on_site
    # the reason of the audit's refusal, by the check that failed; no group fails
    # it (the sweep above), so each failure is planted in the group's audit
    planted = (
        ({"reason": "multiplicity", "measured": 16, "expected": 8},
         "the position a has multiplicity 16 in the pyxtal package's list, and 8 in the "
         "tables."),
        ({"reason": "absent from pyxtal's list", "measured": None, "expected": 8},
         "the position 8a of the tables is not in the pyxtal package's list."),
        ({"reason": "stabilizer order", "measured": 12, "expected": 24},
         "the stabilizer of a point of the position 8a has order 12, where 24 is expected."))
    kept = dict(group._wyckoff_audit)
    try:
        for bad, words in planted:
            group._wyckoff_audit["audit"] = {"ok": False, "offenders": (dict(bad, letter="a"),)}
            refused = tqg.wyckoff_sites_tables(group)
            assert refused.gate == "wyckoff_setting" and words in refused.reason, refused.reason
    finally:
        group._wyckoff_audit.clear()
        group._wyckoff_audit.update(kept)


# ---------------------------------------------------------------------------
# 2.  a model -> its SRSIs
# ---------------------------------------------------------------------------

#: What each case's rows localize to, and the model's invariants.  Per row, in
#: the order of ``r.wyckoff_orbits``: its Wyckoff position, its orbitals and its
#: site irreps; the position its localization is at, and that localization's
#: terms, each a site irrep at a position (none for a row whose site irreps
#: induce a composite band representation); the row's integer-valued invariants
#: and its invariants valued modulo n.  Then the model's two.
MODEL_SRSIS = {
    # -E1 since 2026-09-10 (the served matrices carry the tables' lift); the first
    # two Z-SRSIs count the two 8a spinor EBRs, so the total moved with the name
    "FKM diamond": ((("8a", (0, 1), "-E1", "8a", (("-E1", "8a"),),
                      ("1", "0", "0", "0", "0", "0", "0", "0"), ()),),
                    ("1", "0", "0", "0", "0", "0", "0", "0"), ()),
    "FKM strong": ((("6c", (0, 1), "-E1", "3a", (("-E1g", "3a"), ("-E1u", "3a")),
                     ("0", "0", "1", "1", "0", "0", "0", "0"), ()),),
                   ("0", "0", "1", "1", "0", "0", "0", "0"), ()),
    "Kane-Mele odd": ((("1c", (0,), "-E1", "1c", (("-E1", "1c"),), ("0", "1", "0", "-1"), ()),
                       ("1b", (1,), "-E1", "1b", (("-E1", "1b"),), ("0", "1", "-1", "0"), ())),
                      ("0", "2", "-1", "-1"), ()),
    # two Wyckoff orbits on the one orbit of sites at 1a, each its own row
    "lin3g": ((("1a", (0,), "-E1", "1a", (("-E1", "1a"),), ("0", "1", "0"), (0, 0)),
               ("1a", (1,), "-E2", "1a", (("-E2", "1a"),), ("1", "0", "0"), (0, 0)),
               ("2c", (2, 3), "-E", "2c", (("-E", "2c"),), ("1", "1", "0"), (0, 1))),
              ("2", "2", "0"), (0, 1)),
    # a composite multiplet beside another on one site: each row reads its own
    # orbitals, and the composite one its own site irrep
    "P-43m 3c": ((("3c", (0, 4, 8), "B2", "3c", (("B2", "3c"),),
                   ("0", "-1", "1", "1", "0", "1", "1", "-1", "0", "0"), ()),
                  ("3c", (1, 2, 3, 5, 6, 7), "E", "3c", (),
                   ("1", "1", "0", "1", "1", "-1", "-1", "0", "0", "0"), ())),
                 ("1", "0", "1", "2", "1", "0", "0", "-1", "0", "0"), ()),
}

#: The points each row's symmetry data vector is identified at: every tabulated
#: point of a fully periodic crystal's group, and the in-plane ones of a
#: layer's -- ``#156.50``'s ``A``/``H``/``L`` rows have no ``k`` in a
#: monolayer's zone.
ORBIT_POINTS = {"FKM diamond": ("GM", "L", "W", "X"), "FKM strong": ("GM", "F", "L", "T"),
                "Kane-Mele odd": ("GM", "K", "M"), "lin3g": ("GM", "M", "X"),
                "P-43m 3c": ("GM", "M", "R", "X")}

#: How many bands each fixture has in total (the identity of § 4 is asserted on
#: *all* of them).
ALL_BANDS = {"lin3g": 8, "Kane-Mele odd": 4}

#: The names of the **Z**-valued rows of each fixture's group, as literals.  The
#: Z rows only: ``#99.164`` has five SRSIs and two of them are Z_2, so lin3g's
#: three names stop short of ``theta_4`` and ``theta_5``.  Both routes carry this
#: tuple and both build it by the same expression, so neither is the other's
#: oracle -- the literal is.
Z_ROW_NAMES = {
    "99.164": ("theta_1", "theta_2", "theta_3"),
    "156.50": ("theta_1", "theta_2", "theta_3", "theta_4"),
}

#: The record each case reads: a fixture's name, and the key into it where the
#: fixture holds more than one.
_CASES = {"FKM diamond": ("fkm_undistorted", None), "FKM strong": ("fkm_strong", None),
          "lin3g": ("lin3g_declared_cross_checked", None),
          "Kane-Mele odd": ("km_realized", "odd"),
          "P-43m 3c": ("p43m_composite", None)}


def _record(request, name):
    """The realized group a case names, built only when that case asks for it."""
    fixture, key = _CASES[name]
    got = request.getfixturevalue(fixture)
    return got if key is None else got[key]


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_SRSIS))
def test_srsi_of_model_localizes_every_orbit_and_pins_the_total(request, name):
    """Every Wyckoff orbit's position, its localization and the model's SRSIs, measured.

    A row for each row of ``r.wyckoff_orbits``, its orbitals and site irreps
    the same.  The positions are route G's (the § 1 tests above) and the
    localizations are route C's — the row's own symmetry data vector,
    decomposed over the EBR columns of a position on the closure of its
    variety — and the two must land in the same place, which is the one safety
    gate left in this route.  The symmetry data vector is identified only at
    the points the model can sample, which for a layer are its in-plane ones.
    The distorted Fu-Kane-Mele orbit is on a line the tables list no EBR at, so
    it is localized by the adiabatic shift, and the summary says so.  Each
    row's invariants are the tables' own of its site irreps at its own
    position, the composite one's included.  The totals are pinned as strings
    so a Fraction that stopped being integral would be visible.
    """
    realized = _record(request, name)
    want_rows, want_total, want_zn = MODEL_SRSIS[name]
    got = tqr.srsi_of_model(realized)
    assert got, str(got)
    assert got.group is realized.group
    tables = tqr.srsi_tables(realized.group)
    assert len(got.wyckoff_orbits) == len(want_rows) == len(realized.wyckoff_orbits)
    for row, given, want in zip(got.wyckoff_orbits, realized.wyckoff_orbits, want_rows):
        wyckoff, orbitals, label, at, terms, theta_z, theta_zn = want
        assert (row.wyckoff, row.orbitals, row.label) == (wyckoff, orbitals, label), (name, row)
        assert (row.terms, row.site_symmetry, row.band_representation) == (
            given.terms, given.site_symmetry, given.band_representation), name
        assert row.numerical.kpoints == ORBIT_POINTS[name], (name, row.numerical.kpoints)
        chosen = row.numerical.chosen
        assert chosen is row.numerical.localizations[0]
        assert chosen.wyckoff == at, (name, chosen)
        assert tuple((t.irrep, t.wyckoff) for t in chosen) == terms, (name, chosen)
        assert all(t.multiplicity == 1 for t in chosen)
        assert (row.numerical.composite is None) == bool(terms), name
        assert row.numerical.symmetry_data
        assert tuple(str(v) for v in row.theta_Z.values()) == theta_z, name
        assert tuple(v.value for v in row.theta_Zn.values()) == theta_zn, name
        # the row's invariants are the tables' own of its site irreps there
        assert tables.theta(row.terms) == tqr.Invariants(row.theta_Z, row.theta_Zn), name
        # the agreement over the entries themselves, not the record's flag: every
        # localization of the row lands on the same SRSIs
        assert row.numerical.agree and all(
            e.theta_Z == row.theta_Z and e.theta_Zn == row.theta_Zn
            for e in row.numerical.localizations), row.numerical.localizations
    assert tuple(str(v) for v in got.theta_Z.values()) == want_total, name
    assert tuple(v.value for v in got.theta_Zn.values()) == want_zn, name
    assert all(v.denominator == 1 for v in got.theta_Z.values()), "a model is an atomic limit"
    # the record prints what it did
    text = got.summary()
    assert realized.group.number_str in text
    names = tuple(got.theta_Z)
    span = f"{names[0]} to {names[-1]}" if len(names) > 2 else ", ".join(names)
    assert f"the whole model: {span} = ({', '.join(want_total)})" in text
    for row in got.wyckoff_orbits:
        assert f"{row.wyckoff} ({row.site_symmetry}): " in text
    if any(row.numerical.composite is None and row.wyckoff not in row.slides_onto
           for row in got.wyckoff_orbits):
        assert "slides onto 3a and 3b" in text, name


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(ALL_BANDS))
def test_a_model_s_z_srsis_are_the_band_route_s_on_all_of_its_bands(request, name):
    """The identity that ties the two routes, on lin3g and the odd Kane-Mele regime.

    A model *is* an atomic configuration, so the SRSIs read off its orbitals and
    the ones read off the symmetry content of all its bands are the same
    numbers — computed along completely different paths (site orbits and a
    nonnegative-integer decomposition on one side, the Smith form and a rational
    solve of ``BR . p = y`` on the other).  It is the sharpest single check
    either route has.

    The two records' names are built by the identical expression off the same
    table, so comparing them to each other would catch nothing; they are compared
    to the literals of ``Z_ROW_NAMES`` instead.
    """
    realized = _record(request, name)
    group = realized.group
    model_srsis = tqr.srsi_of_model(realized)
    assert model_srsis, str(model_srsis)
    n_bands = ALL_BANDS[name]
    bands = tqr.srsi_of_bands(tqw.hsp_table(realized, bands=n_bands))
    assert bands, str(bands)
    assert model_srsis.theta_Z == bands.theta_Z, name
    assert all(v.denominator == 1 for v in bands.theta_Z.values()), \
        "all the bands of a model are an atomic limit"
    want_names = Z_ROW_NAMES[group.number_str]
    assert tuple(bands.theta_Z) == want_names, name
    assert tuple(model_srsis.theta_Z) == want_names, name


def test_the_safety_gate_can_tell_a_setting_problem_from_a_modeling_one(
        lin3g_declared_cross_checked):
    """Which letters a content *does* fit, when the geometry's letters do not.

    Two different diagnoses hide behind "the decomposition failed": a content
    that is a band representation of *nothing* (a modelling statement,
    ``wyckoff_orbit_not_localizable``) and one that is a band representation of a
    position the geometry says the orbitals are not on (a **setting** statement,
    ``wyckoff_setting``).  ``_letters_that_do_fit`` is what separates them, and
    it is measured here on lin3g's own EBR columns: each ``1a`` row fits at
    ``a`` and nowhere else, each ``1b`` row at ``b`` and nowhere else, and the
    ``2c`` row at all three — ``2c``'s site symmetry ``mm2`` is a proper subgroup
    of the ``4mm`` at ``1a``/``1b``, so its induced row is the sum of either
    maximal pair.  A geometry saying ``a`` against a content that fits only at
    ``b`` is exactly the disagreement the gate names.
    """
    group = lin3g_declared_cross_checked.group
    data = _tables._ebr_data(group)
    labels = list(data["labels_physics_convention"])
    matrix = np.asarray(data["matrix"], dtype=int)
    points = ("GM", "M", "X")
    got = {}
    for column, (name, position) in enumerate(data["ebrs"]):
        counts = {labels[i]: int(matrix[i, column]) for i in range(len(labels))}
        got[(name, position.split("(")[0])] = tqr._letters_that_do_fit(
            data, counts, points, 4)
    assert got[("-E1↑G(2)", "1a")] == ("a",)
    assert got[("-E2↑G(2)", "1a")] == ("a",)
    assert got[("-E1↑G(2)", "1b")] == ("b",)
    assert got[("-E2↑G(2)", "1b")] == ("b",)
    assert got[("-E↑G(4)", "2c")] == ("a", "b", "c")
    # ... and a content that fits nowhere gives the empty tuple, which is what
    # sends the caller to `wyckoff_orbit_not_localizable` instead
    assert tqr._letters_that_do_fit(data, {labels[0]: 1}, points, 4) == ()


# ---------------------------------------------------------------------------
# 3.  a band subset -> its Z-SRSIs
# ---------------------------------------------------------------------------

def test_srsi_of_bands_reproduces_the_fu_kane_mele_anchors(fkm_strong):
    """The paper's theorem through the public route: fractional iff indicated.

    ``Non-zero SIs <=> fractional Z-SRSIs``, and the **order of the class is the
    common denominator**: quarters for the valence pair of the strong phase,
    whose class has order 4.  The record carries both halves, so the statement
    can be read off one object.
    """
    content = tqw.hsp_table(fkm_strong, bands=2)
    got = tqr.srsi_of_bands(content)
    assert got, str(got)
    assert got.indicator.values == (0, 1) and got.indicator.order == 4
    assert tuple(str(v) for v in got.theta_Z.values()) == (
        "-1/4", "0", "1/2", "1", "1/4", "-1/2", "-1/4", "1/4")
    # the paper's theorem, as a two-route check: a real-space denominator against
    # a momentum-space class order.  (Whether the class is nonzero is
    # `any(indicator.values)` and `order` is an lcm over the same values, so a
    # third term would be free.)
    assert math.lcm(*(v.denominator for v in got.theta_Z.values())) == 4
    assert got.group is fkm_strong.group
    assert got.line() == ("theta_1 to theta_8 = (-1/4, 0, 1/2, 1, 1/4, -1/2, -1/4, 1/4), "
                          "fractional; class (0, 1) in Z2 x Z4, of order 4"), \
        "the record on one line, the invariants and the class side by side"
    assert "Z-valued stable real-space invariants of the bands" in got.summary()
    assert "  fractional   (.theta_Z)" in got.summary()
    # the record's own one-line rendering: the group, the theta_Z and whether
    # they are fractional, which is what a reader checks first
    assert repr(got) == ("BandSRSIs(R-3m1' #166.98, theta_1 to theta_8 = "
                         "(-1/4, 0, 1/2, 1, 1/4, -1/2, -1/4, 1/4), fractional)")
    # one rational site-symmetry vector of many, keyed as the tables take it, and
    # the tables read the same invariants off it
    assert tqr.srsi_tables(fkm_strong.group).theta(got.numerical.p).theta_Z == got.theta_Z


def test_srsi_of_bands_reproduces_the_lin3g_and_kane_mele_anchors(
        lin3g_declared_cross_checked, km_realized):
    """Linearity, and two obstructed limits at different Wyckoff positions.

    ``P6 = P4 + P2`` holds as SRSIs because Theta and the rational solve are both
    linear; and Kane-Mele's two regimes, which decompose at *different* Wyckoff
    positions, carry different Z-SRSIs — the paper's ``Pmm2`` distinction on a
    model.

    The three Z-SRSI triples are pinned, so their sum is arithmetic on the page;
    the **premise** that makes it a statement about linearity is that the three
    blocks' measured symmetry data vectors add, and that is what is asserted.
    """
    blocks = {}
    for name, bands in (("P6", P6), ("P4", list(P4)), ("P2", list(P2))):
        got = tqr.srsi_of_bands(tqw.hsp_table(lin3g_declared_cross_checked, bands=bands))
        assert got, (name, str(got))
        blocks[name] = got
        # integral theta_Z, so the common denominator is 1: the other half of
        # the theorem the Fu-Kane-Mele anchors read at 4 and 2
        assert all(v.denominator == 1 for v in got.theta_Z.values()), name
    assert tuple(str(v) for v in blocks["P4"].theta_Z.values()) == ("1", "1", "0")
    assert tuple(str(v) for v in blocks["P2"].theta_Z.values()) == ("0", "0", "1")
    assert tuple(str(v) for v in blocks["P6"].theta_Z.values()) == ("1", "1", "1")
    # the premise: the measured symmetry data vectors of the three blocks add
    assert tuple(blocks["P6"].indicator._y) == tuple(
        a + b for a, b in zip(blocks["P4"].indicator._y, blocks["P2"].indicator._y))

    got = {}
    for regime in ("odd", "even"):
        km = km_realized[regime]
        out = tqr.srsi_of_bands(tqw.hsp_table(km, bands=2))
        assert out, (regime, str(out))
        got[regime] = tuple(str(v) for v in out.theta_Z.values())
        assert all(v.denominator == 1 for v in out.theta_Z.values())
    # different Wyckoff positions, different real-space invariants -- the two
    # literals above are the statement; `got["odd"] != got["even"]` beside them
    # would be a remark about this file
    assert got["odd"] == ("0", "1", "0", "0")
    assert got["even"] == ("0", "1", "-1", "0")
