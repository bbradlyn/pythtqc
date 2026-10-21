r"""Wave A5.5: the stable-RSI paper's vocabulary as canonical, ``irreptables``' as backend.

The owner's ruling of 2026-09-03, verbatim: *"lets go with the papers vocabulary
as canonical, with a backend conversion function to the irreptables conventions
for when that is needed."*  This file is what says the package does that and
that the doing is licensed by measurement rather than by preference.  Five
statements, in the order the wave's spec puts them:

1. **Coverage.**  The paper's symmetry-indicator rows cover *exactly* the grey
   systems whose EBR matrix has a nontrivial Smith form -- none missing, none
   extra -- so every class this package can serve has a name in the canonical
   vocabulary, and no name belongs to a group with no class.  The thirteen
   space groups the tables omit are checked too: they have no indicator group
   to speak of, in either valuedness.
2. **The determinism gate.**  Serving a class in a vocabulary is only honest if
   the vocabulary can tell the classes apart.  Every element of ``{BS}/{AI}`` of
   every one of the 170 systems is enumerated -- through the rational solve,
   which is a *different* computation from the homomorphism arithmetic the
   package caches -- and the map class -> canonical values is checked to be
   injective.  The converse, that adding an EBR column to a symmetry vector
   moves no canonical value, is the statement that every served form vanishes
   on every EBR, which the coverage sweep checks.
3. **The value gate.**  Every name the conversion pairs agrees in value with its
   counterpart on every EBR column, on random atomic ``p`` and on every
   generator of the indicator group, in all 170 systems.  S1's tie test over 20
   groups, promoted to the whole table and to package data.  A pairing that
   failed this is a bug in the map and never a tolerance.
4. **The fixtures**, in the paper's names: Fu-Kane-Mele's strong and trivial
   phases in ``#166.98``, lin3g's ``P2`` where the indicator group is trivial,
   and a **unitary** group, where the shipped tables have nothing to say and the
   record says so.  The Kane-Mele layer in ``#191.234``, where the two
   vocabularies spell names differently, is read in the paper's names by
   ``tests/test_ebr.py``'s pure-2D layer test, on the record its stack route
   returns.
5. **The backend conversion applied**: ``irreptables_named`` on those fixtures
   is ``irreptables``' own arithmetic, the same numbers computed the other way
   round -- through momentum-space forms rather than real-space ones; the
   spelling map applied to real data, ``as_irreptables()``, is the layer's.

What A5.5 measured that its spec did not expect is in statement 3: with the
qualifier-order rule in the map, **nothing is paper-only**.  The "nine paper
rows with no ``irreptables`` counterpart" S1b recorded is the *derivation's*
count (rows shipping ``derived: null``, where only the literal and
transliterated spellings were tried), not a name-correspondence count.  Those
nine are among the **eleven** rows this wave's reorder rule pairs; the other two
of the eleven (SGs 124, 128) and SG 130's ``Theta2``, which the alphabet rule
pairs, are exactly the three S1b derived from the group's own Smith coordinate
instead.  Nine + three = the twelve rows of that family, and A5.5's map pairs
all twelve.
"""

import collections
import importlib.util
import itertools
import os
import pathlib
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc.refusal import Refusal
from pythtqc import InternalError
from pythtqc import _smith
from pythtqc import _spacegroup
from pythtqc import _srsitables

import tqc_models as tqm                                            # noqa: F401
from conftest import FKM_BOND, OMITTED, fkm_regime, patch_everywhere


def _converter():
    """``scripts/convert_srsi_tables.py``, loaded **by path**.

    Only for its ``grey_numbers()`` -- the grey (type-II) BNS number of every
    space group, *measured* from ``irreptables``' own correpresentation tables
    rather than typed, which is what the coverage statement needs for the
    thirteen groups the SRSI tables omit (they have no shipped record to read a
    ``number_str`` off).  Loaded by path for the same reason ``conftest.py``
    loads ``tqc_models`` that way: independence from ``sys.path`` order.
    """
    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "scripts",
        "convert_srsi_tables.py"))
    spec = importlib.util.spec_from_file_location("convert_srsi_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# the one pass over the tables, and the three sweeps on it
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep(every_table):
    """One walk over the 434 systems carrying every number statements 1-3 assert.

    Written as a fixture rather than repeated per test because the expensive
    half is the walk, not the arithmetic: the determinism gate enumerates all
    1355 classes of the 170 indicator groups **through the rational solve**, and
    the value gate evaluates both vocabularies on every EBR column, four random
    atomic ``p`` and every generator, in the same pass.

    Every statement below is counted as well as checked -- ``vanish_probes``,
    ``compared``, ``classes`` -- because an empty-failure list is only a
    measurement if something was measured, and the consumers pin the counts.
    ``gate[(sg, spinor)]`` is the per-group
    ``(divisors, n_classes, n_collisions)`` this walk measures, which is what
    ``canonical_determinism``'s served record is compared against.
    """
    from irreptables.symmetry_indicators import load_si_table

    # irreptables' named forms: one file per valuedness, read once for the walk
    shipped_tables = {spinor: load_si_table(spinor=spinor, magnetic=True)
                      for spinor in (False, True)}
    rng = np.random.default_rng(20260905)
    out = {
        "systems": 0, "with_rows": set(), "with_divisors": set(),
        "vanish_failures": [], "vanish_probes": 0, "generators": 0, "classes": 0,
        "non_injective": [], "compared": 0, "disagreements": [], "named_none": [],
        "relations": collections.Counter(), "paper_only": [],
        "irreptables_only": [], "modulus_mismatch": [], "reordered": [],
        "paper_rows": 0, "shipped_rows": 0, "gate": {},
    }
    for table, data in every_table:
        key = (table.sg, table.spinor)
        out["systems"] += 1
        u, v, d, matrix = _smith._smith_system(data, table.number_str)
        n_row, n_col = len(matrix), len(matrix[0])
        slots = [i for i, x in enumerate(d) if x > 1]
        divisors = [int(d[i]) for i in slots]
        if table.indicators:
            out["with_rows"].add(key)
        if divisors:
            out["with_divisors"].add(key)

        # (a) every SERVED row is an indicator of its own group
        for form in table.indicators:
            for column in range(n_col):
                total = sum((c * Fraction(table.theta_on_columns[i - 1][column])
                             for i, c in form.coefficients.items()), Fraction(0))
                out["vanish_probes"] += 1
                if total.denominator != 1 or int(total) % form.mod:
                    out["vanish_failures"].append(
                        (table.sg, table.spinor, form.name, column, str(total)))

        # (d) the correspondence, per group and per name
        entry = shipped_tables[table.spinor].get(table.number_str, {})
        forms = dict(entry.get("indicators", {}))
        out["paper_rows"] += len(table.indicators)
        out["shipped_rows"] += len(forms)
        paired = set()
        for form in table.indicators:
            spelled, relation = _srsitables._spelling(form.name)
            if spelled not in forms:
                out["relations"]["paper_only"] += 1
                out["paper_only"].append((table.sg, table.spinor, form.name, spelled))
                continue
            paired.add(spelled)
            out["relations"][relation] += 1
            if relation == "reordered":
                out["reordered"].append((table.sg, form.name, spelled))
            shipped_mod = float(forms[spelled]["mod"])
            if int(round(shipped_mod)) != form.mod:
                out["modulus_mismatch"].append(
                    (table.sg, table.spinor, form.name, spelled, form.mod, shipped_mod))
        for spelled in sorted(set(forms) - paired):
            out["relations"]["irreptables_only"] += 1
            out["irreptables_only"].append((table.sg, table.spinor, spelled))

        out["gate"][key] = (tuple(divisors), 0, 0)
        if not divisors:
            continue
        u_inv = _smith._unimodular_inverse(u)

        # (b) the determinism gate, through the rational solve
        generator_values = []
        for slot in slots:
            y = [Fraction(int(u_inv[i][slot])) for i in range(n_row)]
            generator_values.append(
                table.indicators_of_columns(_srsitables._rational_p(u, v, d, y)))
            out["generators"] += 1
        seen, n_here, collided_here = {}, 0, 0
        for combination in itertools.product(*(range(x) for x in divisors)):
            y = [sum(c * int(u_inv[i][s]) for c, s in zip(combination, slots))
                 for i in range(n_row)]
            solved = table.indicators_of_columns(
                _srsitables._rational_p(u, v, d, [Fraction(x) for x in y]))
            # ... and the same class through the homomorphism arithmetic the
            # package's own gate uses.  Two computations, one answer.
            combined = {f.name: sum(c * generator_values[g][f.name]
                                    for g, c in enumerate(combination)) % f.mod
                        for f in table.indicators}
            assert solved == combined, (table.sg, table.spinor, combination)
            out["classes"] += 1
            n_here += 1
            signature = tuple(sorted(solved.items()))
            if signature in seen:
                collided_here += 1
                out["non_injective"].append(
                    (table.sg, table.spinor, seen[signature], combination, solved))
            seen[signature] = combination
        out["gate"][key] = (tuple(divisors), n_here, collided_here)

        # (c) the value gate, on every name the map pairs
        probes = [[1 if j == c else 0 for j in range(n_col)] for c in range(n_col)]
        probes += [list(map(int, rng.integers(0, 4, size=n_col))) for _ in range(4)]
        vectors = [([Fraction(sum(matrix[i][j] * p[j] for j in range(n_col)))
                     for i in range(n_row)], p) for p in probes]
        for slot in slots:
            y = [Fraction(int(u_inv[i][slot])) for i in range(n_row)]
            vectors.append((y, _srsitables._rational_p(u, v, d, y)))
        shim = _srsitables._TableKey(table.number_str, table.spinor)
        for y, p in vectors:
            named, _touched, note, _coeffs = tqe._named_forms(
                shim, data, matrix, y, "II")
            if named is None:
                out["named_none"].append((table.sg, table.spinor, note))
                break
            paper = table.indicators_of_columns(p)
            for form in table.indicators:
                spelled, _relation = _srsitables._spelling(form.name)
                if spelled not in named:
                    continue
                value = paper[form.name]
                assert isinstance(value, int), (table.sg, form.name, value)
                out["compared"] += 1
                if value % form.mod != named[spelled] % form.mod:
                    out["disagreements"].append(
                        (table.sg, table.spinor, form.name, spelled,
                         value, named[spelled], form.mod))
    return out


# ---------------------------------------------------------------------------
# 1.  coverage
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_the_paper_s_indicator_rows_cover_exactly_the_groups_with_a_class(sweep):
    """170 = 170: an indicator group iff a canonical name for it.

    This is what makes the paper's vocabulary usable as *the* vocabulary rather
    than as one more table to consult.  Measured 2026-09-05 over all 434
    tabulated systems: the systems whose EBR matrix has a nontrivial Smith form
    and the systems the paper prints indicator rows for are the **same set**,
    170 of them (53 single-valued + 117 double-valued).  A group in the first
    set and not the second would be a class with no canonical word for it, which
    :func:`pythtqc.srsi._canonical_reading` raises ``InternalError`` on; a
    group in the second and not the first would be a name for a class that does
    not exist.
    """
    assert sweep["systems"] == 434
    assert len(sweep["with_rows"]) == 170, sorted(sweep["with_rows"])
    assert sweep["with_rows"] == sweep["with_divisors"], (
        sorted(sweep["with_rows"] ^ sweep["with_divisors"]))
    assert sum(1 for _sg, sp in sweep["with_rows"] if not sp) == 53, "single-valued"
    assert sweep["paper_rows"] == 390 and sweep["shipped_rows"] == 390


@pytest.mark.data
def test_the_thirteen_omitted_space_groups_have_no_class_to_speak():
    """The other half of the coverage statement, on the groups with no table.

    Thirteen space groups carry no SRSI row at all, by the tables' own caption,
    so they carry no indicator row either -- and the coverage sweep above cannot
    see them, because it walks the systems that *have* a table.  Measured here
    instead, on their grey EBR matrices directly: **every one of the 26 systems
    has a trivial Smith form**, in both valuednesses.  So there is no class
    anywhere without a canonical name for it, and the ``InternalError`` in
    :func:`pythtqc.srsi._canonical_reading` for a group whose table is absent is
    a guard that the shipped data cannot trip.
    """
    grey = _converter().grey_numbers()
    seen = 0
    for sg in OMITTED:
        for spinor in (False, True):
            assert not tqr.srsi_tables(sg, spinor=spinor)
            data = tqg.ebr_data(_srsitables._TableKey(grey[sg], spinor))
            assert data, (sg, spinor, data)
            assert _smith._divisors_of(data, grey[sg]) == (), (sg, spinor)
            seen += 1
    assert seen == 26


@pytest.mark.data
def test_every_served_indicator_row_vanishes_on_every_ebr_of_its_group(sweep):
    """A form that is not zero on an atomic limit is not an indicator.

    The row :meth:`SRSITables.indicators_of` actually evaluates -- the printed
    one, or S1b's derived one for the four expressions of SGs 147, 162 and 163
    without SOC where the printed one is wrong -- must vanish modulo its own
    modulus on every EBR column of its group.  Measured over all 434 systems:
    **5003 (form, EBR column) evaluations over the 390 served rows, zero
    failures**.  The count is the witness and is pinned here: an empty
    failure list is also what a pass over no rows at all produces.  The same
    check runs per group inside :func:`pythtqc.srsi._canonical_gate`, and a
    failure there raises ``InternalError`` rather than serving the number.
    """
    assert sweep["vanish_probes"] == 5003, "one per (served form, EBR column)"
    assert sweep["vanish_failures"] == []


# ---------------------------------------------------------------------------
# 2.  the determinism gate
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_the_paper_s_names_determine_the_class_in_every_one_of_the_170(sweep):
    """The gate that licenses serving a class in this vocabulary at all.

    Enumerated: one representative ``y = sum_i c_i u^-1 e_slot_i`` per element of
    ``{BS}/{AI}``, its canonical values from the exact rational solve of
    ``BR.p = y``.  Measured 2026-09-05 over the 170 systems with an indicator
    group: **263 generators, 1355 classes, and the map class -> canonical values
    is injective in every single system** -- no two classes anywhere share every
    canonical value.  So the paper's names carry exactly the information the
    Smith coordinates do, which is what makes ``canonical`` a *coordinate change*
    on the class rather than a lossy label.

    Two computations agree on all 1355: the rational solve here, and the
    homomorphism arithmetic :func:`pythtqc.srsi._canonical_gate` caches (an
    indicator is a homomorphism onto ``Z_n``, so a class's values are its
    generators' values combined modulo ``n``).  The package uses the second
    because it costs one solve per generator instead of one per class; this test
    is why it is allowed to.
    """
    assert sweep["generators"] == 263
    assert sweep["classes"] == 1355
    assert sweep["non_injective"] == [], sweep["non_injective"][:5]


@pytest.mark.data
def test_the_gate_the_package_caches_is_the_measured_one(every_table, sweep):
    """:func:`pythtqc.srsi.canonical_determinism` serves what the sweep measured.

    The per-group record the serve path reads, checked against the sweep's own
    ``(divisors, n_classes, n_collisions)`` for that group -- the enumeration
    through the rational solve, which is a different computation from the
    homomorphism arithmetic the gate caches -- on the largest indicator group in
    the tables (``#191.234``, ``Z6 x Z12``, 72 classes), on the anchor
    ``#166.98`` (``Z2 x Z4``, 8) and on ``#2.5`` (``Z2 x Z2 x Z2 x Z4``, 32).
    The pinned tuple beside each is the regression anchor, so a drift in *both*
    computations at once still fails.

    What is *not* asserted here is the gate's injectivity verdict read back out
    of its own record: that is
    ``test_the_gate_measures_injectivity_rather_than_assuming_it``, which hands
    the gate a vocabulary that cannot separate the classes and requires it to
    say so.
    """
    lookup = {(t.sg, t.spinor): t for t, _d in every_table}
    for sg, spinor, divisors, n_classes, n_forms in ((191, True, (6, 12), 72, 7),
                                                     (166, True, (2, 4), 8, 4),
                                                     (2, True, (2, 2, 2, 4), 32, 4)):
        key = (sg, spinor)
        measured = sweep["gate"][key]
        assert measured == (divisors, n_classes, 0), (key, measured)
        gate = tqr.canonical_determinism(lookup[key])
        assert gate["available"], (sg, gate)
        assert (gate["divisors"], gate["n_classes"], len(gate["collisions"])) == measured, \
            "the served record is the sweep's own measurement of this group"
        assert len(gate["names"]) == n_forms
        assert gate["injective"] and gate["collisions"] == ()
        assert gate["vanishes"] and gate["ebr_failures"] == ()
        # `integral` re-derived rather than read back: a fractional generator value
        # is what it reports, so the values themselves are what says it
        assert gate["integral"] and all(
            isinstance(value, int) for record in gate["generators"]
            for value in record.values()), (sg, gate["generators"])


def test_the_gate_measures_injectivity_rather_than_assuming_it(monkeypatch):
    """The negative control: a vocabulary that cannot separate the classes.

    ``injective`` and ``collisions`` are the two fields the serve path refuses
    on, and no system in the shipped tables fails them -- so nothing in the
    suite would notice if the gate *asserted* injectivity instead of measuring
    it (``n_classes = prod(divisors); collisions = ()`` passes every other test
    in this file).  The control is a table the gate must refuse.

    ``#166.98`` served with **two forms that are the same linear form** -- ``z4``
    and a renamed copy of it -- is that table, and it is a physical statement
    rather than a contrivance: ``z4`` is the strong index, and on its own it
    cannot see the weak one.  So the trivial class ``(0, 0)`` and the weak class
    ``(1, 0)`` -- Fu-Kane-Mele's two phases with ``z4 = 0`` -- read
    identically, and the gate must report exactly those four pairs: the three
    ``z2w`` rows it was stripped of are what tell them apart.  Note which
    verdicts do *not* move -- the copy is a real indicator with integer values,
    so ``vanishes``, ``integral`` and ``n_classes`` are unchanged and
    injectivity is the only thing under test.

    ``_canonical_gate`` reads its table back by number and is ``lru_cache``\\ d,
    so the gate is called through its unwrapped self: the poisoned record must
    not be what the next test's serve path finds in the cache.
    """
    real = tqr.srsi_tables(166, spinor=True)
    assert tqr.canonical_determinism(real)["injective"] is True, \
        "the shipped vocabulary of #166.98 does separate its eight classes"

    z4 = next(form for form in real.indicators if form.name == "z4")
    twin = replace(z4, name="z4_twin", name_tex="z_{4}^{\\rm twin}")
    doubled = replace(real, indicators=(z4, twin))

    patch_everywhere(monkeypatch, "_table_of_number", lambda sg, spinor: doubled)
    gate = _srsitables._canonical_gate.__wrapped__(real.number_str, real.spinor)

    assert gate["available"] and gate["names"] == ("z4", "z4_twin")
    assert gate["divisors"] == (2, 4) and gate["n_classes"] == 8, \
        "the same eight classes: only the vocabulary was reduced"
    assert gate["vanishes"] and gate["integral"], \
        "both forms are indicators; injectivity is the only thing that fails"
    assert gate["injective"] is False, "z4 alone does not determine the class"
    assert [(a, b) for a, b, _values in gate["collisions"]] == [
        ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (1, 3))], \
        "every class and its weak partner, named as a pair"
    assert gate["collisions"][0][2] == {"z4": 0, "z4_twin": 0}, \
        "and the reading they share: the trivial and weak phases both have z4 = 0"


# ---------------------------------------------------------------------------
# 3.  the value gate, and the correspondence it licenses
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_the_two_vocabularies_agree_in_value_on_every_name_the_map_pairs(sweep):
    """The gate under the spelling map: same name, same number, everywhere.

    ``irreptables`` ships named linear forms in the **momentum-space** symmetry
    vector; the paper gives named forms in the **real-space** SRSIs.  Where the
    map pairs two names they must give the same number -- on an atomic limit,
    and on a band structure that is not one.  The vectors are every EBR column
    (both sides zero, the easy half), four random nonnegative-integer ``p`` per
    system, and the generators of ``{BS}/{AI}`` themselves, where both sides are
    nonzero by construction.

    Measured 2026-09-05 over all 170 systems: **7295 comparisons, zero
    disagreements**, and ``irreptables`` served a usable form set in every one.
    S1's tie test made the same measurement over 20 chosen groups (3796
    comparisons); this is it over the whole table, which is what promoting the
    map from a test's transliteration dict into package data requires.
    """
    assert sweep["named_none"] == []
    assert sweep["compared"] == 7295
    assert sweep["disagreements"] == [], sweep["disagreements"][:5]


@pytest.mark.data
def test_the_name_correspondence_is_a_bijection_over_the_whole_table(sweep):
    """317 literal + 62 transliterated + 11 reordered = 390 = 390.

    The tally the ruling rests on, measured per group and per name over the 170
    systems: every one of the paper's 390 indicator rows pairs with exactly one
    of ``irreptables``' 390, and **nothing is left over on either side** -- no
    ``paper_only``, no ``irreptables_only``, and no pair whose moduli disagree.
    The three relations are the three differences between the vocabularies: the
    alphabet (``z3m,pi`` against ``z3m,π``), the order of the qualifiers on one
    family, and one letter's case.

    **This corrects an expectation, and the correction is the point.**  S1b
    recorded "nine paper rows with no ``irreptables`` counterpart"; that is the
    *derivation's* count -- rows that ship ``derived: null`` because the
    derivation tried only the literal and transliterated spellings -- and not a
    statement about the vocabularies.  Those nine (SGs 10, 83, 84, 87 and 175,
    all without SOC) are a subset of the **eleven** rows the reorder rule pairs
    here; the other two of the eleven (SGs 124 and 128), together with SG 130's
    ``Theta2`` that the alphabet rule pairs, are exactly the three S1b derived
    from the group's own Smith coordinate instead.  Both counts are pinned --
    the eleven and the nine's names in this test, the derivation's nine by the
    ``tqc_srsi_tables`` golden's "9 with no derived row" -- so neither can drift.
    """
    assert sweep["relations"]["paper_only"] == 0, sweep["paper_only"]
    assert sweep["relations"]["irreptables_only"] == 0, sweep["irreptables_only"]
    assert sweep["modulus_mismatch"] == []
    assert sweep["relations"]["literal"] == 317
    assert sweep["relations"]["transliterated"] == 62
    assert sweep["relations"]["reordered"] == 11
    assert set(sweep["relations"]) <= set(tqr.NAME_RELATIONS), (
        "and the only relations the map produced over the whole table are "
        "published ones -- this is where that claim can fail")
    assert sorted(sweep["reordered"]) == sorted([
        (10, "z2,+", "z2+"), (10, "z2,-", "z2-"),
        (83, "delta2+,pi", "δ2,π+"), (83, "delta2-,pi", "δ2,π-"),
        (84, "delta2+,0", "δ2,0+"), (84, "delta2-,0", "δ2,0-"),
        (87, "delta2+,0", "δ2,0+"),
        (124, "delta2+,pi", "δ2,π+"), (128, "delta2+,pi", "δ2,π+"),
        (175, "z2,+", "z2+"), (175, "z2,-", "z2-")])
    # ... and the nine S1b could not derive against a name are the subset of
    # those eleven whose groups have no Smith coordinate to fall back on
    nine = {(10, "z2,+"), (10, "z2,-"), (83, "delta2+,pi"), (83, "delta2-,pi"),
            (84, "delta2+,0"), (84, "delta2-,0"), (87, "delta2+,0"),
            (175, "z2,+"), (175, "z2,-")}
    assert nine < {(sg, name) for sg, name, _s in sweep["reordered"]}


def test_the_spelling_map_round_trips_on_the_whole_vocabulary():
    """34 names on each side, and the rules are a bijection between them.

    The map is a *convention* -- three rules, no table of names -- and this is
    the statement that the convention is total: every name either vocabulary
    ships maps to a name the other one ships, and back to itself.
    """
    paper, shipped = tqr.indicator_vocabularies()
    assert len(paper) == 34 and len(shipped) == 34
    for name in sorted(paper):
        spelled = tqr.to_irreptables_name(name)
        assert spelled is not None, name
        assert spelled in shipped, (name, spelled)
        assert tqr.from_irreptables_name(spelled) == name
    for name in sorted(shipped):
        back = tqr.from_irreptables_name(name)
        assert back is not None, name
        assert back in paper, (name, back)
        assert tqr.to_irreptables_name(back) == name
    # the three rules, one example each, plus a name that is already shared
    assert tqr.to_irreptables_name("z3m,pi") == "z3m,π"          # alphabet
    assert tqr.to_irreptables_name("delta2+,pi") == "δ2,π+"      # qualifier order
    assert tqr.to_irreptables_name("z2,+") == "z2+"              # ... with no qualifier
    assert tqr.to_irreptables_name("Theta2") == "θ2"             # letter case
    assert tqr.to_irreptables_name("z4") == "z4"                 # literal
    # a name in neither vocabulary is None on both sides, never guessed at
    # (the positive control is the literal `z4` two lines above)
    assert tqr.to_irreptables_name("z99") is None
    assert tqr.to_irreptables_name("z3m,π") is None, "that is the other spelling"
    assert tqr.from_irreptables_name("z3m,pi") is None
    assert tqr.from_irreptables_name("nonsense") is None


def test_name_correspondence_reports_per_group_and_propagates_refusals(
        fkm_canonical, unitary_group):
    """:func:`pythtqc.srsi.name_correspondence` on a group, and on one with no table."""
    pairs = tqr.name_correspondence(fkm_canonical["strong"][1])
    assert [p.relation for p in pairs] == ["literal"] * 4
    assert [(p.paper, p.irreptables, p.mod) for p in pairs] == [
        ("z2w,1", "z2w,1", 2), ("z2w,2", "z2w,2", 2), ("z2w,3", "z2w,3", 2),
        ("z4", "z4", 4)]
    assert all(p.reason == "" for p in pairs)

    # SG 83 without SOC is where the reorder rule earns its place
    pairs83 = tqr.name_correspondence(tqr.srsi_tables(83, spinor=False))
    reordered = {p.paper: p.irreptables for p in pairs83 if p.relation == "reordered"}
    assert reordered == {"delta2+,pi": "δ2,π+", "delta2-,pi": "δ2,π-"}
    assert not [p for p in pairs83 if p.relation.endswith("_only")]

    # a unitary group has no SRSI table at all, and the refusal is propagated
    refused = tqr.name_correspondence(unitary_group)
    assert isinstance(refused, Refusal) and refused.gate == "srsi_scope"


# ---------------------------------------------------------------------------
# 4.  the fixtures, in the paper's names
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fkm_canonical(realized):
    """Fu-Kane-Mele's strong and trivial phases in ``R-3m1' #166.98``, with content.

    The phases are declared with the bond-centre species (``FKM_BOND``), which
    names ``R-3m1' #166.98`` outright from the cell; ``examples/tqc_fkm_diamond.py``
    reads the same phases without it, by descent from ``Fd-3m1' #227.129``.
    """
    out = {}
    for name in ("strong", "trivial"):
        got = realized(fkm_regime(name), decorations=FKM_BOND,
                       characters={0: "s"})
        assert got and got.group.number_str == "166.98", f"{name}: {got}"
        table = tqw.hsp_table(got.rep, bands=2)
        assert table.complete
        out[name] = (table, got.group, got.rep)
    return out


@pytest.fixture(scope="module")
def unitary_group():
    """Fu-Kane-Mele's undistorted diamond cell as the **unitary** group ``#227``.

    ``include_TR=False`` asks the cell without time reversal, which is a
    magnetic type-I group -- and ``X_BS(#227, double-valued) = Z2`` is not
    trivial, so there is a class here and the canonical vocabulary still has
    nothing to say about it.  That is the pairing the fixture is for.
    """
    from pythtb.models import fu_kane_mele

    group = _spacegroup._group_of(fu_kane_mele(1.0, 0.125, [0.0, 0.0, 0.0, 0.0]),
                          include_TR=False)
    assert group and group.number_str == "227", str(group)
    assert tqe._magnetic_type(group) == "I"
    return group


#: The anchor, in the canonical vocabulary: the strong phase at ``(0,1)`` of
#: order 4 and the trivial one at ``(0,0)`` in ``X_BS(#166.98) = Z2 x Z4``, said
#: in the paper's names.  Every name of ``#166.98`` is literal, so the two
#: vocabularies print identically here; ``#191.234``, in ``tests/test_ebr.py``'s
#: pure-2D layer test, is where they do not.
FKM_CANONICAL = {
    "strong": ((0, 1), 4, {"z2w,1": 1, "z2w,2": 1, "z2w,3": 1, "z4": 3}),
    "trivial": ((0, 0), 1, {"z2w,1": 0, "z2w,2": 0, "z2w,3": 0, "z4": 0}),
}


@pytest.mark.parametrize("phase", ["strong", "trivial"])
def test_fu_kane_mele_s_classes_in_the_paper_s_names(fkm_canonical, phase):
    """The literature anchor, spoken canonically -- and converted back.

    ``nu_0 = z4 mod 2`` reads off the canonical dict exactly as it read off the
    backend one, because they are the same numbers: the pinned ``z4`` is 3 in the
    strong phase and 0 in the trivial one, and ``si.irreptables_named`` --
    ``irreptables``' own arithmetic through momentum-space forms -- is that same
    dict.  That is the value gate holding on one fixture where both routes were
    run end to end from a model rather than from a table, on a nonzero class and
    on the zero one.  Every name of ``#166.98`` is literal, so the backend
    conversion is the identity map here and says nothing the pairing does not;
    ``#191.234`` is where it has work to do.
    """
    table, group, rep = fkm_canonical[phase]
    si = tqe.symmetry_indicator(table, group, rep=rep, bands=2)
    assert si, str(si)
    values, order, canonical = FKM_CANONICAL[phase]
    assert si.values == values and si.order == order and si.divisors == (2, 4)
    assert si.canonical == canonical
    assert si.canonical_tex == {"z2w,1": "z_{2w,1}", "z2w,2": "z_{2w,2}",
                                "z2w,3": "z_{2w,3}", "z4": "z_{4}"}
    assert si.irreptables_named == canonical, \
        "irreptables' own arithmetic, the same numbers"
    assert "10.1038/s41467-026-74844-w" in si.canonical_note
    # the canonical line comes first, then the Smith class, then irreptables' names
    lines = si.summary().splitlines()
    order_of = [i for i, line in enumerate(lines)
                if line.startswith(("  canonical:", "  indicator:", "  irreptables:"))]
    assert [lines[i].split(":")[0].strip() for i in order_of] == \
        ["canonical", "indicator", "irreptables"]


def test_a_trivial_indicator_group_has_no_class_to_speak(lin3g_declared_cross_checked):
    """lin3g's fragile ``P2`` in ``P4mm1' #99.164``: ``canonical is None``, and why.

    The group has no elementary divisor above 1, so there is no class in any
    vocabulary -- and the paper's tables carry no indicator row for it either,
    which is the coverage statement seen from one group.  ``None`` with the
    reason, not an empty dict and not a refusal: the record is an answer.
    """
    model, group, _asg, _val, rep = lin3g_declared_cross_checked
    si = tqe.symmetry_indicator(tqw.hsp_table(rep, bands=(4, 5)), group,
                                rep=rep, bands=(4, 5))
    assert si and si.verdict == "trivial indicator group"
    assert si.divisors == () and si.canonical is None
    assert si.canonical_tex == {} and si.as_irreptables() is None
    assert "the indicator group of #99.164 is trivial" in si.canonical_note
    assert "canonical" not in si.summary()
    table = tqr.srsi_tables(99, spinor=True)
    assert table and table.indicators == ()


def test_a_unitary_group_gets_the_smith_class_and_no_canonical_name(unitary_group):
    """Type I: the shipped tables are the grey ones, and nothing is extrapolated.

    ``#227`` without time reversal has ``X_BS = Z2`` -- a real class, served in
    Smith coordinates -- and the canonical vocabulary declines it, because the
    paper's shipped tables are the time-reversal-symmetric ones.
    ``irreptables_named`` is
    ``None`` here for the matching reason on the other side (``irreptables``'
    non-magnetic indicator files are the grey tables under bare space-group
    numbers), so this is the one fixture where a class is served with **no**
    names at all.
    """
    rows = tqe.ebr_rows(unitary_group)
    assert rows
    si = tqe.symmetry_indicator(rows[sorted(rows)[0]], unitary_group)
    assert si and si.magnetic_type == "I"
    assert si.divisors == (2,) and si.structure == "Z2" and si.values == (0,)
    assert si.canonical is None and si.canonical_tex == {}
    assert si.as_irreptables() is None and si.irreptables_named is None
    assert "magnetic group of type I" in si.canonical_note
    assert isinstance(tqr.srsi_tables(unitary_group), Refusal)


# ---------------------------------------------------------------------------
# 5.  the InternalError guards
# ---------------------------------------------------------------------------

def test_a_class_the_paper_s_names_cannot_separate_is_an_internal_error(
        fkm_canonical, monkeypatch):
    """The determinism gate at serve time, forced.

    No system in the shipped tables fails it (statement 2), so the only way to
    exercise the branch is to hand the serve path a gate record that says it
    failed.  What is asserted is the **law**: a vocabulary that cannot tell two
    classes apart is this package failing to apply a convention, so it is a hard
    error naming the collision, never a printed number and never a ``Refusal``.
    """
    table, group, rep = fkm_canonical["strong"]
    real = _srsitables._canonical_gate

    def broken(number_str, spinor):
        record = dict(real(number_str, spinor))
        record["injective"] = False
        record["collisions"] = (((0, 0), (1, 2), {"z4": 0}),)
        return record

    patch_everywhere(monkeypatch, "_canonical_gate", broken)
    with pytest.raises(InternalError, match="do not determine the class"):
        tqe.symmetry_indicator(table, group, rep=rep, bands=2)


def test_a_form_that_is_not_an_indicator_is_an_internal_error(
        fkm_canonical, monkeypatch):
    """The vanishing gate at serve time, forced -- the same law, the other check."""
    table, group, rep = fkm_canonical["strong"]
    real = _srsitables._canonical_gate

    def broken(number_str, spinor):
        record = dict(real(number_str, spinor))
        record["vanishes"] = False
        record["ebr_failures"] = (("z4", 3, "2"),)
        return record

    patch_everywhere(monkeypatch, "_canonical_gate", broken)
    with pytest.raises(InternalError, match="do not vanish on every EBR column"):
        tqe.symmetry_indicator(table, group, rep=rep, bands=2)


def test_a_class_with_no_canonical_name_at_all_is_an_internal_error(
        fkm_canonical, monkeypatch):
    """A grey group with a nontrivial Smith form and no paper row.

    Measured not to exist (statement 1), and forced here: a class this package
    can compute and cannot name is the canonicalization failing to apply, which
    is what ``InternalError`` is for.
    """
    table, group, rep = fkm_canonical["strong"]
    real = tqr.srsi_tables

    def stripped(g):
        return replace(real(g), indicators=())

    patch_everywhere(monkeypatch, "srsi_tables", stripped)
    with pytest.raises(InternalError, match="carries no indicator row"):
        tqe.symmetry_indicator(table, group, rep=rep, bands=2)

    def absent(g):
        return Refusal("srsi_table_absent", "forced, for the guard")

    patch_everywhere(monkeypatch, "srsi_tables", absent)
    with pytest.raises(InternalError, match="carry no row for this space group"):
        tqe.symmetry_indicator(table, group, rep=rep, bands=2)


def test_the_serve_path_survives_the_tables_not_being_installed(
        fkm_canonical, monkeypatch):
    """No SRSI data is an installation state, not a bug -- so it is not an error.

    The converted tables ship as package data, generated from a TeX supplement
    that is not shipped, and an installation that has lost them must still
    serve a Smith class: the absence is forced here.  ``canonical`` is ``None``
    there with the reason, and the backend record is untouched.
    """
    table, group, rep = fkm_canonical["strong"]
    patch_everywhere(monkeypatch, "srsi_status", lambda: {
        "data": False, "single": None, "double": None, "dir": "<forced>",
        "doi": _srsitables._DOI, "reason": "forced, for the guard"})
    si = tqe.symmetry_indicator(table, group, rep=rep, bands=2)
    assert si and si.values == (0, 1) and si.canonical is None
    assert "not installed" in si.canonical_note
    assert si.irreptables_named == {"z2w,1": 1, "z2w,2": 1, "z2w,3": 1, "z4": 3}


# ---------------------------------------------------------------------------
# 6.  the failure channels
# ---------------------------------------------------------------------------

def test_the_package_has_three_failure_channels_and_no_bare_runtime_error():
    """The policy, read off the source of every module.

    What failed chooses the channel: a verdict measured on the model or on the
    shipped tables is a ``Refusal`` and is *returned*; a malformed argument is a
    ``ValueError``; this package's own convention failing to apply is an
    ``InternalError``.  A bare ``RuntimeError`` says which of the three
    happened -- nothing -- so it is not a channel here.  The subclasses are: an
    ``InternalError`` *is* a ``RuntimeError``, which is what a caller catching
    one catches; the base class raised on its own is what nothing does.
    """
    src = pathlib.Path(tqg.__file__).parent
    modules = sorted(src.glob("*.py"))
    assert len(modules) > 5, [p.name for p in modules]
    raised = [f"{path.name}:{n}"
              for path in modules
              for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
              if "raise RuntimeError(" in line]
    assert raised == [], raised


def test_a_table_that_cannot_be_matched_to_the_cell_is_an_internal_error(
        km_odd, monkeypatch):
    """The first guard of the translation repair, forced.

    ``_exact_operations`` rebuilds a noisy cell's translations from the
    tabulated reference-cell operations, which needs the shipped table and the
    identified operation list to be the same group operation by operation.  They
    are, wherever the package identifies a cell at all, so the guard is staged
    twice over: the closure is made to read as broken, so that there is
    something to repair, and the shipped table is handed back one operation
    short, so that the matching has nothing to stand on.  What is asserted is
    the law -- a table this package cannot match to the cell it has just
    identified is its own use of the tables failing, so it *raises* and is never
    a ``Refusal`` about the model.
    """
    group = _spacegroup._group_of(km_odd, include_TR=False)
    assert group, str(group)
    assert group._tables_ready, "the repair's tabulated branch needs a matched cell"

    # `irrep` first, always, and `_group_of` is what touches it: `irreptables.irreps`
    # imports `irrep.utility`, so reaching irreptables before irrep leaves that
    # module half-built when irrep runs its own `from irreptables.irreps import
    # IrrepTable`, and irrep falls back to DummyMethodTables for the rest of the
    # process (see test_fixtures_smoke.py).
    import irreptables.irreps as IT

    real_table = IT.IrrepTable

    def one_short(*args, **kwargs):
        table = real_table(*args, **kwargs)
        table.symmetries.pop()
        return table

    monkeypatch.setattr(IT, "IrrepTable", one_short)
    patch_everywhere(monkeypatch, "_translation_closure", lambda symmetries: 1e-8)

    with pytest.raises(InternalError) as exc:
        _spacegroup._exact_operations(group._sg, 1e-5, True, group.lattice)
    text = str(exc.value)
    assert "no operation-by-operation matching" in text, text
    assert "not about the model" in text, text
    assert isinstance(exc.value, RuntimeError) and not isinstance(exc.value, Refusal)
