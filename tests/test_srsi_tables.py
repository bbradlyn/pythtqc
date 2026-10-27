r"""Wave S1: the stable-RSI tables as verified package data.

The conversion (``scripts/convert_srsi_tables.py``) turns the TeX longtables of
the owner's stable-RSI paper (https://doi.org/10.1038/s41467-026-74844-w) into
``src/pythtqc/data/srsi_{single,double}.json``; this file is what says the
result is the paper's tables and not something else.  Six statements, in the
order the spec puts them:

1. **Resolution.**  Every ``(site-symmetry irrep, Wyckoff letter)`` term of
   every SRSI of all 217 + 217 tabulated groups either carries the index of
   *exactly one* EBR of ``groups.ebr_data``'s own list for that grey group, or
   says which of the two reasons it has none.  The thirteen omitted groups
   refuse with the caption's reason.
2. **Counts**, all measured: rows, moduli, terms, split EBRs.
3. **Every EBR is SI-trivial**, checked on every EBR column of every group —
   with the four shipped expressions that fail it pinned exactly, because a
   defect that is measured and named is worth more than one that is silently
   repaired.
4. **The transport, tied**: the paper's indicators read through the SRSIs
   agree with ``irreptables``' own named forms, on every name the two
   vocabularies pair, in every system with an indicator group — the value gate
   of ``tests/test_vocabulary.py``.
5. **The paper's worked examples.**  SG 212 with SOC, whose rows the
   supplement derives in full, reproduced term by term (SG 25 with SOC, the
   other group it derives in full, is the table ``docs/guide/srsi.md`` prints
   row by row); ``Pmm2``'s own distinguishing example, two EBRs with one
   symmetry vector and different Z_2-SRSIs; and the tables of SG 211 without
   SOC and SG 219 with SOC held to the adiabatic relation their Wyckoff
   positions impose.
6. **The residue**: a Z_n-valued SRSI of an atomic limit is an integer reduced
   modulo its own ``n``.  The paper's theorem ``nonzero symmetry indicator <=>
   fractional Z-SRSI`` on this suite's own band sets is
   ``tests/test_srsi_model.py``'s.

Wave S1b adds three more:

7. **The derived indicator rows.**  Every printed indicator is re-derived from
   the momentum-space side, and the four rows the derivation disagrees with are
   exactly the four that fail statement 3.  The corrections are pinned as
   expressions and checked to be indicators.
8. **The supplement's own adiabatic matrix ``q``**, for the two examples that
   print one.  Every shipped row annihilates it and the Z rows span exactly its
   left null space -- the only check of the **non-EBR** terms there is, since
   the kernel test of statement 3 lives on EBR columns alone.
9. **The full site-symmetry vector as the primitive**: ``site_keys`` is the
   paper's whole basis, ``theta`` evaluates on it -- the keys no EBR column
   reaches included -- and every key moves some SRSI.

Statements 4-6 for a *model* -- the Wyckoff letter in the tables' setting, the
localization of an orbit, the model's own SRSIs and a band subset's -- are
``tests/test_srsi_model.py``.

The strongest check of all is not here but in the converter, which refuses to
write a file unless ``ker BR`` is inside ``ker Theta_Z`` on the EBR columns of
every one of the 434 systems; :func:`test_the_kernel_check_that_gates_the_conversion`
re-runs it on the shipped data so the claim lives in the suite too.  That check
constrains the row *space* and the rank, so it is invariant under any change of
basis of ``Theta_Z`` -- doubling a row included.
:func:`test_the_shipped_rows_generate_the_group_their_structure_names` is its
complement: the rows must **generate** the group ``structure`` names, which
together with it pins ``Theta_Z`` up to ``GL(n_Z, Z)`` in all 434 systems.  The
individual coefficients are pinned only where a worked example or an
indicator reaches them; that is said where it is measured and not glossed over.
"""

import collections
import json
import os
import re
from fractions import Fraction

import numpy as np
import pytest

from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import srsi as tqr
from pythtqc.refusal import Refusal
from pythtqc import _smith
from pythtqc import _spacegroup
from pythtqc import _srsitables

import tqc_models as tqm                                            # noqa: F401
from conftest import OMITTED, patch_everywhere

# lin3g band blocks, from LIN3G.band_sets (the same names test_ebr.py uses)
P4, P2, P6 = (0, 1, 2, 3), (4, 5), 6


def _ebr_keys(data):
    """``ebr_data``'s columns as ``(label, Wyckoff letter)``, in column order."""
    return tuple(_srsitables._ebr_key(name, position) for name, position in data["ebrs"])


# ---------------------------------------------------------------------------
# 1.  resolution
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_every_term_of_every_group_resolves_to_exactly_one_ebr_or_says_why(every_table):
    """The conversion's central claim, re-checked against the installed tables.

    A term carries an EBR column exactly when its Wyckoff letter is one of the
    group's EBR positions *and* the site irrep there induces an elementary band
    representation.  For those, the stored column must be the unique EBR with
    that label and letter — checked against ``ebr_data``'s own list, so this is
    a statement about the shipped data and the installed ``irreptables``
    together, not about the conversion's memory of them.

    The two ways a term can have no column are counted rather than tolerated:
    a **non-maximal** Wyckoff position (the paper's ``p`` runs over all of them,
    an EBR basis over none) and a **maximal** position whose induced band
    representation is composite.  Measured 2026-09-02: without SOC, 8460 terms
    on EBR columns, 160 at maximal positions with no EBR and 7184 at
    non-maximal ones; with SOC, 3097 / 0 / 2231.

    Where the supplement and ``irreptables`` spell a site irrep differently, the
    conversion *derived* the correspondence, by matching each position's
    spellings and pairing what was left over only when exactly one label
    remained on each side, and both of its findings are re-derived here from the
    shipped records, once per position:

    * for the site group ``222`` the supplement writes ``A_1`` where the EBR
      tables write ``A`` — 112 positions without SOC, and none with: exactly
      the terms carrying an ``ebr_label``;
    * ``{}^1E{}^2E`` at site symmetry ``-4`` and ``E`` at ``-4m2`` / ``-42m``
      name site irreps whose induced band representation is **composite**, so no
      EBR list carries them — 25 + 7 + 7 = 39 positions without SOC, none with:
      exactly the maximal terms with no EBR column.
    """
    tally = {"ebr": collections.Counter(), "composite": collections.Counter(),
             "nonmaximal": collections.Counter()}
    aliased_positions, composite_positions = collections.Counter(), collections.Counter()
    for table, data in every_table:
        keys = _ebr_keys(data)
        letters = {letter for _label, letter in keys}
        symbols = {}
        for _name, wyckoff in data["ebrs"]:
            match = _srsitables._POSITION.match(str(wyckoff))
            symbols[match.group(2)] = match.group(3)
        kind = "double" if table.spinor else "single"
        seen_alias, seen_absent = set(), set()
        for row in table.rows:
            for term in row.terms:
                where = (table.sg, kind, row.name, term["irrep"], term["letter"])
                if term["ebr"]:
                    assert term["maximal"] and term["column"] is not None, where
                    label = term.get("ebr_label", term["irrep"])
                    hits = [j for j, key in enumerate(keys)
                            if key == (label, term["letter"])]
                    assert hits == [term["column"]], (where, hits, term["column"])
                    tally["ebr"][kind] += 1
                else:
                    assert term["column"] is None, where
                    assert term["maximal"] == (term["letter"] in letters), where
                    tally["composite" if term["maximal"] else "nonmaximal"][kind] += 1
                # the two spellings the conversion derived, once per position
                position = (table.sg, term["letter"], term["irrep"])
                site = (kind, symbols.get(term["letter"]), term["irrep"])
                if "ebr_label" in term:
                    if position not in seen_alias:
                        seen_alias.add(position)
                        aliased_positions[site + (term["ebr_label"],)] += 1
                elif term["maximal"] and not term["ebr"]:
                    if position not in seen_absent:
                        seen_absent.add(position)
                        composite_positions[site] += 1
    assert dict(tally["ebr"]) == {"single": 8460, "double": 3097}
    assert dict(tally["composite"]) == {"single": 160}
    assert dict(tally["nonmaximal"]) == {"single": 7184, "double": 2231}
    assert dict(aliased_positions) == {("single", "2221',222", "A1", "A"): 112}
    assert dict(composite_positions) == {("single", "-41',-4", "1E2E"): 25,
                                         ("single", "-42m1',-42m", "E"): 7,
                                         ("single", "-4m21',-4m2", "E"): 7}


@pytest.mark.parametrize("spinor", [False, True])
def test_the_thirteen_omitted_groups_refuse_with_the_caption_s_reason(spinor):
    """"SGs with only one site-symmetry irrep in a unit cell" — an answer, not a gap.

    The tables leave thirteen groups out because the group's single SRSI is the
    band count, which needs no table.  The refusal says so and carries the whole
    omitted list, so a caller can tell "not tabulated, and here is why" from
    "not found".  Which thirteen are absent is the counts test's measurement.
    """
    out = tqr.srsi_tables(OMITTED[0], spinor=spinor)
    assert isinstance(out, Refusal) and not out
    assert out.gate == "srsi_table_absent"
    assert "only one site-symmetry irrep" in out.reason
    assert tuple(out.context["omitted_groups"]) == OMITTED
    assert out.context["doi"].endswith("s41467-026-74844-w")


def test_a_space_group_number_outside_the_230_refuses_scope():
    for sg in (0, 231, -3):
        out = tqr.srsi_tables(sg, spinor=False)
        assert not out and out.gate == "srsi_scope", (sg, out)


# ---------------------------------------------------------------------------
# 2.  counts
# ---------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.parametrize(
    "valuedness,groups,srsi_rows,zn,si_groups,si_moduli,split_groups,splittings",
    [("single", 217, 1717, {2: 62}, 53, {2: 67, 4: 3}, 3, 9),
     ("double", 217, 1133, {2: 254, 4: 4}, 117,
      {2: 182, 3: 16, 4: 96, 6: 8, 8: 12, 12: 6}, 51, 202)])
def test_the_measured_counts_of_the_shipped_tables(
        valuedness, groups, srsi_rows, zn, si_groups, si_moduli, split_groups,
        splittings):
    """Every number in the spec's table of the source, re-measured from the data.

    Not read out of the provenance block — recounted from the served records,
    so a provenance that drifted from its own file would fail here.  The
    :math:`\\mathbb{Z}_n` rows: 62 of 1717 without SOC, all mod 2; 258 of 1133
    with SOC, 254 mod 2 and 4 mod 4.  The indicators: 53 groups / 70
    expressions without SOC over moduli {2, 4}, 117 groups / 320 expressions
    with SOC over {2, 3, 4, 6, 8, 12}.  The split EBRs: 3 groups / 9 splittings
    without SOC, 51 / 202 with.  The groups served are the exact complement of
    the caption's thirteen.

    The independent content is the parametrized literals.  The four lines that
    read ``provenance["counts"]`` are a **self-consistency** check and nothing
    more: they compare the converter's self-reported counts with counts measured
    off the same JSON document — two fields one script wrote in one pass — so
    what they catch is a provenance block that drifted from its own file.
    """
    spinor = valuedness == "double"
    tables = [tqr.srsi_tables(sg, spinor=spinor) for sg in range(1, 231)]
    tables = [t for t in tables if t]
    assert len(tables) == groups
    assert sorted(set(range(1, 231)) - {t.sg for t in tables}) == list(OMITTED)
    assert sum(len(t.rows) for t in tables) == srsi_rows
    moduli = collections.Counter(m for t in tables for m in t.moduli if m is not None)
    assert dict(moduli) == zn
    assert sum(1 for t in tables if t.indicators) == si_groups
    si = collections.Counter(f.mod for t in tables for f in t.indicators)
    assert dict(si) == si_moduli
    assert sum(1 for t in tables if t.split_ebrs) == split_groups
    assert sum(len(t.split_ebrs) for t in tables) == splittings

    provenance = tables[0].provenance
    assert provenance["counts"]["groups"] == groups
    assert provenance["counts"]["srsi_rows"] == srsi_rows
    assert provenance["counts"]["si_expressions"] == sum(si.values())
    assert provenance["counts"]["splittings"] == splittings
    assert tuple(provenance["omitted_groups"]) == OMITTED


def test_the_provenance_names_the_paper_and_the_bytes_it_was_built_from():
    """A table you cannot trace to a file and a digest is a table you cannot audit.

    Nor one whose file is not there: both documents are installed, each carrying
    its provenance block and the 217 groups.
    """
    status = tqr.srsi_status()
    assert status["data"] and status["reason"] is None
    for valuedness in ("single", "double"):
        assert os.path.isfile(status[valuedness])
        with open(status[valuedness], encoding="utf-8") as handle:
            document = json.load(handle)
        assert set(document) == {"provenance", "groups"}
        assert len(document["groups"]) == 217
    for spinor in (False, True):
        table = tqr.srsi_tables(2, spinor=spinor)
        provenance = table.provenance
        assert provenance["doi"] == "https://doi.org/10.1038/s41467-026-74844-w"
        assert provenance["converter"] == "scripts/convert_srsi_tables.py"
        assert len(provenance["sources"]) == 3
        for source in provenance["sources"]:
            assert source["file"].endswith(".tex")
            assert re.fullmatch("[0-9a-f]{64}", source["sha256"]), source
            assert source["bytes"] > 0
        assert provenance["irreptables_version"]
        # the label map is the mechanical TeX -> irreptables spelling, and it is
        # a function: 58 distinct spellings without SOC, 36 with
        assert len(provenance["label_map"]) == (36 if spinor else 58)
        assert provenance["label_map"]["\\bar{A}_g\\bar{A}_g" if spinor
                                       else "A_{g}"] == ("-AgAg" if spinor else "Ag")


# ---------------------------------------------------------------------------
# 3.  every EBR is SI-trivial
# ---------------------------------------------------------------------------

#: The four shipped indicator expressions that do NOT vanish on their own
#: group's EBRs, and how many of its columns each fails on.  Measured
#: 2026-09-02 over all 4731 EBR columns of the 434 tabulated systems; these are
#: the only failures, and SGs 147, 162, 163 without SOC are also the only three
#: groups whose indicator rows carry fractional theta-coefficients.
NOT_INDICATORS = {(147, "z2,3"): (2, 12, 14), (147, "z4"): (4, 11, 14),
                  (162, "z2'"): (2, 16, 26), (163, "z2'"): (2, 13, 15)}


@pytest.mark.data
def test_every_ebr_is_symmetry_indicator_trivial_through_the_srsis(every_table):
    """An EBR is an atomic limit, so every indicator of its group vanishes on it.

    Written through the SRSIs that is ``sum_i c_i Theta[i][j] = 0 mod n`` for
    every EBR column ``j``, and it is the paper's own consistency statement
    about the two tables — the SRSI table supplies ``Theta``, the SI table
    supplies ``c`` and ``n``, and neither knows about the other.  Checked on
    every column of every group: 4731 columns, 5003 evaluations of the 390
    expressions — which is also what the two shipped provenance blocks say
    themselves (``si_on_ebr_checks`` 1158 + 3845).

    Four expressions fail, all of them in SGs 147, 162 and 163 without SOC, and
    they are pinned here rather than excused: if a corrected supplement ever
    lands, this test is what says so.
    """
    checked = failed = 0
    seen = {}
    for table, _data in every_table:
        n_ebr = len(table.ebrs)
        for form in table.indicators:
            bad = []
            for column in range(n_ebr):
                p = [0] * n_ebr
                p[column] = 1
                raw = table.theta_unreduced_columns(p)
                total = sum((c * raw[i - 1] for i, c in form.theta.items()),
                            Fraction(0))
                checked += 1
                if total.denominator != 1 or int(total) % form.mod:
                    bad.append(column)
            assert form.vanishes_on_ebrs == (not bad), (table.sg, form.name, bad)
            if bad:
                failed += 1
                key = (table.sg, form.name)
                assert not table.spinor, key
                seen[key] = (form.mod, len(bad), n_ebr)
                assert [c for c, _v in form.ebr_failures] == bad, key
    assert seen == NOT_INDICATORS
    assert (checked, failed) == (5003, 4), (checked, failed)


# ---------------------------------------------------------------------------
# 4.  a configuration for a symmetry vector, over the rationals
# ---------------------------------------------------------------------------

def _rational_p(u, v, d, y):
    """A rational ``p`` with ``EBR . p = y``, exactly.

    ``u M v = R`` with ``R`` diagonal, so ``M p = y`` is ``R z = u y`` in
    ``z = v^-1 p``: divide by the nonzero divisors, take zero on the kernel
    directions, and come back with ``p = v z``.  Exact over
    :class:`~fractions.Fraction`; the caller must already have checked the
    compatibility relations (``y'`` vanishes where ``R``'s row is empty),
    which is what makes a solution exist over the rationals.
    """
    n_col = len(v)
    y_prime = [sum(u[i][j] * y[j] for j in range(len(y))) for i in range(len(u))]
    z = [Fraction(y_prime[i], d[i]) if i < len(d) and d[i] else Fraction(0)
         for i in range(n_col)]
    return [sum((Fraction(v[j][i]) * z[i] for i in range(n_col)), Fraction(0))
            for j in range(n_col)], y_prime


# ---------------------------------------------------------------------------
# 5.  the paper's worked examples
# ---------------------------------------------------------------------------

def _row_terms(table, name):
    """``{(irrep, letter): coefficient}`` for one SRSI, as the paper prints it."""
    row = table.row(name)
    assert row is not None, name
    return {(t["irrep"], t["letter"]): t["coef"] for t in row.terms}


#: SN "SG P4_332 (No. 212) with SOC", the three SRSIs printed after the Hermite
#: decomposition, transcribed as ``{(site irrep, Wyckoff letter): coefficient}``
#: with the supplement's labels put in irreptables' spelling by the conversion's
#: own mechanical map.
SG212_DOUBLE = {
    "theta_1": ({("-1E2E", "a"): 1, ("-1E2E", "b"): 1, ("-EE", "c"): 2,
                 ("-1E2E", "d"): 1, ("-AA", "e"): 2}, None),
    "theta_2": ({("-E1", "a"): 1, ("-E1", "b"): 1, ("-1E2E", "c"): 2,
                 ("-1E2E", "d"): 2, ("-AA", "e"): 4}, None),
    "theta_3": ({("-E1", "b"): 1}, 2),
}


def test_the_supplement_s_worked_example_with_soc_is_the_parsed_rows():
    """SG 212 with SOC: every printed term, coefficient for coefficient.

    The supplement derives two groups in full — the adiabatic-process matrix,
    its Smith decomposition, the Hermite simplification — and then says the
    result "match[es] the SRSIs given in our exhaustive tables".  It does,
    exactly.  The other, SG 25 with SOC, is the table the SRSI page prints row
    by row, term by term (docs/guide/srsi.md), which holds it.
    """
    table = tqr.srsi_tables(212, spinor=True)
    assert table and len(table.rows) == len(SG212_DOUBLE)
    for name, (terms, modulus) in SG212_DOUBLE.items():
        assert _row_terms(table, name) == terms, name
        assert table.row(name).mod == modulus, name


def test_pmm2_reproduces_the_paper_s_own_distinguishing_example():
    """The point of the SRSIs, in the paper's own words, computed here.

    ``(E)_1a`` and ``(E)_1b`` of ``Pmm2`` with SOC induce the **same**
    symmetry-data vector, so no momentum-space diagnostic can tell them apart;
    their :math:`\\mathbb{Z}_2`-SRSIs are ``(0, 0, 0)`` and ``(1, 0, 0)``.  And
    ``(E)_1a + (E)_1b`` shares its SRSIs with ``(E)_1b`` alone while having a
    different symmetry vector — the converse failure, also the paper's.
    """
    table = tqr.srsi_tables(25, spinor=True)
    columns = {letter: j for j, (_label, letter) in enumerate(table.ebrs)}
    assert set(columns) == set("abcd")

    def theta(p):
        return table.theta_columns(p)

    a = [1 if j == columns["a"] else 0 for j in range(4)]
    b = [1 if j == columns["b"] else 0 for j in range(4)]
    assert theta(a) == ((Fraction(1),), {2: (0, 0, 0)})
    assert theta(b) == ((Fraction(1),), {2: (1, 0, 0)})

    data = tqg.ebr_data(_srsitables._TableKey(table.number_str, True))
    matrix = np.asarray(data["matrix"], dtype=int)
    assert (matrix @ np.array(a) == matrix @ np.array(b)).all(), (
        "the two EBRs have the same symmetry-data vector")

    both = [x + y for x, y in zip(a, b)]
    assert theta(both)[1] == theta(b)[1], "same Z2-SRSIs..."
    assert not (matrix @ np.array(both) == matrix @ np.array(b)).all(), (
        "... and a different symmetry-data vector")


def test_i432_without_soc_rows_satisfy_the_adiabatic_relation():
    """SG 211: every Z-valued row annihilates a move off the general position.

    The premise comes from ``pyxtal``'s Wyckoff list of SG 211 rather than from
    the rows being checked: ``48j`` is the general position (multiplicity 48,
    three free parameters, hence trivial site symmetry and a single site irrep
    ``A``), and ``24g`` and ``24h`` have multiplicity 24, so each has a site
    group of order ``48/24 = 2`` and the general orbit covers it two to one.  An
    order-two group has exactly two irreps; that the parsed rows carry ``{A, B}``
    at both letters is then a *consequence*, checked below, and not the premise.

    Moving one orbital from ``48j`` onto such a position therefore induces the
    regular representation there,
    ``Delta p = -[(A)_48j] + [(A)_w] + [(B)_w]``, and **every** Z-valued SRSI
    must annihilate it: ``coef[(A)_j] = coef[(A)_w] + coef[(B)_w]`` at both
    ``w = g`` and ``w = h``, row by row.
    """
    table = tqr.srsi_tables(211, spinor=False)
    assert table and len(table.rows) == 10
    # nothing printed in the supplement is dropped: the 48 terms with no EBR
    # column of their own are served beside the ones that have one
    assert len(table.terms_without_ebr_column) == 48

    # the premise of the arithmetic, from pyxtal's Wyckoff list of SG 211 -- an
    # oracle the shipped rows cannot influence
    wyckoffs = {w["letter"]: w for w in _spacegroup._pyxtal_wyckoff(211)[0]}
    general = max(wyckoffs.values(), key=lambda w: w["multiplicity"])
    assert (general["letter"], general["multiplicity"]) == ("j", 48)
    assert general["n_free"] == 3, "the general position: trivial site symmetry"
    for letter in ("g", "h"):
        assert wyckoffs[letter]["multiplicity"] == 24, letter
        assert general["multiplicity"] // wyckoffs[letter]["multiplicity"] == 2, (
            letter, "a site group of order two, covered two to one by 48j")

    # ... and the parsed rows' site irreps are what those orders predict
    used = collections.defaultdict(set)
    for row in table.rows:
        for term in row.terms:
            used[term["letter"]].add(term["irrep"])
    assert used["j"] == {"A"}, "one site irrep at the general position, and it is A"
    assert used["g"] == used["h"] == {"A", "B"}, "the two irreps of an order-two group"

    for row in table.rows:
        if row.mod is not None:
            continue
        coefficients = {(t["irrep"], t["letter"]): t["coef"] for t in row.terms}

        def coefficient(irrep, letter, _c=coefficients):
            return _c.get((irrep, letter), 0)

        for letter in ("g", "h"):
            assert coefficient("A", "j") == (coefficient("A", letter)
                                             + coefficient("B", letter)), (
                row.name, letter, "the table satisfies the adiabatic relation")


def test_f43c_with_soc_rows_satisfy_the_adiabatic_relation():
    """SG 219: every row admits one factor for a move off the general position.

    The premise is one no row supplies -- ``pyxtal`` says ``96h`` is the general
    position (multiplicity 96, three free parameters, so trivial site symmetry
    and one double-valued site irrep) and ``irreptables``' own EBR list says the
    maximal ``24c`` carries exactly the two site irreps ``{1E1 2E1, 1E2 2E2}``:
    moving one orbital from ``96h`` onto ``24c`` induces some fixed multiple
    ``lambda`` of their sum, and every SRSI must satisfy
    ``coef[(A A)_96h] = lambda (coef[1E1 2E1] + coef[1E2 2E2])`` -- with the
    :math:`\\mathbb{Z}_4` row read modulo 4.  The shipped rows admit
    ``lambda = 2`` in every one of the five.
    """
    table = tqr.srsi_tables(219, spinor=True)
    assert table and len(table.rows) == 5

    # the premise, from pyxtal's Wyckoff list and irreptables' EBR list
    wyckoffs = {w["letter"]: w for w in _spacegroup._pyxtal_wyckoff(219)[0]}
    general = max(wyckoffs.values(), key=lambda w: w["multiplicity"])
    assert (general["letter"], general["multiplicity"]) == ("h", 96)
    assert general["n_free"] == 3, "the general position: trivial site symmetry"
    assert wyckoffs["c"]["multiplicity"] == 24
    at_c = {label for label, letter
            in _ebr_keys(tqg.ebr_data(_srsitables._TableKey(table.number_str, True)))
            if letter == "c"}
    assert at_c == {"-1E12E1", "-1E22E2"}, "24c is maximal: irreptables lists its two"

    # ... and the parsed rows touch exactly those site irreps
    used = collections.defaultdict(set)
    for row in table.rows:
        for term in row.terms:
            used[term["letter"]].add(term["irrep"])
    assert used["h"] == {"-AA"}, "one site irrep at the general position 96h"
    assert used["c"] == at_c

    for name in table.names:
        terms, modulus = _row_terms(table, name), table.row(name).mod
        here = terms.get(("-AA", "h"), 0)
        there = 2 * (terms.get(("-1E12E1", "c"), 0) + terms.get(("-1E22E2", "c"), 0))
        assert ((here - there) % modulus == 0) if modulus else here == there, (
            name, "the table's rows all admit lambda = 2")


# ---------------------------------------------------------------------------
# 6.  the residue of a Z_n-valued SRSI
# ---------------------------------------------------------------------------

def test_a_zn_srsi_of_an_atomic_limit_is_a_residue_modulo_its_own_n():
    """"A residue mod ``n`` is only defined on an integer", where there is a row.

    Fu-Kane-Mele's ``#166.98`` has no :math:`\\mathbb{Z}_n`-valued SRSI in
    either valuedness, so its phases cannot make this half of the theorem at
    all.  ``#99.164`` with SOC is ``Z x Z x Z x Z2 x Z2``, and it is
    made here: an atomic limit -- three orbitals in ``(-E)_1c``, two in
    ``(-E2)_1b``, a nonnegative integer occupation by construction -- has an
    **integral** raw value on every row, which is what makes the residue
    defined, and the reduction then actually reduces rather than being the
    identity: the raw ``2`` and ``3`` come back as ``0`` and ``1`` mod 2.
    """
    table = tqr.srsi_tables(99, spinor=True)
    assert table and table.zn_rows == {2: (3, 4)}, table.structure
    assert table.structure == "Z x Z x Z x Z2 x Z2"

    p = {("-E2", "b"): 2, ("-E", "c"): 3}
    raw = table.theta_unreduced(p)
    assert all(v.denominator == 1 for v in raw), raw
    assert [raw[i] for i in table.zn_rows[2]] == [Fraction(2), Fraction(3)], raw

    z_values, zn = table.theta(p)
    assert z_values == (Fraction(3), Fraction(3), Fraction(2)), z_values
    assert zn == {2: (0, 1)}, zn
    assert all(isinstance(x, int) and 0 <= x < 2
               for row in zn.values() for x in row), zn


# ---------------------------------------------------------------------------
# the check that gates the conversion, re-run on the shipped data
# ---------------------------------------------------------------------------

def _rank(rows):
    """The rank of an integer matrix, by fraction-free elimination: exact."""
    m = [[int(x) for x in row] for row in rows]
    rank, previous = 0, 1
    for column in range(len(m[0]) if m else 0):
        pivot = next((r for r in range(rank, len(m)) if m[r][column]), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        for r in range(rank + 1, len(m)):
            m[r] = [(m[rank][column] * a - m[r][column] * b) // previous
                    for a, b in zip(m[r], m[rank])]
        previous = m[rank][column]
        rank += 1
    return rank


@pytest.mark.data
def test_the_kernel_check_that_gates_the_conversion(every_table):
    r"""``ker BR`` is inside ``ker Theta_Z`` on the EBR columns, in all 434 systems.

    The paper's structural claim is that the :math:`\mathbb{Z}`-SRSIs and the
    symmetry-data vector determine each other, :math:`B = \mathrm{BR}\cdot
    [\Theta^{(0)}]^\ddagger\cdot\theta_\mathbb{Z}`, which forces
    :math:`\ker\Theta^{(0)} = \ker\mathrm{BR}`.  Restricted to the EBR columns —
    a coordinate subspace — both kernels restrict with it, and the inclusion is
    checkable against ``irreptables``' own EBR matrix.

    This is the sharpest available check that the label map, the Wyckoff letters
    and the resolved column indices are *jointly* right: a single mis-resolved
    term would generically break it.  It is also what makes
    :func:`~pythtqc.srsi.srsi_of_columns` well defined on a band set that is not
    an atomic limit, where ``p`` is only determined up to ``ker BR``.

    Exact over the integers, through the Smith factors the tables ship with the
    EBR matrix, checked first to be its Smith form: ``u BR v = R`` with ``R``
    diagonal, so the columns of ``v`` on which ``R`` vanishes span ``ker BR``.
    The ranks are compared too, so the inclusion is an equality.
    """
    checked = kernel_vectors = 0
    for table, data in every_table:
        u, v, d, matrix = _smith._smith_system(data, table.number_str)
        u, v, matrix = (np.asarray(x, dtype=np.int64) for x in (u, v, matrix))
        n_row, n_col = matrix.shape
        # every product below stays far inside int64, so the arithmetic is exact
        assert (int(np.abs(u).max()) * int(np.abs(matrix).max()) * n_row
                * int(np.abs(v).max()) * n_col) < 2 ** 62, (table.sg, table.spinor)
        smith = np.zeros((n_row, n_col), dtype=np.int64)
        for i in range(min(n_row, n_col)):
            smith[i, i] = d[i]
        assert (u @ matrix @ v == smith).all(), (table.sg, table.spinor, "u BR v = R")
        kernel = [j for j in range(n_col) if j >= n_row or d[j] == 0]
        z_rows = np.asarray([[int(x) for x in table.theta_on_columns[i]]
                             for i in table.z_rows], dtype=np.int64)
        assert len(z_rows), (table.sg, table.spinor)
        assert not (z_rows @ v[:, kernel]).any(), (table.sg, table.spinor)
        assert _rank(z_rows) == n_col - len(kernel), (table.sg, table.spinor)
        kernel_vectors += len(kernel)
        checked += 1
    assert checked == 434, checked
    assert kernel_vectors > 0, "a sweep with no kernel at all would be vacuous"


#: The coefficients of every Z_n-valued row of every tabulated system, tallied
#: as ``(n, coefficient)``.  Measured 2026-09-09 over all 638 such terms: they
#: are printed as residues, and every one of them is already in ``[1, n)`` --
#: the mod-2 rows are plain parities of an occupation sum.
ZN_COEFFICIENTS = {(2, 1): 617, (4, 1): 5, (4, 2): 15, (4, 3): 1}


@pytest.mark.data
def test_the_shipped_rows_generate_the_group_their_structure_names(every_table):
    r"""The coefficients at the **value** level, in all 434 systems.

    :func:`test_the_kernel_check_that_gates_the_conversion` is the strongest
    check of the rows there is and it is blind to one whole class of error: it
    constrains :math:`\ker\Theta_\mathbb{Z}` and
    :math:`\operatorname{rank}\Theta_\mathbb{Z}`, which is the row *space*, so
    **any** change of basis of :math:`\Theta_\mathbb{Z}` passes it — doubling a
    row included.  This is the complement of it.  The paper's rows are not
    merely a basis of that space over :math:`\mathbb{Q}`; they are a set of
    *generators of the SRSI group*, and :attr:`~pythtqc.srsi.SRSITables.structure`
    says which group that is, so

    .. math:: \theta:\ \mathbb{Z}^{\text{site keys}} \longrightarrow
              \mathbb{Z}^{n_\mathbb{Z}} \times \prod_j \mathbb{Z}_{n_j}

    has to be **onto**.  Doubling ``theta_1`` would land it in
    :math:`2\mathbb{Z}\times\cdots`: the row would no longer take every integer
    value on an integer occupation, and the ``Z`` the structure string advertises
    would be the wrong group.  Checked exactly over the integers, on the paper's
    *whole* site basis rather than the EBR projection: the columns of the
    coefficient matrix, together with :math:`n_j e_j` for each
    :math:`\mathbb{Z}_n` row, must generate :math:`\mathbb{Z}^{\text{rows}}` —
    every elementary divisor 1.

    **What this reaches, and what it does not.**  All 434 systems and all 2850
    rows, and with the kernel check it pins :math:`\Theta_\mathbb{Z}` up to
    :math:`GL(n_\mathbb{Z},\mathbb{Z})`: the kernel check fixes the rational row
    space, this fixes the row lattice to be saturated inside it, and the freedom
    left is an integral recombination of the rows.  It does **not** see such a
    recombination, and it does not see a single coefficient altered in a way that
    leaves the lattice alone — that is the kernel check's job on a
    :math:`\mathbb{Z}` row and nothing's on a :math:`\mathbb{Z}_n` row.  Fixing
    the individual rows is the job of the worked examples transcribed term by
    term, and of the indicator values ``tests/test_vocabulary.py``'s value gate
    compares with ``irreptables``' own forms in the 170 systems with an
    indicator group; this test is why a non-unimodular corruption of any row is
    no longer invisible.

    The :math:`\mathbb{Z}_n` half has to be pinned the other way round, because a
    modulus makes the lattice statement insensitive to it (77 and 1 generate the
    same thing mod 2).  So the coefficients themselves are tallied against
    :data:`ZN_COEFFICIENTS`: all 638 of them, over the 184 systems with such a
    row — **82 of which have neither an indicator nor a split EBR**, and so no
    other check of those rows anywhere in the suite.
    """
    import sympy
    from sympy.matrices.normalforms import smith_normal_form

    checked = 0
    zn_coefficients = collections.Counter()
    for table, _data in every_table:
        index = {(key.irrep, key.letter): j for j, key in enumerate(table.site_keys)}
        columns = [[0] * len(table.rows) for _ in index]
        for i, row in enumerate(table.rows):
            for term in row.terms:
                columns[index[(str(term["irrep"]), str(term["letter"]))]][i] += \
                    int(term["coef"])
        # n e_i is zero in the group, which is what makes row i a Z_n row
        for n, rows in table.zn_rows.items():
            for i in rows:
                unit = [0] * len(table.rows)
                unit[i] = int(n)
                columns.append(unit)
                for term in table.rows[i].terms:
                    zn_coefficients[(n, int(term["coef"]))] += 1
        lattice = sympy.Matrix(len(table.rows), len(columns),
                               lambda r, c: columns[c][r])
        normal = smith_normal_form(lattice, domain=sympy.ZZ)
        divisors = [abs(normal[i, i]) for i in range(min(normal.shape))]
        assert len(divisors) == len(table.rows), (table.sg, table.spinor, divisors)
        assert all(d == 1 for d in divisors), (
            table.sg, table.spinor, table.structure, divisors,
            "the rows generate a proper subgroup of the one structure names")
        checked += 1
    assert checked == 434, checked
    assert dict(zn_coefficients) == ZN_COEFFICIENTS, dict(zn_coefficients)


# ---------------------------------------------------------------------------
# the loader's gates
# ---------------------------------------------------------------------------

def test_srsi_tables_refuses_a_unitary_group(lin3g, lin3g_declared_cross_checked):
    """The tables are the time-reversal-symmetric ones, and the type is measured.

    And every route that stands on the tables returns the same refusal for a
    unitary group rather than an answer: the band route, the model route, the
    splitting and the pairs the SRSIs do not distinguish.
    """
    group = _spacegroup._group_of(lin3g, include_TR=False)
    assert group, str(group)
    out = tqr.srsi_tables(group)
    assert not out and out.gate == "srsi_scope", str(out)
    assert out.context["magnetic_type"] == "I"
    assert "type-II" in out.reason

    model, _grey, assignment, _validated, rep = lin3g_declared_cross_checked
    for route, refused in (
            ("srsi_of_bands", tqr.srsi_of_bands({}, group)),
            ("srsi_of_model", tqr.srsi_of_model((model, group, assignment, rep))),
            ("splitting", tqr.splitting((model, group, assignment, rep), bands=4)),
            ("stably_equivalent_pairs", tqr.stably_equivalent_pairs(group))):
        assert not refused and refused.gate == "srsi_scope", (route, str(refused))
        assert refused.context["magnetic_type"] == "I", route


def test_srsi_tables_refuses_a_stale_ebr_basis_and_names_both_lists(
        lin3g_declared_cross_checked, monkeypatch):
    """The gate that keeps a stored column index from naming the wrong EBR.

    A row's stored ``column`` is an index into ``ebr_data``'s own EBR list, so an
    ``irreptables`` that serves a *different* list makes every one of them name
    the wrong EBR — and a different **order** is enough, which is the case
    doctored here: the served record's ``ebrs`` reversed, so the columns are the
    same set and only their order is wrong.  The refusal has to notice that and
    hand back both lists plus the ``irreptables`` version the conversion ran
    against, because the caller's fix is to re-run the converter.

    Doctored through ``_document`` rather than on disk: the parsed document is
    ``lru_cache``d, so editing the record in place would leak into every later
    ``_build``.
    """
    _model, group, _asg, _val, _rep = lin3g_declared_cross_checked
    valuedness = "double" if group.spinor else "single"
    real = _srsitables._document(valuedness)
    doctored = dict(real, groups=dict(real["groups"]))
    record = dict(doctored["groups"][str(group.sg_number)])
    record["ebrs"] = list(reversed(record["ebrs"]))
    doctored["groups"][str(group.sg_number)] = record
    patch_everywhere(monkeypatch, "_document",
                        lambda kind: doctored if kind == valuedness else real)

    out = tqr.srsi_tables(group)
    assert not out and out.gate == "srsi_ebr_basis", str(out)
    assert "name the wrong EBRs" in out.reason
    assert "convert_srsi_tables.py" in out.reason
    installed = _ebr_keys(tqg.ebr_data(group))
    assert tuple(out.context["installed"]) == installed
    assert tuple(out.context["expected"]) == tuple(reversed(installed))
    assert set(out.context["expected"]) == set(installed), "only the order is wrong"
    assert out.value == out.threshold == float(len(installed)), "and not the count"
    assert out.context["number_str"] == "99.164"
    assert out.context["converted_against"] == "99.164"
    assert out.context["irreptables_version"] == \
        real["provenance"]["irreptables_version"]

    monkeypatch.undo()
    assert tqr.srsi_tables(group), "the real list is served again"


def test_srsi_of_columns_accepts_a_table_or_a_group_and_refuses_a_wrong_length(
        lin3g_declared_cross_checked):
    """**Two entry points, one reading** — on a ``p`` whose reading is not zero.

    The zero vector would prove only that nothing raises: it is the all-zero
    reading in *every* table with this many columns, so an equality on it cannot
    say the group route reached the right table.  The probe is the mixed one the
    sibling test uses — two orbitals on the first EBR column, one on the last —
    and both halves of its reading are pinned before the three routes are
    compared on it.
    """
    _model, group, _asg, _val, rep = lin3g_declared_cross_checked
    table = tqr.srsi_tables(group)
    p = [0] * len(table.ebrs)
    p[0], p[-1] = 2, 1
    # #99.164 with SOC, measured 2026-09-09: neither half of the reading is zero
    assert table.theta_columns(p) == ((Fraction(1), Fraction(3), Fraction(0)),
                                      {2: (0, 1)})
    assert tqr.srsi_of_columns(group, p) == tqr.srsi_of_columns(table, p) == \
        table.theta_columns(p)
    with pytest.raises(ValueError, match="EBR columns"):
        tqr.srsi_of_columns(table, [0, 1])


def test_the_two_shapes_of_p_have_a_name_each_and_neither_takes_the_other(
        lin3g_declared_cross_checked):
    """**Two entry points, one per shape of ``p``**, and they agree where both speak.

    An EBR column *is* a site key -- ``(site irrep, Wyckoff letter)`` with a
    column index -- so the same configuration written as a vector over the
    columns (``srsi_of_columns``) and as a mapping over the keys (``srsi_of``)
    must give the same SRSIs, both halves.  Measured on lin3g's ``P4mm1'
    #99.164``, every EBR column, singly and in a mixed sum.

    Neither name accepts the other's shape: handing a mapping to the column
    spellings is a :class:`TypeError` naming the one to use, so a ``p`` in the
    wrong order can no longer be read as a right-length sequence by accident.
    """
    group = lin3g_declared_cross_checked.group
    table = tqr.srsi_tables(group)
    assert table, str(table)
    for column, (irrep, letter) in enumerate(table.ebrs):
        vector = [1 if j == column else 0 for j in range(len(table.ebrs))]
        assert tqr.srsi_of_columns(group, vector) == tqr.srsi_of(
            group, {(irrep, letter): 1}), \
            f"#{group.number_str} column {column} ({irrep})_{letter}"
    # and a sum of two of them, which the mapping adds up by key
    mixed = [0] * len(table.ebrs)
    mixed[0], mixed[-1] = 2, 1
    assert tqr.srsi_of_columns(table, mixed) == tqr.srsi_of(
        table, {table.ebrs[0]: 2, table.ebrs[-1]: 1})

    table = tqr.srsi_tables(211, spinor=False)
    mapping = {("B3", "d"): 1}
    for call in (lambda: tqr.srsi_of_columns(table, mapping),
                 lambda: table.theta_columns(mapping),
                 lambda: table.theta_unreduced_columns(mapping),
                 lambda: table.indicators_of_columns(mapping)):
        with pytest.raises(TypeError, match="mapping is"):
            call()


def test_a_sequence_under_the_mapping_spellings_says_which_name_to_use():
    """The mapping spellings refuse the column shape and name the one that takes it.

    ``srsi_of``, ``theta``, ``theta_unreduced`` and ``indicators_of`` take the
    site-key mapping and the ``_columns`` four take the sequence over the EBR
    columns.  Each gate is raised before any arithmetic, so a shape handed to
    the wrong name is answered as the grammar mistake it is and never as a
    number.
    """
    table = tqr.srsi_tables(211, spinor=False)
    vector = [0] * len(table.ebrs)
    vector[16] = 1

    for call, names in ((lambda: tqr.srsi_of(table, vector), "srsi_of_columns"),
                        (lambda: table.theta(vector), "theta_columns"),
                        (lambda: table.theta_unreduced(vector),
                         "theta_unreduced_columns"),
                        (lambda: table.indicators_of(vector),
                         "indicators_of_columns")):
        with pytest.raises(TypeError, match=names):
            call()


def test_srsi_tables_takes_a_number_or_a_group_and_serves_the_same_table(
        lin3g_declared_cross_checked):
    """**One loader**, two shapes, and the ``spinor=`` declaration belongs to one.

    A bare space-group number needs the valuedness said out loud; a measured
    ``GroupData`` already carries it, and saying it twice is a
    :class:`ValueError` rather than a silent preference.  The two routes serve
    the same table: lin3g's group is ``#99.164``, and ``srsi_tables(99,
    spinor=True)`` is that record field for field.

    And "field for field" is all the equality below says, because the group route
    **is** the number route plus the EBR-basis gate — one ``lru_cache``d document,
    one ``_build``, then a check.  An implementation that re-keyed the columns
    into the group's own order is what it would catch, and the thing that carries
    that difference is the gate's own test
    (:func:`test_srsi_tables_refuses_a_stale_ebr_basis_and_names_both_lists`).
    What discriminates here is the rest: the two :class:`ValueError`\\ s, the
    number string, the row and site-key counts and the structure string.
    """
    _model, group, _asg, _val, _rep = lin3g_declared_cross_checked
    assert group.number_str == "99.164" and group.spinor

    by_group = tqr.srsi_tables(group)
    by_number = tqr.srsi_tables(99, spinor=True)
    assert by_group == by_number
    assert by_number.number_str == "99.164" and by_number.spinor

    # the model-free route on the supplement's own worked example
    sg211 = tqr.srsi_tables(211, spinor=False)
    assert sg211 and sg211.number_str == "211.57" and not sg211.spinor
    assert len(sg211.rows) == 10 and len(sg211.site_keys) == 29
    assert sg211.structure == "Z x Z x Z x Z x Z x Z x Z x Z2 x Z2 x Z2"

    with pytest.raises(ValueError, match="two valuednesses"):
        tqr.srsi_tables(211)
    with pytest.raises(ValueError, match="already declares its valuedness"):
        tqr.srsi_tables(group, spinor=True)
    with pytest.raises(ValueError, match="already declares its valuedness"):
        tqr.srsi_tables(group, spinor=False)
    # the calls that take a table do not take a number: they name the loader
    for call in (lambda: tqr.srsi_of(211, {}), lambda: tqr.srsi_of_columns(211, []),
                 lambda: tqr.stably_equivalent_pairs(211),
                 lambda: tqr.canonical_determinism(211),
                 lambda: tqr.name_correspondence(211)):
        with pytest.raises(TypeError, match=r"srsi_tables\(211, spinor=True\)"):
            call()


#: The SRSI names ``pythtqc`` exports at the top level, frozen in wave C1:
#: the entry points a user calls and the records they hand back.  Everything
#: else reaches through ``pythtqc.srsi`` / ``pythtqc.groups``.
FROZEN_TOP_LEVEL = frozenset({
    "srsi_tables", "srsi_of", "srsi_of_columns", "srsi_of_bands", "srsi_of_model",
    "splitting", "nsit_criterion",
    "atomic_realizations", "stably_equivalent_pairs",
    "wyckoff_sites_tables",
    "SRSITables", "BandSRSIs", "ModelSRSIs", "Splitting", "Realizations", "StablePair",
})

#: Reachable through the submodules and deliberately **not** at the top level.
NOT_TOP_LEVEL = ("SRSIRow", "SRSIIndicator", "SiteKey", "SplitEBR", "OrbitSRSIs",
                 "MaximalKey", "AtomicRealization",
                 "SPLITTING_VERDICTS", "GATES", "srsi_status",
                 "to_irreptables_name", "from_irreptables_name", "NamePair",
                 "NAME_RELATIONS", "indicator_vocabularies", "canonical_determinism",
                 "name_correspondence")


def test_the_frozen_api_and_the_names_kept_off_the_top_level():
    """**The C1 freeze**: the top-level namespace, and what stays below it.

    The SRSI names at the top level are exactly the frozen set, each the
    ``pythtqc.srsi`` object itself, and the module's other public names are
    reached through ``pythtqc.srsi``.  The names the freeze retired --
    ``srsi_table``, folded into ``srsi_tables``, and the four arithmetic methods
    that are now ``theta`` / ``theta_unreduced`` -- are kept off every surface
    by the style test's list of names the package does not use
    (``tests/test_register.py``).
    """
    import pythtqc as tqc

    served = {name for name in tqc.__all__
              if getattr(getattr(tqc, name), "__module__", None)
              in (tqr.__name__, _srsitables.__name__)}
    expected = FROZEN_TOP_LEVEL - {"wyckoff_sites_tables"}
    assert served == expected, sorted(served ^ expected)
    assert ("wyckoff_sites_tables" in tqc.__all__
            and tqc.wyckoff_sites_tables is tqg.wyckoff_sites_tables)
    for name in expected:
        assert getattr(tqc, name) is getattr(tqr, name), name
    for name in NOT_TOP_LEVEL:
        assert hasattr(tqr, name), name
        assert not hasattr(tqc, name) and name not in tqc.__all__, name
    assert hasattr(tqg, "WYCKOFF_REFERENCE_TOL") and not hasattr(tqc, "WYCKOFF_REFERENCE_TOL")
    assert tqc.SiteReference is tqg.SiteReference, "what wyckoff_sites_tables returns"
    assert "SiteReference" in tqc.__all__
    assert tqc.srsi is tqr


def test_every_served_record_is_uniform():
    """The C1 record shape: ``number_str`` and ``spinor`` on each.

    And one pair of names for the invariants wherever they appear —
    ``theta_Z`` / ``theta_Zn``, with the totals of a splitting spelled
    ``theta_Z_total`` / ``theta_Zn_total`` so that a record carrying three
    vectors names all three.  ``summary()`` is not asserted over the six here —
    ``callable`` is true of any attribute that exists, so the line said only
    that — but exercised below on the two the freeze gave one to, and
    ``SRSITables.formulas()`` beside it, one line per invariant.
    """
    for record in (tqr.SRSITables, tqr.BandSRSIs, tqr.ModelSRSIs, tqr.Splitting,
                   tqr.Realizations, tqr.StablePair):
        fields = set(record.__dataclass_fields__)
        assert {"number_str", "spinor"} <= fields, record.__name__
    assert "theta_Z" in tqr.BandSRSIs.__dataclass_fields__
    assert "theta_Zn" not in tqr.BandSRSIs.__dataclass_fields__   # the band route
    for record in (tqr.ModelSRSIs, tqr.OrbitSRSIs, tqr.StablePair):
        assert {"theta_Z", "theta_Zn"} <= set(record.__dataclass_fields__), record
    assert {"theta_Z_total", "theta_Zn_total", "theta_Z_v", "theta_Z_c"} <= set(
        tqr.Splitting.__dataclass_fields__)

    # the two records the freeze gave a summary() to, exercised
    table = tqr.srsi_tables(211, spinor=False)
    text = table.summary()
    assert "I4321' #211.57, single-valued" in text and table.structure in text
    assert "theta_10" in text and "17 of them inducing an elementary band" in text
    head, *lines = table.formulas().splitlines()
    assert "multiplicity" in head, "the first line says what a term's coefficient counts"
    assert [line.split(" (")[0].strip() for line in lines] == [row.name for row in table.rows], \
        "formulas() writes every invariant, in the paper's order"
    assert all(" = " in line for line in lines)
    assert lines[0].split(" = ", 1)[1].count(")_") == len(table.rows[0].terms), \
        "each term is one site irrep at its Wyckoff letter"
    pair, = tqr.stably_equivalent_pairs(table)
    assert "B2 and B3 at Wyckoff position d, which no stable real-space invariant " \
        "distinguishes" in pair.summary()
    assert "the invariants both carry" in pair.summary()


def test_the_srsi_module_all_is_complete():
    """Every public function, class and named constant of the module is in ``__all__``."""
    defined = {name for name, value in vars(tqr).items()
               if not name.startswith("_")
               and getattr(value, "__module__", None) == tqr.__name__}
    assert defined <= set(tqr.__all__), sorted(defined - set(tqr.__all__))
    for name in ("GATES", "SPLITTING_VERDICTS", "NAME_RELATIONS"):
        assert name in tqr.__all__, name
    assert len(set(tqr.__all__)) == len(tqr.__all__)


def test_the_gate_list_is_measured_against_the_source():
    """``srsi.GATES`` is every gate the layer can return, checked by grepping for them.

    Every ``Refusal("...")`` literal in ``srsi.py``, in ``_srsitables.py``, the
    tables beneath it, and in ``wyckoff_sites_tables`` — the one function outside
    the two the SRSI layer refuses through — must appear in :data:`srsi.GATES`,
    and nothing else may.  A gate added without an entry there, saying what it measures,
    fails here.
    """
    import ast
    import inspect

    literals = set()
    for source in (inspect.getsource(tqr), inspect.getsource(_srsitables),
                   inspect.getsource(tqg.wyckoff_sites_tables)):
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "Refusal" and node.args):
                first = node.args[0]
                assert isinstance(first, ast.Constant), ast.dump(first)
                literals.add(first.value)
    assert literals == set(tqr.GATES), sorted(literals ^ set(tqr.GATES))
    assert len(tqr.GATES) == 12
    for gate, meaning in tqr.GATES.items():
        assert meaning and "\n" not in meaning, gate


def test_the_terms_without_an_ebr_column_are_served_rather_than_dropped():
    """Nothing the table prints is thrown away by the restriction to EBR columns."""
    table = tqr.srsi_tables(25, spinor=True)
    orphans = table.terms_without_ebr_column
    assert len(orphans) == 5                      # 2e, 2f, 2g, 2h and the general 4i
    assert {t["letter"] for t in orphans} == set("efghi")
    assert all(t["srsi"] == 1 and not t["maximal"] for t in orphans)
    assert {t["irrep"] for t in orphans} == {"-1E2E", "-AA"}
    # the rows themselves also carry every printed term -- theta_1 has nine
    # (four on EBR columns and the five above), theta_2..4 one each -- while
    # `coefficients` carries only the EBR block
    assert [len(row.terms) for row in table.rows] == [9, 1, 1, 1]
    assert [sum(1 for t in row.terms if t["ebr"]) for row in table.rows] == [4, 1, 1, 1]
    assert all(len(row.coefficients) == len(table.ebrs) for row in table.rows)


# ---------------------------------------------------------------------------
# 7.  the derived indicator rows (S1b)
# ---------------------------------------------------------------------------

#: The four corrected rows, as the S1 rider derived them and as the conversion
#: now ships them: ``{index: coefficient}`` over the paper's own theta indices,
#: all of them Z-valued rows.  Each is integer and even like every other row of
#: the tables, and none is a re-normalization of the printed one -- they differ
#: by a rational vector.
CORRECTED = {
    (147, "z2,3"): (2, {1: -2, 2: 2, 3: -4, 4: 4, 5: 2, 6: 4}),
    (147, "z4"): (4, {1: -4, 2: 4, 3: -8, 4: 8}),
    (162, "z2'"): (2, {1: -2, 2: 2, 3: -2, 4: 2, 5: -4, 6: 4}),
    (163, "z2'"): (2, {4: -2, 5: -4}),
}


@pytest.mark.parametrize("sg,name", sorted(CORRECTED))
def test_the_four_corrected_rows_are_the_derived_ones(sg, name):
    """The corrections, pinned as expressions and checked as indicators.

    Two things have to be true of a corrected row and are asserted here: it
    **vanishes on every EBR column** of its group (which is what makes it an
    indicator at all, and what the printed row fails), and it is **integer on a
    Z-basis of ``{BS}``** (an indicator of ``{BS}/{AI}`` takes integer values on
    band structures).  The printed row still carries its own measurement,
    ``vanishes_on_ebrs=False`` with the columns it fails on.
    """
    table = tqr.srsi_tables(sg, spinor=False)
    form = next(f for f in table.indicators if f.name == name)
    modulus, want = CORRECTED[(sg, name)]
    assert form.mod == modulus
    assert form.relation == "different" and form.derived is not None
    assert {k: int(v) for k, v in form.derived.items()} == want
    assert all(v.denominator == 1 for v in form.derived.values())
    assert form.coefficients == form.derived
    assert not form.vanishes_on_ebrs, "the PRINTED row is still flagged"

    n_ebr = len(table.ebrs)
    for column in range(n_ebr):
        p = [1 if j == column else 0 for j in range(n_ebr)]
        raw = table.theta_unreduced_columns(p)
        total = sum((c * raw[i - 1] for i, c in form.derived.items()), Fraction(0))
        assert total.denominator == 1 and int(total) % modulus == 0, (
            column, total, "the derived row must vanish on every EBR")

    data = tqg.ebr_data(_srsitables._TableKey(table.number_str, False))
    u, v, d, matrix = _smith._smith_system(data, table.number_str)
    u_inv = _smith._unimodular_inverse(u)
    checked = 0
    for i, di in enumerate(d):
        if di == 0:
            continue
        y = [Fraction(int(u_inv[r][i])) for r in range(len(u))]
        p, _y_prime = _rational_p(u, v, d, y)
        raw = table.theta_unreduced_columns(p)
        total = sum((c * raw[j - 1] for j, c in form.derived.items()), Fraction(0))
        assert total.denominator == 1, (i, total, "integer on the {BS} basis")
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 8.  the two worked examples, against the supplement's own q
# ---------------------------------------------------------------------------

#: SN "SG I432 (No. 211) without SOC", SEq. (sg211_irrep): the paper's own
#: ordered basis of the site-symmetry representation vector, transcribed into
#: irreptables' spelling (the conversion's mechanical label map).  29 entries.
SG211_BASIS = (
    ("A1", "a"), ("A2", "a"), ("E", "a"), ("T1", "a"), ("T2", "a"),
    ("A1", "b"), ("A2", "b"), ("B1", "b"), ("B2", "b"), ("E", "b"),
    ("A1", "c"), ("A2", "c"), ("E", "c"),
    ("A1", "d"), ("B1", "d"), ("B2", "d"), ("B3", "d"),
    ("A", "e"), ("B", "e"), ("1E2E", "e"),
    ("A1", "f"), ("1E2E", "f"),
    ("A", "g"), ("B", "g"),
    ("A", "h"), ("B", "h"),
    ("A", "i"), ("B", "i"),
    ("A", "j"),
)

#: SN "SG I432 (No. 211) without SOC", SEq. (sg211_q): the adiabatic-process
#: matrix, rows indexed by SG211_BASIS and columns the generators of the
#: adiabatic processes.  Copied from the supplement (the paper's own published
#: content, and the one thing this suite transcribes from it).
SG211_Q = (
    (1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 2, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
    (-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1),
)

#: SN "SG F-43c (No. 219) with SOC", SEq. (sg219_irrep) -- 13 entries, with
#: ``1E2 2E2`` printed BEFORE ``1E1 2E1`` at both 24c and 24d.
SG219_BASIS = (
    ("-E", "a"), ("-1F2F", "a"), ("-E", "b"), ("-1F2F", "b"),
    ("-1E22E2", "c"), ("-1E12E1", "c"), ("-1E22E2", "d"), ("-1E12E1", "d"),
    ("-EE", "e"), ("-1E2E", "e"), ("-1E2E", "f"), ("-1E2E", "g"), ("-AA", "h"),
)

#: SN "SG F-43c (No. 219) with SOC", SEq. (sg219_q).
SG219_Q = (
    (0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0),
    (2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0),
    (0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (-1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0),
    (0, -1, 0, 0, -1, 0, 0, 0, 2, 0, 0),
    (0, 0, -1, 0, 0, 0, 0, -1, 0, 2, 0),
    (0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 2),
    (0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1),
)


@pytest.mark.parametrize("sg,spinor,basis,q,rank,n_z", [
    (211, False, SG211_BASIS, SG211_Q, 22, 7),
    (219, True, SG219_BASIS, SG219_Q, 9, 4)])
def test_every_table_row_annihilates_the_supplement_s_own_adiabatic_matrix(
        sg, spinor, basis, q, rank, n_z):
    r"""**The test of the non-EBR terms there is**, and it is not optional.

    An SRSI is by definition a linear form on the site-symmetry vector that
    every allowed adiabatic process leaves alone, i.e. one that annihilates the
    columns of the process matrix ``q`` -- exactly for a Z-valued row, modulo
    ``n`` for a Z_n-valued one.  The supplement prints ``q`` explicitly for two
    of its worked examples, over its own ordered basis of *all* site-symmetry
    irreps at *all* Wyckoff positions.  So this checks the shipped rows over the
    paper's **whole** basis, non-maximal positions and composite maximal ones
    included, which the conversion's kernel test (an EBR-column statement) could
    not reach at all.

    Three statements, all measured here:

    * the rows touch **exactly** the printed basis -- 29 of 29 for SG 211
      without SOC, 13 of 13 for SG 219 with -- so ``site_keys`` *is* the paper's
      ``p``, and no coordinate of it is missing or invented;
    * every row annihilates every column of ``q``;
    * the Z-valued rows span **exactly** the left null space of ``q``: rank 7 of
      7 for SG 211, 4 of 4 for SG 219.  Not merely inside it -- equal to it,
      which is what says the table is the complete set of Z-SRSIs.
    """
    import sympy

    table = tqr.srsi_tables(sg, spinor=spinor)
    assert table
    keys = {(k.irrep, k.letter) for k in table.site_keys}
    assert keys == set(basis), (sorted(keys ^ set(basis)),
                                "the rows touch exactly the paper's basis")
    assert len(table.site_keys) == len(basis)

    n_row, n_col = len(q), len(q[0])
    assert n_row == len(basis)
    z_rows = []
    for row in table.rows:
        coefficients = {(t["irrep"], t["letter"]): int(t["coef"]) for t in row.terms}
        vector = [coefficients.get(key, 0) for key in basis]
        values = [sum(vector[i] * q[i][j] for i in range(n_row))
                  for j in range(n_col)]
        if row.mod is None:
            assert all(v == 0 for v in values), (row.name, values)
            z_rows.append(vector)
        else:
            assert all(v % row.mod == 0 for v in values), (row.name, row.mod, values)
    matrix = sympy.Matrix([list(r) for r in q])
    assert int(matrix.rank()) == rank
    left_null = matrix.T.nullspace()
    assert len(left_null) == n_z == len(z_rows)
    assert int(sympy.Matrix(z_rows).rank()) == n_z
    joint = sympy.Matrix(z_rows + [[x for x in vector] for vector in
                                   (list(nv) for nv in left_null)])
    assert int(joint.rank()) == n_z, "the Z rows span the whole left null space"


# ---------------------------------------------------------------------------
# 9.  the full site-symmetry vector as the primitive
# ---------------------------------------------------------------------------

def test_theta_evaluates_the_keys_no_ebr_column_can_reach():
    """The coordinates the EBR block drops -- and they are not zero.

    Both kinds, in SG 84 without SOC: ``(A)_8k`` at a **non-maximal** position,
    and ``({}^1E{}^2E)_4e`` at a **maximal** one whose induced band
    representation is composite (site symmetry ``-4``, one of the 39 such
    positions).  The EBR-column form cannot express either -- there is none --
    and the full vector can, with the table's own printed coefficients.

    The headline is ``any(v != 0 ...)``: these coordinates are not in the kernel,
    so the EBR block really does drop information.  The ``z_values == want`` line
    is **not** a check of the coefficients -- its oracle rebuilds them from the
    same ``row.terms`` -- but of ``_theta_of_sites``' arithmetic against the
    printed ones, which is a different and smaller claim; the rational
    multiplicity at the end is the part of that arithmetic with teeth.  What pins
    the coefficients themselves is
    :func:`test_the_shipped_rows_generate_the_group_their_structure_names` and
    the kernel check.
    """
    table = tqr.srsi_tables(84, spinor=False)
    keys = {(k.irrep, k.letter): k for k in table.site_keys}
    assert not keys[("A", "k")].maximal and keys[("A", "k")].column is None
    assert keys[("1E2E", "e")].maximal and not keys[("1E2E", "e")].ebr
    assert keys[("1E2E", "e")].column is None

    for key in (("A", "k"), ("1E2E", "e")):
        z_values, zn = table.theta({key: 1})
        want = tuple(Fraction(_row_terms(table, name).get(key, 0))
                     for name in table.names if table.row(name).mod is None)
        assert z_values == want
        assert any(v != 0 for v in z_values), (key, "a basis vector is not in ker")
    # and rational multiplicities go through, the way they do on the columns
    half = table.theta({("A", "k"): Fraction(1, 2)})[0]
    whole = table.theta({("A", "k"): 1})[0]
    assert half == tuple(v / 2 for v in whole)


def test_theta_refuses_an_unknown_key_by_naming_the_known_ones():
    """A mistyped label is a grammar mistake, so it raises rather than refusing."""
    table = tqr.srsi_tables(25, spinor=True)
    with pytest.raises(ValueError, match=r"no site irrep 'Q' at Wyckoff letter 'a'"):
        table.theta({("Q", "a"): 1})
    with pytest.raises(ValueError, match=r"the tables carry \['-E'\]"):
        table.theta({("-E", "a"): 1, ("Q", "a"): 1})
    with pytest.raises(ValueError, match="no Wyckoff letter 'z' at all"):
        table.theta({("-E", "z"): 1})
    with pytest.raises(ValueError, match="keyed by"):
        table.theta({"-E": 1})


@pytest.mark.data
def test_the_site_keys_are_the_whole_basis_and_no_column_is_empty(every_table):
    """Every key the rows touch moves at least one SRSI.

    The band count is a Z-linear form in the Z-SRSIs (the paper's own
    observation about its SG 211 example: ``nu = theta_1 + theta_2``), so
    adding an orbital of *any* site irrep at *any* position must move some
    Z-SRSI -- no coordinate of ``p`` can have an all-zero column.  Measured on
    every key of every tabulated group.
    """
    checked = collections.Counter()
    for table, _data in every_table:
        kind = "double" if table.spinor else "single"
        for key in table.site_keys:
            # the record site_keys returns is itself a key theta takes
            z_values, zn = table.theta({key: 1})
            assert any(v != 0 for v in z_values) or any(
                v for row in zn.values() for v in row), (table.sg, key)
            checked[kind] += 1
    assert dict(checked) == {"single": 5019, "double": 2604}, dict(checked)
