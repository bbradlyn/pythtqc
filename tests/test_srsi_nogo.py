r"""Wave S2: the paper's **no-go criterion for a band splitting**, in theta-coordinates.

Where ``test_srsi_tables.py`` says the shipped tables are the paper's and
``test_srsi_model.py`` says the two routes onto a model are right, this file says
the one *verdict* the SRSI layer issues is right.

The criterion (SN "SRSI mismatch as a sufficient criterion for band topology"):
the full Hilbert space of an insulator is an atomic insulator, so ``B[H] = B[V] +
B[C]`` and ``theta_Z[H] = theta_Z[V] + theta_Z[C]`` always hold; and if ``V`` and
``C`` are *both* atomic insulators then ``theta_Zn`` must add up modulo ``n`` as
well, for **some** atomic realizations of the two with the right symmetry data
vectors.  When no pair adds up, one of the two sides is not an atomic insulator.

1. **The basis of the realizations** (``SRSITables._maximal_keys``,
   ``SRSITables._band_count_form``).  Every maximal site key with the number of
   bands one orbital of it carries -- the EBR columns and the 39 composite ones.
   The band count is a Z-linear form in the Z-SRSIs, it is *unique* (the EBR
   block of ``Theta0`` has full row rank in all 434 systems), and no maximal key
   has an all-zero column.
2. **The enumeration** (``atomic_realizations``).  One equation per Z-SRSI instead
   of the redundant band-representation system, which is exact because
   ``ker BR = ker Theta0``; the bound is rigorous, so ``INFEASIBLE`` is a proof
   and a truncated enumeration is a refusal rather than an answer.
3. **The pairs no measurement separates** (``SRSITables.stably_equivalent``).
4. **The regression.**  The supplement's own list of 211 symmetry-indicated
   split EBRs: every flag reproduced from the theta arithmetic.  The criterion
   run on the same list with the **full** enumeration of atomic realizations --
   information the supplement does not carry -- is ``examples/tqc_srsi_tables.py``'s
   table; the one flagged splitting it does not diagnose is measured here.
5. **The models**, where ``nsit`` has to be read rather than looked up.

**The vocabulary matters and is asserted.**  A consistent pair does not make a
splitting trivial -- it makes any topology of that splitting *invisible to the
SRSIs*.  The criterion is sufficient, never necessary.
"""

import re
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from pythtqc import ebr as tqe
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc import _srsitables
from pythtqc import _tables

import tqc_models as tqm                                            # noqa: F401
from conftest import FKM_BOND, fkm_regime, km_z2_regime

#: The 36 same-site pairs of distinct site irreps with identical symmetry data
#: and matching Z- and Z_n-SRSIs, ``(space group, Wyckoff letter, site irreps)``,
#: in the order the sweep over the tabulated systems meets them.
SAME_CONTENT_PAIRS = (
    (90, "a", ("B2", "B3")), (90, "b", ("B2", "B3")), (97, "d", ("B2", "B3")),
    (100, "b", ("B1", "B2")), (101, "a", ("B1", "B2")), (101, "b", ("B1", "B2")),
    (102, "a", ("B1", "B2")), (105, "a", ("B1", "B2")), (105, "b", ("B1", "B2")),
    (107, "b", ("B1", "B2")), (108, "b", ("B1", "B2")), (109, "a", ("B1", "B2")),
    (113, "c", ("B1", "B2")), (117, "c", ("B2", "B3")), (117, "d", ("B2", "B3")),
    (120, "a", ("B2", "B3")), (120, "d", ("B2", "B3")), (127, "c", ("B2u", "B3u")),
    (127, "c", ("B2g", "B3g")), (127, "d", ("B2u", "B3u")), (127, "d", ("B2g", "B3g")),
    (128, "d", ("B2", "B3")), (130, "a", ("B2", "B3")), (133, "c", ("B2", "B3")),
    (135, "d", ("B2", "B3")), (137, "d", ("B1", "B2")), (138, "e", ("B1", "B2")),
    (140, "d", ("B2u", "B3u")), (140, "d", ("B2g", "B3g")), (142, "b", ("B2", "B3")),
    (197, "b", ("B1", "B2")), (201, "d", ("B1", "B2")), (204, "b", ("B1u", "B2u")),
    (204, "b", ("B1g", "B2g")), (211, "d", ("B2", "B3")), (230, "c", ("B2", "B3")),
)

#: The ``(n)`` an ``_tables._ebr_data`` EBR name ends in -- that EBR's **own** band
#: count, written by the EBR tables beside the name and nowhere in ``Theta``.
#: Read here rather than through ``srsi._ebr_dimension`` so the band-count form
#: is checked against a number the package's own arithmetic did not produce.
_EBR_BANDS = re.compile(r"\((\d+)\)\s*$")


def _by_modulus(theta_zn):
    """``{n: values}``: invariants valued modulo n, grouped by modulus in the tables' order."""
    out = {}
    for entry in theta_zn.values():
        out.setdefault(entry.modulus, []).append(entry.value)
    return {n: tuple(values) for n, values in out.items()}


def _column(table, index, multiplicity=1):
    """The invariants of ``multiplicity`` copies of one column of the group's list."""
    vector = [0] * len(table._ebrs)
    vector[index] = multiplicity
    return table._theta_columns(vector)


def _implied_bands(table, theta_z):
    """The band count the table's band-count form reads off ``theta_Z``."""
    return sum((table._band_count_form[i] * Fraction(theta_z[table._rows[r].name])
                for i, r in enumerate(table._z_rows)), Fraction(0))


def _p_of(table, realization):
    """A realization's multiplicities over the table's maximal keys, in their order."""
    counts = {(t.irrep, t.wyckoff): t.multiplicity for t in realization}
    return tuple(counts.get((key.term.irrep, key.term.wyckoff), 0)
                 for key in table._maximal_keys)


def _bands_of(table, realization):
    """The bands a realization carries, each key's multiplicity times its dimension."""
    return sum(m * key.dim for m, key in zip(_p_of(table, realization),
                                             table._maximal_keys))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fkm_phases(realized):
    """Fu-Kane-Mele's strong and trivial distorted phases in ``R-3m1' #166.98``."""
    out = {}
    for name in ("strong", "trivial"):
        got = realized(fkm_regime(name), decorations=FKM_BOND,
                       characters={0: "s"})
        assert got, f"{name}: {got}"
        assert got.group.number_str == "166.98"
        out[name] = got
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


@pytest.fixture(scope="module")
def sg211():
    """``I432 #211.57`` without SOC -- the paper's own ``12d`` example."""
    table = tqr.srsi_tables(211, spinor=False)
    assert table, str(table)
    return table


# ---------------------------------------------------------------------------
# 1.  the basis of the realizations: the maximal positions and their band counts
# ---------------------------------------------------------------------------

def test_the_maximal_keys_are_the_ebr_columns_then_the_composite_ones(sg211):
    """The basis the enumeration runs over, in the order it is written in.

    Every EBR column of ``ebr_data``, in that function's own order and with its
    own ``(n)`` as the band count, then the maximal positions whose induced band
    representation is composite.  ``#211.57`` has no composite key -- all 17 of
    its maximal keys are EBR columns -- which is what makes it the clean case to
    pin the order on.

    A key carries the **supplement's** spelling, with the EBR list's alongside it
    where the two differ: ``12d`` here is a ``222`` site, where the supplement
    writes ``A_1`` and the EBR tables write ``A``, so that one key reads
    ``irrep='A1'``, ``ebr_label='A'`` -- the naming convention derived by
    elimination rather than typed.
    """
    keys = sg211._maximal_keys
    assert len(keys) == 17 == len(sg211._ebrs)
    assert [(key.ebr_label or key.irrep, key.letter) for key in keys] == \
        list(sg211._ebrs)
    assert [(index, key.irrep, key.ebr_label) for index, key in enumerate(keys)
            if key.ebr_label] == [(13, "A1", "A")]
    assert [key.dim for key in keys] == [1, 1, 2, 3, 3, 3, 3, 3, 3, 6, 4, 4, 8,
                                         6, 6, 6, 6]
    assert keys[16].label == "(B3)_d"
    assert keys[16].ebr_name.startswith("B3")
    assert "6" in keys[16].ebr_name           # the (n) the dimension came from


@pytest.mark.data
def test_the_band_count_is_a_unique_linear_form_in_the_z_srsis_everywhere():
    """``nu = a . theta_Z``, solved on the EBR columns, over all 434 systems.

    The band count is itself a stable real-space invariant -- no adiabatic move
    creates or destroys a band -- so it *has* to be a Z-linear form in the
    Z-SRSIs, and it is, in every one of the 434 tabulated systems.  It is also
    **unique**: ``Theta0`` restricted to the EBR columns has full row rank in
    every system, so the value the form gives a *composite* maximal key is not a
    choice.  That is what licenses ``_MaximalKey.dim`` there, and the dimension
    is what bounds the enumeration.

    The form is asserted where its right-hand side comes from **outside**
    ``Theta``: on each of the 4731 EBR columns, against the ``(n)`` written into
    that EBR's own ``_tables._ebr_data`` name.  On a *composite* key the same expression
    is how ``_maximal_basis`` computed ``dim`` in the first place, so that line
    restates the construction and is labelled as such.

    Measured: 434 systems, 4731 EBR columns, 39 composite maximal keys in 28
    systems (all without SOC), zero rank-deficient blocks, and **no maximal key
    with an all-zero Theta column**.
    """
    systems = columns = composite = zero = against_ebr_names = 0
    deficient = []
    for sg in range(1, 231):
        for spinor in (False, True):
            table = tqr.srsi_tables(sg, spinor=spinor)
            if not table:
                continue
            systems += 1
            columns += len(table._ebrs)
            block = np.array([[table._theta_on_columns[i][j] for j in range(len(table._ebrs))]
                              for i in table._z_rows], dtype=float)
            if np.linalg.matrix_rank(block) != len(table._z_rows):
                deficient.append((sg, spinor))
            keys = table._maximal_keys
            form = table._band_count_form
            theta_columns = _srsitables._maximal_basis(sg, spinor)[1]
            for index, key in enumerate(keys):
                theta = theta_columns[index]
                if not any(theta):
                    zero += 1
                value = sum((form[i] * Fraction(int(theta[r]))
                             for i, r in enumerate(table._z_rows)), Fraction(0))
                if key.ebr:
                    # the headline: the form's value against a band count it did
                    # not produce -- the (n) in the EBR's own name
                    against_ebr_names += 1
                    nu = int(_EBR_BANDS.search(str(key.ebr_name)).group(1))
                    assert value == nu, (sg, spinor, key, value, nu)
                    assert key.dim == nu, (sg, spinor, key, nu)
                else:
                    composite += 1
                    # ... and here the same expression *is* how `_maximal_basis`
                    # computed `dim`, so this restates the construction
                    assert value == key.dim, (sg, spinor, key, value)
    assert systems == 434
    assert columns == 4731
    assert against_ebr_names == 4731
    assert composite == 39
    assert deficient == [], deficient
    assert zero == 0


def test_the_maximal_key_arithmetic_agrees_with_theta(sg211, monkeypatch):
    """Two paths to the same number, and the fast one is the one that is used.

    ``atomic_realizations`` evaluates over precomputed integer columns; the
    primitive is ``theta`` on the paper's whole basis.  At one copy of a key the
    two read the same stored column, so the loop checks the agreement across
    sources: a mapping key against the same key's EBR column, ``theta(vector)
    == by_sites`` -- the converted record's per-term ``(irrep, wyckoff)`` labels
    against ``_tables._ebr_data``'s own EBR spelling at the same column.  Where the two
    spellings differ -- ``A`` in the EBR list for the supplement's ``A1`` at
    ``12d`` -- the mapping takes the EBR list's spelling too and reads the same
    key.

    What is only the fast path's is the arithmetic it does with those columns:
    **one sum over the keys, reduced once** at the end.  Three copies of
    ``(B3)_12d`` therefore land back on ``theta_Z2 = (0, 0, 1)`` where two land
    on ``(0, 0, 0)``, and a doubled ``(B3)_12d`` beside one ``(A1)_2a`` reads
    ``(1, 0, 0)``; those are written down.  The last block is the negative
    control for "the fast one is the one that is used": with the table's own
    evaluator taken away, the precomputed columns still answer.
    """
    aliased = 0
    for key in sg211._maximal_keys:
        position = key.term.wyckoff
        by_sites = sg211.theta({(key.irrep, position): 1})
        if key.ebr:
            assert _column(sg211, key.column) == by_sites, key
            if key.ebr_label:
                assert sg211.theta({(key.ebr_label, position): 1}) == by_sites, key
                aliased += 1
    assert aliased == 1, "the one key whose two spellings differ"

    def over_keys(multiplicities):
        """``p`` over the maximal keys, from ``{key index: multiplicity}``."""
        p = [0] * len(sg211._maximal_keys)
        for index, multiplicity in multiplicities.items():
            p[index] = multiplicity
        return p

    def zn_of_keys(multiplicities):
        return _by_modulus(tqr._theta_zn_of_keys(sg211, over_keys(multiplicities)))

    # the sum is taken over the keys and reduced modulo each row's own n once,
    # at the end, so a multiplicity above 1 wraps
    assert zn_of_keys({16: 1}) == {2: (0, 0, 1)}
    assert zn_of_keys({16: 2}) == {2: (0, 0, 0)}
    assert zn_of_keys({16: 3}) == {2: (0, 0, 1)}
    assert zn_of_keys({16: 2, 0: 1}) == {2: (1, 0, 0)}

    def no_slow_path(self, p):
        raise AssertionError("the mapping-keyed path was taken")

    with monkeypatch.context() as mp:
        for name in ("theta", "_theta_columns",
                     "theta_unreduced", "_theta_unreduced_columns"):
            mp.setattr(tqr.SRSITables, name, no_slow_path)
        assert zn_of_keys({16: 3}) == {2: (0, 0, 1)}


# ---------------------------------------------------------------------------
# 2.  the enumeration
# ---------------------------------------------------------------------------

def test_atomic_realizations_enumerates_every_realization_of_one_symmetry_vector(sg211):
    """``(B3)_12d``'s own symmetry data, and the four configurations that carry it.

    The paper's ``I432`` example.  Its Z-SRSIs are ``(1, 0, 1, 1, 1, -1, 0)`` and
    the enumeration finds **four** atomic realizations at 6 bands -- ``(B3)_d``,
    ``(B2)_d`` (the pair no measurement separates), ``(T2)_a + (A2)_b`` and
    ``(T1)_a + (B2)_b`` -- of which the first two carry ``theta_Z2 = (0, 0, 1)``
    and the others ``(0, 0, 0)`` and ``(1, 1, 0)``.  That spread over one
    symmetry data vector is the whole reason the criterion needs a quantifier: the
    Z_n-SRSIs are *not* functions of the symmetry data.
    """
    column = _column(sg211, 16)
    theta_z = column.theta_Z
    assert tuple(str(v) for v in theta_z.values()) == ("1", "0", "1", "1", "1", "-1", "0")
    assert _by_modulus(column.theta_Zn) == {2: (0, 0, 1)}

    got = tqr.atomic_realizations(sg211, theta_z)
    assert got and got.realizations, str(got)
    assert _implied_bands(sg211, theta_z) == 6
    assert [realization.label for realization in got.realizations] == [
        "1 (B3)_d", "1 (B2)_d", "1 (T2)_a + 1 (A2)_b", "1 (T1)_a + 1 (B2)_b"]
    assert [_by_modulus(realization.theta_Zn)[2] for realization in got.realizations] == [
        (0, 0, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)]
    assert all(_bands_of(sg211, realization) == 6 for realization in got.realizations)
    # the two that use the same-content pair say so; the other two do not
    assert [len(realization.stably_equivalent) for realization in got.realizations] == \
        [1, 1, 0, 0]
    assert got.realizations[0].stably_equivalent[0].irreps == ("B2", "B3")
    assert "4 atomic realizations with Z-valued" in got.summary()


def _realizations_over_br(table, y, n_bands):
    """Every ``p >= 0`` with ``BR . p = y`` and ``sum_j dim_j p_j == n_bands``.

    The **band-representation** system, by brute force over ``_tables._ebr_data``'s own
    integer matrix -- the redundant system ``atomic_realizations`` replaces with
    one equation per Z-SRSI, enumerated here without touching ``Theta`` at all.
    Exhaustive by the same argument the enumeration's own bound uses: with every
    ``dim_j >= 1`` and ``p >= 0`` the band count caps each ``p_j`` at
    ``n_bands // dim_j``, and the recursion walks all of them.  Returned sorted,
    because a set is what is being compared.
    """
    data = _tables._ebr_data(_srsitables._TableKey(table.group.number_str, table.group.spinor))
    matrix = np.asarray(data["matrix"], dtype=int)
    dims = [int(_EBR_BANDS.search(str(name)).group(1))
            for name, _position in data["ebrs"]]
    target = np.asarray(y, dtype=int)
    found, p = [], [0] * len(dims)

    def walk(j, left):
        if j == len(dims):
            if left == 0 and (matrix @ np.array(p, dtype=int) == target).all():
                found.append(tuple(p))
            return
        for multiplicity in range(left // dims[j] + 1):
            p[j] = multiplicity
            walk(j + 1, left - multiplicity * dims[j])
        p[j] = 0

    walk(0, int(n_bands))
    return sorted(found)


def test_the_theta_enumeration_is_the_band_representation_enumeration(sg211):
    """``ker BR = ker Theta0``: the premise that licenses the whole substitution.

    ``atomic_realizations`` solves one equation per Z-SRSI instead of the redundant
    band-representation system ``BR . p = y``, and that substitution is exact
    only because the two have the same kernel.  The module docstring above and
    ``atomic_realizations``' own assert it; here it is **measured**, by asking
    ``BR`` the same questions by brute force with ``Theta`` never touched:

    * ``(B3)_12d``'s symmetry data vector at 6 bands -- the same **four** ``p``;
    * the fragile ``2 x (B3)_12d - (A1)_2a`` at 11 bands -- **none**, so the
      ``INFEASIBLE`` below is a proof of none over ``BR`` as well.

    ``#211.57``'s 17 maximal keys are exactly its 17 EBR columns in order, so a
    ``p`` over the keys and a ``p`` over the columns are the same vector and the
    two solution sets can be compared entry for entry.
    """
    keys = sg211._maximal_keys
    assert len(keys) == len(sg211._ebrs)
    assert all(key.ebr and key.column == index for index, key in enumerate(keys))
    data = _tables._ebr_data(_srsitables._TableKey(sg211.group.number_str, False))
    matrix = np.asarray(data["matrix"], dtype=int)

    def theta_z_of(index):
        return _column(sg211, index).theta_Z

    got = tqr.atomic_realizations(sg211, theta_z_of(16))
    assert got.realizations and _implied_bands(sg211, got.theta_Z) == 6
    over_br = _realizations_over_br(sg211, matrix[:, 16], 6)
    assert len(over_br) == 4, over_br
    assert sorted(_p_of(sg211, realization) for realization in got.realizations) == over_br

    fragile = {name: 2 * a - theta_z_of(0)[name] for name, a in theta_z_of(16).items()}
    none = tqr.atomic_realizations(sg211, fragile)
    assert not none.realizations and _implied_bands(sg211, fragile) == 11
    assert _realizations_over_br(sg211, 2 * matrix[:, 16] - matrix[:, 0], 11) == []


def test_the_enumeration_bound_is_a_bound_and_not_a_guess(sg211):
    """A multiplicity **at** the bound, where every other realization here is below it.

    ``bound = n_bands // min_j dim_j`` is what makes ``INFEASIBLE`` a proof of
    none: it is the largest any ``p_j`` can be once ``sum_j dim_j p_j =
    n_bands`` with every ``dim_j >= 1`` and ``p >= 0``.  Every other realization in
    this file has every multiplicity equal to 1, so nothing there tells a correct
    bound from a too-generous one.  **Twice** the ``(A1)_2a`` column does: its
    only realization is ``2 (A1)_a`` itself, at 2 bands with ``min(dim) == 1``
    and so ``bound == 2`` -- a bound one smaller would serve ``INFEASIBLE`` on a
    band set that has a realization.

    Twice ``(B3)_12d`` is the same statement with room in it: eleven
    realizations at 12 bands, **four** of them carrying a multiplicity of 2, and
    the brute force over ``BR`` finds the same eleven.
    """
    data = _tables._ebr_data(_srsitables._TableKey(sg211.group.number_str, False))
    matrix = np.asarray(data["matrix"], dtype=int)

    def doubled(index):
        return _column(sg211, index, 2).theta_Z

    def p_of(realization):
        return _p_of(sg211, realization)

    got = tqr.atomic_realizations(sg211, doubled(0))
    assert got.realizations and _implied_bands(sg211, got.theta_Z) == 2
    assert [realization.label for realization in got.realizations] == ["2 (A1)_a"]
    assert got.numerical.bound == 2 and max(p_of(got.realizations[0])) == 2   # exactly reached
    assert _realizations_over_br(sg211, 2 * matrix[:, 0], 2) == \
        [p_of(realization) for realization in got.realizations]

    got = tqr.atomic_realizations(sg211, doubled(16))
    assert got.realizations and _implied_bands(sg211, got.theta_Z) == 12
    assert got.numerical.bound == 12
    assert len(got.realizations) == 11
    assert sum(1 for realization in got.realizations if max(p_of(realization)) == 2) == 4
    assert sorted(p_of(realization) for realization in got.realizations) == \
        _realizations_over_br(sg211, 2 * matrix[:, 16], 12)


def test_a_composite_site_irrep_is_in_the_realizations_basis_and_is_found():
    """The 39 keys no EBR list carries, exercised end to end.

    ``P4_2/m1' #84.52`` has ``{}^1E{}^2E`` at site symmetry ``-4`` on ``2e`` and
    ``2f`` -- site-symmetry irreps whose induced band representation is
    *composite*, so no EBR column carries them, and the enumeration would simply
    miss those configurations if the basis were the EBR block.  Handed that key's
    own ``theta_Z``, ``atomic_realizations`` finds **two** realizations at 4 bands:
    the key itself, and a ``Bg + Bu`` pair -- which is what "composite" means, an
    induced band representation whose content is a sum of EBRs' contents.

    **Which** pair is the measurement, and it tracks the position rather than
    the irrep: the ``2e`` key decomposes at ``2a`` and the ``2f`` key at ``2b``,
    never the other way round.  Both are pinned per key.

    The band count, 4, came from the band-count form rather than from any table,
    and it is what puts the key in the search at all.
    """
    table = tqr.srsi_tables(84, spinor=False)
    assert table, str(table)
    composite = [(index, key) for index, key in enumerate(table._maximal_keys)
                 if not key.ebr]
    assert [(key.irrep, key.letter, key.dim) for _index, key in composite] == [
        ("1E2E", "e", 4), ("1E2E", "f", 4)]
    # and they sit *past* the EBR block, which is what `atomic_realizations`'
    # `rows`/`dims` construction over the same key order assumes
    assert [index for index, _key in composite] == \
        [len(table._ebrs), len(table._ebrs) + 1]
    realizations = {"e": ["1 (1E2E)_e", "1 (Bg)_a + 1 (Bu)_a"],
                    "f": ["1 (1E2E)_f", "1 (Bg)_b + 1 (Bu)_b"]}
    for _index, key in composite:
        own = table.theta({(key.irrep, key.term.wyckoff): 1})
        assert own.theta_Zn == {}, "SG 84 has no Z_n row"
        got = tqr.atomic_realizations(table, own.theta_Z)
        assert got and got.realizations, str(got)
        assert _implied_bands(table, got.theta_Z) == 4
        assert [realization.label for realization in got.realizations] == \
            realizations[key.letter], key


def test_atomic_realizations_returns_none_on_a_fractional_theta_z(sg211):
    """A stable band set has no atomic realization at all, and nothing is enumerated.

    By the paper's theorem a fractional Z-SRSI *is* a nonzero symmetry
    indicator, so no atomic insulator carries that symmetry data.  The record
    comes back with no realization, the fractions as handed in, and no
    enumeration run (``bound == 0``); its summary says the invariants are
    fractional.
    """
    names = [row.name for row in sg211._rows if row.mod is None]
    half = dict(zip(names, [Fraction(1, 2), 0, 0, 0, 0, 0, 0]))
    out = tqr.atomic_realizations(sg211, half)
    assert out and out.realizations == (), str(out)
    assert out.theta_Z == half and list(out.theta_Z) == names
    assert out.numerical.bound == 0
    assert "  fractional   (.theta_Z)" in out.summary().splitlines()
    # the contrast: the same vector with an integral first entry has a realization
    assert tqr.atomic_realizations(sg211, dict(zip(names, [1, 0, 0, 0, 0, 0, 0]))).realizations


def test_atomic_realizations_returns_none_on_a_fragile_theta_z(sg211):
    """**An empty list is a verdict**: integral Z-SRSIs with no realization.

    Constructed from the tables themselves, as a difference of EBR columns --
    ``2 x (B3)_12d - (A1)_2a`` -- which is integral (so the symmetry indicators
    see nothing) and has no nonnegative-integer configuration behind it.  This
    is the affine-monoid statement of Song et al. in theta-coordinates, and the
    arithmetic signature of a fragile band set; it is served, not refused.

    The bound the enumeration used is rigorous -- ``sum_j dim_j p_j = n_bands``
    with every ``dim_j >= 1`` and ``p >= 0`` bounds every ``p_j`` -- so
    ``INFEASIBLE`` here is a proof of none rather than a failure to find one.

    The **other** ``INFEASIBLE`` is the contrast at the end, and it means
    something else: a ``theta_Z`` whose implied band count is not a nonnegative
    integer describes no band set at all, so there is nothing to search and the
    record comes back with ``bound == 0``.
    """
    def column(index):
        return _column(sg211, index).theta_Z

    fragile = {name: 2 * a - column(0)[name] for name, a in column(16).items()}
    got = tqr.atomic_realizations(sg211, fragile)
    assert got, str(got)                       # the record is truthy
    assert got.realizations == ()
    assert _implied_bands(sg211, fragile) == 11    # 2 x 6 - 1, read off theta_Z
    assert got.numerical.bound == 11
    assert got.summary().split(": ", 1)[1].startswith("no atomic realization")
    assert "fractional" not in got.summary()

    # ... and the one that never gets as far as the search: -(A1)_2a's theta_Z
    # implies -1 bands through the band-count form (1, 1, 2, 3, 3, 3, 4)
    negative = {name: -v for name, v in column(0).items()}
    empty = tqr.atomic_realizations(sg211, negative)
    assert empty, str(empty)                   # served, not refused
    assert empty.realizations == ()
    assert _implied_bands(sg211, negative) == -1 and empty.numerical.bound == 0


def test_atomic_realizations_refuses_a_truncated_enumeration(sg211):
    """A no-go is only ever asserted on a **complete** enumeration.

    At ``cap=1`` the search stops after one solution, and neither "no pair adds
    up" nor "none exists" may be claimed from that -- so the cap is a refusal
    and not a short list.  The same input at the default cap serves four.
    """
    theta_z = _column(sg211, 16).theta_Z
    out = tqr.atomic_realizations(sg211, theta_z, cap=1)
    assert not out and out.gate == "srsi_realizations_truncated", str(out)
    # the band-count form was evaluated before the cap fired (`context["cap"]`
    # beside it is the keyword read back, so it is not asserted)
    assert out.context["n_bands"] == 6
    assert len(tqr.atomic_realizations(sg211, theta_z, cap=256).realizations) == 4


def test_the_grammar_mistakes_are_value_errors_not_refusals(sg211):
    """Four caller bugs, each matched on the wording of **its own** check.

    A ``theta_Z`` written as a tuple rather than by name; one naming invariants
    the group does not have; a splitting whose two sides do not add up to the
    total; and a ``theta_Zn`` naming invariants valued modulo n the group does
    not have.  The last two are ``nsit_criterion``'s, and the additivity check
    is the one branch ``splitting`` can never reach: it builds ``theta_Z[C]`` as
    ``total - V``, so it cannot hand in three sets that do not add up.

    Each pattern here is a piece of wording only one of the checks has.
    """
    theta_total = _column(sg211, 16)
    theta_z = theta_total.theta_Z
    with pytest.raises(TypeError, match="theta_Z= is of type tuple; pass a mapping from "
                                        "each integer-valued invariant's name to its value"):
        tqr.atomic_realizations(sg211, tuple(theta_z.values()))
    with pytest.raises(ValueError, match="theta_Z= holds theta_1, theta_2, theta_3; "
                                         r"#211\.57 has the integer-valued invariants"):
        tqr.atomic_realizations(sg211, {"theta_1": 0, "theta_2": 0, "theta_3": 0})
    with pytest.raises(ValueError, match="theta_Z_v= plus theta_Z_c= is"):
        tqr.nsit_criterion(sg211, theta_total, theta_z, theta_z)
    zeros = {name: 0 for name in theta_z}
    wrong_moduli = replace(theta_total, theta_Zn={"theta_8": 0})
    with pytest.raises(ValueError, match="theta_total= holds the invariants valued modulo n "
                                         "theta_8, and #211.57 has"):
        tqr.nsit_criterion(sg211, wrong_moduli, theta_z, zeros)


# ---------------------------------------------------------------------------
# 3.  the site irreps the SRSIs do not distinguish
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_the_same_content_sweep_over_every_tabulated_system():
    """**36 pairs**: same-site site irreps with one symmetry data vector and one set of SRSIs.

    Two distinct site irreps at the same maximal Wyckoff position whose
    ``ebr_data`` columns are identical carry the same symmetry data; where the
    tables give them matching Z- and Z_n-SRSIs as well, the two are stably
    equivalent, by the paper's SN 3B.  Swept over every tabulated system: 36 such
    pairs in 26 space groups, every one of them a pair (never a larger set) and
    every one spinless, tabulated in :data:`SAME_CONTENT_PAIRS`.
    """
    found = []
    for sg in range(1, 231):
        for spinor in (False, True):
            table = tqr.srsi_tables(sg, spinor=spinor)
            if not table:
                continue
            for pair in table.stably_equivalent:
                assert len(pair._columns) == 2, (sg, spinor, pair)
                assert not spinor, (sg, pair)
                assert _column(table, pair._columns[0]) == \
                    _column(table, pair._columns[1]), (sg, pair)
                found.append((sg, pair.wyckoff.lstrip("0123456789"), pair.irreps))
    assert tuple(found) == SAME_CONTENT_PAIRS, found


def test_the_stably_equivalent_pairs_of_a_measured_group_are_its_numbers(
        lin3g_declared_cross_checked):
    """The tables of a measured group hold its number's pairs, and ``P4mm1'`` has none."""
    group = lin3g_declared_cross_checked.group
    assert tqr.srsi_tables(group).stably_equivalent == \
        tqr.srsi_tables(99, spinor=True).stably_equivalent == ()


# ---------------------------------------------------------------------------
# 4.  the split-EBR regression
# ---------------------------------------------------------------------------

def _split_thetas(table, split):
    """``(theta_total, theta_Z[V], theta_Z[C])`` for one printed splitting."""
    theta_total = _column(table, split.parent._column)
    theta_v = _column(table, split.children[0]._column).theta_Z
    theta_c = {name: a - theta_v[name] for name, a in theta_total.theta_Z.items()}
    return theta_total, theta_v, theta_c


@pytest.mark.data
def test_every_split_ebr_flag_is_reproduced_from_the_theta_arithmetic():
    """The supplement's 211 splittings, and its own ``xmark`` re-derived.

    ``Lambda = V (+) C`` with every constituent an EBR: 9 without SOC and 202
    with, of which the captions flag 9 and 194 as SRSI-mismatching.  Two
    statements, both exact:

    * ``theta_Z(parent) = sum theta_Z(children)`` on **all 211** -- the Z-SRSIs
      are linear in ``p`` and the symmetry data vectors add, so this is what a
      splitting *is*;
    * ``srsi_mismatch`` iff some Z_n row of the parent differs from the
      children's sum modulo that row's own ``n`` -- **all 211 flags
      reproduced**, in both directions.

    That is the whole of the supplement's own column, recomputed from the theta
    rows it is printed beside.
    """
    counts = {"single": 0, "double": 0}
    flagged = {"single": 0, "double": 0}
    additive = reproduced = 0
    for sg in range(1, 231):
        for spinor in (False, True):
            table = tqr.srsi_tables(sg, spinor=spinor)
            if not table or not table.split_ebrs:
                continue
            kind = "double" if spinor else "single"
            for split in table.split_ebrs:
                counts[kind] += 1
                flagged[kind] += int(split.srsi_mismatch)

                def column(entry):
                    vector = [0] * len(table._ebrs)
                    vector[int(entry._column)] = 1
                    return vector

                parent = table._theta_unreduced_columns(column(split.parent))
                children = [Fraction(0)] * len(parent)
                for child in split.children:
                    for i, value in enumerate(table._theta_unreduced_columns(column(child))):
                        children[i] += value
                if [parent[i] for i in table._z_rows] == \
                        [children[i] for i in table._z_rows]:
                    additive += 1
                mismatch = any((int(parent[i]) - int(children[i])) % n
                               for n, rows in table._zn_rows.items() for i in rows)
                if mismatch == bool(split.srsi_mismatch):
                    reproduced += 1
    assert counts == {"single": 9, "double": 202}
    assert flagged == {"single": 9, "double": 194}
    assert additive == 211
    assert reproduced == 211


def test_the_one_splitting_the_full_enumeration_does_not_diagnose():
    """``#209.49``, in detail: why the printed pair mismatches and another does not.

    ``F432`` with SOC.  The parent ``(1F2F)_8c`` splits into ``(F)_4a (+)
    (F)_4b``, which the supplement flags: the two children's Z_2-SRSIs sum to
    ``(0, 0, 1, 0, 0)`` where the parent has ``(0, 0, 0, 0, 0)``.

    But ``(F)_4a`` and ``(F)_4b`` induce the **same** symmetry-data vector, so
    each side's symmetry data admits *both* realizations -- and ``a + a`` and
    ``b + b`` both add up.  The supplement prints those two as splittings of the
    same parent, unflagged.  The reading is not that the splitting is trivial:
    it is that any topology it carries is invisible to the SRSIs.
    """
    table = tqr.srsi_tables(209, spinor=True)
    assert table and table._zn_rows == {2: (3, 4, 5, 6, 7)}
    splits = [s for s in table.split_ebrs if s.parent.irrep == "-1F2F"]
    assert len(splits) == 3
    assert [s.srsi_mismatch for s in splits] == [False, True, False]

    flagged = splits[1]
    assert tuple((c.irrep, c.wyckoff) for c in flagged.children) == \
        (("-F", "4a"), ("-F", "4b"))
    theta_total, theta_v, theta_c = _split_thetas(table, flagged)
    assert _by_modulus(theta_total.theta_Zn) == {2: (0, 0, 0, 0, 0)}
    got = tqr.nsit_criterion(table, theta_total, theta_v, theta_c)
    assert got and got.nsit is False
    assert got.valence.realizations and got.conduction.realizations
    valence = got.valence.realizations
    assert [c.label for c in valence] == ["1 (-F)_b", "1 (-F)_a"]
    assert [_by_modulus(c.theta_Zn)[2] for c in valence] == \
        [(0, 0, 1, 0, 0), (0, 0, 0, 0, 0)]
    assert sorted((valence.index(v), got.conduction.realizations.index(c))
                  for v, c in got.consistent_pairs) == [(0, 0), (1, 1)]
    assert ": not nsit, 2 pairs of a valence and a conduction realization adding up " \
        "to the whole model's invariants valued modulo n" in got.summary().splitlines()[0]
    # ... and the printed pair is the one that does not add up
    printed_v = _by_modulus(_column(table, flagged.children[0]._column).theta_Zn)
    printed_c = _by_modulus(_column(table, flagged.children[1]._column).theta_Zn)
    assert printed_v == {2: (0, 0, 0, 0, 0)} and printed_c == {2: (0, 0, 1, 0, 0)}


# ---------------------------------------------------------------------------
# 5.  the models
# ---------------------------------------------------------------------------

def test_the_trivial_phase_runs_the_criterion_and_the_group_has_no_zn_row(fkm_phases):
    """``#166.98`` cannot produce the diagnosis, and the record says which.

    Fu-Kane-Mele's trivial phase has an integral ``theta_Z[V] = (0,0,0,1,0,0,0,0)``
    and an integral complement ``(0,0,1,0,0,0,0,0)``, so the criterion runs and
    both sides come back with exactly one atomic realization -- ``(-E1u)_3a`` and
    ``(-E1g)_3a``.  But ``R-3m1'`` has **no Z_n-valued SRSI at all**, so there is
    no additivity to test: ``nsit`` is false, and the summary's first line says
    it is the group that has no invariants valued modulo n.

    The strong phase is the contrast: its ``theta_Z[V]`` is fractional, which by
    the paper's theorem is a nonzero symmetry indicator, so no atomic insulator
    has the valence pair's symmetry data and there is nothing for an
    enumeration of realizations to do.  ``nsit`` is false, both sides come back
    with no realization and their fractional invariants, and the summary says
    so.

    A consistent pair hides any topology from the SRSIs and never makes a
    splitting trivial, the criterion being sufficient, never necessary.
    """
    got = tqr.splitting(fkm_phases["trivial"], bands=2)
    assert got, str(got)
    assert got.nsit is False and got.theta_Zn == {}
    # the *group*, not the splitting, is what has no Z_n row -- read off the
    # table rather than off the splitting
    tables = tqr.srsi_tables(fkm_phases["trivial"].group)
    assert [name for name in tables.invariants if tables.moduli[name]] == []
    assert got.valence.realizations and got.conduction.realizations
    assert [c.label for c in got.valence.realizations] == ["1 (-E1u)_a"]
    assert [c.label for c in got.conduction.realizations] == ["1 (-E1g)_a"]
    indicator = tqe.symmetry_indicator(tqw.hsp_table(fkm_phases["trivial"], bands=2))
    assert indicator.values == (0, 0) and indicator.order == 1
    first = got.summary().splitlines()[0]
    assert first.endswith(": not nsit, the group having no invariants valued modulo n   "
                          "(.group, .nsit, .theta_Zn)")
    assert got.summary().count("no invariants valued modulo n") == 1, "said once"

    stable = tqr.splitting(fkm_phases["strong"], bands=2)
    assert stable.nsit is False
    assert stable.valence.realizations == () and stable.conduction.realizations == ()
    assert {name: str(v) for name, v in stable.valence.theta_Z.items()} == {
        "theta_1": "-1/4", "theta_2": "0", "theta_3": "1/2", "theta_4": "1",
        "theta_5": "1/4", "theta_6": "-1/2", "theta_7": "-1/4", "theta_8": "1/4"}
    assert tuple(str(v) for v in stable.conduction.theta_Z.values()) == (
        "1/4", "0", "1/2", "0", "-1/4", "1/2", "1/4", "-1/4")
    assert ": not nsit, the valence bands' invariants being fractional" \
        in stable.summary().splitlines()[0]


def test_lin3g_p2_has_one_atomic_realization_and_the_enumeration_is_complete(
        lin3g_declared_cross_checked):
    """The enumeration inside lin3g's NSIT splitting, on its fragile ``P2``.

    Split above its sixth band, lin3g has ``nsit`` true: both sides have
    realizations and no pair of them reproduces the total's
    ``theta_Zn = {2: (0, 1)}``.  Its valence
    set contains the suite's fragile golden ``P2``, whose only atomic
    realization is ``(-E2)_1b``, and this is the enumeration that finds it, on
    ``P2``'s own ``theta_Z``: the basis of maximal keys it is written over with
    each key's band dimension, the band count read off ``theta_Z`` through the
    table's own form rather than trusted, and the per-coordinate bound --
    rigorous, since ``sum_j dim_j p_j = n_bands`` with every ``dim_j >= 1``
    bounds every ``p_j``, so the enumeration is complete.
    """
    tables = tqr.srsi_tables(lin3g_declared_cross_checked.group)
    p2 = {"theta_1": Fraction(0), "theta_2": Fraction(0), "theta_3": Fraction(1)}
    cand = tqr.atomic_realizations(tables, p2)
    assert cand and cand.realizations, str(cand)
    assert [key.label for key in tables._maximal_keys] == [
        "(-E1)_a", "(-E2)_a", "(-E1)_b", "(-E2)_b", "(-E)_c"]
    assert [key.dim for key in tables._maximal_keys] == [2, 2, 2, 2, 4]
    assert tuple(str(v) for v in tables._band_count_form) == ("2", "2", "2")
    assert _implied_bands(tables, p2) == 2 and cand.numerical.bound == 1
    assert [c.label for c in cand.realizations] == ["1 (-E2)_b"]


def test_kane_mele_odd_has_a_fragile_conduction_pair_and_even_has_no_zn_row(
        km_realized):
    """``#156.50`` has no Z_n row either -- and one regime has a fragile side.

    Kane-Mele's **odd** regime: the valence pair's symmetry data is that of
    ``(-E1)_1a`` (one realization), and the conduction pair's
    ``theta_Z = (0, 1, -1, -1)`` has **no** nonnegative realization at two bands
    at all.  That side is fragile, which its symmetry data already shows, so
    the splitting is not NSIT -- ``nsit`` is false whatever the moduli, and the
    summary's first line says the conduction bands have no atomic realization
    rather than claiming a Z_n mismatch the group cannot have.

    The **even** regime splits into ``(-E1)_1b`` and ``(-E1)_1c``, one
    realization each, and ``P3m11'`` has no Z_n row to test them with.
    """
    odd = tqr.splitting(km_realized["odd"], bands=2)
    assert odd, str(odd)
    assert odd.nsit is False
    assert odd.theta_Zn == {}
    assert odd.valence.realizations and not odd.conduction.realizations
    assert odd.conduction.realizations == ()
    assert tuple(str(v) for v in odd.conduction.theta_Z.values()) == ("0", "1", "-1", "-1")
    assert [c.label for c in odd.valence.realizations] == ["1 (-E1)_a"]
    assert ": not nsit, the conduction bands having no atomic realization   " \
        "(.group, .nsit, .conduction.realizations)" in odd.summary().splitlines()[0]

    even = tqr.splitting(km_realized["even"], bands=2)
    assert even, str(even)
    assert even.nsit is False and even.theta_Zn == {}
    assert even.valence.realizations and even.conduction.realizations
    assert [c.label for c in even.valence.realizations] == ["1 (-E1)_b"]
    assert [c.label for c in even.conduction.realizations] == ["1 (-E1)_c"]
    assert ": not nsit, the group having no invariants valued modulo n" \
        in even.summary().splitlines()[0]


def test_the_band_set_is_one_keyword_on_both_routes(lin3g_declared_cross_checked):
    """``bands=`` names the block for ``splitting`` and for the band route alike.

    A list of bands is the same request as their count and the band set is
    required; a table is read on the block it was measured on, and is handed
    alone.
    """
    record = lin3g_declared_cross_checked
    want = tqr.splitting(record, bands=4)
    assert want, str(want)
    by_list = tqr.splitting(record, bands=[0, 1, 2, 3])
    assert by_list, str(by_list)
    assert by_list.nsit == want.nsit
    assert by_list.valence.theta_Z == want.valence.theta_Z
    assert by_list.conduction.theta_Z == want.conduction.theta_Z
    assert by_list.summary() == want.summary(), "the same split, said the same way"

    with pytest.raises(TypeError, match=r"^splitting\(\): no bands= is passed; pass the "
                                        r"valence bands, as their number or their indices\.$"):
        tqr.splitting(record)

    table = tqw.hsp_table(record, bands=[0, 1, 2, 3])
    by_list = tqr.srsi_of_bands(table)
    assert len(table.bands) == 4
    assert by_list.theta_Z == tqr.srsi_of_bands(tqw.hsp_table(record, bands=4)).theta_Z, \
        "a list of bands is the same block as their count"
    with pytest.raises(TypeError, match="pass the table alone"):
        tqr.srsi_of_bands(table, record)
