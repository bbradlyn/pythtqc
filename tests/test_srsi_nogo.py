r"""Wave S2: the paper's **no-go criterion for a band splitting**, in theta-coordinates.

Where ``test_srsi_tables.py`` says the shipped tables are the paper's and
``test_srsi_model.py`` says the two routes onto a model are right, this file says
the one *verdict* the SRSI layer issues is right.

The criterion (SN "SRSI mismatch as a sufficient criterion for band topology"):
the full Hilbert space of an insulator is an atomic insulator, so ``B[H] = B[V] +
B[C]`` and ``theta_Z[H] = theta_Z[V] + theta_Z[C]`` always hold; and if ``V`` and
``C`` are *both* atomic insulators then ``theta_Zn`` must add up modulo ``n`` as
well, for **some** atomic realizations of the two with the right symmetry
vectors.  When no pair adds up, one of the two sides is not an atomic insulator.

1. **The basis of the realizations** (``SRSITables.maximal_keys``,
   ``SRSITables.band_count_form``).  Every maximal site key with the number of
   bands one orbital of it carries -- the EBR columns and the 39 composite ones.
   The band count is a Z-linear form in the Z-SRSIs, it is *unique* (the EBR
   block of ``Theta0`` has full row rank in all 434 systems), and no maximal key
   has an all-zero column.
2. **The enumeration** (``atomic_realizations``).  One equation per Z-SRSI instead
   of the redundant band-representation system, which is exact because
   ``ker BR = ker Theta0``; the bound is rigorous, so ``INFEASIBLE`` is a proof
   and a truncated enumeration is a refusal rather than an answer.
3. **The pairs no measurement separates** (``stably_equivalent_pairs``).
4. **The regression.**  The supplement's own list of 211 symmetry-indicated
   split EBRs: every flag reproduced from the theta arithmetic.  The criterion
   run on the same list with the **full** enumeration of atomic realizations --
   information the supplement does not carry -- is ``examples/tqc_srsi_tables.py``'s
   table; the one flagged splitting it does not diagnose is measured here.
5. **The models**, where the verdict has to be read rather than looked up.

**The vocabulary matters and is asserted.**  A consistent pair does not make a
splitting trivial -- it makes any topology of that splitting *invisible to the
SRSIs*.  The criterion is sufficient, never necessary.
"""

import re
from fractions import Fraction

import numpy as np
import pytest

from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc import _srsitables

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

#: The ``(n)`` an ``ebr_data`` EBR name ends in -- that EBR's **own** band
#: count, written by the EBR tables beside the name and nowhere in ``Theta``.
#: Read here rather than through ``srsi._ebr_dimension`` so the band-count form
#: is checked against a number the package's own arithmetic did not produce.
_EBR_BANDS = re.compile(r"\((\d+)\)\s*$")


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
    keys = sg211.maximal_keys
    assert len(keys) == 17 == len(sg211.ebrs)
    assert [(key.ebr_label or key.irrep, key.letter) for key in keys] == \
        list(sg211.ebrs)
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
    choice.  That is what licenses ``MaximalKey.dim`` there, and the dimension
    is what bounds the enumeration.

    The form is asserted where its right-hand side comes from **outside**
    ``Theta``: on each of the 4731 EBR columns, against the ``(n)`` written into
    that EBR's own ``ebr_data`` name.  On a *composite* key the same expression
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
            columns += len(table.ebrs)
            block = np.array([[table.theta_on_columns[i][j] for j in range(len(table.ebrs))]
                              for i in table.z_rows], dtype=float)
            if np.linalg.matrix_rank(block) != len(table.z_rows):
                deficient.append((sg, spinor))
            keys = table.maximal_keys
            form = table.band_count_form
            theta_columns = _srsitables._maximal_basis(sg, spinor)[1]
            for index, key in enumerate(keys):
                theta = theta_columns[index]
                if not any(theta):
                    zero += 1
                value = sum((form[i] * Fraction(int(theta[r]))
                             for i, r in enumerate(table.z_rows)), Fraction(0))
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
    against ``ebr_data``'s own EBR spelling at the same column.  Where the two
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
    for key in sg211.maximal_keys:
        by_sites = sg211.theta({(key.irrep, key.letter): 1})
        if key.ebr:
            vector = [0] * len(sg211.ebrs)
            vector[key.column] = 1
            assert sg211.theta_columns(vector) == by_sites, key
            if key.ebr_label:
                assert sg211.theta({(key.ebr_label, key.letter): 1}) == by_sites, key
                aliased += 1
    assert aliased == 1, "the one key whose two spellings differ"

    def over_keys(multiplicities):
        """``p`` over the maximal keys, from ``{key index: multiplicity}``."""
        p = [0] * len(sg211.maximal_keys)
        for index, multiplicity in multiplicities.items():
            p[index] = multiplicity
        return p

    # the sum is taken over the keys and reduced modulo each row's own n once,
    # at the end, so a multiplicity above 1 wraps
    assert tqr._theta_zn_of_keys(sg211, over_keys({16: 1})) == {2: (0, 0, 1)}
    assert tqr._theta_zn_of_keys(sg211, over_keys({16: 2})) == {2: (0, 0, 0)}
    assert tqr._theta_zn_of_keys(sg211, over_keys({16: 3})) == {2: (0, 0, 1)}
    assert tqr._theta_zn_of_keys(sg211, over_keys({16: 2, 0: 1})) == {2: (1, 0, 0)}

    def no_slow_path(self, p):
        raise AssertionError("the mapping-keyed path was taken")

    with monkeypatch.context() as mp:
        for name in ("theta", "theta_columns",
                     "theta_unreduced", "theta_unreduced_columns"):
            mp.setattr(tqr.SRSITables, name, no_slow_path)
        assert tqr._theta_zn_of_keys(sg211, over_keys({16: 3})) == {2: (0, 0, 1)}


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
    symmetry vector is the whole reason the criterion needs a quantifier: the
    Z_n-SRSIs are *not* functions of the symmetry data.
    """
    vector = [0] * len(sg211.ebrs)
    vector[16] = 1
    theta_z, theta_zn = sg211.theta_columns(vector)
    assert tuple(str(v) for v in theta_z) == ("1", "0", "1", "1", "1", "-1", "0")
    assert theta_zn == {2: (0, 0, 1)}

    got = tqr.atomic_realizations(sg211, theta_z)
    assert got and got.atomic_possible, str(got)
    assert got.n_bands == 6
    assert got.atomic_possible
    assert [realization.label for realization in got.realizations] == [
        "1 (B3)_d", "1 (B2)_d", "1 (T2)_a + 1 (A2)_b", "1 (T1)_a + 1 (B2)_b"]
    assert [realization.theta_Zn[2] for realization in got.realizations] == [
        (0, 0, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)]
    assert all(realization.n_bands == 6 for realization in got.realizations)
    # the two that use the same-content pair say so; the other two do not
    assert [len(realization.pairs) for realization in got.realizations] == [1, 1, 0, 0]
    assert got.realizations[0].pairs[0].irreps == ("B2", "B3")
    assert "4 atomic realizations of 6 bands" in got.summary()
    # the band count is redundant, and asserting the right one changes nothing
    assert tqr.atomic_realizations(sg211, theta_z, n_bands=6).realizations == \
        got.realizations


def _realizations_over_br(table, y, n_bands):
    """Every ``p >= 0`` with ``BR . p = y`` and ``sum_j dim_j p_j == n_bands``.

    The **band-representation** system, by brute force over ``ebr_data``'s own
    integer matrix -- the redundant system ``atomic_realizations`` replaces with
    one equation per Z-SRSI, enumerated here without touching ``Theta`` at all.
    Exhaustive by the same argument the enumeration's own bound uses: with every
    ``dim_j >= 1`` and ``p >= 0`` the band count caps each ``p_j`` at
    ``n_bands // dim_j``, and the recursion walks all of them.  Returned sorted,
    because a set is what is being compared.
    """
    data = tqg.ebr_data(_srsitables._TableKey(table.number_str, table.spinor))
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

    * ``(B3)_12d``'s symmetry vector at 6 bands -- the same **four** ``p``;
    * the fragile ``2 x (B3)_12d - (A1)_2a`` at 11 bands -- **none**, so the
      ``INFEASIBLE`` below is a proof of none over ``BR`` as well.

    ``#211.57``'s 17 maximal keys are exactly its 17 EBR columns in order, so a
    ``p`` over the keys and a ``p`` over the columns are the same vector and the
    two solution sets can be compared entry for entry.
    """
    keys = sg211.maximal_keys
    assert len(keys) == len(sg211.ebrs)
    assert all(key.ebr and key.column == index for index, key in enumerate(keys))
    data = tqg.ebr_data(_srsitables._TableKey(sg211.number_str, False))
    matrix = np.asarray(data["matrix"], dtype=int)

    def theta_z_of(index):
        vector = [0] * len(sg211.ebrs)
        vector[index] = 1
        return sg211.theta_columns(vector)[0]

    got = tqr.atomic_realizations(sg211, theta_z_of(16))
    assert got.atomic_possible and got.n_bands == 6
    over_br = _realizations_over_br(sg211, matrix[:, 16], 6)
    assert len(over_br) == 4, over_br
    assert sorted(realization.p for realization in got.realizations) == over_br

    fragile = tuple(2 * a - b for a, b in zip(theta_z_of(16), theta_z_of(0)))
    none = tqr.atomic_realizations(sg211, fragile)
    assert not none.atomic_possible and none.n_bands == 11
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
    data = tqg.ebr_data(_srsitables._TableKey(sg211.number_str, False))
    matrix = np.asarray(data["matrix"], dtype=int)

    def doubled(index):
        vector = [0] * len(sg211.ebrs)
        vector[index] = 2
        return sg211.theta_columns(vector)[0]

    got = tqr.atomic_realizations(sg211, doubled(0))
    assert got.atomic_possible and got.n_bands == 2
    assert [realization.label for realization in got.realizations] == ["2 (A1)_a"]
    assert got.bound == 2 and max(got.realizations[0].p) == 2   # exactly reached
    assert _realizations_over_br(sg211, 2 * matrix[:, 0], 2) == \
        [realization.p for realization in got.realizations]

    got = tqr.atomic_realizations(sg211, doubled(16))
    assert got.atomic_possible and got.n_bands == 12 and got.bound == 12
    assert len(got.realizations) == 11
    assert sum(1 for realization in got.realizations if max(realization.p) == 2) == 4
    assert sorted(realization.p for realization in got.realizations) == \
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
    composite = [(index, key) for index, key in enumerate(table.maximal_keys)
                 if not key.ebr]
    assert [(key.irrep, key.letter, key.dim) for _index, key in composite] == [
        ("1E2E", "e", 4), ("1E2E", "f", 4)]
    # and they sit *past* the EBR block, which is what `atomic_realizations`'
    # `rows`/`dims` construction over the same key order assumes
    assert [index for index, _key in composite] == \
        [len(table.ebrs), len(table.ebrs) + 1]
    realizations = {"e": ["1 (1E2E)_e", "1 (Bg)_a + 1 (Bu)_a"],
                    "f": ["1 (1E2E)_f", "1 (Bg)_b + 1 (Bu)_b"]}
    for _index, key in composite:
        theta_z, theta_zn = table.theta({(key.irrep, key.letter): 1})
        assert theta_zn == {}, "SG 84 has no Z_n row"
        got = tqr.atomic_realizations(table, theta_z)
        assert got and got.atomic_possible, str(got)
        assert got.n_bands == 4
        assert [realization.label for realization in got.realizations] == \
            realizations[key.letter], key


def test_atomic_realizations_refuses_a_fractional_theta_z(sg211):
    """A stable band set has no atomic realization at all -- a refusal, not a list.

    By the paper's theorem a fractional Z-SRSI *is* a nonzero symmetry
    indicator, so no atomic insulator carries that symmetry data.  That is a
    stronger statement than an empty list of realizations, so it is a refusal with its
    own gate and not an ``INFEASIBLE`` record.
    """
    out = tqr.atomic_realizations(sg211, [Fraction(1, 2), 0, 0, 0, 0, 0, 0])
    assert not out and out.gate == "srsi_not_atomic", str(out)
    assert "nonzero symmetry indicator" in out.reason
    # (`context["denominators"]` beside it is the argument read back, so it is
    # not asserted here; the contrast that makes the gate a refusal is that the
    # same vector with an integral first entry comes back served)
    assert tqr.atomic_realizations(sg211, [1, 0, 0, 0, 0, 0, 0]).atomic_possible


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
    record comes back with ``bound == 0`` and its own note.
    """
    def column(index):
        vector = [0] * len(sg211.ebrs)
        vector[index] = 1
        return sg211.theta_columns(vector)[0]

    fragile = tuple(2 * a - b for a, b in zip(column(16), column(0)))
    got = tqr.atomic_realizations(sg211, fragile)
    assert got, str(got)                       # the record is truthy
    assert not got.atomic_possible
    assert got.realizations == ()                # `atomic_possible` is `bool(realizations)`
    assert got.n_bands == 11                   # 2 x 6 - 1, read off theta_Z
    assert got.bound == 11
    assert "fragile" in got.note and "no atomic realization" in got.summary()

    # ... and the one that never gets as far as the search: -(A1)_2a's theta_Z
    # implies -1 bands through the band-count form (1, 1, 2, 3, 3, 3, 4)
    empty = tqr.atomic_realizations(sg211, tuple(-v for v in column(0)))
    assert empty, str(empty)                   # served, not refused
    assert not empty.atomic_possible and empty.realizations == ()
    assert empty.n_bands == 0 and empty.bound == 0
    assert "is -1, which is not a nonnegative integer" in empty.note
    assert "fragile" not in empty.note


def test_atomic_realizations_refuses_a_truncated_enumeration(sg211):
    """A no-go is only ever asserted on a **complete** enumeration.

    At ``cap=1`` the search stops after one solution, and neither "no pair adds
    up" nor "none exists" may be claimed from that -- so the cap is a refusal
    and not a short list.  The same input at the default cap serves four.
    """
    vector = [0] * len(sg211.ebrs)
    vector[16] = 1
    theta_z = sg211.theta_columns(vector)[0]
    out = tqr.atomic_realizations(sg211, theta_z, cap=1)
    assert not out and out.gate == "srsi_realizations_truncated", str(out)
    # the band-count form was evaluated before the cap fired (`context["cap"]`
    # beside it is the keyword read back, so it is not asserted)
    assert out.context["n_bands"] == 6
    assert len(tqr.atomic_realizations(sg211, theta_z, cap=256).realizations) == 4


def test_the_grammar_mistakes_are_value_errors_not_refusals(sg211):
    """Four caller bugs, each matched on the wording of **its own** gate.

    A wrong-length ``theta_Z``; a band count no band set has; a splitting whose
    two sides do not add up to the total; and a ``theta_Zn`` at moduli the group
    does not have.  The last two are ``nsit_criterion``'s, and the additivity
    gate is the one branch ``splitting`` can never reach: it builds
    ``theta_Z[C]`` as ``total - V``, so it cannot hand in three vectors that do
    not add up.

    Two of the four messages end in the same phrase ("an arithmetic mistake"),
    so matching on that would not tell them apart: each pattern here is a piece
    of wording only one of the gates has.
    """
    with pytest.raises(ValueError, match="Z-valued SRSIs and theta_Z carries"):
        tqr.atomic_realizations(sg211, [0, 0, 0])
    vector = [0] * len(sg211.ebrs)
    vector[16] = 1
    theta_total = sg211.theta_columns(vector)
    theta_z = theta_total[0]
    with pytest.raises(ValueError, match="theta_Z implies 6 bands"):
        tqr.atomic_realizations(sg211, theta_z, n_bands=7)
    with pytest.raises(ValueError, match="not those of all the bands"):
        tqr.nsit_criterion(sg211, theta_total, theta_z, theta_z)
    zeros = tuple(0 for _ in theta_z)
    with pytest.raises(ValueError, match=r"moduli \[2\] and theta_Zn carries \[3\]"):
        tqr.nsit_criterion(sg211, (theta_z, {3: (0, 0, 0)}), theta_z, zeros)


# ---------------------------------------------------------------------------
# 3.  the site irreps the SRSIs do not distinguish
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_the_same_content_sweep_over_every_tabulated_system():
    """**36 pairs**: same-site site irreps with one symmetry vector and one set of SRSIs.

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
            for pair in tqr.stably_equivalent_pairs(table):
                assert len(pair.columns) == 2, (sg, spinor, pair)
                assert not spinor, (sg, pair)
                first = [1 if j == pair.columns[0] else 0
                         for j in range(len(table.ebrs))]
                second = [1 if j == pair.columns[1] else 0
                          for j in range(len(table.ebrs))]
                assert table.theta_columns(first) == table.theta_columns(second), (sg, pair)
                found.append((sg, pair.letter, pair.irreps))
    assert tuple(found) == SAME_CONTENT_PAIRS, found


def test_stably_equivalent_pairs_takes_a_measured_group_too(lin3g_declared_cross_checked):
    """A ``GroupData`` works as well as a table, and ``P4mm1'`` has no such pair."""
    _model, group, _assignment, _validated, _rep = lin3g_declared_cross_checked
    assert tqr.stably_equivalent_pairs(group) == ()
    assert tqr.stably_equivalent_pairs(tqr.srsi_tables(group)) == ()


# ---------------------------------------------------------------------------
# 4.  the split-EBR regression
# ---------------------------------------------------------------------------

def _split_thetas(table, split):
    """``(theta_total, theta_Z[V], theta_Z[C])`` for one printed splitting."""
    def column(entry):
        vector = [0] * len(table.ebrs)
        vector[int(entry["column"])] = 1
        return vector

    theta_total = table.theta_columns(column(split.parent))
    theta_v = table.theta_columns(column(split.children[0]))[0]
    theta_c = tuple(a - b for a, b in zip(theta_total[0], theta_v))
    return theta_total, theta_v, theta_c


@pytest.mark.data
def test_every_split_ebr_flag_is_reproduced_from_the_theta_arithmetic():
    """The supplement's 211 splittings, and its own ``xmark`` re-derived.

    ``Lambda = V (+) C`` with every constituent an EBR: 9 without SOC and 202
    with, of which the captions flag 9 and 194 as SRSI-mismatching.  Two
    statements, both exact:

    * ``theta_Z(parent) = sum theta_Z(children)`` on **all 211** -- the Z-SRSIs
      are linear in ``p`` and the symmetry vectors add, so this is what a
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
                    vector = [0] * len(table.ebrs)
                    vector[int(entry["column"])] = 1
                    return vector

                parent = table.theta_unreduced_columns(column(split.parent))
                children = [Fraction(0)] * len(parent)
                for child in split.children:
                    for i, value in enumerate(table.theta_unreduced_columns(column(child))):
                        children[i] += value
                if [parent[i] for i in table.z_rows] == \
                        [children[i] for i in table.z_rows]:
                    additive += 1
                mismatch = any((int(parent[i]) - int(children[i])) % n
                               for n, rows in table.zn_rows.items() for i in rows)
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
    assert table and table.zn_rows == {2: (3, 4, 5, 6, 7)}
    splits = [s for s in table.split_ebrs if s.parent["irrep"] == "-1F2F"]
    assert len(splits) == 3
    assert [s.srsi_mismatch for s in splits] == [False, True, False]

    flagged = splits[1]
    assert tuple(c["irrep"] + "@" + c["letter"] for c in flagged.children) == \
        ("-F@a", "-F@b")
    theta_total, theta_v, theta_c = _split_thetas(table, flagged)
    assert theta_total[1] == {2: (0, 0, 0, 0, 0)}
    got = tqr.nsit_criterion(table, theta_total, theta_v, theta_c)
    assert got and got.verdict == "invisible to the SRSIs"
    assert not got.nsit
    assert got.v_atomic_possible and got.c_atomic_possible
    assert [c.label for c in got.realizations_v.realizations] == \
        ["1 (-F)_b", "1 (-F)_a"]
    assert [c.theta_Zn[2] for c in got.realizations_v.realizations] == \
        [(0, 0, 1, 0, 0), (0, 0, 0, 0, 0)]
    assert sorted(got.consistent_pairs) == [(0, 0), (1, 1)]
    assert "invisible to the SRSIs" in got.summary()
    assert "invisible to the stable real-space invariants" in got.reason
    # ... and the printed pair is the one that does not add up
    printed_v = table.theta_columns(
        [1 if j == flagged.children[0]["column"] else 0
         for j in range(len(table.ebrs))])[1]
    printed_c = table.theta_columns(
        [1 if j == flagged.children[1]["column"] else 0
         for j in range(len(table.ebrs))])[1]
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
    no additivity to test: the verdict is ``"no Z_n SRSIs"``, said out loud, and
    not a vacuous ``nsit=False``.

    The strong phase is the contrast: its ``theta_Z[V]`` is fractional, which by
    the paper's theorem is a nonzero symmetry indicator, so no atomic insulator
    has the valence pair's symmetry data and there is nothing for an
    enumeration of realizations to do.  The record is ``stable topological``
    with no realizations on either side, and its summary says so.

    The two verdicts are two of the four a splitting has, and there is no
    fifth: the others are ``diagnosed topological`` and ``invisible to the
    SRSIs`` -- a consistent pair hides any topology from the SRSIs and never
    makes a splitting trivial, the criterion being sufficient, never necessary.
    """
    got = tqr.splitting(fkm_phases["trivial"], bands=2)
    assert got, str(got)
    assert got.verdict == "no Z_n SRSIs" and not got.nsit
    # the *group*, not the splitting, is what has no Z_n row -- read off the
    # table rather than off the verdict that was selected by it
    assert tqr.srsi_tables(fkm_phases["trivial"].group).zn_rows == {}
    assert got.v_atomic_possible and got.c_atomic_possible
    assert [c.label for c in got.realizations_v.realizations] == ["1 (-E1u)_a"]
    assert [c.label for c in got.realizations_c.realizations] == ["1 (-E1g)_a"]
    assert got.indicator.values == (0, 0) and got.indicator.order == 1
    assert "no Z_n-valued SRSI" in got.reason
    assert "no Z_n SRSIs: the group has no Z_n-valued" in got.summary()
    assert got.summary().count("no Z_n SRSIs") == 1, "the spelled-out verdict, once"

    stable = tqr.splitting(fkm_phases["strong"], bands=2)
    assert stable.verdict == "stable topological" and not stable.nsit
    assert stable.realizations_v is None and stable.realizations_c is None
    assert "stable topological: the bands asked for carry a nonzero symmetry " \
        "indicator" in stable.summary()

    # ... and the four verdicts a splitting can have, these two among them
    assert tqr.SPLITTING_VERDICTS == ("diagnosed topological", "invisible to the SRSIs",
                                      "no Z_n SRSIs", "stable topological")


def test_lin3g_p2_has_one_atomic_realization_and_the_enumeration_is_complete(
        lin3g_declared_cross_checked):
    """The enumeration inside lin3g's diagnosed splitting, on its fragile ``P2``.

    Split above its sixth band, lin3g is ``diagnosed topological``: no pair of
    realizations reproduces the total's ``theta_Zn = {2: (0, 1)}``.  Its valence
    set contains the suite's fragile golden ``P2``, whose only atomic
    realization is ``(-E2)_1b``, and this is the enumeration that finds it, on
    ``P2``'s own ``theta_Z``: the basis of maximal keys it is written over with
    each key's band dimension, the band count read off ``theta_Z`` through the
    table's own form rather than trusted, and the per-coordinate bound --
    rigorous, since ``sum_j dim_j p_j = n_bands`` with every ``dim_j >= 1``
    bounds every ``p_j``, so the enumeration is complete.
    """
    tables = tqr.srsi_tables(lin3g_declared_cross_checked.group)
    cand = tqr.atomic_realizations(tables, (Fraction(0), Fraction(0), Fraction(1)),
                                   n_bands=2)
    assert cand and cand.atomic_possible, str(cand)
    assert [key.label for key in cand.keys] == [
        "(-E1)_a", "(-E2)_a", "(-E1)_b", "(-E2)_b", "(-E)_c"]
    assert [key.dim for key in cand.keys] == [2, 2, 2, 2, 4]
    assert tuple(str(v) for v in tables.band_count_form) == ("2", "2", "2")
    assert cand.n_bands == 2 and cand.bound == 1
    assert [c.label for c in cand.realizations] == ["1 (-E2)_b"]


def test_kane_mele_odd_is_diagnosed_by_its_conduction_pair_and_even_is_not(
        km_realized):
    """``#156.50`` has no Z_n row either -- and one regime is still diagnosed.

    Kane-Mele's **odd** regime: the valence pair's symmetry data is that of
    ``(-E1)_1a`` (one realization), and the conduction pair's
    ``theta_Z = (0, 1, -1, -1)`` has **no** nonnegative realization at two bands
    at all.  A side that is not an atomic insulator is a side that is not an
    atomic insulator, whatever the moduli, so the verdict is
    ``diagnosed topological`` -- by the *realization* half of the criterion, and
    the record's ``reason`` says exactly that rather than claiming a Z_n
    mismatch the group cannot have.

    The **even** regime splits into ``(-E1)_1b`` and ``(-E1)_1c``, one
    realization each, and ``P3m11'`` has no Z_n row to test them with:
    ``no Z_n rows``.
    """
    odd = tqr.splitting(km_realized["odd"], bands=2)
    assert odd, str(odd)
    assert odd.verdict == "diagnosed topological" and odd.nsit
    assert odd.moduli == ()
    assert odd.v_atomic_possible and not odd.c_atomic_possible
    assert odd.realizations_c.realizations == ()
    assert tuple(str(v) for v in odd.theta_Z_c) == ("0", "1", "-1", "-1")
    assert [c.label for c in odd.realizations_v.realizations] == ["1 (-E1)_a"]
    assert "the conduction bands have" in odd.reason and "realization half" in odd.reason
    assert "fragile side" in odd.reason

    even = tqr.splitting(km_realized["even"], bands=2)
    assert even, str(even)
    assert even.verdict == "no Z_n SRSIs" and not even.nsit
    assert even.v_atomic_possible and even.c_atomic_possible
    assert [c.label for c in even.realizations_v.realizations] == ["1 (-E1)_b"]
    assert [c.label for c in even.realizations_c.realizations] == ["1 (-E1)_c"]


def test_the_band_set_is_one_keyword_on_both_routes(lin3g_declared_cross_checked):
    """``bands=`` names the block for ``splitting`` and for the band route alike.

    A list of bands is the same request as their count, the band set is
    required, and beside a table measured on its own block a ``bands=`` naming a
    different set is a contradiction rather than an override.
    """
    record = lin3g_declared_cross_checked
    want = tqr.splitting(record, bands=4)
    assert want, str(want)
    by_list = tqr.splitting(record, bands=[0, 1, 2, 3])
    assert by_list, str(by_list)
    assert by_list.verdict == want.verdict
    assert by_list.theta_Z_v == want.theta_Z_v
    assert by_list.theta_Z_c == want.theta_Z_c
    assert by_list.n_bands_v == want.n_bands_v == 4
    assert by_list.n_bands_c == want.n_bands_c
    assert by_list.summary() == want.summary(), "the same split, said the same way"

    with pytest.raises(TypeError, match="missing required argument"):
        tqr.splitting(record)

    _model, group, _assignment, _validated, rep = record
    table = tqw.hsp_table(rep, bands=4)
    by_list = tqr.srsi_of_bands(table, group, rep=rep, bands=[0, 1, 2, 3])
    assert by_list.n_bands == 4
    assert by_list.theta_Z == tqr.srsi_of_bands(table, group, rep=rep).theta_Z, \
        "with no bands= the table's own block is what is read"
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tqr.srsi_of_bands(table, group, rep=rep, bands=2)
