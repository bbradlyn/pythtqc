r"""The operations by index, and the Seitz symbol each one is written as.

``operations`` writes one row per operation: the index every per-operation
record is keyed on, the Seitz symbol ``{R|t}`` of the tables' conventional
setting, the Cartesian name a ``characters=`` declaration is written in, and
the rotation and translation in the model's own reduced coordinates.  ``seitz``
gives one symbol by index and ``operation_index`` reads a symbol back.

What is checked here, and how each check can fail
-------------------------------------------------
**The symbol names the operation.**  Parsed back, a row's symbol carries
``irrep``'s own ``rotation_refUC`` and ``translation_refUC`` for that
operation, and ``operation_index`` on it gives the index it came from.  The
second half is what makes the naming a *bijection*: two operations written the
same way would send the lookup to the first of them and the test would fail on
the second.

**The sense is the turn the crystal makes.**  For every rotation of order above
two the symbol carries a ``+`` or a ``-``, and the check builds the Cartesian
rotation by ``±2π/n`` about the axis the symbol names — Rodrigues, from the
direction and the angle alone — and compares it with the operation's own
``rotation_cart``.  A sense read the wrong way round gives the transpose and
the comparison fails.

**The translation the tables carry is the shortest representative of itself.**
A Seitz symbol's translation is fixed only modulo the crystal's lattice, and on
a centred lattice the conventional cell holds more than one representative of
it: ``Fd-3m``'s twofold is ``1/4,1/4,0`` and ``3/4,3/4,0`` at once.  What is
written is the group's own tabulated translation, so that a symbol read here and
a general position read off the tables are the same string; what a *written*
symbol is matched against is the shortest representative, computed from
``refUC`` and ``shiftUC`` by a route the printout does not use.  The two are
checked against each other, operation by operation: one vector modulo the
crystal's lattice always, and equal outright outside the three groups of
:data:`NOT_SHORTEST`.

**A non-symmorphic group's general positions, as literal strings.**  ``Pnma``'s
eight operations are written out in full, which pins the whole spelling at once
— the symbols, the directions, the fractions, the separators — against the
general-position list of ITA's own entry for the group.

**The three ways of naming a group's operations.**  What ``realized_group``
measured, its group and its representation all read as one table.

**The verdicts.**  A symbol no operation is written as is refused
``unknown_operation`` with the nearest spellings the group does carry; a group
whose reference setting was never determined is refused ``tables_not_ready``;
an index the group has no operation for raises.

The census
----------
:data:`CENSUS` is fourteen builds over thirteen space groups, 364 operations,
chosen so that every part of the spelling is reached: the ``P`` centring (ten
builds), ``I`` (two), ``F`` and ``R``; symmorphic groups and groups whose
operations carry screws and glides; five two-dimensional slab cells and nine
fully periodic ones; eleven spinful models and three spinless; and all sixteen
rotation parts the symbols have -- ``1``, ``-1``, ``m``, ``2``, ``3±``, ``4±``,
``6±``, ``-3±``, ``-4±`` and ``-6±``.  Eleven rows are realized records, the
builds other modules use, through the session cache of
``conftest.realized_once``, so a whole-suite run pays for most of them once.
The other three -- undistorted Fu-Kane-Mele, the free-standing Kane-Mele
layer and ``p4mmm_s_2g`` -- are Hamiltonians that obey their cell's whole
group, so the group their cell allows *is* the group served, and they are read
off the cell alone: a symbol names an operation of a group, and nothing in the
naming asks for the Hamiltonian.  The ``A``, ``B`` and ``C`` centrings and the
monoclinic and triclinic systems are not reached: no fixture realizes a group
with one.
"""

from dataclasses import replace

import numpy as np
import pytest

from pythtqc import groups as tqg
from pythtqc.refusal import Refusal
from pythtqc import _seitz
from pythtqc import _spacegroup

import tqc_models as tqm                                            # noqa: F401
from conftest import FKM_BOND, fkm_regime, km_z2_regime

#: One build per row: the zero-argument builder, the keyword arguments, and the
#: group it realizes.  The number is asserted, so a build that drifts to another
#: group is a failure here rather than a silently different census.  A row whose
#: keyword arguments are ``None`` is read off its cell (``_group_of``): its
#: Hamiltonian obeys every operation the cell allows.
CENSUS = {
    # -- groups read off the cell alone --
    "fkm_diamond": (fkm_regime("uniform"), None, "227.129"),
    # -- the shared builds, already in the session cache in a whole-suite run --
    "fkm_strong": (fkm_regime("strong"),
                   dict(decorations=FKM_BOND, characters={0: "s"}), "166.98"),
    "lin3g": (tqm.lin3g, dict(typat=[0, 0, 1, 1],
                              characters=["s", "dx2-y2", "s", "s"]), "99.164"),
    "km_odd": (km_z2_regime("odd"),
               dict(typat=[0, 1], cross_check=False, characters={0: "A1"}), "156.50"),
    "pwv_fragile": (tqm.pwv_fragile,
                    dict(cross_check=False, characters={0: "A1"}), "183.186"),
    "km_inversion": (tqm.kane_mele_inversion,
                     dict(environment="substrate", cross_check=False,
                          characters={0: "A1"}), "183.186"),
    # -- the free-standing layer group, which is where `-6` and `-3` are --
    "km_inversion_free": (tqm.kane_mele_inversion, None, "191.234"),
    # -- screws and glides --
    "pnma": (tqm.pnma_s_4a, {}, "62.442"),
    "p4122": (tqm.p4122_atomic, {}, "91.104"),
    "p3121": (tqm.p3121_atomic, {}, "152.34"),
    "p62": (tqm.p62_atomic, {}, "171.122"),
    "i213": (tqm.i213_flicker2018, {}, "199.13"),
    "i41_soc": (tqm.i41_soc, {}, "80.30"),
    # -- the rotoinversions of a tetragonal layer --
    "p4mmm": (tqm.p4mmm_s_2g, None, "123.340"),
}


def record(realized, key):
    """One realized census row, from the session cache."""
    build, kw, number_str = CENSUS[key]
    assert kw is not None, f"{key} is read off its cell"
    out = realized(build, **kw)
    assert out, f"{key}: {out}"
    assert out.group.number_str == number_str, f"{key}: {out.group.number_str}"
    return out


def census_group(realized, key):
    """``(group, rep)`` of one census row: a realized row's, or a cell row's and ``None``."""
    build, kw, number_str = CENSUS[key]
    if kw is not None:
        out = record(realized, key)
        return out.group, out.rep
    group = _spacegroup._group_of(build())
    assert group, f"{key}: {group}"
    assert group.number_str == number_str, f"{key}: {group.number_str}"
    return group, None


#: The groups where ``irreptables`` keeps a translation that is not the shortest
#: representative of its operation: the ``c``-glide F-centred cubic groups,
#: where ``1/2,1/2,1/2`` stands where ``0,0,1/2`` would be shorter.  Measured
#: over all 230 space groups in both valuednesses: 108 operations of these three
#: and none of the other 227.
NOT_SHORTEST = (219, 226, 228)


# ---------------------------------------------------------------------------
# 1. the symbol names the operation, says how it turns, and carries the
#    tables' own translation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(CENSUS))
def test_every_operation_resolves_to_a_symbol_that_names_it_back(realized, key):
    """Every operation has a symbol, and the symbol has that ``ind``.

    On a realized row the operations are the ``ind`` of ``rep.matrices``, on a
    cell row the group's own.  Parsed back, the symbol carries exactly the
    rotation and the translation ``irrep`` computes for that operation in the
    conventional setting (``rotation_refUC``, ``translation_refUC``), and the
    reverse lookup returns the index it was written from — which it could not
    do if two operations were written the same way.

    **The sense.**  ``n+`` turns by ``+2π/n`` about the axis the symbol names,
    ``n-`` by ``−2π/n``.  The comparison is against the operation's own
    Cartesian rotation, built here from the direction and the angle alone, so a
    sense read the wrong way round gives the transpose and fails.

    **The translation.**  What is written is the translation the group's own
    tables carry, so that a symbol here and a printed general position are the
    same string.  What is *matched* against is the shortest representative,
    computed from ``refUC`` and ``shiftUC`` and reduced modulo the crystal's
    lattice — a different route to the same operation, which the printout does
    not use.  The two must be one vector modulo that lattice, always; and
    outside #219, #226 and #228 they must be equal outright, the tables keeping
    the shortest representative everywhere else.  A reduction that lost a
    centring translation, or an origin shift the two routes disagreed about,
    fails here.
    """
    group, rep = census_group(realized, key)
    named = group if rep is None else rep
    table = tqg.operations(named)
    assert not isinstance(table, Refusal), str(table)
    listed = (sorted(int(s.ind) for s in group.symmetries) if rep is None
              else sorted(int(i) for i in rep.matrices))
    assert [row.ind for row in table.values()] == listed

    centering = _seitz._centering_of(group)
    by_ind = {int(s.ind): s for s in group.symmetries}
    for row in table.values():
        sym = by_ind[row.ind]
        parsed = _seitz._parse_seitz(row.seitz, centering)
        assert parsed is not None, f"{key} #{row.ind}: {row.seitz!r} does not parse"
        wanted, wanted_tr = parsed
        # `_seitz_key` is `rotation_refUC` and `translation_refUC` put in the
        # quantities the symbol is written from; `_matches` compares the two.
        assert _seitz._matches(_seitz._seitz_key(sym, group), bool(sym.time_reversal),
                            wanted, wanted_tr), f"{key} #{row.ind}: {row.seitz}"
        assert tqg.operation_index(named, row.seitz) == row.ind, row.seitz
        assert tqg.seitz(named, row.ind) == row.seitz

    # the sense is the turn the operation makes
    lattice = np.asarray(group.lattice, dtype=float)
    refUC = np.asarray(group._refUC, dtype=float)
    seen = set()
    for sym in group.symmetries:
        det, order, axis, sense, _ = _seitz._seitz_key(sym, group)
        if order <= 2:
            continue
        seen.add(("" if det > 0 else "-") + str(order))
        n = (refUC @ np.asarray(axis, dtype=float)) @ lattice
        n = n / np.linalg.norm(n)
        cross = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
        angle = sense * 2.0 * np.pi / order
        turn = (np.eye(3) + np.sin(angle) * cross
                + (1.0 - np.cos(angle)) * (cross @ cross))
        assert np.abs(det * np.asarray(sym.rotation_cart, dtype=float)
                      - turn).max() < 1e-6, f"{key} #{int(sym.ind)}"
    if key in ("fkm_diamond", "km_inversion_free", "p4mmm"):
        assert seen, f"{key} carries no rotation of order above two"

    # the translation is the one the tables carry
    assert group.sg_number not in NOT_SHORTEST, (
        f"{key} reaches #{group.sg_number}: the outright half below is no longer "
        "the claim for it")
    tabulated = _seitz._tabulated_translations(group)
    for sym in group.symmetries:
        rotation = np.asarray(sym.rotation_refUC(refUC), dtype=int)
        # the fall-back would make this test vacuous, so it is ruled out first
        assert (rotation.tobytes(), bool(sym.time_reversal)) in tabulated, (
            f"{key} #{int(sym.ind)}: the tables carry no entry for this rotation")
        shortest = _seitz._seitz_key(sym, group)[4]
        written = _seitz._written_translation(sym, group, tabulated, shortest)
        difference = np.array([float(a) - float(b) for a, b in zip(written, shortest)])
        assert _spacegroup._is_translation(difference, centering), (
            f"{key} #{int(sym.ind)}: the tables carry {[str(p) for p in written]} "
            f"and the shortest representative is {[str(p) for p in shortest]}, "
            "which differ by no lattice translation")
        assert written == shortest, (
            f"{key} #{int(sym.ind)}: the tables carry {[str(p) for p in written]}, "
            f"not the shortest representative {[str(p) for p in shortest]}")


# ---------------------------------------------------------------------------
# 2. a non-symmorphic group's general positions, as literal strings
# ---------------------------------------------------------------------------

#: ``Pnma``'s eight operations as ITA's entry for the group writes them: the
#: identity and the inversion, three twofold screws and three glides, with the
#: translations of its general-position list.
PNMA_SEITZ = ("{1|0,0,0}", "{2_001|1/2,0,1/2}", "{2_010|0,1/2,0}",
              "{2_100|1/2,1/2,1/2}", "{-1|0,0,0}", "{m_001|1/2,0,1/2}",
              "{m_010|0,1/2,0}", "{m_100|1/2,1/2,1/2}")


def test_the_general_positions_of_pnma_are_written_as_the_tables_write_them(realized):
    """``Pnma``'s eight unitary operations, spelled out.

    The whole spelling is pinned here at once — which symbol names which kind
    of operation, the direction of the axis (of a reflection's **normal**), the
    fractions and the separators — against a group whose lattice is primitive,
    so that the translations carry no representative freedom at all.
    """
    group = record(realized, "pnma").group
    unitary = [row for row in tqg.operations(group).values() if not row.time_reversal]
    assert tuple(row.seitz for row in unitary) == PNMA_SEITZ
    # the grey partners are the same eight with a prime
    antiunitary = [row for row in tqg.operations(group).values() if row.time_reversal]
    assert tuple(row.seitz for row in antiunitary) == tuple(s + "'" for s in PNMA_SEITZ)


# ---------------------------------------------------------------------------
# 3. a symbol is read as the operation it names, not as a string
# ---------------------------------------------------------------------------

def test_a_symbol_is_read_as_the_operation_it_names_and_not_as_a_string(realized):
    """The two spellings of an axis, and any representative of a translation.

    ``n+`` about ``[uvw]`` and ``n-`` about ``[-u-v-w]`` turn the same way, and
    the printed tables choose between them per group; on a centred lattice the
    conventional cell holds several translations of one operation.  A symbol
    written either way, and with any of those translations, is the same
    operation and resolves to it.  Asked of diamond's ``Fd-3m1'``, read off the
    cell.
    """
    group, _ = census_group(realized, "fkm_diamond")
    threefold = tqg.operation_index(group, "{3+_111|0,0,0}")
    assert threefold == tqg.operation_index(group, "{3-_-1-1-1|0,0,0}")
    assert tqg.seitz(group, threefold) == "{3+_111|0,0,0}"

    twofold = tqg.operation_index(group, "{2_001|1/4,1/4,0}")
    # the F centring: the same operation, the other three representatives
    for other in ("{2_001|3/4,3/4,0}", "{2_001|1/4,3/4,1/2}", "{2_001|3/4,1/4,1/2}"):
        assert tqg.operation_index(group, other) == twofold, other
    # and the spacing, the underscore and the braces' padding are free
    for other in ("{ 2_001 | 1/4,1/4,0 }", "{2 001|1/4,1/4,0}", "{2001|1/4,1/4,0}"):
        assert tqg.operation_index(group, other) == twofold, other


# ---------------------------------------------------------------------------
# 4. the verdicts
# ---------------------------------------------------------------------------

def test_a_symbol_no_operation_is_written_as_is_refused(realized):
    """``unknown_operation``, naming the nearest spellings the group carries.

    ``P3m1`` has no fourfold axis, so a fourfold symbol names no operation of
    it; what a reader is handed back is the threefold that is nearest to what
    was written, not a bare failure.
    """
    group = record(realized, "km_odd").group
    got = tqg.operation_index(group, "{4+_001|0,0,0}")
    assert isinstance(got, Refusal) and got.gate == "unknown_operation"
    assert got.context["nearest"], str(got)
    assert all(name in {row.seitz for row in tqg.operations(group).values()}
               for name in got.context["nearest"])
    assert "{3+_001|0,0,0}" in got.context["nearest"]
    # text that is no symbol at all is the same verdict, not an exception
    assert tqg.operation_index(group, "C4(z)").gate == "unknown_operation"


def test_a_group_whose_reference_setting_is_unknown_is_refused(realized):
    """``tables_not_ready``: with no ``refUC`` there is no setting to write in."""
    group = record(realized, "km_odd").group
    without = replace(group, _tables_ready=False, _setup_note="no reference cell")
    for got in (tqg.operations(without), tqg.seitz(without, 1),
                tqg.operation_index(without, "{1|0,0,0}")):
        assert isinstance(got, Refusal) and got.gate == "tables_not_ready", str(got)


def test_an_index_the_group_has_no_operation_for_raises(realized):
    """An index out of the group is a mistake about the group, not a verdict on it."""
    out = record(realized, "km_odd")
    with pytest.raises(KeyError, match="no operation #99"):
        tqg.seitz(out.group, 99)
    with pytest.raises(TypeError, match="r.group"):
        tqg.operations(out.model)


def test_the_measured_record_is_read_as_its_group(realized):
    """``r``, ``r.group`` and ``r.rep`` are three ways of naming the same operations.

    A reader holding what :func:`realized_group` measured reaches for it first,
    and on a record every operation of the group is represented, so the three
    carry one table between them.
    """
    out = record(realized, "km_odd")
    assert len(out.rep.matrices) == out.group.size, "not every operation is represented"
    rows = tqg.operations(out)
    assert ([row.seitz for row in rows.values()]
            == [row.seitz for row in tqg.operations(out.group).values()]
            == [row.seitz for row in tqg.operations(out.rep).values()])
    assert tqg.seitz(out, 5) == tqg.seitz(out.group, 5) == "{m_100|0,0,0}"
    assert tqg.operation_index(out, "{m_100|0,0,0}") == 5
    # `r.operations` and `rep.operations` are `tqc.operations` read on the record
    assert out.operations == rows == out.rep.operations
