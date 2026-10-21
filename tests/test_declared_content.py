r"""P1b: **the declared content** — what the caller says the orbitals are.

The hoppings cannot measure what an orbital *is*.  They measure which operations
are symmetries once the orbitals are stated, so a declaration the hoppings are
not covariant under is not a wrong declaration: it is a crystal with less
symmetry, and the group **descends** exactly as it does for a hopping that
breaks an operation.  ``character_mismatch`` is gone with no shim.

Three things make that work, and this file pins all three on measured numbers.

**A declaration is read as content.**  ``characters=`` resolves once, in the site
tables of the group the loop is on, into one
:class:`pythtqc._sitechars.DeclaredContent` per declared multiplet — the
constraint ``{ind: chi(g)}``, the label, the dimension and the group it was read
in — and every re-identification below **re-keys** those ``.ind`` through the
step's own pairing rather than resolving the label again.  Operations the smaller
group no longer has drop out of the constraint, and that dropping *is* the
restriction of the declared irrep to the smaller site group: ``A2`` of ``3m``
served as ``A`` of ``3``, and a restriction that has become reducible spelled as
the sum of its irreps (``B2 + B1``) rather than left empty.

**A declaration is a site's content.**  It is read on the multiplets the
Hamiltonian measures, which it never changes: words naming exactly those
multiplets one multiplet at a time, and any other spelling as the one content
the site's words add up to, part of what the site holds where some of its
orbitals are left out.  So ``{0: "s", 1: "dx2-y2"}`` on a square lattice's
``p_x, p_y`` pair that nothing couples says the pair is ``A1 + B1``, which the
hoppings carry as faithfully as the ``p`` doublet, and the fourfold stays.

**The verdict is per operation, on the detector's own solution spaces.**  Does
the space hold a nonsingular member block-diagonal on the declared channels, and
do that member's per-channel scalars agree with the declared characters up to one
scalar ``λ(g)``?  What survives is the largest subgroup on which a real character
of *that subgroup* meets every requirement — the first of the maximal ones by the
tie rule where several tie — and the identity alone always does, which is why a
descent by declaration always lands somewhere and never refuses.  Its floor is
``P1``.

One refusal remains, and it is honest.  ``_build_rep`` on a **fixed** group says
``declaration_breaks_group`` and names the subgroup ``realized_group`` descends
to.  Where no antiunitary operation is represented, a **complex** character of
the whole group can meet every requirement while no real one does: nothing in
such a group's measurement forbids one, so P1c **serves** the content with that
twist rather than lowering a group the Hamiltonian has.

Section 6 is P1c's own: a measurement is read **once**.  The verdict stage
chooses as well as judging — the member of each solution space the
representation is assembled from, its free blocks settled by the declaration
where there is one and by the least twist where there is not, on every site —
and the build assembles and reports from those choices without reading a
solution space again.

Section 8 is **which orbital a declaration speaks for**.  The hoppings fix
every multiplet's block up to one one-dimensional character per part of the
crystal they couple, so a declaration anywhere constrains its part's
character, the reference orbital's no more than any other: declaring one
sublattice of an orbit is the same statement as declaring the other, and
declarations of one part no character meets together are a smaller group.
Everything undeclared is assigned by the lowest-``l`` rule, and each row's
``determined_by`` says whether the declarations left its site irrep a choice.
"""

import numpy as np
import pytest
from pythtb import Lattice, TBModel
from pythtb.models import haldane, kane_mele

from pythtqc import _sitechars
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import spin as tqsp
from pythtqc import symmetry as tqs
from pythtqc.models import cubic_3c_perp_doublet
from pythtqc.refusal import Refusal
from pythtqc import _build
from pythtqc import _declaration
from pythtqc import _measure
from pythtqc import _multiplets
from pythtqc import _spacegroup
from pythtqc import _tables

import tqc_models as tqm                                            # noqa: F401

#: Kane-Mele at ``delta = 0`` with the two sublattices declared different
#: species: ``P3m11' #156.50``, 12 of 12, and the site group ``3m`` the
#: ``A1``/``A2`` declarations below are written in.
KM_ARGS = (0.0, 1.0, 0.3, 0.25)


def _km():
    return kane_mele(*KM_ARGS)


def _square_pxpy(angle=0.0):
    """A ``p_x, p_y`` pair at one ``4mm`` site of a square lattice, spinless.

    Slater-Koster nearest neighbours, ``t_sigma = 1.0`` along the bond and
    ``t_pi = 0.3`` across it.  Nothing couples ``p_x`` to ``p_y``, so the
    commutant of ``H(k)`` is a torus and the layout measures one
    two-dimensional channel: the same model P1a's tests read ``E`` at ``1a``.
    ``angle`` writes the pair in a basis turned by it inside the site: at
    ``π/4`` the orbitals are ``(p_x ± p_y)/√2``.
    """
    c, s = np.cos(angle), np.sin(angle)
    turn = np.array([[c, -s], [s, c]])
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]] * 2,
                        periodic_dirs=[0, 1]), spinful=False)
    for bond, along in (([1, 0], np.diag([1.0, 0.3])), ([0, 1], np.diag([0.3, 1.0]))):
        hop = turn @ along @ turn.T
        for a in range(2):
            for b in range(2):
                if abs(hop[a, b]) > 1e-15:
                    m.set_hop(float(hop[a, b]), a, b, bond, mode="add")
    return m


def _tri_pxpy(t_sigma=1.0, t_pi=0.3, orbital_field=0.0):
    r"""A ``p_x, p_y`` pair at the one ``6mm`` site of a triangular lattice, spinless.

    Slater-Koster nearest neighbours over the three bond directions of the
    lattice ``a1 = (1, 0)``, ``a2 = (1/2, sqrt3/2)`` — ``t_ab = t_sigma n_a n_b
    + t_pi (delta_ab - n_a n_b)`` with ``n`` the unit bond vector, the same form
    the ``p_x, p_y`` honeycomb is built from.  The threefold axis mixes the two
    orbitals, so the layout measures one two-dimensional channel and the site
    has no one-dimensional channel at all.

    ``orbital_field`` adds the on-site ``i λ (p_x† p_y − p_y† p_x)``, the
    orbital angular momentum along ``z``.  It is odd under time reversal and
    under every mirror of the site — an axial quantity in the plane — so it
    leaves the sixfold axis and takes the six mirrors, the antiunitary
    operations with them, and what is left is a crystal with no antiunitary
    operation at all.
    """
    lat = np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2]])
    inv = np.linalg.inv(lat)
    m = TBModel(Lattice(lat.tolist(), [[0.0, 0.0]] * 2, periodic_dirs=[0, 1]),
                spinful=False)
    for d in ([1.0, 0.0], [0.5, np.sqrt(3.0) / 2], [-0.5, np.sqrt(3.0) / 2]):
        n = np.asarray(d, dtype=float)
        n = n / np.linalg.norm(n)
        t = t_sigma * np.outer(n, n) + t_pi * (np.eye(2) - np.outer(n, n))
        cell = np.rint(np.asarray(d, dtype=float) @ inv).astype(int).tolist()
        for a in range(2):
            for b in range(2):
                if abs(t[a, b]) > 1e-15:
                    m.set_hop(t[a, b], a, b, cell, mode="add")
    if orbital_field:
        m.set_hop(1j * float(orbital_field), 0, 1, [0, 0], mode="add")
    return m


def _decoupled(hoppings):
    """Decoupled orbitals at the one site of a square lattice, each with its own isotropic hopping.

    Equal hoppings make the orbitals degenerate, and then every matrix on them
    commutes with ``H(k)``; different ones leave one projector per orbital.
    """
    n = len(hoppings)
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]] * n,
                        periodic_dirs=[0, 1]), spinful=False)
    for o, t in enumerate(hoppings):
        m.set_hop(float(t), o, o, [1, 0], mode="add")
        m.set_hop(float(t), o, o, [0, 1], mode="add")
    return m


def _lieb_pair(couplings=(0.0, 0.0, 0.0), jitter=0.0):
    """Two identical Lieb lattices on the same sites, orbitals ``0-2`` and ``3-5``.

    Each copy hops ``-1`` from its corner to its two edge sites; ``couplings``
    joins the copies on site, at the corner and the two edges in turn, which
    a fourfold axis carries onto one another only where the edges' are equal.
    ``jitter`` turns one bond of the first copy off the other three, which the
    fourfold then carries onto one another only within it.
    """
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0, 0], [0.5, 0], [0, 0.5]] * 2,
                        periodic_dirs=[0, 1]), spinful=False)
    for off in (0, 3):
        for R in ([0, 0], [-1, 0]):
            m.set_hop(-1.0, off, off + 1, R, mode="add")
        for R in ([0, 0], [0, -1]):
            m.set_hop(-1.0, off, off + 2, R, mode="add")
    if jitter:
        m.set_hop(-float(jitter), 0, 1, [0, 0], mode="add")
    for o, c in enumerate(couplings):
        if c:
            m.set_hop(float(c), o, o + 3, [0, 0], mode="add")
    return m


def _lines(dim=2, coupling=0.0, odd=False):
    """Three orbitals at the one site of a hexagonal lattice, each hopping along its own line.

    Line ``i`` runs along ``a1``, ``a2`` or ``a2 - a1``: every orbital hops
    ``-1`` to its nearest neighbours and ``0.35`` to the second along its own
    line alone, and in three dimensions ``-0.4`` along the axis.  Nothing
    couples the three unless ``coupling`` says so; then orbital 0 hops to the
    other two along its own line, even under the mirror across it, or odd where
    ``odd`` says so.
    """
    steps = [np.asarray(v[:dim]) for v in ([1, 0, 0], [0, 1, 0], [-1, 1, 0])]
    lat = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2, 0.0], [0.0, 0.0, 1.3]])
    m = TBModel(Lattice(lat[:dim, :dim].tolist(), [[0.0] * dim] * 3,
                        periodic_dirs=list(range(dim))), spinful=False)
    for i, own in enumerate(steps):
        for step in steps:
            m.set_hop(-1.0, i, i, step.tolist(), mode="add")
        if dim == 3:
            m.set_hop(-0.4, i, i, [0, 0, 1], mode="add")
        m.set_hop(0.35, i, i, (2 * own).tolist(), mode="add")
    for j in ((1, 2) if coupling else ()):
        m.set_hop(coupling, 0, j, steps[0].tolist(), mode="add")
        m.set_hop(-coupling if odd else coupling, j, 0, steps[0].tolist(), mode="add")
    return m


def _names(record, reason="declaration"):
    """The operations one descent record's ``reason`` refuted, by name."""
    return {b["name"] for b in record["dropped"] if b["refuted_by"] == reason}


# ---------------------------------------------------------------------------
# 1.  Kane-Mele: the owner's two calls (A2 on both sublattices is in section 8)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_kane_mele_declared_a1_a2_descends_to_p3(realized):
    """``A2`` of ``3m`` is even under ``C3`` and odd under the three mirrors.

    Every hopping of this model is ``C3``-symmetric, so a declaration that gives
    the two sublattices *different* parities under the mirrors states a crystal
    whose mirrors are not symmetries — and the group is ``P3``, not the trivial
    group.  The six operations it breaks are the three mirrors and their primed
    partners: a site character is a character of the **unitary** site group, so
    ``m(x)`` and ``m(x)'`` stand or fall together, and the record counts both
    under ``declaration``.

    The declaration is carried across the descent as content, never resolved
    again: ``A2`` of ``3m`` has no name in ``3``, so the constraint is re-keyed
    through the step's own pairing and the operations the smaller group no
    longer has drop out.  And ``typat=[0, 1]`` has already split the cell, so
    the declaration reduces nothing before the hoppings are read.
    """
    declaration = {0: "A1", 1: "A2"}
    out = realized(_km, typat=[0, 1], cross_check=False, characters=declaration)
    assert out, str(out)
    assert (out.group.number_str, out.group.name, out.group.size) == ("143.2", "P31'", 6)
    assert len(out.validated.indices) == 6
    assert out.rep.characters.irreps == ("-1E2E", "-1E2E")

    d = out.descent
    assert d["from"] == {"number_str": "156.50", "name": "P3m11'", "size": 12}
    assert d["to"] == {"number_str": "143.2", "name": "P31'", "size": 6}
    assert d["steps"][0]["refuted_by"] == {"declaration": 6}
    assert d["dropped_by"] == {"declaration": 6}
    assert _names(d) == {"m(x)", "m([010])", "m([1-10])",
                         "m(x)'", "m([010])'", "m([1-10])'"}
    assert "the declared orbital content broke 6 of them" in d["note"]
    assert ("(6 of 12 coset representatives of the lattice translations broken by the "
            "declared content") in repr(out)

    rec = out.rep.characters.orbitals[1]
    assert rec.declared_irrep == declaration[1]
    assert rec.declared_in == "3m in P3m11' #156.50"
    assert rec.site_symmetry == "3"
    assert rec.irrep == "-1E2E", (
        "the declared irrep restricted to the returned site group, in the "
        "vocabulary the record returns: both orbital characters of a threefold "
        "axis lift to its one spinor pair")
    assert set(rec.declaration_broke) == _names(d)

    line = [ln for ln in out.rep.characters.summary().splitlines()
            if ln.strip().startswith("orbital 1:")]
    assert len(line) == 1, "one line per declaration that lowered the group"
    assert f"declared {declaration[1]!r}, read at the 3m in P3m11' #156.50" in line[0]
    assert "the declaration breaks" in line[0] and "returned as -1E2E of 3" in line[0]

    # re-keyed, never re-resolved: the served site group has no A2 to resolve
    # into, and every kept operation was carried through the step's pairing
    served = {int(s.ind) for s in out.group.symmetries if not s.time_reversal}
    assert len(served) == 3
    table = out.rep.characters.site_tables[1]
    assert "A2" not in table.labels, "the served site group has no A2 to resolve into"
    assert set(table.indices) <= served
    pairs = out.descent["steps"][0]["pairs"]
    assert {b for _a, b in pairs} == served, "every kept operation was re-keyed"
    assert len(out.rep.characters.orbitals[1].character) == len(table.indices)

    # the split stated by typat costs the declaration nothing: no cell reduction
    assert _reduction(out) is None and "cell lowered" not in repr(out)
    assert out.rep.characters.orbitals[1].reduced_in == "", "nothing was reduced"


# ---------------------------------------------------------------------------
# 2.  lin3g: one site with two channels, and the tie rule
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_lin3g_both_1a_orbitals_declared_alike_breaks_the_fourfold():
    """The *positions* allow both ``1a`` orbitals to be ``s``; the hoppings do not.

    Their measured relative character is ``B1`` of ``4mm`` — odd under the
    fourfold and under the two diagonal mirrors — so declaring them alike states
    a crystal without those four operations (eight with their primed partners).

    Three orbitals at one ``6mm`` site, each hopping along its own line and
    nothing coupling them, declared ``A1 + A2 + B1``: the sixfold carries each
    line onto the next, so in ``P6mm1'`` they are one orbit of three sectors of
    what the Hamiltonian conserves, and no character of one line's own group
    induces that content.  In ``Cmm2`` the mirror across line 0 leaves it an
    orbit of its own and the other two one orbit, each orbit with a character
    of its own, and the content is carried there: a smaller group splits the
    orbits by itself, so every subgroup is asked on its own.  Line 0 coupled
    to the pair by a hopping of ``1e-7``, odd under the mirror across it,
    below a declared precision of ``1e-5``, and the three declared
    ``A1 + A2 + B2``: every operation holds exactly, and line 0 and the pair
    are two parts of the crystal coupled only below the precision, where a
    coupling counts as absent, so the crystal is ``Cmm2`` again, the
    projection leaves the coupling out, and the model returned holds the two
    parts apart.
    """
    out = tqg.realized_group(tqm.lin3g(), typat=[0, 0, 1, 1], cross_check=False,
                             characters={0: "s", 1: "s"})
    assert out, str(out)
    assert (out.group.number_str, out.group.name) == ("25.58", "Pmm21'")
    assert len(out.validated.indices) == out.group.size == 8

    d = out.descent
    assert d["from"]["number_str"] == "99.164" and d["from"]["size"] == 16
    assert d["steps"][0]["refuted_by"] == {"declaration": 8}
    assert _names(d) == {"C4(z)", "C4^-1(z)", "m([110])", "m([1-10])",
                         "C4(z)'", "C4^-1(z)'", "m([110])'", "m([1-10])'"}
    assert out.rep.characters.irreps == ("-E", "-E", "-E", "-E")
    assert out.rep.characters.orbitals[1].site_symmetry == "mm2"
    assert out.rep.characters.orbitals[1].determined_by == "declaration"

    # three uncoupled lines at one site: the descent lands on Cmm2, not below it
    lines = tqg.realized_group(_lines(), typat=[0] * 3, cross_check=False,
                               characters={(0, 1, 2): "A1 + A2 + B1"})
    assert lines, str(lines)
    assert (lines.group.number_str, lines.group.name) == ("35.166", "Cmm21'")
    assert lines.descent["from"]["number_str"] == "183.186"
    assert lines.descent["steps"][0]["refuted_by"] == {"declaration": 16}
    assert lines.rep.multiplets.by_site == (((0,), (1, 2)),)
    assert lines.rep.characters.irreps == ("B2", "A2 + A1", "A2 + A1")

    # ... and coupled below a declared precision: the same crystal, the coupling left out
    near = tqg.realized_group(_lines(coupling=1e-7, odd=True), typat=[0] * 3,
                              cross_check=False, precision=1e-5,
                              characters={(0, 1, 2): "A1 + A2 + B2"})
    assert near, str(near)
    assert near.group.number_str == "35.166"
    parts = near.symmetrization.parts
    assert near.symmetrization.near == () and parts["orbitals"] == ((0,), (1, 2))
    assert parts["exact"] < parts["coupling"] < 1e-6
    assert "2 parts of the crystal are coupled only below" in near.symmetrization.summary()
    assert not [h for h in near.model.hoppings
                if (h["from_orbital"] == 0) != (h["to_orbital"] == 0)], \
        "nothing couples line 0 to the pair in the model returned"


@pytest.mark.slow
def test_lin3g_1a_declared_a_doublet_keeps_the_diagonal_mm2():
    """``E`` on a pair the Hamiltonian splits: the diagonal mirrors survive.

    ``chi(C2) = −2`` against a measured ``+2`` reads as ``λ(C2) = −1``, and
    ``chi(σ_v) = 0`` against ``+2`` disallows the axial mirrors outright.  The
    fourfold subgroup would need ``λ(C4) = ±i``, which the antiunitary anchor
    forbids; the diagonal ``mm2`` has ``λ = B1`` or ``B2`` and is the unique
    largest, so that is what is served, and the ``2c`` orbitals are inferred
    ``C2``-odd there.

    Two identical orbitals at a square site that nothing couples are another
    matter: they are copies of one another, one multiplet whose representation
    the hoppings leave open, the doublet among them.  Declared ``E`` together,
    they carry it, in the tables' own matrices of ``E`` on ``(p_x, p_y)``, and
    ``P4mm1'`` is kept whole; the declaration made the choice, and no band
    table says the labels rest on one.  So do two identical Lieb
    lattices on the same sites coupled on site only below a declared precision
    of ``1e-6``, by ``1e-8`` and unequally at the two edges: the sectors read at
    that precision hold them as copies, where such a coupling counts as absent,
    so the projection holds them as exact copies -- nothing left between them --
    and the doublet declared at the corner is carried; undeclared, each copy
    is ``A1``, as on the exact copies.

    Coupled on site above that precision, the two lattices are a bonding and an
    antibonding lattice that nothing joins and no operation exchanges, each
    carrying a character of its own.  Coupled alike at every site they keep the
    fourfold axis, each ``A1``; coupled unequally at the two edges, the hoppings
    break it.  The doublet declared at the corner is no symmetry of the
    coupling there, and the group is the largest whose restriction of it, ``B2 +
    B1``, the bonding and the antibonding state carry -- ``Pmm21'``, tied with the
    diagonal ``Cmm21'``.  What lies below the precision -- edges coupled unequally
    by ``1e-8``, a bond of one lattice off the others by ``1e-8``, which joins the
    bonding lattice to the antibonding one -- counts as absent, and the
    projection leaves it out: the fourfold is kept.
    """
    out = tqg.realized_group(tqm.lin3g(), typat=[0, 0, 1, 1], cross_check=False,
                             characters={(0, 1): "E"})
    assert out, str(out)
    assert (out.group.number_str, out.group.name, out.group.size) == ("35.166",
                                                                      "Cmm21'", 8)
    d = out.descent
    assert d["steps"][0]["refuted_by"] == {"declaration": 8}
    assert _names(d) == {"C4(z)", "C4^-1(z)", "m(x)", "m(y)",
                         "C4(z)'", "C4^-1(z)'", "m(x)'", "m(y)'"}
    assert d["tied"] == (), "the diagonal mm2 is the unique largest reading"

    rec = out.rep.characters
    assert rec.irreps[2:] == ("-1E2E", "-1E2E"), "the 2c orbitals are inferred C2-odd"
    assert [o.site_symmetry for o in rec.orbitals][2:] == ["2", "2"]
    assert [(o.irrep, o.multiplet, o.declared_irrep) for o in rec.orbitals[:2]] == \
        [("-E", (0,), "E"), ("-E", (1,), "E")], (
        "E of 4mm subduced to the diagonal mm2 is reducible, B1 + B2, one on each of the "
        "pair's multiplets, and both lift to the one spinor corep of mm2: each row is its "
        "own multiplet's -E, the declared E kept beside it")

    # ... and E on two identical orbitals nothing couples: carried, nothing broken
    copies = tqg.realized_group(_decoupled((1.0, 1.0)), typat=[0, 0], cross_check=False,
                                characters={(0, 1): "E"})
    assert copies, str(copies)
    assert copies.group.number_str == "99.164" and copies.descent is None
    assert copies.rep.multiplets.by_site == (((0, 1),),)
    assert [(o.irrep, o.determined_by) for o in copies.rep.characters.orbitals] == \
        [("E", "declaration")] * 2
    (space,) = copies.rep.characters.multiplicity_spaces
    assert (space.content, space.determined_by, space.n_representations) == (
        "E", "declaration", 11)
    assert not tqw.hsp_table(copies.rep, bands=2).note, \
        "the declaration states what the copies carry: no choice of it to disclaim"
    # ... returned in the tables' own matrices of E on (p_x, p_y), time reversal
    # the identity on the two real orbitals, whatever basis of the copies was read
    syms = {int(s.ind): s for s in copies.group.symmetries}
    for ind, M in copies.rep.matrices.items():
        assert np.allclose(M, np.asarray(syms[int(ind)].rotation_cart)[:2, :2], atol=1e-12), ind

    # ... and copies on several sites coupled only below a declared precision: held as
    # exact copies, nothing left between them, the doublet carried
    for said, rows in (({(0, 3): "E"}, [("E", "declaration")] * 2),
                       (None, [("A1", "lowest-l")] * 2)):
        held = tqg.realized_group(_lieb_pair((1.3e-8, 0.7e-8, -0.9e-8)), cross_check=False,
                                  precision=1e-6, characters=said)
        assert held, str(held)
        assert held.group.number_str == "99.164" and held.descent is None
        corner = [held.rep.characters.orbitals[o] for o in (0, 3)]
        assert [(o.irrep, o.determined_by) for o in corner] == rows
        alike = held.symmetrization.copies
        assert alike["orbitals"] == ((0, 1, 2, 3, 4, 5),) and alike["copies"] == (2,)
        assert alike["exact"] < alike["off"] < 1e-6
        assert not [h for h in held.model.hoppings
                    if (h["from_orbital"] < 3) != (h["to_orbital"] < 3)], \
            "nothing couples the copies in the model returned"

    # ... and the copies coupled on site above it: a bonding and an antibonding
    # lattice, each its own character, what joins them below it left out
    unequal, alike = (1.3e-3, 0.7e-3, -0.9e-3), (1e-3,) * 3
    for couplings, jitter, said, group, corner, held in (
            (unequal, 0.0, None, "25.58", ("A1", "lowest-l"), None),
            (unequal, 0.0, {(0, 3): "E"}, "25.58", ("B2 + B1", "declaration"), None),
            (alike, 0.0, None, "99.164", ("A1", "lowest-l"), None),
            (alike, 0.0, {(0, 3): "E"}, "25.58", ("B2 + B1", "declaration"), None),
            ((1e-3 + 1.3e-8, 1e-3 + 0.7e-8, 1e-3 - 0.9e-8), 0.0, None, "99.164",
             ("A1", "lowest-l"), "near"),
            ((1e-8,) * 3, 1e-8, {(0, 3): "E"}, "99.164", ("E", "declaration"), "copies"),
            (alike, 1e-8, None, "99.164", ("A1", "lowest-l"), "shares")):
        out = tqg.realized_group(_lieb_pair(couplings, jitter=jitter), precision=1e-6,
                                 characters=said, cross_check=said is None and not jitter)
        assert out, str(out)
        assert out.group.number_str == group, (couplings, jitter, said)
        assert [(out.rep.characters.orbitals[o].irrep,
                 out.rep.characters.orbitals[o].determined_by) for o in (0, 3)] == [corner] * 2
        assert max(out.rep.residuals.values()) < 1e-12, "the matrices returned are exact"
        if said and couplings == alike:
            assert out.descent["tied"] == ((1, 2, 5, 6), (1, 2, 7, 8)), "tied with Cmm21'"
        sym = out.symmetrization
        assert (sym is None) == (held is None)
        if held == "near":
            assert not sym.copies and not sym.shares
        elif held == "copies":
            assert sym.copies["copies"] == (2,) and not sym.shares
        elif held == "shares":
            assert sym.shares["sectors"] == (2,) and not sym.copies
            assert sym.shares["exact"] < sym.shares["coupling"] < 1e-6


@pytest.mark.slow
def test_lin3g_a_doublet_beside_two_s_orbitals_keeps_the_first_mirror_by_the_tie_rule():
    """Two readings, each consistent alone and not together: the tie rule keeps one.

    Declaring the ``2c`` pair ``s`` as well contradicts ``λ(C2) = −1`` outright,
    and each diagonal mirror is then consistent **on its own** while the two
    together are not — the hoppings give both mirrors the same ``1a`` block where
    ``E`` needs different ones.  Nothing measured separates the two readings, so
    the tie rule chooses: neither keeps a rotation but the identity, and the
    mirror whose normal is ``[110]`` ranks before ``[1-10]`` in the conventional
    cell.  ``Cm1' #8.33`` is served, the ``1a`` doublet restricted to that mirror
    as two copies of ``-1E2E``, one on each of its rows, and both readings are
    recorded in the rule's order.  The ``2c`` pair lands on the general position, whose site group is
    the identity alone: each row is that group's own spinor pair, and the band
    representation the pair induces is written at ``2a`` on the mirror, two
    copies of its ``-1E2E``.
    """
    out = tqg.realized_group(tqm.lin3g(), typat=[0, 0, 1, 1], cross_check=False,
                             characters={(0, 1): "E", 2: "s", 3: "s"})
    assert out, str(out)
    assert (out.group.number_str, out.group.name, out.group.size) == ("8.33", "Cm1'", 4)
    d = out.descent
    assert d["steps"][0]["refuted_by"] == {"declaration": 12}
    assert d["tied"] == ((1, 7), (1, 8)), \
        "{E, m([110])} and {E, m([1-10])}, the two maximal readings, in the rule's order"
    assert "{E, m([110])} and {E, m([1-10])}" in d["note"]
    assert "its principal axis, [110], comes before [1-10]" in d["note"]
    assert {"m([1-10])", "m([1-10])'"} <= _names(d) and "m([110])" not in _names(d), \
        "the first reading is the one kept"
    rec = out.rep.characters
    assert [(o.wyckoff, o.site_symmetry, o.irrep, o.named_by) for o in rec.orbitals] == \
        [("2a", "m", "-1E2E", "tables")] * 2 + [("4b", "1", "-AA", "point group")] * 2
    doublet, pair = rec.band_representations
    assert [(m, name, position.split("(")[0]) for m, name, position in pair.terms] == \
        [(2, "-1E2E↑G(2)", "2a")]
    assert pair.sentence() in rec.summary() and doublet.sentence() not in rec.summary()


# ---------------------------------------------------------------------------
# 3.  a declaration is a site's content, read on the measured multiplets
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_square_doublet_declared_s_and_d_keeps_the_fourfold():
    """``{0: "s", 1: "dx2-y2"}`` on a measured doublet: a site's content, and the fourfold stays.

    Nothing couples ``p_x`` to ``p_y``, so the fourfold carries orbital 0's
    hopping along ``x`` onto orbital 1's along ``y`` and the pair is one
    measured multiplet, whose solution spaces leave the sign of each orbital
    open.  It is then carried as faithfully by the ``p`` doublet ``E`` as by an
    ``s`` beside a ``d``, ``A1 + B1``: the Hamiltonian cannot tell the two
    apart, and the declaration says which.  ``A1 + B1`` is the site's content,
    so ``P4mm1'`` is kept whole, its ``s`` the sum of the two orbitals as
    written and its ``d`` their difference, and the fourfold carries one onto
    itself and the other onto minus itself.  Undeclared, the lowest-``l`` rule
    reads the same pair ``E``.  So it does in every basis the pair is written
    in, ``(p_x ± p_y)/√2`` among them: there the fourfold's solution space holds
    a diagonal member, and what binds the two orbitals into one multiplet is
    that each holds half of each of the two sectors the Hamiltonian conserves,
    ``p_x`` and ``p_y``.

    The same reading at the ``3m`` site of ``R3m``, its ``p`` shell turned
    inside the site by ``1e-8``: the shell is one measured multiplet, and
    ``{0: "A1", (1, 2): "A1 + A2"}`` states it ``2xA1 + A2`` where it holds
    ``A1 + E``, so the threefold and two of the mirrors are broken by the
    declaration and the crystal is ``Cm1'``, as with the shell aligned.  The
    rows are the shell's content on the one mirror, ``A'' + 2xA'``, and the
    matrices are the Hamiltonian's own blocks, exact, so nothing is projected
    and the call needs no symmetrization.
    """
    out = tqg.realized_group(_square_pxpy(), typat=[0, 0], cross_check=False,
                             characters={0: "s", 1: "dx2-y2"})
    assert out, str(out)
    assert (out.group.number_str, out.group.name, out.group.size) == ("99.164",
                                                                      "P4mm1'", 16)
    assert out.descent is None and len(out.validated.indices) == 16
    assert out.rep.multiplets.by_site == (((0, 1),),), "the multiplets are the measured ones"

    rec = out.rep.characters
    assert set(rec.irreps) == {"B1 + A1"}, "the site holds A1 + B1, the pair as a whole"
    assert [o.determined_by for o in rec.orbitals] == ["declaration"] * 2
    assert rec.orbitals[1].declared_irrep == "B1", "dx2-y2 resolved in 4mm"
    assert rec.orbitals[1].declared_in == "4mm in P4mm1' #99.164"
    assert rec.orbitals[1].site_symmetry == "4mm"
    frame = {name: ind for ind, name in rec.frame.items()}
    chi = rec.orbitals[0].character
    for name, want in (("C2(z)", 2.0), ("C4(z)", 0.0), ("m(x)", 2.0), ("m([110])", 0.0)):
        assert complex(chi[frame[name]]).real == pytest.approx(want, abs=1e-9), name
    s_and_d = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    four = s_and_d.T @ np.asarray(out.rep.matrices[frame["C4(z)"]]) @ s_and_d
    assert np.allclose(four, np.diag(np.diag(four)), atol=1e-12) and \
        sorted(np.diag(four).real.round(12)) == [-1.0, 1.0], \
        "the fourfold carries the sum of the two orbitals onto plus or minus itself, " \
        "and the difference onto the other sign"
    # what the Hamiltonian conserves at the site: p_x and p_y, which nothing
    # couples, are two sectors, and the fourfold exchanges them
    conserved = out.conserved
    assert [(s["orbitals"], s["copies"], s["dim"]) for s in conserved["sectors"]] == \
        [((0,), 1, 1), ((1,), 1, 1)]
    assert conserved["images"][frame["C4(z)"]] == (1, 0)
    assert conserved["images"][frame["m(x)"]] == (0, 1)

    # the same pair written in a basis turned inside the site, (p_x +- p_y)/sqrt(2)
    # at pi/4 among them: one multiplet, E undeclared and A1 + B1 declared, in every
    # frame
    for angle in (0.0, 0.3, np.pi / 8, np.pi / 4, np.pi / 2):
        model = _square_pxpy(angle)
        for characters, rows, how in ((None, "E", "lowest-l"),
                                      ({0: "s", 1: "dx2-y2"}, "B1 + A1", "declaration")):
            if angle == 0.0 and characters is not None:
                continue                         # read above
            got = tqg.realized_group(model, typat=[0, 0], cross_check=False,
                                     characters=characters)
            assert got, (angle, str(got))
            assert (got.group.number_str, got.descent) == ("99.164", None), angle
            assert got.rep.multiplets.by_site == (((0, 1),),), angle
            assert got.rep.characters.irreps == (rows, rows), angle
            assert [o.determined_by for o in got.rep.characters.orbitals] == [how] * 2
            assert [s["copies"] for s in got.conserved["sectors"]] == [1, 1], angle

    # the 3m site of R3m, its p shell turned inside it by 1e-8
    shell, kw = tqm.r3m_p_shell(angle=1e-8), {"decorations": tqm.R3M_SHADOW}
    said = {0: "A1", (1, 2): "A1 + A2"}
    for symmetrize in ("auto", False):
        turned = tqg.realized_group(shell, characters=said, symmetrize=symmetrize, **kw)
        assert turned, str(turned)
        assert (turned.group.name, turned.group.number_str, turned.group.size) == (
            "Cm1'", "8.33", 4)
        assert turned.descent["from"]["number_str"] == "160.66"
        assert turned.descent["steps"][0]["refuted_by"] == {"declaration": 8}
        assert turned.rep.multiplets.by_site == (((0, 1, 2),),)
        assert turned.rep.characters.irreps == ("A'' + 2xA'",) * 3
        assert turned.symmetrization is None, "the model's own blocks are exact"
        assert max(turned.rep.residuals.values()) < 1e-13


@pytest.mark.slow
def test_a_measured_multiplet_is_assigned_the_lowest_l_its_characters_allow():
    """The doublet's site irrep is assigned by the lowest-``l`` rule, from its characters.

    Two dimensions at a ``6mm`` site can carry ``E1`` -- the ``p_x, p_y`` pair --
    or ``E2`` -- the ``d_xy, d_x²₋y²`` one -- and the hoppings cannot tell them
    apart: they differ by the one-dimensional character ``B2``, which rescales
    each operation's block on its own.  What settles it is the lowest ``l`` able
    to carry the site's content, so the doublet is p-like — ``E1``, the
    character ``−2`` under ``C2(z)``, which is what being p-like is — whatever
    basis the model writes the pair in; the row names no function, since
    nothing was declared.

    An on-site ``i λ (p_x† p_y − p_y† p_x)`` keeps the sixfold axis and breaks
    every mirror and time reversal with it, so the crystal is ``P6 #168`` and
    carries no antiunitary operation: there is no reality anchor, and with no
    one-dimensional channel at the site there was nothing at all to pin the
    multiplet's phase with.  The pair's own block carries the same characters;
    the site group ``6`` has only one-dimensional irreps and tells the pair
    apart into the conjugate ``1E2 + 2E2``, and the group is returned rather
    than refused.

    Declared, the doublet carries what the declaration names.  ``E1`` is the
    undeclared labelling itself.  ``E2`` is ``E1`` times ``B1``/``B2`` — ``-1``
    on ``C2`` and the two sixfolds, ``+1`` on ``C3`` — which is the statement
    that the pair is ``d_xy, d_x²₋y²`` rather than ``p_x, p_y``: another
    labelling of this crystal and not a reason to lower it.  The verdict stage
    measures it so: with no one-dimensional multiplet, the pair's own blocks,
    settled into a representation, are the scale -- a ``(p_x, p_y)`` pair's
    blocks carry ``E1`` -- and against them the declared ``E2`` differs from
    the ``E1`` they carry by ``-1`` on the sixfolds and the twofold alone -- a
    real character of the group, so it breaks no operation.  The matrices returned carry it, the six mirrors
    reading ``0`` as on any two-dimensional irrep of ``6mm``, and an irrep
    label names no function.
    """
    for model, group, size, label in ((_tri_pxpy(), "183.186", 24, "E1"),
                                      (_tri_pxpy(orbital_field=0.4), "168", 6,
                                       "1E2 + 2E2")):
        out = tqg.realized_group(model, cross_check=False)
        assert out, str(out)
        assert out.group.number_str == group and out.group.size == size
        rec = out.rep.characters
        assert rec.irreps == (label,) * 2
        frame = {name: ind for ind, name in rec.frame.items()}
        chi = rec.orbitals[0].character
        assert complex(chi[frame["C2(z)"]]).real == pytest.approx(-2.0, abs=1e-9)
        assert complex(chi[frame["C3(z)"]]).real == pytest.approx(-1.0, abs=1e-9)
        assert max(out.rep.residuals.values()) < 1e-13
        assert out.rep.product_residual < 1e-13

    other = tqg.realized_group(_tri_pxpy(), cross_check=False, characters={(0, 1): "E2"})
    assert other, str(other)
    assert other.group.number_str == "183.186" and other.group.size == 24
    assert other.descent is None and len(other.validated.indices) == 24
    assert other.rep.multiplets.by_site == (((0, 1),),)
    rec = other.rep.characters
    assert rec.irreps == ("E2",) * 2
    assert rec.orbitals[0].site_symmetry == "6mm"
    assert rec.orbitals[0].declared_in == "6mm in P6mm1' #183.186"
    assert {o.function for o in rec.orbitals} == {""}, "an irrep label names no function"
    assert [o.determined_by for o in rec.orbitals] == ["declaration"] * 2
    # the verdict read the declaration against the pair's own settled blocks,
    # which carry E1: the two differ by -1 exactly on the sixfolds and the
    # twofold, the value of B1 and of B2 there
    flipped = {r["name"] for r in rec.checks if abs(r["measured"] - r["declared"]) > 1e-9}
    assert flipped == {"C6(z)", "C6^-1(z)", "C2(z)"}
    assert all(abs(r["declared"] + r["measured"]) < 1e-9 for r in rec.checks
               if r["name"] in flipped)
    frame = {name: ind for ind, name in rec.frame.items()}
    chi = rec.orbitals[0].character
    assert complex(chi[frame["C2(z)"]]).real == pytest.approx(2.0, abs=1e-9)
    assert complex(chi[frame["C6(z)"]]).real == pytest.approx(-1.0, abs=1e-9)
    mirrors = [ind for name, ind in frame.items() if name.startswith("m(") and ind in chi]
    assert len(mirrors) == 6 and all(abs(complex(chi[ind])) < 1e-9 for ind in mirrors), \
        "the six mirrors read zero on a two-dimensional irrep of 6mm"


# ---------------------------------------------------------------------------
# 4.  the one refusal that is not a descent
# ---------------------------------------------------------------------------

def test_a_declaration_a_complex_twist_serves_is_served_with_it():
    """``i41_soc``'s unitary group, ``4a`` declared ``B``: the twist, applied.

    The declaration needs ``λ`` odd on ``2_z``, and ``B(2_z) = B(4⁺)² = +1`` for
    every real character of ``Z4`` — but ``λ(4⁺) = ±i`` serves it exactly.  This
    group carries no antiunitary operation to forbid that: such a character
    preserves the covariance, the canonical norm and the whole product table, so
    nothing measured here can see it and it is a physical twist rather than a
    reason to lower a group the Hamiltonian has.  P1c therefore **serves** it —
    ``declaration_needs_complex_twist`` is retired — and what it does is exchange
    the ``P`` labels the corep naming rests on, level by level, which is measured
    below.

    Which of ``±i`` is returned the declaration cannot say, and the orbitals
    read alike under both; the band labels decide.  The two conjugates label
    the bands at Γ and ``M`` differently, and the one whose labels come first
    in the tables' order is returned: ``λ(4⁺) = -i``, under which the lowest
    pair at Γ reads ``-GM5 + -GM8`` where ``+i`` would read ``-GM6 + -GM7``.
    """
    model = tqm.i41_soc("generic", lam=0.4)
    group = _spacegroup._group_of(model, include_TR=False)
    assert group and group.sg_number == 80, str(group)
    assignment = _tables._assign_orbitals(model, group)
    validated = _measure._validate_group(model, group, nk=4)

    default = _build._build_rep(model, group, assignment, validated=validated, n_battery=5)
    assert default, str(default)
    assert default.characters.irreps == ("-1E + -2E", "-1E + -2E") + ("2x-A",) * 4, (
        "the 4a pair's states are the spinor pair the tables name at that position; "
        "the 8b orbitals sit at the general position, which the EBR table does not "
        "carry, so each is named by its trivial site group itself: a spin doublet "
        "there is two copies of that group's one double-valued irrep")

    out = _build._build_rep(model, group, assignment, validated=validated, n_battery=5,
                         characters={0: "B"})
    assert out, str(out)
    assert out.characters.irreps == ("-1E + -2E", "-1E + -2E") + ("2x-A",) * 4, \
        "B and A of a twofold axis lift to the one spinor pair; the 8b rows are named " \
        "by their trivial site group"
    assert out.characters.orbitals[0].declared_as == "B"
    assert out.characters.orbitals[0].determined_by == "declaration"
    lat = np.asarray(group.lattice, dtype=float)
    named = {_sitechars.operation_name(out._operation(int(i)), lat): int(i)
             for i in out.indices}
    assert set(named) == {"E", "C2(z)", "C4(z)", "C4^-1(z)"}, sorted(named)
    four = named["C4(z)"]

    def _ratio(ind):
        """The scalar the declared matrix is of the undeclared one, read off the two."""
        a, b = np.asarray(default.matrices[ind]), np.asarray(out.matrices[ind])
        rho = complex(np.vdot(a, b) / np.vdot(a, a))
        assert float(np.abs(b - rho * a).max()) < 1e-12, ind
        return rho

    twist = {ind: _ratio(ind) for ind in named.values()}
    assert abs(twist[named["C2(z)"]] + 1.0) < 1e-9, "the declaration asks for it"
    assert abs(twist[four] * twist[named["C4^-1(z)"]] - 1.0) < 1e-9
    assert abs(twist[four] + 1j) < 1e-9, "the band labels take -i"
    assert [key for choice in out.characters._labellings["choices"]
            for key, _labels in choice["others"]] == ["bands"]
    lowest = tqw.hsp_table(out, bands=2)
    assert lowest.content["GM"] == {"-GM5": 1, "-GM8": 1}
    assert lowest.note.endswith("come first in the tables' order")
    assert all(abs(abs(v) - 1.0) < 1e-9 for v in twist.values())
    assert max(out.residuals.values()) < 1e-13 and out.product_residual < 1e-13, \
        "a complex character preserves the covariance and the product table"

    # the P labels, exchanged level by level -- which is what the twist DOES
    P = tqg.high_symmetry_points(group)["P"]
    states = tqw.solve_at(model, P)
    w = np.asarray(states[0], dtype=float).reshape(-1)[:4]
    got = []
    for block in tqw.multiplets(w):
        a = tqw.corep_content(default, P, states=states, bands=list(block))
        b = tqw.corep_content(out, P, states=states, bands=list(block))
        assert a and b, (block, str(a), str(b))
        got.append((a.counts, b.counts))
    assert got == [({"-P3": 1}, {"-P4": 1}),
                   ({"-P4": 2}, {"-P3": 2}),
                   ({"-P3": 1}, {"-P4": 1})], got

    ok = _build._build_rep(model, group, assignment, validated=validated, n_battery=5,
                        characters={0: "A"})
    assert ok, str(ok)
    assert max(float(np.abs(np.asarray(ok.matrices[i])
                            - np.asarray(default.matrices[i])).max())
               for i in default.indices) == 0.0


# ---------------------------------------------------------------------------
# 5.  the caps on the two enumerations
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_measured_blocks_enumeration_is_capped_and_says_so():
    """The two enumerations that grow with the model, each bounded by ``max_candidates``.

    A multi-dimensional multiplet whose solution spaces are more than
    one-dimensional has its block fixed only up to what the model conserves
    inside it.  Where the conserved operators are projectors no operation
    exchanges -- four decoupled orbitals with different hoppings, stated as one
    multiplet -- each orbital carries a representation of its own, and the
    representations of the multiplet are every choice of one for each.  Where
    they are a whole matrix algebra -- the same four orbitals with one hopping,
    which conserve a fourfold pseudospin -- they are the sums of the group's
    irreducible representations the multiplicity space could carry.  Past the
    ceiling each build refuses rather than starting a search that would not
    finish, and the message names the multiplet, what it measured there and the
    declaration that settles it.
    """
    def _bundle(model, n=4):
        group = _spacegroup._group_of(model, typat=[0] * n)
        assert group, str(group)
        assignment = _tables._assign_orbitals(model, group, typat=[0] * n)
        return model, group, assignment, _measure._validate_group(model, group)

    # the orbitals nothing couples, each its own choice: 4 x 4 x 4 x 4 ways
    model, group, assignment, validated = _bundle(_decoupled((1.0, 1.1, 1.2, 1.3)))
    out = _build._build_rep(model, group, assignment, validated=validated,
                         multiplets=[[(0, 1, 2, 3)]], max_candidates=64)
    assert not out and out.gate == "rep_search_too_large", str(out)
    assert out.context["multiplet"] == (0, 1, 2, 3) and out.context["sets"] == 4
    assert out.value == 256.0 and out.threshold == 64.0
    assert "4 sets of states" in out.reason and "`characters=`" in out.reason
    wide = _build._build_rep(model, group, assignment, validated=validated,
                          multiplets=[[(0, 1, 2, 3)]])
    assert wide, str(wide)
    assert [(o.irrep, o.determined_by) for o in wide.characters.orbitals] == \
        [("A1", "lowest-l")] * 4, "each orbital read on its own, an s each"

    # ... and the pseudospin's own enumeration: the sums of dimension four
    model, group, assignment, validated = _bundle(_decoupled((1.0,) * 4))
    out = _build._build_rep(model, group, assignment, validated=validated,
                         multiplets=[[(0, 1, 2, 3)]], max_candidates=16)
    assert not out and out.gate == "rep_search_too_large", str(out)
    assert out.context["multiplet"] == (0, 1, 2, 3) and out.context["multiplicity"] == 4
    assert out.context["n_candidates"] == 46
    assert out.value == 46.0 and out.threshold == 16.0
    assert "fourfold pseudospin" in out.reason and "46 sums" in out.reason
    assert "`characters=`" in out.reason


# ---------------------------------------------------------------------------
# 6.  the record carries the declaration across every re-identification
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_several_maximal_readings_of_a_declaration_name_the_first_by_the_tie_rule(realized):
    """Two maximal readings, neither containing the other: the refusal names the first.

    On Kane-Mele with no sublattice potential, read on a substrate, the two
    sublattices are one orbit of ``P6mm1' #183.186``.  Declaring ``s`` on one
    and ``A2`` on the other reads relative character ``A2``, and two readings
    of that are maximal -- the sixfold rotations and the ``3m``.  The refusal
    ``_build_rep`` serves on that group names the first by the tie rule, the
    one with the most rotations, says why it is first, and names it as the
    subgroup ``realized_group`` descends to.
    """
    km = realized(tqm.kane_mele_inversion, environment="substrate", cross_check=False,
                  characters={0: "A1"})
    assert km and km.group.number_str == "183.186", str(km)
    wide = _build._build_rep(km.model, km.group, km.assignment, validated=km.validated,
                          characters=["s", "A2"])
    assert not wide and wide.gate == "declaration_breaks_group", str(wide)
    assert wide.context["relative_character"] == "A2"
    assert "several readings are maximal" in wide.reason
    assert len(wide.context["tied"]) == 2
    assert set(wide.context["tied"][0]) <= set(wide.context["subgroup"])
    assert len(wide.context["subgroup"]) < 24
    assert "the first by the tie rule" in wide.reason
    assert "it holds 6 rotations (proper operations) where the next holds 3" in wide.reason
    assert "realized_group" in wide.reason and "descends" in wide.reason


# ---------------------------------------------------------------------------
# 6.  one reading: the verdict stage chooses, and the build assembles
# ---------------------------------------------------------------------------

def _two_sites(coupling=0.0):
    """Two ``s``-like orbitals on inequivalent sites of a square lattice.

    ``1a`` at the origin with isotropic hoppings ``-1`` and ``1b`` at the cell
    centre with ``0.4``; on-site energies 0 and 1 and, unless ``coupling`` says
    otherwise, **nothing between them**.  The second site is then a channel the
    hoppings never reach: what it carries relative to the first is not a
    measurement this Hamiltonian can make, on this site or on any other.
    """
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]],
                        periodic_dirs=[0, 1]), spinful=False)
    m.set_onsite([0.0, 1.0])
    m.set_hop(-1.0, 0, 0, [1, 0])
    m.set_hop(-1.0, 0, 0, [0, 1])
    m.set_hop(0.4, 1, 1, [1, 0])
    m.set_hop(0.4, 1, 1, [0, 1])
    if coupling:
        for lv in ([0, 0], [-1, 0], [0, -1], [-1, -1]):
            m.set_hop(coupling, 0, 1, lv)
    return m


@pytest.mark.slow
def test_a_free_block_on_another_site_takes_the_least_twist(realized):
    """Undeclared, the decoupled second site reads ``A1`` — the trivial relative character.

    The least twist is no longer confined to one site.  Nothing couples orbital
    1 to orbital 0, so its solution-space block is free at every operation and
    the convention names it, exactly as it names two decoupled orbitals *at* one
    site.  What was decoupled is still said — ``conserved`` counts the matrix
    that commutes with ``H`` beyond the identity — and the row's
    ``determined_by`` says the lowest-``l`` rule made the choice.

    The control: one hopping in each of the four diagonal directions keeps the
    fourfold axis and ties the two channels together, so the relative character
    is read off the solution spaces instead of being named, and nothing
    commutes with ``H`` beyond the identity.
    """
    out = realized(_two_sites, typat=[0, 1], cross_check=False)
    assert out, str(out)
    assert out.group.number_str == "99.164" and out.descent is None
    assert len(out.validated.indices) == out.group.size == 16
    assert out.rep.characters.irreps == ("A1", "A1")
    assert out.rep.characters.orbitals[1].determined_by == "lowest-l"
    assert out.conserved["dim"] == 1, "the relative phase commutes with H at every k"
    assert [(s["orbitals"], s["sites"], s["copies"], s["dim"])
            for s in out.conserved["sectors"]] == [((0,), (0,), 1, 1), ((1,), (1,), 1, 1)], \
        "each site a sector of its own: the projector onto either commutes with H"

    coupled = tqg.realized_group(_two_sites(0.3), typat=[0, 1], cross_check=False)
    assert coupled, str(coupled)
    assert coupled.group.number_str == "99.164" and coupled.descent is None
    assert coupled.rep.characters.irreps == ("A1", "A1")
    assert coupled.conserved["dim"] == 0, "nothing commutes with H beyond the identity"
    assert [(s["orbitals"], s["copies"], s["dim"]) for s in coupled.conserved["sectors"]] == \
        [((0, 1), 1, 2)], "the hoppings join the two sites into one sector"


@pytest.mark.slow
@pytest.mark.parametrize("declaration,labels,determined_by", [
    ({0: "s", 1: "s"}, ("A1", "A1"), ("declaration", "declaration")),
    ({0: "s", 1: "dx2-y2"}, ("A1", "B1"), ("declaration", "declaration")),
    ({1: "dx2-y2"}, ("A1", "B1"), ("lowest-l", "declaration")),
    ({0: "dx2-y2"}, ("B1", "A1"), ("declaration", "lowest-l")),
])
def test_a_free_block_takes_the_declared_character_and_nothing_breaks(
        declaration, labels, determined_by):
    """The declaration settles the free block, and no operation is broken by it.

    This is the call the committed tree crashed on.  Nothing couples orbital 1,
    so the hoppings measure nothing about its character — there is no
    measurement for a declaration to contradict, and a verdict that said
    otherwise was a second reading of a measurement the verdict stage had
    already made.  ``{1: "dx2-y2"}`` is served on the whole of ``P4mm1'`` for the
    same reason ``{1: "s"}`` is: a decoupled orbital can be anything.  Declared
    on one site alone, either site, the other is assigned by the lowest-``l``
    rule, and nothing on this path raises.
    """
    out = tqg.realized_group(_two_sites(), typat=[0, 1], cross_check=False,
                             characters=declaration)
    assert out, str(out)
    assert out.group.number_str == "99.164" and out.descent is None
    assert out.rep.characters.irreps == labels
    assert tuple(o.determined_by for o in out.rep.characters.orbitals) == determined_by
    assert max(out.rep.residuals.values()) < 1e-13
    assert out.rep.product_residual < 1e-13


@pytest.mark.slow
def test_the_verdict_stage_chooses_and_the_build_does_not_re_measure(realized):
    """The choices the build assembles from are the verdict stage's, per operation.

    ``_verdicts_and_choices`` is the one reading: per operation it returns the
    class scalars the hoppings pin (``chosen``) and the classes they leave open
    (``free_classes``), and ``_build_rep`` assembles the matrices from those
    rather than measuring them again.  Handing the same record in by hand
    therefore changes nothing that is served.
    """
    base = realized(_two_sites, typat=[0, 1], cross_check=False)
    assert base, str(base)
    model, group, assignment, validated, _rep = base.bundle
    assert group.number_str == "99.164", str(group)
    layout = _multiplets._site_multiplets(model, group, assignment, None, validated)
    u_pos = [p for p, s in enumerate(group.symmetries) if not s.time_reversal]
    got = _declaration._verdicts_and_choices(model, group, assignment, None, validated,
                                    layout, (), u_pos)
    assert not isinstance(got, Refusal), str(got)
    assert got["nclass"] == 2 and got["free_classes"] == frozenset({1}), \
        "the reference class is pinned to 1 by construction; the other is free"
    assert all(abs(complex(v) - 1.0) < 1e-12
               for (ind, cls), v in got["chosen"].items() if cls == 0)
    assert not [k for k in got["chosen"] if k[1] == 1], \
        "nothing was measured about the free class, so nothing was recorded for it"

    plain = _build._build_rep(model, group, assignment, validated=validated)
    handed = _build._build_rep(model, group, assignment, validated=validated, _choices=got)
    assert plain and handed, (str(plain), str(handed))
    assert max(float(np.abs(np.asarray(plain.matrices[i])
                            - np.asarray(handed.matrices[i])).max())
               for i in plain.indices) == 0.0
    assert plain.characters.irreps == handed.characters.irreps == ("A1", "A1")


# ---------------------------------------------------------------------------
# 7.  step 1: what the declared content takes out of the cell
# ---------------------------------------------------------------------------
#
# P2.  A declaration says what the crystal IS, and both kinds of statement it
# can make reduce the cell before any Hamiltonian is read: two sites of one
# orbit declared different content are different sites, and a declared
# *function* whose span its site group does not preserve leaves only the
# operations that do.  One rule, one identification, provenance ``realized``:
# the operations are gone from the group, so no verdict below ever meets them
# and nothing downstream re-decides it.

def _reduction(out):
    """The step 1 record, off the decoration where it lives."""
    return (out.decoration or {}).get("declared")


def _rect_pair():
    """Two orbitals at one ``mm2`` site of a rectangular lattice, spinless.

    A site group that makes every ``p`` shorthand one-dimensional, which is
    where the joint form of one is a grammar error rather than a doublet.
    """
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.4]], [[0.0, 0.0]] * 2,
                        periodic_dirs=[0, 1]), spinful=False)
    m.set_hop(1.0, 0, 0, [1, 0], mode="add")
    m.set_hop(0.3, 1, 1, [1, 0], mode="add")
    m.set_hop(0.2, 0, 0, [0, 1], mode="add")
    m.set_hop(0.5, 1, 1, [0, 1], mode="add")
    return m


def _stacked_pair():
    """One ``s``-like orbital at each of the two ``2g`` sites ``(0, 0, ±z)`` of ``P4/mmm``, spinless.

    Each site's group is ``4mm``, and the operations exchanging the two are
    the eight that turn ``z`` over.  ``p_x`` declared at one site alone is kept
    by ``mm2`` there, and both ``mmm`` and ``-4m2`` hold that ``mm2`` together
    with four exchanging operations: two largest groups inside the operations
    the declared content is carried by, one order, which only the tie rule
    tells apart.
    """
    m = TBModel(Lattice([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.6]],
                        [[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]], periodic_dirs=[0, 1, 2]),
                spinful=False)
    for o in (0, 1):
        m.set_hop(-1.0, o, o, [1, 0, 0])
        m.set_hop(-1.0, o, o, [0, 1, 0])
    m.set_hop(-0.7, 0, 1, [0, 0, 0])
    m.set_hop(-0.3, 1, 0, [0, 0, 1])
    return m


@pytest.mark.slow
def test_kane_mele_a1_a2_with_no_typat_splits_the_cell_before_it_descends():
    """The species split is a **cell** statement, not a descent.

    With no ``typat`` the two sublattices are one orbit of ``P6mm1' #183.186``,
    and declaring them ``A1`` and ``A2`` says they are different atoms.  The
    twelve operations that exchange them — ``C2``, ``C6``, ``C6⁻¹`` and three
    of the six mirrors, each with its primed partner — are therefore not
    operations of the declared crystal at all, and the cell is identified as
    ``P3m11' #156.50`` instead.  Only then do the hoppings get a verdict, and
    they break the three mirrors that *fix* each sublattice, exactly as they do
    when ``typat=[0, 1]`` says the same thing about the atoms.

    So the eighteen operations this call used to charge to a descent are two
    different statements: twelve about what the crystal is, six about what its
    Hamiltonian does.

    What is compared is what the two sites are declared to hold, as the words
    state it.  Written ``-E1``, the second site's content is the corep ``A1``
    and ``A2`` both lift to, which is what an ``A2`` at the first site lifts
    to too: one species, and the crystal is ``P6mm1' #183.186`` whole.
    """
    out = tqg.realized_group(_km(), cross_check=False, characters={0: "A1", 1: "A2"})
    assert out, str(out)
    assert (out.group.number_str, out.group.size) == ("143.2", 6)

    red = _reduction(out)
    assert red is not None, "the split is on the decoration, not on `descent`"
    assert red["from"] == {"number_str": "183.186", "name": "P6mm1'", "size": 24}
    assert red["to"] == {"number_str": "156.50", "name": "P3m11'", "size": 12}
    assert red["n_excluded"] == 12
    assert {e["name"] for e in red["excluded"]} == {
        "C2(z)", "C6^-1(z)", "C6(z)", "m([2-10])", "m(y)", "m([110])",
        "C2(z)'", "C6^-1(z)'", "C6(z)'", "m([2-10])'", "m(y)'", "m([110])'"}
    assert {e["reason"] for e in red["excluded"]} == {
        "declared content differs across the orbit"}
    assert all(e["sites"] == (0, 1) and e["orbitals"] == (0, 1)
               for e in red["excluded"])

    # the descent is the hoppings' half, and it starts from the reduced cell
    d = out.descent
    assert d["from"]["number_str"] == "156.50" and len(d["dropped"]) == 6
    assert d["to"]["number_str"] == "143.2"
    assert _names(d) == {"m(x)", "m([010])", "m([1-10])",
                         "m(x)'", "m([010])'", "m([1-10])'"}
    assert not {b["name"] for b in d["dropped"]} & {e["name"] for e in red["excluded"]}, \
        "nothing is charged twice: what was never there was never broken"
    assert out.rep.characters.irreps == ("-1E2E", "-1E2E")

    text = repr(out)
    assert ("cell lowered by the declared content from #183.186 "
            "(12 of 24 coset representatives of the lattice translations excluded") in text
    assert text.index("cell lowered") < text.index("descended from")

    corep = tqg.realized_group(_km(), cross_check=False, characters={0: "A2", 1: "-E1"})
    assert corep, str(corep)
    assert (corep.group.number_str, len(corep.validated.indices)) == ("183.186", 24)
    assert _reduction(corep) is None and corep.descent is None
    assert corep.rep.characters.irreps == ("-E1", "-E1")


@pytest.mark.slow
def test_one_px_at_a_3m_site_is_the_crystal_with_one_mirror():
    """A declared **function** whose span its site group does not preserve.

    ``px`` names a Cartesian polynomial.  At a ``3m`` site the orbit it
    generates is the two-dimensional ``E``, so declaring ``px`` for the *one*
    orbital there says the crystal carries that one direction — and ``C3``,
    ``C3⁻¹`` and two of the three mirrors carry ``p_x`` out of its own span and
    are not symmetries of it.  What is left is the one mirror that carries
    ``p_x`` to ±itself, and the cell is ``Cm1' #8.33``.

    This used to refuse ``ambiguous_multiplet`` — and offer ``{(0,): 'px'}``,
    which is not a spelling at all.  It is not a grammar error: it is a
    crystal with less symmetry, measured before any hopping is read, so the
    identification's provenance is ``realized`` and ``descent`` is empty.  The
    words a declaration at the ``m`` site is written in are ``A'`` and ``A''``,
    as every table prints them.

    ``p_x`` on one sublattice and ``p_y`` on the other have the **same**
    stabilizer in ``3m`` -- of three mirrors sixty degrees apart only one has
    ``p_x`` and ``p_y`` for eigenvectors -- so the cell loses the same eight
    operations, and what is left is a statement the hoppings then contradict:
    ``p_x`` is odd under that mirror and ``p_y`` even, while the sublattices
    carry the same parity, so the mirror is broken by the declaration and the
    group is ``P11'``.  Eight operations were never in that crystal; two were
    broken in it.

    With no ``typat`` the two sublattices are one orbit, and ``p_x`` declared on
    the first alone leaves the second undeclared: an operation carrying site 0
    onto site 1 is compatible whatever it carries ``p_x`` onto there, and two
    such compose into a threefold that turns ``p_x`` out of its own span.  The
    compatible operations are no group, and the cell keeps the largest group
    inside them, ``Cmm21' #35.166``; the four the declared function is
    carried by that it leaves out, two sixfolds and two mirrors with their
    primed partners, are excluded beside the four that turn ``p_x`` at its own
    site, and the free-standing layer reads the same crystal.  Declared at
    both sites -- ``p_x`` at the second too, the function the operations kept
    carry the first site's onto -- it is the same crystal, with the same
    matrices: declaring a function at every site of an orbit states what
    declaring it at one does.  In three dimensions, where the cell's group is
    cubic, ``p_y`` declared at one ``3c`` site of the ``p`` pairs of
    ``cubic_3c_perp_doublet``, nothing at the other two, keeps ``Pm-31'``,
    the twelve rotations of the tetrahedron and their products with inversion,
    which carry ``p_y`` onto ``p_z`` and ``p_x`` at the other two sites; the
    hoppings then break the inversion coset, as they do undeclared, and
    nothing else, so ``P231'`` is returned.  ``p_y`` is the ``B2`` of ``222``
    the hoppings carry, and no one-dimensional character of ``T`` restricts to
    it: it is measured as its own character, and the declaration compared
    with that.  And where
    two largest groups tie -- ``p_x`` at one of the two ``(0, 0, ±z)`` sites of
    ``P4/mmm``, kept with ``mmm`` and with ``-4m2`` alike -- the tie rule keeps
    ``-4m2``, whose fourfold rotoinversion comes first, and each operation only
    ``mmm`` holds says so.
    """
    out = tqg.realized_group(_km(), typat=[0, 1], cross_check=False,
                             characters={0: "px"})
    assert out, str(out)
    assert (out.group.name, out.group.number_str, out.group.size) == ("Cm1'", "8.33", 4)
    assert out.group.provenance == "realized", "a cell statement, not a descent"
    assert out.descent is None

    red = _reduction(out)
    assert red["from"] == {"number_str": "156.50", "name": "P3m11'", "size": 12}
    assert red["to"]["number_str"] == "8.33" and red["n_excluded"] == 8
    assert {e["name"] for e in red["excluded"]} == {
        "C3(z)", "C3^-1(z)", "m([010])", "m([1-10])",
        "C3(z)'", "C3^-1(z)'", "m([010])'", "m([1-10])'"}
    assert {e["reason"] for e in red["excluded"]} == {
        "the declared function at this site is not carried to itself"}
    assert all(e["sites"] == (0, 0) for e in red["excluded"])

    # the one mirror that survives is the one p_x is an eigenvector of, and the
    # row is that eigenvalue: p_x is ODD under m(x), so the label is the site
    # group's odd one-dimensional irrep and nothing here is two-dimensional.
    row = out.rep.characters.orbitals[0]
    assert row.site_symmetry == "m" and row.dim == 2 and row.function == "px"
    assert row.irrep == "-1E2E", (
        "the spinor pair of a site group with one mirror: both of its orbital "
        "characters lift to it, and a spinor character is zero on a mirror, which "
        "is why the parity below is read off the declaration and not off the row")
    assert row.generator_character["m(x)"].real == pytest.approx(-1.0)
    mirror = next(s for s in out.group.symmetries
                  if not s.time_reversal
                  and not np.allclose(np.asarray(s.rotation_cart, dtype=float),
                                      np.eye(3)))
    R_inv = np.linalg.inv(np.asarray(mirror.rotation_cart, dtype=float))
    px = _sitechars._apply(np.eye(3), _sitechars.SHORTHANDS["px"], 1)
    image = _sitechars._apply(R_inv, _sitechars.SHORTHANDS["px"], 1)
    assert np.allclose(image, -px), "the kept mirror carries p_x to -p_x"
    table = out.rep.characters.site_tables[0]
    odd = [i for i, lab in enumerate(table.labels)
           if table.character_on(i, int(mirror.ind)).real < 0]
    assert len(odd) == 1 and row.irrep == table.labels[odd[0]], \
        "the odd row of m, not the even one"
    assert set(table.declarable) == {"A\'", "A\'\'"}, \
        "the orbital names a declaration is still written in"
    assert not [lab for t in out.rep.characters.site_tables.values() for lab in t.labels
                if "_" in lab], "no site group is served a generic suffix"

    # where the word was read and where it resolves are two different places,
    # and the record says both
    assert row.declared_in == "3m in P3m11' #156.50"
    assert row.reduced_in == "m in Cm1' #8.33"
    assert out.rep.characters.declarations[0]["reduced_in"] == "m in Cm1' #8.33"
    line = [ln for ln in out.rep.characters.summary().splitlines()
            if ln.strip().startswith("orbital 0:")]
    assert len(line) == 1 and "the declared content lowered the cell" in line[0]
    assert "m in Cm1' #8.33" in line[0] and "3m in P3m11' #156.50" in line[0]

    # the undeclared sublattice imposes nothing and is read relative to it
    assert out.rep.characters.orbitals[1].determined_by == "hamiltonian"

    # p_x and p_y: both sites reduce to the same mirror, and then it is broken
    both = tqg.realized_group(_km(), typat=[0, 1], cross_check=False,
                              characters={0: "px", 1: "py"})
    assert both, str(both)
    assert (both.group.number_str, both.group.size) == ("1.2", 2)
    red = _reduction(both)
    assert red["from"]["number_str"] == "156.50" and red["n_excluded"] == 8
    assert red["to"]["number_str"] == "8.33"
    assert both.descent["from"]["number_str"] == "8.33"
    assert both.descent["dropped_by"] == {"declaration": 2}
    assert _names(both.descent) == {"m(x)", "m(x)'"}
    rows = both.rep.characters.orbitals
    assert rows[0].declared_irrep != rows[1].declared_irrep, \
        "opposite parities under the one mirror, which is why it breaks"
    assert all(r.reduced_in == "m in Cm1' #8.33" for r in rows)

    # one orbit, its second site undeclared: the largest group the declared
    # function is carried by, and nothing descends
    outside = _declaration._OUTSIDE_THE_GROUP
    turned = "the declared function at this site is not carried to itself"

    def _by_reason(red):
        names = {}
        for e in red["excluded"]:
            names.setdefault(e["reason"], set()).add(e["name"])
        return names

    def primed(names):
        return names | {n + "'" for n in names}

    orbit = tqg.realized_group(_km(), characters={0: "px"})
    assert orbit, str(orbit)
    assert (orbit.group.name, orbit.group.number_str, orbit.group.size) == (
        "Cmm21'", "35.166", 8)
    assert orbit.group.provenance == "realized" and orbit.descent is None
    red = _reduction(orbit)
    assert red["from"] == {"number_str": "183.186", "name": "P6mm1'", "size": 24}
    assert red["to"]["number_str"] == "35.166" and red["n_excluded"] == 16
    assert _by_reason(red) == {
        turned: primed({"C3(z)", "C3^-1(z)", "m([010])", "m([1-10])"}),
        outside: primed({"C6(z)", "C6^-1(z)", "m([2-10])", "m([110])"})}
    assert all(e["sites"] == (0, 1) and e["orbitals"] == (0,)
               for e in red["excluded"] if e["reason"] == outside), \
        "each carries the declared site onto the undeclared one"
    assert orbit.cross_check["outcome"] == "agree"
    assert orbit.cross_check["free_standing"]["number_str"] == "35.166"
    assert [o.determined_by for o in orbit.rep.characters.orbitals] == \
        ["declaration", "hamiltonian"]
    # declared at every site of the orbit, the second site's function the one the
    # kept operations carry the first site's onto: the same crystal
    every = tqg.realized_group(_km(), cross_check=False, characters={0: "px", 1: "px"})
    assert every, str(every)
    assert every.group.number_str == "35.166" and every.descent is None
    assert every.rep.characters.irreps == orbit.rep.characters.irreps
    assert [o.determined_by for o in every.rep.characters.orbitals] == ["declaration"] * 2
    assert max(float(np.abs(np.asarray(every.rep.matrices[i])
                            - np.asarray(orbit.rep.matrices[i])).max())
               for i in orbit.rep.indices) < 1e-14, "one crystal, one representation"

    # ... and in three dimensions, where the cell's group is cubic
    cubic = tqg.realized_group(cubic_3c_perp_doublet(), characters={0: "py"})
    assert cubic, str(cubic)
    red = _reduction(cubic)
    assert red["from"] == {"number_str": "221.93", "name": "Pm-3m1'", "size": 96}
    assert red["to"] == {"number_str": "200.15", "name": "Pm-31'", "size": 48}
    by_reason = _by_reason(red)
    assert set(by_reason) == {turned, outside} and len(by_reason[outside]) == 32
    assert by_reason[turned] == primed({"C4(x)", "C4^-1(x)", "S4(x)", "S4^-1(x)",
                                        "C2([011])", "C2([01-1])", "m([011])",
                                        "m([01-1])"}), \
        "the operations of the site's 4/mmm that turn p_y into p_z"
    assert all(e["sites"][0] == 0 for e in red["excluded"])
    # the hoppings break the inversion coset of m-3, as they do undeclared, and
    # nothing else: p_y is what they carry, and so are p_z and p_x where the
    # threefolds carry it
    assert (cubic.group.name, cubic.group.number_str, cubic.group.size) == (
        "P231'", "195.2", 24)
    d = cubic.descent
    assert d["from"]["number_str"] == "200.15" and "declaration" not in d["dropped_by"]
    assert d["steps"][0]["refuted_by"] == {"no symmetry matrix": 24}
    rows = cubic.rep.characters.orbitals
    assert cubic.rep.characters.irreps == ("B2", "B1", "B3", "B1", "B3", "B2")
    assert rows[0].declared_irrep == rows[0].irrep
    assert [o.determined_by for o in rows] == ["declaration"] + ["hamiltonian"] * 5
    assert all((o.wyckoff, o.site_symmetry) == ("3c", "222") for o in rows)
    assert {k: complex(v).real for k, v in rows[0].generator_character.items()} == \
        pytest.approx({"E": 1.0, "C2(z)": -1.0, "C2(y)": 1.0, "C2(x)": -1.0}), \
        "p_y's B2 of 222, which no one-dimensional character of T restricts to"

    # two largest groups of one order: the tie rule keeps one, and says why
    stacked = tqg.realized_group(_stacked_pair(), characters={0: "px"})
    assert stacked, str(stacked)
    red = _reduction(stacked)
    assert red["from"]["number_str"] == "123.340"
    assert red["to"] == {"number_str": "115.284", "name": "P-4m21'", "size": 16}
    assert _by_reason(red) == {
        turned: primed({"C4(z)", "C4^-1(z)", "m([110])", "m([1-10])"}),
        _declaration._OUTSIDE_THE_TIE: primed({"C2(x)", "C2(y)", "i", "m(z)"})}
    note = next(e["note"] for e in red["excluded"] if e["name"] == "i")
    assert "{E, C2(z), C2(y), C2(x), i, m(z), m(y), m(x)}" in note
    assert "the tie rule" in note and "keeps the first since" in note


@pytest.mark.slow
def test_a_declared_function_whose_orbit_is_the_multiplet_reduces_nothing(realized):
    """``px`` and ``py`` on the **pair** at a ``3m`` site is the doublet itself.

    The functions declared at a site generate an orbit under its site group;
    where that orbit is as wide as the orbitals they were declared for, what
    was declared *is* the orbit — and an orbit is invariant, so nothing is
    reduced.  The ``p_x, p_y`` honeycomb declared orbital by orbital is
    therefore the same crystal as the undeclared one, down to the matrices,
    with every row now the declaration's.
    """
    # the undeclared honeycomb is the session's shared record of that call
    bare = realized(tqm.pxpy_honeycomb)
    out = tqg.realized_group(tqm.pxpy_honeycomb(), cross_check=False,
                             characters={0: "px", 1: "py", 2: "px", 3: "py"})
    assert bare and out, (str(bare), str(out))
    assert out.group.number_str == "183.186" and out.group.size == 24
    assert _reduction(out) is None and out.descent is None
    assert out.rep.characters.irreps == ("E",) * 4
    assert [o.determined_by for o in out.rep.characters.orbitals] == ["declaration"] * 4
    assert [o.function for o in out.rep.characters.orbitals] == ["px", "py", "px", "py"], \
        "the words written, orbital by orbital, on a two-dimensional multiplet"
    assert [o.function for o in bare.rep.characters.orbitals] == [""] * 4
    assert max(float(np.abs(np.asarray(bare.rep.matrices[i])
                            - np.asarray(out.rep.matrices[i])).max())
               for i in bare.rep.indices) == 0.0


def test_descend_false_does_not_turn_off_a_cell_reduction():
    """``descend=False`` asks for the group the **cell** allows — reduction and all.

    A step 1 reduction is not a descent: it says which operations the declared
    crystal has, which is a question about the cell, and ``descend=False`` is
    the instruction not to lower a group the *Hamiltonian* breaks.  So the
    reduced cell is still the cell, and the refusal that follows names the
    subgroup of **it** the hoppings leave.
    """
    out = tqg.realized_group(_km(), characters={0: "A1", 1: "A2"}, descend=False)
    assert not out and out.gate == "declaration_breaks_group", str(out)
    assert out.context["number_str"] == "156.50", \
        "the cell the declaration left, not the one no `typat` allows"
    assert out.context["subgroup"] == (1, 2, 3, 7, 8, 9), "the six operations of P31'"
    assert "descend=False" in out.reason


@pytest.mark.slow
def test_a_joint_shorthand_of_the_wrong_dimension_suggests_a_valid_spelling():
    """``{(0, 1): "px"}`` where ``px`` is one-dimensional, and the way out.

    The joint form names the content its orbitals hold together, so it is a
    grammar error wherever that content is not of the dimension of the
    orbitals it names — at an ``mm2`` site ``p_x`` spans ``B1`` alone, and one
    ``B1`` is not what two orbitals hold.  (Where the site group *does* make
    the shorthand a doublet, the joint form names that doublet:
    ``{(0, 1): "dz2"}`` at a ``23`` site is P4₂32's paper's own spelling,
    pinned in ``test_characters``.)
    Every spelling a message offers has to be one this package accepts, which
    is what P2 fixed: the ``{(0,): 'px'}`` a single-orbital site used to be
    offered was never a spelling at all.  What a site holds is counted in
    Wannier functions where a spinor basis is declared, so a doublet written on
    one of them is offered the contents one Wannier function holds rather than
    the other spelling of the same doublet, and only contents that fit the site
    are listed.
    """
    out = tqg.realized_group(_rect_pair(), characters={(0, 1): "px"})
    assert not out and out.gate == "ambiguous_multiplet", str(out)
    assert "the joint form names the content its orbitals hold together" in out.reason
    assert "1-dimensional against the 2 orbitals named" in out.reason
    assert "{0: 'B1', 1: 'B1'}" in out.reason
    assert tqg.realized_group(_rect_pair(), cross_check=False,
                              characters={0: "B1", 1: "B1"}), \
        "the offered spelling is one this package accepts"

    # the invalid one: a site carrying one orbital was told to name two of them;
    # it is offered what one orbital with spin holds, which E and the
    # -1E2E + -E1 it becomes are not
    single = tqg.realized_group(_km(), typat=[0, 1], characters={0: "E"})
    assert not single and single.gate == "ambiguous_multiplet", str(single)
    assert "{(0,)" not in single.reason, "that was never a spelling"
    assert ("no 2-dimensional multiplet sits on it at all: name a content of one orbital "
            "instead (A1, A2, -E1).") in single.reason
    for word in single.reason.split("instead (")[1].split(")")[0].split(", "):
        assert tqg.realized_group(_km(), typat=[0, 1], cross_check=False,
                                  characters={0: word}), f"the offered {word!r} is accepted"
    # ... and -1E2E, which E becomes only beside -E1, is no content of one orbital
    lone = tqg.realized_group(_km(), typat=[0, 1], characters={0: "-1E2E"})
    assert not lone and lone.gate == "unknown_label", str(lone)
    assert "'-1E2E' is not a site irrep times the spinor, or a sum of them." in lone.reason
    # ... and E in its coreps is named as written, counted in orbitals
    doublet = tqg.realized_group(_km(), typat=[0, 1], characters={0: "-1E2E + -E1"})
    assert not doublet and doublet.gate == "ambiguous_multiplet", str(doublet)
    assert "orbital 0: '-1E2E + -E1' holds 2 orbitals, and the 1 orbital" in doublet.reason

    # a declared spinor basis counts Wannier functions: a doublet written on one
    # of them, by one orbital or by the pair, is told what one Wannier function
    # holds, and neither spelling sends the reader to the other
    flat = tqsp.spinless_presentation(_km())[0]
    for words in ({0: "E"}, {(0, 1): "E"}):
        out = tqg.realized_group(flat, typat=[0, 0, 1, 1], spin="interleaved",
                                 characters=words)
        assert not out and out.gate == "ambiguous_multiplet", str(out)
        assert ("Site 0 carries 1 Wannier function, the spin pair (0, 1), so no multiplet "
                "of 2 Wannier functions sits on it at all: name a content of one Wannier "
                "function instead (A1, A2, -E1).") in out.reason, out.reason
        assert "{0: 'E'}" not in out.reason and "{(0, 1): 'E'}" not in out.reason
    # ... and where the site holds two, the joint form names one orbital of each:
    # one site of a square lattice carrying two spinor Wannier functions
    one = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]],
                          periodic_dirs=[0, 1]), spinful=True)
    one.set_onsite([0.3, -0.3], mode="add")
    for R in ([1, 0], [0, 1]):
        one.set_hop(-1.0, 0, 0, R, mode="add")
        one.set_hop(-0.5, 1, 1, R, mode="add")
        one.set_hop(0.2, 0, 1, R, mode="add")
        one.set_hop(0.2, 1, 0, R, mode="add")
    two = tqsp.spinless_presentation(one)[0]
    pair = tqg.realized_group(two, spin="interleaved", characters={0: "E"})
    assert not pair and "e.g. {(0, 2): 'E'}" in pair.reason, str(pair)
    assert tqg.realized_group(two, spin="interleaved", cross_check=False,
                              characters={(0, 2): "E"}), "the offered spelling is accepted"


# ---------------------------------------------------------------------------
# 8.  which orbital a declaration speaks for
# ---------------------------------------------------------------------------
#
# The hoppings fix how the orbitals they couple transform relative to one
# another, so a declaration on any orbital states what every orbital coupled to
# it is: one one-dimensional character per coupled part of the crystal, which
# every declaration in the part constrains alike.  An undeclared orbital, the
# reference orbital included, constrains nothing, and the lowest-``l`` rule
# chooses among the characters the declarations leave, reading the reference
# orbital's site first (lin3g's ``d_x²₋y²`` at ``1a`` is ``B1`` of ``4mm``
# relative to the ``s``).  Declarations of one part that no character meets
# together break the operations they disagree on, and the group descends,
# exactly as any declaration-versus-hoppings conflict does.


def _px_at_2c():
    """``s`` at ``1a`` and one ``p_x`` at each ``2c`` site, spinless, square.

    A model **built to carry p_x**, with the hoppings a ``p_x`` has:

    * ``1a -> 2c`` across the ``x`` bond is the ``s``-``p`` σ overlap, which is
      **odd** along the bond: ``+t`` to ``(+½, 0)`` and ``-t`` to ``(-½, 0)``.
    * ``1a -> 2c`` across the ``y`` bond is the π one, and an ``s`` has no π
      overlap with a ``p`` perpendicular to the bond: it is zero.  The mirror
      says the same thing — ``m(x)`` carries that bond to itself while ``p_x``
      at ``(0, ½)`` is odd under it and ``s`` is even, so any amplitude there
      would be a crystal without ``m(x)``.
    * ``2c -- 2c`` across a lattice vector is σ along ``x`` and π along ``y``,
      the same pair of numbers at **both** sites, which is what makes this the
      nematic crystal rather than the vector one.  The four ``2c -- 2c``
      diagonals carry one amplitude between them — for two ``p_x`` orbitals on a
      diagonal bond ``t_σ n_x² + t_π n_y²`` is the same number on all four — and
      that is what ties the second ``2c`` orbital's character to the first's
      instead of leaving it a free block.

    The fourfold carries ``p_x`` at ``(½, 0)`` onto ``p_y`` at ``(0, ½)``, so
    these hoppings are **not** ``C4``-covariant: undeclared, the model descends
    from ``P4mm1' #99.164`` to ``Pmm21' #25.58`` on ``"no symmetry matrix"``.  That is
    the point of the pair of tests below: the declaration ``{0: "s", 1: "px",
    2: "px"}`` says so *before* a hopping is read (a cell statement, no descent),
    and ``{0: "s", 1: "s", 2: "s"}`` says the opposite and breaks the operations
    the parities differ on.
    """
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]],
                        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
                        periodic_dirs=[0, 1]), spinful=False)
    m.set_onsite([0.0, 1.5, 1.5])
    m.set_hop(+0.6, 0, 1, [0, 0])           # s -> p_x, sigma, +x
    m.set_hop(-0.6, 0, 1, [-1, 0])          # ... and -x: the other sign
    for o in (1, 2):
        m.set_hop(-0.9, o, o, [1, 0])       # p_x -- p_x along x: sigma
        m.set_hop(+0.3, o, o, [0, 1])       # p_x -- p_x along y: pi
    for lv in ([0, 0], [0, -1], [1, 0], [1, -1]):
        m.set_hop(0.2, 1, 2, lv)            # the four 2c diagonals
    return m


def _triclinic(origin=True):
    """One or two orbitals in a triclinic cell — the site groups ``-1`` and ``1``."""
    at = [[0.0, 0.0, 0.0]] if origin else [[0.21, 0.33, 0.13], [0.62, 0.17, 0.44]]
    m = TBModel(Lattice([[1.0, 0.0, 0.0], [0.27, 1.13, 0.0], [0.19, 0.11, 1.41]],
                        at, periodic_dirs=[0, 1, 2]), spinful=False)
    for o in range(len(at)):
        m.set_hop(-1.0 + 0.1 * o, o, o, [1, 0, 0])
        m.set_hop(-0.4, o, o, [0, 1, 0])
        m.set_hop(-0.2, o, o, [0, 0, 1])
    return m


@pytest.mark.slow
def test_a_declaration_on_either_sublattice_states_the_same_crystal(realized):
    """``{1: "A2"}`` on Kane-Mele is the global ``A2``: nothing descends.

    The two sublattices are measured to carry the *same* parity under the three
    mirrors, so declaring the second ``A2`` states that the first is ``A2`` as
    well: the hoppings couple them, and one one-dimensional character assigns
    both.  That is the crystal ``{0: "A2", 1: "A2"}`` states, ``P3m11' #156.50``
    whole, with the same matrices; the undeclared reference constrains nothing
    and is read relative to the declared sublattice.  The same label on both
    sublattices is a **twist**, not a broken mirror: ``A2`` is a real character
    of the whole group, and it is what the matrices carry -- every mirror's
    matrix is minus the undeclared crystal's.

    In a group with no antiunitary operation every one-dimensional character
    is such a labelling, the complex ones included.  Haldane's model with a
    sublattice mass and its sublattices declared different atoms is ``P3 #143``,
    time reversal broken; declared ``{0: "1E"}`` it stays ``P3``, both rows
    ``1E``, orbital 1 fixed by the hoppings relative to orbital 0: the crystal
    ``{0: "1E", 1: "1E"}`` states, with the same matrices.  ``1E`` is an irrep
    of the substrate's ``3`` and of nothing in the free-standing layer's ``-6``,
    so the two environments are not compared.
    """
    out = tqg.realized_group(_km(), typat=[0, 1], cross_check=False, characters={1: "A2"})
    both = realized(_km, typat=[0, 1], cross_check=False, characters={0: "A2", 1: "A2"})
    assert out and both, (str(out), str(both))
    assert (out.group.number_str, out.group.name, out.group.size) == ("156.50", "P3m11'", 12)
    assert out.descent is None and len(out.validated.indices) == 12
    assert both.group.number_str == "156.50" and both.descent is None
    assert out.rep.characters.irreps == both.rep.characters.irreps == ("-E1", "-E1")
    assert [o.determined_by for o in out.rep.characters.orbitals] == \
        ["hamiltonian", "declaration"], "the reference read relative to the declared row"
    assert [o.determined_by for o in both.rep.characters.orbitals] == ["declaration"] * 2
    frame = out.rep.characters.frame
    mirror = next(ind for ind, name in frame.items() if name == "m(x)")
    assert complex(out.rep.characters.orbitals[0].character[mirror]).real == \
        pytest.approx(-1.0), "orbital 0 is A2 too: odd under the mirror"
    assert max(float(np.abs(np.asarray(out.rep.matrices[i])
                            - np.asarray(both.rep.matrices[i])).max())
               for i in out.rep.indices) < 1e-14, "one crystal, one representation"
    # the declared A2 is a character of the whole group, and it is what the
    # matrices carry: every mirror's matrix is minus the undeclared one
    plain = realized(_km, typat=[0, 1], cross_check=False)
    assert plain, str(plain)
    for ind in both.rep.indices:
        if both.rep._operation(ind).time_reversal:
            continue
        name = both.rep.characters.frame[int(ind)]
        sign = -1.0 if name.startswith("m(") else 1.0
        assert np.abs(np.asarray(both.rep.matrices[ind])
                      - sign * np.asarray(plain.rep.matrices[ind])).max() < 1e-12, name

    one, two = (tqg.realized_group(haldane(0.5, 1.0, 0.15), typat=[0, 1], characters=words)
                for words in ({0: "1E"}, {0: "1E", 1: "1E"}))
    assert one and two, (str(one), str(two))
    for r in (one, two):
        assert (r.group.number_str, r.group.name, len(r.validated.indices)) == ("143", "P3", 3)
        assert not _names(r.descent), "the declaration breaks nothing"
        assert r.rep.characters.irreps == ("1E", "1E")
        w = r.cross_check
        assert (w["outcome"], w["free_standing"], w["substrate"]["number_str"]) == \
            ("not compared", None, "143")
    assert [o.determined_by for o in one.rep.characters.orbitals] == \
        ["declaration", "hamiltonian"]
    assert max(float(np.abs(np.asarray(one.rep.matrices[i])
                            - np.asarray(two.rep.matrices[i])).max())
               for i in one.rep.indices) < 1e-14, "one crystal, one representation"


def test_a_site_group_with_no_rotation_axis_is_named_by_its_own_parity():
    """The declaration vocabulary of a site group with no rotation axis.

    An orbital character is named ``A`` or ``B`` from the character of the
    principal rotation axis, and three site groups have none: the trivial group,
    ``-1`` and ``m``.  The first two are named by dimension and by ``chi(i)``,
    as every table prints them; ``m``'s one reflection is its own parity
    reference, ``A'`` and ``A''``, which the one-``p_x`` crystal above reads.

    In ``P11'`` no little group holds more than the identity, so an orbital's
    symmetry content is read at no point; its band count alone fixes the sum,
    and each orbital induces its own ``A`` at ``1a``.
    """
    inv = tqg.realized_group(_triclinic(), nk=3)
    assert inv, str(inv)
    assert inv.group.name == "P-11'"
    assert inv.rep.characters.orbitals[0].site_symmetry == "-1"
    assert set(inv.rep.characters.site_tables[0].labels) == {"Ag", "Au"}

    plain = tqg.realized_group(_triclinic(origin=False), typat=[0, 1], nk=3)
    assert plain, str(plain)
    assert plain.group.name == "P11'"
    assert plain.rep.characters.site_tables[0].labels == ("A",)
    assert [(o.orbitals, o.terms, o.note)
            for o in plain.rep.characters.band_representations] == \
        [((0,), ((1, "A↑G(1)", "1a(11',1)"),), ""), ((1,), ((1, "A↑G(1)", "1a(11',1)"),), "")]

    for rec in (inv.rep.characters, plain.rep.characters):
        assert not [lab for t in rec.site_tables.values() for lab in t.labels
                    if "_" in lab], "no site group is served a generic suffix"


@pytest.mark.slow
def test_px_declared_across_the_2c_orbit_splits_the_cell_and_nothing_descends():
    """The nematic crystal, on a model whose hoppings are the ones a ``p_x`` has.

    ``C4`` carries the declared ``p_x`` at ``(½, 0)`` onto ``p_y`` at ``(0, ½)``,
    which is not what that site was declared to carry, so the fourfold and the
    two diagonal mirrors are not operations of the declared crystal at all: the
    cell is ``Pmm21' #25.58``, measured before a hopping is read.  The hoppings
    then obey every operation that is left — they were built to — so there is no
    descent, and the representation is the undeclared model's to the bit.
    """
    bare = tqg.realized_group(_px_at_2c(), typat=[0, 1, 1], cross_check=False)
    out = tqg.realized_group(_px_at_2c(), typat=[0, 1, 1], cross_check=False,
                             characters={0: "s", 1: "px", 2: "px"})
    assert bare and out, (str(bare), str(out))
    assert (out.group.name, out.group.number_str, out.group.size) == ("Pmm21'",
                                                                      "25.58", 8)
    assert out.group.provenance == "realized" and out.descent is None

    red = _reduction(out)
    assert red["from"] == {"number_str": "99.164", "name": "P4mm1'", "size": 16}
    assert red["n_excluded"] == 8
    assert {e["name"] for e in red["excluded"]} == {
        "C4(z)", "C4^-1(z)", "m([110])", "m([1-10])",
        "C4(z)'", "C4^-1(z)'", "m([110])'", "m([1-10])'"}
    assert {e["reason"] for e in red["excluded"]} == {
        "declared content differs across the orbit"}
    assert all(e["sites"] == (1, 2) and e["orbitals"] == (1, 2)
               for e in red["excluded"])

    rec = out.rep.characters
    assert rec.irreps == ("A1", "B1", "B1"), "the s, and p_x by its parities"
    assert [o.determined_by for o in rec.orbitals] == ["declaration"] * 3
    for o in rec.orbitals[1:]:
        gen = {k: complex(v).real for k, v in o.generator_character.items()}
        assert gen == pytest.approx({"E": 1.0, "C2(z)": -1.0,
                                     "m(x)": -1.0, "m(y)": 1.0}), \
            "p_x is odd under the twofold and under m(x), even under m(y)"
        assert o.reduced_in == "mm2 in Pmm21' #25.58"
    assert max(float(np.abs(np.asarray(bare.rep.matrices[i])
                            - np.asarray(out.rep.matrices[i])).max())
               for i in out.rep.indices) == 0.0, \
        "the declaration states the crystal these hoppings already are"


@pytest.mark.slow
def test_the_same_px_hoppings_declared_s_break_what_the_parities_differ_on():
    """The negative control: ``{0: "s", 1: "s", 2: "s"}`` on the same model.

    Three ``s`` orbitals is one content across the ``2c`` orbit, so the cell is
    not reduced at all and the whole of ``P4mm1' #99.164`` gets a verdict.  The
    hoppings break the four operations that exchange the two ``2c`` sites
    (``"no symmetry matrix"``: the ``1a -> 2c`` bond along ``x`` is not the one along
    ``y``), and the **declaration** breaks the two the parities differ on:
    ``p_x``'s bond is odd under ``C2(z)`` and under ``m(x)`` while an ``s`` at
    ``1a`` is even under both, and the reference's own ``s`` anchors the global
    character at ``+1``, so no twist can absorb it.  What is left is the one
    mirror both parities agree on, ``Pm1' #6.19``, and there every row is even.
    """
    out = tqg.realized_group(_px_at_2c(), typat=[0, 1, 1], cross_check=False,
                             characters={0: "s", 1: "s", 2: "s"})
    assert out, str(out)
    assert (out.group.name, out.group.number_str, out.group.size) == ("Pm1'",
                                                                      "6.19", 4)
    assert _reduction(out) is None, "one content across the orbit reduces nothing"

    d = out.descent
    assert d["from"]["number_str"] == "99.164" and len(d["dropped"]) == 12
    assert d["steps"][0]["refuted_by"] == {"declaration": 4, "no symmetry matrix": 8}
    assert _names(d) == {"C2(z)", "m(x)", "C2(z)'", "m(x)'"}
    assert _names(d, "no symmetry matrix") == {
        "C4(z)", "C4^-1(z)", "m([110])", "m([1-10])",
        "C4(z)'", "C4^-1(z)'", "m([110])'", "m([1-10])'"}
    note = next(b["note"] for b in d["dropped"] if b["name"] == "m(x)")
    assert "the hoppings carry orbitals (1,) with -1" in note and "state +1" in note, \
        "the note says what was declared and what the hoppings carry"

    rec = out.rep.characters
    assert rec.irreps == ("A'", "A'", "A'"), "the kept mirror is m(y), even on all three"
    assert [o.determined_by for o in rec.orbitals] == ["declaration"] * 3
    assert [o.site_symmetry for o in rec.orbitals] == ["m"] * 3
