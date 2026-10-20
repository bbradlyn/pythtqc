r"""Phase A3a regressions: declared orbital characters (``_build_rep(characters=)``).

What this suite closes, and why each number is the one that closes it:

* **undeclared, every orbital is assigned by the lowest-``l`` rule.**
  ``characters=None`` reproduces the representation ``_build_rep`` built before
  the option existed — asserted on the matrices, and on the corep census through
  ``tests/test_baseline_r4.py``, whose ``r1/lin3g`` record pins
  ``{-GM6: 2, -GM7: 1}`` at Gamma, ``{-M6: 1, -M7: 2}`` at M and ``{-X5: 3}``
  for the declared build this file measures the default against bit for bit —
  and each row says whether its site irrep rests on that rule
  (``determined_by``).
* **lin3g's own physics is that labelling.**  Declaring the fixture's stated
  orbital content — ``s`` and ``d_x2-y2`` on the ``1a`` quartet, ``s`` on the
  ``2c`` pair — changes nothing: the model's hoppings and the caller's physics
  agree, which is the honest way to find that out.
* **a B-type declaration is another labelling.**  Declaring orbital 0 to be
  ``B1`` -- or orbital 1 to be ``A1``, the same statement, since the hoppings
  couple the two -- reproduces ``reference_orbital=1`` at machine precision,
  and under that reference ``-GM6`` and ``-GM7`` swap wholesale, partitions
  untouched: a relabelling, stated as physics instead of an index.
* **the returned matrices are induced from the returned content**: every
  site block's trace is its rows' character, and every block between sites is
  the model's own transport, a member of the operation's solution space.
* **the refusal gates**: ``unknown_label`` (with the site group's own valid
  labels and the generator frame), ``ambiguous_multiplet`` (a declaration that
  cannot say which orbitals form a multiplet, or names one of the wrong
  dimension), ``declaration_breaks_group`` (with the measured-versus-declared
  table and the subgroup the content *is* covariant on); and a declaration
  outside the grammar, which raises ``ValueError`` before anything is
  measured.
* **a declaration the hoppings do not respect is a smaller group, not an
  error.**  A site *position* allows every irrep of its site group as often as
  one likes; the *Hamiltonian* does not.  Declaring both of lin3g's ``1a``
  orbitals ``s`` is position-legal, and what it describes is a crystal without
  the fourfold: ``_build_rep`` is handed a fixed group and cannot lower one, so
  it names the ``mm2`` subgroup ``realized_group`` descends to and prints the
  measured relative character (``B1``) that says why.  Every descent this
  implies is pinned in ``test_declared_content.py``.
* **naming is not C4v-only**: Kane-Mele's ``3m`` sites and P4_232's ``23``
  sites are exercised.
* **the names are the tables' own.**  A row of a computed site table is named
  by the band representation it induces, matched against the EBR entries at the
  site's Wyckoff position, so the last section here is about that: the
  diamond's ``8a`` reads ``A1 A2 E T1 T2`` and takes declarations in those
  words; a Wyckoff position whose entries share a Gamma content is separated at
  the other high-symmetry points; and the induction itself is checked against
  every EBR column of every position the baseline gate's fixtures sit at, in
  both valuednesses, which is content against content and reads no name.
* **the generator-character dict is naming-free** and reproduces the label
  form on two different site-group families.

The one thing the machinery cannot catch is a *mis-oriented* declaration
(``B1`` where the model means ``B2``): it satisfies every gate and relabels
every corep at once.  That is what the echo record exists for, so the
structural tests here assert that it covers every orbital and prints the frame.
"""

import dataclasses
import itertools
import re

import numpy as np
import pytest
from pythtb import Lattice, TBModel

from pythtqc import _sitechars as tqc_sitechars
from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import sewing as tqw
from pythtqc import symmetry as tqs
from pythtqc.models import cubic_3c_axis_p, cubic_3c_perp_doublet

import tqc_models
# The oracle below runs on exactly the models the baseline gate builds, so the
# builders come from that module rather than being written out a second time:
# two copies of a fixture recipe are two fixtures the day one of them is edited.
import test_baseline_r4 as _baseline
# the jitter a numerical model's noise is modelled by, the one the symmetrizing
# pass's own tests use
from test_symmetrize import jitter

GATE = tqs.EXACT_TOL   # the exact floor of _build_rep's covariance gate


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------

C_SLAB = 30.0                      # slab lattice constant of the 3D presentation


def realized_bundle_2d(model, typat, **kwargs):
    """``(model3d, group, assignment, validated, rep)`` in the *realized* group.

    Phase A4 promoted this recipe into ``tqc.groups`` — ``_slab_presentation``
    (which now *measures* the bit-identity of the re-presented ``H(k)``),
    ``_decorated_cell`` (two different species at ``(0, 0, ±dz)``, removing the
    ``σ_h``-composed operations a flat orbital point set only *looks*
    symmetric under) and ``realized_group``, which chains them and whose
    result unpacks as exactly the tuple these tests destructure.  This wrapper
    is all that is left of the local copy.
    """
    out = tqg.realized_group(model, typat=typat, c=C_SLAB, **kwargs)
    assert out, str(out)
    return out


#: One ``solve_ham`` at one k -> the ``(evals, evecs)`` pair sewing takes.  The
#: library's own, so a test and a caller feed the readouts the same object.
solve_at = tqw.solve_at


def rebuild(bundle, **kwargs):
    """The same representation, with the declarations under test."""
    model, group, assignment, validated, _ = bundle
    return tqs._build_rep(model, group, assignment, validated=validated, **kwargs)


def same_matrices(a, b, atol=0.0):
    """``max |a - b|`` over the represented operations, against ``atol``.

    The default is **exactly zero**, because every positive comparison in this
    file is two builds of the same model in the same process and the twist is a
    multiplication by exactly ``±1`` that ``_build_rep`` skips altogether when it
    is trivial: the matrices are bit-identical, and "bit for bit" is what the
    tests below say.  The one comparison that wants slack is the *negative*
    control, which passes ``atol=1e-9`` explicitly against a measured 2.0.
    """
    return max(float(np.abs(a.matrices[i] - b.matrices[i]).max()) for i in a.indices) <= atol


@pytest.fixture(scope="module")
def km_realized(km_odd):
    """Kane-Mele (Z2-odd) in the group it realizes: P3m11' (#156.50)."""
    return realized_bundle_2d(km_odd, [0, 1])


# ---------------------------------------------------------------------------
# 1.  the default is the lowest-l rule
# ---------------------------------------------------------------------------

def test_undeclared_orbitals_follow_the_lowest_l_rule_and_say_so(lin3g_undeclared):
    """The lowest-``l`` rule, pinned as a contract rather than as an accident.

    The hoppings fix how the orbitals transform relative to one another, not
    what they are, so without a declaration a labelling has to be *chosen*:
    the rule puts every orbital at the lowest angular momentum the
    Hamiltonian's symmetry allows, and among what that leaves the letter the
    orbitals read as, then the band labels, and last the group's order of
    operations decide.  The reference orbital therefore reads **s-like**.

    What that choice does *not* touch is the orbitals' content relative to one
    another: orbital 1 is undeclared and comes back ``B1`` -- ``d_x2-y2`` --
    because the hoppings say so.  Which rows the choice decides is on the
    record: the ``1a`` pair's site coreps differ between labellings (``-E1``
    against ``-E2``), so both rows are ``lowest-l``; the ``2c`` sites' ``mm2``
    has one spinor corep, ``-E``, which every labelling gives them, so theirs
    is ``hamiltonian``.
    """
    rep = lin3g_undeclared.rep
    rec = rep.characters

    # (1) the rows: the atomic-limit table, and which rows the rule decides
    assert max(rep.residuals.values()) <= GATE
    assert rec.residual < 1e-12, "the blocks factorize into character x internal factor"
    assert rec.irreps == ("-E1", "-E2", "-E", "-E")
    assert [o.function for o in rec.orbitals] == [""] * 4, \
        "an undeclared row names no function: its irrep and characters say what it is"
    assert [o.determined_by for o in rec.orbitals] == \
        ["lowest-l", "lowest-l", "hamiltonian", "hamiltonian"]
    assert [o.wyckoff for o in rec.orbitals] == ["1a", "1a", "2c", "2c"]
    assert rec.orbitals[int(rep._reference["orbital"])].irrep == "-E1"

    # (2) deterministic, and the report says what the labelling is rather than
    # hiding it
    again = rebuild(lin3g_undeclared)
    assert again, str(again)
    assert same_matrices(again, rep)
    assert again.characters.orbitals == rec.orbitals
    text = rec.summary()
    assert tqs._LOWEST_L_LEGEND in text, "the one sentence of legend, verbatim"
    assert "the hoppings fix how the orbitals transform relative to one another" in text
    assert "lowest-l" in repr(rec)
    assert "global character" not in text and "twist" not in text


@pytest.mark.slow
def test_declaring_nothing_and_declaring_the_truth_agree_bit_for_bit(
        lin3g_undeclared, lin3g_declared_cross_checked):
    """lin3g's own orbital content *is* the default convention.

    The fixture says orbitals 0,1 carry the quartet at ``1a`` and orbitals 2,3
    the ``s`` EBR at ``2c``.  Declaring exactly that reproduces the default
    representation to machine precision and every declaration comes back
    consistent — the model's hoppings and the caller's physics agree.  The
    ``function`` column echoes the function words declared, orbital by orbital,
    and nothing else: an irrep label names no function.  End to end, through
    ``realized_group``, the declaration ``["s", "dx2-y2", "s", "s"]`` returns the
    undeclared crystal's group and its matrices, entry for entry.
    """
    _, _, _, _, rep = lin3g_undeclared
    for spec in ({2: "A1"}, {2: "s"}, {3: "s"}, ["s", "dx2-y2", "s", "s"],
                 ["A1", "B1", "A1", "A1"], {1: "dx2-y2"}, {1: "B1"}):
        out = rebuild(lin3g_undeclared, characters=spec)
        assert out, f"{spec}: {out}"
        assert same_matrices(rep, out), f"{spec} moved the representation"
        assert out.characters.irreps == ("-E1", "-E2", "-E", "-E")
        written = spec if isinstance(spec, list) else [spec.get(o) for o in range(4)]
        assert [o.function for o in out.characters.orbitals] == \
            [tqc_sitechars.canonical_shorthand(w) for w in written], \
            f"{spec}: the declared function words, and nothing undeclared"

    declared = lin3g_declared_cross_checked
    assert declared.group.number_str == lin3g_undeclared.group.number_str
    assert set(declared.rep.matrices) == set(rep.matrices)
    assert same_matrices(rep, declared.rep), "the declaration moved a matrix"


@pytest.mark.slow
def test_a_declaration_fixes_the_labelling_as_far_as_its_site_symmetry_reaches(
        lin3g_undeclared):
    """What a declaration settles is every labelling it rules out, and no more.

    ``{2: "s"}`` is true, and exactly what the hoppings measure at ``2c``
    relative to an ``s``-like orbital 0; but the ``mm2`` of ``2c`` cannot tell
    the ``s`` labelling of the ``1a`` pair from the ``B1`` one, both trivial on it,
    so the ``1a`` rows stay ``lowest-l``.  A declaration at the ``4mm`` site --
    on the reference or on its partner -- sees every character of the group,
    and then every undeclared row is fixed by the Hamiltonian relative to it.
    Naming the reference elsewhere (``reference_orbital=2``) changes nothing a
    declaration says about its own row.
    """
    away = rebuild(lin3g_undeclared, characters={2: "s"})
    assert away, str(away)
    assert [o.determined_by for o in away.characters.orbitals] == \
        ["lowest-l", "lowest-l", "declaration", "hamiltonian"], \
        "the 2c declaration leaves the 1a labelling open"
    assert away.characters.orbitals[2].note.startswith("'s' resolved to A1"), \
        "a declaration is echoed in the words it was written in"

    for spec in ({0: "dx2-y2"}, {1: "dx2-y2"}):
        fixed = rebuild(lin3g_undeclared, characters=spec)
        assert fixed, str(fixed)
        rows = fixed.characters.orbitals
        assert [o.determined_by for o in rows] == \
            ["declaration" if o.orbital in spec else "hamiltonian" for o in rows], spec

    anchored = rebuild(lin3g_undeclared, characters={2: "s"}, reference_orbital=2)
    assert anchored, str(anchored)
    assert anchored.characters.orbitals[2].determined_by == "declaration"

    out = rebuild(lin3g_undeclared, characters=["s", "dx2-y2", "s", "s"])
    assert out, str(out)
    assert {o.determined_by for o in out.characters.orbitals} == {"declaration"}
    assert all("status" not in d for d in out.characters.declarations)
    # the checks table carries the measured-vs-declared comparison the verdict
    # made.  One row per (declaration, operation it constrains) -- and
    # `resolve_spec` constrains every operation of the site group, not one per
    # class
    assert len(out.characters.checks) == 24, \
        "8+8 at the 4mm site, 4+4 at the two mm2 sites"
    for row in out.characters.checks:
        assert abs(row["measured"] - row["declared"]) < 1e-9
        assert set(row) == {"orbitals", "ind", "name", "measured", "declared"}


def _pxpy_with_s(s_first):
    """The honeycomb ``(p_x, p_y)`` pairs with an ``s`` orbital at the hexagon centre.

    The pairs and their hoppings are ``tqc_models.pxpy_honeycomb``'s own; the
    ``s`` at ``1a`` is σ-bonded to the six nearest ``p`` sites, ``0.7`` times
    the bond direction's component along each ``p``, with an on-site energy
    ``0.3``.  ``s_first`` lists the ``s`` orbital first instead of last: the
    same crystal, its orbitals in another order.
    """
    base = tqc_models.pxpy_honeycomb()
    lat = np.asarray(tqc_models.PXPY_LATTICE, dtype=float)
    r_a = np.asarray(tqc_models.PXPY_SITES["A"], dtype=float)
    r_b = np.asarray(tqc_models.PXPY_SITES["B"], dtype=float)
    shift, s = (1, 0) if s_first else (0, 4)
    orbs = [list(r_a), list(r_a), list(r_b), list(r_b)]
    orbs.insert(s, [0.0, 0.0])
    model = TBModel(Lattice(lat.tolist(), orbs, periodic_dirs=[0, 1]), spinful=False)
    onsite = [0.0] * 5
    onsite[s] = 0.3
    model.set_onsite(onsite, mode="set")
    for h in base.hoppings:
        model.set_hop(complex(h["amplitude"]), int(h["from_orbital"]) + shift,
                      int(h["to_orbital"]) + shift,
                      [int(x) for x in h.get("lattice_vector", [0, 0])], mode="add")
    pairs = {"A": (shift, shift + 1), "B": (shift + 2, shift + 3)}
    for cell in itertools.product(range(-2, 3), repeat=2):
        for name, r in (("A", r_a), ("B", r_b)):
            bond = (r + np.asarray(cell)) @ lat
            if abs(float(np.linalg.norm(bond)) - 1.0 / np.sqrt(3.0)) > 1e-6:
                continue
            for n, orb in zip(bond / np.linalg.norm(bond), pairs[name]):
                if abs(n) > 1e-14:
                    model.set_hop(0.7 * float(n), s, orb, list(cell), mode="add")
    return model


def _pxpy_s_last():
    """The σ-bonded ``s`` listed after the four ``p`` orbitals."""
    return _pxpy_with_s(False)


def _pxpy_s_first():
    """The σ-bonded ``s`` listed before the four ``p`` orbitals."""
    return _pxpy_with_s(True)


@pytest.mark.slow
def test_the_honeycomb_doublet_bonded_to_an_s_is_assigned_alike_in_either_order(realized):
    """The honeycomb ``(p_x, p_y)`` pair with an ``s`` bonded to it, in either order.

    The doublet is ``E`` at the ``3m`` site under every labelling -- ``E ⊗ A2 =
    E`` -- so the pair's own site content cannot choose between two labellings
    that differ by a sign on every mirror.  The crystal's content can: with an ``s``
    at the hexagon centre σ-bonded to the pairs, the hoppings pin the ``s`` and
    the pairs to one another, and the ``s`` reads ``A1``, the lowest angular
    momentum at the hexagon centre, whichever orbital is listed first -- the
    reference's own site when the ``s`` comes first, the whole crystal when the
    pair does.  Either way every operation that fixes a site is, on that site's
    pair, the Cartesian matrix of the operation on ``(x, y)`` -- ``+R`` on each
    mirror -- which is what a pair σ-bonded to an ``s`` carries.  The two
    orders are one crystal, and the two atomic-limit tables and representations
    are one another's, orbital for orbital.
    """
    out = realized(_pxpy_s_first, cross_check=False)
    assert out, str(out)
    assert out.group.number_str == "183.186"
    rep, rec = out.rep, out.rep.characters
    for a, b in ((1, 2), (3, 4)):
        assert rec[a].irrep == rec[b].irrep == "E"
        for ind in rec[a].site_group:
            R = np.asarray(rep._operation(ind).rotation_cart, dtype=float)[:2, :2]
            block = np.asarray(rep.matrices[ind])[np.ix_([a, b], [a, b])]
            assert float(np.abs(block - R).max()) < 1e-12, rec.frame[ind]

    assert (rec[0].wyckoff, rec[0].irrep) == ("1a", "A1")
    twin = realized(_pxpy_s_last, cross_check=False)
    assert twin, str(twin)
    perm = {o: o - 1 for o in range(1, 5)} | {0: 4}

    def row(o, rename=lambda x: x):
        return (o.wyckoff, o.site_symmetry, o.function, o.irrep, o.named_by,
                o.determined_by, tuple(sorted(rename(m) for m in o.multiplet)),
                tuple(sorted((k, round(complex(v).real, 9), round(complex(v).imag, 9))
                             for k, v in o.generator_character.items())))

    assert {perm[o.orbital]: row(o, perm.get) for o in rec.orbitals} == \
        {o.orbital: row(o) for o in twin.rep.characters.orbitals}, \
        "the two orders return one atomic-limit table"
    P = np.zeros((5, 5))
    for o, q in perm.items():
        P[q, o] = 1.0
    assert tuple(twin.rep.indices) == tuple(rep.indices)
    assert max(float(np.abs(P @ np.asarray(rep.matrices[i]) @ P.T
                            - np.asarray(twin.rep.matrices[i])).max())
               for i in rep.indices) < 1e-12, "and one representation"


# ---------------------------------------------------------------------------
# 2.  the deliberate twist
# ---------------------------------------------------------------------------

def test_declaring_either_orbital_of_the_pair_is_the_same_twist(lin3g_undeclared):
    """``{0: "B1"}`` and ``{1: "A1"}`` are one statement, and one crystal.

    The relative character between lin3g's two ``1a`` orbitals is measured, and
    it is ``B1``: the hoppings couple them, through the ``2c`` pair, so what one
    of them is fixes what the other is.  ``B1`` on orbital 0 and ``A1`` on
    orbital 1 therefore say the same thing -- orbital 0 is the
    ``d_x2-y2``-like partner -- and both return one representation, every row
    the hoppings relate to the declared one read relative to it.  An undeclared
    orbital imposes nothing, the reference orbital included: which of the two
    was declared moves which row reads ``declaration``, and nothing else.

    The twist is a relabelling asked for in physics rather than as an index:
    ``{0: "B1"}`` and ``reference_orbital=1`` agree at machine precision
    (measured 3e-16; asserted at 1e-14), and under that reference every corep
    label swaps wholesale while the phase-invariant partition stays where it was.
    """
    model, group, assignment, validated, rep = lin3g_undeclared
    a = rebuild(lin3g_undeclared, characters={0: "B1"})
    assert a, str(a)
    assert a.characters.irreps == ("-E2", "-E1", "-E", "-E"), \
        "the declared content is the one returned, and the partner reads relative to it"
    assert [o.determined_by for o in a.characters.orbitals] == \
        ["declaration", "hamiltonian", "hamiltonian", "hamiltonian"]

    other = tqs._build_rep(model, group, assignment, validated=validated, reference_orbital=1)
    assert max(float(np.abs(a.matrices[i] - other.matrices[i]).max())
               for i in rep.indices) < 1e-14
    hsp = tqg.high_symmetry_points(group)
    for name, want, flip in (("GM", {"-GM6": 2, "-GM7": 1}, {"-GM7": 2, "-GM6": 1}),
                             ("M", {"-M6": 1, "-M7": 2}, {"-M7": 1, "-M6": 2}),
                             ("X", {"-X5": 3}, {"-X5": 3})):
        states = solve_at(model, hsp[name])
        before = tqw.corep_content(rep, hsp[name], states=states, bands=6)
        after = tqw.corep_content(other, hsp[name], states=states, bands=6)
        assert before and after
        assert before.counts == want and after.counts == flip, name
        assert before.partition == after.partition, \
            "the phase-invariant content must not move"

    b = rebuild(lin3g_undeclared, characters={1: "A1"})
    assert b, str(b)
    assert b.characters.irreps == a.characters.irreps, "one crystal, whichever was declared"
    assert [o.determined_by for o in b.characters.orbitals] == \
        ["hamiltonian", "declaration", "hamiltonian", "hamiltonian"], \
        "the undeclared reference is read relative to the declared partner"
    assert max(float(np.abs(a.matrices[i] - b.matrices[i]).max())
               for i in rep.indices) < 1e-14


# ---------------------------------------------------------------------------
# 3.  the refusal gates
# ---------------------------------------------------------------------------

def test_unknown_label_names_the_site_group_and_prints_the_frame(lin3g_undeclared,
                                                                monkeypatch):
    """A label from the wrong group refuses with what *this* site group offers.

    Which words a site takes depends on the site group the measurement reaches,
    so a word naming nothing there is a refusal.  A declaration outside the
    grammar, such as a key that names no orbital or names one twice, a list of
    the wrong length, or a value that is neither a word nor a mapping of
    characters, raises ``ValueError`` before anything is measured, in
    ``realized_group`` as in ``_build_rep``.
    """
    for bad, says in (("s", "given the str 's'"),
                      (["s", "s"], "the list has 2 entries"),
                      ({7: "s"}, "names orbital 7"),
                      ({-1: "s"}, "names orbital -1"),
                      ({(0, 0): "s"}, "orbital 0 is named twice, in the key (0, 0)"),
                      ({(0, 1): "A1 + B1", 0: "s"}, "in the key (0, 1) and in the key 0"),
                      ({True: "s"}, "a bool"),
                      ({"0": "s"}, "a str"),
                      ({(): "s"}, "names no orbital"),
                      ({0: 5}, "given the int 5"),
                      ({0: None}, "leaves out an orbital"),
                      ({0: {}}, "empty mapping"),
                      ({0: {"C4(z)": "minus"}}, "a character is a finite number"),
                      ({0: {1.5: 1}}, "named by its index or its name"),
                      (["s", "s", "s", 5], "entry 3 is given the int 5")):
        with pytest.raises(ValueError, match=re.escape(says)) as raised:
            rebuild(lin3g_undeclared, characters=bad)
        assert str(raised.value).startswith("`characters=` is a list with one entry per orbital")

    def measured(*args, **kwargs):
        raise AssertionError("the declaration's grammar is read before any group")

    monkeypatch.setattr(tqg, "_group_of", measured)
    with pytest.raises(ValueError, match="orbital 0 is named twice"):
        tqg.realized_group(lin3g_undeclared.source, characters={0: "s", (0, 1): "E"})
    monkeypatch.undo()

    out = rebuild(lin3g_undeclared, characters={2: "E"})
    assert not out and out.gate == "unknown_label"
    assert "A1" in out.reason and "B2" in out.reason
    assert "m(x)" in out.reason and "C2(z)" in out.reason, "the generator frame is printed"
    assert set(out.context["irreps"]) == {"-E", "A1", "A2", "B1", "B2"}, (
        "the site corep the row is returned under, and the orbital name a "
        "declaration states -- both are declaration words at a spinful site")
    assert set(out.context["site_group"]) == {1, 2, 5, 6}, \
        "mm2 at 2c is E, C2(z), m(y), m(x) -- the four operations that fix the site"

    wrong = rebuild(lin3g_undeclared, characters={0: "A1g"})
    assert not wrong and wrong.gate == "unknown_label"
    assert "E" in wrong.context["irreps"], "4mm does have a two-dimensional irrep"

    junk = rebuild(lin3g_undeclared, characters={0: {"C6": 1.0}})
    assert not junk and junk.gate == "unknown_label"
    assert "names no operation" in junk.reason


def test_a_hoppings_forbidden_declaration_names_what_was_measured(lin3g_undeclared):
    """The **positions** allow both ``1a`` orbitals to be ``s``; the Hamiltonian does not.

    This is the A3b(a) program.  Nothing about the *geometry* forbids two
    ``s`` orbitals at ``1a`` — ``A1`` is an irrep of that site's ``4mm`` and
    the position hosts as many of them as one likes — so ``_assign_orbitals``
    is happy and every gate that looks at the cell is happy.  What forbids it
    is the **hoppings**: lin3g's orbital 1 is the ``d_x2-y2``-like partner of
    the Wieder quartet, and the representation the covariance search pinned
    measures its character relative to orbital 0 as ``B1``, absolutely.

    The refusal must therefore *name* that, not merely tabulate numbers: a
    caller who wrote ``{0: "s", 1: "s"}`` needs to read "you said A1 and A1,
    the hoppings say the second one is B1" -- and, since a declaration is never
    second-guessed, which crystal that describes: the ``mm2`` subgroup
    ``realized_group`` descends to.
    """
    out = rebuild(lin3g_undeclared, characters={0: "s", 1: "s"})
    assert not out and out.gate == "declaration_breaks_group"
    assert out.context["orbitals"] == (1,), "the second declaration is the one that fails"
    assert out.context["subgroup"] == (1, 2, 5, 6, 9, 10, 13, 14)

    # the named verdict -- this is the sentence the caller acts on
    assert "declared (0,)=-E1, (1,)=-E1" in out.reason, \
        "the declaration is echoed in the vocabulary the measured line is written in"
    assert "carry -E2, not the declared -E1" in out.reason, (
        "what the states carry, and what the declaration says they carry, both "
        "in the vocabulary the record returns its labels in")
    assert "relative character B1" in out.reason, (
        "the ratio of the measured character to the declared one is a twist and "
        "not a site representation, so it is named in the orbital vocabulary")
    assert out.context["measured_irrep"] == "-E2"
    assert out.context["relative_character"] == "B1"
    assert out.context["declared_irreps"] == (((0,), "A1"), ((1,), "A1"))

    # ... and the numeric table underneath it still says which operation demands it
    rows = [r for r in out.context["checks"] if r["orbitals"] == (1,)]
    bad = [r for r in rows if abs(r["measured"] - r["declared"]) > 1e-9]
    assert {r["name"] for r in bad} == {"C4(z)", "C4^-1(z)", "m([110])", "m([1-10])"}, \
        "B1 of 4mm is exactly odd under C4 and under the diagonal mirrors"
    assert all(abs(r["declared"] / r["measured"] + 1.0) < 1e-9 for r in bad)

    # the same statement in labels rather than shorthands, and the reverse order
    by_label = rebuild(lin3g_undeclared, characters={0: "A1", 1: "A1"})
    assert not by_label and by_label.gate == "declaration_breaks_group"
    assert by_label.context["relative_character"] == "B1"

    other_way = rebuild(lin3g_undeclared, characters={1: "s", 0: "s"})
    assert not other_way and other_way.gate == "declaration_breaks_group"
    assert other_way.context["orbitals"] == (1,), \
        "the reference orbital's own declaration is read first whichever order it " \
        "was written in, so the declaration named is the same one either way"
    assert other_way.context["relative_character"] == "B1", \
        "B1 is its own inverse, so the relative character is the same either way"
    assert other_way.context["subgroup"] == out.context["subgroup"], \
        "and the crystal it describes is the same one, written the other way round"

    # the near miss is the twist, not a contradiction: B1 of 4mm restricts to A1
    # of mm2, so the 2c declarations survive the relabelling untouched and only
    # the 1a pair changes hands
    consistent = rebuild(lin3g_undeclared, characters=["dx2-y2", "s", "s", "s"])
    assert consistent, str(consistent)
    assert consistent.characters.irreps == ("-E2", "-E1", "-E", "-E")

    clash = rebuild(lin3g_undeclared, characters=["dx2-y2", "dx2-y2", None, None])
    assert not clash and clash.gate == "declaration_breaks_group", \
        "the two 1a orbitals differ by a measured B1; no convention makes them equal"
    assert clash.context["subgroup"] == out.context["subgroup"], \
        "a global twist does not move which operations the declaration breaks"


def test_the_flat_list_refuses_when_it_cannot_group_a_multiplet(lin3g_undeclared,
                                                                monkeypatch):
    """A 2D label on one orbital of a two-orbital site is ambiguous by construction.

    Orbitals of two sites written as one key are no multiplet at all: the sites
    are read off the centres alone, so that is grammar, and it raises before
    anything is measured.
    """
    out = rebuild(lin3g_undeclared, characters=["E", None, None, None])
    assert not out and out.gate == "ambiguous_multiplet"
    assert "mapping form" in out.reason and "2-dimensional" in out.reason
    assert out.context["dim"] == 2 and out.context["orbitals"] == (0,)

    with pytest.raises(ValueError, match=re.escape("the key (0, 2) names orbitals of "
                                                   "sites [0, 1]")):
        rebuild(lin3g_undeclared, characters={(0, 2): "A1"})

    def measured(*args, **kwargs):
        raise AssertionError("the sites are read off the centres before any group")

    monkeypatch.setattr(tqg, "_group_of", measured)
    with pytest.raises(ValueError, match=re.escape("names orbitals of sites")):
        tqg.realized_group(lin3g_undeclared.source, typat=[0, 0, 1, 1],
                           characters={(1, 3): "A1"})


def test_a_two_dimensional_declaration_the_model_only_holds_on_a_subgroup(lin3g_undeclared):
    """lin3g's ``1a`` pair is ``A1 + B1``, not ``E`` — so ``E`` is a smaller crystal.

    The declaration is a layout as well as a character, so the covariance search
    is run on the declared doublet and finds nothing on all of ``4mm``; the
    per-operation reading on the detector's own solution spaces then says which
    operations it is: the two axial mirrors outright (``chi(sigma_v) = 0``
    against a measured ``+2``) and the fourfold with them, leaving the
    **diagonal** ``mm2`` where ``lambda = B1`` or ``B2`` serves it.
    """
    out = rebuild(lin3g_undeclared, characters={(0, 1): "E"})
    assert not out and out.gate == "declaration_breaks_group"
    assert out.context["subgroup"] == (1, 2, 7, 8, 9, 10, 15, 16), \
        "E, C2(z), m([110]), m([1-10]) and their primed partners"
    rows = {r["name"]: r for r in out.context["checks"]}
    assert abs(rows["C2(z)"]["measured"] - 2.0) < 1e-9, "two 1D characters, both +1 on C2"
    assert abs(rows["C2(z)"]["declared"] + 2.0) < 1e-9, "E has chi(C2) = -2"
    assert abs(rows["m(x)"]["measured"] - 2.0) < 1e-9
    assert abs(rows["m(x)"]["declared"]) < 1e-9, "E has chi(sigma_v) = 0"


# ---------------------------------------------------------------------------
# 4.  multiplets, and a second site-group family
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_p4232_declares_a_two_dimensional_multiplet_jointly(p4232_decorated):
    """The ``23`` site's doublet, declared as one object on the partner pair.

    The two orbitals of P4_232's ``2a`` site are one measured channel, and its
    block is the two-dimensional physically irreducible ``1E2E`` (a
    complex-conjugate pair glued by time reversal).  ``chi (x) 1E2E`` is
    ``1E2E`` again for every real character here, so every labelling gives the
    rows the same site irrep: undeclared they are already ``hamiltonian``, and
    a two-dimensional declaration of them is a check that settles nothing new.
    The paper's own words for the site -- ``d_z2`` and ``d_x2-y2``, which
    "together form a time-reversal symmetric irrep" -- declare it jointly.

    Stating the measured multiplets as ``multiplets=`` serves the identical
    representation, which is what makes a stated partition a declaration about
    the orbitals and not a second convention.
    """
    _, _, _, _, rep = p4232_decorated
    rec = rep.characters
    assert rec.irreps == ("1E2E",) * 4
    assert [o.dim for o in rec.orbitals] == [2, 2, 2, 2], (
        "the echo record's own reading of 'the site block is the two-dimensional "
        "1E2E' -- chi(E) = 2 on all four rows, one per orbital of the two 2a sites")
    assert [o.multiplet for o in rec.orbitals] == [(0, 1), (0, 1), (2, 3), (2, 3)], \
        "and the two orbitals of each site carry that character jointly"
    assert rec.residual < 1e-13, (
        "on a measured channel the residual is the single-valuedness check on "
        "the site blocks; this model is spinless, so they close exactly")

    assert {o.determined_by for o in rec.orbitals} == {"hamiltonian"}
    assert [o.wyckoff for o in rec.orbitals] == ["2a"] * 4
    for spec in ({(0, 1): "E"}, {0: "1E2E"}, {(0, 1): "dz2"}):
        out = rebuild(p4232_decorated, n_battery=4, characters=spec)
        assert out, f"{spec}: {out}"
        assert out.characters.irreps == ("1E2E",) * 4, spec
        assert same_matrices(rep, out)
        declared = {int(a) for d in out.characters.declarations for a in d["orbitals"]}
        assert [o.determined_by for o in out.characters.orbitals] == [
            "declaration" if o.orbital in declared else "hamiltonian"
            for o in out.characters.orbitals], spec

    assert len(rep.indices) == 48
    assert rep.multiplets.by_site == (((0, 1),), ((2, 3),))
    assert rep.multiplets.dims == ((2,), (2,))
    stated = rebuild(p4232_decorated, n_battery=4,
                     multiplets=[list(chans) for chans in rep.multiplets.by_site])
    assert stated, str(stated)
    assert stated.multiplets.determined_by == "declaration"
    assert stated.multiplets.by_site == rep.multiplets.by_site
    assert same_matrices(rep, stated)

    wrong = rebuild(p4232_decorated, n_battery=4, characters={(0, 1): "A"})
    assert not wrong and wrong.gate == "ambiguous_multiplet", (
        "the joint form names the content its orbitals hold together, and a "
        "one-dimensional irrep is not what two orbitals hold -- grammar about "
        "the declaration, never a measurement about the model")
    assert "{0: 'A', 1: 'A'} is the spelling" in wrong.reason
    assert wrong.context["dim"] == 1 and wrong.context["orbitals"] == (0, 1)


@pytest.mark.slow
def test_p4232_refuses_an_irrep_of_the_wrong_dimension_as_grammar(p4232_decorated):
    """A three-dimensional irrep is not what **two** orbitals carry jointly.

    ``T`` is a perfectly good irrep of P4_232's ``2a`` site group — the
    computed table lists it, and ``irreptables``' own EBR basis names it — so
    *the position allows it*.  What no Hamiltonian can make of it is a
    statement about two orbitals: the joint form names the content its
    orbitals hold together, and its dimension is the number of orbitals named.
    That is grammar about underspecified input, so it is ``ambiguous_multiplet`` and
    not a measurement about the model -- and the message says which spelling
    declares three separate ``T`` orbitals instead.
    """
    model, group, assignment, validated, rep = p4232_decorated
    table = rep.characters.site_tables[0]
    assert "T" in table.labels and dict(zip(table.labels, table.dims))["T"] == 3, \
        "the site group really does possess a three-dimensional irrep"

    out = rebuild(p4232_decorated, n_battery=4, characters={(0, 1): "T"})
    assert not out and out.gate == "ambiguous_multiplet"
    assert "3-dimensional against the 2 orbitals named" in out.reason
    assert out.context["dim"] == 3 and out.context["orbitals"] == (0, 1)

    # And the same physics one level down: with the layout STATED to be one
    # channel per orbital, the two orbitals of that site would have to carry two
    # 1D characters, which no signed permutation of them can make covariant.
    # Said with nothing declared, the refusal comes from the search's own gates
    # -- and the record says which gate, and the message which multiplet the
    # physics has.
    assert [tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals] \
        == [(0, 1), (2, 3)], "four orbitals, two to a site, on the two 2a sites"
    record = {}
    orb = tqs._build_rep(model, group, assignment, validated=validated,
                         multiplets=[[(0,), (1,)], [(2,), (3,)]], n_battery=4,
                         _search_record=record)
    assert not orb and orb.gate == "rep_not_covariant"
    assert orb.value > 1e-3
    assert "the tensor structure is too narrow for the multiplets it was given" in \
        orb.reason
    assert ("a site whose orbitals mix under its site-symmetry group is one multiplet"
            in orb.reason), "and the message names the multiplet the physics has"
    assert orb.context["multiplets"].determined_by == "declaration"
    search = record["search"]
    assert search["n_survivors"] == 0 and search["n_covariant"] == 0
    assert search["n_distinct"] == 4, "four distinct candidate ansatz orientations"
    assert search["killed_by"] == {"covariance": 4, "internal_factor": 0,
                                   "product_table": 0}
    assert search["best"] > 1e-3, "not a near miss: the ansatz cannot host this site"
    assert all(c["killed_by"] == "covariance" for c in search["candidates"])

    # ... and said as a DECLARATION -- the same layout, with the content the
    # caller means by it -- the answer is the physics rather than the search:
    # what was described is a crystal whose 23 site group is not a symmetry at
    # all, and the verdict names the subgroup that is left.
    declared = tqs._build_rep(model, group, assignment, validated=validated,
                              multiplets=[[(0,), (1,)], [(2,), (3,)]],
                              n_battery=4, characters={0: "A", 1: "A"})
    assert not declared and declared.gate == "declaration_breaks_group"
    assert len(declared.context["subgroup"]) < 24
    assert {b["ind"] for b in declared.context["broken"]}, "and which operations it breaks"


def test_kane_mele_covers_the_3m_site_group_family(km_realized):
    """Naming and declaration on ``3m``, so nothing here is C4v-only.

    The site point group's own double-valued names — read off the tables at the
    origin of a symmorphic group of ``3m``, which is what names a row the
    entries at its position do not — are the tables' own names here, where
    the entries name every row directly.  Each orbit's band representation is
    then its row's own elementary one at its own position, and nothing is
    printed under the table for it.  Written in a corep's own name, ``-E1``, a
    declaration states the corep content alone, which ``A1`` and ``A2`` of the
    site both lift to: on an orbital the hoppings carry as ``A2`` it is
    carried as it is.
    """
    _, group, _, _, rep = km_realized
    rec = rep.characters
    table = rec.site_tables[0]
    assert table.order == 6 and set(table.labels) == {"-E1", "-1E2E + -E1"}, (
        "the double-valued site coreps the tables name at this position: the two "
        "one-dimensional orbital characters both lift to the spinor doublet, and "
        "the orbital E lifts to the pair")
    assert tqe._point_group_names(group, table) == table.labels
    assert [(orbit.orbitals, orbit.wyckoff, orbit.terms)
            for orbit in rec.band_representations] == [
        ((0,), "1c", ((1, "-E1↑G(2)", "1c(3m1',3m)"),)),
        ((1,), "1b", ((1, "-E1↑G(2)", "1b(3m1',3m)"),))]
    assert not any(orbit.sentence() in rec.summary() for orbit in rec.band_representations)
    assert set(table.declarable) == {"A1", "A2", "E"}, "the orbital names still declare"
    assert rec.irreps == ("-E1", "-E1"), "two s-like sites, equal by measurement"

    # the table a refusal prints: each row by its orbital name with its own
    # characters, a zero at rounding read 0, and the corep it becomes with spin
    printed = tqc_sitechars.table_summary(table).splitlines()
    assert printed[0].split()[-2:] == ["with", "spin"]
    rows = {line.split()[0]: line.split()[1:] for line in printed[1:]}
    assert rows["A1"] == ["1", "1", "1", "-E1"] and rows["A2"] == ["1", "1", "-1", "-E1"]
    assert rows["E"] == ["2", "-1", "0", "-1E2E", "+", "-E1"]
    # ... and what fits a site holding one orbital, as a hint offers it: the E
    # doublet and the pair it becomes hold two
    assert tqc_sitechars.fitting_words(table, 1) == ("A1", "A2", "-E1")
    assert tqc_sitechars.fitting_words(table, 2) == ("A1", "A2", "E", "-E1", "-1E2E + -E1")

    # A multiplet the site's own operations bind across irreps -- a p shell
    # whose orbitals are turned inside the site -- is named from its character
    # alone: the rows it decomposes into, each row's corep terms counted
    # together, so the -E1 of A1 and the one inside E's pair are one term.
    row = {name: p for p, name in enumerate(table.declarable)}

    def of(**copies):
        return {int(i): sum(n * table.character_on(row[name], i) for name, n in copies.items())
                for i in table.indices}

    assert tqs._named_content(table, of(A1=1, E=1)) == ("2x-E1 + -1E2E", "tables")
    assert tqs._named_content(table, of(E=2))[0] == "2x-1E2E + 2x-E1", \
        "a multiplicity multiplies every term of a row whose own name is a sum"
    # named only as directly as its least direct part, and not at all where a
    # part has no name: naming the sum would claim one for that row too
    weaker = dataclasses.replace(table, named_by=tuple(
        "point group" if p == row["E"] else n for p, n in enumerate(table.named_by)))
    assert tqs._named_content(weaker, of(A1=1, E=1)) == ("2x-E1 + -1E2E", "point group")
    blank = dataclasses.replace(
        table, labels=tuple("" if p == row["E"] else n for p, n in enumerate(table.labels)),
        named_by=tuple("unnamed" if p == row["E"] else n for p, n in enumerate(table.named_by)))
    assert tqs._named_content(blank, of(A1=1, E=1)) == ("", "unnamed")

    assert [o.determined_by for o in rec.orbitals] == ["hamiltonian"] * 2, \
        "A1 and A2 of 3m lift to the one -E1, so every labelling gives both rows -E1"
    same = rebuild(km_realized, characters={0: "s"})
    assert same and same_matrices(rep, same)
    assert [o.determined_by for o in same.characters.orbitals] == \
        ["declaration", "hamiltonian"], \
        "3m separates both characters, so the second row is fixed by the hoppings"

    twisted = rebuild(km_realized, characters={0: "A2"})
    assert twisted, str(twisted)
    assert not same_matrices(rep, twisted, atol=1e-9)
    assert twisted.characters.irreps == ("-E1", "-E1"), (
        "A1 and A2 of 3m are one double-valued site corep -- the same label off a "
        "different crystal, which the matrices below are what separates")
    assert max(twisted.residuals.values()) <= GATE
    assert twisted.product_residual < 1e-13

    # the corep's own name states the corep content alone: `-E1` on the second
    # sublattice, whose orbital the hoppings carry as A2 once the first is
    # declared A2, is that crystal, and not the A1 whose mirrors would clash
    corep = rebuild(km_realized, characters={0: "A2", 1: "-E1"})
    assert corep, str(corep)
    assert same_matrices(twisted, corep)
    assert [o.determined_by for o in corep.characters.orbitals] == ["declaration"] * 2
    assert [o.declared_irrep for o in corep.characters.orbitals] == ["A2", "-E1"], \
        "the corep declared is recorded as written"
    assert round(corep.characters.orbitals[1].generator_character["m(x)"].real) == -1, \
        "and the orbital factor under it is the A2 the hoppings carry"

    clash = rebuild(km_realized, characters=["s", "A2"])
    assert not clash and clash.gate == "declaration_breaks_group"
    assert clash.context["subgroup"] == (1, 2, 3, 7, 8, 9), \
        "one sublattice even under the mirrors and the other odd is P3, not P3m1"


# ---------------------------------------------------------------------------
# 5.  the naming-free path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bundle_name,label,returned,gens", [
    ("lin3g", "B1", "-E2", {"C4": -1.0, "sigma_v(x)": 1.0}),
    ("km", "A2", "-E1", {"C3": 1.0, "m(x)": -1.0}),
])
def test_generator_dicts_reproduce_the_label_form(lin3g_undeclared, km_realized,
                                                  bundle_name, label, returned, gens):
    """The escape hatch no naming convention can spoil, on two site groups.

    ``returned`` is the double-valued site corep the row is returned under,
    which is what ``label`` -- the orbital character a declaration states --
    lifts to on these spinful fixtures.  The operations' own ``.ind`` are the
    last-resort keys, and bypass naming entirely.
    """
    bundle = lin3g_undeclared if bundle_name == "lin3g" else km_realized
    by_label = rebuild(bundle, characters={0: label})
    by_gens = rebuild(bundle, characters={0: gens})
    assert by_label and by_gens, f"{by_label}\n{by_gens}"
    assert same_matrices(by_label, by_gens)
    assert by_gens.characters.orbitals[0].irrep == returned
    assert "generator characters" in by_gens.characters.orbitals[0].note
    if bundle_name == "lin3g":
        frame = bundle.rep.characters.frame
        c4 = next(i for i, n in frame.items() if n == "C4(z)")
        mx = next(i for i, n in frame.items() if n == "m(x)")
        by_ind = rebuild(bundle, characters={0: {c4: -1.0, mx: 1.0}})
        assert by_ind, str(by_ind)
        assert same_matrices(by_ind, by_label), \
            "the generator characters name the same content the label does"


# ---------------------------------------------------------------------------
# 6.  the echo record itself
# ---------------------------------------------------------------------------

def test_the_echo_record_covers_every_orbital_and_prints_the_frame(lin3g_undeclared):
    """The safety net: one row per orbital, each with its site group and frame."""
    model, _, _, _, rep = lin3g_undeclared
    rec = rep.characters
    assert len(rec) == int(model.norb)
    assert [row.site for row in rec.orbitals] == [0, 0, 1, 2], (
        "the fixture's 1a/2c split, read off the record: orbitals 0 and 1 share "
        "the 1a site, orbitals 2 and 3 sit on the two 2c sites one each")
    for row in rec.orbitals:
        assert set(row.site_group) <= set(rep.indices)
        assert row.generator_character, "every row carries its own frame"
        assert row.dim == 2 * len(row.multiplet), (
            "`dim` is the dimension of what the label names -- the site irrep the "
            "states carry -- and on a spinful model that is the orbital multiplet "
            "times the two spin states")
    assert [(row.wyckoff, row.site_symmetry, len(row.site_group))
            for row in rec.orbitals] == [("1a", "4mm", 8), ("1a", "4mm", 8),
                                         ("2c", "mm2", 4), ("2c", "mm2", 4)]

    text = rec.summary()
    head = text.splitlines()[0].split("  ")
    assert [w.strip() for w in head if w.strip()] == [
        "orbital", "Wyckoff", "site symmetry", "function", "irrep", "named by",
        "determined by", "orbital characters"], "the columns of an atomic-limit table"
    assert "global character" not in text and "offset" not in text, (
        "the character relating two labellings of the orbitals is the package's own "
        "and never printed; each row's determined_by says what its site irrep "
        "rests on")
    assert "C4(z)" in text and "m(x)" in text
    assert not hasattr(rec, "twist") and not hasattr(rec, "gauge")


def _axis_p_at_2c():
    """``p_x`` at ``(½, 0)`` and ``p_y`` at ``(0, ½)`` of a square lattice, spinless.

    Each ``2c`` site's ``p`` along its own axis, with Slater-Koster nearest
    neighbours: ``-0.9`` along that axis, ``0.3`` across it, and between the two
    sites the ``±0.25`` a ``p_x`` and a ``p_y`` share across a diagonal bond, its
    sign the product of the bond's two components.  The fourfold carries ``p_x``
    at ``(½, 0)`` onto ``p_y`` at ``(0, ½)``, and that onto ``-p_x`` one cell
    over: an orbit that permutes the functions with a sign, in ``P4mm1'``
    (#99.164, 16 operations).
    """
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]],
                        periodic_dirs=[0, 1]), spinful=False)
    m.set_onsite([0.0, 0.0])
    m.set_hop(-0.9, 0, 0, [1, 0])            # p_x -- p_x along x: sigma
    m.set_hop(0.3, 0, 0, [0, 1])             # ... and along y: pi
    m.set_hop(-0.9, 1, 1, [0, 1])            # p_y -- p_y along y: sigma
    m.set_hop(0.3, 1, 1, [1, 0])             # ... and along x: pi
    for lv, sign in (([0, 0], -1.0), ([1, 0], 1.0), ([0, -1], 1.0), ([1, -1], -1.0)):
        m.set_hop(0.25 * sign, 0, 1, lv)     # p_x -- p_y across the four diagonals
    return m


@pytest.mark.slow
def test_an_orbit_that_permutes_the_frame_reads_each_site_its_own_content(realized):
    """``p`` along each ``3c`` site's own fourfold axis: ``A2u`` at every row.

    A Wyckoff orbit whose operations carry one site's orbital onto a *different*
    function at the next is what a transported content is for: ``p_x`` at
    ``(0,½,½)``, ``p_y`` at ``(½,0,½)`` and ``p_z`` at ``(½,½,0)``, one orbital
    apiece, so every multiplet is one-dimensional.  Each row carries the ``A2u``
    of its own site's ``4/mmm``, its characters named in that site's own axes,
    named by the tables.  The hoppings refute the trivial labelling -- ``A2u``
    is the restriction of no one-dimensional character of this group -- but not
    the whole class ``{A2u, B2u, A2g, B2g}`` the labellings they leave open
    give the row, so undeclared every row is ``lowest-l``, ``A2u`` its
    lowest-``l`` member.

    A transport read off hoppings known only to a precision reads the same, on
    a square crystal with one: ``p_x`` at ``(½, 0)`` and ``p_y`` at
    ``(0, ½)``, each along its own site's axis, which the fourfold carries
    onto one another with a sign.  Jittered at ``1e-7`` with
    ``precision=1e-6`` declared, each transported sign misses its exact value
    by the model's own noise and is read as that sign within the precision: the
    group is ``P4mm1'`` and each row the ``p`` along its own site's axis,
    ``B1`` and ``B2``, the model projected onto its symmetric part by the size
    of the jitter.
    """
    out = realized(cubic_3c_axis_p)
    assert out, str(out)
    assert (out.group.number_str, out.group.size) == ("221.93", 96)
    assert len(out.validated.indices) == 96 and out.descent is None
    rec = out.rep.characters
    assert rec.irreps == ("A2u",) * 3
    assert [o.named_by for o in rec.orbitals] == ["tables"] * 3
    assert [(o.wyckoff, o.site_symmetry) for o in rec.orbitals] == [("3c", "4/mmm")] * 3
    assert [o.determined_by for o in rec.orbitals] == ["lowest-l"] * 3
    chi = rec.orbitals[0].generator_character
    assert complex(chi["C4(x)"]).real == pytest.approx(1.0, abs=1e-9)
    assert complex(chi["C2(y)"]).real == pytest.approx(-1.0, abs=1e-9), \
        "odd under a twofold perpendicular to its own axis, which is what A2u is"
    assert max(out.rep.residuals.values()) <= GATE
    assert out.rep.product_residual < 1e-13

    noisy = tqg.realized_group(jitter(_axis_p_at_2c(), rel=1e-7), precision=1e-6,
                               cross_check=False)
    assert noisy, str(noisy)
    assert (noisy.group.name, noisy.group.size) == ("P4mm1'", 16)
    assert noisy.rep.characters.irreps == ("B1", "B2")
    for row, own, other in ((0, "m(x)", "m(y)"), (1, "m(y)", "m(x)")):
        chi = noisy.rep.characters.orbitals[row].generator_character
        assert {k: complex(v).real for k, v in chi.items()} == pytest.approx(
            {"E": 1.0, "C2(z)": -1.0, own: -1.0, other: 1.0}), \
            "the p along its own site's axis: odd under the twofold and the mirror reversing it"
    assert 0.0 < noisy.symmetrization.defect < 1e-6, \
        "the hoppings moved by the jitter's own size and no more"


# ---------------------------------------------------------------------------
# 6b. the returned matrices are the representation induced from the returned
#     content, with nothing hidden
# ---------------------------------------------------------------------------

#: What each fixture of the acceptance test stands for.
INDUCED_FIXTURES = {
    "perp_doublet": "spinless 3D: the E doublet at 3c of P-43m, named by the point group",
    "pxpy_honeycomb": "spinless 2D: the (p_x, p_y) doublet at 2b of P6mm",
    "lin3g": "spinful 2D: s and d_x2-y2 at 1a, s at 2c of P4mm",
    "pnma_s_4a": "spinful 3D: s at 4a of Pnma, every mirror and screw a transport",
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(INDUCED_FIXTURES))
def test_the_returned_matrices_are_induced_from_the_returned_content(name, realized,
                                                                    request):
    """Nothing hidden: the matrices are the representation the returned content induces.

    Two statements, on each fixture of :data:`INDUCED_FIXTURES`, and neither
    reads which orbital is which function.

    * **Every site block's trace is its rows' character.**  On every operation
      that fixes a site, the trace of the matrix over each multiplet there --
      the orbital factor's, the spin lift divided out orbital by orbital where
      the states carry spin -- is the character its rows carry, as many times
      as the multiplet holds copies of their irrep.
    * **Every block between sites is the model's own transport.**  Each matrix
      is a member of its operation's measured solution space: what carries one
      site's orbitals onto another's is what the Hamiltonian carries them
      with, in each site's own frame, and not a copy of one site's block.

    The characters of the doublet's rows vanish on most of its operations and
    print as a table prints them; they are returned as the complex numbers
    they are.
    """
    builders = {"perp_doublet": (cubic_3c_perp_doublet, {}),
                "pxpy_honeycomb": (tqc_models.pxpy_honeycomb, {}),
                "pnma_s_4a": (tqc_models.pnma_s_4a, {})}
    if name in builders:
        build, kw = builders[name]
        bundle = realized(build, **kw)
    else:
        bundle = request.getfixturevalue("lin3g_undeclared")
    assert bundle, str(bundle)
    rep = bundle.rep
    rows = rep.characters.orbitals
    if name == "perp_doublet":
        assert all(isinstance(v, complex) for o in rows
                   for v in (*o.character.values(), *o.generator_character.values()))

    nspin = int(rep.model.nspin)
    blocks, members = 0, 0
    for ind in rep.indices:
        sym = rep._operation(ind)
        D = np.asarray(rep.matrices[int(ind)])
        if not sym.time_reversal:
            lift = (np.asarray(sym.spinor_rotation, dtype=complex) if nspin == 2
                    else np.eye(1))
            for multiplet in dict.fromkeys(o.multiplet for o in rows):
                row = rows[multiplet[0]]
                if int(ind) not in row.character:
                    continue                  # the operation moves this site
                trace = sum(complex(np.vdot(lift, D[np.ix_(st, st)]) / np.vdot(lift, lift))
                            for st in ([a * nspin + t for t in range(nspin)]
                                       for a in multiplet))
                copies = len(multiplet) / complex(row.generator_character["E"]).real
                assert abs(trace - copies * complex(row.character[int(ind)])) < 1e-9, \
                    f"{name}: #{ind} on {multiplet}"
                blocks += 1
        space = np.asarray(bundle.validated._bases[int(ind)])
        A = space.reshape(space.shape[0], -1).T
        coef = np.linalg.lstsq(A, D.reshape(-1), rcond=None)[0]
        assert float(np.abs(A @ coef - D.reshape(-1)).max()) < 1e-9, \
            f"{name}: #{ind} is not the model's own"
        members += 1
    assert members == len(rep.indices) and blocks >= len(rep.indices) // 2


# ---------------------------------------------------------------------------
# 7.  the character mathematics itself
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_computed_character_tables_pass_their_own_orthogonality_check(
        lin3g_undeclared, km_realized, p4232_decorated):
    """Row orthogonality is the table's self-check; nothing here is tabulated.

    Burnside's ``sum d_p^2 = |G|`` is the other theorem, and the **merged**
    tables are why it is written the way it is below.  A row
    ``_merge_conjugate_pairs`` produced — P4_232's ``1E2E`` at its ``2a`` site —
    is not an irreducible character but the *sum* of a complex-conjugate pair of
    inequivalent ones, glued because time reversal makes them one physically
    irreducible object.  Each half of that pair contributes ``(d/2)^2`` to the
    sum of squares, so the merged row contributes ``d^2/2`` and not ``d^2``:
    measured on the ``23`` site, ``dims = (2, 1, 3)`` sums to **14** against an
    order of **12**, and ``4/2 + 1 + 9 = 12`` is the identity that holds.  That
    is the only self-check the merged branch has in either file, which is why
    the ``23`` site is in this loop at all.
    """
    for bundle, orders, n_merged in ((lin3g_undeclared, {8, 4}, 0),
                                     (km_realized, {6}, 0),
                                     (p4232_decorated, {12}, 2)):
        _, _, _, _, rep = bundle
        seen, merged_rows = set(), 0
        for table in rep.characters.site_tables.values():
            assert table.orthogonality < 1e-10, table.labels
            assert all(table.labels), "every row of these site groups is named"
            dims = np.asarray(table.dims, dtype=int)
            merged = np.asarray(table.merged, dtype=bool)
            assert len(merged) == len(dims)
            assert int(np.sum(np.where(merged, dims ** 2 // 2, dims ** 2))) == table.order
            merged_rows += int(np.count_nonzero(merged))
            seen.add(table.order)
        assert seen == orders
        assert merged_rows == n_merged, (
            "and the merged branch is reached exactly where the site group has a "
            "complex-conjugate pair for time reversal to glue")


def test_mirrors_are_named_by_their_normal(lin3g_undeclared):
    """The frame the labels are interpretable in, asserted rather than assumed.

    ``_sitechars``' documented contract is the Hermann-Mauguin reading: the axis
    in a mirror's name is the plane's **normal**, so ``m(x)`` here *is* Mulliken's
    ``σ_v(yz)``.  Every label in this file is read in that frame, so the frame is
    checked the way the library builds it — from the matrices.

    Each operation of the frame is classified by its own ``rotation_cart``: a
    mirror is the element with ``det = -1`` and ``tr = +1`` (both
    basis-independent, so neither needs the naming code), its normal is the
    non-degenerate ``-1`` eigenvector (measured here, not taken from
    ``_sitechars``), and the name the frame must carry is that direction spelled
    by :func:`_sitechars._axis_name` — plus the ``'`` the record appends to an
    antiunitary operation.  Nothing is skipped and the count is pinned, so
    neither a rename nor a misread axis can leave the loop vacuous: the four
    vertical mirrors and their four time-reversed copies are all eight checked.
    """
    _, group, _, _, rep = lin3g_undeclared
    frame = rep.characters.frame
    lattice = np.asarray(group.lattice, dtype=float)
    pos = {int(s.ind): s for s in group.symmetries}
    assert set(frame) == set(pos) and len(frame) == 16, "the frame names every operation"

    checked = []
    for ind, name in sorted(frame.items()):
        sym = pos[ind]
        R = np.asarray(sym.rotation_cart, dtype=float)
        if (int(round(float(np.linalg.det(R)))) != -1
                or abs(float(np.trace(R)) - 1.0) > 1e-7):
            continue                      # an inversion or a rotoinversion, not a mirror
        assert np.allclose(R, R.T, atol=1e-12), "a reflection is symmetric: 1 - 2 n n^T"
        w, V = np.linalg.eigh(R)
        assert [int(round(float(x))) for x in w] == [-1, 1, 1], \
            "and its spectrum is one inverted direction and a fixed plane"
        normal = np.real(V[:, int(np.argmin(np.abs(w + 1.0)))])
        assert np.allclose(R @ normal, -normal, atol=1e-12), "the normal is inverted"
        plane = np.eye(3) - np.outer(normal, normal)
        assert np.allclose(R @ plane, plane, atol=1e-12), "and the plane is fixed"
        want = "m(" + tqc_sitechars._axis_name(normal, lattice) + ")"
        if bool(sym.time_reversal):
            want += "'"
        assert name == want, (
            f"#{ind} is the mirror whose normal is {np.round(normal, 6)}, so the "
            f"frame must spell it {want!r}, not {name!r}")
        checked.append(name)

    assert len(checked) == 8, (
        "lin3g's realized P4mm1' has four vertical mirrors -- m(x), m(y) and the "
        "two diagonals -- and a time-reversed copy of each; every one is named "
        f"here, and the loop checked {checked}")
    assert set(checked) == {"m(x)", "m(y)", "m([110])", "m([1-10])",
                            "m(x)'", "m(y)'", "m([110])'", "m([1-10])'"}


@pytest.mark.slow
def test_group_provenance_rides_into_the_character_tables(lin3g_undeclared):
    """A table can be read without the group in hand: it carries the stamp.

    ``realized_group`` stamps its record before the representation is built, so
    ``rep.group`` carries it and every table built from that representation
    echoes it — which is the point of the field, since a symmetry vector's
    labels mean one thing when the group was measured on ``H`` and another when
    a cell merely allowed it.

    **Both stamps are exercised, and wave R4c is why there are two here.**  This
    module's fixture takes the default reading of a ``dim_r = 2`` model, which is
    the layer **on a substrate**: the ``σ_h``-composed operations are absent from
    the cell by construction, the cell allows ``P4mm1' #99.164`` outright, the
    Hamiltonian obeys all sixteen and nothing is cut — ``"realized"``.  Declared
    **free-standing**, the same model's cell is the slab point set as given, so
    it allows ``P4/mmm1' #123.340`` — 32 operations, ``σ_h`` and everything
    composed with it included — the Hamiltonian breaks half of them, and the
    pipeline lowers the cell to the sixteen that hold and serves the same
    ``P4mm1' #99.164`` this module has always labelled its bands with:
    ``"descended"``.  Every statement ``"realized"`` makes is true of a descended
    record — the group was measured on ``H``, not merely allowed by a geometry —
    and it additionally says the cell was an upper bound the measurement had to
    cut.  The tables must carry whichever of the two the record has, not a
    hard-coded value: the field exists to be read, and a table that flattened the
    distinction would be lying about where its labels came from.  Both values are
    asserted below, on one model and one served group.
    """
    _, group, _, _, rep = lin3g_undeclared
    assert group.provenance == "realized" and rep.group.provenance == "realized"
    assert group.number_str == "99.164" and group.size == 16

    k = tqg.high_symmetry_points(group)["GM"]
    table = tqw.characters(rep, k, bands=6)
    assert isinstance(table, tqw.CharacterTable), str(table)
    assert table.group_provenance == group.provenance == "realized"

    hsp = tqw.hsp_table(rep, bands=6)
    assert hsp.group_provenance == group.provenance == "realized"
    assert hsp.records, "the realized group resolves its high-symmetry points"
    assert all(rec.table.group_provenance == group.provenance
               for rec in hsp.records.values())

    # ... and the other stamp, on the same model and the same served group: the
    # free-standing cell is an upper bound the measurement had to cut
    down = tqg.realized_group(lin3g_undeclared.source, typat=[0, 0, 1, 1], c=C_SLAB,
                              environment="free-standing")
    assert down, str(down)
    assert down.group.number_str == "99.164" and down.group.size == 16
    assert down.descent["from"]["number_str"] == "123.340"
    assert down.group.provenance == down.rep.group.provenance == "descended"
    down_table = tqw.characters(down.rep, k, bands=6)
    assert down_table.group_provenance == "descended"
    assert tqw.hsp_table(down.rep, bands=6).group_provenance == "descended"


# ---------------------------------------------------------------------------
# 8.  the names the tables carry for a Wyckoff position
# ---------------------------------------------------------------------------
#
# A computed site-group table has characters but no names.  The names a
# physicist looks a position up by are the tables' own, and what connects the
# two is the band representation a site irrep induces: that is what an EBR
# column holds, so a row is named by inducing it and matching, never by a
# character lookup that would need an orientation convention nothing measures.


@pytest.fixture(scope="module")
def spinless_diamond():
    """One spinless ``s`` on each diamond site: the ``8a`` position of Fd-3m1'.

    The same model the P4 review's prototype measured the naming on, with the
    site group ``-43m`` whose five rows ``_mulliken`` could only call
    ``A_a``, ``A_b``, ``E``, ``T_a``, ``T_b``.
    """
    lat = Lattice([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
                  [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=False)
    m.set_onsite([0.0, 0.0])
    for lv in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]):
        m.set_hop(-1.0, 1, 0, lv)
    out = tqg.realized_group(m)
    assert out, str(out)
    return out


@pytest.mark.slow
def test_the_diamond_site_irreps_are_the_names_the_tables_print(spinless_diamond,
                                                                fkm_undistorted):
    """``8a`` of Fd-3m1' reads ``A1 A2 E T1 T2``, and declarations use those words.

    The rows the tables name are reached three ways here, and all three are the
    physics rather than the naming: the *label set* itself, which irreptables
    prints for that position; the **shorthands**, whose spans are built from the
    site group's own action on Cartesian polynomials and land on ``A1``, ``T2``
    and ``E`` for ``s``, the ``p`` triple and the ``d`` pair — that is a
    statement about the diamond and holds whatever anything is called; and the
    **declaration grammar**, where the tables' words are now what a caller may
    write and the placeholder is not.

    ``{0: "T2"}`` is refused ``ambiguous_multiplet`` and not ``unknown_label``:
    this model carries one orbital per site, so a three-dimensional irrep is not
    the dimension of its channel, which is a grammar statement about the
    declaration and not about the name.

    The second half is the same position with spin.  What the states carry
    there is each row times the operations' spin-1/2 lift, so the label they
    are returned under is the double-valued site corep — but a declaration
    still states the **orbital**, and the orbital factor is the same
    single-valued object at the same position whichever the model.  So the
    words a declaration may use are the same five, row for row, and the corep's
    own name declares too: ``-E1`` and ``s``, the shared record's declaration,
    resolve to one content, character for character.
    """
    r = spinless_diamond
    assert r.group.number_str == "227.129"
    assert r.assignment.wyckoff_tables == ("a", "a")
    table = r.rep.characters.site_tables[0]

    assert set(table.labels) == {"A1", "A2", "E", "T1", "T2"}
    assert dict(zip(table.labels, (int(d) for d in table.dims))) == \
        {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
    assert set(table.named_by) == {"tables"}, "every row of -43m at 8a is tabulated"
    assert all(o.named_by == "tables" for o in r.rep.characters.orbitals)
    assert all(" tables " in line for line in r.rep.characters.summary().splitlines()[1:3]), \
        "the named-by column says where each printed name comes from"

    for spec, want, dim in (("s", "A1", 1), ("px", "T2", 3), ("py", "T2", 3),
                            ("pz", "T2", 3), ("dz2", "E", 2), ("dx2-y2", "E", 2)):
        _, label, got, _note = tqc_sitechars.resolve_spec(r.group, table, spec,
                                                          "orbital 0")
        assert (label, got) == (want, dim), spec
    # a shell names the content its harmonics carry here, and a sum of irreps
    # the content they add up to, in any order
    for spec, want, dim in (("p", {"T2"}, 3), ("d", {"E", "T2"}, 5),
                            ("T2 + A1", {"A1", "T2"}, 4)):
        _, label, got, _note = tqc_sitechars.resolve_spec(r.group, table, spec,
                                                          "orbital 0")
        assert (set(label.split(" + ")), got) == (want, dim), spec

    assert r.rep.characters.irreps == ("A1", "A1"), "undeclared, the lowest-l rule's A1"
    out = tqs._build_rep(r.model, r.group, r.assignment, validated=r.validated,
                         characters={0: "A2"})
    assert out, str(out)
    assert out.characters.irreps == ("A2", "A2"), \
        "both sites of the orbit carry what was declared on one of them"

    gone = tqs._build_rep(r.model, r.group, r.assignment, validated=r.validated,
                          characters={0: "A_b"})
    assert not gone and gone.gate == "unknown_label", "the placeholder names nothing"
    assert set(gone.context["irreps"]) == {"A1", "A2", "E", "T1", "T2"}

    wide = tqs._build_rep(r.model, r.group, r.assignment, validated=r.validated,
                          characters={0: "T2"})
    assert not wide and wide.gate == "ambiguous_multiplet", \
        "a three-dimensional irrep on a one-orbital channel is a dimension, not a name"

    # The same position in the DOUBLE group: what states carrying spin would
    # carry there is each of those rows times the operations' spin-1/2 lift,
    # which the tables name with their own double-valued entries.  `A1` lifts
    # to the fundamental spinor `-E1`, the `p` triple to `-E2 + -F` and the `d`
    # pair to `-F` -- the tetrahedral site's whole double-valued vocabulary,
    # read by inducing and matching and not by any convention here.  The
    # spinful model is undistorted Fu-Kane-Mele, `fu_kane_mele(1.0, 0.125,
    # [0, 0, 0, 0])` with `s` declared: the session's shared record.
    spinful = fkm_undistorted
    assert spinful, str(spinful)
    assert spinful.group.number_str == "227.129" and spinful.group.spinor
    dtable = spinful.rep.characters.site_tables[0]
    assert dict(zip(dtable.declarable, dtable.labels)) == {
        "A2": "-E2", "A1": "-E1", "E": "-F",
        "T1": "-E1 + -F", "T2": "-E2 + -F"}
    assert set(dtable.named_by) == {"tables"}
    assert tqe._point_group_names(spinful.group, dtable) == dtable.labels, (
        "-43m's own double-valued names, read at the origin of a symmorphic group "
        "and carried onto 8a of Fd-3m operation by operation, are the names the "
        "tables give every row there")
    assert spinful.rep.characters.irreps == ("-E1", "-E1"), "an s orbital is -E1"
    assert [o.dim for o in spinful.rep.characters.orbitals] == [2, 2]

    # The orbital factor is the same single-valued object whether or not the
    # states carry spin, so it carries the same name at the same position: row
    # for row, the words a declaration here may use are the labels the spinless
    # model above is returned under.  The two are read from the two table files
    # of the one group and nothing else differs between them.
    assert dtable.declarable == table.labels
    assert not any("_" in word for word in tqc_sitechars.declaration_words(dtable)), \
        "no placeholder survives in the words a declaration at this site may use"
    # a sum written in the coreps the rows are returned as states the corep
    # content alone, and is named in the coreps; a sum in the orbital names
    # states that content, and is named in them
    for spec, want, coreps in (("-E2 + 2x-F", {"-E2", "2x-F"}, True),
                               ("-F + -E1", {"-E1", "-F"}, True), ("d", {"E", "T2"}, False),
                               ("T2 + E", {"E", "T2"}, False)):
        _, label, _dim, _note = tqc_sitechars.resolve_spec(spinful.group, dtable, spec,
                                                           "orbital 0")
        assert set(label.split(" + ")) == want, spec
        assert tqc_sitechars.named_in_coreps(dtable, spec) is coreps, spec

    # ... and the corep's own name declares as well as the orbital name it lifts
    # from, which is what the shared record declares: the two resolve to one
    # constraint on the site group's operations and one dimension, which is what
    # the matrices are built from, each named as it was written
    as_corep = tqc_sitechars.resolve_spec(spinful.group, dtable, "-E1", "orbital 0")
    as_orbital = tqc_sitechars.resolve_spec(spinful.group, dtable, "s", "orbital 0")
    assert as_corep[0] == as_orbital[0] and as_corep[2] == as_orbital[2] == 1, \
        "the corep's name and the orbital name it lifts from are one declaration"
    assert (as_corep[1], as_orbital[1]) == ("-E1", "A1")
    assert len(as_corep[0]) == len(dtable.indices)


@pytest.mark.slow
def test_a_tie_at_gamma_is_broken_at_the_other_high_symmetry_points():
    """Two EBRs with one Γ content are told apart where they differ.

    ``B1↑G`` and ``B2↑G`` at ``2c`` of P4mm are both ``GM5`` at Γ and part at X
    and M, so a Γ content alone cannot name a row of that site's ``mm2``.  A
    square lattice with one ``s`` on each of ``(½, 0)`` and ``(0, ½)`` sits
    exactly there, and its four rows come back with four different names — which
    is the induction at a point away from Γ, with the site-displacement phases,
    doing the separating.
    """
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]],
                  periodic_dirs=[0, 1])
    m = TBModel(lat, spinful=False)
    m.set_onsite([0.0, 0.0])
    m.set_hop(-1.0, 0, 0, [1, 0])
    m.set_hop(-1.0, 1, 1, [0, 1])
    for lv in ([0, 0], [1, 0], [0, -1], [1, -1]):
        m.set_hop(0.25, 0, 1, lv)

    on_substrate = tqg.realized_group(m, environment="substrate")
    assert on_substrate, str(on_substrate)
    assert on_substrate.group.number_str == "99.164"
    assert on_substrate.assignment.wyckoff_tables == ("c", "c")
    ties = tqe._gamma_ties(on_substrate.group)
    assert {frozenset(names) for names in ties["2c(mm21',mm2)"]} == \
        {frozenset({"B1↑G(2)", "B2↑G(2)"})}, \
        "the tables say these two share their Γ content and nothing else does"
    assert set(ties) == {"2c(mm21',mm2)"}, "and this is the group's only tied position"
    table = on_substrate.rep.characters.site_tables[0]
    assert set(table.labels) == {"A1", "A2", "B1", "B2"}
    assert set(table.named_by) == {"tables"}


def _baseline_realized(key, realized):
    """One record of ``tests/test_baseline_r4.py``, by its key, read as the gate reads it.

    The gate's own builder and keywords, through the session's ``realized``, so a
    session that runs the gate measures each of these models once.
    """
    section, _, rest = key.partition("/")
    build, kw = _baseline.record_call(section, rest)
    return realized(build, **kw)


#: One fixture of the baseline gate per (group, Wyckoff letter, valuedness) the
#: gate's fixtures reach, with what each one stands for.  The 42 non-refusal
#: records reach **18** such cases and 24 of them are repeats — a second fixture
#: of the same group at the same position in the same valuedness makes the same
#: claim about the same table — so one representative of each is taken and the
#: repeats are left to the gate itself.  Measured over all 42 (the whole set,
#: read once), these 18 cover every case and leave none behind.  The chiral
#: ``P41'`` layer is one case in both readings: free-standing, the tie rule keeps
#: the fourfold rotations, and the group is the substrate reading's ``#75.2``.
ORACLE_FIXTURES = {
    "layer/pxpy|free-standing": "#191.234, single-valued, -6m2 at c",
    "layer/kane_mele|free-standing": "#191.234, double-valued, -6m2 at c",
    "layer/pxpy|substrate": "#183.186, single-valued, 3m at b",
    "layer/kane_mele|substrate": "#183.186, double-valued, 3m at b",
    "layer/buckled_graphene|substrate": "#156.50, single-valued, 3m at b and c",
    "r1/km_odd|substrate": "#156.50, double-valued, 3m at b and c",
    "layer/buckled_graphene|free-standing": "#164.86, single-valued, 3m at d",
    "layer/square_s_pz|substrate": "#99.164, single-valued, 4mm at a",
    "r1/lin3g|substrate": "#99.164, double-valued, 4mm at a and mm2 at c",
    "layer/p41_chiral|substrate": "#75.2, double-valued, 2 at c",
    "layer/square_s_pz|free-standing": "#123.340, single-valued, 4/mmm at a",
    "ebr/p4mm_s_1a|soc=0|free-standing": "#123.340, double-valued, 4/mmm at a",
    "layer/square_s_pz_mixed|free-standing": "#47.250, single-valued, mmm at a",
    "ebr/mm2_s|soc=0|free-standing": "#47.250, double-valued, mmm at a",
    "layer/square_s_pz_mixed|substrate": "#25.58, single-valued, mm2 at a",
    "ebr/mm2_s|soc=0|substrate": "#25.58, double-valued, mm2 at a",
    "r1/fkm_diamond": "#227.129, double-valued, -43m at a",
    "r1/fkm_bond_weak": "#166.98, double-valued, 3m at c, which the EBR table omits",
}

BASELINE_KEYS = sorted(ORACLE_FIXTURES)

#: Five of them run in the slow tier -- single- and double-valued, layers read on
#: a substrate and free-standing, the diamond, and a decorated diamond whose
#: position the EBR table omits -- and the whole set runs with the table sweeps.
ORACLE_SLOW = frozenset({"r1/lin3g|substrate", "layer/kane_mele|free-standing",
                         "layer/buckled_graphene|substrate", "r1/fkm_bond_weak",
                         "r1/fkm_diamond"})

# Read at collection, not in a test of its own: every representative has to be a
# record of the gate, so that a fixture renamed there cannot leave the oracle
# standing on something the gate no longer measures.
for _key in BASELINE_KEYS:
    _section, _, _rest = _key.partition("/")
    assert _rest in _baseline.BASELINE[_section], (
        f"{_key} is no longer a record of tests/test_baseline_r4.py")
assert ORACLE_SLOW <= set(BASELINE_KEYS)


def _peel(content, columns):
    """``content`` as a sum of ``columns``, or ``None`` if it is not one.

    An exact search over non-negative integer combinations, memoized on what is
    left: the vectors are a handful of labels with multiplicities of one or two,
    so the whole space is tiny.
    """
    keys = sorted({lab for one in [content, *columns.values()] for lab in one})
    cols = {name: tuple(one.get(lab, 0) for lab in keys)
            for name, one in columns.items()}
    seen: dict = {}

    def go(rest):
        if not any(rest):
            return ()
        if rest in seen:
            return seen[rest]
        seen[rest] = None
        for name, column in cols.items():
            if any(column) and all(a <= b for a, b in zip(column, rest)):
                sub = go(tuple(b - a for a, b in zip(column, rest)))
                if sub is not None:
                    seen[rest] = (name,) + sub
                    break
        return seen[rest]

    return go(tuple(content.get(lab, 0) for lab in keys))


@pytest.mark.parametrize("key", [
    pytest.param(key, marks=pytest.mark.slow if key in ORACLE_SLOW else pytest.mark.data)
    for key in BASELINE_KEYS])
def test_every_ebr_column_at_a_fixture_position_comes_from_its_computed_rows(key, realized):
    """The induction oracle: the tables' own columns, rebuilt from measured rows.

    For every group the baseline gate's fixtures reach, and at every Wyckoff
    position one of their sites sits at, the band representations the computed
    site-group rows induce — at every high-symmetry point that presentation can
    sample — account for **every** EBR column the tables carry at that position.
    The claim is about a (group, position, valuedness), so one fixture of the
    gate stands for each and :data:`ORACLE_FIXTURES` says which.
    Nothing here reads a name: it compares content with content, so a wrong
    site-displacement phase, a wrong conjugation or a wrong spin lift moves a
    multiplicity and the comparison fails.

    A spinless group's rows are the single-valued site irreps and the map onto
    the columns is one to one.  A spinful group's rows are single-valued too —
    the character mathematics is the same — and what they induce is the row
    times the operations' own spin-½ lift, which is a **sum** of the position's
    double-valued columns: ``E`` at Kane-Mele's ``2b`` carries ``-E1`` and
    ``-1E2E`` together, and the two ``T`` rows of the diamond's ``8a`` each
    carry ``-F`` with one of ``-E1``, ``-E2``.  Both are checked the same way,
    by decomposing each row's content over the columns and asking that the
    columns together are covered.

    The bond-decorated diamonds sit at a position R-3m's EBR table does not
    carry at all, which is asserted rather than skipped.
    """
    r = _baseline_realized(key, realized)
    assert r, f"{key}: {r}"
    group, assignment = r.group, r.assignment
    rows = tqe.ebr_rows(group)
    assert rows, f"{key}: {rows}"
    points = tqg.high_symmetry_points(group)
    assert points, f"{key}: {points}"
    syms = group.symmetries
    place = {int(s.ind): p for p, s in enumerate(syms)}
    kept = {int(i) for i in r.validated.indices}
    lift = ({int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
             for s in syms} if group.spinor else None)

    seen, checked = set(), 0
    for site in range(len(assignment.site_group)):
        letter = assignment.wyckoff_tables[site]
        fixers = tuple(int(i) for i in assignment.site_group[site]
                       if not syms[place[int(i)]].time_reversal and int(i) in kept)
        anti = tuple(int(i) for i in assignment.site_group[site]
                     if syms[place[int(i)]].time_reversal and int(i) in kept)
        if (fixers, anti, letter) in seen:
            continue
        seen.add((fixers, anti, letter))
        columns = {name: {lab: mult for lab, mult in content.items()
                          if tqe.kpoint_of_label(lab) in points}
                   for name, content in tqe._rows_at_letter(rows, letter).items()}
        if not columns:
            assert letter not in {tqe._letter_only(name.split("@")[1]) for name in rows}, \
                f"{key}: site {site} sits at {letter}, which the EBR table does carry"
            continue
        table = tqc_sitechars.site_group_table(group, fixers, antiunitary=anti)
        used = set()
        for row in range(len(table.labels)):
            chi = {int(i): (table.character_on(row, int(i))
                            * (1.0 if lift is None else lift[int(i)]))
                   for i in table.indices}
            content: dict = {}
            for name in points:
                content.update(tqe._induced_content(
                    group, assignment, site, chi,
                    k=None if name == "GM" else points[name], kpname=name))
            parts = _peel(content, columns)
            assert parts is not None, (
                f"{key}: site {site} ({letter}) row {table.labels[row]} induces "
                f"{content}, which is no sum of the columns {sorted(columns)}")
            used.update(parts)
        assert used == set(columns), (
            f"{key}: site {site} ({letter}) leaves {sorted(set(columns) - used)} "
            "unaccounted for")
        checked += 1
    assert seen, f"{key}: no site of this fixture was examined"
