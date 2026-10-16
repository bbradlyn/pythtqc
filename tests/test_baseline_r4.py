"""The R4 baseline: every served field of the goldens, pinned field for field.

This module exists for one reason — wave P1 is a **performance** wave, and its
hard rule is that no result may change.  ``tests/data/r4_baseline.json`` was
captured on the R4e tree before anything was touched (the commit that closed the
R series), and every entry here recomputes the same call and compares the answer
against that file field for field: the group's number and name, its size and the
kept count, the site labels, the corep content at every high-symmetry point of
the fixture's own occupation, the symmetry indicator's values and class, the
cross-check outcome and its row counts, the descent summary (``from``, ``to``,
``n_dropped`` and the per-step ``refuted_by`` histogram), the branch
tie, the product residual and the orbital centres.

What is pinned and what is only bounded
---------------------------------------
:data:`RESIDUAL_TOL` is an **absolute** 1e-12 window and :func:`compare` applies
it to every float in a record.  Two fields live orders of magnitude below it and
are therefore *bounded* rather than digit-pinned — a change inside the window is
invisible, and naming them is the honest form of "no result may change":

``product_residual``
    42 of the 55 records hold one -- every record outside the ``srsi`` section;
    measured 0.0 (7 of them) to 3.6e-15, more than two decades below the
    window, so any value in ``[0, 1e-12]`` passes.  That is deliberate: the
    recomputed residual drifts by tens of percent between machines (1.577e-15
    against the stored 1.196e-15 for ``kane_mele|free-standing``), so pinning its
    digits would make the gate machine-dependent rather than strict.
``induced.max_dev``
    the 8 ``ebr`` records' measured ``D(g) = λ(g)·S(g)`` deviation, 2.4e-16 …
    3.5e-16.  Its companions in the same record — ``n_unitary`` and the whole
    ``lambda`` list — are exact integers and *are* pinned.

Every other field is pinned: the integers, the names, the labels, the histograms,
and ``orb`` (the only other float, served rounded to 12 decimals, on which the
same window is an exact comparison).  For the two residual fields the gate says
only that they stayed at roundoff.

``kept`` and ``n_ops_represented`` are not two further facts beside ``size``:
measured over the 42 records that carry them, none a refusal, all three are
equal in 42 of 42, because ``realized_group`` settles on the group the
Hamiltonian obeys and ``_build_rep`` represents all of it.  They are the **settle
invariant**, pinned so that a descent which stops early becomes visible.

``occ`` is the baseline's *question*, not its answer: :func:`compare` skips it and
``snapshot`` is handed it, so the file silently chooses the occupation each record
is read at.  Whether a fixture is gapped there is not asked here: the lin3g and
Fu-Kane-Mele examples' goldens print their gaps, and ``test_fixtures_smoke.py``
checks the published ones of the hexagonal fragile model and the ``p_x, p_y``
honeycomb.

Four sections, each captured in both readings where the fixture is 2D:

``layer``
    eleven 2D fixtures, in the free-standing and the substrate reading.
    ``tqc_models.kane_mele_inversion()`` is ``kane_mele(0.0, -1.0, 0.3, 0.0)``,
    the model the ``kane_mele`` entry builds with the same keywords, so its
    records would be those records field for field and it has none of its own.
``r1``
    the fixtures of the R1 baseline (``scratchpad/r1_baseline.json``), which the
    orchestrator's verification scripts have compared against since wave R1.
``ebr``
    the **pure-EBR self-decomposition**: one spinful ``s`` at ``1a`` of a square
    layer serves ``{-GM7, -M7, -X5}`` — the tabulated ``Ē₁@1a`` row, not
    ``Ē₂@1a``'s ``{-GM6, -M6, -X5}`` — and one at the origin of a rectangular
    layer serves ``{-GM5, -S5, -X5, -Y5}``, with and without a Rashba term.
    That the served content **is** the induced representation's is the check,
    and each entry also carries the measured ``D(g) = λ(g)·S(g)`` deviation
    against the tables' own lifts.  The four ``mm2_s`` records were captured
    again once the builders were called by keyword: a bare number had been read
    as the ``x`` hopping, which left the ``soc=0.2`` pair with no Rashba term
    and the ``soc=0`` pair with no ``x`` bond.
``srsi``
    the SRSI outputs of the SRSI examples' models: θ vectors, the splitting
    verdicts and the consistent pairs, plus the tables of SGs 211/147/162/163 in
    both valuednesses.  The pair *lists* are empty for almost all of these groups
    (measured: ``pairs`` is ``[]`` in 7 of the 8 ``tables_*`` records — only
    ``tables_211_single`` has one — and ``consistent_pairs`` in 4 of the 6
    splitting records), so what carries there is the adjacent count
    (``n_pairs == 0``) and not the serialized pair content.

Every record is read through ``conftest``'s ``realized``, which makes each call
once a session, with named builders (:func:`record_call` spells each call), so a
record the session also reads elsewhere is measured once: the undistorted and
bond-decorated diamonds, the flat-band layers, Kane-Mele's ``p_z`` layer read
free-standing and the declared lin3g are calls other modules' fixtures make, the
bond-decorated phases serve the ``r1`` and ``srsi`` sections from one
measurement each, and ``test_characters``' EBR-column oracle reads 18 of these
records.

A field that comes back different is a **result changed**, whatever it cost, and
that is what this file is here to catch.
"""

import io
import json
import os
from fractions import Fraction

import numpy as np
import pytest
from pythtb import Lattice, TBModel
from pythtb.models import kane_mele

import tqc_models
from conftest import FKM_BOND, fkm_regime, km_z2_regime, p41_band
from pythtqc import ebr as tqe
from pythtqc import sewing as tqw
from pythtqc import srsi as tqr
from pythtqc.refusal import Refusal

#: A gate: the module runs whole, with ``-m gates``, before a commit, and a bare
#: ``pytest`` leaves it out.
pytestmark = pytest.mark.gates

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "r4_baseline.json")

with io.open(BASELINE_PATH, encoding="utf-8") as _fh:
    BASELINE = json.load(_fh)

#: The **absolute** window every float in a record is compared in.  On the two
#: residual fields (``product_residual``, ``induced.max_dev``) it is a *ceiling*
#: three or more decades above the quantity and not a pin on its digits; on ``orb``,
#: served rounded to 12 decimals, it is an exact comparison.  See the module
#: docstring's "What is pinned and what is only bounded".
RESIDUAL_TOL = 1e-12


# ---------------------------------------------------------------------------
# the gate's own builders
# ---------------------------------------------------------------------------
#
# These four are this file's.  `test_layer.py` has a `square_s_pz` of the same
# name and a different model: its parity-mixing term is an on-site s-p_z
# coupling, which the free-standing reading keeps in `P4/mmm1' #123.340`, where
# the one below is an antisymmetric bond along x (with other on-site energies
# and hoppings as well), which the free-standing reading measures as
# `Pmmm1' #47.250`.

def square_s_pz(mix=0.0):
    """s and p_z on one site of a square lattice; ``mix`` couples them along x."""
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]],
                  periodic_dirs=[0, 1])
    m = TBModel(lat, spinful=False)
    m.set_onsite([0.0, 1.0])
    m.set_hop(-1.0, 0, 0, [1, 0])
    m.set_hop(-1.0, 0, 0, [0, 1])
    m.set_hop(0.4, 1, 1, [1, 0])
    m.set_hop(0.4, 1, 1, [0, 1])
    if mix:
        m.set_hop(mix, 0, 1, [1, 0])
        m.set_hop(-mix, 1, 0, [1, 0])
    return m


def buckled_graphene(d=0.15, c=30.0):
    """A ``dim_r = 3`` buckled honeycomb slab, read from its real geometry."""
    lat = Lattice([[1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0],
                   [0.0, 0.0, float(c)]],
                  [[1.0 / 3.0, 1.0 / 3.0, float(d) / float(c)],
                   [2.0 / 3.0, 2.0 / 3.0, -float(d) / float(c)]],
                  periodic_dirs=[0, 1])
    m = TBModel(lat, spinful=False)
    m.set_onsite([0.0, 0.0])
    for lv in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        m.set_hop(-1.0, 0, 1, lv)
    return m


def p4mm_spinful_s(rashba=0.0):
    """One spinful ``s`` at ``1a`` of a square layer: a pure EBR of ``p4mm``."""
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]],
                        periodic_dirs=[0, 1]), spinful=True)
    m.set_onsite([0.0])
    m.set_hop(-1.0, 0, 0, [1, 0])
    m.set_hop(-1.0, 0, 0, [0, 1])
    if rashba:
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        m.set_hop(1j * rashba * sy, 0, 0, [1, 0], mode="add")
        m.set_hop(-1j * rashba * sx, 0, 0, [0, 1], mode="add")
    return m


def mm2_spinful_s(tx=-1.0, ty=-0.6, soc=0.0):
    """One spinful ``s`` at the origin of a rectangular layer: a pure EBR of ``mm2``."""
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.3]], [[0.0, 0.0]],
                        periodic_dirs=[0, 1]), spinful=True)
    m.set_onsite([0.0])
    m.set_hop(tx, 0, 0, [1, 0])
    m.set_hop(ty, 0, 0, [0, 1])
    if soc:
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        m.set_hop(1j * soc * sy, 0, 0, [1, 0], mode="add")
        m.set_hop(-1j * soc * sx, 0, 0, [0, 1], mode="add")
    return m


# ---------------------------------------------------------------------------
# the snapshot, identical to the one the file was captured with
# ---------------------------------------------------------------------------

def _num(x):
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (str, int, float, bool)) or x is None:
        return _num(x)
    v = _num(x)
    return v if isinstance(v, (str, int, float, bool)) or v is None else repr(x)


def _descent_summary(r):
    d = r.descent
    if d is None:
        return None
    return {
        "from": {k: _num(v) for k, v in d["from"].items()},
        "to": {k: _num(v) for k, v in d["to"].items()},
        "n_dropped": len(d["dropped"]),
        "iterations": int(d["iterations"]),
        "steps": [{"refuted_by": {str(k): int(v) for k, v in
                                  (s.get("refuted_by") or {}).items()},
                   "from": str(s["from"]["number_str"]),
                   "to": str(s["to"]["number_str"]),
                   "n_dropped": int(s["n_dropped"])}
                  for s in d["steps"]],
    }


def snapshot(r, occ):
    """Every served field of one ``realized_group`` answer, as the file holds it."""
    out = {
        "number_str": str(r.group.number_str), "name": str(r.group.name),
        "size": int(r.group.size), "kept": len(r.validated.indices),
        "labels": list(r.rep.characters.irreps),
        "product_residual": float(r.rep.product_residual),
        "time_reversal": r.time_reversal.get("outcome"),
        "orb": np.round(np.asarray(r.model.orb_vecs, float), 12).tolist(),
        "environment": None if r.environment is None else str(r.environment),
        # the orientation branch is pinned by whether the build settled the
        # tie between the two branches by convention (both True and False occur)
        "branch_tie": _num(getattr(r.rep, "_branch_tie", None)),
        # kept == size == n_ops_represented in 42 of the 42 records: the settle
        # invariant, not three independent facts (see the module docstring)
        "n_ops_represented": len(getattr(r.rep, "matrices", {}) or {}),
    }
    w = r.cross_check
    out["cross_check"] = None if w is None else {
        "outcome": None if w.get("outcome") is None else str(w["outcome"]),
        "agrees": w.get("agrees"),
        "free_standing": (None if w.get("free_standing") is None
                          else str(w["free_standing"]["number_str"])),
        "substrate": (None if w.get("substrate") is None
                      else str(w["substrate"]["number_str"])),
        "n_rows": len(w.get("rows") or ()),
        "missing": len(w.get("missing") or ()), "extra": len(w.get("extra") or ()),
        "flipping": len(w.get("flipping") or ()),
        "unmapped": len(w.get("unmapped") or ()),
        "refusal_gate": (None if not w.get("refusal") else str(w["refusal"]["gate"])),
    }
    out["descent"] = _descent_summary(r)
    t = tqw.hsp_table(r.rep, bands=occ)
    if isinstance(t, Refusal):
        out["content"] = {"__refusal__": str(t.gate)}
        out["indicator"] = {"__refusal__": str(t.gate)}
        return out
    out["content"] = {str(k): int(v) for k, v in t.counts.items()}
    try:
        si = tqe.symmetry_indicator(t, r.group, rep=r.rep,
                                    bands=occ if isinstance(occ, int) else None)
    except Exception as exc:      # a served crash is a served fact; see the report
        out["indicator"] = {"__exception__": type(exc).__name__,
                            "args": str(exc)[:200]}
        return out
    if isinstance(si, Refusal):
        out["indicator"] = {"__refusal__": str(si.gate)}
    else:
        # `si.structure` is a *string* ("trivial", "Z6 x Z12"), so `list(...)`
        # explodes it per character and the baseline holds
        # ["t","r","i","v","i","a","l"].  It still discriminates -- a different
        # class gives a different list, and a length change is reported -- but the
        # diff reads per character and the capture is unreadable.  `str(...)` is
        # what this wants; changing it would have to be captured afresh, so the
        # shape stays frozen with the file.
        out["indicator"] = {"values": [int(v) for v in si.values],
                            "structure": _jsonable(list(si.structure)),
                            "order": _num(si.order)}
    return out


def compare(got, want, where=""):
    """Field-for-field, floats inside :data:`RESIDUAL_TOL`; returns the differences.

    Symmetric on dict keys (``set(want) | set(got)``), so a field served now but
    absent from the baseline is reported as loudly as the reverse.  The float
    branch is a *window*, not a pin — see :data:`RESIDUAL_TOL` for which fields
    that actually bounds and which it compares exactly.
    """
    bad = []
    if isinstance(want, dict) and isinstance(got, dict):
        for key in sorted(set(want) | set(got)):
            if key in ("seconds", "occ"):
                # a timing and the occupation this run was asked for: an input
                # and a measurement of the machine, neither a served result
                continue
            if key not in want:
                bad.append(f"{where}.{key}: served now, absent from the baseline "
                           f"({got[key]!r})")
            elif key not in got:
                bad.append(f"{where}.{key}: in the baseline ({want[key]!r}), "
                           "not served now")
            else:
                bad += compare(got[key], want[key], f"{where}.{key}")
        return bad
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return [f"{where}: length {len(got)} against the baseline's {len(want)}"]
        for i, (g, w) in enumerate(zip(got, want)):
            bad += compare(g, w, f"{where}[{i}]")
        return bad
    if isinstance(want, float) or isinstance(got, float):
        try:
            if abs(float(got) - float(want)) <= RESIDUAL_TOL:
                return []
        except (TypeError, ValueError):
            pass
        return [f"{where}: {got!r} against the baseline's {want!r}"]
    if got != want:
        return [f"{where}: {got!r} against the baseline's {want!r}"]
    return []


# ---------------------------------------------------------------------------
# the calls, one per record, through the session's shared records
# ---------------------------------------------------------------------------
#
# `realized(builder, **keywords)` is conftest's session cache: the call
# `realized_group(builder(), **keywords)` is made once a session, keyed on where
# the builder is written, what it closes over, and the keywords.  A builder
# written out again in another file is therefore another record, so the
# builders below are named once -- the very objects conftest's fixtures and the
# other modules pass where they make the same call, this file's otherwise -- and
# `record_call` spells each record's call for this file and for
# `test_characters`' EBR-column oracle alike.

def record_call(section, key):
    """``(builder, keywords)`` of one record, read as ``realized(builder, **keywords)``.

    ``section`` is ``"layer"``, ``"r1"`` or ``"ebr"`` and ``key`` the record's
    key there, whose last field is the reading where the fixture is 2D.
    """
    if section == "layer":
        name, environment = key.split("|")
        build, kw = LAYER_2D[name]
        return build, dict(kw, environment=environment)
    if section == "r1":
        name, _, environment = key.partition("|")
        build, kw = R1_BUILDERS[name]
        return build, (dict(kw, environment=environment) if environment else dict(kw))
    if section == "ebr":
        name, soc_word, environment = key.split("|")
        return (ebr_layer(name, float(soc_word.split("=")[1])),
                dict(environment=environment))
    raise KeyError(f"{section!r} is not a section of the baseline")


# ---------------------------------------------------------------------------
# section 1: the 2D fixtures, both readings
# ---------------------------------------------------------------------------

LAYER_2D = {
    "kane_mele": (lambda: kane_mele(0.0, -1.0, 0.3, 0.0), {}),
    "kane_mele_rashba": (lambda: kane_mele(0.0, -1.0, 0.3, 0.25), {}),
    "km_odd": (lambda: tqc_models.kane_mele_z2("odd"), dict(typat=[0, 1])),
    "lin3g": (lambda: tqc_models.lin3g(), dict(typat=[0, 0, 1, 1])),
    "pwv_fragile": (lambda: tqc_models.pwv_fragile(), {}),
    "pxpy": (lambda: tqc_models.pxpy_honeycomb(),
             dict(typat=[0, 0, 0, 0])),
    # the builders conftest's `p41_free_standing`, `p41_substrate` and
    # `p41_chiral_substrate` pass: three of these four records are those
    "p41_flat_band": (tqc_models.p41_flat_band, {}),
    "p41_chiral": (p41_band(True), {}),
    "square_s_pz": (lambda: square_s_pz(0.0), {}),
    "square_s_pz_mixed": (lambda: square_s_pz(0.3), {}),
    "buckled_graphene": (buckled_graphene, {}),
}


@pytest.mark.parametrize("key", sorted(BASELINE["layer"]))
def test_the_2d_fixtures_serve_the_r4_baseline_in_both_readings(key, realized):
    """Every field of every 2D golden, in both readings, against the R4 capture."""
    build, kw = record_call("layer", key)
    want = BASELINE["layer"][key]
    r = realized(build, **kw)
    if "__refusal__" in want:
        assert isinstance(r, Refusal), f"{key} refused {want['__refusal__']} at R4"
        assert str(r.gate) == want["__refusal__"]
        return
    assert not isinstance(r, Refusal), f"{key}: {r}"
    bad = compare(snapshot(r, want["occ"]), want, key)
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------
# section 2: the R1 baseline's own fixtures
# ---------------------------------------------------------------------------

#: Each fixture's model builder and keywords; a 2D record adds the reading its
#: key names.  Where the session makes the same call elsewhere the builder is the
#: one that call passes: the undistorted diamond is conftest's
#: ``fkm_undistorted``, Kane-Mele's ``p_z`` layer read free-standing
#: ``km_inversion_free_standing``, the declared lin3g in the default reading (the
#: ``srsi`` section's) ``lin3g_declared_cross_checked``, and the bond-decorated
#: phases, in the bond-centre crystal field ``FKM_BOND``, are records several
#: modules read.  ``fu_kane_mele_z2(regime)`` is
#: ``fu_kane_mele(1.0, 0.125, dt)`` with the regime's ``dt``, and
#: ``kane_mele_inversion()`` is ``kane_mele(0.0, -1.0, 0.3, 0.0)``.
R1_BUILDERS = {
    "lin3g": (tqc_models.lin3g,
              dict(typat=[0, 0, 1, 1], characters=["s", "dx2-y2", "s", "s"])),
    "km_odd": (km_z2_regime("odd"), dict(typat=[0, 1], characters={0: "A1"})),
    "km_even": (km_z2_regime("even"), dict(typat=[0, 1], characters={0: "A1"})),
    "fkm_diamond": (fkm_regime("uniform"), dict(characters={0: "s"})),
    "km_layer_sigma_h": (tqc_models.kane_mele_inversion, dict(characters={0: "pz"})),
    "fkm_bond_strong": (fkm_regime("strong"),
                        dict(decorations=FKM_BOND, characters={0: "s"})),
    "fkm_bond_weak": (fkm_regime("weak"),
                      dict(decorations=FKM_BOND, characters={0: "s"})),
    "fkm_bond_trivial": (fkm_regime("trivial"),
                         dict(decorations=FKM_BOND, characters={0: "s"})),
}


@pytest.mark.parametrize("key", sorted(BASELINE["r1"]))
def test_the_r1_fixtures_serve_the_r4_baseline(key, realized):
    """The R1 goldens, every field, 2D ones in both readings."""
    build, kw = record_call("r1", key)
    want = BASELINE["r1"][key]
    r = realized(build, **kw)
    if "__refusal__" in want:
        assert isinstance(r, Refusal), f"{key} refused {want['__refusal__']} at R4"
        assert str(r.gate) == want["__refusal__"]
        return
    assert not isinstance(r, Refusal), f"{key}: {r}"
    bad = compare(snapshot(r, want["occ"]), want, key)
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------
# section 3: the pure-EBR self-decomposition
# ---------------------------------------------------------------------------

#: Each builder called by keyword: ``p4mm_spinful_s`` names its Rashba strength
#: ``rashba``, and ``mm2_spinful_s`` names it ``soc`` after the ``x`` and ``y``
#: hoppings it takes first, so a bare positional number there is the ``x`` hopping.
EBR_BUILDERS = {"p4mm_s_1a": lambda soc: p4mm_spinful_s(rashba=soc),
                "mm2_s": lambda soc: mm2_spinful_s(soc=soc)}


def ebr_layer(name, soc):
    """A named zero-argument builder for one pure-EBR layer at one Rashba strength.

    The closure cells tell the layers and their strengths apart in
    ``realized``'s key, as ``conftest.fkm_regime``'s tell the Fu-Kane-Mele
    regimes apart.
    """
    def build():
        return EBR_BUILDERS[name](soc)
    build.__qualname__ = f"ebr_layer.{name}"
    return build


def _induced_record(r):
    """``D(g)`` against ``λ(g)·S(g)`` on every represented unitary operation."""
    devs, lam = [], []
    mats = getattr(r.rep, "matrices", {}) or {}
    for sym in r.group.symmetries:
        if sym.time_reversal or int(sym.ind) not in mats:
            continue
        D = np.asarray(mats[int(sym.ind)], dtype=complex)
        S = np.asarray(sym.spinor_rotation, dtype=complex)
        dp, dm = float(np.abs(D - S).max()), float(np.abs(D + S).max())
        devs.append(min(dp, dm))
        lam.append(1 if dp <= dm else -1)
    return {"n_unitary": len(devs), "max_dev": max(devs) if devs else None,
            "lambda": lam}


@pytest.mark.parametrize("key", sorted(BASELINE["ebr"]))
def test_a_pure_ebr_serves_its_own_induced_representation(key, realized):
    """The served content of a single-orbital EBR is the induced representation's.

    A spinful ``s`` at ``1a`` of ``p4mm`` serves ``{-GM7, -M7, -X5}`` — the
    tabulated ``Ē₁@1a`` row, and not ``Ē₂@1a``'s ``{-GM6, -M6, -X5}`` — and one
    at the origin of an ``mm2`` layer serves ``{-GM5, -S5, -X5, -Y5}``, with and
    without a Rashba term, in both readings.  Beside the content, every
    represented unitary operation's matrix is measured against that operation's
    own tabulated lift up to a real character.
    """
    build, kw = record_call("ebr", key)
    want = BASELINE["ebr"][key]
    r = realized(build, **kw)
    if "__refusal__" in want:
        assert isinstance(r, Refusal), f"{key} refused {want['__refusal__']} at R4"
        assert str(r.gate) == want["__refusal__"]
        return
    assert not isinstance(r, Refusal), f"{key}: {r}"
    got = snapshot(r, want["occ"])
    got["induced"] = _induced_record(r)
    bad = compare(got, want, key)
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------
# section 4: the SRSI outputs of the SRSI examples' models
# ---------------------------------------------------------------------------

def _srsi_model_record(ms):
    if isinstance(ms, Refusal):
        return {"__refusal__": str(ms.gate)}
    out = {"theta_Z": [_num(v) for v in ms.theta_Z]}
    for attr in ("theta_Zn", "names", "moduli", "structure", "number_str", "spinor"):
        v = getattr(ms, attr, None)
        if v is not None:
            out[attr] = _jsonable(v)
    out["orbits"] = _jsonable([
        {k: _jsonable(v) for k, v in o.items()
         if k in ("letter", "wyckoff", "site_irrep", "irrep", "multiplicity", "n")}
        if isinstance(o, dict) else str(o) for o in (ms.orbits or ())])
    return out


def _srsi_bands_record(b):
    if isinstance(b, Refusal):
        return {"__refusal__": str(b.gate)}
    out = {"theta_Z": [_num(v) for v in b.theta_Z], "stable": _num(b.stable)}
    for attr in ("names", "denominator", "p", "n_bands", "number_str"):
        v = getattr(b, attr, None)
        if v is not None:
            out[attr] = _jsonable(v)
    ind = getattr(b, "indicator", None)
    out["indicator_values"] = (None if ind is None or isinstance(ind, Refusal)
                               else _jsonable(list(ind.values)))
    return out


def _splitting_record(s):
    if isinstance(s, Refusal):
        return {"__refusal__": str(s.gate)}
    out = {}
    for attr in ("number_str", "spinor", "verdict", "nsit", "theta_Z_total",
                 "theta_Zn_total", "theta_Z_v", "theta_Z_c", "moduli",
                 "v_atomic_possible", "c_atomic_possible", "n_bands_v",
                 "n_bands_c", "same_content"):
        v = getattr(s, attr, None)
        if v is not None:
            out[attr] = _jsonable(v)
    out["n_consistent_pairs"] = len(getattr(s, "consistent_pairs", ()) or ())
    out["consistent_pairs"] = _jsonable(
        list(getattr(s, "consistent_pairs", ()) or ())[:12])
    out["repr"] = repr(s)[:800]
    return out


def _srsi_case(key, realized):
    """Recompute one entry of the ``srsi`` section."""
    if key.startswith("fkm_"):
        # the ``r1`` section's bond-decorated phase: one record for both sections
        build, kw = R1_BUILDERS["fkm_bond_" + key[len("fkm_"):]]
        r = realized(build, **kw)
        if isinstance(r, Refusal):
            return {"__refusal__": str(r.gate)}
        t2 = tqw.hsp_table(r.rep, bands=2)
        return {"srsi_of_model": _srsi_model_record(tqr.srsi_of_model(r)),
                "srsi_of_bands_2": _srsi_bands_record(
                    tqr.srsi_of_bands(t2, r.group, rep=r.rep, bands=2)),
                "splitting_2": _splitting_record(tqr.splitting(r, bands=2))}
    if key == "lin3g":
        # the ``r1`` section's declaration, in the default reading
        build, kw = R1_BUILDERS["lin3g"]
        r = realized(build, **kw)
        if isinstance(r, Refusal):
            return {"__refusal__": str(r.gate)}
        return {"srsi_of_model": _srsi_model_record(tqr.srsi_of_model(r)),
                "splitting_6": _splitting_record(tqr.splitting(r, bands=6)),
                "splitting_4": _splitting_record(tqr.splitting(r, bands=4))}
    if key == "p41_chiral":
        # the chiral layer of section 1, in the default reading
        r = realized(LAYER_2D["p41_chiral"][0])
        if isinstance(r, Refusal):
            return {"__refusal__": str(r.gate)}
        tv = tqw.hsp_table(r.rep, bands=2)
        return {"number_str": str(r.group.number_str),
                "srsi_of_model": _srsi_model_record(tqr.srsi_of_model(r)),
                "srsi_of_bands_2": _srsi_bands_record(
                    tqr.srsi_of_bands(tv, r.group, rep=r.rep, bands=2)),
                "splitting_2": _splitting_record(tqr.splitting(r, bands=2))}
    _, num, valuedness = key.split("_")
    tb = tqr.srsi_tables(int(num), spinor=(valuedness == "double"))
    if isinstance(tb, Refusal):
        return {"__refusal__": str(tb.gate)}
    pairs = tqr.stably_equivalent_pairs(tb)
    return {"n_rows": len(getattr(tb, "rows", ()) or ()),
            "names": _jsonable(list(getattr(tb, "names", ()) or ())),
            "moduli": _jsonable(list(getattr(tb, "moduli", ()) or ())),
            "repr": repr(tb)[:400],
            "n_pairs": len(pairs), "pairs": _jsonable(list(pairs)[:40])}


@pytest.mark.parametrize("key", sorted(BASELINE["srsi"]))
def test_the_srsi_outputs_serve_the_r4_baseline(key, realized):
    """θ vectors, splitting verdicts, consistent pairs and the tables themselves."""
    bad = compare(_srsi_case(key, realized), BASELINE["srsi"][key], key)
    assert not bad, "\n".join(bad)
