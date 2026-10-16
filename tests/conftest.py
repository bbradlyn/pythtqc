"""Fixtures for the ``pythtqc`` test suite, and the ``sys.path`` setup it needs.

The suite runs against the **in-repo** ``src/pythtqc``, with no install step:
this file puts ``src/`` on ``sys.path`` before anything imports the package.

The golden builders live in ``fixture_models.py`` and are loaded **by path**
under the name ``tqc_models`` (``import tqc_models``), not by ordinary import:
pytest prepends this directory to ``sys.path`` when collecting, so a
by-path load is what makes the test imports independent of ``sys.path`` order
and of any same-named module elsewhere in the environment.
"""

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

# this repo's package (pythtqc), run from the checkout rather than an install
sys.path.insert(0, os.path.join(_HERE, "..", "src"))


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register before exec so self-reference works
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _load_tqc_models():
    if "tqc_models" in sys.modules:
        return sys.modules["tqc_models"]
    return _load_by_path("tqc_models", os.path.join(_HERE, "fixture_models.py"))


tqc_models = _load_tqc_models()


def pytest_configure(config):
    """Silence the magnetic-scope notice for the whole suite.

    Many fixtures here break time reversal on purpose, so the notice would say
    the same thing over and over; the test that asserts it uses
    ``pytest.warns``, which records the warning whatever this filter says.  The
    filter is added here rather than in ``pyproject.toml``'s ``filterwarnings``:
    naming the class there makes pytest import the package while it reads its
    configuration, before ``pytest-cov`` has started, and every line that runs at
    import -- each ``def``, ``class`` and module-level statement -- then goes
    unrecorded by a coverage run.
    """
    config.addinivalue_line("filterwarnings",
                            "ignore::pythtqc.refusal.MagneticScopeWarning")


# ---------------------------------------------------------------------------
# the session-scoped realized-group cache
# ---------------------------------------------------------------------------
#
# 337 top-level `realized_group` calls run over this suite and only 203 of them
# are distinct: 134 are the same builder with the same keyword arguments, rebuilt
# because the module-scoped fixture that holds it lives in one file and the next
# file wants the same answer (measured 2026-09-06, before this cache; the twelve
# worst are the lin3g quartet at 11 rebuilds, free-standing Kane-Mele at 9, the
# bond-decorated Fu-Kane-Mele phases at 9, and the two #156 regimes at 8 each).
# `realized_group` is a measurement of a Hamiltonian and its answer is a frozen
# record, so the second call can only produce what the first one did.
#
# `realized_once(builder, **kwargs)` runs `realized_group(builder(), **kwargs)`
# at most once per session and hands the SAME record to every caller after that.
# It is keyed on the builder — its module, qualified name, definition line, and
# the closure cells and default arguments a `lambda` in a parametrize list
# carries its parameters in — together with the keyword arguments rendered
# stably, so two builders that look alike but close over different numbers get
# different entries.  A cache miss is never silent: what comes back is whatever
# the call returned, refusals included.
#
# The model must come from the builder rather than from a model fixture: a
# fixture value has no name to key on, and it is the (builder, kwargs) pair that
# names the call.

_REALIZED: dict = {}

#: What the cache did, so the saving is a measurement and not a claim:
#: ``calls - entries`` is the number of ``realized_group`` builds it removed.
_REALIZED_STATS = {"calls": 0, "entries": 0}


def _render(value):
    """A stable rendering of one keyword argument."""
    if isinstance(value, dict):
        return {str(k): _render(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [_render(v) for v in value]
    if hasattr(value, "tolist"):
        return _render(value.tolist())
    return repr(value)


def _builder_key(builder):
    """What names a builder: where it is written, and what it closes over."""
    code = getattr(builder, "__code__", None)
    closure = getattr(builder, "__closure__", None)
    return (
        getattr(builder, "__module__", "?"),
        getattr(builder, "__qualname__", repr(builder)),
        None if code is None else (code.co_filename, code.co_firstlineno),
        () if not closure else tuple(repr(c.cell_contents) for c in closure),
        repr(getattr(builder, "__defaults__", None)),
    )


def realized_once(builder, **kwargs):
    """``realized_group(builder(), **kwargs)``, computed at most once per session.

    ``builder`` is a **callable that returns the model** — ``tqc_models.lin3g``,
    not the ``lin3g`` fixture — because the key is the call, and a fixture value
    has no name.  Everything else is forwarded verbatim.
    """
    import pythtqc.groups as _tqg

    key = (_builder_key(builder), repr(_render(dict(sorted(kwargs.items())))))
    _REALIZED_STATS["calls"] += 1
    if key not in _REALIZED:
        _REALIZED[key] = _tqg.realized_group(builder(), **kwargs)
        _REALIZED_STATS["entries"] = len(_REALIZED)
    return _REALIZED[key]


@pytest.fixture(scope="session")
def realized():
    """The session-scoped :func:`realized_once`, as a fixture."""
    return realized_once


def fkm_regime(regime):
    """A **named** zero-argument builder for one Fu-Kane-Mele regime.

    ``realized_once`` keys on the builder, and a ``lambda`` written in one test
    file is a different builder from an identical ``lambda`` in the next one.
    The four regimes therefore share one builder here — the closure cell is
    what tells them apart — so the same distorted diamond is measured once for
    the whole session however many files ask for it.
    """
    def build():
        return tqc_models.fu_kane_mele_z2(regime)
    build.__qualname__ = f"fkm_regime.{regime}"
    return build


def km_z2_regime(regime):
    """A named zero-argument builder for one ``#156`` Kane-Mele regime."""
    def build():
        return tqc_models.kane_mele_z2(regime)
    build.__qualname__ = f"km_z2_regime.{regime}"
    return build


def p41_band(chiral):
    """A named zero-argument builder for the SRSI paper's flat-band model.

    ``chiral=True`` adds the next-nearest pattern that breaks the vertical
    mirrors.  The closure cell tells the two apart in ``realized_once``'s key,
    as ``fkm_regime``'s does for the Fu-Kane-Mele regimes.
    """
    def build():
        return tqc_models.p41_flat_band(chiral=chiral)
    build.__qualname__ = f"p41_band.{'chiral' if chiral else 'plain'}"
    return build


#: The bond-centre crystal field the distorted Fu-Kane-Mele phases are declared
#: with: one non-orbital species at the midpoint of the detuned bond, which names
#: ``R-3m1' #166.98`` outright from the cell.  ``examples/tqc_fkm_diamond.py``
#: reads the same phases without it, by descent from ``Fd-3m1' #227.129``.
FKM_BOND = (([0.125, 0.125, 0.125], "bond"),)


# ---------------------------------------------------------------------------
# the records every module shares
# ---------------------------------------------------------------------------
#
# One fixture per call, made once per session -- through `realized_once` where
# the call is `realized_group` -- and handed to every test that asks for it by
# name.  Each call's keyword arguments are spelled here once, because two
# spellings of one call are two cache entries -- and a keyword passed explicitly
# is not the same call as the same value left to its default where the record
# says which it was (`environment_declared`).  No fixture below shares its name
# with a fixture anywhere else in `tests/`, so a test adopts one only by asking
# for it.


@pytest.fixture(scope="session")
def p4232_descended():
    """The ``P4_232`` golden, reached by the lattice descent: ``P4_2321' #208.45``.

    ``realized_group(tqc_models.p4232_cano2018())``.  The ``2a``
    orbitals alone are a body-centred point set, ``Im-3m1' #229.141``; the
    Hamiltonian breaks the centring translation, and the record descends to the
    48 operations it obeys (``group.provenance == "descended"``).  Each site's
    ``(d_z2, d_x2-y2)`` pair is one two-dimensional multiplet.
    """
    out = realized_once(tqc_models.p4232_cano2018)
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def p4232_decorated():
    """The ``P4_232`` golden in the paper's crystal-field group: ``#208``, 48 operations.

    ``(model, group, assignment, validated, rep)``, built by hand rather than by
    ``realized_group``: the group is ``_group_of`` of the decorated cell
    ``tqc_models.p4232_symmetry_cell()`` -- the 24-fold orbit of a generic point
    around the two ``2a`` sites -- read off the geometry before the Hamiltonian is
    consulted, and on ``tqc_models.p4232_cano2018()`` the chain is
    ``_assign_orbitals``, ``_validate_group(nk=5)`` and ``_build_rep(n_battery=4)``.
    """
    import numpy as np
    from pythtb import Lattice, TBModel

    import pythtqc.groups as _tqg
    import pythtqc.symmetry as _tqs

    model = tqc_models.p4232_cano2018()
    cell = tqc_models.p4232_symmetry_cell()
    decorated = TBModel(Lattice(np.eye(3), cell["positions"].tolist(),
                                periodic_dirs=[0, 1, 2]), spinful=False)
    group = _tqg._group_of(decorated, typat=cell["typat"].tolist())
    assert group and group.sg_number == 208, str(group)
    assignment = _tqg._assign_orbitals(model, group)
    validated = _tqg._validate_group(model, group, nk=5)
    rep = _tqs._build_rep(model, group, assignment, validated=validated, n_battery=4)
    assert rep, str(rep)
    return model, group, assignment, validated, rep


@pytest.fixture(scope="session")
def fkm_undistorted():
    """Undistorted Fu-Kane-Mele on diamond, ``s`` declared: ``Fd-3m1' #227.129``.

    ``realized_group(tqc_models.fu_kane_mele_z2("uniform"), characters={0: "s"})``,
    all 96 operations of the cell's group obeyed.  The same cache entry as
    ``realized(fkm_regime("uniform"), characters={0: "s"})``, the spelling
    ``test_characters``' EBR-column oracle passes.
    """
    out = realized_once(fkm_regime("uniform"), characters={0: "s"})
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def km_inversion_free_standing():
    """Kane-Mele at ``delta = rashba = 0`` as a free-standing layer: ``P6/mmm1' #191.234``.

    ``realized_group(tqc_models.kane_mele_inversion(),
    environment="free-standing", characters={0: "pz"})``.  The horizontal mirror
    is real here, so all 48 operations of the layer group are obeyed, and the
    declared ``p_z`` is ``A2''`` of the ``-6m2`` site group.
    """
    out = realized_once(tqc_models.kane_mele_inversion,
                        environment="free-standing", characters={0: "pz"})
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def p41_free_standing():
    """The SRSI paper's flat-band model as a free-standing layer: ``P4mm1' #99.164``.

    ``realized_group(tqc_models.p41_flat_band(), environment="free-standing")``:
    a descent from the cell's lattice supergroup ``P4/mmm1' #123.340``, 64
    operations, to the 16 the Hamiltonian obeys.
    """
    out = realized_once(tqc_models.p41_flat_band, environment="free-standing")
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def p41_substrate():
    """The same model declared on a substrate: ``P4mm1' #99.164``, 16 operations.

    ``realized_group(tqc_models.p41_flat_band(), environment="substrate")``,
    the environment declared rather than left to its default.
    """
    out = realized_once(tqc_models.p41_flat_band, environment="substrate")
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def p41_chiral_substrate():
    """The chiral flat-band model declared on a substrate: ``P41' #75.2``.

    ``realized_group(tqc_models.p41_flat_band(chiral=True),
    environment="substrate")``: the next-nearest pattern breaks the four
    vertical mirrors, and the record descends from ``P4mm1' #99.164`` to the 8
    operations it obeys.
    """
    out = realized_once(p41_band(True), environment="substrate")
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def lin3g_undeclared():
    """lin3g with nothing declared, on a substrate: ``P4mm1' #99.164``.

    ``realized_group(tqc_models.lin3g(), typat=[0, 0, 1, 1])``: every orbital
    assigned by the lowest-angular-momentum rule -- the site irreps ``s``,
    ``d_x2-y2``, ``s``, ``s`` carry -- and the free-standing reading
    cross-checked, as by default.
    """
    out = realized_once(tqc_models.lin3g, typat=[0, 0, 1, 1])
    assert out, str(out)
    return out


@pytest.fixture(scope="session")
def lin3g_declared_cross_checked():
    """lin3g with its orbitals declared ``s, d_x2-y2, s, s``: ``P4mm1' #99.164``.

    ``realized_group(tqc_models.lin3g(), typat=[0, 0, 1, 1],
    characters=["s", "dx2-y2", "s", "s"])``, with the free-standing reading
    cross-checked, as by default.  The same declaration with
    ``cross_check=False`` is a different record, which several modules share
    through ``realized``.
    """
    out = realized_once(tqc_models.lin3g, typat=[0, 0, 1, 1],
                        characters=["s", "dx2-y2", "s", "s"])
    assert out, str(out)
    return out


#: The thirteen space groups the SRSI tables omit, from the caption's own rule.
OMITTED = (1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170)


@pytest.fixture(scope="session")
def every_table():
    """``[(SRSITables, ebr_data), ...]`` for every tabulated system, both valuednesses.

    One pass over the 434 systems, shared by every sweep over the SRSI tables.
    Each EBR list is fetched by the table's own grey BNS number and valuedness
    (``srsi._TableKey``), with no measured group; a space group with no table
    refuses ``srsi_table_absent``.
    """
    import pythtqc.groups as _tqg
    import pythtqc.srsi as _tqr

    out = []
    for sg in range(1, 231):
        for spinor in (False, True):
            table = _tqr.srsi_tables(sg, spinor=spinor)
            if not table:
                assert table.gate == "srsi_table_absent", (sg, spinor, table.gate)
                continue
            data = _tqg.ebr_data(_tqr._TableKey(table.number_str, spinor))
            assert data, f"#{table.number_str}: {data}"
            out.append((table, data))
    return out


@pytest.fixture(scope="module")
def lin3g():
    """The primary golden: 2D spinful fragile quartet + s@2c EBR, 8 bands."""
    return tqc_models.lin3g()


@pytest.fixture(scope="module")
def km_odd():
    """Kane-Mele in the Z2-nontrivial (QSH) regime, 4 bands."""
    return tqc_models.kane_mele_z2("odd")


@pytest.fixture(scope="module")
def km_even():
    """Kane-Mele in the Z2-trivial regime, 4 bands."""
    return tqc_models.kane_mele_z2("even")


@pytest.fixture(scope="module")
def p4232():
    """The 3D nonsymmorphic spinless golden (SG 208) at the paper's Fig. 3a parameters."""
    return tqc_models.p4232_cano2018()
