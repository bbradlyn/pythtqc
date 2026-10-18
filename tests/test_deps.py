"""Hard dependency check for the ``tqc`` pipeline: real tables, not silent stubs.

``irrep`` imports every ``irreptables`` entry point inside a ``try/except
ImportError`` and, on failure, rebinds the name to
``irrep.utility.DummyMethodTables``.  That fallback is not a stub that raises at
import: it is an object that only complains when it is *called*, and several
``irrep`` code paths call it lazily or not at all.  A ``tqc`` run against a
half-installed ``irreptables`` would therefore not crash -- it would quietly
produce results with no character tables behind them.  These tests are the
tripwire: they assert the real tables are bound, that they load *complete*
character tables for two spinless and two spinful groups -- counts and k-point
sets, not a lower bound a single surviving row would satisfy -- and that the
table versions are the ones the rest of the suite pinned its labels against.

Which of the two of them is bound is decided by import order, and the last two
tests run a child interpreter each to measure it: ``irreptables.irreps`` imports
``irrep.utility``, so importing ``irreptables`` first makes ``irrep``'s own
``from irreptables.irreps import IrrepTable`` run against a half-built module,
and the fallback is bound for the rest of that process.  ``pythtqc.groups``
imports ``irrep.spacegroup_irreps`` at module scope for exactly that reason.

``ortools`` is likewise a hard requirement (the EBR / symmetry-indicator work
needs its integer programming), and it has no fallback at all -- so a plain
import is the whole check.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))


def _child(body):
    """Run ``body`` in a fresh interpreter against this checkout; give back stdout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_SRC] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


#: The oldest table versions the suite holds: every pinned label, corep name,
#: k-point name and conjugated point in it was measured against these, and holds
#: on irrep 3.3.0 as well.  A later release is accepted, as ``pyproject.toml``
#: accepts it.  Those pins are properties of the tables and of nothing in this
#: package, so a release that moves a label fails the tests that pin it and the
#: stored baseline: read such failures against the release first, and
#: re-measure the affected pins in the same commit.
TABLE_VERSIONS = {"irrep_version": "3.2.0", "irreptables_version": "3.1.0"}


def test_irrep_did_not_fall_back_to_dummy_tables():
    """None of ``irrep``'s three ``irreptables`` entry points is the dummy stub."""
    from irrep.utility import DummyMethodTables
    from irrep.spacegroup_irreps import IrrepTable
    from irrep.bandstructure import load_ebr_data, load_si_table

    for name, obj in (("IrrepTable", IrrepTable),
                      ("load_ebr_data", load_ebr_data),
                      ("load_si_table", load_si_table)):
        assert not isinstance(obj, DummyMethodTables), (
            f"irrep fell back to DummyMethodTables for {name}: irreptables is not "
            f"importable, and every downstream irrep result would be meaningless")


def test_the_table_versions_are_at_least_the_ones_the_suite_was_measured_against():
    """``tables_status`` reports an ``irrep`` / ``irreptables`` no older than the pins assume."""
    from packaging.version import Version
    from pythtqc import groups as tqg

    status = tqg.tables_status()
    assert status["tables"], status["reason"]
    got = {k: status[k] for k in TABLE_VERSIONS}
    older = {k: v for k, v in got.items() if Version(v) < Version(TABLE_VERSIONS[k])}
    assert not older, (
        f"character tables are {got}, older than {TABLE_VERSIONS}: every pinned irrep "
        f"label, corep name and conjugated point in this suite was measured against "
        f"the latter, and pyproject.toml asks for them")


#: ``(sg_number, spinor, n_irreps, n_symmetries, kpoints)`` of the four tables
#: whose completeness is pinned below.
REAL_TABLES = (
    # P4_332 -- nonsymmorphic, spinless.  NOT a group any fixture needs: the 3D
    # fixture's group is 208, not 212 (see fixture_models.py's module
    # docstring), and 212 stays in this list only as a second nonsymmorphic
    # spinless table.
    ("212", False, 15, 24, {"GM", "M", "R", "X"}),
    # P4_232 -- the group the 3D fixture actually realizes
    ("208", False, 20, 24, {"GM", "M", "R", "X"}),
    # P4mm -- spinful (double-valued) coreps
    ("99", True, 10, 8, {"A", "GM", "M", "R", "X", "Z"}),
    # P4/mmm -- spinful; the lin3g slab cell's group
    ("123", True, 20, 16, {"A", "GM", "M", "R", "X", "Z"}),
)


def test_irreptables_load_real_character_tables():
    """``irreptables`` returns its whole table for each group, not a first row.

    The counts and the k-point set are pinned exactly, and that is the point of
    the test rather than a tightening of it.  A ``min_irreps`` of 1 stood here
    until wave P6, in all four rows -- 10x to 20x below the true counts (15, 20,
    10, 20) and one-sided on top of that -- so a table that came back holding
    one irrep out of fifteen satisfied every assertion in the file whose entire
    reason for existing is that a half-installed ``irreptables`` fails quietly.
    A truncated table is exactly what the structural statements catch: the full
    k-point set (the two cubic groups carry ``{GM,M,R,X}``, the two tetragonal
    ones also ``A`` and ``Z``) and the operation count, which was not checked at
    all.

    Provenance of the numbers: measured from ``irreptables`` 3.1.0 under
    ``irrep`` 3.2.0, the versions every pinned label, corep name and conjugated
    point in this suite is a property of.  They are regression pins on the
    installed tables -- a genuine upstream table revision is expected to move
    them, and to be read here as the notification that it did.
    """
    from irreptables.irreps import IrrepTable

    for sg_number, spinor, n_irreps, n_symmetries, kpoints in REAL_TABLES:
        table = IrrepTable(sg_number, spinor, magnetic=False)
        assert len(table.irreps) == n_irreps, (
            f"IrrepTable({sg_number!r}, {spinor}) loaded {len(table.irreps)} irreps, "
            f"expected {n_irreps} -- a truncated table is the failure this file exists for")
        assert len(table.symmetries) == n_symmetries, sg_number
        # every entry really is a labelled irrep at a named k-point
        for irrep in table.irreps:
            assert irrep.name and irrep.kpname, sg_number
        assert {ir.kpname.upper() for ir in table.irreps} == kpoints, sg_number


@pytest.mark.slow
def test_irreptables_imported_first_is_reported_as_the_import_order_it_is():
    """A child that imports ``irreptables.irreps`` first reads the honest cause.

    ``irreptables`` is installed in this environment, so the fallback bound in
    that child is not the missing package: it is the order the two were imported
    in, and ``tables_status`` says which of the two it is.
    """
    out = _child("""
        import json
        import irreptables.irreps          # noqa: F401 - before irrep, on purpose
        from pythtqc import groups as tqg
        print("STATUS", json.dumps(tqg.tables_status()))
    """)
    status = json.loads(out.split("STATUS", 1)[1])
    assert status["tables"] is False and "DummyMethodTables" in status["reason"]
    assert status["irreptables_version"] == TABLE_VERSIONS["irreptables_version"]
    assert "imported before" in status["reason"], status["reason"]
    assert "not installed" not in status["reason"], status["reason"]


@pytest.mark.slow
def test_importing_pythtqc_first_binds_the_real_tables_whatever_follows():
    """``import pythtqc`` leaves ``irreptables.irreps`` safe to import after it.

    ``pythtqc.groups`` imports ``irrep.spacegroup_irreps`` at module scope, so
    irrep binds the real ``IrrepTable`` before anything else can spring the
    trap; the group of a one-site square lattice's cell is then identified with
    its tables in place: the whole layer group of the square lattice,
    ``P4/mmm1' #123.340``.
    """
    out = _child("""
        import json
        import pythtqc                     # noqa: F401 - binds the real IrrepTable
        import irreptables.irreps          # noqa: F401 - nothing left to spring
        from pythtb import Lattice, TBModel
        from pythtqc import groups as tqg

        lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], periodic_dirs=[0, 1])
        m = TBModel(lattice=lat, spinful=False)
        m.set_onsite([0.0])
        m.set_hop(-1.0, 0, 0, [1, 0])
        m.set_hop(-1.0, 0, 0, [0, 1])
        g = tqg._group_of(m)
        print("GROUP", json.dumps({"served": bool(g),
                                   "tables_ready": bool(getattr(g, "_tables_ready", False)),
                                   "number_str": str(getattr(g, "number_str", ""))}))
    """)
    group = json.loads(out.split("GROUP", 1)[1])
    assert group["served"] is True and group["tables_ready"] is True
    assert group["number_str"] == "123.340"


def test_ortools_is_installed():
    """``ortools`` (integer programming for the EBR / indicator solvers) imports."""
    import ortools                    # noqa: F401
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    x = model.NewIntVar(0, 3, "x")
    model.Add(x == 2)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(x) == 2
