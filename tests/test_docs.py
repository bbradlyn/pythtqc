"""Every user page, run as written: each line a page shows as printed is what its code prints.

The user pages -- the root ``README.md``, the pages of ``docs/``, its map
``docs/README.md`` among them, and every page
under ``docs/start``, ``docs/guide``, ``docs/background`` and
``docs/reference`` -- carry their code in ``python`` blocks, with what each
statement prints under it as comment lines.  Each page is run through
``scripts/check_page.py``: in a fresh process, from the repository root, its
blocks in order and its statements one at a time, each statement's printout
against the lines under it, a printed number that is rounding matching any
other rounding.  The pages are read off the tree by that script's
listing, so a page is under this test the day it exists.  A failure is a
unified diff per statement, at the page's own line numbers.

Two pages are held to more than their code: ``docs/reference/functions.md``
and ``docs/reference/records.md`` are written from the docstrings by
``scripts/build_reference.py``, and must be what that script writes now.

The docstrings are held to their examples, with the functions' page: each ``>>>``
example in a docstring of ``src/pythtqc`` runs as a doctest and must print what it
shows.  An example runs with its module's own names, the package's public names and
numpy as ``np``.

And the pages together build as the documentation site, ``mkdocs build
--strict`` by ``mkdocs.yml``: every page in its navigation, every link and
anchor resolved.  ``mkdocs serve`` previews it.

To rewrite the outputs after a deliberate change:

* ``TQC_UPDATE_DOC_OUTPUTS=1 pytest tests/test_docs.py -m gates`` rewrites
  every page's output lines from what its code prints and then **fails**, so
  that a rewrite is never a green run; re-run without the variable to see it
  pass, and read the change as ``git diff docs README.md``.
* ``python scripts/check_page.py --update <page>`` rewrites one page.
* ``python scripts/build_reference.py`` rewrites the two reference pages.
"""

import difflib
import importlib.util
import os
import subprocess
import sys

import pytest

#: Every page with code runs as a subprocess, and a page that measures a large
#: model costs tens of seconds: a gate, run whole with ``-m gates`` before a
#: commit, and a bare ``pytest`` leaves it out.
pytestmark = pytest.mark.gates

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, ".."))


def _load(name, relative):
    """A module of this tree, loaded by path: ``scripts/`` and ``tests/`` are not packages."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_page = _load("tqc_check_page", os.path.join("scripts", "check_page.py"))
build_reference = _load("tqc_build_reference", os.path.join("scripts", "build_reference.py"))

#: every user page, read off the tree, so a new page cannot slip in unrun
PAGES = check_page.user_pages(ROOT)

#: the functions' reference, with which the docstrings' examples are run
FUNCTIONS = "docs/reference/functions.md"

#: the pages written from the docstrings, each with the function that writes it
WRITTEN = {FUNCTIONS: build_reference.render,
           "docs/reference/records.md": build_reference.render_records}


def _reference_drift(page):
    """How a reference page differs from what the docstrings make it now."""
    want = WRITTEN[page]()
    with open(os.path.join(ROOT, page), "rb") as handle:
        have = handle.read().decode("utf-8").replace("\r\n", "\n")
    if have == want:
        return []
    diff = list(difflib.unified_diff(have.splitlines(), want.splitlines(),
                                     f"{page} (in the tree)",
                                     f"{page} (the docstrings now)", lineterm=""))
    more = f"\n... and {len(diff) - 120} lines more" if len(diff) > 120 else ""
    return [f"{page} is not what the docstrings make it; "
            f"`python scripts/build_reference.py` writes it\n" + "\n".join(diff[:120]) + more]


def _docstring_examples():
    """How the docstrings' examples fail, each run as a doctest; empty when all pass.

    Every docstring of ``src/pythtqc`` with ``>>>`` lines, public or private, runs in
    its module's namespace with numpy as ``np`` and the package's public names added.
    The few examples that run on names they do not build are listed in ``supplied``,
    from the object's dotted name to the function that builds those names.
    """
    import doctest
    import pkgutil

    import numpy

    import pythtqc

    def lin3g_fourfold_rotation():
        """lin3g's fourfold rotation: its matrix, the operation, the group, the anchor."""
        from pythtqc._matrices import RepAnchor
        r = pythtqc.realized_group(pythtqc.models.lin3g())
        group, matrices = r.group, r.rep.matrices
        c4 = pythtqc.operation_index(group, "{4+_001|0,0,0}")
        tr = pythtqc.operation_index(group, "{1|0,0,0}'")
        return {"D": matrices[c4], "op": group._symmetries[c4 - 1], "group": group,
                "anchor": RepAnchor(antiunitary=matrices[tr], rows=(0, 1), cols=(0, 1))}

    def kane_mele_one_center_moved():
        """Kane-Mele with the first center moved by 1e-8 along the first lattice vector."""
        from pythtb import Lattice, TBModel
        from pythtb.models import kane_mele
        km = kane_mele(0.0, -1.0, 0.3, 0.25)
        moved = TBModel(Lattice(km.lat_vecs, km.orb_vecs + [[1e-8, 0.0], [0.0, 0.0]],
                                periodic_dirs=km.periodic_dirs), spinful=True)
        moved.set_onsite(km.onsite, mode="set")
        for h in km.hoppings:
            moved.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                          h.get("lattice_vector", [0, 0]), mode="add")
        return {"moved": moved}

    supplied = {"pythtqc._matrices.normalize_rep": lin3g_fourfold_rotation,
                "pythtqc.symmetrization._symmetrize_positions": kane_mele_one_center_moved}

    public = {name: getattr(pythtqc, name) for name in pythtqc.__all__}
    finder = doctest.DocTestFinder(exclude_empty=True)
    problems, ran = [], 0
    names = ["pythtqc"] + [f"pythtqc.{m.name}" for m in pkgutil.iter_modules(pythtqc.__path__)]
    for module in map(importlib.import_module, names):
        for test in finder.find(module, globs={**public, "np": numpy, **vars(module)}):
            if test.name in supplied:
                test.globs.update(supplied[test.name]())
            out = []
            result = doctest.DocTestRunner(verbose=False).run(test, out=out.append)
            ran += result.attempted
            if result.failed:
                problems.append(f"the example in the docstring of {test.name} "
                                f"prints otherwise\n" + "".join(out))
    if not ran:
        problems.append("no docstring example was found in src/pythtqc")
    return problems


@pytest.mark.parametrize("page", PAGES)
def test_page_prints_what_it_shows(page):
    """One page's code, run as written, against the output lines under each statement."""
    update = os.environ.get("TQC_UPDATE_DOC_OUTPUTS") == "1"
    run = check_page.check(os.path.join(ROOT, page), update=update)
    if update:
        pytest.fail(f"outputs rewritten for {page}: {len(run.rewritten)} statement(s); "
                    f"re-run without the variable" + "".join("\n\n" + p for p in run.problems))
    problems = list(run.problems)
    if page in WRITTEN:
        problems += _reference_drift(page)
    if page == FUNCTIONS:
        problems += _docstring_examples()
    assert not problems, "\n\n".join(problems)


def test_the_pages_build_as_a_site_with_every_link_resolved(tmp_path):
    """The documentation site, built strictly from the pages as they are.

    ``mkdocs.yml`` makes a page left out of the navigation, a dead link and a dead
    anchor warnings, and a strict build fails on a warning, so this is the test
    that the site and the tree agree.
    """
    if any(importlib.util.find_spec(name) is None for name in ("mkdocs", "material")):
        pytest.fail("the site needs MkDocs and its Material theme: pip install '.[test]'")
    env = dict(os.environ, PYTHONUTF8="1", NO_MKDOCS_2_WARNING="1")
    run = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir",
         str(tmp_path / "site")],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
