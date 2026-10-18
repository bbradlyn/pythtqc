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
listing, the one the register scans too, so a page is under this test the day
it exists.  A failure is a unified diff per statement, at the page's own line
numbers.

Two pages are held to more than their code:

* ``docs/reference/functions.md`` is written from the docstrings by
  ``scripts/build_reference.py``, and must be what that script writes now;
* ``docs/reference/refusals.md``, once it exists, carries a heading for every
  gate a public call can return, by the list ``tests/test_register.py`` holds
  the docstrings to.

And the pages together build as the documentation site, ``mkdocs build
--strict`` by ``mkdocs.yml``: every page in its navigation, every link and
anchor resolved.  ``mkdocs serve`` previews it.

To rewrite the outputs after a deliberate change:

* ``TQC_UPDATE_DOC_OUTPUTS=1 pytest tests/test_docs.py -m gates`` rewrites
  every page's output lines from what its code prints and then **fails**, so
  that a rewrite is never a green run; re-run without the variable to see it
  pass, and read the change as ``git diff docs README.md``.
* ``python scripts/check_page.py --update <page>`` rewrites one page.
* ``python scripts/build_reference.py`` rewrites the function reference.
"""

import difflib
import importlib.util
import os
import re
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
register = _load("tqc_test_register", os.path.join("tests", "test_register.py"))

#: every user page, read off the tree, so a new page cannot slip in unrun
PAGES = check_page.user_pages(ROOT)

FUNCTIONS = "docs/reference/functions.md"
REFUSALS = "docs/reference/refusals.md"


def _reference_drift():
    """How the function reference differs from what the docstrings make it now."""
    want = build_reference.render()
    with open(os.path.join(ROOT, FUNCTIONS), "rb") as handle:
        have = handle.read().decode("utf-8").replace("\r\n", "\n")
    if have == want:
        return []
    diff = list(difflib.unified_diff(have.splitlines(), want.splitlines(),
                                     f"{FUNCTIONS} (in the tree)",
                                     f"{FUNCTIONS} (the docstrings now)", lineterm=""))
    more = f"\n... and {len(diff) - 120} lines more" if len(diff) > 120 else ""
    return [f"{FUNCTIONS} is not what the docstrings make it; "
            f"`python scripts/build_reference.py` writes it\n" + "\n".join(diff[:120]) + more]


def _unheaded_gates():
    """Every gate a public call can return that no heading of the refusal reference names."""
    headings = [text for _, text, _ in check_page.headings(os.path.join(ROOT, REFUSALS))]
    missing = [gate for gate in register.returned_gates()
               if not any(re.search(rf"\b{re.escape(gate)}\b", text) for text in headings)]
    if not missing:
        return []
    return [f"{REFUSALS} has no heading for {len(missing)} of the gates a public call can "
            f"return: " + ", ".join(missing)]


@pytest.mark.parametrize("page", PAGES)
def test_page_prints_what_it_shows(page):
    """One page's code, run as written, against the output lines under each statement."""
    update = os.environ.get("TQC_UPDATE_DOC_OUTPUTS") == "1"
    run = check_page.check(os.path.join(ROOT, page), update=update)
    if update:
        pytest.fail(f"outputs rewritten for {page}: {len(run.rewritten)} statement(s); "
                    f"re-run without the variable" + "".join("\n\n" + p for p in run.problems))
    problems = list(run.problems)
    if page == FUNCTIONS:
        problems += _reference_drift()
    if page == REFUSALS:
        problems += _unheaded_gates()
    assert not problems, "\n\n".join(problems)


def test_the_pages_build_as_a_site_with_every_link_resolved(tmp_path):
    """The documentation site, built strictly from the pages as they are.

    ``mkdocs.yml`` makes a page left out of the navigation, a dead link and a dead
    anchor warnings, and a strict build fails on a warning, so this is the test
    that the site and the tree agree.
    """
    if importlib.util.find_spec("mkdocs") is None:
        pytest.fail("the site needs MkDocs and its Material theme: pip install '.[test]'")
    env = dict(os.environ, PYTHONUTF8="1", NO_MKDOCS_2_WARNING="1")
    run = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir",
         str(tmp_path / "site")],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
