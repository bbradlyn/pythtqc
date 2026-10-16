"""The ten worked examples in ``examples/``, each against its golden stdout.

This module tests one thing: that the examples print what we expect them to
print.  Each script is run once, in a subprocess, in quick mode, with its figure
directory pointed at a ``tmp_path``, and its stdout is compared character for
character with ``tests/data/example_goldens/<script>.out``.  The one
substitution is the figure directory on the script's last line: that is this
run's temporary directory, which no golden can hold.  A failure is a unified
diff between the golden and what the script printed.

Every number the scripts print is measured rather than typed, so the goldens are
the measurement record: a convention change, a table update or a refactor that
moves one of them shows up as a diff here.  The captures are quick mode, which
changes only the k-point density of the figures -- no script prints its mesh --
so a full-mode run matches the same files.

To regenerate after a deliberate change:

* ``TQC_UPDATE_EXAMPLE_GOLDENS=1 pytest tests/test_examples.py -m gates``
  rewrites every golden and then **fails**, so that a regeneration is never a
  green run; re-run without the variable to see it pass, and read the change as
  ``git diff tests/data/example_goldens``.
* ``python scripts/update_example_goldens.py [script ...]`` rewrites one golden
  (or all ten) outside pytest.
"""

import difflib
import os
import subprocess
import sys

import pytest

#: Every test here runs an example as a subprocess; the ten together cost a few
#: minutes.  A gate: an example whose printed numbers have moved is a result
#: changed, so the module runs whole, with ``-m gates``, before a commit, and a
#: bare ``pytest`` leaves it out.
pytestmark = pytest.mark.gates

_HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.normpath(os.path.join(_HERE, "..", "examples"))
GOLDENS = os.path.join(_HERE, "data", "example_goldens")

#: the checkout's own package, which is what the rest of the suite imports
#: (``conftest.py`` puts it on ``sys.path``; a subprocess needs ``PYTHONPATH``).
SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))

#: generous -- the scripts measure 8-60 s each on the reference machine.
TIMEOUT = 600

#: every example, read off the directory, so an eleventh script cannot slip in
#: unread: it parametrizes the test and fails on its missing golden.
SCRIPTS = sorted(f for f in os.listdir(EXAMPLES)
                 if f.startswith("tqc_") and f.endswith(".py"))


def _env(figdir):
    """The example's environment: quick mode, the figure directory, and the checkout.

    ``PYTHONIOENCODING`` is set because the corep labels (``↑``, ``−``) must
    survive a cp1252 default on Windows, and ``MPLBACKEND`` because a headless
    run must not open a window.
    """
    env = dict(os.environ)
    env.update(TQC_EXAMPLE_QUICK="1", TQC_EXAMPLE_FIGDIR=str(figdir),
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    env["PYTHONPATH"] = os.pathsep.join(
        [SRC] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _run(script, figdir):
    """Run one example in quick mode; return its stdout.

    A subprocess rather than ``runpy``: the scripts set the matplotlib backend,
    reconfigure ``sys.stdout`` to UTF-8 and rely on ``sys.path[0]`` being
    ``examples/`` for their ``_path`` shim, and a child gives all three for free
    while leaving the pytest process's own state alone.  ``text=True``
    universal-newline translates on decode, so what comes back is LF-only
    whatever the platform wrote.
    """
    path = os.path.join(EXAMPLES, script)
    assert os.path.exists(path), path
    proc = subprocess.run([sys.executable, path], env=_env(figdir),
                          timeout=TIMEOUT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, (
        f"{script} exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}")
    return proc.stdout


def _normalize(out, figdir):
    """The one machine-dependent string, replaced by a token.

    Every script ends ``print("\\nDone — figure written to", OUTDIR)`` with
    ``OUTDIR`` the directory this run handed it.  Replacing the directory rather
    than dropping the line keeps the rest of that line under exact comparison,
    and it degrades gracefully if a script ever prints the directory twice.
    """
    return out.replace(str(figdir), "<FIGDIR>")


def _read_golden(path):
    """One golden, LF-only whatever the checkout did to it.

    The goldens are stored LF-only and pinned there by ``.gitattributes``; the
    normalization is belt and braces for a checkout that arrived some other way.
    """
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8").replace("\r\n", "\n")


def _write_golden(path, text):
    """Write one golden, LF-only, on any platform."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_prints_what_we_expect(script, tmp_path):
    """One script's stdout, character for character, against its golden."""
    measured = _normalize(_run(script, tmp_path), tmp_path)
    path = os.path.join(GOLDENS, script + ".out")
    if os.environ.get("TQC_UPDATE_EXAMPLE_GOLDENS") == "1":
        _write_golden(path, measured)
        pytest.fail(f"golden rewritten for {script}; re-run without the variable")
    assert os.path.exists(path), (
        f"no golden for {script}: write one with "
        f"`python scripts/update_example_goldens.py {script}`")
    want = _read_golden(path)
    assert measured == want, "\n".join(
        difflib.unified_diff(want.splitlines(), measured.splitlines(),
                             f"{script}.out (golden)", f"{script} (measured)",
                             lineterm=""))
