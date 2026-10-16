"""Rewrite the example goldens outside pytest.

``python scripts/update_example_goldens.py`` runs all ten examples and rewrites
``tests/data/example_goldens/<script>.out``; naming one or more scripts rewrites
only those, which is the author's path while rewriting a single example (eight
seconds for ``tqc_sg80_soc_doublet.py`` rather than four minutes for the ten).
Each script prints what changed, as a line count, and ``git diff
tests/data/example_goldens`` is the review surface.

The run environment is not duplicated here: ``tests/test_examples.py`` is loaded
by path and its ``_run``, ``_normalize`` and ``_write_golden`` are the ones
used, so the goldens this writes are by construction the ones the test reads.
"""

import argparse
import importlib.util
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_MODULE = os.path.normpath(os.path.join(_HERE, "..", "tests", "test_examples.py"))


def _load_test_module():
    """``tests/test_examples.py``, loaded by path (``tests/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("tqc_test_examples", _TEST_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    tests = _load_test_module()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", nargs="*", default=[],
                        help=f"one or more of: {', '.join(tests.SCRIPTS)} "
                             f"(default: all of them)")
    args = parser.parse_args(argv)

    scripts = args.script or tests.SCRIPTS
    unknown = [s for s in scripts if s not in tests.SCRIPTS]
    if unknown:
        parser.error(f"not an example: {', '.join(unknown)}")

    for script in scripts:
        path = os.path.join(tests.GOLDENS, script + ".out")
        before = tests._read_golden(path) if os.path.exists(path) else ""
        with tempfile.TemporaryDirectory(prefix="tqc_goldens_") as figdir:
            after = tests._normalize(tests._run(script, figdir), figdir)
        tests._write_golden(path, after)
        old, new = before.splitlines(), after.splitlines()
        added = len([line for line in new if line not in old])
        removed = len([line for line in old if line not in new])
        print(f"{script:<45} {len(new):>4} lines  "
              f"{'+%d/-%d' % (added, removed) if before else 'new golden'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
