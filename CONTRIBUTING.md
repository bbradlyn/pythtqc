# Contributing to pythtqc

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/bbradlyn/pythtqc/issues). A report is most useful
with the model that shows it, the call, and what it returned or raised.

## Setting up

```bash
pip install -e ".[dev]"
```

## The tests

The suite has four tiers, by mark:

```bash
pytest                        # the fast tier
pytest -m gates               # every page and example against what it prints, the stored baseline, and the site
pytest -m slow                # the slower tests
pytest -m data                # the sweeps over the shipped tables
pytest -m "slow or not slow"  # everything, about an hour
```

Run the test modules a change touches, whole, with `-m "slow or not slow"`, and
the gates before opening a pull request. A new test is fast unless it is marked,
so a costly one is marked `slow`.

## The documentation

Every page under `docs/` runs as written: the lines a page shows as printed are
what its code prints, and `tests/test_docs.py` holds every page to that. After a
deliberate change, `python scripts/check_page.py --update <page>` rewrites a
page's printed lines; they are never typed by hand. The function reference,
`docs/reference/functions.md`, is written from the docstrings by
`python scripts/build_reference.py`. `mkdocs serve` previews the site.

The text a reader meets, the docstrings and comments, the pages and the
examples, is written for a physicist: physics words, the present tense, what a
call does and what it returns. `tests/test_register.py` checks it.

## Conventions the code keeps

- A `Refusal` is a value a call returns, saying why it cannot answer and what it
  measured; `InternalError` is the package disagreeing with itself, a bug to
  report.
- What a user declares, the atom types (`typat=`) and the orbitals' content
  (`characters=`), defines the crystal, and the Hamiltonian is measured against
  it.
- Matrices and labels are returned in the tables' convention: spin along `+z` in
  a right-handed frame, the labels those of `irrep`'s tables.
