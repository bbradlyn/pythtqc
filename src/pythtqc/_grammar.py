"""The grammar of ``characters=`` and ``multiplets=``, read where a call starts
and before anything is measured: a malformed declaration raises ``ValueError``
naming the spelling it expected.
"""

from __future__ import annotations

import numbers

import numpy as np

from . import _sitechars
from ._matrices import SiteMultiplets


def _spec_key(spec):
    """A comparable identity for one declaration value.

    Labels compare case-insensitively; a **generator mapping**, of characters
    or of matrices, compares by its content rather than by object identity,
    because the two members of a declared spinor pair can be handed the same
    mapping written twice.  Anything else falls
    back to identity, which is what the ordered-list grouping does.
    """
    if isinstance(spec, str):
        return ("label", spec.lower())
    if isinstance(spec, dict):
        # sorted on the key alone: a complex value is not orderable, and two
        # entries can stringify to the same key (operation #1 and its name "1")
        return ("chi", tuple((str(k), _value_key(v)) for k, v in
                             sorted(spec.items(), key=lambda kv: str(kv[0]))))
    return ("id", id(spec))


def _value_key(value):
    """One value of a generator mapping as a comparable: a character, or a matrix's entries."""
    if _character_value(value):
        return complex(value)
    matrix = _matrix_value(value)
    if matrix is None:
        return ("id", id(value))
    return ("matrix", matrix.shape, tuple(complex(x) for x in matrix.ravel()))


# ---- the grammar of a declaration ---------------------------------------------
# `characters=` and `multiplets=` are read as grammar once, where a call starts
# and before anything is measured, and a malformed one raises `ValueError`
# stating the argument's grammar.  Which words name something at a site is
# read later, in the site group the measurement reaches.

#: ``characters=`` in one sentence, which every error about its grammar opens with.
_CHARACTERS_GRAMMAR = (
    "`characters=` is a list with one entry per orbital, each a word or None, or a "
    "mapping from an orbital, or from several orbitals of one site written together, "
    "to a word or to a mapping from operations to characters, or to matrices on "
    "those orbitals")

#: ``multiplets=`` in one sentence, which every error about its grammar opens with.
_MULTIPLETS_GRAMMAR = (
    "`multiplets=` is one list per site of the multiplets it holds, each written as "
    "its orbitals, such as [[(0, 1)], [(2,)]], or the SiteMultiplets a representation "
    "carries")


def _shown(value, width=60) -> str:
    """``value`` as a grammar error quotes it: its type, and its repr cut to ``width``."""
    if value is None:
        return "None"
    text = repr(value)
    if len(text) > width:
        text = text[:width - 3] + "..."
    return f"the {type(value).__name__} {text}"


def _orbital_index(value) -> bool:
    """Whether ``value`` can name an orbital: an integer, and not a truth value."""
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)))


def _character_value(value) -> bool:
    """Whether ``value`` can be a character: a finite number, and not a truth value."""
    return (isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))
            and bool(np.isfinite(complex(value))))


def _matrix_value(value):
    """``value`` as a complex square matrix of finite numbers, or ``None`` where it is none.

    A nested list or an array of two dimensions whose entries are all numbers;
    a string, a truth value or a ragged list is no matrix.
    """
    if isinstance(value, (str, bytes, dict)) or _character_value(value):
        return None
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size
            or matrix.dtype.kind not in "iufc"):
        return None
    matrix = matrix.astype(complex)
    return matrix if bool(np.all(np.isfinite(matrix))) else None


def _declaration_grammar(norb, *, characters=None, multiplets=None, spin=None,
                         site_of=None):
    r"""Raise ``ValueError`` where ``characters=`` or ``multiplets=`` is malformed.

    The grammar of the two declarations, read where :func:`.._build._build_rep`,
    :func:`..groups.realized_group` and :func:`..symmetrization.symmetrize`
    start, before anything is measured.  ``characters=`` is a list with one
    entry per orbital, each a word, a mapping from operations to characters or
    ``None``, or a mapping from an orbital, or from several orbitals of one
    site together, to a word or to such a mapping: each orbital is named once,
    a mapping of characters names at least one operation, by its index or its
    name, and gives each a finite number, a mapping of matrices gives each a
    square matrix of finite numbers, as wide as the Wannier functions its
    orbitals are, and gives every operation one or the other, and the two
    orbitals of a declared spin pair, the up and down states of one Wannier
    function, are given one word.  ``multiplets=`` is one list per site of the
    multiplets it holds, each named by its orbitals, or a
    :class:`SiteMultiplets`: the multiplets hold every orbital once, each the
    orbitals of one site, and each holds both states of a declared spin pair.
    Whether a word names something at its site depends on the site group the
    measurement reaches, and is read there.

    ``norb`` is the model's orbital count, ``spin`` the declared
    :class:`..spin.SpinBasis` or ``None``, and ``site_of`` each orbital's site,
    or ``None`` where the sites are not read yet.  An entry whose value is a
    :class:`.._sitechars.DeclaredContent` is a content already read, carried
    from one group to the next, and passes.

    Returns the multiplets ``multiplets=`` states, every site's in one
    sequence, or ``None`` where it states none.
    """
    pairs = () if spin is None else tuple((int(u), int(d)) for u, d in spin.pairs)
    if characters is not None:
        _characters_grammar(characters, int(norb), pairs, site_of)
    if multiplets is None:
        return None
    return _multiplets_grammar(multiplets, int(norb), pairs, site_of)


def _names_joint_keys(characters) -> bool:
    """Whether ``characters=`` has a key of several orbitals, whose site the grammar reads."""
    return isinstance(characters, dict) and any(
        isinstance(key, (tuple, list)) and len(key) > 1 for key in characters)


def _characters_grammar(characters, norb, pairs, site_of=None) -> None:
    """The grammar of ``characters=``, the first half of :func:`_declaration_grammar`.

    ``site_of`` is each orbital's site, where the sites are read: a key of
    several orbitals names what one site holds together.
    """

    def malformed(what):
        raise ValueError(f"{_CHARACTERS_GRAMMAR}; {what}.")

    named = []                         # (orbitals, value, where it was written)
    if isinstance(characters, dict):
        for key, value in characters.items():
            if isinstance(value, _sitechars.DeclaredContent):
                continue
            if _orbital_index(key):
                orbs = (int(key),)
            elif isinstance(key, (tuple, list)) and not key:
                malformed("the key () names no orbital")
            elif isinstance(key, (tuple, list)) and all(_orbital_index(o) for o in key):
                orbs = tuple(int(o) for o in key)
            else:
                malformed(f"it is given the key {key!r}, a {type(key).__name__}")
            named.append((orbs, value, f"the key {key!r}"))
    elif isinstance(characters, (list, tuple)):
        if len(characters) != norb:
            malformed(f"this model has {norb} orbitals, and the list has "
                      f"{len(characters)} entr{'y' if len(characters) == 1 else 'ies'}")
        named = [((i,), value, f"entry {i}") for i, value in enumerate(characters)
                 if value is not None
                 and not isinstance(value, _sitechars.DeclaredContent)]
    else:
        malformed(f"it is given {_shown(characters)}")
    holder = {}
    for orbs, value, where in named:
        for o in orbs:
            if not 0 <= o < norb:
                malformed(f"{where} names orbital {o}, and this model's orbitals are 0 "
                          f"to {norb - 1}")
            if o in holder:
                malformed(f"it names each orbital once, and orbital {o} is named twice, in "
                          + (where if holder[o][1] == where
                             else f"{holder[o][1]} and in {where}"))
            holder[o] = (value, where)
        if site_of is not None and len(orbs) > 1:
            sites = sorted({int(site_of[o]) for o in orbs})
            if len(sites) > 1:
                malformed(f"{where} names orbitals of sites {sites}")
        if isinstance(value, str):
            continue
        if isinstance(value, dict):
            if not value:
                malformed(f"{where} is given an empty mapping, and a mapping from "
                          "operations to characters names at least one operation")
            # A MATRIX is the operation's matrix on the orbitals named, one row
            # and one column per Wannier function: a declared spin pair is one
            pair_of = {int(o): c for c, pair in enumerate(pairs) for o in pair}
            width = len({pair_of.get(int(o), ("orbital", int(o))) for o in orbs})
            kinds = set()
            for op, chi in value.items():
                if not (_orbital_index(op) or isinstance(op, str)):
                    malformed(f"{where} names an operation by {_shown(op)}, and an "
                              "operation is named by its index or its name, such as 3 "
                              "or 'C4(z)'")
                if _character_value(chi):
                    kinds.add("character")
                    continue
                matrix = _matrix_value(chi)
                if matrix is None:
                    malformed(f"{where} gives the operation {op!r} {_shown(chi)}, and a "
                              "character is a finite number and a matrix a square array "
                              "of them")
                if matrix.shape[0] != width:
                    noun = "Wannier function" if pairs else "orbital"
                    malformed(f"{where} gives the operation {op!r} a {matrix.shape[0]}x"
                              f"{matrix.shape[0]} matrix, and a matrix on {width} "
                              f"{noun}{'' if width == 1 else 's'} is {width}x{width}, "
                              f"one row and one column per {noun} in the order written")
                kinds.add("matrix")
            if len(kinds) > 1:
                malformed(f"{where} gives some operations characters and others matrices, "
                          "and a mapping gives every operation a character or every "
                          "operation a matrix")
            continue
        if value is None:
            malformed(f"{where} is given None, and a mapping leaves out an orbital it "
                      "declares nothing about")
        malformed(f"{where} is given {_shown(value)}")
    # the two states of one Wannier function hold one content
    for c, (up, dn) in enumerate(pairs):
        if (up in holder and dn in holder
                and _spec_key(holder[up][0]) != _spec_key(holder[dn][0])):
            raise ValueError(
                f"{_CHARACTERS_GRAMMAR}, and gives the two orbitals of a declared spin pair "
                f"one word: they are the up and down states of one Wannier function, "
                f"which has one site character. Orbitals {up} and {dn}, spin pair {c}, "
                f"are given {holder[up][0]!r} and {holder[dn][0]!r}; give both the same "
                "word, or declare only one of them.")


def _multiplets_grammar(multiplets, norb, pairs, site_of) -> tuple:
    """The grammar of ``multiplets=``, the second half of :func:`_declaration_grammar`.

    Returns the multiplets it states, every site's in one sequence.
    """

    def malformed(what):
        raise ValueError(f"{_MULTIPLETS_GRAMMAR}; {what}.")

    if isinstance(multiplets, SiteMultiplets):
        stated = multiplets.by_site
    elif isinstance(multiplets, (list, tuple)):
        stated = multiplets
    else:
        malformed(f"it is given {_shown(multiplets)}")
    flat = []
    for i, site in enumerate(stated):
        if not isinstance(site, (list, tuple)):
            malformed(f"entry {i} is {_shown(site)}, where the list of one site's "
                      "multiplets goes")
        for orbs in site:
            if not isinstance(orbs, (list, tuple)):
                malformed(f"entry {i}, {site!r}, holds {_shown(orbs)} where a multiplet "
                          "goes" + _per_site_spelling(stated, norb, site_of))
            if not orbs:
                malformed(f"it holds no empty multiplet, and entry {i} holds one")
            for o in orbs:
                if not _orbital_index(o):
                    malformed(f"the multiplet {tuple(orbs)!r} names an orbital by "
                              f"{_shown(o)}, and an orbital is named by its index")
                if not 0 <= int(o) < norb:
                    malformed(f"the multiplet {tuple(orbs)!r} names orbital {int(o)}, out "
                              f"of range for a model with {norb} orbitals")
            flat.append(tuple(int(o) for o in orbs))
    holder = {}
    for j, orbs in enumerate(flat):
        for o in orbs:
            if o in holder:
                malformed(f"it partitions every orbital of the model into multiplets, and "
                          f"orbital {o} is named twice, in {orbs!r}"
                          + ("" if holder[o] == j else f" and {flat[holder[o]]!r}"))
            holder[o] = j
    missing = tuple(o for o in range(norb) if o not in holder)
    if missing:
        malformed("it partitions every orbital of the model into multiplets, and no "
                  f"multiplet holds orbital{'s' if len(missing) > 1 else ''} "
                  + (str(missing) if len(missing) > 1 else str(missing[0])))
    for c, (up, dn) in enumerate(pairs):
        if holder[up] != holder[dn]:
            malformed(f"each multiplet holds both states of a declared spin pair, the up "
                      f"and down states of one Wannier function, and spin pair {c}, "
                      f"orbitals ({up}, {dn}), is split between {flat[holder[up]]!r} and "
                      f"{flat[holder[dn]]!r}")
    if site_of is not None:
        for orbs in flat:
            sites = sorted({int(site_of[o]) for o in orbs})
            if len(sites) > 1:
                malformed(f"each multiplet holds the orbitals of one site, and the "
                          f"multiplet {orbs!r} spreads over sites {sites}")
    return tuple(flat)


def _per_site_spelling(stated, norb, site_of) -> str:
    """The one-list-per-site spelling of multiplets written as one flat list, or ``""``.

    ``[(0,), (1,)]`` lists multiplets where ``multiplets=`` lists sites; where
    every entry is a multiplet of orbitals of this model, the same multiplets
    are offered one list per site, by ``site_of`` where the sites are read.
    """
    if not stated or not all(
            isinstance(m, (list, tuple)) and m
            and all(_orbital_index(o) and 0 <= int(o) < norb for o in m)
            for m in stated):
        return ""
    by_site = {}
    for m in stated:
        site = 0 if site_of is None else int(site_of[int(m[0])])
        by_site.setdefault(site, []).append(tuple(int(o) for o in m))
    spelled = [by_site[s] for s in sorted(by_site)]
    return f"; one list per site, {spelled!r} states these multiplets"
