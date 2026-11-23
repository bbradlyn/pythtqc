"""The grammar of ``characters=`` and ``site_multiplets=``.

It is read where a call starts, before anything is measured.  A malformed
declaration raises ``ValueError``, whose message, after the call's name,
states the argument's grammar in one sentence.
"""

from __future__ import annotations

import numbers

import numpy as np

from . import _sitechars
from ._matrices import SiteMultiplets


def _spec_key(spec):
    """A comparable identity for one value of ``characters=``.

    ``spec`` is the value an orbital is given.  A word compares
    case-insensitively.  A mapping, of characters or of matrices, compares by
    its entries (:func:`_value_key`), so that the two orbitals of a declared
    spin pair given one mapping written twice are given one value.  Anything
    else compares by identity.
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
    """One value of a mapping of ``characters=``, as a comparable.

    ``value`` is a character, returned as a complex number, or a matrix,
    returned as its shape and entries.  Anything else is returned as its
    identity.
    """
    if _character_value(value):
        return complex(value)
    matrix = _matrix_value(value)
    if matrix is None:
        return ("id", id(value))
    return ("matrix", matrix.shape, tuple(complex(x) for x in matrix.ravel()))


# The grammar of a declaration.  Which words name something at a site is read
# later, in the site group the measurement reaches.

#: ``characters=`` in one sentence, which every error about its grammar states
#: after the call's name.
_CHARACTERS_GRAMMAR = (
    "`characters=` is a list with one entry per orbital, or a mapping from an "
    "orbital, or from several orbitals of one site written as one key, such "
    "as (0, 1), and each entry or value is a word or a mapping from "
    "operations to characters or to matrices on those orbitals, or, in a "
    "list, None")

#: ``site_multiplets=`` in one sentence, which every error about its grammar
#: states after the call's name.
_MULTIPLETS_GRAMMAR = (
    "`site_multiplets=` is a SiteMultiplets, or one list per site of the "
    "multiplets it holds, each written as its orbitals, such as [[(0, 1)], "
    "[(2,)]], and each orbital of the model is in exactly one multiplet")


def _shown(value, width=60) -> str:
    """``value`` as an error of grammar quotes it.

    Returns its type and its repr, cut to ``width`` characters, or ``"None"``.
    """
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
    """``value`` as a square complex matrix of finite numbers, or ``None``.

    A nested list or an array of two dimensions, square and nonempty, whose
    entries are all finite numbers, is a matrix.  A string, a mapping, a
    single number, a truth value or a ragged list is not.
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
    r"""The multiplets ``site_multiplets=`` states, both declarations checked.

    ``norb`` is the model's number of orbitals, and ``characters`` and
    ``multiplets`` the values of ``characters=`` and ``site_multiplets=``, or
    ``None``.  ``spin`` is the declared :class:`..spin.SpinBasis` or ``None``,
    and ``site_of`` each orbital's site, or ``None`` where the sites are not
    read.  :func:`_characters_grammar` and :func:`_multiplets_grammar` check
    the two declarations, and a malformed one raises ``ValueError`` there.
    Whether a word names something at its site is not read here: it depends
    on the site group the measurement reaches.  Returns the multiplets, every
    site's in one sequence, or ``None`` where ``multiplets`` is ``None``.
    """
    pairs = () if spin is None else tuple((int(u), int(d)) for u, d in spin.pairs)
    if characters is not None:
        _characters_grammar(characters, int(norb), pairs, site_of)
    if multiplets is None:
        return None
    return _multiplets_grammar(multiplets, int(norb), pairs, site_of)


def _names_joint_keys(characters) -> bool:
    """Whether ``characters``, the value of ``characters=``, has a joint key.

    A joint key names several orbitals, and the grammar checks that they lie
    at one site, so the sites are read before it.
    """
    return isinstance(characters, dict) and any(
        isinstance(key, (tuple, list)) and len(key) > 1 for key in characters)


def _characters_grammar(characters, norb, pairs, site_of=None) -> None:
    """Raise ``ValueError`` where ``characters=`` is malformed.

    ``characters`` is the value of ``characters=``, ``norb`` the model's
    number of orbitals, ``pairs`` the ``(up, down)`` orbitals of each declared
    spin pair, and ``site_of`` each orbital's site, or ``None`` where the sites
    are not read.  ``characters=`` is a list with one entry per orbital, or a
    mapping from an orbital, or from several orbitals of one site written
    together.  Each value is a word, or a mapping from operations to
    characters or to matrices, and a list's entry may be ``None``.  Each
    orbital is named once, and is one of the model's.  A key of several
    orbitals names orbitals of one site, where ``site_of`` is given.  A
    mapping from operations names at least one, by its index or its name.  It
    gives every operation a finite number, or every operation a square matrix
    with a row for each of its orbitals, a spin pair counted as one spinful
    orbital.  The
    two orbitals of a spin pair, where both are declared, are given one value.
    A value that is
    a :class:`.._sitechars.DeclaredContent`, a declaration already read and
    carried from one group to the next, is passed over.
    """

    def malformed(what):
        raise ValueError(f"realized_group(): {_CHARACTERS_GRAMMAR}; {what}.")

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
                malformed(f"orbital {o} is named twice, in "
                          + (where if holder[o][1] == where
                             else f"{holder[o][1]} and in {where}"))
            holder[o] = (value, where)
        if site_of is not None and len(orbs) > 1:
            sites = sorted({int(site_of[o]) for o in orbs})
            if len(sites) > 1:
                malformed(f"{where} names orbitals of sites {sites}; give the "
                          "orbitals of each site a key of their own")
        if isinstance(value, str):
            continue
        if isinstance(value, dict):
            if not value:
                malformed(f"{where} is given an empty mapping; name at least one "
                          "operation")
            # A matrix is the operation's matrix on the orbitals named, one row
            # and one column per orbital, a spin pair counting as one spinful
            # orbital
            pair_of = {int(o): c for c, pair in enumerate(pairs) for o in pair}
            width = len({pair_of.get(int(o), ("orbital", int(o))) for o in orbs})
            kinds = set()
            for op, chi in value.items():
                if not (_orbital_index(op) or isinstance(op, str)):
                    malformed(f"{where} names an operation by {_shown(op)}; name it "
                              "by its index or its name, such as 3 or 'C4(z)'")
                if _character_value(chi):
                    kinds.add("character")
                    continue
                matrix = _matrix_value(chi)
                if matrix is None:
                    malformed(f"{where} gives the operation {op!r} {_shown(chi)}; give "
                              "it a finite number, or a square matrix of finite numbers")
                if matrix.shape[0] != width:
                    noun = "Wannier function" if pairs else "orbital"
                    malformed(f"{where} gives the operation {op!r} a {matrix.shape[0]}x"
                              f"{matrix.shape[0]} matrix on {width} "
                              f"{noun}{'' if width == 1 else 's'}; give a {width}x{width} "
                              f"matrix, one row and one column per {noun} in the order "
                              "written")
                kinds.add("matrix")
            if len(kinds) > 1:
                malformed(f"{where} gives some operations characters and others matrices; "
                          "give every operation a character, or every operation a "
                          "matrix")
            continue
        if value is None:
            malformed(f"{where} is given None; leave the orbital out of the mapping "
                      "to declare nothing about it")
        malformed(f"{where} is given {_shown(value)}")
    # the up and the down state of one spinful orbital are given one value
    for c, (up, dn) in enumerate(pairs):
        if (up in holder and dn in holder
                and _spec_key(holder[up][0]) != _spec_key(holder[dn][0])):
            raise ValueError(
                f"realized_group(): {_CHARACTERS_GRAMMAR}. Orbitals {up} and {dn}, "
                f"spin pair {c}, "
                f"are given {holder[up][0]!r} and {holder[dn][0]!r}; give both the same "
                "value, or declare only one of them.")


def _multiplets_grammar(multiplets, norb, pairs, site_of) -> tuple:
    """The multiplets ``site_multiplets=`` states, its grammar checked.

    ``multiplets`` is the value of ``site_multiplets=``, and ``norb``,
    ``pairs`` and ``site_of`` are as :func:`_characters_grammar` takes them.
    ``site_multiplets=`` is one list per site of the multiplets it holds, each
    a nonempty list of orbitals, or a :class:`SiteMultiplets`.  Its multiplets
    hold every orbital of the model once, both orbitals of each spin pair in
    one multiplet, and, where ``site_of`` is given, each multiplet the
    orbitals of one site.  Returns the multiplets, every site's in one
    sequence, each multiplet its orbitals in the order written.  Raises
    ``ValueError`` where ``site_multiplets=`` is malformed.
    """

    def malformed(what):
        raise ValueError(f"realized_group(): {_MULTIPLETS_GRAMMAR}; {what}.")

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
                malformed(f"entry {i} holds an empty multiplet")
            for o in orbs:
                if not _orbital_index(o):
                    malformed(f"the multiplet {tuple(orbs)!r} names an orbital by "
                              f"{_shown(o)}; name it by its index")
                if not 0 <= int(o) < norb:
                    malformed(f"the multiplet {tuple(orbs)!r} names orbital {int(o)}, out "
                              f"of range for a model with {norb} orbitals")
            flat.append(tuple(int(o) for o in orbs))
    holder = {}
    for j, orbs in enumerate(flat):
        for o in orbs:
            if o in holder:
                malformed(f"orbital {o} is named twice, in {orbs!r}"
                          + ("" if holder[o] == j else f" and {flat[holder[o]]!r}"))
            holder[o] = j
    missing = tuple(o for o in range(norb) if o not in holder)
    if missing:
        malformed("no multiplet holds "
                  f"orbital{'s' if len(missing) > 1 else ''} "
                  + (str(missing) if len(missing) > 1 else str(missing[0])))
    for c, (up, dn) in enumerate(pairs):
        if holder[up] != holder[dn]:
            malformed(f"spin pair {c}, orbitals ({up}, {dn}), is split between "
                      f"{flat[holder[up]]!r} and {flat[holder[dn]]!r}; place both "
                      "orbitals of a spin pair in one multiplet")
    if site_of is not None:
        for orbs in flat:
            sites = sorted({int(site_of[o]) for o in orbs})
            if len(sites) > 1:
                malformed(f"the multiplet {orbs!r} spreads over sites {sites}; give "
                          "each multiplet the orbitals of one site")
    return tuple(flat)


def _per_site_spelling(stated, norb, site_of) -> str:
    """The multiplets of a flat list, offered one list per site.

    ``stated`` is the value of ``site_multiplets=``, ``norb`` the model's
    number of orbitals, and ``site_of`` each orbital's site or ``None``.  A
    flat list such as ``[(0,), (1,)]`` lists multiplets where
    ``site_multiplets=`` lists sites.  Where every entry of ``stated`` is a
    nonempty list of this model's orbitals, returns the clause of the error
    that offers them grouped by ``site_of``, all at one site where it is
    ``None``.  For ``[(0,), (1,)]`` with ``site_of = [0, 1]`` the clause is
    ``"; one list per site, [[(0,)], [(1,)]] states these multiplets"``.
    Returns ``""`` otherwise.
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
