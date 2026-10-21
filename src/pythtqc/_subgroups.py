"""The subgroups of a group that lie inside the operations the Hamiltonian keeps,
and the tie rule that picks one where several of the same order qualify with
nothing measured between them.
"""

from __future__ import annotations

import numpy as np

from ._seitz import _ROTATION_ORDER, _direction_text, _rotation_axis


# --------------------------------------------------------------------------
# an operation as a comparable key, and whether it keeps a layer's side
# --------------------------------------------------------------------------

def _operation_key(sym, tol=1e-6) -> tuple:
    r"""One operation as a comparable key: ``(rotation, translation mod 1, coset)``.

    A space-group operation is its integer rotation together with its
    translation **modulo the lattice**, and the antiunitary coset is a third
    coordinate rather than a detail — a primed operation is not the unprimed one.
    Used to match the two environments' selections against each other, which is
    the only place two independently identified groups of the *same* cell have
    to be compared operation for operation.

    The reduction snaps ``1 − ε`` and ``ε`` alike onto ``0``, so a translation
    that spglib returns just below one is the same key as one just above zero.
    """
    R = np.rint(np.asarray(sym.rotation, dtype=float)).astype(int)
    t = np.asarray(sym.translation, dtype=float).reshape(-1)

    def _reduced(v):
        w = float(v) % 1.0
        return 0.0 if min(w, 1.0 - w) < tol else round(w, 6)

    return (tuple(int(x) for x in R.ravel()), tuple(_reduced(x) for x in t),
            bool(sym.time_reversal))


def _side_preserving(sym, padded) -> bool | None:
    r"""Does this operation keep the layer's side?  ``None`` if it is not a layer op.

    A layer operation is block-diagonal in the non-periodic direction —
    ``diag(g₂, ±1)`` — and the sign is what a substrate decides: ``+1`` keeps
    each side where it is, ``−1`` exchanges them and a substrate removes it.
    ``None`` where the rotation mixes the plane with the vacuum direction, which
    is not a layer operation at all.
    """
    R = np.rint(np.asarray(sym.rotation, dtype=float)).astype(int)
    d = int(padded)
    per = [i for i in range(3) if i != d]
    if (R[per, d] != 0).any() or (R[d, per] != 0).any():
        return None
    if abs(int(R[d, d])) != 1:                  # pragma: no cover - integer cell
        return None
    return int(R[d, d]) == 1


def _descent_closure(group, kept) -> tuple:
    r"""Products of the kept set that leave it — the descent's own closure gate.

    A set of operations can only *be* a group's operation set if it closes, and
    the kept set is a **measurement**: the detector's verdicts are per operation
    (per k-action class, strictly), and nothing forces the survivors to compose.
    :attr:`ValidatedGroup.closed` reports the same thing for the whole surviving
    set; this asks it of the **unitary** part alone, which is the part a descent
    is built on.  Returns the offending ``(a, b, ab)`` triples of ``.ind``, empty
    when the set closes.  The product is taken modulo lattice translations,
    through the group's own product table.
    """
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    keep = {int(s.ind) for s in kept}
    leaks = []
    for a in sorted(keep):
        for b in sorted(keep):
            c = int(group.symmetries[int(prod[at[a], at[b]])].ind)
            if c not in keep:
                leaks.append((a, b, c))
    return tuple(leaks)


def _subgroups_inside(group, kept, avoid=None) -> set:
    r"""Every subgroup of ``group`` whose operations all lie in ``kept``.

    Enumerated the only way that is both complete and cheap: a subgroup is the
    **join of its cyclic subgroups**, so the cyclic closures of the kept
    operations are computed first and then joined pairwise until nothing new
    appears, with any closure that leaves ``kept`` abandoned as soon as it does.
    The cell groups this runs on have at most 192 operations and the kept sets
    at most a few dozen, so the whole lattice is found in milliseconds.

    ``kept`` is the **unitary** kept set (the descent's own), and the products
    are taken modulo lattice translations through the group's own product table.
    ``avoid`` is a set of pairs of ``.ind``, each a pair no subgroup returned may
    hold both of -- two operations whose matrices on these states carry a sign
    no single-valued representation has -- and a closure is abandoned as soon as
    it holds one; every subgroup of a subgroup free of such pairs is free of
    them too, so the enumeration stays complete.  Returns a set of frozensets of
    ``.ind``; ``{E}`` is always among them.  The subgroups inside one set are
    found once per group and kept on it, since every reading that searches them
    asks the same group again.
    """
    keep = frozenset(int(i) for i in kept)
    memo = getattr(group, "_subgroups", None)
    key = (keep, frozenset(frozenset(int(i) for i in pair) for pair in (avoid or ())))
    if memo is not None and key in memo:
        return set(memo[key])
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    ind_of = [int(s.ind) for s in group.symmetries]
    apart: dict = {}
    for pair in (avoid or ()):
        a, b = (int(i) for i in pair)
        apart.setdefault(a, set()).add(b)
        apart.setdefault(b, set()).add(a)

    def close(seed):
        cur, frontier = set(seed), list(seed)
        if apart and any(apart.get(x, set()) & cur for x in cur):
            return None
        while frontier:
            a = frontier.pop()
            for b in list(cur):
                for x, y in ((a, b), (b, a)):
                    c = ind_of[int(prod[at[x], at[y]])]
                    if c not in cur:
                        if c not in keep or (apart and apart.get(c, set()) & cur):
                            return None
                        cur.add(c)
                        frontier.append(c)
        return frozenset(cur)

    subs = {c for c in (close({a}) for a in sorted(keep)) if c is not None}
    frontier = set(subs)
    while frontier:
        fresh = set()
        for s in frontier:
            for t in subs:
                j = close(s | t)
                if j is not None and j not in subs and j not in fresh:
                    fresh.add(j)
        subs |= fresh
        frontier = fresh
    if memo is not None:
        memo[key] = frozenset(subs)
    return subs


def _resolve_closure(group, kept):
    r"""The group inside a kept set that does not close — what it cannot carry, removed.

    The measurement's survivors need not compose: the detector's verdicts are
    per operation, and nothing forces the set they leave to be a group.  An
    operation sitting on the wrong side of the declared tolerance breaks
    closure while its partners keep it.  A **layer** cell's ``σ_h`` fibers,
    both members of each in the cell, do not: the tensor test holds each member
    to its own lift, on the multiplets where each is one orbital and, in a
    spinful model, on the whole solution space whatever the multiplets, and
    keeps one member of each fiber the Hamiltonian breaks -- on the SRSI
    paper's ``P41'`` cell read free-standing ``C4`` and ``C4⁻¹``, ``S4`` and
    ``S4⁻¹`` refuted by tensor structure -- and no layer fixture reaches this
    step.

    So: enumerate the subgroups inside the kept set (:func:`_subgroups_inside`)
    and return the largest.  Where the largest order is reached by **more than
    one** subgroup, the measurement does not choose between them, and the one
    returned is the first by the tie rule every such choice is made by
    (:func:`_tie_ranked`): on a layer the most operations that keep its side,
    then the most rotations, then the highest-order rotation, then its axis in
    the conventional cell.  Their intersection would keep less than any of them
    carries.

    Returns ``(resolved, dropped, tied)`` — the returned set of ``.ind``; per
    dropped operation ``(ind, why)`` with ``why`` ``"closure"`` for one no
    candidate subgroup carries — the kept set does not close with it — and
    ``"tie"`` for one some candidates carry and the one kept does not; and the
    candidates in the rule's order, the one kept first, or ``()`` where the
    largest is unique.
    """
    subs = _subgroups_inside(group, kept)
    top = max(len(s) for s in subs)
    ranked = _tie_ranked(group, [s for s in subs if len(s) == top])
    keep = sorted(int(i) for i in kept)
    in_some = set().union(*(set(c) for c in ranked))
    resolved = frozenset(ranked[0])
    dropped = tuple((i, "tie" if i in in_some else "closure")
                    for i in keep if i not in resolved)
    return resolved, dropped, (ranked if len(ranked) > 1 else ())


def _setting_axes(group) -> dict:
    r"""Each unitary operation's determinant, rotation order and axis, in the conventional cell.

    ``{ind: (det, order, axis)}``, read the way a Seitz symbol is written
    (:func:`.._seitz._seitz_key`): the rotation carried into the tables' own conventional
    setting, the order that of its proper part ``det·R``, and the axis that
    part turns about -- a mirror's normal -- as the integer direction
    :func:`.._seitz._primitive_direction` writes, empty for the identity and inversion.
    A group with no conventional setting is read in its own cell.
    """
    ref = getattr(group, "_refUC", None)
    out = {}
    for sym in group.symmetries:
        if bool(sym.time_reversal):
            continue
        rotation = (sym.rotation_refUC(np.asarray(ref, dtype=float)) if ref is not None
                    else sym.rotation)
        rotation = np.rint(np.asarray(rotation, dtype=float)).astype(int)
        det = int(round(float(np.linalg.det(rotation))))
        proper = det * rotation
        order = _ROTATION_ORDER[int(round(float(np.trace(proper))))]
        out[int(sym.ind)] = (det, order, () if order == 1 else _rotation_axis(proper))
    return out


def _direction_rank(axis) -> tuple:
    r"""Where a direction of the conventional cell stands in the tie rule.

    ``axis`` is ``[uvw]``, its first nonzero index positive.  Fewest nonzero
    indices first, then the smallest sum of their sizes, then the larger
    ``w``, ``u`` and ``v`` in that order: ``[001]``, ``[100]``, ``[010]`` --
    the ``c``, ``a`` and ``b`` axes -- then ``[101]``, ``[011]``, ``[110]`` and
    ``[1-10]``, and ``[111]`` first among the body diagonals.  No direction at
    all -- inversion's -- stands before every one.
    """
    if not axis:
        return ()
    u, v, w = (int(x) for x in axis)
    return (sum(1 for x in (u, v, w) if x), abs(u) + abs(v) + abs(w), (-w, -u, -v))


def _principal_axis(rows) -> tuple:
    r"""The axis the tie rule reads third, from a subgroup's ``(det, order, axis)`` rows.

    That of its highest-order rotation, or where it keeps no rotation but the
    identity, of its highest-order improper operation; the first in the order
    :func:`_direction_rank` fixes where several share that order, and ``()``
    where no operation has an axis.
    """
    proper = [r for r in rows if r[0] > 0]
    top = max(r[1] for r in proper)
    pool = [r for r in proper if r[1] == top and r[2]]
    if not pool:
        improper = [r for r in rows if r[0] < 0 and r[2]]
        if improper:
            highest = max(r[1] for r in improper)
            pool = [r for r in improper if r[1] == highest]
    return min((r[2] for r in pool), key=_direction_rank, default=())


def _layer_sides(group) -> dict:
    r"""Which unitary operations of a layer group keep the layer's side, ``{ind: bool}``.

    ``True`` for ``R[2,2] = +1`` along the one non-periodic direction
    (:func:`_side_preserving`), ``False`` for an operation that exchanges the
    two sides.  Empty where the group is no layer's -- a fully periodic crystal
    has no non-periodic direction -- and where none of its operations flips the
    side, a substrate's group among them: the tie rule's first key then has
    nothing to count.
    """
    padded = tuple(int(d) for d in getattr(group, "vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return {}
    sides = {int(s.ind): _side_preserving(s, padded[0]) is True
             for s in group.symmetries if not s.time_reversal}
    return {} if all(sides.values()) else sides


def _tie_key(subgroup, axes, sides=None) -> tuple:
    r"""The key the tie rule takes the smallest of.

    Among subgroups of one order, each read through ``axes``
    (:func:`_setting_axes`), and on a layer through ``sides``
    (:func:`_layer_sides`): first, on a layer, the most operations that keep
    the layer's side, since those hold in every environment and are what the
    layer keeps on a substrate; then the most rotations -- proper operations,
    the identity among them -- so that a subgroup keeping rotations stands
    before one keeping mirrors or rotoinversions; then the highest order among
    its rotations; then its principal axis (:func:`_principal_axis`) in the
    order :func:`_direction_rank` fixes; then every operation's order and axis
    the same way; and last the operations' indices, so that the rule is total.
    Where ``sides`` is empty the first entry is ``0`` for every subgroup, and the
    rule is the rest of it.
    """
    rows = [axes[int(i)] for i in subgroup]
    proper = [r for r in rows if r[0] > 0]
    every = sorted((-r[1], r[0] < 0, _direction_rank(r[2])) for r in rows)
    kept_side = sum(1 for i in subgroup if (sides or {}).get(int(i)))
    return (-kept_side, -len(proper), -max(r[1] for r in proper),
            _direction_rank(_principal_axis(rows)), every,
            tuple(sorted(int(i) for i in subgroup)))


def _tie_ranked(group, subgroups) -> tuple:
    r"""Subgroups of one order the measurement does not choose between, as the tie rule ranks them.

    Wherever the group a Hamiltonian obeys is the largest subgroup of some set
    and several subgroups of that order qualify -- the survivors that do not
    close (:func:`_resolve_closure`), the readings a declaration leaves
    (:func:`.._declaration._covariant_subgroup`), the subgroups on which spinless
    states carry an ordinary representation (:func:`.._descent._projective_class_refuted`)
    -- the one kept is the first here, never their intersection, which would
    keep less than any of them carries.  The rule, read in the conventional
    cell of ``group`` (:func:`_tie_key`): on a layer whose group has operations
    that flip its side, first the most operations that keep the side
    (``R[2,2] = +1``), since those hold in every environment and are what the
    layer keeps on a substrate; then the most rotations, so that a subgroup
    keeping rotations stands before one keeping mirrors or rotoinversions; then
    the highest-order rotation; then its axis, ``c`` before ``a`` before ``b``
    before the diagonals; then every other operation's axis the same way.

    Returns the subgroups, each as its ``.ind`` in order, the one kept first.
    """
    axes = _setting_axes(group)
    sides = _layer_sides(group)
    return tuple(sorted((tuple(sorted(int(i) for i in s)) for s in subgroups),
                        key=lambda s: _tie_key(s, axes, sides)))


def _tie_rule_words(group) -> str:
    r"""The tie rule as the record states it: its first keys, in words.

    The layer's own key is said only where it can decide something -- on a
    layer whose group has operations that flip its side (:func:`_layer_sides`).
    """
    return (("on a layer the most operations that keep its side, then " if
             _layer_sides(group) else "")
            + "the most rotations, then the highest-order rotation, then its axis in "
            "the conventional cell")


def _tie_reason(group, ranked) -> str:
    r"""Why the tie rule keeps the first of ``ranked`` rather than the second, in words.

    The first key of :func:`_tie_key` on which the two differ, said as what the
    one kept has: more operations that keep a layer's side, more rotations, a
    rotation of higher order, or a principal axis that comes first in the
    conventional cell.
    """
    axes = _setting_axes(group)
    sides = _layer_sides(group)
    kept, other = ([axes[int(i)] for i in s] for s in ranked[:2])
    mine, theirs = (_tie_key(s, axes, sides) for s in ranked[:2])
    if mine[0] != theirs[0]:
        return (f"it holds {-mine[0]} operations that keep the layer's side "
                f"(R[2,2] = +1) where the next holds {-theirs[0]}, each counted once per "
                "coset of the lattice translations")
    if mine[1] != theirs[1]:
        return (f"it holds {-mine[1]} rotations (proper operations) where the next "
                f"holds {-theirs[1]}, each counted once per coset of the lattice "
                "translations")
    if mine[2] != theirs[2]:
        return (f"its highest rotation is {-mine[2]}-fold where the next one's is "
                f"{-theirs[2]}-fold")
    if mine[3] != theirs[3]:
        here, there = _principal_axis(kept), _principal_axis(other)
        if not here:
            # only inversion is left to read, and no direction at all stands
            # before every one
            return (f"it keeps inversion where the next keeps an axis, "
                    f"[{_direction_text(there)}], and no direction stands before any")
        return (f"its principal axis, [{_direction_text(here)}], comes before "
                f"[{_direction_text(there)}] in the conventional cell")
    if mine[4] != theirs[4]:
        return ("its other operations' orders and axes come first in the conventional "
                "cell")
    return "its operations come first in the group's own order"
