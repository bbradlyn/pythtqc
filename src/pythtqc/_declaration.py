"""What ``characters=`` declares, read on the multiplets the Hamiltonian measures.

Each site's declared site irreps are read in its site group, and the operations
that carry them are kept.  Each operation's solution space then gives a verdict
on the declaration, with the scalars the hoppings pin.  Where the declaration
breaks the group and the group cannot be lowered, the refusal is
``declaration_breaks_group``.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from . import _sectors
from . import _sitechars
from .refusal import InternalError, Refusal
from ._tolerance import CHARACTER, IN_SPACE, SPREAD, ZERO, _IN_BLOCKS, noise_floor
from ._text import _counted_as, _fmt_char, _label_terms
from ._seitz import _record_name
from ._subgroups import (
    _closes, _descent_closure, _resolve_closure, _subgroups_inside, _tie_ranked,
    _tie_reason, _tie_rule_words)
from ._detector import _READOFF_MAX_BYTES
from ._matrices import (
    RepAnchor, _ansatz_blocks, _atom_blocks, _atoms_of_site, _bfs_words,
    _channel_classes, _channel_permutation, _characters_of_subgroup, _closure,
    _filled_check, _identity_position, _in_blocks, _lift_of, _minimal_generators,
    _multiplets_of, _primed_partners, _product_table_residual, _propagated,
    _proportionality, _settle_unit)
from ._grammar import _declaration_grammar
from ._naming import _SiteTableOf, _contained, _site_tables
from ._multiplets import (
    _MultipletBlocks, _channel_perms, _multiplet_blocks, _reversal_scalars,
    _transported_lifts)


@dataclass(frozen=True, eq=False)
class _Row:
    """One row of the measured-versus-declared table.

    A row is one declaration under one operation.  A declaration over several
    multiplets is read as the sum of their characters.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals the declaration names.
    ind : int
        The operation's index in the group.
    name : str
        The operation's name from its Cartesian rotation
        (:func:`.._sitechars.operation_name`).
    measured : complex
        The character the hoppings give the orbitals, read against the scale
        of their coupled part.
    declared : complex
        The character the declaration states.
    part : int
        The coupled part of the crystal the declaration is read in: the
        multiplets the hoppings connect, directly or through others.
    required : complex or None
        The value the part's one-dimensional character has to take for the
        declaration to hold.  ``None`` where the measured character vanishes,
        or where the declared matrices are not proportional to the measured.
    matrices : tuple of ndarray or None
        The declared and the measured generator matrix, where the declaration
        states matrices; ``None`` otherwise.
    """

    orbitals: tuple
    ind: int
    name: str
    measured: complex
    declared: complex
    part: int
    required: complex | None
    matrices: tuple | None = None


@dataclass(frozen=True)
class _ReadTogether:
    """A declaration read over several multiplets at once, operation by operation.

    Attributes
    ----------
    decl : dict
        The declaration, as :func:`_normalize_declaration` returns it.
    theta : dict
        The character each of its multiplets carries at each operation,
        ``{multiplet: {ind: character}}``, read against the multiplet's own
        coupled part's scale.  On an orbit of sectors built by induction, it
        is read under the representation that orbit is assembled with.
    part : dict or None
        The coupled part each multiplet is in, ``{multiplet: part}``; ``None``
        on an orbit of sectors built by induction, which the sectors alone
        weigh.
    site_group : tuple of int or None
        The indices of the operations kept that fix the declaration's site.
    table_of : _SiteTableOf or None
        Called with a subgroup of the site group, the site's character table
        on it.
    lift_trace : dict or None
        The trace of each operation's spin-1/2 lift, by index.
    """

    decl: dict
    theta: dict
    part: dict | None = None
    site_group: tuple | None = None
    table_of: object = None
    lift_trace: dict | None = None


@dataclass(frozen=True)
class _CovariantSubgroup:
    """The largest subgroup a declaration is covariant on, and what it leaves out.

    Attributes
    ----------
    subgroup : frozenset of int
        The indices of the subgroup returned.
    candidates : tuple
        If nonempty, the subgroups of the largest order that tie, in the tie
        rule's order, the one returned first.
    broken : tuple of int
        The indices of the operations asked about that the subgroup leaves out.
    """

    subgroup: frozenset
    candidates: tuple
    broken: tuple


@dataclass(frozen=True, eq=False)
class _Verdict:
    """A verdict per operation off the solution spaces, and what the hoppings pin.

    Attributes
    ----------
    rows : tuple of _Row
        The measured-versus-declared table.
    parts : tuple of frozenset
        The coupled parts of the crystal as sets of multiplet orbits, the
        anchor's first.
    requirements : tuple of dict
        Per part, what each operation asks of its one-dimensional character,
        ``{ind: value}``, ``None`` where the declaration breaks the operation.
    scalars : tuple of dict
        Per part, that value before it is rounded onto a sign or a phase.
    matrices : tuple of dict
        Per part, the requirements at the operations declared generator
        matrices were read at.
    joint : tuple of _ReadTogether
        The declarations read over several coupled parts, or on part of a site.
    by_sectors : tuple of _ReadTogether
        The declarations on orbits of sectors built by induction.
    reasons : dict
        One sentence per broken operation, by index.
    classes : ndarray of int
        The multiplet orbit of each multiplet.
    nclass : int
        The number of multiplet orbits.
    chosen : dict
        The class scalar the hoppings pin, ``{(ind, class): value}``.
    within : dict
        The ratio an orbit of a part free of the anchor holds to that part's
        first, ``{(ind, class): value}``.
    free : frozenset
        The ``(ind, class)`` pairs the hoppings leave open.
    free_classes : frozenset of int
        The classes free at every operation that was measured.
    seen : dict
        The classes measured at each operation, ``{ind: frozenset}``.
    anchor : int
        The multiplet every scalar is read against, orbital 0's.
    reference_orbital : int
        The orbital the lowest-angular-momentum rule reads the crystal from first.
    declared : tuple
        What was declared, as :func:`_declared_key` writes it.
    space_tol : float
        How far outside the blocks a member may sit and count as inside,
        dimensionless.
    indices : frozenset of int
        The indices of the operations read.
    blocks : _MultipletBlocks
        The multiplets' own blocks, with every orbit of sectors built by induction.
    reversal : ndarray or None
        What time reversal carries on each orbit; ``None`` where it was not read.
    search : SectorSearch or None
        What the declaration asks of the sectors on any subgroup; ``None``
        where they read nothing.
    """

    rows: tuple
    parts: tuple
    requirements: tuple
    scalars: tuple
    matrices: tuple
    joint: tuple
    by_sectors: tuple
    reasons: dict
    classes: object
    nclass: int
    chosen: dict
    within: dict
    free: frozenset
    free_classes: frozenset
    seen: dict
    anchor: int
    reference_orbital: int
    declared: tuple
    space_tol: float
    indices: frozenset
    blocks: object
    reversal: object
    search: object


def _covariant_subgroup(group, kept_u, requirements, *, unitary_only=False, tol=CHARACTER,
                        joint=(), search=None, matrices=()):
    r"""The largest subgroup the declared site irreps are covariant on.

    The hoppings fix every multiplet's block up to one one-dimensional
    character per coupled part of the crystal, the multiplets they connect
    directly or through others.  A subgroup ``S`` keeps the declaration where
    every part has a one-dimensional character of ``S`` that meets all of its
    requirements.  The representation on each part is then the declared one
    restricted to ``S``, times that character.  Where ``search`` is given and
    complete, every operation having a reading on the sectors, the sectors
    decide (:func:`_subgroup_read`).  The requirements decide where the sectors
    read nothing, or where the subgroup they keep fails the declared matrices.
    Where several subgroups of the largest order tie, the first by the tie rule
    (:func:`.._subgroups._tie_ranked`) is returned and all are recorded.  The
    identity always keeps the declaration.

    Parameters
    ----------
    group : ..groups.GroupData
        The group the operations belong to.
    kept_u : iterable of int
        The indices of the unitary operations the measurement kept.
    requirements : dict or sequence of dict
        Per coupled part, ``{ind: value}``, the value the part's character has
        to take at each unitary operation for its declarations to hold,
        ``None`` where the operation is broken on its own.  An operation absent
        from a mapping constrains nothing.  A single mapping is one part.
    unitary_only : bool, default False
        Whether the group keeps no antiunitary operation.  With
        ``unitary_only=True`` every one-dimensional character of a subgroup,
        the complex ones included, is a candidate; otherwise only the real
        ones are.
    tol : float, default CHARACTER
        How far a character may sit from the value a requirement asks and
        still meet it (:func:`_meets`).  Callers pass the declared tolerance
        where that is larger.
    joint : sequence of _ReadTogether, default ()
        The declarations read over several coupled parts, or on part of a
        site (:attr:`_Verdict.joint`).  A subgroup keeps each where one
        character for each part it spans carries it (:func:`_joint_holds`).
    search : _sectors.SectorSearch or None, default None
        What the declaration asks of the sectors on any subgroup; ``None``
        where the sectors read nothing.
    matrices : sequence of dict, default ()
        Per coupled part, the requirements at the operations declared
        generator matrices were read at (:attr:`_Verdict.matrices`).

    Returns
    -------
    _CovariantSubgroup or Refusal
        The subgroup, the subgroups that tie, and the operations of ``kept_u``
        left out.  Where ``kept_u`` is no group and holds several largest
        subgroups, only the operations the declaration breaks on their own are
        left out.  The refusal ``rep_search_too_large`` where the sectors have
        more representations to weigh than ``search.max_candidates``
        (:func:`_subgroup_read`).
    """

    def _carries_matrices(sub):
        """Whether each part's declared matrices are met inside ``sub`` by one character of it."""
        inside = {int(i) for i in sub}
        asked = [dict(m) for m in (matrices or ()) if any(int(g) in inside for g in m)]
        if not asked:
            return True
        chars = (_characters_of_subgroup(group, sub, roots=bool(unitary_only))
                 or [{int(i): 1.0 + 0.0j for i in sub}])
        return all(any(_meets(chi, m, over=inside, tol=tol) for chi in chars) for m in asked)

    if search is not None and search.complete:
        got = _subgroup_read(group, kept_u, search)
        if isinstance(got, Refusal) or (got is not None and _carries_matrices(got.subgroup)):
            return got
    parts = ([requirements] if isinstance(requirements, dict)
             else [dict(part) for part in requirements])
    joint = tuple(joint or ())
    keep = {int(i) for i in kept_u}

    def _barred(i):
        return any(part.get(int(i), "free") is None for part in parts)

    # The group the survivors form without the declaration.  The operations a
    # measurement keeps need not compose: on a layer, the partner of a kept
    # operation under the mirror through the plane (sigma_h), whose lift
    # coincides with the kept one's, survives beside it until the closure step
    # removes it.  So the declaration is read inside the group the survivors
    # do form, and such a partner is neither kept by the declaration nor
    # charged to it.
    closes = _closes(group, keep)
    if closes:
        # the survivors are a group, the one largest inside them
        widest = [frozenset(keep)]
    else:
        plain = _subgroups_inside(group, keep)
        widest = sorted((s for s in plain if len(s) == max(len(t) for t in plain)),
                        key=sorted) if plain else []
    if len(widest) != 1:
        # The survivors are not a group, and no largest subgroup of them is
        # unique.  A declaration is read on a group, and the closure resolution
        # finds that group first, keeping the candidate the tie rule ranks
        # first.  So only the operations the declaration breaks on their own
        # are charged to it here, and the declaration is read again on the
        # group that resolution keeps.
        bad = {i for i in keep if _barred(i)}
        return _CovariantSubgroup(subgroup=frozenset(keep - bad), candidates=(),
                                  broken=tuple(sorted(bad)))
    baseline = set(widest[0])
    allowed = {i for i in baseline if not _barred(i)}

    def _met(sub):
        """Whether every coupled part has a character of ``sub`` meeting its requirements.

        Where declarations read together (``joint``) speak inside ``sub``, one
        character for each part they span meets its own part's requirements
        and carries those declarations at once.
        """
        inside = {int(i) for i in sub}
        asked = [i for i, wants in enumerate(parts) if any(int(g) in inside for g in wants)]
        together = [j for j in joint
                    if any(int(g) in inside for row in j.theta.values() for g in row)]
        if not asked and not together:
            return True
        chars = (_characters_of_subgroup(group, sub, roots=bool(unitary_only))
                 or [{int(i): 1.0 + 0.0j for i in sub}])
        own = {i: [chi for chi in chars if _meets(chi, parts[i], over=inside, tol=tol)]
               for i in asked}
        if not all(own[i] for i in asked):
            return False
        if not together:
            return True
        spanned = sorted({int(i) for j in together for i in j.part.values()})
        options = [own.get(i, chars) for i in spanned]
        for combo in itertools.product(*options):
            pick = dict(zip(spanned, combo))
            if all(_joint_holds(j, pick, inside, tol=tol) for j in together):
                return True
        return False

    if closes and _closes(group, allowed) and _met(frozenset(allowed)):
        # the operations the declaration allows are a group carrying it, the one
        # largest consistent subgroup, with no tie
        return _CovariantSubgroup(
            subgroup=frozenset(allowed), candidates=(),
            broken=tuple(sorted(i for i in baseline if i not in allowed)))
    consistent = {frozenset(int(i) for i in sub)
                  for sub in _subgroups_inside(group, allowed) if _met(sub)}
    if not consistent:                   # pragma: no cover - {E} always is
        return _CovariantSubgroup(subgroup=frozenset(), candidates=(),
                                  broken=tuple(sorted(keep)))
    # The largest consistent subgroup, with the tie rule only where several
    # share the largest order.  Maximality by inclusion alone would let a
    # small subgroup nothing constrains, such as the one generated by a
    # sigma_h partner the closure step removes, stand as a candidate beside the
    # group the declaration reaches.  A tie is two consistent subgroups of the
    # same order, and those are the candidates returned.
    top = max(len(s) for s in consistent)
    ranked = _tie_ranked(group, [s for s in consistent if len(s) == top])
    best = frozenset(ranked[0])
    candidates = ranked if len(ranked) > 1 else ()
    return _CovariantSubgroup(subgroup=best, candidates=candidates,
                              broken=tuple(sorted(i for i in baseline if i not in best)))


def _cyclic(group, ind):
    """The cyclic group of the operation ``ind`` in ``group``, by index."""
    prod = group._get_product_table(get_diff=False)
    at = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    ind_of = [int(s.ind) for s in group._symmetries]
    out, x = {int(ind)}, int(ind)
    while True:
        x = ind_of[int(prod[at[int(ind)], at[x]])]
        if x in out:
            return frozenset(out)
        out.add(x)


def _subgroup_read(group, kept_u, search):
    r"""The largest subgroup the declaration is carried on, read off the sectors.

    ``kept_u`` holds the indices of the unitary operations the measurement
    kept, and ``search`` what the declaration asks of the sectors of the
    Hamiltonian's conserved matrices (:class:`.._sectors.SectorSearch`).
    Where ``kept_u`` is a group, or holds one largest subgroup, that group's
    subgroups are asked from the largest order down
    (:meth:`.._sectors.SectorSearch.met`).  The first order at which any
    carries the declaration is kept, the tie rule choosing among several
    (:func:`.._subgroups._tie_ranked`).  Where ``kept_u`` holds several
    largest subgroups, only the operations whose own cyclic group does not
    carry the declaration are left out, and the declaration is read again on
    the group the closure resolution keeps.  Returns the
    :class:`_CovariantSubgroup`; ``None`` where the sectors do not read the
    group; the ``rep_search_too_large`` refusal where the representations to
    weigh at once are more than ``search.max_candidates``.

    Raises
    ------
    InternalError
        Where no subgroup carries the declaration, not even the identity alone.
    """

    keep = frozenset(int(i) for i in kept_u)
    try:
        if _closes(group, keep):
            # the survivors are a group, the one largest inside them, and where
            # it carries the declaration it is the subgroup kept, with no tie
            baseline = keep
            if search.met(baseline):
                return _CovariantSubgroup(subgroup=baseline, candidates=(), broken=())
        else:
            plain = _subgroups_inside(group, keep)
            top = max(len(s) for s in plain) if plain else 0
            widest = sorted((s for s in plain if len(s) == top), key=sorted)
            if len(widest) != 1:
                bad = {i for i in keep if not search.met(_cyclic(group, i))}
                return _CovariantSubgroup(subgroup=frozenset(keep - bad), candidates=(),
                                          broken=tuple(sorted(bad)))
            baseline = frozenset(widest[0])
        carried, size = [], None
        for sub in sorted(_subgroups_inside(group, baseline), key=lambda s: (-len(s), sorted(s))):
            if size is not None and len(sub) < size:
                break
            if search.met(sub):
                size = len(sub)
                carried.append(frozenset(sub))
    except _sectors._Unread:
        return None
    except _sectors._TooMany as many:
        return Refusal(
            "rep_search_too_large",
            f"comparing the declaration of orbitals {many.orbitals} with the "
            f"{many.count:,} representations the hoppings leave open on "
            f"{many.copies} identical cop{'y' if many.copies == 1 else 'ies'} of their "
            f"states exceeds max_candidates = {search.max_candidates:,}. Declare fewer "
            "of these orbitals, or pass a larger `max_candidates=`.",
            value=float(many.count), threshold=float(search.max_candidates),
            context={"orbitals": many.orbitals, "copies": many.copies,
                     "n_candidates": many.count, "max_candidates": search.max_candidates})
    if not carried:
        raise InternalError(
            f"no subgroup of {group.name} #{group.number_str}, not even the identity "
            "alone, "
            "carries the declaration on the sectors of the Hamiltonian's conserved "
            "matrices. Report it, with the model and the call.")
    ranked = _tie_ranked(group, carried)
    best = frozenset(ranked[0])
    return _CovariantSubgroup(subgroup=best, candidates=ranked if len(ranked) > 1 else (),
                              broken=tuple(sorted(i for i in baseline if i not in best)))


def _declaration_breaks(group, assignment, decls, verdict, *, standing, subgroup, broken,
                        readings, not_lowered, site_tables=None, admissible=None):
    r"""The ``declaration_breaks_group`` refusal.

    :func:`.._build._build_rep`, handed a fixed group it cannot lower, returns
    it.  The verdict is the one :func:`..groups.realized_group` descends on.
    The reason names the unitary operations the declaration is not covariant
    under and their primed partners, the subgroup it is covariant on, the
    subgroups that tie, what the failing declaration measured in site-irrep
    names, and the measured-versus-declared rows of the broken operations.

    Parameters
    ----------
    group : ..groups.GroupData
        The group the representation was asked for on.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites in ``group``.
    decls : sequence of dict
        The declarations, as :func:`_normalize_declaration` returns them.
    verdict : _Verdict
        As :func:`_verdicts_and_choices` returns it.
    standing : iterable of int
        The indices of the operations the verdict was read on, unitary and
        antiunitary.
    subgroup : iterable of int
        The indices of the operations the declaration is covariant on
        (:func:`_covariant_subgroup`), unitary alone or with their primed
        partners; the partners are added here.
    broken : iterable of int
        The indices of the operations it breaks, in the same form.
    readings : sequence of tuple of int
        The subgroups that tie, in the tie rule's order; empty where none do.
    not_lowered : str
        The sentence of the reason that says why the group is not lowered.
    site_tables : dict or None, default None
        The site character tables of ``group``
        (:func:`.._naming._site_tables`); ``None`` computes them here.
    admissible : list of dict or None, default None
        The real one-dimensional characters of the unitary operations,
        ``{ind: value}`` each; ``None`` computes them here.

    Returns
    -------
    Refusal
        ``declaration_breaks_group``, its context carrying ``group``.
    """

    lattice = np.asarray(group._lattice, dtype=float)
    by_ind = {int(s.ind): s for s in group._symmetries}
    standing = sorted({int(i) for i in standing})
    # each operation as the refusal lists it, and as the frame a declaration is
    # written in names it: a site's own operations by their rotations alone
    names = {i: _record_name(by_ind[i], lattice) for i in standing}
    frame = {i: (_sitechars.operation_name(by_ind[i], lattice)
                 + ("'" if by_ind[i].time_reversal else "")) for i in standing}
    unitary = [i for i in standing if not by_ind[i].time_reversal]
    # A site character is a character of the unitary site group, so the verdict
    # is about unitary operations.  The group returned is the grey group they
    # generate, and the primed copy of an operation stands or falls with it.  A
    # primed partner is counted as what it is, never as a unitary operation.
    partners = _primed_partners(group, names)
    broken_u = sorted({int(i) for i in broken if not by_ind[int(i)].time_reversal})
    kept_u = sorted({int(i) for i in subgroup if not by_ind[int(i)].time_reversal})
    primed = [partners[i] for i in broken_u if i in partners]
    every = sorted(set(broken_u) | set(primed))
    kept = sorted(set(kept_u) | {partners[i] for i in kept_u if i in partners})
    rows = tuple(verdict.rows)
    reasons = verdict.reasons or {}
    read_together = tuple(verdict.joint) + tuple(verdict.by_sectors)
    why_of = {i: str(reasons.get(i) or _joint_reason(read_together, i) or (
        "the declared content is carried by this operation on its own, but no group "
        "of operations containing it and those kept carries every declared multiplet "
        "in its declared irrep at once")) for i in broken_u}
    for i in broken_u:
        if i in partners:
            why_of[partners[i]] = (
                f"the primed copy of #{i} {names[i]}: a site character is a character "
                "of the unitary site group, so the two stand or fall together")

    # What the failing declaration measured, in site-irrep names, read off the
    # site tables and the one-dimensional characters of the group
    if site_tables is None:
        try:
            site_tables = _site_tables(group, assignment, set(unitary),
                                       set(standing) - set(unitary),
                                       range(len(assignment.site_orbitals)))
        except Exception:                        # noqa: BLE001 - the names are a courtesy
            site_tables = {}
    if admissible is None:
        admissible = _characters_of_subgroup(group, unitary)
    named = _mismatch_named(decls, rows, admissible, site_tables,
                            ref_orb=int(verdict.reference_orbital))

    def _listed(inds):
        return ", ".join(f"#{i} {names[i]}" for i in inds)

    read = ""
    if readings:

        read = ("\n  several subgroups the declaration is covariant on are maximal, and none "
                "contains the others: "
                + "; ".join("{" + ", ".join(f"#{i}" for i in r) + "}" for r in readings)
                + ". The one named above is the first by the tie rule of "
                f"`realized_group`: {_tie_rule_words(group)}. It is first since "
                f"{_tie_reason(group, readings)}.")
    why = "\n".join(f"    #{i} {names[i]}: {why_of[i]}" for i in every)
    table = "\n".join(
        f"    #{r.ind:<4d} {r.name:<14s} orbitals {tuple(r.orbitals)}"
        f"   measured {_fmt_char(r.measured)}"
        f"   declared {_fmt_char(r.declared)}"
        for r in rows if int(r.ind) in set(broken_u))
    return Refusal(
        "declaration_breaks_group",
        f"the declaration is not covariant under {len(broken_u)} of the "
        f"{len(unitary)} unitary operations of {group.name} #{group.number_str}, "
        f"{_counted_as(group.spinor)}: "
        + (_listed(broken_u) or "(none)")
        + (f", nor under their primed partners {_listed(primed)}" if primed else "")
        + f". It is covariant on the {len(kept)} operations "
        + _listed(kept)
        + f", the subgroup of {group.name} that `realized_group` descends to. {not_lowered}"
        + read
        + (f"\n{named['verdict']}" if named else "")
        + (f"\n  why each is broken:\n{why}" if why else "")
        + (f"\n  measured and declared characters:\n{table}" if table else "")
        + "\n  frame: " + "; ".join(f"#{i} = {frame[i]}" for i in standing),
        value=float(len(every)), threshold=0.0,
        context=dict({
            "group": group, "number_str": str(group.number_str),
            "subgroup": tuple(kept),
            "broken": tuple({"ind": int(i), "name": names[i], "note": why_of[i],
                             "rows": _returned_rows(r for r in rows
                                                    if int(r.ind) == int(i))}
                            for i in every),
            "tied": tuple(tuple(int(i) for i in r) for r in readings),
            "checks": _returned_rows(rows),
            "declarations": tuple({"orbitals": tuple(d["orbitals"]),
                                   "declared_as": d["spec"],
                                   "declared_irrep": d["label"]} for d in decls),
            "frame": dict(frame)}, **(named.get("named", {}) if named else {})))


def _joint_reason(joint, ind):
    """The sentence saying why a declaration read together breaks one operation.

    ``joint`` holds the declarations read over several multiplets at once
    (:class:`_ReadTogether`), and the first measured at the operation of
    index ``ind`` is named.  Returns ``None`` where none was measured there.
    """
    for j in joint:
        if not any(int(ind) in row for row in j.theta.values()):
            continue
        d = j.decl
        said = d.get("label") or repr(d.get("spec"))
        orbs = tuple(int(o) for o in d["orbitals"])
        if d.get("partial"):
            return (f"orbitals {orbs} are declared {said}, part of what their site holds, "
                    "and no labeling the hoppings leave open puts it inside the site's "
                    "content on a group holding this operation and those kept")
        return (f"orbitals {orbs} are declared {said} together, and no labeling the "
                "hoppings leave open carries that content on a group holding this "
                "operation and those kept")
    return None


def _declared_key(decls):
    """A comparable key for the declarations ``decls``.

    The key holds each declaration's orbitals, its characters rounded to
    twelve decimals, and whether it is written in double-valued site irreps.
    A verdict handed down from :func:`..groups.realized_group` is reused only
    where its key equals this one.
    """
    return tuple(sorted(
        (tuple(int(o) for o in d["orbitals"]),
         tuple(sorted((int(k), round(complex(v).real, 12), round(complex(v).imag, 12))
                      for k, v in d["constraints"].items())),
         bool(d.get("corep")))
        for d in decls))


def _multiplet_scale(group, layout, positions, cperm, classes, ref, blocks, *,
                     reversal=None, reversal_image=None, reach):
    r"""One multiplet's blocks round its orbit, settled into a representation.

    This is the scale a coupled part reads its declared characters against
    where it holds no one-dimensional multiplet, or where the lift of its
    one-dimensional multiplet is no representation (:func:`_lifted_scale`).
    The Hamiltonian fixes each operation's block on the orbit of ``ref`` only
    up to the phase of the solution space it was read from.  The blocks are
    settled as the build settles its matrices
    (:func:`.._matrices._settle_unit`): each generator in its canonical norm,
    the time reversal anchored, the rest propagated along the generators'
    words.  What is left is fixed up to one one-dimensional character, which
    the part chooses.

    Parameters
    ----------
    group : ..groups.GroupData
        The group the operations belong to.
    layout : SiteMultiplets
        The multiplets of each site.
    positions : iterable of int
        The positions in ``group`` of the unitary operations read.
    cperm : mapping
        Per position, where the operation carries each multiplet.
    classes : ndarray of int
        The multiplet orbit of each multiplet.
    ref : int
        The multiplet whose orbit is settled.
    blocks : mapping
        Each operation's block on each multiplet, ``{ind: {multiplet: block}}``.
    reversal : int or None, default None
        The position of the time reversal that anchors the phases; ``None``
        where the group keeps none.
    reversal_image : sequence of int or None, default None
        Where that time reversal carries each multiplet.
    reach : float
        The composition defect up to which noisy blocks are projected onto the
        nearest exact representation.

    Returns
    -------
    dict or None
        ``{position: block}``, each block from ``ref`` to its image under the
        operation.  ``None`` where the operations are no group, a block is
        missing, or the blocks settle into no representation.
    """
    syms = group._symmetries
    prod, _tdiff, spf = group._get_product_table(get_diff=True)
    ident = _identity_position(group)
    u_pos = sorted({int(p) for p in positions})
    if ident not in u_pos or set(_closure(u_pos, prod, ident)) != set(u_pos):
        return None
    gens = _minimal_generators(u_pos, prod, ident)
    if gens is None:
        return None
    words, parent = _bfs_words(gens, prod, ident, u_pos)
    if set(words) != set(u_pos):
        return None
    ref = int(ref)
    members = [c for c in range(int(layout._nchan)) if int(classes[c]) == int(classes[ref])]
    states = [int(a) for c in members for a in layout._chan_states[c]]
    local = {a: i for i, a in enumerate(states)}
    n = len(states)

    def _local(c):
        return np.asarray([local[int(a)] for a in layout._chan_states[int(c)]], dtype=int)

    def _raw(p, image):
        """The matrix of the operation at position ``p`` on the orbit of ``ref``.

        ``image`` is where the operation carries each multiplet.  The matrix
        is assembled from ``blocks``; ``None`` where one is missing or has the
        wrong shape.
        """
        got = blocks.get(int(syms[p].ind), {})
        M = np.zeros((n, n), dtype=complex)
        for c in members:
            S = got.get(int(c))
            if S is None or int(image[c]) not in members:
                return None
            rows, cols = _local(image[c]), _local(c)
            if np.asarray(S).shape != (len(rows), len(cols)):
                return None
            M[np.ix_(rows, cols)] = S
        return M

    Dgen = [_raw(gp, cperm[gp]) for gp in gens]
    if any(M is None for M in Dgen):
        return None
    T_pos, au_of, DT = None, {}, None
    if reversal is not None:
        T_pos = int(reversal)
        DT = _raw(T_pos, reversal_image)
        if DT is None:
            return None
        au_of = {T_pos: ident}
    tree = {"gens": gens, "words": words, "parent": parent, "spf": spf, "ident": ident,
            "nstate": n, "T_pos": T_pos, "au_of": au_of}
    kept = list(u_pos) + ([T_pos] if T_pos is not None else [])
    cols = tuple(int(i) for i in _local(ref))

    def _anchor_of(p, anchor_T):
        image = reversal_image if p == T_pos else cperm[p]
        return RepAnchor(antiunitary=anchor_T,
                         rows=tuple(int(i) for i in _local(image[ref])), cols=cols)

    bad, D, _norm, _anchor_T, _DT_n, _defect, _roots = _settle_unit(
        _propagated(Dgen, DT, **tree), group=group, tree=tree, kept=kept, u_pos=u_pos,
        prod=prod, anchor_of=_anchor_of, reach=reach)
    # the build's own product-table gate: what the table settles is a
    # representation, and a set of phases that is not one is no scale
    if bad is not None or _product_table_residual(D, u_pos, syms, prod, spf) > 1e-9:
        return None
    return {p: D[p][np.ix_(_local(cperm[p][ref]), _local(ref))] for p in u_pos}


def _lifted_scale(group, layout, positions, cperm, classes, ref, blocks, *, bases,
                  intertwiners, reversal=None, reversal_image=None, reach):
    r"""The scale of a one-dimensional multiplet whose lift is no representation.

    A coupled part holding a one-dimensional multiplet reads its declared
    characters against that multiplet's lift, carried round its Wyckoff orbit
    with the signs the hoppings give it
    (:func:`.._multiplets._transported_lifts`).  Where the multiplet's
    character on its site group is the restriction of no one-dimensional
    character of the group, those blocks are no representation, and they are
    settled into one by :func:`_multiplet_scale`.

    ``ref`` is the multiplet, and ``blocks`` the multiplets' blocks handed
    down, ``{ind: {multiplet: block}}``.  ``bases`` holds the measured solution
    spaces and ``intertwiners`` one member of each, by index; they carry the
    lift round the orbit where ``blocks`` holds no block.  ``group``,
    ``layout``, ``positions``, ``cperm``, ``classes``, ``reversal``,
    ``reversal_image`` and ``reach`` are as for :func:`_multiplet_scale`.
    Returns the settled blocks, ``{position: block}``.  Returns ``None`` where
    the lift is a representation of the operations, where the blocks settle
    into none, or where ``reversal`` is given without ``reversal_image``.
    """
    syms = group._symmetries
    prod, _tdiff, spf = group._get_product_table(get_diff=True)
    ident = _identity_position(group)
    u_pos = sorted({int(p) for p in positions})
    if reversal is not None and reversal_image is None:
        return None
    images = {int(p): cperm[p] for p in u_pos}
    if reversal is not None:
        images[int(reversal)] = reversal_image
    members = [int(c) for c in range(int(layout._nchan))
               if int(classes[c]) == int(classes[int(ref)])]
    # the orbit's blocks as the build assembles them: the ones handed down, the
    # signs the hoppings give the orbit relative to ``ref``, or the lift alone
    carried = None
    if len(members) > 1 and not all(int(c) in blocks.get(int(syms[p].ind), {})
                                    for p in images for c in members):
        carried = _transported_lifts(syms, list(images), images, layout, members, int(ref),
                                     bases, intertwiners, prod, ident)
    lifted = {}
    for p in images:
        ind = int(syms[p].ind)
        got = blocks.get(ind, {})
        lift = _lift_of(syms[p], layout._spinful)
        lifted[ind] = {c: got[c] if c in got else (carried or {}).get(p, {}).get(c, lift)
                       for c in members}
    states = [int(a) for c in members for a in layout._chan_states[c]]
    local = {a: i for i, a in enumerate(states)}
    raw = {}
    for p in u_pos:
        M = np.zeros((len(states), len(states)), dtype=complex)
        for c in members:
            image = int(images[p][c])
            if image not in members:      # pragma: no cover - a class is an orbit
                return None
            rows = [local[int(a)] for a in layout._chan_states[image]]
            cols = [local[int(a)] for a in layout._chan_states[c]]
            M[np.ix_(rows, cols)] = lifted[int(syms[p].ind)][c]
        raw[p] = M
    if _product_table_residual(raw, u_pos, syms, prod, spf) <= noise_floor(
            ZERO, getattr(layout, "_tol", None)):
        return None                       # the lift is a representation
    return _multiplet_scale(group, layout, u_pos, cperm, classes, int(ref), lifted,
                            reversal=reversal, reversal_image=reversal_image, reach=reach)


@dataclass(frozen=True, eq=False)
class _Anchor:
    """The multiplets' permutations and orbits, the anchor, and their blocks.

    Attributes
    ----------
    classes : ndarray of int
        The multiplet orbit of each multiplet.
    cperm : dict
        Where each operation carries each multiplet, by position.
    layout : SiteMultiplets
        The multiplets of each site (:func:`_verdicts_and_choices`).
    measured_of : dict
        Each operation's internal block on each multiplet, by index.
    multiplicity : tuple
        The orbits of sectors the blocks were built for, each with its
        candidates.
    nchan : int
        The number of multiplets.
    nclass : int
        The number of multiplet orbits.
    positions : list of int
        The positions of the unitary operations read.
    ref : int
        The anchor, orbital 0's multiplet.
    ref_cls : int
        The anchor's orbit.
    reps : dict
        The multiplet each orbit is read at, the anchor for its own.
    site_layout : SiteMultiplets
        Each site's orbitals as one multiplet.
    syms : list
        The group's operations, in the group's order.
    """

    classes: object
    cperm: object
    layout: object
    measured_of: object
    multiplicity: object
    nchan: object
    nclass: object
    positions: object
    ref: object
    ref_cls: object
    reps: object
    site_layout: object
    syms: object


@dataclass(frozen=True, eq=False)
class _OperationsRead:
    """What each operation's solution space says of the multiplets.

    Attributes
    ----------
    bases : dict
        Each operation's solution space, by index.
    block_tol : float or None
        The tolerance the multiplets were measured at.
    broken : dict
        The operations the declared multiplets break, by index.
    carried : dict
        Each orbit's block across each solution space, ``{ind: {class: array}}``.
    chan_states : tuple
        The states of each multiplet.
    chosen : dict
        The class scalar the hoppings pin, ``{(ind, class): value}``.
    free : set
        The ``(ind, class)`` pairs the hoppings leave open.
    in_tol : float
        How far outside the blocks a member may sit and count as inside.
    lattice : ndarray
        The group's lattice vectors.
    rows : list
        The measured-versus-declared table, filled by :func:`_characters_read`.
    seen : dict
        The classes measured at each operation, ``{ind: set}``.
    spaces : dict
        The in-block solution space of each operation, by position, which
        :func:`_characters_read` reads.
    spread_tol : float
        The spread within which a ratio counts as pinned.
    su2 : bool
        Whether the states carry the spin-1/2 lift.
    want_of : dict
        The declaration of each declared multiplet.
    why : dict
        One sentence per broken operation, by index.
    zero : float
        The size below which a block or a ratio vanishes.
    """

    bases: object
    block_tol: object
    broken: object
    carried: object
    chan_states: object
    chosen: object
    free: object
    in_tol: object
    lattice: object
    rows: object
    seen: object
    spaces: object
    spread_tol: object
    su2: object
    want_of: object
    why: object
    zero: object


@dataclass(frozen=True, eq=False)
class _PartsRead:
    """The coupled parts of the crystal and the part each declaration reads in.

    Attributes
    ----------
    free_classes : frozenset of int
        The orbits free at every operation that was measured.
    induced_classes : list of frozenset
        The orbits of each orbit of sectors built by induction.
    joint_decls : list
        The declarations read over several parts, or on part of a site.
    on_induced : list
        The declarations on an orbit of sectors built by induction.
    part_of : dict
        The coupled part of each orbit.
    part_of_decl : dict
        The part each declaration is read in, ``None`` where it is read with
        others.
    parts : list of frozenset
        The coupled parts as sets of orbits, the anchor's first.
    spoken : set of int
        The parts some declaration is read in.
    within : dict
        The ratio an orbit of a part free of the anchor holds to that part's
        first, ``{(ind, class): value}``.
    """

    free_classes: object
    induced_classes: object
    joint_decls: object
    on_induced: object
    part_of: object
    part_of_decl: object
    parts: object
    spoken: object
    within: object


@dataclass(frozen=True, eq=False)
class _Scales:
    """The scale each part's characters are read against.

    Attributes
    ----------
    reversal_image : list or None
        Where time reversal carries each multiplet; ``None`` where it is not
        read.
    roots : dict
        Per part, whether its character may take any unit phase.
    scales : dict
        Per part, the multiplet of its scale and the scale's blocks.
    """

    reversal_image: object
    roots: object
    scales: object


@dataclass(frozen=True, eq=False)
class _CharactersRead:
    """The declarations read over several parts, multiplet by multiplet.

    Attributes
    ----------
    joint_theta : dict
        Per declaration, by its ``id``, each multiplet's character at each
        operation, ``{id: {multiplet: {ind: value}}}``.
    """

    joint_theta: object


def _verdicts_and_choices(model, group, assignment, spin, validated, layout, decls,
                          positions, *, classes=None, ref_orb=0,
                          blocks=None, space_tol=IN_SPACE, tol=CHARACTER,
                          max_candidates=2 ** 20, reversal=None, sectors=None):
    r"""One verdict per operation, and the scalars it pins, off the solution spaces.

    This is the one place a solution space is read for what it says about the
    orbitals.  The descent by declaration, the representation's choices and the
    table of what each orbital carries at its site (``rep._characters``) are
    written from what is returned.  Each unitary operation is read inside the
    site blocks its own permutation of the sites allows, and three things are
    read there.

    * The multiplet verdict.  Where no member of the space is block-diagonal
      on the multiplets and nonsingular there, the hoppings mix multiplets.
      Where ``decls`` is not empty and the multiplets themselves were declared
      (``layout.determined_by == "declaration"``), the operation is then
      broken by the declaration.
    * The scalar of each multiplet orbit relative to the anchor, orbital 0's
      multiplet.  The hoppings pin it where every member of the space carries
      the same ratio, and leave it free where the ratio varies.
    * The character of each declared multiplet the operation fixes, read
      against a scale in its own coupled part (:func:`_lifted_scale`,
      :func:`_multiplet_scale`).  Two multiplets of one part that need
      different scalars break the operation.  So does a scalar other than
      ``±1`` where the group keeps an antiunitary operation, or other than a
      unit phase where it keeps none.  A declaration written in double-valued
      site irreps is not read at an operation whose spin-1/2 lift has zero
      trace.

    A declaration over several coupled parts, or on part of a site, is read
    multiplet by multiplet, as one requirement on those parts together.
    Antiunitary operations are not read.  Whoever reads the verdict breaks the
    primed copy of a broken operation with it.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : ..groups.GroupData
        The group whose operations are read.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites in ``group``.
    spin : ..spin.SpinBasis or None
        The declared spin basis; ``None`` where none is declared.
    validated : ..groups.ValidatedGroup
        The operations measured, with their solution spaces.
    layout : SiteMultiplets
        The multiplets of each site.  The thresholds on the blocks, the
        scalars and the ratios read the tolerance the multiplets were measured
        at where that is larger than their exact defaults
        (:func:`.._tolerance.noise_floor`).
    decls : sequence of dict
        The declarations, as :func:`_normalize_declaration` returns them.
    positions : iterable of int
        The positions in ``group`` of the operations read; the antiunitary
        ones are dropped.
    classes : ndarray of int or None, default None
        The multiplet orbit of each multiplet; ``None`` reads them off the
        multiplets' permutations.
    ref_orb : int, default 0
        The reference orbital, which the lowest-angular-momentum rule reads
        first.  It is recorded on the verdict.
    blocks : _MultipletBlocks or None, default None
        The multiplets' blocks.  ``None`` builds them here where the
        multiplets were measured (:func:`.._multiplets._multiplet_blocks`).
    space_tol : float, default IN_SPACE
        How far a member may sit outside the blocks and still count as in
        them (:func:`.._matrices._in_blocks`), dimensionless.
    tol : float, default CHARACTER
        How far a character read off the spaces may sit from a value and
        still be read as it; pass the declared tolerance where that is larger.
    max_candidates : int, default 2**20
        The most representations tried for an orbit of sectors where the
        blocks are built here.  It also bounds the representations the
        sectors' search on the declaration weighs at once, past which
        :func:`_subgroup_read` refuses ``rep_search_too_large``.
    reversal : int or None, default None
        The position of the time reversal that anchors the scales' phases;
        ``None`` where the group keeps none.
    sectors : .._sectors.SectorReading or None, default None
        The sectors read earlier on the same model by
        :func:`..groups.realized_group`; ``None`` reads them here.

    Returns
    -------
    _Verdict or Refusal
        The verdict, or the refusal the block enumeration made.
    """
    anc = _anchor_read(model, group, assignment, spin, validated, layout, positions,
                       classes=classes, blocks=blocks, max_candidates=max_candidates)
    if not isinstance(anc, _Anchor):
        return anc
    ops = _operations_read(model, group, assignment, validated, layout, decls, anc,
                           space_tol=space_tol)
    prt = _parts_read(decls, anc, ops)
    scl = _scales_read(model, group, assignment, spin, validated, layout, anc, ops, prt,
                       reversal=reversal)
    chv = _characters_read(model, anc, ops, prt, scl, tol=tol)
    return _verdict_made(model, group, assignment, spin, validated, layout, decls, anc, ops, prt,
                         scl, chv, ref_orb=ref_orb, space_tol=space_tol, tol=tol,
                         max_candidates=max_candidates, reversal=reversal, sectors=sectors)


def _anchor_read(model, group, assignment, spin, validated, layout, positions, *, classes, blocks,
                 max_candidates):
    """The multiplets' permutations and orbits, the anchor, and their blocks.

    ``model``, ``group``, ``assignment``, ``spin``, ``validated``, ``layout``,
    ``positions``, ``classes``, ``blocks`` and ``max_candidates`` are as for
    :func:`_verdicts_and_choices`.  The anchor is orbital 0's multiplet.  The
    blocks are built here where the multiplets were measured and none are
    handed in.  Returns the :class:`_Anchor`, or the refusal that the block
    enumeration (:func:`.._multiplets._multiplet_blocks`) made.
    """
    syms = group._symmetries
    positions = [p for p in positions if not bool(syms[p].time_reversal)]
    cperm = _channel_perms(model, assignment, syms, positions, layout, spin)
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    site_layout = _multiplets_of(model, assignment, spin,
                             tuple((tuple(orbs),) for orbs in site_orbs))
    nchan = int(layout._nchan)
    chan_of = np.asarray(layout._chan_of, dtype=int)
    # The anchor every scalar is read against is orbital 0's multiplet,
    # whichever orbital the lowest-angular-momentum rule reads the crystal
    # from, so that every build of this group reads the same verdict
    ref = int(chan_of[0])
    if classes is None:
        classes, nclass = _channel_classes(cperm, positions, nchan)
    else:
        classes = np.asarray(classes, dtype=int)
        nclass = int(classes.max()) + 1 if classes.size else 0
    ref_cls = int(classes[ref]) if nclass else 0
    reps = {}
    for cls in range(nclass):
        members = np.flatnonzero(classes == cls)
        reps[cls] = ref if cls == ref_cls else int(members[0])

    measured_of = dict(blocks or {})
    multiplicity = tuple(getattr(blocks, "multiplicity", ()) or ())
    if layout._measured and not measured_of:
        got = _multiplet_blocks(model, group, assignment, spin, validated, layout,
                                [p for p, s in enumerate(syms)
                                 if int(s.ind) in set(validated._indices)],
                                ref_chan=ref, max_candidates=int(max_candidates))
        if isinstance(got, Refusal):
            return got
        measured_of = got
        multiplicity = tuple(got.multiplicity)
    return _Anchor(classes=classes, cperm=cperm, layout=layout, measured_of=measured_of,
                   multiplicity=multiplicity, nchan=nchan, nclass=nclass, positions=positions,
                   ref=ref, ref_cls=ref_cls, reps=reps, site_layout=site_layout, syms=syms)


def _operations_read(model, group, assignment, validated, layout, decls, anc, *, space_tol):
    """The multiplet verdict and the relative scalars, operation by operation.

    ``model``, ``group``, ``assignment``, ``validated``, ``layout``, ``decls``
    and ``space_tol`` are as for :func:`_verdicts_and_choices`, and ``anc`` is
    what :func:`_anchor_read` returns.  Each operation at ``anc.positions`` is
    read inside the site blocks its own permutation of the sites allows.  Where
    no member of the space inside those blocks is block-diagonal on the
    multiplets and nonsingular, the operation is broken by the declaration if
    ``decls`` is not empty and the multiplets were declared
    (``layout.determined_by == "declaration"``), with a sentence naming the
    multiplets the hoppings mix; otherwise it is not read.  Where a member is,
    the scalar of each multiplet orbit relative to the anchor is pinned where
    every member carries the same ratio, and free where the ratio varies.  An
    operation with no solution space held, one whose solution space is too
    large to read (``_READOFF_MAX_BYTES``), or one with no member inside its
    site blocks is not read.  Returns the :class:`_OperationsRead`, its
    ``rows`` empty for :func:`_characters_read` to fill.
    """
    cperm = anc.cperm
    measured_of = anc.measured_of
    nchan = anc.nchan
    nclass = anc.nclass
    positions = anc.positions
    ref = anc.ref
    reps = anc.reps
    site_layout = anc.site_layout
    syms = anc.syms

    lattice = np.asarray(group._lattice, dtype=float)
    chan_states, su2 = layout._chan_states, layout._spinful
    want_of = {int(c): d for d in decls for c in d["channels"]}
    bases = getattr(validated, "_bases", None) or {}
    # the multiplets' blocks are tested at the tolerance they were measured at
    block_tol = getattr(layout, "_tol", None)
    in_tol = noise_floor(space_tol, block_tol)
    zero, spread_tol = noise_floor(ZERO, block_tol), noise_floor(SPREAD, block_tol)
    rows, broken, why = [], {}, {}
    chosen, free, seen = {}, set(), {}
    # the in-block solution spaces the character verdict reads, once the
    # coupled parts they are read in are known, and what each orbit's block
    # is across each space
    spaces, carried = {}, {}
    for p in positions:
        ind = int(syms[p].ind)
        basis = bases.get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            member = validated._intertwiners.get(ind)
            if member is None:
                continue
            basis = np.asarray(member)[None, :, :]
        basis = np.asarray(basis)
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            continue                      # the read-off's own ceiling: not measured
        site_image = _channel_permutation(assignment, p, model, site_layout)
        space = _in_blocks(basis, _ansatz_blocks(site_layout._chan_states, site_image, 1),
                           tol=in_tol)
        if not len(space):
            continue                      # the operation's own tensor structure
        image = cperm[p]
        blocks_d = _ansatz_blocks(chan_states, image, 1)
        V = _in_blocks(space, blocks_d, tol=in_tol)
        ok, unfilled = (False, list(range(len(blocks_d)))) if not len(V) else \
            _filled_check(V, blocks_d, tol=block_tol)
        if not ok:
            if not decls or layout.determined_by != "declaration":
                # nothing stated these multiplets: the Hamiltonian measured them,
                # so multiplets the space cannot carry are the operation's own
                # tensor structure and `_ansatz_refuted` owns that verdict, not
                # this one
                continue
            touched = sorted({int(c) for c in unfilled}
                             | {int(image[int(c)]) for c in unfilled})
            named = [c for c in touched if c in want_of] or touched
            broken[ind] = None
            why[ind] = (
                "the measured solution space holds no nonsingular member that is "
                "block-diagonal on the declared multiplets: the hoppings mix "
                + ", ".join(f"the multiplet of orbitals {tuple(layout._chan_orbs[c])}"
                            for c in named)
                + " with the rest of the site")
            continue
        lift = _lift_of(syms[p], su2)
        blocks_here = measured_of.get(ind, {})
        lift_of = [blocks_here.get(int(c),
                                   None if layout._chan_kind[int(c)] == "measured" else lift)
                   for c in range(nchan)]

        # One scalar per multiplet orbit, relative to the anchor
        S_ref = lift_of[ref]
        if S_ref is not None:
            rrows = np.asarray(chan_states[int(image[ref])], dtype=int)
            rcols = np.asarray(chan_states[ref], dtype=int)
            nu_c = np.asarray([_proportionality(V[j][np.ix_(rrows, rcols)], S_ref)[0]
                               for j in range(len(V))])
            base = float(np.abs(nu_c).max()) if nu_c.size else 0.0
            if base > zero:
                for cls in range(nclass):
                    c0 = reps[cls]
                    S0 = lift_of[c0]
                    if S0 is None:
                        continue
                    crow = np.asarray(chan_states[int(image[c0])], dtype=int)
                    ccol = np.asarray(chan_states[c0], dtype=int)
                    t = np.asarray([_proportionality(V[j][np.ix_(crow, ccol)], S0)[0]
                                    for j in range(len(V))])
                    sig = complex(np.vdot(nu_c, t) / np.vdot(nu_c, nu_c))
                    spread = float(np.abs(t - sig * nu_c).max())
                    seen.setdefault(ind, set()).add(int(cls))
                    carried.setdefault(ind, {})[int(cls)] = t
                    if spread > spread_tol * max(float(np.abs(t).max()), base, 1.0):
                        free.add((ind, int(cls)))
                    else:
                        chosen[(ind, int(cls))] = sig
        if want_of:
            spaces[p] = (V, image, lift, lift_of)
    return _OperationsRead(bases=bases, block_tol=block_tol, broken=broken, carried=carried,
                           chan_states=chan_states, chosen=chosen, free=free, in_tol=in_tol,
                           lattice=lattice, rows=rows, seen=seen, spaces=spaces,
                           spread_tol=spread_tol, su2=su2, want_of=want_of, why=why, zero=zero)


def _parts_read(decls, anc, ops):
    """The coupled parts of the crystal and the part each declaration reads in.

    ``decls`` is as for :func:`_verdicts_and_choices`, and ``anc`` and ``ops``
    are what :func:`_anchor_read` and :func:`_operations_read` return.  A
    coupled part is the multiplet orbits the hoppings connect, directly or
    through others: the anchor's part, and the orbits free of the anchor
    joined where their blocks keep one ratio across a solution space.  An
    orbit of sectors built by induction is one part, and the anchor's where it
    holds the anchor's multiplet; its orbits are then set to ``1`` in
    ``ops.chosen`` and taken out of ``ops.free``.  A declaration on the
    multiplets of one part, stating all its site holds, is read in that part.
    One over several parts, or on part of a site, is read multiplet by
    multiplet, as one requirement on the parts it spans (``joint_decls``).
    One on an orbit built by induction is left to the sectors.  Returns the
    :class:`_PartsRead`.
    """
    classes = anc.classes
    multiplicity = anc.multiplicity
    nclass = anc.nclass
    ref_cls = anc.ref_cls
    carried = ops.carried
    chosen = ops.chosen
    free = ops.free
    seen = ops.seen
    spread_tol = ops.spread_tol
    zero = ops.zero

    # An orbit is free where nothing measured it: free at every operation whose
    # solution space spoke about it.  That is a decoupled multiplet.  This
    # Hamiltonian does not fix what it carries relative to the anchor, on this
    # site or any other, and the labeling settles it
    # (:func:`.._labeling._labeling`), as a declaration on it states it where
    # there is one.
    free_classes = frozenset(
        cls for cls in range(nclass)
        if any(cls in got for got in seen.values())
        and all((ind, cls) in free for ind, got in seen.items() if cls in got))
    # Identical copies, and uncoupled orbitals a symmetry exchanges that the
    # group's characters do not carry, are one part of the crystal each: an
    # orbit of sectors the hoppings couple to nothing else.  Its multiplets are
    # assembled with the blocks of one representation of it
    # (:func:`.._multiplets._induced_blocks`), and the labeling chooses which
    # among all of them.  So no scalar is read between its multiplets, which
    # carry those blocks as they stand, since a scalar read at one multiplet of
    # an orbit lying on several sectors says nothing of the others.  The orbit
    # is pinned to the anchor where it holds the anchor's multiplet, and free
    # of it otherwise.
    induced_classes = []
    for rec in multiplicity:
        if rec.induced is None or not rec.n_admitted:
            continue
        own = frozenset(int(classes[int(c)]) for c in rec.multiplets)
        induced_classes.append(own)
        if ref_cls in own:
            free_classes = free_classes - own
            for ind in seen:
                for cls in own:
                    free.discard((int(ind), int(cls)))
                    chosen[(int(ind), int(cls))] = 1.0 + 0.0j
        else:
            free_classes = free_classes | own

    # The coupled parts.  The hoppings fix every multiplet's block up to one
    # one-dimensional character per part of the crystal they connect: the
    # anchor's part, with every orbit they pin to it, and the orbits they
    # leave free of it, which are one part where they pin one another.  Two
    # orbits pin one another where their blocks keep one ratio across a
    # solution space.  A declaration is a constraint on its own part's
    # character, read against a scale inside that part, so a part the hoppings
    # never reach measures nothing about it.
    joined = {int(cls): int(cls) for cls in free_classes}

    for got in carried.values():
        both = [(int(cls), t) for cls, t in sorted(got.items())
                if int(cls) in joined and float(np.abs(t).max()) > zero]
        for (a, ta), (b, tb) in itertools.combinations(both, 2):
            sig = complex(np.vdot(ta, tb) / np.vdot(ta, ta))
            if float(np.abs(tb - sig * ta).max()) <= spread_tol * max(
                    float(np.abs(tb).max()), float(np.abs(ta).max()), 1.0):
                joined[max(_joined(joined, a), _joined(joined, b))] = min(
                    _joined(joined, a), _joined(joined, b))
    for own in induced_classes:
        if ref_cls in own:
            continue
        first = min(own)
        for cls in own:
            a, b = _joined(joined, int(first)), _joined(joined, int(cls))
            if a != b:
                joined[max(a, b)] = min(a, b)
    apart = {}
    for cls in sorted(joined):
        apart.setdefault(_joined(joined, cls), set()).add(int(cls))
    parts = ([frozenset(cls for cls in range(nclass) if cls not in free_classes)]
             + [frozenset(apart[k]) for k in sorted(apart)])
    part_of = {int(cls): i for i, part in enumerate(parts) for cls in part}
    # ... and the ratio every member of a space holds between an orbit of such
    # a part and the part's first, where the operation's space speaks of both;
    # one between the multiplets of an orbit built by induction, which carry
    # their blocks as they stand
    within = {}
    for ind, got in carried.items():
        for part in parts[1:]:
            lead_cls = min(part)
            if any(part == own for own in induced_classes):
                for cls in part:
                    if int(cls) != int(lead_cls):
                        within[(int(ind), int(cls))] = 1.0 + 0.0j
                continue
            t0 = got.get(int(lead_cls))
            if t0 is None or float(np.abs(t0).max()) <= zero:
                continue
            for cls in part:
                if int(cls) != int(lead_cls) and int(cls) in got:
                    within[(int(ind), int(cls))] = complex(
                        np.vdot(t0, got[int(cls)]) / np.vdot(t0, t0))
    # Every declaration is read once, in one of two ways.  One on the
    # multiplets of one coupled part is a requirement on that part's
    # one-dimensional character, operation by operation.  One spread over
    # several parts, or one stating part of what its site holds, is one
    # requirement on what those parts carry together: read here per multiplet,
    # at every operation fixing its site, and met where one labeling of each
    # part it spans carries it (:func:`_covariant_subgroup`).
    part_of_decl, joint_decls, on_induced = {}, [], []
    induced = frozenset(cls for own in induced_classes for cls in own)
    for d in decls:
        held = {part_of.get(int(classes[int(c)])) for c in d["channels"]}
        if any(int(classes[int(c)]) in induced for c in d["channels"]):
            # a declaration on an orbit built by induction: what the orbit
            # carries is no measurement, so nothing is read here, and the
            # sectors' own reading of every representation of it is the verdict
            part_of_decl[id(d)] = None
            on_induced.append(d)
            continue
        if len(held) == 1 and not d.get("partial"):
            part_of_decl[id(d)] = held.pop()
        else:
            part_of_decl[id(d)] = None
            if d["channels"]:
                joint_decls.append(d)
    spoken = ({i for i in part_of_decl.values() if i is not None}
              | {part_of.get(int(classes[int(c)])) for d in joint_decls for c in d["channels"]})
    spoken = {int(i) for i in spoken if i is not None}
    return _PartsRead(free_classes=free_classes, induced_classes=induced_classes,
                      joint_decls=joint_decls, on_induced=on_induced, part_of=part_of,
                      part_of_decl=part_of_decl, parts=parts, spoken=spoken, within=within)


def _scales_read(model, group, assignment, spin, validated, layout, anc, ops, prt, *, reversal):
    """The scale each part's characters are read against.

    ``model``, ``group``, ``assignment``, ``spin``, ``validated``, ``layout``
    and ``reversal`` are as for :func:`_verdicts_and_choices`, and ``anc``,
    ``ops`` and ``prt`` are what the three stages before it return.  A scale
    is sought for each part some declaration is read in.  A part holding a
    one-dimensional multiplet, whose block is the tables' own lift, reads
    against one such multiplet, the part's lead (:func:`_lead_of`) where the
    lead is one.  It reads against the blocks :func:`_lifted_scale` returns,
    and against the lift itself where that returns ``None``, which it does
    where the lift is a representation, where the blocks settle into none, or
    where time reversal's image of the multiplets cannot be read.  Any other
    part reads against its lead multiplet's blocks, settled into a
    representation (:func:`_multiplet_scale`), where the multiplets were
    measured; none of these has a scale where ``reversal`` is given and time
    reversal's image of the multiplets cannot be read.  A part's character may
    take any unit phase where ``reversal`` is ``None``, and a sign otherwise.
    Returns the :class:`_Scales`.  A part with no scale is absent from it, and
    nothing is read in that part.
    """
    classes = anc.classes
    cperm = anc.cperm
    measured_of = anc.measured_of
    nchan = anc.nchan
    positions = anc.positions
    ref = anc.ref
    syms = anc.syms
    bases = ops.bases
    in_tol = ops.in_tol
    part_of = prt.part_of
    spoken = prt.spoken

    # The scale each part's characters are read against.  A member of a
    # solution space is a representation matrix times one free complex number,
    # and the scalar a declared multiplet needs says something about the
    # physics only once that number is fixed.  It is fixed in the gauge the
    # representation returned is written in, the canonical induced form: one
    # multiplet's transport block carries that multiplet's own internal factor
    # and nothing more.  A one-dimensional multiplet's factor is pinned, the
    # tables' own lift, and where the part has one that is its scale.  Where
    # that lift, carried round its orbit, is no representation of the group,
    # the multiplet's own character is measured and its settled blocks are the
    # scale (:func:`_lifted_scale`).  Where those blocks settle into no
    # representation, or time reversal's image cannot be read, the lift itself
    # is the scale.  A wider multiplet's block is the model's own and carries
    # the member's free phase, so in a part of such multiplets the scale is one
    # of its multiplets' blocks, settled into a representation of the group.
    # With no antiunitary operation to anchor them, those blocks are fixed only
    # up to a unit phase per generator, which the part's one-dimensional
    # character can then carry.  Either way the part's character may take any
    # unit phase where the group keeps no antiunitary operation, and a sign
    # where it keeps one.  Where it keeps none, every one-dimensional
    # character, the complex ones included, is a labeling nothing measured
    # tells apart.
    reversal_image = None
    if reversal is not None:
        reversal_image = _channel_perms(model, assignment, syms, [int(reversal)], layout,
                                        spin)[int(reversal)]
    scales, roots, wider = {}, {}, []
    for i in sorted(spoken):
        lead = _lead_of([c for c in range(nchan) if part_of.get(int(classes[c])) == i], ref)
        tables = [c for c in range(nchan) if part_of.get(int(classes[c])) == i
                  and layout._chan_kind[c] == "tables"]
        if tables:
            scale_chan = lead if lead in tables else int(tables[0])
            # the one-dimensional multiplet's own character, settled, where its
            # lift is no representation; the lift itself where it is one, or
            # where nothing settles (``None``)
            scales[i] = (scale_chan, _lifted_scale(
                group, layout, positions, cperm, classes, scale_chan, measured_of,
                bases=bases, intertwiners=validated._intertwiners, reversal=reversal,
                reversal_image=reversal_image, reach=min(1e-2, 100.0 * in_tol)))
            roots[i] = reversal is None
        elif layout._measured and measured_of:
            wider.append((i, lead))
    if wider:
        for i, lead in wider:
            if reversal is not None and reversal_image is None:
                break
            scale_of = _multiplet_scale(group, layout, positions, cperm,
                                        classes, lead, measured_of, reversal=reversal,
                                        reversal_image=reversal_image,
                                        reach=min(1e-2, 100.0 * in_tol))
            if scale_of is not None:
                scales[i] = (lead, scale_of)
                roots[i] = reversal is None
    return _Scales(reversal_image=reversal_image, roots=roots, scales=scales)


def _characters_read(model, anc, ops, prt, scl, *, tol):
    """The character verdict for every declared multiplet the operations fix.

    ``model`` and ``tol`` are as for :func:`_verdicts_and_choices`, and
    ``anc``, ``ops``, ``prt`` and ``scl`` are what the four stages before it
    return.  At each operation, each declared multiplet the operation fixes is
    read against its part's scale (:func:`_theta`); a multiplet the hoppings
    leave free, or a part with no scale at that operation, gives nothing.  A
    declaration read in one part gives one :class:`_Row` per operation,
    appended to ``ops.rows``: the character measured, summed over its
    multiplets, the one declared, and the value the part's character has to
    take, ``None`` where the measured character is within ``tol`` of zero.
    Declared generator matrices are read as matrices (:func:`_matrix`), the
    value being the scalar that carries the measured matrix onto the declared
    one, ``None`` where none does within ``tol``.  A declaration in
    double-valued site irreps is not read where the operation's spin-1/2 lift
    has zero trace.  Returns the :class:`_CharactersRead`, the declarations
    over several parts, or on part of a site, read multiplet by multiplet.
    """
    classes = anc.classes
    positions = anc.positions
    syms = anc.syms
    chan_states = ops.chan_states
    lattice = ops.lattice
    rows = ops.rows
    spaces = ops.spaces
    want_of = ops.want_of
    zero = ops.zero
    joint_decls = prt.joint_decls
    part_of = prt.part_of
    part_of_decl = prt.part_of_decl
    scales = scl.scales

    # The character verdict, for what was declared

    nspin = int(getattr(model, "nspin", 1) or 1)

    pre, joint_theta = [], {}
    for p in positions:
        if p not in spaces:
            continue
        V, image, lift, lift_of = spaces[p]
        ind = int(syms[p].ind)
        name = _sitechars.operation_name(syms[p], lattice)
        nus = {}
        for i, (scale_chan, scale_of) in scales.items():
            if scale_of is None:
                S_norm = lift_of[scale_chan]
                if S_norm is None:        # pragma: no cover - a guard, not a path
                    continue
            elif p in scale_of:
                S_norm = scale_of[p]
            else:
                continue                  # nothing to fix the scale against
            nrows = np.asarray(chan_states[int(image[scale_chan])], dtype=int)
            ncols = np.asarray(chan_states[scale_chan], dtype=int)
            nu = np.asarray([_proportionality(V[j][np.ix_(nrows, ncols)], S_norm)[0]
                             for j in range(len(V))])
            floor = float(np.abs(nu).max()) if nu.size else 0.0
            if floor > zero:
                nus[i] = (nu, floor)
        silent = abs(complex(np.trace(lift))) <= zero
        # A declaration over several parts, or on part of a site: per
        # multiplet, what it carries against its own part's scale, at every
        # operation that fixes all of its multiplets
        for d in joint_decls:
            if ind not in d["constraints"] or (d.get("corep") and silent):
                continue                  # double-valued site irreps say nothing here
            if any(int(image[int(x)]) != int(x) for x in d["channels"]):
                continue
            got = {}
            for x in d["channels"]:
                i = part_of.get(int(classes[int(x)]))
                if i not in nus:
                    break                 # no scale in its part at this operation
                theta = _theta(anc, ops, x, V, lift, *nus[i])
                if theta is None:
                    break
                got[int(x)] = theta
            if len(got) != len(d["channels"]):
                continue
            for x, theta in got.items():
                joint_theta.setdefault(id(d), {}).setdefault(int(x), {})[ind] = theta
        if not nus:
            continue
        read = set()
        for c in sorted(want_of):
            if int(image[int(c)]) != int(c):
                continue
            d = want_of[int(c)]
            # a declaration over several multiplets is read once, as the sum of
            # their characters
            if id(d) in read or part_of_decl[id(d)] is None:
                continue
            read.add(id(d))
            if part_of_decl[id(d)] not in nus:
                continue                  # no scale in its part at this operation
            if ind not in d["constraints"] or (d.get("corep") and silent):
                continue                  # double-valued site irreps say nothing here
            nu, floor = nus[part_of_decl[id(d)]]
            if ind in (d.get("matrices") or {}):
                # generator matrices are read as matrices: the declared one is
                # the measured one times the value the part's character has to
                # take here, or no multiple of it
                if any(int(image[int(x)]) != int(x) for x in d["channels"]):
                    continue
                B = _matrix(ops, nspin, d, V, lift, nu, floor)
                if B is None:
                    continue
                M = np.asarray(d["matrices"][ind], dtype=complex)
                lam, gap = _proportionality(M, B)
                pre.append({"orbitals": tuple(d["orbitals"]), "ind": ind, "name": name,
                            "measured": complex(np.trace(B)),
                            "declared": complex(np.trace(M)),
                            "part": int(part_of_decl[id(d)]),
                            "required": (lam if gap <= tol * max(1.0, float(np.abs(M).max()))
                                         else None),
                            "matrices": (M, B)})
                continue
            thetas = []
            for x in d["channels"]:
                if int(image[int(x)]) != int(x):
                    break
                theta = _theta(anc, ops, x, V, lift, nu, floor)
                if theta is None:
                    break
                thetas.append(theta)
            if len(thetas) != len(d["channels"]):
                continue
            pre.append({"orbitals": tuple(d["orbitals"]), "ind": ind, "name": name,
                        "measured": thetas[0] if len(thetas) == 1 else complex(sum(thetas)),
                        "declared": complex(d["constraints"][ind]),
                        "part": int(part_of_decl[id(d)])})

    for r in pre:
        if "matrices" in r:
            rows.append(_Row(**r))        # its scalar was read off the matrices
            continue
        theta, want = complex(r["measured"]), complex(r["declared"])
        rows.append(_Row(**r, required=None if abs(theta) <= tol else want / theta))
    return _CharactersRead(joint_theta=joint_theta)


def _verdict_made(model, group, assignment, spin, validated, layout, decls, anc, ops, prt, scl,
                  chv, *, ref_orb, space_tol, tol, max_candidates, reversal, sectors):
    """The verdict, what each operation asks of each coupled part.

    ``model``, ``group``, ``assignment``, ``spin``, ``validated``, ``layout``,
    ``decls``, ``ref_orb``, ``space_tol``, ``tol``, ``max_candidates``,
    ``reversal`` and ``sectors`` are as for :func:`_verdicts_and_choices`, and
    ``anc``, ``ops``, ``prt``, ``scl`` and ``chv`` are what the five stages
    before it return.  Each part's rows become one requirement per operation
    (:func:`_needs_per_operation`), starting from the operations the
    multiplets break, and the sentence for each operation broken is added to
    ``ops.why``.  The declarations over several parts, or on part of a site,
    and those on an orbit of sectors built by induction, are kept as
    :class:`_ReadTogether`.  Time reversal's scalar on each multiplet orbit is
    read where ``reversal`` is given and its solution space and its image of
    the multiplets can be read; the verdict's ``reversal`` is ``None``
    otherwise.  The sectors' search (:func:`_declaration_search`) is read
    where ``decls`` is not empty.  Returns the :class:`_Verdict`.
    """
    classes = anc.classes
    cperm = anc.cperm
    measured_of = anc.measured_of
    multiplicity = anc.multiplicity
    nchan = anc.nchan
    nclass = anc.nclass
    positions = anc.positions
    ref = anc.ref
    syms = anc.syms
    bases = ops.bases
    block_tol = ops.block_tol
    broken = ops.broken
    chan_states = ops.chan_states
    chosen = ops.chosen
    free = ops.free
    rows = ops.rows
    seen = ops.seen
    su2 = ops.su2
    why = ops.why
    zero = ops.zero
    free_classes = prt.free_classes
    induced_classes = prt.induced_classes
    joint_decls = prt.joint_decls
    on_induced = prt.on_induced
    part_of = prt.part_of
    parts = prt.parts
    within = prt.within
    reversal_image = scl.reversal_image
    roots = scl.roots
    joint_theta = chv.joint_theta

    # Every declaration constrains its own part's character, wherever it sits:
    # what each operation asks of that character, part by part.  An operation
    # the multiplets break is broken in every part, and one that two
    # declarations of a part ask different values of is broken there; which
    # characters meet the rest is the subgroup search's question.
    requirements, scalars, by_matrices = [], [], []
    for i in range(len(parts)):
        req, raw, said = dict(broken), {}, {}
        _needs_per_operation([r for r in rows if r.part == i], req, raw, said,
                             tol=tol, roots=roots.get(i, False))
        requirements.append(req)
        scalars.append(raw)
        # ... and what the part's declared matrices ask, which a reading of
        # characters alone does not hear (:func:`_covariant_subgroup`)
        asked = {int(r.ind) for r in rows if r.part == i and r.matrices is not None}
        by_matrices.append({ind: req[ind] for ind in sorted(asked) if ind in req})
        for ind, sentence in said.items():
            why.setdefault(int(ind), sentence)
    # ... and every declaration read together over its multiplets, per
    # multiplet, with what reads it on a smaller group: the site group the
    # operations kept leave it, and the trace of each operation's own
    # spin-1/2 lift
    kept_inds = {int(syms[p].ind) for p in positions}
    lift_trace = {int(syms[p].ind): complex(np.trace(_lift_of(syms[p], su2)))
                  for p in positions}
    joint = []
    for d in joint_decls:
        theta = joint_theta.get(id(d))
        if not theta:
            continue
        site = int(d["site"])
        joint.append(_ReadTogether(
            decl=d, theta=theta,
            part={int(c): int(part_of[int(classes[int(c)])]) for c in d["channels"]},
            site_group=tuple(int(i) for i in assignment.site_group[site]
                             if int(i) in kept_inds),
            table_of=_SiteTableOf(group, assignment, site, corep=bool(d.get("corep"))),
            lift_trace=lift_trace))
    # ... and every declaration on an orbit built by induction, which the
    # sectors alone weigh: what its multiplets carry under the representation
    # they are assembled with, at every operation fixing all of them, so that
    # an operation it breaks is said of its own orbitals
    by_sectors = []
    held_by = {int(c): rec for rec in multiplicity
               if rec.induced is not None and rec.n_admitted
               for c in rec.multiplets}
    for d in on_induced:
        if not all(int(x) in held_by for x in d["channels"]):
            continue
        theta = {}
        for p in positions:
            ind = int(syms[p].ind)
            if ind not in d["constraints"] or any(int(cperm[p][int(x)]) != int(x)
                                                   for x in d["channels"]):
                continue
            if d.get("corep") and abs(complex(np.trace(_lift_of(syms[p], su2)))) <= zero:
                continue                  # double-valued site irreps say nothing here
            for x in d["channels"]:
                got = held_by[int(x)].candidates[0]["contents"].get(int(x), {})
                if ind in got:
                    theta.setdefault(int(x), {})[ind] = complex(got[ind])
        if theta:
            by_sectors.append(_ReadTogether(decl=d, theta=theta))

    # What time reversal carries on each orbit of multiplets, read here once:
    # its own internal factor on every multiplet, times the scalar its
    # solution space pins on each orbit, which is one on orbitals that are
    # real functions (:func:`_reversal_scalars`)
    reversal_scalars = None
    if reversal is not None and reversal_image is not None:
        t_ind = int(syms[int(reversal)].ind)
        basis_t = bases.get(t_ind)
        if basis_t is not None and len(np.asarray(basis_t)):
            basis_t = np.asarray(basis_t)
            n_t = int(basis_t.shape[1])
            lift_t = _lift_of(syms[int(reversal)], su2)
            got_t = measured_of.get(t_ind, {})
            lifts_t = [got_t.get(int(c), None if layout._chan_kind[int(c)] == "measured"
                                 else lift_t) for c in range(nchan)]
            if (basis_t.shape[0] * n_t * n_t * 16 <= _READOFF_MAX_BYTES
                    and all(S is not None for S in lifts_t)):
                reversal_scalars = _reversal_scalars(
                    basis_t, _ansatz_blocks(chan_states, reversal_image, 1), lifts_t,
                    classes, ref, tol=noise_floor(_IN_BLOCKS, block_tol))
                # an orbit built by induction carries time reversal's blocks of
                # its own representation as they stand
                if reversal_scalars is not None and induced_classes:
                    reversal_scalars = np.array(reversal_scalars, dtype=complex)
                    for own in induced_classes:
                        for cls in own:
                            reversal_scalars[int(cls)] = 1.0 + 0.0j
    # What a declaration asks of the sectors, on any subgroup: read off the
    # conserved matrices the multiplets carry and what this round's operations
    # do to them (:class:`.._sectors.SectorSearch`)
    search = None
    if decls:
        search = _declaration_search(model, group, assignment, spin, validated, layout,
                                     decls, positions, reversal=reversal, sectors=sectors,
                                     tol=tol, max_candidates=max_candidates)
    return _Verdict(rows=tuple(rows), parts=tuple(parts),
                    requirements=tuple(requirements), scalars=tuple(scalars),
                    matrices=tuple(by_matrices),
                    joint=tuple(joint), by_sectors=tuple(by_sectors),
                    reasons=why,
                    classes=classes, nclass=int(nclass), chosen=chosen, within=within,
                    free=frozenset(free), free_classes=free_classes,
                    seen={int(k): frozenset(v) for k, v in seen.items()},
                    anchor=int(ref), reference_orbital=int(ref_orb),
                    declared=_declared_key(decls), space_tol=float(space_tol),
                    indices=frozenset(int(syms[p].ind) for p in positions),
                    blocks=_MultipletBlocks(measured_of, multiplicity=multiplicity),
                    reversal=reversal_scalars, search=search)


def _joined(joined, cls):
    while joined[cls] != cls:
        cls = joined[cls]
    return cls


def _theta(anc, ops, x, V, lift, nu, floor):
    """Multiplet ``x``'s character at one operation, against its part's scale.

    ``V`` holds the members of the operation's in-block solution space,
    ``lift`` is the operation's own internal factor
    (:func:`.._matrices._lift_of`), ``nu`` the scale's coefficient in each
    member and ``floor`` the largest of their moduli; ``anc`` and ``ops`` are
    what :func:`_anchor_read` and :func:`_operations_read` return.  On a
    measured multiplet the character is the orbital one, the spinor divided
    out atom by atom.  Returns the character, or ``None`` where its ratio to
    ``nu`` varies across the members beyond ``ops.spread_tol``: the hoppings
    leave it free.
    """
    layout = anc.layout
    chan_states = ops.chan_states
    spread_tol = ops.spread_tol
    su2 = ops.su2

    cols = np.asarray(chan_states[int(x)], dtype=int)
    if layout._chan_kind[int(x)] == "measured":
        # the multiplet's orbital character: the block's trace with the
        # spinor divided out, atom by atom, which is what the declared
        # single-valued irrep is a statement about
        atoms = _atom_blocks(chan_states[int(x)], su2)
        f = np.asarray([sum(_proportionality(V[j][np.ix_(a, a)], lift)[0]
                            for a in atoms) for j in range(len(V))])
    else:
        f = np.asarray([_proportionality(V[j][np.ix_(cols, cols)], lift)[0]
                        for j in range(len(V))])
    theta = complex(np.vdot(nu, f) / np.vdot(nu, nu))
    gap = float(np.abs(f - theta * nu).max())
    if gap > spread_tol * max(float(np.abs(f).max()), floor, 1.0):
        return None                   # free: the hoppings do not measure it
    return theta


def _matrix(ops, nspin, d, V, lift, nu, floor):
    """The orbital matrix of declaration ``d`` at one operation, on its scale.

    The matrix is the orbital factor of the block the operation carries
    ``d``'s orbitals with: the spinor divided out atom by atom, one row and
    one column per atom in the order ``d`` names them
    (:func:`_declared_atoms`).  Declared generator matrices are compared with
    it.  ``ops``, ``V``, ``lift``, ``nu`` and ``floor`` are as for
    :func:`_theta`, and ``nspin`` is the model's number of spin states per
    orbital.  Returns the matrix, or ``None`` where its ratio to ``nu`` varies
    across the members beyond ``ops.spread_tol``: the hoppings leave it free.
    """
    chan_states = ops.chan_states
    spread_tol = ops.spread_tol
    su2 = ops.su2

    atoms = _declared_atoms(d, chan_states, su2, nspin)
    B = np.asarray([[[_proportionality(V[j][np.ix_(a, b)], lift)[0] for b in atoms]
                     for a in atoms] for j in range(len(V))])
    M = np.tensordot(np.conj(nu), B, axes=(0, 0)) / np.vdot(nu, nu)
    gap = float(np.abs(B - nu[:, None, None] * M[None, :, :]).max())
    if gap > spread_tol * max(float(np.abs(B).max()), floor, 1.0):
        return None                   # free: the hoppings do not measure it
    return M


def _declaration_search(model, group, assignment, spin, validated, layout, decls, positions,
                        *, reversal=None, sectors=None, tol=CHARACTER, max_candidates=2 ** 20):
    r"""What the declarations ask of the sectors on any subgroup, or ``None``.

    ``model``, ``group``, ``assignment``, ``spin``, ``validated``, ``layout``,
    ``decls``, ``tol`` and ``max_candidates`` are as for
    :func:`_verdicts_and_choices`, and ``positions`` holds the positions in
    ``group`` of the unitary operations read.  The sectors are ``sectors``
    where it is handed in, and are read here otherwise
    (:func:`.._sectors.read_sectors`), at the tolerance the multiplets were
    measured at.  They are carried onto the multiplets
    (:func:`.._sectors.carried_reading`), with what the operations do to them
    (:func:`.._sectors.sector_action`).  ``reversal`` is time reversal's
    position, read only where it is the pure time reversal.  Returns the
    :class:`.._sectors.SectorSearch`, or ``None`` where no sectors are read or
    an operation has no reading on them.
    """

    eps = getattr(layout, "_tol", None) or None
    reading = (sectors if sectors is not None else
               _sectors.read_sectors(model, group, assignment, spin, validated, tol=eps))
    if reading is None:
        return None
    carried = _sectors.carried_reading(reading, layout, tol=eps)
    action = _sectors.sector_action(carried, model, group, assignment, spin, validated, tol=eps)
    pure = None
    if reversal is not None:
        sym = group._symmetries[int(reversal)]
        if (np.array_equal(np.rint(np.asarray(sym.rotation)).astype(int), np.eye(3, dtype=int))
                and np.allclose(np.asarray(sym.translation) % 1.0, 0.0, atol=ZERO)):
            pure = int(reversal)
    search = _sectors.SectorSearch(carried, action, group, assignment, decls, positions,
                                   reversal=pure, tol=tol, max_candidates=max_candidates)
    return search if search.complete else None


def _reference_declaration(decls, ref_orb):
    """The declaration in ``decls`` naming the reference orbital ``ref_orb``.

    The reference orbital is the one the lowest-angular-momentum rule reads
    first.  A declaration that names it inside a multiplet counts.  Returns
    the first such declaration, or ``None`` where none names it.
    """
    return next((d for d in decls if int(ref_orb) in
                 {int(o) for o in d["orbitals"]}), None)


def _needs_per_operation(rows, req, raw, why, *, tol=CHARACTER, roots=False):
    """What a coupled part's declarations ask of its character, per operation.

    ``rows`` is the measured-versus-declared table of one part.  For each
    operation, ``req`` receives the value ``λ(g)`` the part's one-dimensional
    character has to take for the declarations to hold, or ``None`` where no
    value does; ``raw`` that value before it is rounded onto a sign or a
    phase; and ``why`` one sentence per operation broken, saying what was
    declared and what the hoppings carry.  An operation already in ``req``,
    broken by the multiplets, is left alone.  Two rows asking values more than
    ``tol`` apart break the operation, as does a row whose measured character
    vanishes where its declared one does not, or a declared matrix that no
    scalar carries the measured one onto.  A value other than a sign breaks
    it too; with ``roots=True``, for a group with no antiunitary operation, a
    value of unit modulus is kept as a phase instead.  Returns ``req``,
    ``raw`` and ``why``, filled in place.  Which characters meet these values
    is :func:`_covariant_subgroup`'s to read.
    """
    by_ind: dict = {}
    for r in rows:
        by_ind.setdefault(int(r.ind), []).append(r)
    for ind, group_rows in by_ind.items():
        if ind in req:                    # the multiplets already broke this operation
            continue
        need, first = None, None
        for r in group_rows:
            if need == "broken":
                break
            orbs = tuple(r.orbitals)
            if r.required is None:
                if r.matrices is not None:
                    # a declared matrix no scalar carries the measured one onto
                    need = "broken"
                    why[ind] = (f"orbitals {orbs} are declared "
                                f"{_stated(r, 'declared')} under this operation, and the "
                                f"hoppings carry them with {_stated(r, 'measured')} up to "
                                "one scalar, which is no multiple of it")
                elif abs(complex(r.declared)) > tol:
                    need = "broken"
                    why[ind] = (f"orbitals {orbs} are declared "
                                f"{_fmt_char(r.declared)} under this operation, and "
                                "the character the hoppings carry them with vanishes")
                continue
            lam = complex(r.required)
            if need is None:
                need, first = lam, r
            elif abs(need - lam) > tol and (r.matrices is not None
                                            or first.matrices is not None):
                why[ind] = (f"the declarations of orbitals {tuple(first.orbitals)} and "
                            f"{orbs} ask their part's one-dimensional character for "
                            f"{_fmt_char(need)} and {_fmt_char(lam)} under this operation, "
                            "and it takes one value")
                need = "broken"
            elif abs(need - lam) > tol:
                # both measured characters are nonzero here: a row whose
                # measured character vanishes carries no requirement
                ratio = complex(r.measured) / complex(first.measured)
                base = complex(first.declared)
                said = (f"state {_fmt_char(complex(r.declared) / base)}"
                        if abs(base) > tol else "are not in that ratio")
                why[ind] = (f"relative to orbitals {tuple(first.orbitals)}, the "
                            f"hoppings carry orbitals {orbs} with {_fmt_char(ratio)} "
                            f"under this operation, and the declarations "
                            f"({tuple(first.orbitals)} "
                            f"{_fmt_char(first.declared)}, {orbs} "
                            f"{_fmt_char(r.declared)}) {said}")
                need = "broken"
        if need is None:
            continue
        if need == "broken":
            req[ind] = None
            continue
        raw[ind] = complex(need)
        if roots and min(abs(need - 1.0), abs(need + 1.0)) > tol:
            if abs(abs(need) - 1.0) > tol:
                req[ind] = None
                why[ind] = (f"orbitals {tuple(first.orbitals)} are declared "
                            f"{_stated(first, 'declared')} under this operation and "
                            f"the hoppings carry them with {_stated(first, 'measured')}, "
                            f"a ratio of {_fmt_char(need)} that is not a phase, and so "
                            "the value of no one-dimensional character")
            else:
                req[ind] = complex(need) / abs(need)
        elif min(abs(need - 1.0), abs(need + 1.0)) > tol:
            req[ind] = None
            why[ind] = (f"orbitals {tuple(first.orbitals)} are declared "
                        f"{_stated(first, 'declared')} under this operation and the "
                        f"hoppings carry them with {_stated(first, 'measured')}, a "
                        f"ratio of {_fmt_char(need)} that is not a sign: an antiunitary "
                        "operation of the group leaves only a sign between two "
                        "labelings of one crystal")
        else:
            req[ind] = complex(round(float(np.real(need))))
    return req, raw, why


def _stated(row, which) -> str:
    """What one row of the measured-versus-declared table states, as text.

    ``row`` is a :class:`_Row`, and ``which`` is ``"declared"`` or
    ``"measured"``.  Returns that character, or for a declaration of generator
    matrices that matrix, formatted for a sentence.
    """
    if row.matrices is not None:
        return _sitechars._fmt_matrix(row.matrices[0 if which == "declared" else 1])
    return _fmt_char(getattr(row, which))


def _declared_atoms(d, chan_states, su2, nspin):
    """The states of each atom declaration ``d`` names, in its order.

    An atom is one orbital, or the two states of one Wannier function, which
    carry an operation's spin-1/2 lift together
    (:func:`.._matrices._atom_blocks`).  ``chan_states`` holds each
    multiplet's states, ``su2`` says whether the states carry the spin-1/2
    lift, and ``nspin`` is the model's number of spin states per orbital.
    Returns one array of state indices per atom of ``d``'s multiplets that
    ``d``'s orbitals name, in the order they first name them: the rows and
    columns of declared generator matrices.
    """
    atoms = [a for c in d["channels"] for a in _atom_blocks(chan_states[int(c)], su2)]
    out, taken = [], set()
    for o in d["orbitals"]:
        for k, a in enumerate(atoms):
            if k not in taken and int(o) in {int(s) // int(nspin) for s in a}:
                taken.add(k)
                out.append(np.asarray(a, dtype=int))
                break
    return out


def _measure_channel_characters(D, syms, kept_u, cperm, layout, chan):
    """The orbital character of multiplet ``chan`` at each operation fixing it.

    ``D`` is the representation :func:`.._build._build_rep` built on the
    scalars the verdict pinned, before any labeling is chosen, ``D[p]`` the
    matrix of the operation at position ``p``; the Hamiltonian is not read.
    ``syms`` are the group's operations, ``kept_u`` the positions of the
    unitary ones read, ``cperm[p][c]`` the multiplet the operation at ``p``
    carries ``c`` onto, and ``layout`` the multiplets.  A one-dimensional
    multiplet's block is ``chi(g)`` times the operation's own spin-1/2 lift,
    and ``chi`` is read by least squares; a wider multiplet's character is the
    trace of its block, read atom by atom the same way.  The lift is divided
    out where ``layout._spinful`` says the states carry it, a declared spin
    pair as much as a native spinful model, so ``chi`` is a single-valued
    character of the site group.  Returns ``({ind: chi}, residual)``,
    ``residual`` the worst least-squares defect of a one-dimensional block,
    ``0.0`` for a wider multiplet.
    """
    chi, worst = {}, 0.0
    states = np.asarray(layout._chan_states[chan], dtype=int)
    measured = layout._chan_kind[int(chan)] == "measured"
    atoms = _atom_blocks(layout._chan_states[chan], layout._spinful) if measured else ()
    for p in kept_u:
        if int(cperm[p][chan]) != int(chan):
            continue
        S = _lift_of(syms[p], layout._spinful)
        if measured:
            chi[int(syms[p].ind)] = complex(sum(
                _proportionality(D[p][np.ix_(a, a)], S)[0] for a in atoms))
            continue
        B = D[p][np.ix_(states, states)]
        rho, res = _proportionality(B, S)
        chi[int(syms[p].ind)] = complex(rho)
        worst = max(worst, float(res))
    return chi, worst


def _declaration_entries(characters, model, assignment, *, spin=None):
    r"""``characters=`` parsed into its entries and the atoms they name.

    ``characters`` is the declaration as written, on the orbitals of
    ``model``, whose sites ``assignment`` gives; ``spin`` is the declared spin
    basis, or ``None``.  Step 1 (:func:`_declared_site_contents`), before any
    multiplet is measured, and :func:`_normalize_declaration` read this one
    parse, so the two name the same orbitals.  In the list form, the orbitals
    of one site written with the same word are one entry, and a per-operation
    declaration is one entry per orbital.  An atom is one orbital, or the two
    of a declared spin pair, one Wannier function.  Returns
    ``(entries, atom_of, atom_orbs)``: ``entries`` one
    ``(orbitals, spec, joint)`` per declaration in the order written,
    ``joint`` true for a key naming several orbitals; ``atom_of`` maps each
    orbital to its atom's key, ``(site, lowest orbital)``; and ``atom_orbs``
    maps that key to the atom's orbitals.

    Raises
    ------
    ValueError
        Where the declaration is malformed
        (:func:`.._grammar._declaration_grammar`).
    """
    _declaration_grammar(int(model.norb), characters=characters, spin=spin)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    atom_of, atom_orbs = {}, {}
    for s, site_atoms in enumerate(_atoms_of_site(site_orbs, spin)):
        for a in site_atoms:
            key = (int(s), int(min(a)))
            atom_orbs[key] = tuple(int(o) for o in a)
            for o in a:
                atom_of[int(o)] = key
    raw = []                    # (the orbitals of one declaration, its spec, joint?)
    if isinstance(characters, dict):
        for key, spec in characters.items():
            joint = isinstance(key, (tuple, list)) and len(key) > 1
            orbs = tuple(int(k) for k in (key if isinstance(key, (tuple, list)) else (key,)))
            raw.append((orbs, spec, bool(joint)))
    else:
        seq = list(characters)
        by_site = {}
        for a, spec in enumerate(seq):
            if spec is None:
                continue
            key = (int(site_of[a]), spec.lower() if isinstance(spec, str) else id(spec))
            by_site.setdefault(key, []).append((a, spec))
        for (site, _), entries in by_site.items():
            orbs = tuple(a for a, _ in entries)
            spec = entries[0][1]
            if not isinstance(spec, str):
                for a in orbs:                   # a per-operation declaration never auto-groups
                    raw.append(((a,), spec, False))
                continue
            raw.append((orbs, spec, False))
    return tuple(raw), atom_of, atom_orbs


def _declared_site_contents(model, group, assignment, characters, *, spin=None):
    r"""What each site was declared to hold, read in ``group``'s site tables.

    Step 1's reading of ``characters``, made before any Hamiltonian is read.
    ``model``, ``assignment`` and ``spin`` are as for
    :func:`_declaration_entries`.  Each word is read on the site group
    ``group`` gives its site: a shorthand as its Cartesian polynomial
    (:data:`.._sitechars.SHORTHANDS`), a label as its characters, and a
    declaration already read (:class:`.._sitechars.DeclaredContent`) as the
    characters it carries.  A shorthand names a function, whose span need not
    be invariant under the site group; a label names site irreps, invariant
    by construction.

    Returns ``(contents, read_in)``.  ``contents`` maps each declared site to
    ``"polys"``, the declared functions' polynomials, emptied where their
    orbit under the site group is no wider than the atoms they are written
    on; ``"atoms"``, the atoms the functions are written on; ``"chars"``, one
    ``{ind: chi}`` per declaration, a span's own character standing for a
    function; ``"entries"``, the site's entries
    (:func:`_declaration_entries`); ``"corep"``, whether any of them is
    written in double-valued site irreps; and ``"whole"``, the characters of
    all the site was declared to hold.  ``read_in`` maps each declaration's
    sorted orbitals to the site group and group its word was read in.

    ``"whole"`` sums the site's declarations, however they partition its
    orbitals, on the operations they all state: ``"p"`` on a ``p`` shell, and
    ``pz`` beside ``E`` on its orbitals, are both ``A1 + E`` at a ``3m`` site.
    Words that name parts of one wider irrep add up to whole copies of it
    (``px`` and ``py`` are one ``E``), the count rounded up where they make no
    whole number.  A declared spin pair counts once.

    Raises
    ------
    _sitechars.DeclarationError
        ``gate="unknown_label"`` where a declared word names nothing in its
        site group.
    ValueError
        Where the declaration is malformed
        (:func:`.._grammar._declaration_grammar`).
    InternalError
        Where a site's character table cannot be made
        (:func:`.._sitechars.site_group_table`).  Its one caller,
        :func:`.._descent._declared_cell_reduction`, returns it as the
        refusal ``site_group``.
    """
    entries, atom_of, _atom_orbs = _declaration_entries(
        characters, model, assignment, spin=spin)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    kept_u = {int(s.ind) for s in group._symmetries if not s.time_reversal}
    kept_au = {int(s.ind) for s in group._symmetries if s.time_reversal}
    tables = _site_tables(group, assignment, kept_u, kept_au,
                          sorted({int(site_of[a]) for orbs, _s, _j in entries
                                  for a in orbs}))
    contents: dict = {}
    read_in: dict = {}
    for orbs, spec, joint in entries:
        site = int(site_of[orbs[0]])
        table = tables[site]
        where = f"orbital{'s' if len(orbs) > 1 else ''} {orbs if len(orbs) > 1 else orbs[0]}"
        rec = contents.setdefault(int(site), {"polys": [], "chars": [], "entries": [],
                                              "atoms": set(), "pool": {}, "corep": False})
        read_in[tuple(sorted(int(a) for a in orbs))] = (
            f"{_sitechars.point_group_symbol(group, table.indices)} in "
            f"{group.name} #{group.number_str}")
        named = {atom_of[int(a)] for a in orbs}
        if isinstance(spec, _sitechars.DeclaredContent):
            # already read, on this same cell: carried, never read twice
            chi = {int(k): complex(v) for k, v in spec.constraints.items()
                   if int(k) in set(table.indices)}
            dim, corep = int(spec.dim), bool(spec.corep)
            if spec.functions:
                rec["polys"].extend(spec.functions)
                rec["atoms"].update(named)
        elif _sitechars.shorthand_key(spec) is not None:
            poly = _sitechars.SHORTHANDS[_sitechars.shorthand_key(spec)]
            span, dim = _sitechars.span_character(group, table, [poly])
            rec["polys"].append(poly)
            rec["atoms"].update(named)
            chi = {int(ind): complex(span[table.class_of[int(ind)]]) for ind in table.indices}
            corep = False
        else:
            cons, _label, dim, _note = _sitechars.resolve_spec(group, table, spec, where)
            chi = {int(k): complex(v) for k, v in cons.items()}
            corep = _sitechars.named_in_coreps(table, spec)
        rec["chars"].append(chi)
        rec["entries"].append((orbs, spec, joint))
        rec["corep"] = rec["corep"] or bool(corep)
        # the orbitals written with the same characters at this site, pooled:
        # parts of a wider irrep add up to copies of it, and a declared spinor
        # pair written twice, once per member, is one atom
        part = rec["pool"].setdefault(_declared_character_key(chi),
                                      {"chi": chi, "dim": max(1, int(dim)), "atoms": set()})
        part["atoms"].update(named)
    for rec in contents.values():
        pieces = list(rec.pop("pool").values())
        asked = set.intersection(*(set(p["chi"]) for p in pieces)) if pieces else set()
        whole = {}
        for p in pieces:
            copies = len(p["atoms"]) if p["dim"] == 1 else max(
                1, -(-len(p["atoms"]) // p["dim"]))
            for ind in asked:
                whole[int(ind)] = whole.get(int(ind), 0.0) + copies * complex(p["chi"][ind])
        rec["whole"] = whole
    # Whether the declared functions name their orbit or themselves.  The
    # functions declared at a site generate an orbit under its site group.
    # Where that orbit is as wide as the atoms they were declared for, what
    # was declared is the orbit, such as the doublet `px` and `py` span at a
    # `3m` site, or the `1E2E` pair one `dz2` names at a `23` site, and an
    # orbit is invariant, so it reduces nothing.  Where the orbit is wider than
    # the atoms named, what was declared is the functions themselves, and a
    # site group need not preserve their span.
    for site, rec in contents.items():
        if not rec["polys"]:
            continue
        _chi, rank = _sitechars.span_character(group, tables[site], rec["polys"])
        if int(rank) <= len(rec["atoms"]):
            rec["polys"] = []
    return contents, read_in


def _content_compatible(group, assignment, contents):
    r"""Step 1: the operations of the cell that carry the declaration.

    ``contents`` is the first of what :func:`_declared_site_contents` returns,
    and ``assignment`` holds the orbitals' sites in ``group``.  A unitary
    operation ``g`` is compatible where, at every declared site ``a`` whose
    image ``b`` is declared too, both of these hold.

    * Where both sites keep declared functions (``"polys"``), ``g`` carries
      the span of ``a``'s onto the span of ``b``'s
      (:func:`.._sitechars.carries_span`).  With ``b = a`` this is the
      stabilizer of a declared function.
    * ``chi_b(g h g^-1) = chi_a(h)`` for every ``h`` fixing ``a`` that both
      sites' characters state, ``chi`` being each site's ``"whole"`` and the
      conjugation taken in the group's product table modulo lattice
      translations.  Where either site is declared in double-valued site
      irreps, the operations whose spin-1/2 lift has zero trace are not
      compared.

    An operation carrying a declared site onto an undeclared one is compatible
    there.  A primed operation stands or falls with its unitary partner, a
    site character being a character of the unitary site group.  The
    operations kept are the largest subgroup inside the compatible ones
    (:func:`_largest_compatible_group`).  Returns ``(kept, excluded)``: the
    indices kept, a frozenset over all operations, and for each operation left
    out a record of its ``"reason"``, ``"sites"``, ``"orbitals"``, ``"ind"``,
    ``"name"`` and ``"note"`` (:func:`_excluded_named`).
    """
    syms = group._symmetries
    at = {int(s.ind): p for p, s in enumerate(syms)}
    prod = group._get_product_table(get_diff=False)
    perm = np.asarray(assignment.permutations, dtype=int)
    unitary = [int(s.ind) for s in syms if not s.time_reversal]
    spans = {s: _sitechars.span_basis(rec["polys"]) if rec["polys"] else None
             for s, rec in contents.items()}
    fixed = {s: tuple(int(i) for i in assignment.site_group[s] if int(i) in at
                      and not syms[at[int(i)]].time_reversal)
             for s in contents}
    # the inverse of each unitary operation, read off the group's own product
    # table modulo lattice translations; `get_inverse_table` is not used, since
    # it is asked about the antiunitary coset too and warns that it has no
    # inverse
    ident = _identity_position(group)
    inverse = {a: b for a in unitary for b in unitary
               if int(prod[at[a], at[b]]) == ident}

    def _conjugate(g, h):
        """``g h g^-1`` as an index, modulo lattice translations."""
        gi = inverse[int(g)]
        return int(syms[int(prod[at[int(g)], int(prod[at[int(h)], at[gi]])])].ind)

    # the operations a declaration in double-valued site irreps states nothing
    # under: those whose spin-1/2 lift has zero trace
    silent = ({ind for ind, v in _sitechars.lift_squares(
                  {ind: syms[at[ind]].rotation_cart for ind in unitary}).items() if v <= 1e-9}
              if any(rec.get("corep") for rec in contents.values()) else set())

    def _matched(g, a, b):
        """Whether ``g`` carries site ``a``'s declaration onto site ``b``'s.

        The two sites' whole declarations are compared:
        ``chi_b(g h g^-1) = chi_a(h)`` on every ``h`` fixing ``a`` that both
        state.  The relation is symmetric, and among declared sites the
        operations it keeps are closed under products.  Two sites declared
        different dimensions differ on the identity.
        """
        chi_a, chi_b = contents[a]["whole"], contents[b]["whole"]
        coreps = bool(contents[a].get("corep") or contents[b].get("corep"))
        for h in fixed[a]:
            if int(h) not in chi_a or (coreps and int(h) in silent):
                continue
            c = _conjugate(g, h)
            if c not in chi_b:
                # b's declaration states nothing at this operation: leaving it
                # out restricts the comparison to the operations both state
                continue
            if abs(complex(chi_b[c]) - complex(chi_a[int(h)])) > 1e-6:
                return False
        return True

    def _orbitals(*sites):
        return tuple(sorted({int(o) for s in sites
                             for orbs, _spec, _joint in contents[int(s)]["entries"]
                             for o in orbs}))

    excluded: dict = {}
    for g in unitary:
        for a in sorted(contents):
            b = int(perm[at[g], int(a)])
            if b not in contents:
                continue                      # an undeclared site imposes nothing
            if (spans[a] is not None and spans[b] is not None
                    and not _sitechars.carries_span(group, g, spans[a], spans[b])):
                excluded[g] = {
                    "reason": ("the declared function at this site is not carried to "
                               "itself" if b == a else
                               "the declared function is not carried onto the image "
                               "site's own declared content"),
                    "sites": (int(a), int(b)), "orbitals": _orbitals(a, b)}
                break
            if not _matched(g, a, b):
                excluded[g] = {
                    "reason": ("the declared content is not carried to itself" if b == a
                               else "declared content differs across the orbit"),
                    "sites": (int(a), int(b)), "orbitals": _orbitals(a, b)}
                break
    partners = _primed_partners(group, [int(s.ind) for s in syms])
    for g, rec in list(excluded.items()):
        mate = partners.get(int(g))
        if mate is not None:
            excluded[int(mate)] = dict(rec)
    for g, rec in excluded.items():
        _excluded_named(group, int(g), rec)
    return _largest_compatible_group(group, assignment, _orbitals(*contents), excluded)


#: Why step 1 leaves out an operation that carries the declaration on its
#: own: the operations kept are the largest group inside the compatible ones
#: (:func:`_largest_compatible_group`).  ``_OUTSIDE_THE_GROUP`` is said where
#: no subgroup of that order holds the operation, ``_OUTSIDE_THE_TIE`` where
#: one does and the tie rule keeps another.
_OUTSIDE_THE_GROUP = ("the declared content is carried by this operation on its own, but "
                      "not by every product of it with the operations kept")

_OUTSIDE_THE_TIE = ("the declared content is carried by a group holding this operation, as "
                    "large as the one kept, and the tie rule keeps the other")


def _outside_the_group(rec) -> bool:
    """Whether step 1 excluded an operation only for the largest group's sake.

    ``rec`` is the operation's record in step 1's exclusions.  Returns true
    where its reason is ``_OUTSIDE_THE_GROUP`` or ``_OUTSIDE_THE_TIE``: the
    operation carries the declaration on its own.
    """
    return rec.get("reason") in (_OUTSIDE_THE_GROUP, _OUTSIDE_THE_TIE)


def _excluded_named(group, ind, rec, *, said=""):
    """A step 1 exclusion with its operation's index, name and note added.

    ``rec`` is the record of the operation ``ind`` of ``group``, holding its
    ``"reason"``, ``"sites"`` and ``"orbitals"``, and ``said`` is appended to
    its note.  Sets ``rec["ind"]``, ``rec["name"]``
    (:func:`.._seitz._record_name`) and ``rec["note"]``, the sentence saying
    it in words, and returns ``rec``.
    """

    sym = next(s for s in group._symmetries if int(s.ind) == int(ind))
    rec["ind"] = int(ind)
    rec["name"] = _record_name(sym, group._lattice)
    rec["note"] = (f"#{int(ind)} {rec['name']}: {rec['reason']} (site "
                   + (f"{rec['sites'][0]}" if rec["sites"][0] == rec["sites"][1]
                      else f"{rec['sites'][0]} -> site {rec['sites'][1]}")
                   + f", orbitals {rec['orbitals']})" + said)
    return rec


def _largest_compatible_group(group, assignment, declared, excluded):
    r"""Step 1's operations kept: the largest group inside the compatible ones.

    ``excluded`` holds the operations that do not carry the declaration,
    unitary and primed alike (:func:`_content_compatible`), ``declared`` the
    orbitals the declaration names, and ``assignment`` the orbitals' sites in
    ``group``.  The compatible operations are the unitary ones left.  Where
    they compose, as they do wherever the declaration states every site of each
    orbit it touches, as a function at all of them or at none, nothing is
    added.  Where they do not, the operations kept are the largest subgroup of
    ``group`` inside them, found by the enumeration and the tie rule the
    descent uses (:func:`.._subgroups._resolve_closure`), each primed operation
    with its unitary partner.  Each operation left out is added to ``excluded``
    with the reason ``_OUTSIDE_THE_GROUP``, or ``_OUTSIDE_THE_TIE`` where a
    subgroup of the largest order holds it and the tie rule keeps another, the
    note then naming the subgroups that tie and why the first is kept.  Its
    ``"sites"`` are a declared site and its image, an undeclared image where
    there is one, and its ``"orbitals"`` the declared ones there.  Returns
    ``(kept, excluded)``: the indices kept, a frozenset over all operations,
    and ``excluded`` with those added.
    """

    syms = group._symmetries
    compatible = [s for s in syms if not s.time_reversal and int(s.ind) not in excluded]
    if _descent_closure(group, compatible):
        _group, dropped, tied = _resolve_closure(group, [int(s.ind) for s in compatible])
        at = {int(s.ind): p for p, s in enumerate(syms)}
        perm = np.asarray(assignment.permutations, dtype=int)
        site_of = np.asarray(assignment.site_of_orbital, dtype=int)
        named = sorted({int(o) for o in declared})
        sites = sorted({int(site_of[o]) for o in named})
        lattice = np.asarray(group._lattice, dtype=float)

        def _set(inds):
            return "{" + ", ".join(_record_name(syms[at[int(i)]], lattice) for i in inds) + "}"

        said = ""
        if tied:
            said = ("; the largest groups the declared content is carried by tie, "
                    + " and ".join(_set(c) for c in tied)
                    + f", and the tie rule -- {_tie_rule_words(group)} -- keeps the first "
                    f"since {_tie_reason(group, tied)}")
        partners = _primed_partners(group, [int(s.ind) for s in syms])
        for g, why in dropped:
            images = [(a, int(perm[at[int(g)], a])) for a in sites]
            a, b = next(((a, b) for a, b in images if b not in sites), images[0])
            rec = {"reason": _OUTSIDE_THE_TIE if why == "tie" else _OUTSIDE_THE_GROUP,
                   "sites": (int(a), int(b)),
                   "orbitals": tuple(o for o in named if int(site_of[o]) in (a, b))}
            for ind in (int(g), partners.get(int(g))):
                if ind is not None and ind not in excluded:
                    excluded[ind] = _excluded_named(group, ind, dict(rec),
                                                    said=said if why == "tie" else "")
    kept = frozenset(int(s.ind) for s in syms if int(s.ind) not in excluded)
    return kept, excluded


def _normalize_declaration(characters, model, assignment, group, tables, layout,
                           *, spin=None, read_in=None):
    """The declarations of ``characters``, each resolved in its site's table.

    ``characters`` is the declaration as written; ``model``, ``assignment``
    and ``spin`` are as for :func:`_declaration_entries`; ``group`` is the
    group the words are resolved in, ``tables`` the character table of each
    site (:func:`.._naming._site_tables`), and ``layout`` the multiplets the
    Hamiltonian measures.  ``read_in`` is step 1's reading
    (:func:`_declared_site_contents`), which says where each word was read
    before step 1 lowered the cell; without it a word is recorded as read
    where it resolves.

    The piece counted is the atom: one orbital, or the two of a declared spin
    pair, one Wannier function, never split.  A joint key names what its atoms
    hold together, an irrep, a sum of irreps or a shell, whose dimension is
    the number of atoms: ``{(0, 2): "E"}`` on spin pairs ``(0, 1)`` and
    ``(2, 3)`` is a doublet of the two pairs.  A word whose irrep is wider
    than the one orbital it is written on names that orbital's part of it:
    the whole multiplet measured there where that has the irrep's dimension
    (``{0: "E'"}`` on a ``p_x, p_y`` pair), and otherwise the parts written
    with one irrep at one site add up to whole copies of it.  Generator
    matrices are read on whole measured multiplets, one declaration per
    multiplet where they are block-diagonal on several.

    Returns a list of the declarations, each a mapping holding its
    ``"orbitals"``, ``"channels"`` (its multiplets), ``"site"``, ``"spec"``,
    ``"label"``, ``"dim"``, ``"note"``, ``"constraints"`` (``{ind: chi}``),
    ``"joint"``, ``"corep"``, ``"matrices"`` and ``"content"``, the
    :class:`.._sitechars.DeclaredContent` it resolved to, which a later group
    re-keys rather than resolving the word again.  :func:`_declared_blocks`
    then reads them on the measured multiplets.

    Raises
    ------
    _sitechars.DeclarationError
        ``gate="ambiguous_multiplet"`` for a joint key whose irrep's dimension
        is not the number of atoms it names, for parts of an irrep that make no
        whole number of copies at their site, and for generator matrices on
        part of a measured multiplet, mixing two, at a site whose declarations
        do not each name a whole measured multiplet, or at a site of several
        measured multiplets with a declaration in double-valued site irreps;
        ``gate="unknown_label"`` from resolving a word.
    ValueError
        Where the declaration is malformed
        (:func:`.._grammar._declaration_grammar`).
    """
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    paired = spin is not None              # each atom a declared (up, dn) pair
    chan_of, chan_orbs = layout._chan_of, layout._chan_orbs
    # the indivisible pieces, which is what the dimension check counts
    raw, atom_of, atom_orbs = _declaration_entries(characters, model, assignment,
                                                   spin=spin)

    def _atoms_named(orbs):
        """The distinct atoms of the orbitals ``orbs``, in the order named."""
        out = []
        for a in orbs:
            k = atom_of[int(a)]
            if k not in out:
                out.append(k)
        return tuple(out)

    def _members(keys):
        """The orbitals of the atoms ``keys``, atom by atom."""
        return tuple(int(o) for k in keys for o in atom_orbs[k])

    # The word written for each orbital, whichever declaration it came in: both
    # states of a declared spinor pair, one Wannier function, and every orbital
    # of a declaration already resolved.  A multiplet the Hamiltonian glued
    # and declared orbital by orbital (``{0: "px", 1: "py"}`` on a
    # ``p_x, p_y`` pair) resolves once per orbital to the whole multiplet, and
    # each of those declarations carries every word its orbitals were written
    # with, so that the one a later round keeps still says all of them.  A
    # shell is a function word of every orbital it was written on (``"p"``);
    # an irrep or a sum of irreps names site irreps and no function.
    said = {}
    for orbs, spec, _joint in raw:
        if isinstance(spec, _sitechars.DeclaredContent):
            for a, w in spec.words:
                said.setdefault(int(a), str(w))
            continue
        w = _sitechars.canonical_shorthand(spec) or _sitechars.shell_key(spec) or ""
        if w:
            for a in orbs:
                for o in atom_orbs[atom_of[int(a)]]:
                    said.setdefault(int(o), w)
    # ... and what was written for each orbital, word or not: one declaration
    # can cover orbitals written as different words, `px` on one and `py` on
    # its partner, and each row says its own
    written = {}
    for orbs, spec, _joint in raw:
        if isinstance(spec, _sitechars.DeclaredContent):
            for a, w in (spec.written or tuple((int(o), spec.spec) for o in spec.orbitals)):
                written.setdefault(int(a), w)
            continue
        for a in orbs:
            for o in atom_orbs[atom_of[int(a)]]:
                written.setdefault(int(o), spec)

    decls, placed, parts, declared_with = [], [], {}, {}

    def _bound(site, dim, named):
        """The measured multiplets at ``site`` a message offers, and the others it names.

        Returns ``(bound, own)``, each multiplet as its atoms: ``bound`` those
        of ``dim`` atoms holding an atom of ``named`` and not declared whole,
        ``own`` the other multiplets holding an atom of ``named``.
        """
        whole = {frozenset(int(o) for o in d["orbitals"]) for d in decls
                 if int(d["site"]) == int(site)}
        bound, own = [], []
        for m in layout.by_site[site]:
            ks = _atoms_named(tuple(int(o) for o in m))
            if not set(ks) & set(named):
                continue
            if len(ks) == int(dim) and frozenset(int(o) for o in m) not in whole:
                bound.append(ks)
            else:
                own.append(ks)
        return tuple(bound), tuple(own)

    def _one(rec, members, ks):
        base = rec["base"]
        content = _sitechars.DeclaredContent(
            orbitals=tuple(int(o) for o in members),
            spec=(base.spec if base is not None else rec["spec"]),
            label=str(rec["label"]), dim=int(rec["dim"]), declared_in=rec["read_where"],
            constraints={int(k): complex(v) for k, v in rec["cons"].items()},
            note=str(rec["note"]), broke=(base.broke if base is not None else ()),
            functions=tuple(rec["functions"]), reduced_in=str(rec["reduced_where"]),
            words=tuple((int(o), said[int(o)]) for o in members if int(o) in said),
            written=tuple((int(o), written[int(o)]) for o in members if int(o) in written),
            corep=bool(rec["corep"]), mulliken=tuple(rec["mulliken"]),
            matrices=dict(rec["matrices"]))
        decls.append({"orbitals": content.orbitals,
                      "channels": tuple(sorted({int(chan_of[int(o)])
                                                for o in content.orbitals})),
                      "site": rec["site"], "spec": content.spec, "label": content.label,
                      "dim": content.dim, "note": content.note,
                      "constraints": content.constraints, "content": content,
                      "joint": bool(rec["joint"]), "corep": content.corep,
                      "matrices": content.matrices})
        placed.append(tuple(ks))

    for orbs, spec, joint in raw:
        site = int(site_of[orbs[0]])
        keys = _atoms_named(orbs)
        # the grammar gives both members of a pair one word, so a repeat of the
        # same pair is the same declaration written twice (once per member)
        # rather than a second, independent one
        if paired and keys in placed:
            continue
        table = tables[site]
        where = f"orbital{'s' if len(orbs) > 1 else ''} {orbs if len(orbs) > 1 else orbs[0]}"
        cons, label, dim, note = _sitechars.resolve_spec(group, table, spec, where)
        base = spec if isinstance(spec, _sitechars.DeclaredContent) else None
        # a name the tables give a row Mulliken's convention calls by another:
        # the declaration's note names that row, once, where the name is read
        mulliken = (base.mulliken if base is not None
                    else _sitechars.mulliken_names(group, table, spec))
        if mulliken and base is None:
            note = (note + "; " if note else "") + _sitechars.mulliken_clause(mulliken)
        site_symbol = _sitechars.point_group_symbol(group, table.indices)
        here = f"{site_symbol} in {group.name} #{group.number_str}"
        # Where the declared word was read, which is not always where it
        # resolves: step 1 reads it in the cell's own site groups and may lower
        # the cell before anything is resolved, and the declaration keeps both,
        # as declared_in and reduced_in.
        first = (read_in or {}).get(tuple(sorted(int(a) for a in orbs)))
        read_where = (base.declared_in if base is not None else
                      (first if first is not None else here))
        reduced_where = (base.reduced_in if base is not None else
                         ("" if first is None or first == here else here))
        key = _sitechars.shorthand_key(spec)
        functions = ((base.functions if base is not None else ())
                     if key is None else (_sitechars.SHORTHANDS[key],))
        # a sum written in double-valued site irreps states their
        # multiplicities and nothing finer, which every comparison with the
        # Hamiltonian's blocks reads
        corep = (base.corep if base is not None
                 else _sitechars.named_in_coreps(table, spec))
        # generator matrices, on every operation the declared ones generate:
        # which basis of the declared site irreps the orbitals are
        here_inds = {int(i) for i in table.indices}
        matrices = ({int(k): M for k, M in base.matrices.items() if int(k) in here_inds}
                    if base is not None
                    else _sitechars.generator_matrices(group, table, spec, where))
        rec = {"site": site, "spec": spec, "label": label, "dim": dim, "cons": cons,
               "note": note, "base": base, "read_where": read_where,
               "reduced_where": reduced_where, "functions": functions, "joint": joint,
               "corep": corep, "mulliken": mulliken, "matrices": matrices}
        # every atom of the site written with this word, which a message counts
        said_with = declared_with.setdefault((site, _declared_character_key(cons)), [])
        said_with.extend(k for k in keys if k not in said_with)

        if joint and dim != len(keys):
            n = len(keys)
            noun = "Wannier function" if paired else "orbital"
            # a declaration in double-valued site irreps is counted in what
            # holds it, each orbital with both its spin states
            wide = (f"holds {dim} {noun}{'' if dim == 1 else 's'}" if paired or corep
                    else f"is {dim}-dimensional")
            # the spelling to offer: separate words for a declaration of
            # dimension one, and otherwise what the site holds says what fits
            bound, own = _bound(site, dim, keys)
            fix = ((". Write {" + ", ".join(f"{int(min(atom_orbs[k]))}: {label!r}" for k in keys)
                    + f"}} for {n} separate {label} {noun}s.")
                   if dim == 1 else
                   ". " + _multiplet_spelling(site, _site_atoms(atom_orbs, site), dim, spec,
                                              table, paired=paired, bound=bound, own=own))
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"{where}: {label!r} {wide}, and the key names "
                f"{n} {noun}{'' if n == 1 else 's'}" + fix,
                {"orbitals": orbs, "site": site, "declared_irrep": label, "dim": dim,
                 "multiplet": tuple(atom_orbs[k] for k in keys)})
        if matrices:
            # Matrices are read on whole multiplets, the orbitals the hoppings
            # bind together at the site.  Block-diagonal on several, they are
            # one declaration per multiplet.  A declaration carried down to a
            # smaller group, whose multiplets have come apart, is read so too.
            members = _members(keys)
            touched = sorted({int(chan_of[int(o)]) for o in members})
            bound = " and ".join(str(tuple(int(o) for o in chan_orbs[c])) for c in touched)
            context = {"orbitals": orbs, "site": site, "multiplets": tuple(
                tuple(int(o) for o in chan_orbs[c]) for c in touched)}
            if any(not set(int(o) for o in chan_orbs[c]) <= set(members) for c in touched):
                raise _sitechars.DeclarationError(
                    "ambiguous_multiplet",
                    f"{where}: the generator matrices name part of a multiplet the hoppings "
                    f"measure. The hoppings bind these orbitals as {bound}. Name every "
                    "orbital of a multiplet in one key.",
                    context)
            place = [[i for i, k in enumerate(keys) if int(chan_of[int(atom_orbs[k][0])]) == c]
                     for c in touched]
            if len(place) > 1:
                if any(float(np.abs(M[np.ix_(a, b)]).max()) > CHARACTER
                       for M in matrices.values() for a in place for b in place if a is not b):
                    raise _sitechars.DeclarationError(
                        "ambiguous_multiplet",
                        f"{where}: the generator matrices mix {bound}, which the hoppings "
                        "measure as separate multiplets. Give each multiplet its own key "
                        "and its own matrices.", context)
                for idx in place:
                    part = {int(i): M[np.ix_(idx, idx)] for i, M in matrices.items()}
                    chi = {int(i): complex(np.trace(M)) for i, M in part.items()}
                    whole = all(int(i) in chi for i in table.indices)
                    _one(dict(rec, cons=chi, dim=len(idx), matrices=part,
                              label=(_sitechars.content_label(
                                  table, [chi[int(cls[0])] for cls in table.classes])
                                  if whole else "")),
                         _members(tuple(keys[i] for i in idx)), tuple(keys[i] for i in idx))
                continue
        if dim == 1 and len(keys) > 1:
            for k in keys:                       # k copies of a 1D label = k declarations
                _one(rec, atom_orbs[k], (k,))
            continue
        if dim == len(keys):
            _one(rec, _members(keys), keys)
            continue
        if len(keys) == 1 and dim > 1:
            # a declaration wider than the one orbital it is written on, where
            # the Hamiltonian glued a multiplet of its dimension: the
            # declaration is about the whole multiplet measured there
            c = int(chan_of[int(orbs[0])])
            if int(layout._dim(c)) == int(dim):
                _one(rec, tuple(int(o) for o in chan_orbs[c]), keys)
                continue
        # Part of a declaration: the orbitals written with it at one site are
        # pooled, and add up, below, to the copies of it they hold together
        pool = parts.setdefault((site, _declared_character_key(cons)),
                                {"rec": rec, "keys": [], "orbs": []})
        for k in keys:
            if k not in pool["keys"]:
                pool["keys"].append(k)
        pool["orbs"].extend(int(a) for a in orbs)

    for (site, _chi), pool in parts.items():
        rec, keys = pool["rec"], tuple(pool["keys"])
        dim, table = int(rec["dim"]), tables[site]
        if len(keys) % dim:
            orbs = tuple(pool["orbs"])
            noun = "Wannier function" if paired else "orbital"
            wide = (f"holds {dim} {noun}{'' if dim == 1 else 's'}" if paired or rec["corep"]
                    else f"is {dim}-dimensional")
            left = len(keys) % dim
            # every atom of the site written with the word, those a multiplet of
            # its dimension took whole among them, is what the reason counts
            said_with = declared_with[(site, _chi)]
            if len(said_with) > len(keys) == left:
                named = tuple(sorted(int(o) for k in said_with for o in atom_orbs[k]))
                over = tuple(int(min(atom_orbs[k])) for k in keys)
                count = (f"and {len(said_with)} {noun}s of site {site} are declared with it, "
                         f"{noun}{'s' if len(over) > 1 else ''} "
                         f"{over[0] if len(over) == 1 else over} left over. ")
            else:
                named = orbs
                count = (f"and {left} {noun}{'s' if left > 1 else ''} of site {site} declared "
                         f"with it {'is' if left == 1 else 'are'} left over. ")
            bound, own = _bound(site, dim, keys)
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"orbital{'s' if len(named) > 1 else ''} "
                f"{named if len(named) > 1 else named[0]}: {rec['label']!r} {wide}, "
                + count
                + _multiplet_spelling(site, _site_atoms(atom_orbs, site), dim, rec["spec"],
                                      table, paired=paired, bound=bound, own=own),
                {"orbitals": orbs, "site": site, "declared_irrep": rec["label"],
                 "dim": dim,
                 "multiplet": tuple(atom_orbs[k] for k in keys)})
        copies = len(keys) // dim
        cons = {int(k): copies * complex(v) for k, v in rec["cons"].items()}
        label = rec["label"]
        if copies > 1:
            chi = [cons.get(int(cls[0]), 0.0) for cls in table.classes]
            named = _sitechars.corep_label if rec["corep"] else _sitechars.content_label
            label = named(table, chi) or f"{copies}x{label}"
        _one(dict(rec, cons=cons, label=label, dim=len(keys)), _members(keys), keys)
    # A site with declared matrices is read multiplet by multiplet: a
    # declaration there naming no whole measured multiplet would make its
    # declarations one (:func:`_site_reading`)
    measured = tuple(tuple(tuple(int(o) for o in m) for m in site) for site in layout.by_site)
    for site in sorted({int(d["site"]) for d in decls if d["matrices"]}):
        here = [d for d in decls if int(d["site"]) == site]
        named = {int(o) for d in here for o in d["orbitals"]}
        touched = [m for m in measured[site] if set(m) & named]
        context = {"site": site, "orbitals": tuple(sorted(named)),
                   "multiplets": measured[site]}
        bound = " and ".join(str(m) for m in measured[site])
        if {frozenset(d["orbitals"]) for d in here} != {frozenset(m) for m in touched}:
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"site {site}: generator matrices are declared here, and the declarations "
                f"at this site do not each name one whole multiplet the hoppings measure "
                f"there, {bound}. Give each multiplet its own key.", context)
        if any(d["corep"] for d in here) and len(measured[site]) > 1:
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"site {site}: generator matrices are declared here, and a declaration at "
                f"this site is a double-valued site irrep, where the hoppings measure "
                f"several multiplets, {bound}. Give each multiplet its own key, naming its "
                "orbital factor.", context)
    return decls


def _declared_character_key(cons):
    """A hashable key for the declared character ``cons``, ``{ind: chi}``.

    The key holds each operation's index with the real and imaginary parts of
    its character rounded to nine decimals, in the order of the indices.
    Declarations pooled at one site with equal keys are counted as parts of
    one declaration.
    """
    return tuple(sorted((int(k), round(complex(v).real, 9), round(complex(v).imag, 9))
                        for k, v in cons.items()))


def _site_atoms(atom_orbs, site) -> tuple:
    """The atoms of site ``site``, each given by its orbitals.

    ``atom_orbs`` maps each atom, keyed by its site and its lowest orbital, to
    its orbitals (:func:`_declaration_entries`).  An atom is one orbital, or a
    declared spin pair.  Returns the atoms in the order of their lowest
    orbitals.
    """
    return tuple(atom_orbs[k] for k in sorted(k for k in atom_orbs if int(k[0]) == int(site)))


def _multiplet_spelling(site, atoms, dim, spec, table, *, paired, bound=(), own=()) -> str:
    """The spelling an ``ambiguous_multiplet`` message suggests, as a sentence.

    ``site`` is the site's index and ``atoms`` its atoms, each given by its
    orbitals (:func:`_site_atoms`): its orbitals, or with ``paired`` its
    declared spin pairs, each one Wannier function.  ``spec`` is the word
    declared, ``dim`` its dimension counted in atoms, and ``table`` the site's
    character table.  ``bound`` holds the multiplets of ``dim`` atoms the
    hoppings measure at the site that hold an atom the declaration names and
    are not declared whole, and ``own`` the other measured multiplets holding
    an atom it names, each as the keys of its atoms.  Where ``bound`` holds
    one, the sentence names them and offers the joint form on the first.
    Otherwise, where the site holds fewer than ``dim`` atoms, it says what the
    site holds and lists the site irreps that fit it
    (:func:`.._sitechars.fitting_words`).  Otherwise it names each multiplet
    of ``own`` with its width and lists the site irreps that fit that width.
    Without ``own``, where ``dim > 1``, it offers the joint form on the
    site's first ``dim`` atoms, each named by its lowest orbital.
    """
    n, dim = len(atoms), int(dim)
    by_key = {(int(site), int(min(a))): a for a in atoms}
    noun = "Wannier function" if paired else "orbital"
    if bound and dim > 1:
        spelled = [tuple(int(min(by_key[k])) for k in m) for m in bound]
        held = [tuple(sorted(int(o) for k in m for o in by_key[k])) for m in bound]
        many = len(spelled) > 1
        return (f"The hoppings bind {' and '.join(str(m) for m in held)} as "
                f"{'multiplets' if many else 'a multiplet'} of its dimension. "
                f"{'Name each that holds it in its own key' if many else 'Name it in one key'}"
                f", as {{{spelled[0]}: {spec!r}}}.")
    if own and n >= dim:
        out = []
        for width in sorted({len(m) for m in own}):
            these = [m for m in own if len(m) == width]
            names = []
            for m in these:
                orbs = tuple(sorted(int(o) for k in m for o in by_key[k]))
                names.append(f"orbital {orbs[0]}" if len(orbs) == 1 else f"orbitals {orbs}")
            room = f"one {noun}" if width == 1 else f"{width} {noun}s"
            words = _sitechars.fitting_words(table, width)
            out.append(f"The hoppings measure {' and '.join(names)} as "
                       f"{'a multiplet' if len(these) == 1 else 'multiplets'} of {room}. "
                       f"Declare a word for {room} there"
                       + (f", such as {', '.join(words)}" if words else "") + ".")
        return " ".join(out)
    if n >= dim > 1:
        joint = tuple(int(min(a)) for a in atoms[:dim])
        return ("Name the orbitals that hold it together in one key, as "
                f"{{{joint}: {spec!r}}}.")
    if paired:
        held = (f"{n} Wannier function{'' if n == 1 else 's'}, the spin "
                f"pair{'' if n == 1 else 's'} " + " and ".join(str(tuple(a)) for a in atoms))
        wide = f"a multiplet of {dim} Wannier functions"
        room = "one Wannier function" if n == 1 else f"at most {n} Wannier functions"
    else:
        held = f"{n} orbital{'' if n == 1 else 's'} {tuple(int(a[0]) for a in atoms)}"
        wide = f"a multiplet of {dim} orbitals"
        room = "one orbital" if n == 1 else f"at most {n} orbitals"
    words = _sitechars.fitting_words(table, n)
    return (f"Site {site} holds {held}, too few for {wide}. Declare a word for "
            f"{room}" + (f", such as {', '.join(words)}" if words else "") + ".")


def _declared_blocks(decls, layout, assignment, tables, *, spin=None):
    """The declarations ``decls`` read on the multiplets the hoppings measure.

    ``decls`` are as :func:`_normalize_declaration` returns them, ``layout``
    holds each site's measured multiplets, ``assignment`` each site's orbitals,
    and ``tables`` the site character tables the declarations were read in, by
    site.  ``spin`` is the declared :class:`..spin.SpinBasis`, whose
    ``(up, dn)`` pairs are each one atom of two orbitals, or ``None``.  A site
    whose declarations are exactly the measured multiplets they touch keeps
    them, each read on its own multiplet.  At any other site, and at a site of
    several multiplets with a declaration in double-valued site irreps, the
    declarations become one, their sum, read on all of the site's multiplets
    (:func:`_site_reading`).  Whether the multiplets carry it is the verdict's
    comparison of characters (:func:`_verdicts_and_choices`).

    Returns the declarations in the order written, a site's one declaration in
    the place of its first.
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    pair = 1 if spin is None else 2
    measured = tuple(tuple(tuple(int(o) for o in ch) for ch in site) for site in layout.by_site)
    # each declaration once, and the same orbitals declared twice (a check on
    # itself, which is how a carried record spells a measured pair) are one
    # declaration, not two
    groups_of = {}
    for d in decls:
        groups_of.setdefault(int(d["site"]), {})[tuple(int(o) for o in d["orbitals"])] = d
    merged = {}
    for s in range(len(site_orbs)):
        merged.update(_site_reading(site_orbs[s], measured[s], groups_of.get(s, {}),
                                    tables.get(s), pair))
    out, placed = [], set()
    for d in decls:
        one = merged.get((int(d["site"]), tuple(int(o) for o in d["orbitals"])))
        if one is None:
            out.append(d)
        elif id(one) not in placed:
            placed.add(id(one))
            out.append(one)
    return out


def _site_reading(orbitals, measured, groups, table, pair):
    """One site's declarations read on its measured multiplets.

    ``orbitals`` are the site's orbitals, ``measured`` its measured multiplets,
    each given by its orbitals, and ``groups`` its declarations keyed by their
    orbitals.  ``table`` is the site's character table and ``pair`` the number
    of orbitals in one atom.  Where the declarations are exactly the measured
    multiplets they touch, and none is in double-valued site irreps at a site
    of several multiplets, each stays on its own multiplet and ``{}`` is
    returned.  Otherwise returns ``{(site, orbitals): declaration}``, every
    declaration's orbitals keyed to the one declaration they add up to, read
    on all of the site's multiplets (:func:`_block_content`).
    """
    if not groups:
        return {}
    named = {int(o) for g in groups for o in g}
    touched = [m for m in measured if set(m) & named]
    corep = any(groups[g].get("corep") for g in groups)
    if ({frozenset(g) for g in groups} == {frozenset(m) for m in touched}
            and not (corep and len(measured) > 1)):
        return {}
    one = _block_content([groups[g] for g in groups], list(measured), table, pair)
    if one is None:                       # pragma: no cover - a guard, not a path
        raise InternalError(
            f"the declarations at the site of orbitals {tuple(sorted(orbitals))} do not add "
            "up to site irreps as wide as the orbitals they name. Report it, with the "
            "model and the call.")
    site = int(next(iter(groups.values()))["site"])
    return {(site, tuple(int(o) for o in g)): one for g in groups}


def _block_content(groups, multiplets, table, pair):
    """A site's declarations as one, their sum, read on its measured multiplets.

    ``groups`` are the site's declarations, as :func:`_normalize_declaration`
    returns them, ``multiplets`` its measured multiplets, ``table`` its
    character table and ``pair`` the number of orbitals in one atom.  Each
    declaration is decomposed into the site irreps of ``table``, and their sum
    is what was declared.  Where the sum is narrower than the multiplets, it is
    part of what they hold.  Returns a declaration of the same form, without
    ``"matrices"``, with ``"partial"`` set where it is part, ``"orbitals"`` the
    orbitals named where it is part and every orbital of the multiplets
    otherwise, ``"block"`` every orbital of the multiplets, ``"groups"`` the
    declarations summed, ``"joint"`` true, and ``"channels"`` empty until the
    multiplets of ``"block"`` are written there.  Returns ``None`` where
    ``table`` is ``None``, where a declaration lacks a character on a class, is
    no sum of site irreps with nonnegative integer multiplicities, or is not as
    wide as the atoms it names, and where the sum is wider than the multiplets.
    """
    if table is None:
        return None
    counts = {}
    for d in groups:
        chi = np.asarray([complex(d["constraints"].get(int(cls[0]), np.nan))
                          for cls in table.classes])
        if not np.all(np.isfinite(chi)):
            return None
        width = 0
        for p, n in enumerate(_sitechars.decompose(table, chi)):
            n = complex(n)
            if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or n.real < -0.5:
                return None
            if round(n.real):
                counts[p] = counts.get(p, 0) + int(round(n.real))
                width += int(round(n.real)) * int(round(float(np.real(table.dims[p]))))
        if width != len(d["orbitals"]) // pair:
            return None
    declared = sum(int(round(float(np.real(table.dims[p])))) * n for p, n in counts.items())
    total = sum(len(m) // pair for m in multiplets)
    if declared > total:
        return None
    partial = declared < total
    corep = any(bool(d.get("corep")) for d in groups)

    orbitals = tuple(o for m in multiplets for o in m)
    named = tuple(int(o) for d in groups for o in d["orbitals"])
    chi = sum(n * np.asarray(table.characters[p], dtype=complex) for p, n in counts.items())
    label = _sitechars.content_label(table, chi)
    specs = tuple(dict.fromkeys(d["spec"] for d in groups
                                if not isinstance(d["spec"], dict)))
    cons = {int(ind): complex(sum(complex(d["constraints"][int(ind)]) for d in groups))
            for ind in table.indices}
    first = groups[0]["content"]
    content = _sitechars.DeclaredContent(
        orbitals=(named if partial else orbitals),
        spec=(specs[0] if len(specs) == 1 else (specs or groups[0]["spec"])),
        label=label, dim=int(declared), declared_in=first.declared_in, constraints=cons,
        note="; ".join(dict.fromkeys(str(d["note"]) for d in groups if d["note"])),
        broke=tuple(dict.fromkeys(b for d in groups for b in d["content"].broke)),
        functions=tuple(f for d in groups for f in d["content"].functions),
        reduced_in=first.reduced_in,
        words=tuple(w for d in groups for w in d["content"].words),
        written=tuple(w for d in groups for w in d["content"].written),
        corep=corep,
        mulliken=tuple(dict.fromkeys(t for d in groups for t in d["content"].mulliken)))
    return {"orbitals": content.orbitals, "block": orbitals, "channels": (),
            "site": int(groups[0]["site"]), "spec": content.spec, "label": label,
            "dim": int(declared), "note": content.note, "constraints": cons,
            "content": content, "joint": True, "partial": bool(partial),
            "groups": tuple(groups), "corep": content.corep}


def _mismatch_named(decls, rows, admissible, site_tables, *, ref_orb=0):
    """The lines and context naming the declaration a refusal is to act on.

    The refusal is ``declaration_breaks_group``.  ``decls`` are the
    declarations (:func:`_normalize_declaration`), ``rows`` the
    measured-versus-declared rows (:class:`_Row`), ``admissible`` the
    one-dimensional characters of the unitary operations, ``{ind: value}``
    each, and ``site_tables`` the site character tables, by site.
    ``ref_orb`` is the reference orbital.  The declarations are read in the
    order the lowest-angular-momentum rule reads the crystal: the reference
    orbital's own first where it has one, then the others as written.  The
    one named is the first that no character of ``admissible`` meets
    together with those before it in its coupled part.  So which declaration
    is named does not depend on where the reference orbital's own was written.

    Returns ``{}`` where every declaration is met.  Otherwise returns
    ``{"verdict": text, "named": context}``, the lines and the context
    :func:`_name_the_mismatch` makes, the context with the failing
    declaration's ``orbitals``, ``declared_as`` and ``declared_irrep`` added.
    """
    def _needs(d):
        """What the rows of the declaration ``d`` ask of its part's character.

        Returns ``{ind: value}``: each row's required value, and ``None`` for a
        row that has none and whose declared character is not zero.
        """
        out = {}
        for r in rows:
            if tuple(r.orbitals) != tuple(d["orbitals"]):
                continue
            if r.required is not None:
                out[int(r.ind)] = complex(r.required)
            elif abs(complex(r.declared)) > 1e-6:
                out[int(r.ind)] = None
        return out

    part_of = {tuple(r.orbitals): int(r.part) for r in rows}
    first = _reference_declaration(decls, ref_orb)
    order = ([first] if first is not None else []) + [d for d in decls if d is not first]
    failed, still_free, left = None, None, {}
    for d in order:
        if tuple(d["orbitals"]) not in part_of:
            continue                      # nothing was measured about it
        part = part_of[tuple(d["orbitals"])]
        survivors = left.get(part, list(admissible))
        needs = _needs(d)
        keep = [chi for chi in survivors if _meets(chi, needs)]
        if not keep:
            failed, still_free = d, survivors
            break
        left[part] = keep
    if failed is None:
        return {}
    measured = {}
    for d in decls:
        measured[tuple(d["orbitals"])] = {
            int(r.ind): complex(r.measured) for r in rows
            if tuple(r.orbitals) == tuple(d["orbitals"])}
    verdict, named = _name_the_mismatch(failed, measured, still_free, decls, site_tables)
    return {"verdict": verdict,
            "named": dict(named, orbitals=tuple(failed["orbitals"]),
                          declared_as=failed["spec"], declared_irrep=failed["label"])}


def _name_the_mismatch(failed, measured, still_free, decls, site_tables, tol=1e-6):
    """The lines naming what a ``declaration_breaks_group`` measured.

    ``failed`` is the declaration named (:func:`_mismatch_named`), ``measured``
    the character measured on each declaration's orbitals,
    ``{orbitals: {ind: value}}``, and ``still_free`` the one-dimensional
    characters its coupled part could still carry when it was read.
    ``decls`` holds every declaration in the order written, ``site_tables``
    the site character tables by site, and ``tol`` the size below which a
    declared character counts as zero.  Two names are read off the site table
    of ``failed``.  The measured irrep is the measured character times each
    character of ``still_free``, named as the record returns labels,
    double-valued where the states carry spin; where the names differ, all
    are listed.  The relative character, measured over declared at each
    operation, is the one-dimensional character the declaration would need,
    named in the orbital-factor spelling (:func:`_label_character`); for a
    declaration in double-valued site irreps it is read only where a
    character was measured.  Where declared matrices are not carried but
    their irrep is, the lines say so.  Where the measured and the declared
    double-valued names are one, the lines name the two orbital factors.

    Returns ``(text, context)``: the lines, and for the refusal's context a
    mapping of ``measured_irrep``, ``relative_character`` and
    ``declared_irreps`` (each declaration's orbitals with its label).
    """
    tab = site_tables.get(int(failed["site"]))
    theta = measured[tuple(failed["orbitals"])]

    meas, rels, meas_orb = [], [], []
    for chi in (still_free or [None]):
        gauged = (dict(theta) if chi is None
                  else {ind: complex(chi.get(int(ind), 1.0)) * v for ind, v in theta.items()})
        lab = _label_character(tab, gauged)
        if lab and lab not in meas:
            meas.append(lab)
        lab = _label_character(tab, gauged, orbital=True)
        if lab and lab not in meas_orb:
            meas_orb.append(lab)
        # a declaration in double-valued site irreps states nothing where the
        # spin-1/2 lift has zero trace, and nothing was measured there
        rel = {ind: complex(gauged.get(int(ind), 0.0)) / complex(want)
               for ind, want in failed["constraints"].items() if abs(complex(want)) > tol
               and (not failed.get("corep") or int(ind) in gauged)}
        lab = _label_character(tab, rel, orbital=True)
        if lab and lab not in rels:
            rels.append(lab)
    measured_irrep = " or ".join(meas)
    rel_label = " or ".join(rels)

    def _declared(d):
        """The name of the declaration ``d``, read off its characters.

        It is named against its site's table as the record returns labels, a
        sum by its terms; where that names nothing, by its label, else by the
        word written.
        """
        return (_label_character(site_tables.get(int(d["site"])), d["constraints"],
                                 sums=True) or d["label"] or d["spec"])

    at = next((i for i, d in enumerate(decls) if d is failed), len(decls))
    said = ", ".join(f"{tuple(d['orbitals'])}={_declared(d)}" for d in decls[:at])
    lines = []
    if said:
        lines.append(f"  declared {said}, {tuple(failed['orbitals'])}="
                     f"{_declared(failed)}")
    wanted = _declared(failed)
    # with spin, two orbital factors can give one double-valued name
    # (``A1`` and ``A2`` of ``3m`` both give ``-E1``): the line then names them
    wanted_orb = ("" if failed.get("corep") else
                  _label_character(tab, failed["constraints"], orbital=True))
    if failed.get("matrices") and measured_irrep and measured_irrep == wanted:
        # declared matrices whose traces the hoppings carry, in another basis
        lines.append(
            f"  measured: orbital(s) {tuple(failed['orbitals'])} carry {measured_irrep}, "
            "as declared, in another basis than the declared matrices.")
    elif (measured_irrep and measured_irrep == wanted and wanted_orb and meas_orb
          and wanted_orb not in meas_orb):
        lines.append(
            f"  measured: orbital(s) {tuple(failed['orbitals'])} carry the orbital factor "
            f"{' or '.join(meas_orb)}, not the declared {wanted_orb}, both {wanted} with "
            "spin" + (f"; relative character {rel_label}." if rel_label else "."))
    else:
        lines.append(
            f"  measured: orbital(s) {tuple(failed['orbitals'])} carry "
            f"{measured_irrep or 'a character matching no single irrep of this site group'}"
            f", not the declared {wanted or 'character'}"
            + (f"; relative character {rel_label}." if rel_label else
               "; the measured character over the declared one is no character of this "
               "site group."))
    return "\n".join(lines), {"measured_irrep": measured_irrep,
                              "relative_character": rel_label,
                              "declared_irreps": tuple(
                                  (tuple(d["orbitals"]), d["label"]) for d in decls)}


def _label_character(table, chi, tol=1e-6, *, sums=False, orbital=False):
    """The name of the character ``chi`` in ``table``, or ``""`` where none.

    It is the label :func:`_named_content` returns for the same ``table``,
    ``chi``, ``tol``, ``sums`` and ``orbital``, without where the name comes
    from.  ``sums`` is off by default: a relative character is
    one-dimensional, and no sum names one.
    """
    return _named_content(table, chi, tol, sums=sums, orbital=orbital)[0]


#: How directly a row's name comes off the tables, most direct first.  A sum of
#: rows is named as directly as its least direct part.
_NAMED_BY_RANK = {"tables": 0, "invariants": 1, "point group": 2, "Mulliken": 3}


def _named_content(table, chi, tol=1e-6, *, sums=True, orbital=False):
    """The name of the site irreps ``chi`` holds, and where it comes from.

    ``table`` is a site character table (:class:`.._sitechars.SiteGroupTable`)
    and ``chi`` a character, ``{ind: value}``.  A row of ``table`` that matches
    ``chi`` within ``tol`` on every operation ``chi`` gives is returned with
    its own name and its ``named_by``.  ``sums`` says whether a reducible
    character, given on every operation of the table, is named by the site
    irreps it decomposes into.  The terms are counted across rows, since a
    row's own name can be a sum on a spinful site, so a ``p`` shell with spin
    at a ``3m`` site is ``2x-E1 + -1E2E``.  A sum is named as directly as its
    least direct part (:data:`_NAMED_BY_RANK`).  ``orbital`` asks for a row's
    orbital-factor spelling where it has one
    (:attr:`.._sitechars.SiteGroupTable.declarable`), for a one-dimensional
    character that is not a site representation, such as a measured character
    over a declared one.

    Returns ``(label, named_by)``.  Returns ``("", "unnamed")`` where
    ``table`` is ``None``, where the row matched or a term of the sum has no
    name, and where no row matches and ``chi`` is not named as a sum: with
    ``sums`` off, with an operation of the table missing from ``chi``, or
    where ``chi`` is no sum of site irreps with nonnegative integer
    multiplicities.
    """
    if table is None:
        return "", "unnamed"

    def name(p):
        if orbital and p < len(table.declarable) and table.declarable[p]:
            return table.declarable[p]
        return table.labels[p]

    def where(p):
        return str(table.named_by[p]) if p < len(table.named_by) else "unnamed"

    for p in range(len(table.labels)):
        if all(abs(table.character_on(p, ind) - complex(chi.get(int(ind), 0.0))) <= tol
               for ind in table.indices if int(ind) in chi):
            return (name(p), where(p)) if name(p) else ("", "unnamed")
    if not sums or any(int(ind) not in chi for ind in table.indices):
        return "", "unnamed"
    mult = _sitechars.decompose(
        table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    counts, how = {}, set()
    for p in range(len(mult)):
        n = complex(mult[p])
        if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or round(n.real) < 0:
            return "", "unnamed"
        if round(n.real) > 0:
            if not name(p):
                # one of the parts has no name; naming the sum would claim one
                return "", "unnamed"
            for term, k in _label_terms(name(p)):
                counts[term] = counts.get(term, 0) + int(round(n.real)) * k
            how.add(where(p))
    if not counts:
        return "", "unnamed"
    label = " + ".join(term if k == 1 else f"{k}x{term}" for term, k in counts.items())
    return label, max(how, key=lambda w: _NAMED_BY_RANK.get(w, len(_NAMED_BY_RANK)))


def _copies_of_one_irrep(table, chi, tol=1e-6):
    """The character ``chi`` as ``m`` copies of one site irrep, split.

    ``table`` is the site character table and ``chi`` a character
    ``{ind: value}``; ``tol`` bounds a multiplicity read as zero and a
    multiplicity's distance from an integer.  Where ``chi`` is ``m`` copies of
    one site irrep, ``m`` at least 2, returns ``(chi / m, m)``, so that each
    orbital of a multiplet holding copies of one irrep is labeled with that
    irrep.  Returns ``(chi, 1)`` otherwise, and where ``table`` is ``None`` or
    ``chi`` lacks a class.
    """
    if table is None or any(int(cls[0]) not in chi for cls in table.classes):
        return chi, 1
    mult = _sitechars.decompose(
        table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    held = [complex(n) for n in mult if abs(complex(n)) > tol]
    if len(held) != 1:
        return chi, 1
    n = held[0]
    if abs(n.imag) > tol or abs(n.real - round(n.real)) > tol or round(n.real) < 2:
        return chi, 1
    m = int(round(n.real))
    return {int(ind): complex(v) / m for ind, v in chi.items()}, m


def _one_copy(table, chi, k, tol=CHARACTER):
    """``chi`` divided by ``k``, where it is ``k`` copies of one character.

    ``table`` is the site character table and ``chi`` a character
    ``{ind: value}``.  Returns ``chi / k`` where every site irrep's
    multiplicity in ``chi`` is a multiple of ``k`` within ``tol``, and
    ``None`` otherwise.  Where ``table`` is ``None`` or ``chi`` lacks a class,
    returns ``chi / k`` unchecked.
    """
    if table is None or any(int(cls[0]) not in chi for cls in table.classes):
        return {int(ind): complex(v) / k for ind, v in chi.items()}
    mult = _sitechars.decompose(
        table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    for n in mult:
        n = complex(n) / k
        if abs(n.imag) > tol or abs(n.real - round(n.real)) > tol:
            return None
    return {int(ind): complex(v) / k for ind, v in chi.items()}


def _distance_from_trivial(chi):
    """A sort key putting one-dimensional characters nearest the trivial first.

    ``chi`` is a character, ``{ind: value}``.  The key is first the number of
    operations on which ``chi`` is not ``+1``.  Then, operation by operation
    in index order, the larger real part comes first, and then the larger
    imaginary part.  So on a fourfold axis of a group with no antiunitary
    operation, ``+i`` at ``4+`` comes before ``-i``.  It is the last key of
    the labeling's choice (:func:`.._labeling._labeling`), read only where
    the keys before it tie.
    """
    vals = [complex(v) for _, v in sorted(chi.items())]
    return (sum(1 for v in vals if abs(v - 1.0) > 1e-9),
            tuple((-round(float(v.real), 9), -round(float(v.imag), 9)) for v in vals))


def _meets(chi, needs, *, over=None, tol=CHARACTER):
    """Whether the character ``chi`` meets what declarations ask of it.

    ``chi`` is a one-dimensional character, ``{ind: value}``, an operation it
    does not name taking ``1``.  ``needs`` maps an operation's index to the
    value the coupled part's character must take there for the Hamiltonian to
    carry the declaration, ``None`` where no value does; an operation it does
    not name asks nothing.  ``over``, where given, holds the indices the test
    is read on alone: a subgroup, or the operations a declaration was measured
    at.  ``tol`` is the largest difference from a value that meets it.  Every
    reading of a declaration comes to this test.
    """
    for ind, want in needs.items():
        if over is not None and int(ind) not in over:
            continue
        if want is None or abs(complex(chi.get(int(ind), 1.0)) - complex(want)) > tol:
            return False
    return True


def _declared_needs(d, raw, lift_trace, *, tol=1e-6):
    """What the declaration ``d`` asks of its coupled part's character.

    ``raw`` holds each multiplet's orbital character, ``{ind: value}`` by
    multiplet (:func:`_multiplet_characters`), with or without a candidate
    character of its part multiplied in, and ``lift_trace`` the trace of each
    operation's spin-1/2 lift.  The declared multiplets carry the declaration
    under a character ``chi`` where ``chi(g)`` times the sum of their
    characters is the declared character.  Returns ``{ind: value}``, the
    declared character over that sum, or ``None`` where the sum is below
    ``tol`` and the declared character is not.  An operation is left out
    where both are below ``tol``, where a multiplet has no character there,
    as under an operation that moves it, and, for a declaration in
    double-valued site irreps, where the spin-1/2 lift has zero trace.
    """
    out = {}
    for ind, want in d["constraints"].items():
        ind = int(ind)
        if d.get("corep") and abs(complex(lift_trace.get(ind, 1.0))) <= 1e-9:
            continue
        got = [raw[int(c)].get(ind) for c in d["channels"]]
        if any(v is None for v in got):
            continue
        theta = complex(sum(got))
        if abs(theta) > tol:
            out[ind] = complex(want) / theta
        elif abs(complex(want)) > tol:
            out[ind] = None
    return out


def _holds_part(d, assigned, raw, site_tables, lift_trace, *, unread=False):
    """Whether the multiplets of a declaration on part of a site hold it.

    ``d`` is a declaration on part of what its site holds
    (:func:`_block_content`).  ``raw`` holds each multiplet's character,
    ``{ind: value}`` by multiplet, and ``assigned(c)`` the one-dimensional
    character multiplet ``c``'s coupled part carries, so that the multiplets
    of several parts are read together.  ``site_tables`` gives the site
    character tables by site, and ``lift_trace`` the trace of each
    operation's spin-1/2 lift.  The multiplets hold the declaration where
    what they carry contains it (:func:`.._naming._contained`).  A
    declaration in double-valued site irreps is compared in them: ``2x-E1``
    is part of the ``2x-E1 + -1E2E`` of a ``p`` shell with spin at a ``3m``
    site, although no sum of single-valued site irreps lifting to it is part
    of the shell's ``A1 + E``.
    Returns ``unread`` where a class of the site group has no operation both
    declared and measured on every multiplet, and ``False`` where the site
    has no table.
    """
    tab = site_tables.get(int(d["site"]))
    if tab is None:
        return False
    held, want, asked = [], [], []
    for cls in tab.classes:
        ind = next((int(g) for g in cls if int(g) in d["constraints"]
                    and all(int(g) in raw[int(c)] for c in d["channels"])), None)
        if ind is None:
            return unread
        held.append(sum(complex(assigned(int(c)).get(ind, 1.0)) * complex(raw[int(c)][ind])
                        for c in d["channels"]))
        want.append(complex(d["constraints"][ind]))
        asked.append(abs(complex(lift_trace.get(ind, 1.0))) > ZERO)
    return _contained(tab, held, want, corep=bool(d.get("corep")), asked=asked)


def _joint_holds(j, pick, inside, *, tol=CHARACTER):
    """Whether a declaration read together is carried under ``pick``.

    ``j`` is a declaration read over several multiplets at once
    (:class:`_ReadTogether`).  ``pick`` gives each coupled part it spans a
    one-dimensional character, ``{ind: value}`` by part, ``inside`` holds the
    indices of the subgroup asked about, and ``tol`` is as for
    :func:`_meets`.  A declaration on the whole of its site is carried where
    its multiplets together carry exactly what it declares.  One on part of
    its site is carried where what they carry contains it, on the site group
    within ``inside`` (:func:`_holds_part`); a class of that site group
    nothing measured does not count against it.
    """
    d = j.decl
    raw = {int(c): {int(g): v for g, v in row.items() if int(g) in inside}
           for c, row in j.theta.items()}

    def assigned(c):
        return pick[j.part[int(c)]]

    if not d.get("partial"):
        carried = {c: {g: complex(assigned(c).get(g, 1.0)) * complex(v) for g, v in row.items()}
                   for c, row in raw.items()}
        return _meets({}, _declared_needs(d, carried, j.lift_trace), tol=tol)
    table = j.table_of(tuple(int(g) for g in j.site_group if int(g) in inside))
    return _holds_part(d, assigned, raw, {int(d["site"]): table}, j.lift_trace,
                       unread=True)


def _multiplet_characters(D, syms, u_pos, cperm, layout, prod, spf):
    """Each multiplet's orbital character on ``D``, and the reading's defect.

    ``D``, ``syms``, ``cperm`` and ``layout`` are as for
    :func:`_measure_channel_characters`, which reads each multiplet, and
    ``u_pos`` is its ``kept_u``, the positions of the unitary operations
    read; ``prod`` is the product table of ``syms`` and ``spf`` its spin
    signs, by position (``_get_product_table``).  Returns
    ``(raw, residual)``: ``raw[c]`` is multiplet ``c``'s character over the
    operations that fix it, ``{ind: value}``, and ``residual`` the largest
    defect of those readings.  On a measured multiplet the residual also
    holds how far its blocks are from multiplying as the operations do, the
    product of two blocks set against the block of the product times the
    sign ``spf`` the spin-1/2 lifts carry in it, by position.
    """
    raw, worst = {}, 0.0
    chan_states = layout._chan_states
    for c in range(len(chan_states)):
        chi, res = _measure_channel_characters(D, syms, u_pos, cperm, layout, c)
        raw[c] = chi
        worst = max(worst, res)
    if layout._measured:
        pos_of_ind = {int(s.ind): p for p, s in enumerate(syms)}
        for c in range(len(chan_states)):
            if layout._chan_kind[c] != "measured":
                continue
            st = np.asarray(chan_states[c], dtype=int)
            block = {ind: D[pos_of_ind[int(ind)]][np.ix_(st, st)] for ind in raw[c]}
            for a_ind, A in block.items():
                for b_ind, B in block.items():
                    pa, pb = pos_of_ind[a_ind], pos_of_ind[b_ind]
                    ab = int(syms[int(prod[pa, pb])].ind)
                    if ab in block:
                        worst = max(worst, float(np.abs(
                            A @ B - int(spf[pa, pb]) * block[ab]).max()))
    return raw, worst


def _lead_of(chans, first):
    """The lead multiplet of the coupled part whose multiplets are ``chans``.

    It is ``first`` where the part holds it, and the part's lowest-numbered
    multiplet otherwise.
    """
    return int(first) if int(first) in {int(c) for c in chans} else int(min(chans))


def _returned_rows(rows):
    """The measured-versus-declared rows ``rows`` as a record returns them.

    ``rows`` holds :class:`_Row` records.  Returns one mapping a row, in
    order, each holding its ``orbitals``, ``ind``, ``name``, ``measured`` and
    ``declared``; a row's ``part``, ``required`` and ``matrices`` are not
    returned.
    """
    return tuple({"orbitals": r.orbitals, "ind": r.ind, "name": r.name,
                  "measured": r.measured, "declared": r.declared} for r in rows)
