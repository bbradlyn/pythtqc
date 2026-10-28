"""What a declaration states, read on the multiplets the Hamiltonian measures:
each site's declared content in its site group, the operations that carry it,
one verdict per operation with the scalars it pins, and the refusal where the
declared content breaks the group.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import _sectors
from . import _sitechars
from .refusal import InternalError, Refusal
from ._tolerance import CHARACTER, IN_SPACE, SPREAD, ZERO, _IN_BLOCKS, noise_floor
from ._text import _counted_as, _fmt_char, _label_terms
from ._seitz import _record_name
from ._subgroups import (
    _descent_closure, _resolve_closure, _subgroups_inside, _tie_ranked, _tie_reason,
    _tie_rule_words)
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


def _covariant_subgroup(group, kept_u, requirements, *, unitary_only=False, tol=CHARACTER,
                        joint=(), search=None, matrices=()):
    r"""The largest subgroup the declared orbital content is covariant on.

    Read off the sectors of the Hamiltonian's conserved matrices wherever they
    read the round (``search``, :class:`.._sectors.SectorSearch`): a subgroup
    carries the declaration where one representation per orbit of sectors under
    **that subgroup**, of those the hoppings leave open, gives every declared
    site its content (:func:`_subgroup_read`).  A smaller group splits the
    orbits by itself -- three uncoupled lines a threefold axis exchanges are one
    orbit of ``6mm`` and two of ``mm2``, each with a character of its own -- so
    nothing is ruled out on the larger group's orbits before the smaller one is
    asked.  Where the sectors read nothing the requirements below decide.
    Declared generator matrices are a statement the sectors' reading, of
    contents, does not hear: ``matrices`` holds what they ask, one mapping per
    part as ``requirements`` holds it, and where a subgroup the sectors keep
    has no character meeting it, the requirements decide instead, the
    matrices' among them.

    The hoppings fix every multiplet's block up to one one-dimensional
    character per **coupled part** of the crystal — the multiplets they
    connect, directly or through others — so each part's declarations are
    read against a scale of its own and ask for a character of their own.
    ``requirements`` holds what the measurement left, one mapping per part,
    with one entry per unitary operation it could speak about: ``None`` where
    the operation is *individually* disallowed — two declared multiplets of
    the part it fixes need different scalars, or the scalar one of them needs
    is not ``±1`` — and otherwise the scalar ``λ(g)`` the part's character has
    to take there.  An operation absent from a mapping constrains nothing
    there.  A single mapping is one part.

    Every declaration is such a constraint, wherever it sits: an undeclared
    orbital, the reference orbital included, imposes none.  The declared content
    is then a representation covariant with ``H`` on a subgroup ``S`` exactly
    when every part has a one-dimensional **real** character of ``S`` meeting
    all of its requirements inside it, and the returned representation on each
    part is ``λ·D|_S``; which of the characters left the crystal is assigned is
    the lowest-angular-momentum rule's to choose, and the build chooses it once.
    Those subgroups are closed under intersection but not under join, so the
    largest is not always unique; where several of the largest order tie, each
    a consistent reading on its own, the one returned is the first by the tie
    rule every such choice is made by (:func:`.._subgroups._tie_ranked`) — never
    their intersection, which would keep less than any of them carries — and
    all of them are recorded in the rule's order, because which readings were
    possible is the physics of what was declared.  The floor is the identity,
    which is always consistent: a descent by declaration never refuses.

    ``unitary_only`` is a group with no antiunitary operation represented,
    where the residual freedom is larger than the real characters: **every**
    one-dimensional character of a subgroup is a candidate labelling there —
    roots of unity per generator — so those are what is searched, and a declared
    content met only by a complex character is **returned** with it rather than
    descended on.  Nothing in such a group's measurement forbids one: it
    preserves the covariance, the canonical norm and the whole product table.

    ``joint`` holds the contents read together (:func:`_verdicts_and_choices`):
    one spread over several coupled parts, or one stating part of what its
    site holds.  Each asks something of several parts at once, so a subgroup
    keeps it only where one character of the subgroup for each part it spans
    carries what was declared: the same content where the whole site was
    declared, and a content containing it where part of the site was
    (:func:`_joint_holds`).

    ``tol`` is how far a character may sit from the value a requirement asks
    and still meet it (:func:`_meets`): the requirements are read off the
    solution spaces, so it is the declared tolerance where that is larger than
    its exact default.

    Returns ``{"subgroup", "candidates", "broken"}`` — the returned set of
    ``.ind``, the maximal candidates in the rule's order where several tie, the
    one returned first, and the operations of ``kept_u`` outside ``S`` — or the
    :class:`..refusal.Refusal` of a search past its ceiling.
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
        if got is not None and _carries_matrices(got["subgroup"]):
            return got
    parts = ([requirements] if isinstance(requirements, dict)
             else [dict(part) for part in requirements])
    joint = tuple(joint or ())
    keep = {int(i) for i in kept_u}

    def _barred(i):
        return any(part.get(int(i), "free") is None for part in parts)

    # WHAT THE GROUP WOULD BE WITHOUT THE DECLARATION.  The survivors of a
    # measurement need not compose -- a sigma_h fiber partner whose lift
    # coincides with a kept operation's survives beside it, and the closure step
    # is what removes those -- so the question is asked inside the group they
    # DO form, and such a fiber partner is neither kept by the declaration nor
    # blamed on it: it leaves by the closure resolution, which is its own verdict.
    plain = _subgroups_inside(group, keep)
    widest = sorted((s for s in plain if len(s) == max(len(t) for t in plain)),
                    key=sorted) if plain else []
    if len(widest) != 1:
        # THE SURVIVORS ARE NOT A GROUP AND NO LARGEST SUBGROUP OF THEM IS
        # UNIQUE.  A declaration is read on a *group*, and there is not one here
        # yet -- the closure resolution speaks first, and where the candidates
        # tie it keeps the one the tie rule ranks first.  So only the operations
        # the declaration disallows *individually* are its own, and the
        # character question is asked again on the next round, on whatever
        # group that resolution lands on.
        bad = {i for i in keep if _barred(i)}
        return {"subgroup": frozenset(keep - bad), "candidates": (),
                "broken": tuple(sorted(bad))}
    baseline = set(widest[0])
    allowed = {i for i in baseline if not _barred(i)}

    def _met(sub):
        """Whether every coupled part has a character of ``sub`` meeting its requirements.

        And, where contents read together speak inside ``sub``, whether one
        character of each part they span meets those and its own part's at once.
        """
        inside = {int(i) for i in sub}
        asked = [i for i, wants in enumerate(parts) if any(int(g) in inside for g in wants)]
        together = [j for j in joint
                    if any(int(g) in inside for row in j["theta"].values() for g in row)]
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
        spanned = sorted({int(i) for j in together for i in j["part"].values()})
        options = [own.get(i, chars) for i in spanned]
        for combo in itertools.product(*options):
            pick = dict(zip(spanned, combo))
            if all(_joint_holds(j, pick, inside, tol=tol) for j in together):
                return True
        return False

    consistent = {frozenset(int(i) for i in sub)
                  for sub in _subgroups_inside(group, allowed) if _met(sub)}
    if not consistent:                   # pragma: no cover - {E} always is
        return {"subgroup": frozenset(), "candidates": (),
                "broken": tuple(sorted(keep))}
    # The LARGEST consistent subgroup, and the tie rule only where the largest
    # is not unique.  Maximality by inclusion alone would let a small subgroup
    # nothing constrains -- the sigma_h fiber partners that survive beside a kept
    # operation, which the closure step removes for exactly this reason -- stand
    # beside the group the declaration reaches as a candidate.  A tie needs two consistent subgroups
    # of the SAME size, which is the case the record names.
    top = max(len(s) for s in consistent)
    ranked = _tie_ranked(group, [s for s in consistent if len(s) == top])
    best = frozenset(ranked[0])
    candidates = ranked if len(ranked) > 1 else ()
    return {"subgroup": best, "candidates": candidates,
            "broken": tuple(sorted(i for i in baseline if i not in best))}


def _cyclic(group, ind):
    """The operations one unitary operation generates, by ``.ind``."""
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    ind_of = [int(s.ind) for s in group.symmetries]
    out, x = {int(ind)}, int(ind)
    while True:
        x = ind_of[int(prod[at[int(ind)], at[x]])]
        if x in out:
            return frozenset(out)
        out.add(x)


def _subgroup_read(group, kept_u, search):
    r"""The largest subgroup the declared content is carried on, read off the sectors.

    The subgroups inside the group the survivors form are asked from the
    largest down (:meth:`.._sectors.SectorSearch.met`), and the first order at
    which any carries the declaration is the one kept, the tie rule choosing
    among several (:func:`.._subgroups._tie_ranked`); the identity alone always
    carries it.  Where the survivors form no one largest group, the closure
    resolution speaks first and the declaration is asked again on the group it
    lands on: only an operation the declaration rules out on its own -- whose
    own cyclic group carries no representation that holds it -- is refuted
    here.  Returns what :func:`_covariant_subgroup` does; ``None`` where the
    sectors do not read the group, and the ``rep_search_too_large``
    :class:`..refusal.Refusal` where the representations to weigh at once are
    more than the search's ``max_candidates``.
    """

    keep = frozenset(int(i) for i in kept_u)
    try:
        plain = _subgroups_inside(group, keep)
        top = max(len(s) for s in plain) if plain else 0
        widest = sorted((s for s in plain if len(s) == top), key=sorted)
        if len(widest) != 1:
            bad = {i for i in keep if not search.met(_cyclic(group, i))}
            return {"subgroup": frozenset(keep - bad), "candidates": (),
                    "broken": tuple(sorted(bad))}
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
            f"the declared content of orbitals {many.orbitals} is weighed against every "
            f"representation the hoppings leave open on the sectors those orbitals lie on -- "
            f"{many.copies} identical cop{'y' if many.copies == 1 else 'ies'} of each -- and the "
            f"{many.count:,} to weigh at once are more than max_candidates = "
            f"{search.max_candidates:,}. Declare less of what the copies hold, or raise "
            "`max_candidates=` if the search is worth waiting for.",
            value=float(many.count), threshold=float(search.max_candidates),
            context={"orbitals": many.orbitals, "copies": many.copies,
                     "n_candidates": many.count, "max_candidates": search.max_candidates})
    if not carried:
        raise InternalError(
            f"no subgroup of {group.name} #{group.number_str}, the identity alone included, "
            "carries the declared content on the sectors of the Hamiltonian's conserved "
            "matrices, where the identity always carries it: the copies as they stand hold "
            "every declared dimension.")
    ranked = _tie_ranked(group, carried)
    best = frozenset(ranked[0])
    return {"subgroup": best, "candidates": ranked if len(ranked) > 1 else (),
            "broken": tuple(sorted(i for i in baseline if i not in best))}


def _declaration_breaks(group, assignment, decls, verdict, *, standing, subgroup, broken,
                        readings, not_lowered, site_tables=None, admissible=None):
    r"""The ``declaration_breaks_group`` refusal, made in one place.

    The verdict is made once, per operation, on the detector's own solution
    spaces, the declaration read on the measured multiplets
    (:func:`_verdicts_and_choices` and
    :func:`_covariant_subgroup`, which is also what
    :func:`..groups.realized_group` descends on).  This writes it down for the
    two calls that do not lower the group — :func:`.._build._build_rep`, handed a fixed
    group, and :func:`..groups.realized_group` told ``descend=False``, which
    ``not_lowered`` says in one sentence: which operations the declared content
    is not covariant under, which subgroup it **is** covariant on, the maximal
    readings where several tie, that subgroup the first of them by the tie rule,
    and in irrep names what the failing declaration measured.

    ``standing`` are the operations the verdict was read on, unitary and
    antiunitary.  ``subgroup`` and ``broken`` are the verdict, as unitary
    operations or with their primed partners already added; the partners are
    added here wherever they are missing.  ``site_tables`` and ``admissible``
    — the site character tables and the real one-dimensional characters of
    the group read — are computed here when they are not handed in.
    """

    lattice = np.asarray(group.lattice, dtype=float)
    by_ind = {int(s.ind): s for s in group.symmetries}
    standing = sorted({int(i) for i in standing})
    # each operation as the refusal lists it, and as the frame a declaration is
    # written in names it: a site's own operations by their rotations alone
    names = {i: _record_name(by_ind[i], lattice) for i in standing}
    frame = {i: (_sitechars.operation_name(by_ind[i], lattice)
                 + ("'" if by_ind[i].time_reversal else "")) for i in standing}
    unitary = [i for i in standing if not by_ind[i].time_reversal]
    # A site character is a character of the UNITARY site group, so the verdict
    # is about unitary operations -- but the group returned is the grey one they
    # generate, and the primed copy of an operation stands or falls with it.  A
    # primed partner is counted as what it is, and never as a unitary operation.
    partners = _primed_partners(group, names)
    broken_u = sorted({int(i) for i in broken if not by_ind[int(i)].time_reversal})
    kept_u = sorted({int(i) for i in subgroup if not by_ind[int(i)].time_reversal})
    primed = [partners[i] for i in broken_u if i in partners]
    every = sorted(set(broken_u) | set(primed))
    kept = sorted(set(kept_u) | {partners[i] for i in kept_u if i in partners})
    rows = tuple(verdict["rows"])
    reasons = verdict.get("reasons") or {}
    read_together = tuple(verdict.get("joint", ())) + tuple(verdict.get("by_sectors", ()))
    why_of = {i: str(reasons.get(i) or _joint_reason(read_together, i) or (
        "the declared content is carried by this operation on its own, but no group "
        "of operations containing it and those kept carries every declared multiplet "
        "in its declared irrep at once")) for i in broken_u}
    for i in broken_u:
        if i in partners:
            why_of[partners[i]] = (
                f"the primed copy of #{i} {names[i]}: a site character is a character "
                "of the unitary site group, so the two stand or fall together")

    # WHAT THE FAILING DECLARATION MEASURED, in irrep names: the same naming on
    # both calls, from the site tables and the readings of the group read
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
                            ref_orb=int(verdict.get("reference_orbital", 0)))

    def _listed(inds):
        return ", ".join(f"#{i} {names[i]}" for i in inds)

    read = ""
    if readings:

        read = ("\n  several readings are maximal and none contains the others -- "
                + "; ".join("{" + ", ".join(f"#{i}" for i in r) + "}" for r in readings)
                + " -- and the subgroup named above is the first by the tie rule "
                f"`realized_group` states: {_tie_rule_words(group)}. It is first since "
                f"{_tie_reason(group, readings)}.")
    why = "\n".join(f"    #{i} {names[i]}: {why_of[i]}" for i in every)
    table = "\n".join(
        f"    #{r['ind']:<4d} {r['name']:<14s} orbitals {tuple(r['orbitals'])}"
        f"   measured {_fmt_char(r['measured'])}"
        f"   declared {_fmt_char(r['declared'])}"
        for r in rows if int(r["ind"]) in set(broken_u))
    return Refusal(
        "declaration_breaks_group",
        f"the declared orbital content is not covariant under {len(broken_u)} of the "
        f"{len(unitary)} unitary operations of {group.name} #{group.number_str}, "
        f"{_counted_as(group.spinor)}: "
        + (_listed(broken_u) or "(none)")
        + (f", nor under their primed partners {_listed(primed)}" if primed else "")
        + ". A declaration says what the orbitals are, so a Hamiltonian whose hoppings "
        "do not respect it is not a wrong declaration -- it is a crystal with less "
        f"symmetry. The declared content is covariant on the {len(kept)} operations "
        + _listed(kept)
        + ", and `realized_group` descends to them and returns that group, with its own "
        f"labels, EBR rows and indicators. {not_lowered}"
        + read
        + (f"\n{named['verdict']}" if named else "")
        + (f"\n  why the declared content is not carried by each:\n{why}" if why else "")
        + (f"\n  measured vs declared:\n{table}" if table else "")
        + "\n  frame: " + "; ".join(f"#{i} = {frame[i]}" for i in standing),
        value=float(len(every)), threshold=0.0,
        context=dict({
            "group": group, "number_str": str(group.number_str),
            "subgroup": tuple(kept),
            "broken": tuple({"ind": int(i), "name": names[i], "note": why_of[i],
                             "rows": _returned_rows(r for r in rows
                                                    if int(r["ind"]) == int(i))}
                            for i in every),
            "tied": tuple(tuple(int(i) for i in r) for r in readings),
            "checks": _returned_rows(rows),
            "declarations": tuple({"orbitals": tuple(d["orbitals"]),
                                   "declared_as": d["spec"],
                                   "declared_irrep": d["label"]} for d in decls),
            "frame": dict(frame)}, **(named.get("named", {}) if named else {})))


def _joint_reason(joint, ind):
    """Why a content read together breaks one operation, in words; ``None`` where none speaks there.

    ``joint`` holds the contents the verdict stage reads over several
    multiplets at once (:func:`_verdicts_and_choices`); the first measured at
    ``ind`` is named.
    """
    for j in joint:
        if not any(int(ind) in row for row in j["theta"].values()):
            continue
        d = j["decl"]
        said = d.get("label") or repr(d.get("spec"))
        orbs = tuple(int(o) for o in d["orbitals"])
        if d.get("partial"):
            return (f"orbitals {orbs} are declared {said}, part of what their site holds, "
                    "and no labelling the hoppings leave open puts it inside the site's "
                    "content on a group holding this operation and those kept")
        return (f"orbitals {orbs} are declared {said} together, and no labelling the "
                "hoppings leave open carries that content on a group holding this "
                "operation and those kept")
    return None


def _declared_key(decls):
    """A comparable identity for one set of declared contents.

    What a reading of the solution spaces was made *about*, so that a reading
    handed down from :func:`..groups.realized_group` is used only where it
    answers the same question.
    """
    return tuple(sorted(
        (tuple(int(o) for o in d["orbitals"]),
         tuple(sorted((int(k), round(complex(v).real, 12), round(complex(v).imag, 12))
                      for k, v in d["constraints"].items())),
         bool(d.get("corep")))
        for d in decls))


def _multiplet_scale(group, layout, positions, cperm, classes, ref, blocks, *,
                     reversal=None, reversal_image=None, reach):
    r"""``{position: block}`` — one multiplet's own transport block, its phase settled.

    The scale a part of the crystal holding no one-dimensional multiplet reads
    its declared characters against, taken at the multiplet ``ref`` of that
    part, and the scale of a one-dimensional multiplet whose tables' lift is
    no representation of the group (:func:`_lifted_scale`).  The Hamiltonian
    fixes an operation's block of the multiplet, carried round its Wyckoff
    orbit (:func:`_multiplet_blocks`), only up to the free phase of the
    solution space it was read from, and a declared character is a statement
    about the representation, not about one member of a space.  So the orbit's
    own blocks are settled exactly as the build settles its matrices
    (:func:`_settle_unit`): each generator in its canonical norm and the
    time-reversal anchor, the rest propagated along the generators' word tree,
    and the multiplication table choosing the roots where positivity's choice
    does not multiply.  What is left is a representation of the group on the
    orbit, fixed up to one one-dimensional character, and that character is the
    part's to choose: its declarations are read against this scale with every
    character admitted (:func:`_covariant_subgroup`), so the frame the numbers
    are written in, the settling's own, constrains nothing.

    ``positions`` are the unitary operations read, ``cperm`` where each
    carries each multiplet, ``blocks`` the multiplets' blocks by ``.ind``;
    ``reversal`` is the position of the time reversal kept, with
    ``reversal_image`` its multiplet permutation, and ``reach`` the composition
    defect up to which noisy blocks are projected onto the nearest exact
    representation.  Returns ``{position: block}`` -- the block at the
    multiplet's image and the multiplet -- or ``None`` where the operations are
    no group, a block is missing, or the blocks settle into no representation
    of the operations: nothing is measured on this scale then.
    """
    syms = group.symmetries
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
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
        """One operation's matrix on the reference orbit, off the model's own blocks."""
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
    r"""``{position: block}`` -- a one-dimensional multiplet's scale, where its lift is no representation.

    A part holding a one-dimensional multiplet reads its declared characters
    against that multiplet's lift, the tables' own, carried round its Wyckoff
    orbit with the signs the hoppings give it (:func:`_transported_lifts`): the
    multiplet's own character is then the part's one-dimensional character,
    which the labelling chooses.  That holds wherever its character on its site
    group is the restriction of a one-dimensional character of the group.
    Where it is not, the lift is no representation, and the multiplet's
    character is measured: ``p_y`` at a ``3c`` site of ``P23``, whose three
    twofolds are carried onto one another by the threefolds while ``p_y`` is
    odd under two of them, is the ``B2`` of ``222`` that no character of ``T``
    restricts to.  The orbit's blocks are then settled into a representation
    as the build settles its matrices (:func:`_multiplet_scale`), the
    multiplication table choosing the signs, and that is the scale the
    declared characters are read against.

    ``ref`` is the multiplet, ``blocks`` the multiplets' blocks by ``.ind`` as
    the build assembles them, ``bases`` and ``intertwiners`` the measured
    solution spaces, the rest as for :func:`_multiplet_scale`.  Returns
    ``None`` where the lift is a representation of the operations or the
    orbit's blocks settle into none, and the settled blocks otherwise.
    """
    syms = group.symmetries
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
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


def _verdicts_and_choices(model, group, assignment, spin, validated, layout, decls,
                          positions, *, classes=None, ref_orb=0,
                          blocks=None, space_tol=IN_SPACE, tol=CHARACTER,
                          max_candidates=2 ** 20, reversal=None, sectors=None):
    r"""One verdict per operation, and the scalars it pins, off the solution spaces.

    The second step of the pipeline, and the only place a solution space is read
    for what it says about the orbitals.  Everything downstream — the descent by
    declaration, the representation's own discrete choices, the characters
    record — is written from what comes back here, so that a measurement is made
    once and read once.

    Per operation, inside the site blocks its own permutation allows (a member
    carrying one site's orbitals onto a site the operation does not send them to
    is evidence about the *operation*, which is :func:`.._multiplets._ansatz_defects`'s
    business, and no partition into multiplets absorbs it):

    * **the multiplet verdict.**  No member block-diagonal on the multiplets
      and nonsingular there means the Hamiltonian *mixes* multiplets that are
      said to be separate, and where a ``multiplets=`` states them beside a
      declaration, the operation is broken by the declaration.  Measured
      multiplets a space cannot carry are the operation's own tensor
      structure, which :func:`.._multiplets._ansatz_defects` reads.
    * **the relative scalar, per multiplet orbit.**  The scalar relating the
      member's block at the class's representative multiplet to that
      multiplet's own internal factor, against the anchor's — orbital 0's
      multiplet — which is exactly the freedom :func:`.._build._build_rep` reads the
      representation off with, one scalar per multiplet orbit per
      generator.  Where every member of the space carries the same ratio the
      hoppings **pin** it and the number is measured; where the ratio varies
      over the space they leave it **free**, which is a multiplet decoupled from
      the rest, and nothing was measured there at all.
    * **the character verdict**, for every declared multiplet the operation fixes:
      the scalar relating the member's block to the multiplet's internal factor,
      normalized by the transport block of a multiplet whose factor is fixed
      beyond the member's free phase, so that what is read is in the canonical
      induced gauge the record is written in.  The hoppings fix every block up
      to one one-dimensional character per **coupled part** of the crystal —
      the multiplets they connect, directly or through others: the anchor's
      part, and the orbits of multiplets they leave free of it,
      one part where they pin one another — so a declared multiplet is read
      against a scale inside its own part.  Where the part has a
      one-dimensional multiplet that scale is its lift, the tables' own, or,
      where that lift carried round its orbit is no representation of the
      group, the multiplet's own character, its blocks settled into one
      (:func:`_lifted_scale`).  In a part of wider multiplets it is one of
      them, its blocks settled into a representation of the group as the build
      settles its matrices (:func:`_multiplet_scale`); ``reversal`` is the position of the
      time reversal that anchors those phases, ``None`` where the group keeps
      none.  The scale's own one-dimensional character is the part's to choose:
      each part's declarations ask for one character of their own, and every
      character is admitted against them (:func:`_covariant_subgroup`).  A free
      block imposes nothing, because nothing was measured.  Two multiplets of
      one part needing different scalars break the operation, and so does a
      scalar no one-dimensional character can take: not ``±1`` where the group
      keeps an antiunitary operation, which leaves only a sign, and not a unit
      phase where it keeps none, every one-dimensional character, the complex
      ones included, being a labelling there.  A declaration written in the
      site coreps states the corep content and nothing finer, so it is not read
      at an operation whose spin-1/2 lift has zero trace, where every orbital
      content lifting to it reads zero.
    * **the contents read together**: one declared over multiplets of several
      coupled parts, and one stating part of what its site holds.  Each is
      read multiplet by multiplet, against its own part's scale, at every
      operation fixing all of them, and is one requirement on those parts'
      labellings together: the declared content carried by them, or contained
      in what they carry where part of a site was declared
      (:func:`_covariant_subgroup`).  A multiplet free at an operation leaves
      that operation unread.

    An antiunitary operation is not asked: a site character is a character of
    the site group's unitary operations, and the primed copy of a broken
    operation is broken with it, which the records downstream carry.

    ``max_candidates`` bounds the one enumeration this reading can run, the
    representations tried for an orbit of sectors where the blocks are built
    here, none having been handed down (:func:`_multiplet_blocks`), exactly as
    it bounds the build's.

    ``space_tol`` is how far a member may sit outside the blocks and still
    count as being in them (:func:`_in_blocks`).  It is a *statement about the
    model*, not a knob: an exactly symmetric Hamiltonian's solution space is in
    block form to machine precision, and a numerical one's is in block form to
    its own defect.  :func:`.._build._build_rep` therefore hands down its
    ``internal_tol`` — the same dimensionless "how far from the ideal block"
    the near-symmetric recipe loosens — so that a declaration read on a
    wannier90-scale model is measured rather than silently found unmeasurable.
    The multiplet verdict reads the declared tolerance the multiplets were
    measured at as well, where it is larger (:func:`.._tolerance.noise_floor`): the blocks
    a declared multiplet is tested in are the blocks of that measurement, so a
    frame off the ideal one by less than the declared precision holds a
    declared multiplet as it holds a measured one.  ``tol`` is how far a
    character read off the spaces may sit from a value and still be read as
    it: :data:`.._tolerance.CHARACTER` for a model built from exact numbers,
    and the declared tolerance where that is larger.  Every other threshold
    here -- a scalar that vanishes, a ratio the hoppings pin -- reads the
    declared tolerance the multiplets were measured at in the same way.

    Returns a mapping — ``"rows"`` the measured-versus-declared table in
    :attr:`AtomicLimitTable.checks`'s own shape, ``"parts"`` the coupled parts
    as sets of multiplet orbits, the anchor's first,
    ``"requirements"`` and ``"scalars"`` one mapping per part, the verdict per
    operation (``None`` where the declaration breaks it) and the raw scalar,
    ``"matrices"`` the requirements of each part at the operations declared
    generator matrices were read at,
    ``"joint"`` the contents read together, each with what its multiplets carry
    per operation and the part each is in, ``"by_sectors"`` the contents
    declared on orbits built by induction, which the sectors alone weigh, each
    with what its multiplets carry under the representation they are assembled
    with, ``"reasons"`` one sentence per broken operation, ``"classes"``/``"nclass"``
    the multiplet orbits, ``"chosen"`` the pinned class scalar at ``(ind,
    class)``, ``"within"`` the ratio an orbit of a part free of the anchor
    holds to that part's first at ``(ind, class)``, ``"free"`` the ``(ind,
    class)`` pairs the hoppings leave open, ``"free_classes"`` the classes free
    at every operation that was measured, ``"seen"`` what was measured where,
    ``"blocks"`` the multiplets' own blocks with every orbit of sectors built
    by induction and its candidates (:func:`_multiplet_blocks`), ``"reversal"`` what time reversal
    carries on each orbit (:func:`_reversal_scalars`), and ``"anchor"``,
    ``"reference_orbital"``, ``"declared"``, ``"space_tol"`` and ``"indices"``
    — what the reading was made about, so that one handed down can be checked
    against the question being asked — or the :class:`..refusal.Refusal` the
    multiplet permutation or the block enumeration made.  ``spin`` is the
    declared :class:`..spin.SpinBasis` itself, or ``None``.
    """
    syms = group.symmetries
    positions = [p for p in positions if not bool(syms[p].time_reversal)]
    cperm = _channel_perms(model, assignment, syms, positions, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    site_layout = _multiplets_of(model, assignment, spin,
                             tuple((tuple(orbs),) for orbs in site_orbs))
    nchan = int(layout._nchan)
    chan_of = np.asarray(layout._chan_of, dtype=int)
    # THE ANCHOR every scalar is read against: orbital 0's multiplet, whichever
    # orbital the lowest-angular-momentum rule later reads the crystal from, so
    # that every build of this group reads the one reading
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
                                 if int(s.ind) in set(validated.indices)],
                                ref_chan=ref, max_candidates=int(max_candidates))
        if isinstance(got, Refusal):
            return got
        measured_of = got
        multiplicity = tuple(got.multiplicity)

    lattice = np.asarray(group.lattice, dtype=float)
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

        # ---- the choice, one scalar per multiplet orbit -----------------------
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

    # A class is free where NOTHING measured it: free at every operation whose
    # solution space spoke about it.  That is a decoupled multiplet -- what it
    # carries relative to the anchor is a question this Hamiltonian cannot
    # answer, on this site or any other -- and it is the labelling's to settle
    # (:func:`.._labelling._labelling`), as a declaration states it.
    free_classes = frozenset(
        cls for cls in range(nclass)
        if any(cls in got for got in seen.values())
        and all((ind, cls) in free for ind, got in seen.items() if cls in got))
    # IDENTICAL COPIES, and uncoupled orbitals a symmetry exchanges that the
    # group's characters do not carry, are one part of the crystal each, an
    # orbit of sectors the hoppings couple to nothing else: its multiplets are
    # assembled with the blocks of one representation of it
    # (:func:`.._multiplets._induced_blocks`), and which one it carries is the labelling's
    # choice among all of them.  So no scalar is read between its multiplets,
    # which carry those blocks as they stand -- a scalar read at one multiplet
    # of an orbit of multiplets lying on several sectors says nothing of the
    # others -- and it is pinned to the anchor where it holds the anchor's
    # multiplet and free of it otherwise.
    induced_classes = []
    for rec in multiplicity:
        if rec.get("induced") is None or not rec["n_admitted"]:
            continue
        own = frozenset(int(classes[int(c)]) for c in rec["multiplets"])
        induced_classes.append(own)
        if ref_cls in own:
            free_classes = free_classes - own
            for ind in seen:
                for cls in own:
                    free.discard((int(ind), int(cls)))
                    chosen[(int(ind), int(cls))] = 1.0 + 0.0j
        else:
            free_classes = free_classes | own

    # THE COUPLED PARTS.  The hoppings fix every multiplet's block up to one
    # one-dimensional character per part of the crystal they connect: the
    # anchor's part, every orbit they pin to it, and the orbits
    # they leave free of it, which are one part where they pin one another --
    # two orbits whose blocks keep one ratio across a solution space.  A
    # declaration is a constraint on its own part's character, read against a
    # scale inside that part, so a part the hoppings never reach measures
    # nothing about it.
    joined = {int(cls): int(cls) for cls in free_classes}

    def _joined(cls):
        while joined[cls] != cls:
            cls = joined[cls]
        return cls

    for got in carried.values():
        both = [(int(cls), t) for cls, t in sorted(got.items())
                if int(cls) in joined and float(np.abs(t).max()) > zero]
        for (a, ta), (b, tb) in itertools.combinations(both, 2):
            sig = complex(np.vdot(ta, tb) / np.vdot(ta, ta))
            if float(np.abs(tb - sig * ta).max()) <= spread_tol * max(
                    float(np.abs(tb).max()), float(np.abs(ta).max()), 1.0):
                joined[max(_joined(a), _joined(b))] = min(_joined(a), _joined(b))
    for own in induced_classes:
        if ref_cls in own:
            continue
        first = min(own)
        for cls in own:
            a, b = _joined(int(first)), _joined(int(cls))
            if a != b:
                joined[max(a, b)] = min(a, b)
    apart = {}
    for cls in sorted(joined):
        apart.setdefault(_joined(cls), set()).add(int(cls))
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
    # EVERY DECLARATION IS READ ONCE, in one of two ways.  One content on the
    # multiplets of one coupled part is a requirement on that part's
    # one-dimensional character, operation by operation.  One spread over
    # several parts, or one stating part of what its site holds, is one
    # requirement on what those parts carry together: read here per multiplet,
    # at every operation fixing its site, and met where one labelling of each
    # part it spans carries it (:func:`_covariant_subgroup`).
    part_of_decl, joint_decls, on_induced = {}, [], []
    induced = frozenset(cls for own in induced_classes for cls in own)
    for d in decls:
        held = {part_of.get(int(classes[int(c)])) for c in d["channels"]}
        if any(int(classes[int(c)]) in induced for c in d["channels"]):
            # a content declared on an orbit built by induction: what it
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

    # THE SCALE each part's characters are read against.  A member of a
    # solution space is a representation matrix times one free complex number,
    # and the scalar a declared multiplet needs is only a statement about the
    # physics once that number is fixed -- in the gauge the record is written
    # in, which is the canonical induced form: one multiplet's transport block
    # carries that multiplet's own internal factor and nothing more.  A
    # one-dimensional multiplet's factor is PINNED, the tables' own lift, and
    # where the part has one that is its scale -- unless that lift, carried
    # round its orbit, is no representation of the group, where the
    # multiplet's own character is measured and its settled blocks are the
    # scale (:func:`_lifted_scale`).  A wider multiplet's block is
    # the model's own and carries the member's free phase, so in a part of
    # multiplets the scale is one of its own multiplets' blocks, settled into
    # a representation of the group; with no antiunitary operation to anchor
    # them they are fixed only up to a unit phase per generator, which the
    # part's one-dimensional character can then carry.  Either way, what the
    # part's character may take is a unit phase where the group keeps no
    # antiunitary operation -- every one-dimensional character, the complex
    # ones included, is then a labelling nothing measured tells apart -- and a
    # sign where it keeps one.
    reversal_image = None
    if reversal is not None:
        got = _channel_perms(model, assignment, syms, [int(reversal)], layout, spin)
        reversal_image = None if isinstance(got, Refusal) else got[int(reversal)]
    scales, roots, wider = {}, {}, []
    for i in sorted(spoken):
        lead = _lead_of([c for c in range(nchan) if part_of.get(int(classes[c])) == i], ref)
        tables = [c for c in range(nchan) if part_of.get(int(classes[c])) == i
                  and layout._chan_kind[c] == "tables"]
        if tables:
            scale_chan = lead if lead in tables else int(tables[0])
            # the lift, where it is a representation; the one-dimensional
            # multiplet's own character, settled, where it is not
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

    # ---- the character verdict, for what was declared -----------------------
    def _theta(x, V, lift, nu, floor):
        """Multiplet ``x``'s character against its part's scale ``nu``, or ``None`` where it is free."""
        cols = np.asarray(chan_states[int(x)], dtype=int)
        if layout._chan_kind[int(x)] == "measured":
            # the multiplet's ORBITAL character: the block's trace with the
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

    nspin = int(getattr(model, "nspin", 1) or 1)

    def _matrix(d, V, lift, nu, floor):
        """Declaration ``d``'s orbital matrix against its part's scale ``nu``, or ``None`` where it is free.

        The orbital factor of the block the operation carries its orbitals
        with, the spinor divided out atom by atom, one row and one column per
        atom in the order the declaration names them: what declared generator
        matrices are a statement about.
        """
        atoms = _declared_atoms(d, chan_states, su2, nspin)
        B = np.asarray([[[_proportionality(V[j][np.ix_(a, b)], lift)[0] for b in atoms]
                         for a in atoms] for j in range(len(V))])
        M = np.tensordot(np.conj(nu), B, axes=(0, 0)) / np.vdot(nu, nu)
        gap = float(np.abs(B - nu[:, None, None] * M[None, :, :]).max())
        if gap > spread_tol * max(float(np.abs(B).max()), floor, 1.0):
            return None                   # free: the hoppings do not measure it
        return M

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
        # ONE CONTENT OVER SEVERAL PARTS, OR PART OF A SITE'S: per multiplet,
        # what it carries against its own part's scale, at every operation that
        # fixes all of its multiplets
        for d in joint_decls:
            if ind not in d["constraints"] or (d.get("corep") and silent):
                continue                  # a corep content says nothing here
            if any(int(image[int(x)]) != int(x) for x in d["channels"]):
                continue
            got = {}
            for x in d["channels"]:
                i = part_of.get(int(classes[int(x)]))
                if i not in nus:
                    break                 # no scale in its part at this operation
                theta = _theta(x, V, lift, *nus[i])
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
            # a content declared over several multiplets is read once, as the
            # sum of theirs
            if id(d) in read or part_of_decl[id(d)] is None:
                continue
            read.add(id(d))
            if part_of_decl[id(d)] not in nus:
                continue                  # no scale in its part at this operation
            if ind not in d["constraints"] or (d.get("corep") and silent):
                continue                  # a corep content says nothing here
            nu, floor = nus[part_of_decl[id(d)]]
            if ind in (d.get("matrices") or {}):
                # GENERATOR MATRICES are read as matrices: the declared one is
                # the measured one times the value the part's character has to
                # take here, or it is no multiple of it and nothing makes it so
                if any(int(image[int(x)]) != int(x) for x in d["channels"]):
                    continue
                B = _matrix(d, V, lift, nu, floor)
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
                theta = _theta(x, V, lift, nu, floor)
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
            rows.append(r)                # its scalar was read off the matrices
            continue
        theta, want = complex(r["measured"]), complex(r["declared"])
        rows.append(dict(r, required=None if abs(theta) <= tol else want / theta))
    # EVERY DECLARATION CONSTRAINS ITS OWN PART'S CHARACTER, wherever it sits:
    # what each operation asks of that character, part by part.  An operation
    # the multiplets break is broken in every part, and one two declarations
    # of a part ask different values of is broken there; which characters meet
    # the rest is the subgroup search's question, never an anchor's.
    requirements, scalars, by_matrices = [], [], []
    for i in range(len(parts)):
        req, raw, said = dict(broken), {}, {}
        _needs_per_operation([r for r in rows if r["part"] == i], req, raw, said,
                             tol=tol, roots=roots.get(i, False))
        requirements.append(req)
        scalars.append(raw)
        # ... and what the part's declared matrices ask, which a reading of
        # contents alone does not hear (:func:`_covariant_subgroup`)
        asked = {int(r["ind"]) for r in rows if r["part"] == i and "matrices" in r}
        by_matrices.append({ind: req[ind] for ind in sorted(asked) if ind in req})
        for ind, sentence in said.items():
            why.setdefault(int(ind), sentence)
    # ... and every content read together over its multiplets, per multiplet,
    # with what reads it on a smaller group: the site group the operations
    # kept leave it, and the trace of each operation's own spin-1/2 lift
    kept_inds = {int(syms[p].ind) for p in positions}
    lift_trace = {int(syms[p].ind): complex(np.trace(_lift_of(syms[p], su2)))
                  for p in positions}
    joint = []
    for d in joint_decls:
        theta = joint_theta.get(id(d))
        if not theta:
            continue
        site = int(d["site"])
        joint.append({"decl": d, "theta": theta,
                      "part": {int(c): int(part_of[int(classes[int(c)])])
                               for c in d["channels"]},
                      "site_group": tuple(int(i) for i in assignment.site_group[site]
                                          if int(i) in kept_inds),
                      "table_of": _SiteTableOf(group, assignment, site,
                                               corep=bool(d.get("corep"))),
                      "lift_trace": lift_trace})
    # ... and every content declared on an orbit built by induction, which the
    # sectors alone weigh: what its multiplets carry under the representation
    # they are assembled with, at every operation fixing all of them, so that
    # an operation it breaks is said of its own orbitals
    by_sectors = []
    held_by = {int(c): rec for rec in multiplicity
               if rec.get("induced") is not None and rec["n_admitted"]
               for c in rec["multiplets"]}
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
                continue                  # a corep content says nothing here
            for x in d["channels"]:
                got = held_by[int(x)]["candidates"][0]["contents"].get(int(x), {})
                if ind in got:
                    theta.setdefault(int(x), {})[ind] = complex(got[ind])
        if theta:
            by_sectors.append({"decl": d, "theta": theta})

    # WHAT TIME REVERSAL CARRIES ON EACH ORBIT OF MULTIPLETS, read here once: its
    # own internal factor on every multiplet, times the scalar its solution space
    # pins on each orbit -- one on orbitals that are real functions
    # (:func:`_reversal_scalars`)
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
    # WHAT A DECLARATION ASKS OF THE SECTORS, on any subgroup: read off the
    # conserved matrices the multiplets carry and what this round's operations
    # do to them (:class:`.._sectors.SectorSearch`)
    search = None
    if decls:
        search = _declaration_search(model, group, assignment, spin, validated, layout,
                                     decls, positions, reversal=reversal, sectors=sectors,
                                     tol=tol, max_candidates=max_candidates)
    return {"rows": tuple(rows), "parts": tuple(parts),
            "requirements": tuple(requirements), "scalars": tuple(scalars),
            "matrices": tuple(by_matrices),
            "joint": tuple(joint), "by_sectors": tuple(by_sectors),
            "reasons": why,
            "classes": classes, "nclass": int(nclass), "chosen": chosen, "within": within,
            "free": frozenset(free), "free_classes": free_classes,
            "seen": {int(k): frozenset(v) for k, v in seen.items()},
            "anchor": int(ref), "reference_orbital": int(ref_orb),
            "declared": _declared_key(decls), "space_tol": float(space_tol),
            "indices": frozenset(int(syms[p].ind) for p in positions),
            "blocks": _MultipletBlocks(measured_of, multiplicity=multiplicity),
            "reversal": reversal_scalars, "search": search}


def _declaration_search(model, group, assignment, spin, validated, layout, decls, positions,
                        *, reversal=None, sectors=None, tol=CHARACTER, max_candidates=2 ** 20):
    r"""What a declaration asks of the sectors on any subgroup, or ``None`` where they read nothing.

    The sectors read where the model entered (``sectors``, read here where none
    is handed in), as the round's multiplets carry them
    (:func:`.._sectors.carried_reading`), and what the round's operations do to
    them (:func:`.._sectors.sector_action`); ``reversal`` is time reversal's
    position, which the search reads only where it is the pure one.
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
        sym = group.symmetries[int(reversal)]
        if (np.array_equal(np.rint(np.asarray(sym.rotation)).astype(int), np.eye(3, dtype=int))
                and np.allclose(np.asarray(sym.translation) % 1.0, 0.0, atol=ZERO)):
            pure = int(reversal)
    search = _sectors.SectorSearch(carried, action, group, assignment, decls, positions,
                                   reversal=pure, tol=tol, max_candidates=max_candidates)
    return search if search.complete else None


def _reference_declaration(decls, ref_orb):
    """The declaration made **about the reference orbital**, or ``None``.

    The reference orbital (:func:`.._build._build_rep`'s ``reference_orbital``) is where
    the lowest-angular-momentum rule reads the crystal first, and its own
    declaration is read first among the declarations for the same reason; it
    constrains the one-dimensional character as every declaration does, and no
    more.  A declaration that names the reference orbital inside a multiplet is
    one of them — what the multiplet is *is* what the reference orbital is.
    """
    return next((d for d in decls if int(ref_orb) in
                 {int(o) for o in d["orbitals"]}), None)


def _needs_per_operation(rows, req, raw, why, *, tol=CHARACTER, roots=False):
    """Which operations the declared content is carried by, read off the measured table.

    One reading of the measured-versus-declared table ``rows`` of one coupled
    part of the crystal, written into ``req`` (the scalar ``λ(g)`` the part's
    one-dimensional character has to take for its declared content to be what
    the hoppings carry, ``None`` where the declaration breaks the operation
    outright), ``raw`` (that scalar before it is rounded onto ``±1``) and
    ``why`` (one sentence per broken operation, in the words of the physics:
    what was declared, and what the hoppings carry).  An operation already in
    ``req`` was broken by the multiplets and is left alone.

    ``roots`` is a table read in a group with no antiunitary operation, where
    every one-dimensional character, the complex ones included, is a labelling
    nothing measured tells apart: a scalar of unit modulus is then the value of
    one and kept as it is, where an antiunitary operation leaves only a sign.

    Nothing is decided here and nothing is measured: the comparison was made on
    the solution spaces (:func:`_verdicts_and_choices`), and this turns those
    numbers into one number per operation.  Which characters meet them is
    :func:`_covariant_subgroup`'s question.
    """
    by_ind: dict = {}
    for r in rows:
        by_ind.setdefault(int(r["ind"]), []).append(r)
    for ind, group_rows in by_ind.items():
        if ind in req:                    # the multiplets already broke this operation
            continue
        need, first = None, None
        for r in group_rows:
            if need == "broken":
                break
            orbs = tuple(r["orbitals"])
            if r["required"] is None:
                if "matrices" in r:
                    # a declared matrix no scalar carries the measured one onto
                    need = "broken"
                    why[ind] = (f"orbitals {orbs} are declared "
                                f"{_stated(r, 'declared')} under this operation, and the "
                                f"hoppings carry them with {_stated(r, 'measured')} up to "
                                "one scalar, which is no multiple of it")
                elif abs(complex(r["declared"])) > tol:
                    need = "broken"
                    why[ind] = (f"orbitals {orbs} are declared "
                                f"{_fmt_char(r['declared'])} under this operation, and "
                                "the character the hoppings carry them with vanishes")
                continue
            lam = complex(r["required"])
            if need is None:
                need, first = lam, r
            elif abs(need - lam) > tol and ("matrices" in r or "matrices" in first):
                why[ind] = (f"the declarations of orbitals {tuple(first['orbitals'])} and "
                            f"{orbs} ask their part's one-dimensional character for "
                            f"{_fmt_char(need)} and {_fmt_char(lam)} under this operation, "
                            "and it takes one value")
                need = "broken"
            elif abs(need - lam) > tol:
                # both measured characters are nonzero here: a row whose
                # measured character vanishes carries no requirement
                ratio = complex(r["measured"]) / complex(first["measured"])
                base = complex(first["declared"])
                said = (f"state {_fmt_char(complex(r['declared']) / base)}"
                        if abs(base) > tol else "are not in that ratio")
                why[ind] = (f"relative to orbitals {tuple(first['orbitals'])}, the "
                            f"hoppings carry orbitals {orbs} with {_fmt_char(ratio)} "
                            f"under this operation, and the declarations "
                            f"({tuple(first['orbitals'])} "
                            f"{_fmt_char(first['declared'])}, {orbs} "
                            f"{_fmt_char(r['declared'])}) {said}")
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
                why[ind] = (f"orbitals {tuple(first['orbitals'])} are declared "
                            f"{_stated(first, 'declared')} under this operation and "
                            f"the hoppings carry them with {_stated(first, 'measured')}, "
                            f"a ratio of {_fmt_char(need)} that is not a phase, and so "
                            "the value of no one-dimensional character")
            else:
                req[ind] = complex(need) / abs(need)
        elif min(abs(need - 1.0), abs(need + 1.0)) > tol:
            req[ind] = None
            why[ind] = (f"orbitals {tuple(first['orbitals'])} are declared "
                        f"{_stated(first, 'declared')} under this operation and the "
                        f"hoppings carry them with {_stated(first, 'measured')}, a "
                        f"ratio of {_fmt_char(need)} that is not a sign: an antiunitary "
                        "operation of the group leaves only a sign between two "
                        "labellings of one crystal")
        else:
            req[ind] = complex(round(float(np.real(need))))
    return req, raw, why


def _stated(row, which) -> str:
    """What one row of the measured-versus-declared table says, ``"declared"`` or ``"measured"``.

    A character, or for a declaration of generator matrices the matrix itself,
    which is what was declared and what the hoppings carry.
    """
    if "matrices" in row:
        return _sitechars._fmt_matrix(row["matrices"][0 if which == "declared" else 1])
    return _fmt_char(row[which])


def _declared_atoms(d, chan_states, su2, nspin):
    """The states of each atom a declaration names, in the order it names them.

    An atom is one orbital, or the two states of one Wannier function, which
    carry an operation's spin-1/2 lift together (:func:`.._matrices._atom_blocks`);
    declared generator matrices have one row and one column per atom, in the
    order the declaration's orbitals were written.  ``nspin`` is the model's
    number of spin states per orbital, which a state's index is divided by to
    read its orbital.
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
    """``({ind: chi(g)}, residual)`` of one multiplet over the operations that fix it.

    This reads **the representation**, not the Hamiltonian: ``D`` is what
    :func:`.._build._build_rep` settled on the scalars the verdict stage pinned, before
    any labelling is chosen, so the number that comes back is the character
    a labelling then multiplies, with the product table anchoring it.  No
    verdict is made here — the declaration's verdict was made once, on the
    solution spaces (:func:`_verdicts_and_choices`), and this writes the
    record.

    On a **one-dimensional** multiplet the block of a fixed multiplet is ``chi(g)``
    times the operation's own internal (spin) factor, and ``chi`` is extracted
    by least-squares proportionality — the spinor part is *divided out*, which
    is what makes the declaration a statement about the **orbital** factor and
    what makes ``chi`` a single-valued character of the site point group.  On a
    **multi-dimensional** multiplet the block *is* the site representation and
    the character is its trace — read the same way, atom by atom
    (:func:`_atom_blocks`), so that the spinor is divided out there too and the
    character stays a single-valued character of the site point group.  On a
    spinless model the two are the same number.

    ``layout._spinful`` says whether a one-dimensional multiplet carries the
    operations' SU(2) lift: true for a native spinful model, and equally true
    for a spinless model whose orbital index was **declared** to encode spin
    (:class:`..spin.SpinBasis`), where the multiplet is a declared ``(up, dn)``
    pair and the block is the same ``2x2`` object.  It is that flag, not
    ``nspin``, that decides.
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
    r"""``characters=`` parsed into ``(entries, atom_of, atom_orbs)`` — grammar only.

    The first half of :func:`_normalize_declaration`, kept on its own because
    **step 1** reads the same declaration before any site group has been
    reduced and before any multiplet has been measured
    (:func:`_declared_site_contents`): one parse, so the two can never
    disagree about which orbitals were named.

    ``entries`` is one ``(orbitals, spec, joint)`` per declaration in the order
    it was written; ``atom_of`` and ``atom_orbs`` are the **atoms** — one
    orbital, or the two states of one declared spinor pair, which is one
    Wannier function and never split.

    Raises
    ------
    ValueError
        Where the declaration is malformed (:func:`_declaration_grammar`), which
        every call that takes one has asked before anything was measured.
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
    r"""What each site was declared to carry — read in the **cell group**'s site tables.

    Step 1's reading of ``characters=``, made before any Hamiltonian is: per
    site, the Cartesian polynomials the declared shorthands are
    (:data:`.._sitechars.SHORTHANDS`) and the characters the declared labels
    are, each on the site group the cell allows.  A shorthand is a *function*
    and a label an abstract *content*, and the difference is the whole of
    step 1: a function's span need not be invariant under the site group, a
    label's is invariant by construction.

    Returns ``(contents, read_in)`` — ``contents`` keyed by site with
    ``"polys"``, ``"chars"`` (one ``{ind: chi}`` per declaration, the span's
    own character standing in for a function), ``"entries"``, and ``"whole"``
    and ``"corep"``: the whole content the site was declared to hold, and
    whether any of it was written in the site coreps; ``read_in``
    keyed by the declaration's orbitals with the site group and group its word
    was read in, which is what :attr:`.._sitechars.DeclaredContent.declared_in`
    then says however far the cell is afterwards reduced.

    **The whole content** is the sum of what the site's declarations state,
    however they partition its orbitals: ``"p"`` on a ``p`` shell and ``pz``
    beside ``E`` on its orbitals are one content, ``A1 + E`` at a ``3m`` site.
    A word naming a content wider than the orbitals it is written on names part
    of it, and the parts written with one content at one site add up to whole
    copies of it -- ``px`` and ``py`` are one ``E`` -- counted as one copy where
    they make no whole number of them, since a word says which content its
    orbital belongs to.  A declared spinor pair counts once.

    Raises
    ------
    _sitechars.DeclarationError
        ``gate="unknown_label"`` where a declared word names nothing in this
        cell's site groups.  On a free-standing layer it says that the group to
        read the word in is two operations away, and the declaration is carried
        (``groups.py``).
    ValueError
        Where the declaration is malformed (:func:`_declaration_grammar`).
    """
    entries, atom_of, _atom_orbs = _declaration_entries(
        characters, model, assignment, spin=spin)
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    kept_u = {int(s.ind) for s in group.symmetries if not s.time_reversal}
    kept_au = {int(s.ind) for s in group.symmetries if s.time_reversal}
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
        # the orbitals written with one content at this site, pooled: parts of
        # a wider content add up to copies of it, and a declared spinor pair
        # written twice, once per member, is one atom
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
    # WHICH OF THE TWO the declared functions name, read exactly as
    # :func:`_normalize_declaration` reads them.  The functions declared at a
    # site generate an orbit under its site group; where that orbit is as wide
    # as the orbitals they were declared for, the content declared IS the
    # orbit -- the doublet `px` and `py` span at a `3m` site, or the `1E2E`
    # pair one `dz2` names at a `23` site -- and an orbit is invariant, so it
    # reduces nothing.  Where the orbit is wider than the orbitals named, what
    # was declared is the functions themselves, and a span of those is exactly
    # what a site group need not preserve.
    for site, rec in contents.items():
        if not rec["polys"]:
            continue
        _chi, rank = _sitechars.span_character(group, tables[site], rec["polys"])
        if int(rank) <= len(rec["atoms"]):
            rec["polys"] = []
    return contents, read_in


def _content_compatible(group, assignment, contents):
    r"""Step 1: which operations of the cell the **declared content** leaves.

    A declaration is a statement about the crystal, made before any
    Hamiltonian is read, so an operation the declared content is not carried
    by is not an operation of the cell at all.  Two halves, measured over the
    unitary operations and inherited by their primed partners (a site
    character is a character of the *unitary* site group, so ``m(x)`` and
    ``m(x)'`` stand or fall together):

    **The functions.**  The declared shorthands at site ``a`` span ``V_a``
    (:func:`.._sitechars.span_basis`).  An operation carrying ``a`` onto ``b``
    is compatible when it carries ``V_a`` **onto** ``V_b``
    (:func:`.._sitechars.carries_span`) — with ``b = a`` that is the
    stabilizer of a declared function, which is why a single ``px`` at a
    ``3m`` site is a crystal with one mirror and nothing more.

    **The characters.**  Sites ``a`` and ``b = g·a`` carry the same declared
    content when ``chi_b(g h g⁻¹) = chi_a(h)`` for every ``h`` fixing ``a``,
    the conjugation taken through the group's own product table, modulo
    lattice translations, which is the sense in which a site group is
    measured.  ``chi`` is each site's **whole** declared content, whatever
    partition of its orbitals the words were written on (``"whole"`` of
    :func:`_declared_site_contents`): a ``p`` shell declared ``"p"`` at one
    site and ``p_z`` beside its doublet at the next is one content.  What is
    compared is characters, never words: an irrep label, the shorthand whose
    orbit it is and the shell or sum that holds the same content are one
    declaration wherever their characters are one.  A function's own span
    character stands in where one end names a function and the other a label.
    Where either site's content was written in the site coreps it states the
    corep content alone, and the two are compared as corep content: on the
    operations whose spin-1/2 lift has a nonzero trace, where every orbital
    content lifting to one corep content has its characters, and not on the
    twofolds and mirrors, where every one of them lifts to zero.

    A site nothing was declared about imposes nothing: its content is the
    transported one, which is the inference a partial declaration rests on.

    **The largest group.**  Where the declaration states every site of each
    orbit it touches, and in the same terms at each -- a function at all of
    them or at none -- compatibility is a relation between the sites' declared
    contents and the compatible operations are a group.  Otherwise they need
    not be one: an operation carrying a declared site onto one with nothing to
    compare there -- left undeclared, or declared a content where the first
    holds a function -- is compatible whatever it carries the declared content
    onto, and two such can compose into one the declared content is not
    carried by: ``p_y`` declared at one ``3c`` site of ``Pm-3m`` and nothing at
    the other two, where a fourfold axis and a threefold one together carry
    ``p_y`` onto ``p_z`` at its own site.  The declared crystal's operations
    form a group, so the operations kept are the largest subgroup of the
    cell's group inside the compatible ones, the first by the tie rule every
    such choice is made by where several of that order exist
    (:func:`_largest_compatible_group`), and an undeclared site carries what it
    transports onto it: there ``m-3``, the twelve rotations of the
    tetrahedron and their products with inversion, which carries ``p_y`` onto
    ``p_z`` and ``p_x`` at the other two sites.  The compatible operations it
    leaves out are excluded beside the others, each with its reason.

    Returns ``(kept, excluded)`` — the ``.ind`` kept as a frozenset over
    **all** operations, and ``{ind: {"reason", "note", "orbitals", "sites"}}``
    for the rest.
    """
    syms = group.symmetries
    at = {int(s.ind): p for p, s in enumerate(syms)}
    prod = group.get_product_table(get_diff=False)
    perm = np.asarray(assignment.permutations, dtype=int)
    unitary = [int(s.ind) for s in syms if not s.time_reversal]
    spans = {s: _sitechars.span_basis(rec["polys"]) if rec["polys"] else None
             for s, rec in contents.items()}
    fixed = {s: tuple(int(i) for i in assignment.site_group[s] if int(i) in at
                      and not syms[at[int(i)]].time_reversal)
             for s in contents}
    # the inverse of each UNITARY operation, read off the group's own product
    # table modulo lattice translations -- never `get_inverse_table`, which is
    # asked about the antiunitary coset too and warns that it has no inverse
    ident = _identity_position(group)
    inverse = {a: b for a in unitary for b in unitary
               if int(prod[at[a], at[b]]) == ident}

    def _conjugate(g, h):
        """``g h g⁻¹`` as an ``.ind``, modulo lattice translations."""
        gi = inverse[int(g)]
        return int(syms[int(prod[at[int(g)], int(prod[at[int(h)], at[gi]])])].ind)

    # the operations a content written in the site coreps states nothing under:
    # those whose spin-1/2 lift has zero trace
    silent = ({ind for ind, v in _sitechars.lift_squares(
                  {ind: syms[at[ind]].rotation_cart for ind in unitary}).items() if v <= 1e-9}
              if any(rec.get("corep") for rec in contents.values()) else set())

    def _matched(g, a, b):
        """Does ``a``'s whole declared content transport onto ``b``'s?

        ``chi_b(g h g⁻¹) = chi_a(h)`` on every ``h`` fixing ``a`` both state --
        a relation between the two sites' whole contents, so it is symmetric,
        and among declared sites what it keeps is closed under products.  Two
        sites declared contents of different dimensions differ on the identity.
        """
        chi_a, chi_b = contents[a]["whole"], contents[b]["whole"]
        coreps = bool(contents[a].get("corep") or contents[b].get("corep"))
        for h in fixed[a]:
            if int(h) not in chi_a or (coreps and int(h) in silent):
                continue
            c = _conjugate(g, h)
            if c not in chi_b:
                # the smaller group no longer asks this question, and that
                # dropping IS the restriction of the declared content
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


#: Why step 1 leaves out an operation the declared content is carried by on its
#: own: the operations kept are the largest group inside the compatible ones
#: (:func:`_largest_compatible_group`).  ``_OUTSIDE_THE_GROUP`` is said where
#: no subgroup of that order holds the operation, ``_OUTSIDE_THE_TIE`` where
#: one does and the tie rule keeps another.
_OUTSIDE_THE_GROUP = ("the declared content is carried by this operation on its own, but "
                      "not by every product of it with the operations kept")

_OUTSIDE_THE_TIE = ("the declared content is carried by a group holding this operation, as "
                    "large as the one kept, and the tie rule keeps the other")


def _outside_the_group(rec) -> bool:
    """Whether a step 1 exclusion is the largest group's, rather than the declared content's own."""
    return rec.get("reason") in (_OUTSIDE_THE_GROUP, _OUTSIDE_THE_TIE)


def _excluded_named(group, ind, rec, *, said=""):
    """One step 1 exclusion given its operation's ``ind``, ``name`` and the ``note`` saying it in words."""

    sym = next(s for s in group.symmetries if int(s.ind) == int(ind))
    rec["ind"] = int(ind)
    rec["name"] = _record_name(sym, group.lattice)
    rec["note"] = (f"#{int(ind)} {rec['name']}: {rec['reason']} (site "
                   + (f"{rec['sites'][0]}" if rec["sites"][0] == rec["sites"][1]
                      else f"{rec['sites'][0]} -> site {rec['sites'][1]}")
                   + f", orbitals {rec['orbitals']})" + said)
    return rec


def _largest_compatible_group(group, assignment, declared, excluded):
    r"""Step 1's kept operations: the largest group inside the ones the declared content is carried by.

    ``excluded`` holds the operations the declared content is not carried by,
    unitary and primed alike (:func:`_content_compatible`), and ``declared``
    the orbitals the declaration names.  The compatible operations are
    whatever is left, and they need not compose: a declared site carried onto
    an undeclared one constrains nothing there, nor does a declared function
    carried onto a site declared a content alone, so two compatible operations
    can make one the declared content is not carried by.  The crystal declared
    has a group of operations, and it is the largest subgroup of ``group``
    whose unitary operations all lie among the compatible ones -- the question
    the descent asks of operations measured to hold that do not compose, asked
    here of the operations the declared content is carried by, and answered by
    the same enumeration and the same tie rule
    (:func:`.._subgroups._resolve_closure`).  A primed operation stands or falls
    with its unitary partner, as a site character is a character of the unitary
    site group.

    Where the compatible operations already compose, which they do wherever
    the declaration states every site of an orbit, as a function at all of
    them or at none, nothing is added.  Otherwise each operation left out is
    excluded with ``_OUTSIDE_THE_GROUP`` as its reason, or
    ``_OUTSIDE_THE_TIE`` where a subgroup of the largest order holds it and
    the tie rule keeps another, whose note then names the subgroups that tie
    and why the first is kept.  Its ``sites`` are a declared site and its
    image, the first declared site carried onto an undeclared one where there
    is one, and its ``orbitals`` the declared ones there.

    Returns ``(kept, excluded)`` -- the ``.ind`` kept as a frozenset over all
    operations, and ``excluded`` with the operations left out added.
    """

    syms = group.symmetries
    compatible = [s for s in syms if not s.time_reversal and int(s.ind) not in excluded]
    if _descent_closure(group, compatible):
        _group, dropped, tied = _resolve_closure(group, [int(s.ind) for s in compatible])
        at = {int(s.ind): p for p, s in enumerate(syms)}
        perm = np.asarray(assignment.permutations, dtype=int)
        site_of = np.asarray(assignment.site_of_orbital, dtype=int)
        named = sorted({int(o) for o in declared})
        sites = sorted({int(site_of[o]) for o in named})
        lattice = np.asarray(group.lattice, dtype=float)

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
    """Parse ``characters=`` into the declared contents, each with the orbitals it is declared on (grammar only).

    A declaration names **what a set of orbitals holds**.  The piece the
    grammar counts is the **atom** — one orbital, or the two states of one
    declared spinor pair, which is one Wannier function and never split — and
    a content's dimension is a number of atoms: ``{(0, 2): "E"}`` on an
    interleaved spinor basis with pairs ``(0, 1)`` and ``(2, 3)`` declares the
    two *pairs* jointly as a doublet, while ``{(0, 1): "s"}`` is one atom.
    The joint form names the content its orbitals hold together -- an irrep,
    a sum of irreps or a shell -- and that content's dimension has to be the
    number of atoms it names.

    A word whose content is wider than the one orbital it is written on --
    ``px`` at a ``3m`` site, whose orbit is the doublet ``E`` -- names that
    orbital's part of the content.  Where the multiplet measured there (in
    ``layout``) has the content's own dimension, the part is that whole
    multiplet (``{0: "E'"}`` on a ``p_x, p_y`` pair).  Elsewhere the parts
    written with one content at one site add up, and have to make whole
    copies of it: ``["pz", "px", "py"]`` on a ``p`` shell whose orbitals are
    turned inside the site is ``A1`` on ``p_z`` and one ``E`` on the other
    two, however the Hamiltonian binds them.  How the declared contents are
    then read against the measured multiplets is :func:`_declared_blocks`'.

    Each declaration comes back with the :class:`.._sitechars.DeclaredContent`
    it resolved to — the constraint ``{ind: chi(g)}``, the label, the dimension,
    the Cartesian polynomials a shorthand names and the group it was read in —
    which is what every later re-identification re-keys rather than resolving
    the label again.

    ``read_in`` is step 1's own reading, keyed by the declaration's orbitals:
    where the word was read before the declared content reduced the cell, and
    what site group it resolves in now (:func:`_declared_site_contents`).
    Without it the two are one place, written once.

    Raises
    ------
    _sitechars.DeclarationError
        ``gate="ambiguous_multiplet"`` for a joint declaration whose content
        is not the dimension of the orbitals it names, or parts of a content
        that make no whole number of copies of it at their site;
        ``gate="unknown_label"`` from the label resolution itself.
    ValueError
        Where the declaration is malformed (:func:`_declaration_grammar`).
    """
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    paired = spin is not None              # each atom a declared (up, dn) pair
    chan_of, chan_orbs = layout._chan_of, layout._chan_orbs
    # the indivisible pieces, which is what the dimension check counts
    raw, atom_of, atom_orbs = _declaration_entries(characters, model, assignment,
                                                   spin=spin)

    def _atoms_named(orbs):
        """The distinct indivisible pieces a declaration names, order-preserving."""
        out = []
        for a in orbs:
            k = atom_of[int(a)]
            if k not in out:
                out.append(k)
        return tuple(out)

    def _members(keys):
        """Every orbital of those pieces, each piece in its own order."""
        return tuple(int(o) for k in keys for o in atom_orbs[k])

    # THE WORD WRITTEN FOR EACH ORBITAL, whichever declaration it came in: both
    # states of a declared spinor pair, one Wannier function, and every orbital
    # of a content already resolved.  A multiplet the Hamiltonian glued and
    # declared orbital by orbital (``{0: "px", 1: "py"}`` on a ``p_x, p_y``
    # pair) resolves once per orbital to the whole multiplet, and each of those
    # contents carries every word its orbitals were written with, so that the
    # one a later round keeps still says all of them.  A shell is a function
    # word of every orbital it was written on (``"p"``); an irrep or a sum of
    # irreps names a content and no function.
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
    # ... and WHAT WAS WRITTEN for each orbital, word or not: one declaration can
    # cover orbitals written as different words, `px` on one and `py` on its
    # partner, and each row says its own
    written = {}
    for orbs, spec, _joint in raw:
        if isinstance(spec, _sitechars.DeclaredContent):
            for a, w in (spec.written or tuple((int(o), spec.spec) for o in spec.orbitals)):
                written.setdefault(int(a), w)
            continue
        for a in orbs:
            for o in atom_orbs[atom_of[int(a)]]:
                written.setdefault(int(o), spec)

    decls, placed, parts = [], [], {}

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
        # WHERE THE DECLARED WORD WAS READ, which is not always where it
        # resolves: step 1 reads it in the cell's own site groups and may lower
        # the cell before anything is resolved, and the record has to say both.
        first = (read_in or {}).get(tuple(sorted(int(a) for a in orbs)))
        read_where = (base.declared_in if base is not None else
                      (first if first is not None else here))
        reduced_where = (base.reduced_in if base is not None else
                         ("" if first is None or first == here else here))
        key = _sitechars.shorthand_key(spec)
        functions = ((base.functions if base is not None else ())
                     if key is None else (_sitechars.SHORTHANDS[key],))
        # a sum written in the site coreps states the corep content and nothing
        # finer, which every comparison with the Hamiltonian's blocks reads
        corep = (base.corep if base is not None
                 else _sitechars.named_in_coreps(table, spec))
        # GENERATOR MATRICES, on every operation the declared ones generate: a
        # statement about which basis of the content the orbitals are, which
        # the traces in `cons` do not make
        here_inds = {int(i) for i in table.indices}
        matrices = ({int(k): M for k, M in base.matrices.items() if int(k) in here_inds}
                    if base is not None
                    else _sitechars.generator_matrices(group, table, spec, where))
        rec = {"site": site, "spec": spec, "label": label, "dim": dim, "cons": cons,
               "note": note, "base": base, "read_where": read_where,
               "reduced_where": reduced_where, "functions": functions, "joint": joint,
               "corep": corep, "mulliken": mulliken, "matrices": matrices}

        if joint and dim != len(keys):
            n = len(keys)
            noun = "Wannier function" if paired else "orbital"
            # a content in the site coreps is counted in what holds it, each
            # orbital with both its spin states
            wide = (f"holds {dim} {noun}{'' if dim == 1 else 's'}" if paired or corep
                    else f"is {dim}-dimensional")
            # the spelling to offer: separate words for a content of one, and
            # otherwise what the site holds says what fits it
            fix = (("; {" + ", ".join(f"{int(min(atom_orbs[k]))}: {label!r}" for k in keys)
                    + f"}} is the spelling for {n} separate {label} {noun}s.")
                   if dim == 1 else
                   ". " + _multiplet_spelling(site, _site_atoms(atom_orbs, site), dim, spec,
                                              table, paired=paired))
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"{where}: the joint form names the content its {noun}s hold together, and "
                f"{label!r} {wide} against the {n} {noun}{'' if n == 1 else 's'} named, "
                "so it is not what they hold jointly" + fix,
                {"orbitals": orbs, "site": site, "declared_irrep": label, "dim": dim,
                 "multiplet": tuple(atom_orbs[k] for k in keys)})
        if matrices:
            # MATRICES ARE READ ON WHOLE MULTIPLETS, the orbitals the hoppings
            # bind together at the site: a matrix on part of one says how the
            # hoppings would have to keep apart what they mix, and one mixing
            # two says how they would have to mix what they keep apart.  Over
            # several, block-diagonal on them, they are one declaration per
            # multiplet -- a declaration carried down to a smaller group, whose
            # multiplets have come apart, reads so
            members = _members(keys)
            touched = sorted({int(chan_of[int(o)]) for o in members})
            bound = " and ".join(str(tuple(int(o) for o in chan_orbs[c])) for c in touched)
            context = {"orbitals": orbs, "site": site, "multiplets": tuple(
                tuple(int(o) for o in chan_orbs[c]) for c in touched)}
            if any(not set(int(o) for o in chan_orbs[c]) <= set(members) for c in touched):
                raise _sitechars.DeclarationError(
                    "ambiguous_multiplet",
                    f"{where}: generator matrices are read on whole multiplets the hoppings "
                    f"measure, their orbitals named together, and the hoppings bind these "
                    f"orbitals as {bound}: name every orbital of a multiplet in one key.",
                    context)
            place = [[i for i, k in enumerate(keys) if int(chan_of[int(atom_orbs[k][0])]) == c]
                     for c in touched]
            if len(place) > 1:
                if any(float(np.abs(M[np.ix_(a, b)]).max()) > CHARACTER
                       for M in matrices.values() for a in place for b in place if a is not b):
                    raise _sitechars.DeclarationError(
                        "ambiguous_multiplet",
                        f"{where}: the matrices declared mix {bound}, which the hoppings "
                        "measure as separate multiplets: give each of those multiplets its "
                        "own key and its own matrices.", context)
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
            # a content wider than the one orbital it is written on, where the
            # HAMILTONIAN glued a multiplet of its dimension: the declaration is
            # about the whole multiplet measured there
            c = int(chan_of[int(orbs[0])])
            if int(layout._dim(c)) == int(dim):
                _one(rec, tuple(int(o) for o in chan_orbs[c]), keys)
                continue
        # A PART OF A CONTENT: the orbitals written with it at one site add up,
        # below, to the copies of it they hold together
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
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"orbital{'s' if len(orbs) > 1 else ''} "
                f"{orbs if len(orbs) > 1 else orbs[0]}: {rec['label']!r} {wide}, "
                f"and the {len(keys)} {noun}{'s' if len(keys) > 1 else ''} of site {site} "
                f"written with it make{'s' if len(keys) == 1 else ''} no whole number of "
                f"copies of it, so which {noun}s hold it is not said. "
                + _multiplet_spelling(site, _site_atoms(atom_orbs, site), dim, rec["spec"],
                                      table, paired=paired),
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
    # A SITE WITH DECLARED MATRICES is read multiplet by multiplet, which every
    # declaration at it has to allow: one naming no whole measured multiplet
    # makes the site one content over all of them (:func:`_site_reading`), and
    # one multiplet's matrices are no statement about that
    measured = tuple(tuple(tuple(int(o) for o in m) for m in site) for site in layout.by_site)
    for site in sorted({int(d["site"]) for d in decls if d["matrices"]}):
        here = [d for d in decls if int(d["site"]) == site]
        named = {int(o) for d in here for o in d["orbitals"]}
        touched = [m for m in measured[site] if set(m) & named]
        if ({frozenset(d["orbitals"]) for d in here} != {frozenset(m) for m in touched}
                or (any(d["corep"] for d in here) and len(measured[site]) > 1)):
            raise _sitechars.DeclarationError(
                "ambiguous_multiplet",
                f"site {site}: generator matrices are read on the multiplet they name, and "
                f"the declarations at this site do not each name a whole multiplet the "
                f"hoppings measure there ({', '.join(str(m) for m in measured[site])}), "
                "so the site is read as one content over all of them: give each multiplet "
                "its own key.",
                {"site": site, "orbitals": tuple(sorted(named)),
                 "multiplets": measured[site]})
    return decls


def _declared_character_key(cons):
    """A comparable identity for one declared character, ``{ind: chi(g)}``."""
    return tuple(sorted((int(k), round(complex(v).real, 9), round(complex(v).imag, 9))
                        for k, v in cons.items()))


def _site_atoms(atom_orbs, site) -> tuple:
    """The atoms of one site, each its orbitals: its orbitals, or its declared spin pairs."""
    return tuple(atom_orbs[k] for k in sorted(k for k in atom_orbs if int(k[0]) == int(site)))


def _multiplet_spelling(site, atoms, dim, spec, table, *, paired) -> str:
    """The spelling to suggest for a ``dim``-dimensional content at one site.

    ``atoms`` are the site's own: its orbitals, or, where ``paired``, its
    declared spin pairs, each one Wannier function, and ``dim`` counts them.  A
    suggestion has to be a declaration this package accepts, so the joint form
    is offered only where the site holds enough of them to make one, each named
    by one of its orbitals; where it does not, the message says what the site
    holds and lists the contents that fit it (:func:`.._sitechars.fitting_words`).
    """
    n, dim = len(atoms), int(dim)
    if n >= dim > 1:
        joint = tuple(int(min(a)) for a in atoms[:dim])
        return ("Use the mapping form and name the partners jointly, e.g. "
                f"{{{joint}: {spec!r}}}.")
    if paired:
        held = (f"{n} Wannier function{'' if n == 1 else 's'}, the spin "
                f"pair{'' if n == 1 else 's'} " + " and ".join(str(tuple(a)) for a in atoms))
        wide = f"no multiplet of {dim} Wannier functions"
        room = "one Wannier function" if n == 1 else f"at most {n} Wannier functions"
    else:
        held = f"{n} orbital{'' if n == 1 else 's'} {tuple(int(a[0]) for a in atoms)}"
        wide = f"no {dim}-dimensional multiplet"
        room = "one orbital" if n == 1 else f"at most {n} orbitals"
    words = _sitechars.fitting_words(table, n)
    return (f"Site {site} carries {held}, so {wide} sits on it at all: name a content of "
            f"{room} instead" + (f" ({', '.join(words)})" if words else "") + ".")


def _declared_blocks(decls, layout, assignment, tables, *, spin=None):
    r"""``decls`` — a declaration read on the multiplets the Hamiltonian measures.

    A declaration states what each site holds; how the model's orbitals are
    partitioned inside a site is its basis, which a Wannier basis turned inside
    its site need not honour.  So the multiplets are always the measured ones,
    and each site's declarations are read on them one of two ways:

    * where the declared contents *are* the measured multiplets they touch, each
      is read on its own multiplet: those multiplets are the Hamiltonian's own
      subspaces, not a choice of basis;
    * otherwise the site's declarations are one content over all of its
      measured multiplets, the sum of what was declared there
      (:func:`_site_reading`) -- a content declared orbital by orbital on a
      multiplet the Hamiltonian binds whole, one declared whole over several
      multiplets, a word on one orbital of a multiplet.  Where some of the
      site's orbitals were declared nothing, what was declared is part of what
      the site holds.

    **A content written in the site coreps** states the corep content alone,
    and which of a site's multiplets holds which of its coreps is the basis's,
    so it is read over the site's multiplets whole wherever the site holds
    several: ``2x-E1`` on a ``(p_x, p_y)`` pair measured apart from its
    ``p_z`` is part of the site's ``2x-E1 + -1E2E``.

    Whether the content is what the site's multiplets carry is then the
    verdict's comparison of characters (:func:`_verdicts_and_choices`): one
    content spread over several coupled parts of the crystal is one
    requirement on their characters together, and a part of what a site holds
    has to be contained in it.  ``layout`` holds the measured multiplets,
    ``tables`` the site tables the declarations were read in, and ``spin`` --
    the declared :class:`..spin.SpinBasis`, whose ``(up, dn)`` pairs are each
    one atom of two orbitals, or ``None`` -- is what the contents were read
    with.

    Returns the declarations to read, in the order written, a site's one
    content in the place of its first.
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    pair = 1 if spin is None else 2
    measured = tuple(tuple(tuple(int(o) for o in ch) for ch in site) for site in layout.by_site)
    # each declared content once, and the SAME orbitals declared twice (a
    # check on itself, which is how a carried record spells a measured pair)
    # are one content, not two
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
    """``{(site, orbitals): content}`` -- one site's declarations, read on its measured multiplets.

    ``orbitals`` are the site's, ``measured`` its multiplets and ``groups`` its
    declared contents keyed by their orbitals (:func:`_declared_blocks`), and
    ``pair`` how many orbitals make one atom.  Where the declared contents are
    exactly the measured multiplets they touch, each is read on its own
    multiplet and nothing comes back -- unless one was written in the site
    coreps over a site of several multiplets, which states the site's corep
    content.  Otherwise every declared content of the site comes back as the
    one content they add up to over all of its multiplets
    (:func:`_block_content`).
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
            f"the declarations at the site of orbitals {tuple(sorted(orbitals))} were each "
            "resolved in its site group, and their sum is no content of that group as wide "
            "as the orbitals they name. A content resolved in a site table is a sum of its "
            "irreps by construction, so the two disagreeing is a defect in this package, "
            "not a tolerance to loosen.")
    site = int(next(iter(groups.values()))["site"])
    return {(site, tuple(int(o) for o in g)): one for g in groups}


def _block_content(groups, multiplets, table, pair):
    r"""One declared content over a site's measured multiplets, or ``None`` where it is none.

    ``groups`` are the site's declarations, ``multiplets`` its measured
    multiplets and ``pair`` how many orbitals make one atom.  What was declared
    is the sum of the groups' contents, each a sum of the site's irreps as wide
    as the orbitals it names; narrower than the multiplets, it is part of what
    they hold.  Whether they carry it is the verdict's comparison of
    characters (:func:`_verdicts_and_choices`).  ``None`` where a declaration is
    no such sum, or the content is wider than the multiplets.
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
    """The ``verdict=`` and ``named=`` halves of a ``declaration_breaks_group`` message.

    Which declaration is the one a reader has to act on: reading the
    declarations in the order the lowest-angular-momentum rule reads the
    crystal — the reference orbital's own first where it has one, then the
    others as written — the first that no one-dimensional character meets
    together with those before it in its coupled part of the crystal.  So
    which declaration is named does not depend on where the reference's own
    was written.  Nothing is measured here and nothing is decided;
    :func:`_name_the_mismatch` turns those numbers into irrep names.
    """
    def _needs(d):
        """What the rows measured for one declaration ask of its part's character."""
        out = {}
        for r in rows:
            if tuple(r["orbitals"]) != tuple(d["orbitals"]):
                continue
            if r["required"] is not None:
                out[int(r["ind"])] = complex(r["required"])
            elif abs(complex(r["declared"])) > 1e-6:
                out[int(r["ind"])] = None
        return out

    part_of = {tuple(r["orbitals"]): int(r.get("part", 0)) for r in rows}
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
            int(r["ind"]): complex(r["measured"]) for r in rows
            if tuple(r["orbitals"]) == tuple(d["orbitals"])}
    verdict, named = _name_the_mismatch(failed, measured, still_free, decls, site_tables)
    return {"verdict": verdict,
            "named": dict(named, orbitals=tuple(failed["orbitals"]),
                          declared_as=failed["spec"], declared_irrep=failed["label"])}


def _name_the_mismatch(failed, measured, still_free, decls, site_tables, tol=1e-6):
    r"""What a ``declaration_breaks_group`` measured, said *in irrep names*.

    The numeric measured-vs-declared table below the message is complete but
    unreadable; what a reader needs is "the declaration said A1, the hoppings
    say B1".  Both halves of that are computable here:

    * the **measured label** of the failing multiplet — its character times
      each one-dimensional character still admissible when the declaration was
      tested,
      named against its own site table and so in the vocabulary the record
      **returns** its labels in: the double-valued site corep where the states
      carry spin, the single-valued site irrep where they do not.  When the
      earlier declarations already pinned the one-dimensional character there is
      exactly one such name; otherwise the alternatives are all listed, because the
      ambiguity is real.
    * the **relative character** ``θ_measured / θ_declared``, which is a
      character of the site group in its own right — it is exactly the
      one-dimensional character the declaration would need and no labelling
      can supply.  It is a one-dimensional character and not a
      site representation, so it is named in the orbital-factor vocabulary
      rather than the returned one.  For the standing example (lin3g's two
      ``1a`` orbitals, both declared ``s``) it is ``B1``, and it stays ``B1``
      when the two declarations are written the other way round, because both
      are read in the gauge the *earlier* declarations already pinned rather
      than in the reference orbital's.

    Nothing about the spin frame is said here.  A declaration is what the
    declaration states the orbitals are, and a Hamiltonian that does not respect it
    is a crystal with less symmetry rather than a transcription to check: the
    verdict is the smaller group, and the message is about which operations it
    keeps.

    This names; it never decides.  Every number it reads was measured once, by
    :func:`_verdicts_and_choices`, and the verdict it is spelling out was made
    by :func:`_covariant_subgroup`.

    Returns ``(text, context)``: the block to splice into the message, and the
    machine-readable half (``measured_irrep``, ``relative_character``,
    ``declared_irreps``) for the ``Refusal``'s context.
    """
    tab = site_tables.get(int(failed["site"]))
    theta = measured[tuple(failed["orbitals"])]

    meas, rels = [], []
    for chi in (still_free or [None]):
        gauged = (dict(theta) if chi is None
                  else {ind: complex(chi.get(int(ind), 1.0)) * v for ind, v in theta.items()})
        lab = _label_character(tab, gauged)
        if lab and lab not in meas:
            meas.append(lab)
        # a content written in the coreps states nothing where the spin-1/2
        # lift has zero trace, and nothing was measured there
        rel = {ind: complex(gauged.get(int(ind), 0.0)) / complex(want)
               for ind, want in failed["constraints"].items() if abs(complex(want)) > tol
               and (not failed.get("corep") or int(ind) in gauged)}
        lab = _label_character(tab, rel, orbital=True)
        if lab and lab not in rels:
            rels.append(lab)
    measured_irrep = " or ".join(meas)
    rel_label = " or ".join(rels)

    def _declared(d):
        """A declaration named as the record returns its irrep, against its own site."""
        return (_label_character(site_tables.get(int(d["site"])), d["constraints"],
                                 sums=True) or d["label"] or d["spec"])

    at = next((i for i, d in enumerate(decls) if d is failed), len(decls))
    said = ", ".join(f"{tuple(d['orbitals'])}={_declared(d)}" for d in decls[:at])
    lines = []
    if said:
        lines.append(f"  declared {said}, {tuple(failed['orbitals'])}="
                     f"{_declared(failed)}")
    wanted = _declared(failed)
    if failed.get("matrices") and measured_irrep and measured_irrep == wanted:
        # declared matrices whose traces the hoppings carry: what they do not
        # carry is the basis, which no irrep name says
        lines.append(
            f"  measured: orbital(s) {tuple(failed['orbitals'])} carry {measured_irrep}, "
            "as declared, and no one-dimensional character the hoppings leave open gives "
            "them the declared matrices: the hoppings carry that content in another basis.")
    else:
        lines.append(
            f"  measured: orbital(s) {tuple(failed['orbitals'])} carry "
            f"{measured_irrep or 'a character matching no single irrep of this site group'}"
            f", not the declared {wanted or 'character'}"
            + (f"; relative character {rel_label}." if rel_label else
               "; the ratio measured/declared is not a character of this site group either, "
               "so no other labelling accommodates it."))
    return "\n".join(lines), {"measured_irrep": measured_irrep,
                              "relative_character": rel_label,
                              "declared_irreps": tuple(
                                  (tuple(d["orbitals"]), d["label"]) for d in decls)}


def _label_character(table, chi, tol=1e-6, *, sums=False, orbital=False):
    """Name a character against a table's rows — ``""`` when no row matches it.

    The label :func:`_named_content` gives, without where it comes from.
    ``sums`` names a **reducible** character as the sum of the irreps it
    decomposes into; off by default, because a relative character is
    one-dimensional by construction and a sum there would name an object that
    cannot exist.
    """
    return _named_content(table, chi, tol, sums=sums, orbital=orbital)[0]


#: How directly a row's name comes off the tables, most direct first.  A sum of
#: rows is named as directly as its least direct part.
_NAMED_BY_RANK = {"tables": 0, "point group": 1, "Mulliken": 2}


def _named_content(table, chi, tol=1e-6, *, sums=True, orbital=False):
    """``(label, named_by)`` — the site irreps a character holds, and where their name comes from.

    A character one row of the table matches is that row, with the row's own
    name and provenance.  With ``sums``, a **reducible** character is named by
    the irreps it decomposes into, which are fixed by the character alone
    whatever basis the orbitals are written in: a ``p`` shell whose orbitals
    are turned inside a ``3m`` site so that no subset of them carries the site
    group is ``A1 + E`` all the same, one row of the table per term.  Where the
    states carry spin a row's own name can already be a sum -- ``E ⊗ ½`` at a
    ``3m`` site is ``-1E2E + -E1`` -- and the terms are counted across the
    rows, so the same shell with its spin is ``2x-E1 + -1E2E``: the site
    content an aligned basis spreads over two multiplets, ``-E1`` and
    ``-1E2E + -E1``, spelled once.  An irrep subduced to a smaller site group
    is named the same way (``E`` of ``4mm`` restricted to ``mm2`` is
    ``B1 + B2``).

    A sum is named from the tables only as far as its least direct part is --
    ``tables``, then ``point group`` -- and a sum one of whose rows has no name
    is not named at all, ``("", "unnamed")``, since naming it would claim a
    name for that row too.  The generator characters stay the whole statement
    there.

    ``orbital`` asks for the row's **orbital-factor** spelling where it has one
    (:attr:`.._sitechars.SiteGroupTable.declarable`).  That is what a character
    which is not a site representation at all wants — the ratio of a measured
    character to a declared one is a one-dimensional character of the site
    group, the character the declaration would need, and naming it after the
    double-valued corep of the row with those values would name the wrong kind
    of object.
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
    """``(chi_one, m)`` — a character that is ``m`` copies of one irrep, split.

    A multiplet the Hamiltonian binds can hold several copies of one site
    irrep, written in a basis no orbital subset splits (``A'`` and ``A'`` at a
    mirror site, rotated into one another), and every vector of such an
    isotypic component carries that irrep.  So its rows are the irrep's:
    ``chi_one`` is the character of one copy and ``m`` how many the multiplet
    holds.  Anything else — an irrep, or a sum of different ones that the
    site's own operations glue — comes back unchanged with ``m = 1``.
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
    """``chi / k`` where the content ``chi`` is ``k`` copies of one content, and ``None`` otherwise.

    Read on the site table: every irrep's multiplicity a multiple of ``k``.  A
    content the table cannot read is taken as it stands, divided.
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
    """Sort key ordering one-dimensional characters by their distance from the trivial one.

    First how many operations the character is not ``+1`` on; then, operation by
    operation in ``.ind`` order, the largest real part and — where a group with
    no antiunitary operation admits complex characters and two of them are
    conjugate — the largest imaginary part.  That last clause puts
    ``λ(4⁺) = +i`` before ``−i`` on a unitary-only fourfold axis, the two being
    equally far from trivial and equally real: the one turning with the axis
    first.  As the labelling's last key it decides only where the content,
    the letter and the band labels all tie; on the spin-orbit ``I4_1`` model
    the band labels decide, for ``−i``.
    """
    vals = [complex(v) for _, v in sorted(chi.items())]
    return (sum(1 for v in vals if abs(v - 1.0) > 1e-9),
            tuple((-round(float(v.real), 9), -round(float(v.imag), 9)) for v in vals))


def _meets(chi, needs, *, over=None, tol=CHARACTER):
    r"""Whether a one-dimensional character meets what declarations ask of it.

    Every reading of a declaration comes to this one test.  ``needs`` maps an
    operation's ``.ind`` to the value the character of the declared
    multiplets' coupled part has to take there for the declared content to be
    what the Hamiltonian carries, ``None`` where no value does; an operation it
    does not name asks nothing.  ``over`` asks the question on those operations
    alone -- a subgroup, or the operations a declaration was measured at.
    """
    for ind, want in needs.items():
        if over is not None and int(ind) not in over:
            continue
        if want is None or abs(complex(chi.get(int(ind), 1.0)) - complex(want)) > tol:
            return False
    return True


def _declared_needs(d, raw, lift_trace, *, tol=1e-6):
    r"""``{ind: value}`` -- what one declaration asks of its part's character, on the matrices.

    ``raw`` holds each multiplet's orbital character on the representation
    built before any labelling is chosen (:func:`_multiplet_characters`); the
    declared multiplets carry the declared content under a character ``chi``
    exactly where ``chi(g)`` times the sum of their characters is the declared
    one, at every operation the declaration speaks about.  A content written in
    the site coreps states nothing at an operation whose spin-1/2 lift has zero
    trace, and an operation that moves one of the multiplets asks nothing.
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
    """Does a block, its multiplets under ``assigned(c)``, hold what was declared on part of it?

    What is left of the block's content once the declared content is taken
    away has to be a content of the site group, the part nothing was declared
    about.  A content written in the site coreps states the corep content
    alone, so what the block holds has to contain it as coreps
    (:func:`.._sitechars.corep_counts`): ``2x-E1`` is part of the spin-orbit
    ``p`` shell's ``2x-E1 + -1E2E`` at a ``3m`` site although no orbital
    content lifting to it is part of the shell's ``A1 + E``.  Where a row's
    lift has no name, what the block holds has to contain an orbital content
    lifting to it.  ``assigned(c)`` is the character multiplet ``c``'s coupled
    part carries, so the blocks of several parts are read together; ``unread``
    is what comes back where a class of the site group was not measured on
    every multiplet of the block, and so says nothing either way.
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
    """Whether one content read together is carried under one labelling of the parts it spans.

    ``j`` is one of the contents the verdict stage reads together
    (:func:`_verdicts_and_choices`): what each of its multiplets carries, per
    operation, against its own coupled part's scale.  ``pick`` gives each part
    it spans a character, and ``inside`` is the subgroup asked about.  A
    content declared on the whole of its site is carried where its
    multiplets together carry exactly it; one declared on part of the site
    where what they carry contains it, on the site group the subgroup leaves
    (:func:`_holds_part`), an operation of that site group nothing measured
    leaving the question open.
    """
    d = j["decl"]
    raw = {int(c): {int(g): v for g, v in row.items() if int(g) in inside}
           for c, row in j["theta"].items()}

    def assigned(c):
        return pick[j["part"][int(c)]]

    if not d.get("partial"):
        carried = {c: {g: complex(assigned(c).get(g, 1.0)) * complex(v) for g, v in row.items()}
                   for c, row in raw.items()}
        return _meets({}, _declared_needs(d, carried, j["lift_trace"]), tol=tol)
    table = j["table_of"](tuple(int(g) for g in j["site_group"] if int(g) in inside))
    if table is None:                     # pragma: no cover - a site group closes
        return True
    return _holds_part(d, assigned, raw, {int(d["site"]): table}, j["lift_trace"],
                       unread=True)


def _multiplet_characters(D, syms, u_pos, cperm, layout, prod):
    r"""``(raw, residual)`` -- every multiplet's orbital character on one representation.

    Read once, on the representation built before any labelling is chosen:
    ``raw[c]`` is multiplet ``c``'s character over the operations that fix it
    (:func:`_measure_channel_characters`), and ``residual`` the worst defect of
    that reading -- on a measured multiplet also how far its blocks are from
    closing as a single-valued representation of the site group, which a
    spinful model's are not (the block carries the spinor too).
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
                    ab = int(syms[int(prod[pos_of_ind[a_ind], pos_of_ind[b_ind]])].ind)
                    if ab in block:
                        worst = max(worst, float(np.abs(A @ B - block[ab]).max()))
    return raw, worst


def _lead_of(chans, first):
    """A coupled part's lead multiplet: ``first`` where the part holds it, its first otherwise."""
    return int(first) if int(first) in {int(c) for c in chans} else int(min(chans))


def _returned_rows(rows):
    """The measured-versus-declared rows as a record returns them.

    What each operation would have needed of the one-dimensional character,
    which coupled part of the crystal it was read in, and the matrices a row
    of declared generator matrices was read off, are the package's own reading
    of the row and stay inside it; a returned row says what was measured and
    what was declared, as characters.
    """
    return tuple({k: v for k, v in r.items() if k not in ("required", "part", "matrices")}
                 for r in rows)
