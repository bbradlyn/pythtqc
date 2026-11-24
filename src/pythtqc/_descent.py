"""The descent: the cell lowered to the group the Hamiltonian obeys.

Each round gives one verdict per operation of the cell, read off its solution
space, the hoppings, the orbital centers, the declaration, the lattice and the
operations combined with time reversal. The cell is lowered to the group of the
unitary operations that hold, and each step records what it dropped, by cause.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math

import numpy as np

from . import _bloch
from . import _sitechars
from .refusal import InternalError, Refusal
from ._tolerance import (
    ASSEMBLED, CHARACTER, EXACT, HEADROOM, IN_SPACE, IN_SPAN, SINGULAR, SNAP, ZERO,
    _CLASS_MARGIN, _EIGEN_WINDOW_CEIL, _EXACT_CLOSURE, _IN_BLOCKS,
    _TIME_REVERSAL_SQUARE_MARGIN, eigen_window, noise_floor)
from ._text import _DOUBLE_GROUP, _PER_COSET, _fmt_char, _listed
from ._spacegroup import (
    GroupData, _CosetView, _group_of, _magnetic_type, _pad_cell, _primitive_lattice)
from ._seitz import _record_name
from ._tables import _assign_orbitals
from ._subgroups import (
    _descent_closure, _operation_key, _resolve_closure, _side_preserving,
    _subgroups_inside, _tie_ranked, _tie_reason, _tie_rule_words)
from ._detector import _READOFF_MAX_BYTES
from ._matrices import (
    _ansatz_blocks, _atoms_of_site, _block_members, _channel_permutation,
    _characters_of_subgroup, _commuting_pair_phases, _filled_check,
    _generic_members, _identity_position, _minimal_generators, _multiplets_of,
    _proportionality, _rephasable, _unitarized)
from ._centers import _position_defect
from ._naming import _site_tables
from ._multiplets import _ansatz_defects, _multiplet_blocks
from ._declaration import (
    _content_compatible, _covariant_subgroup, _declared_blocks,
    _declared_site_contents, _joint_reason, _largest_compatible_group,
    _normalize_declaration, _outside_the_group, _returned_rows,
    _verdicts_and_choices)
from ._layers import _decorated_cell
from ._measure import _detection_grid, _max_hamiltonian


@dataclass(frozen=True)
class _DeclarationVerdict:
    """Which operations the declaration breaks, and the subgroup it leaves.

    Attributes
    ----------
    refuted : dict
        ``{ind: entry}`` for each operation the declaration breaks, and for the
        same operation combined with time reversal where the group holds it.
        Each entry names the orbitals, what was declared and what was measured.
    subgroup : tuple of int
        The indices of the operations kept: the unitary ones that carry the
        declared site irreps, each with its product with time reversal where
        the group holds it.
    candidates : tuple
        If nonempty, the largest subgroups that tie, in the tie rule's order.
    rows : tuple of dict
        The declared and the measured characters, as a record returns them.
    """

    refuted: dict
    subgroup: tuple
    candidates: tuple
    rows: tuple


@dataclass(frozen=True, eq=False)
class _Square:
    """An antiunitary operation's square, against the sign the group's table gives it.

    Attributes
    ----------
    ind : int
        The operation's index in the group.
    asked : int
        The sign the group's multiplication table gives the square, ``+1`` or ``-1``.
    measured : float
        The sign of the square of the member the verdict rests on, where that
        square is a multiple of the identity; ``nan`` otherwise.
    obeyed : bool
        Whether a member of the sign asked was found.
    member : ndarray or None
        The member of the sign asked; ``None`` where the solution space holds none.
    """

    ind: int
    asked: int
    measured: float
    obeyed: bool
    member: object


@dataclass(frozen=True, eq=False)
class _AntiunitaryOp:
    """What one round measured of one operation combined with time reversal.

    Attributes
    ----------
    sym : object
        The operation itself.
    entry : dict
        The operation as :attr:`RealizedGroup.beyond_grey` lists it
        (:func:`_antiunitary_entry`).
    member : ndarray or None
        Its matrix; ``None`` where it is refuted.
    space : ndarray or None
        The solution space the matrix is a member of, larger than the matrix
        alone where the Hamiltonian conserves an operator; ``None`` where the
        operation is refuted or the detector recorded no space.
    """

    sym: object
    entry: dict
    member: object
    space: object


@dataclass(frozen=True)
class _AntiunitaryRound:
    """What one round measured of the antiunitary operations it asks last.

    Attributes
    ----------
    group : GroupData
        The round's group.
    ops : dict
        Each operation asked last here, ``{ind: _AntiunitaryOp}``.
    detector : frozenset of int
        The operations the detector refuted, time reversal among them where its
        square is not the sign asked.
    refuted : frozenset of int
        Every operation refuted in this round, by the detector, the verdict
        stage or its own square.
    n : int
        How many operations were asked last here.
    closed : bool
        Whether the operations the detector kept close in this group.
    reversal : bool
        Whether this is the group time reversal was measured on.
    itself : str
        If nonempty, what broke time reversal itself, in words, on the round
        of the group time reversal was measured on.
    returned : dict or None
        Once carried onto the group returned, ``{index here: index in the group
        returned}``; ``None`` before.
    """

    group: object
    ops: dict
    detector: frozenset
    refuted: frozenset
    n: int
    closed: bool
    reversal: bool
    itself: str = ""
    returned: dict | None = None


@dataclass(frozen=True)
class _AntiunitaryReading:
    """Which antiunitary operations hold, read once against the group returned.

    Attributes
    ----------
    verdicts : dict
        What decided each operation, ``{(round, ind): reason}``.
    held : tuple
        The ``(round, ind)`` that hold, in order.
    """

    verdicts: dict
    held: tuple


def _antiunitary_entry(group, validated, ind) -> dict:
    """One antiunitary operation, as :attr:`RealizedGroup.beyond_grey` lists it.

    ``ind`` is the operation's index in ``group``, the group it was measured
    in, and ``validated`` the detector's verdicts there.  Returns its ``ind``;
    ``of``, the number of ``group``; its ``name`` (:func:`_record_name`); its
    ``rotation`` and ``translation`` in the cell's reduced coordinates; and the
    ``residual`` and ``verdict`` the detector measured.  Where the translation
    is not a lattice vector of the cell the name carries it in Seitz form:
    ``{E|1/2,0,0}'``, the half translation of an antiferromagnet's magnetic
    cell combined with time reversal, which ``E'`` alone would print as time
    reversal.
    """
    sym = next(s for s in group._symmetries if int(s.ind) == int(ind))
    return {"ind": int(ind),
            "name": _record_name(sym, group._lattice),
            "of": str(group.number_str),
            "rotation": tuple(tuple(int(round(x)) for x in row)
                              for row in np.asarray(sym.rotation, dtype=float)),
            "translation": tuple(float(x) for x in
                                 np.asarray(sym.translation, dtype=float).reshape(-1)),
            "residual": float(validated.residuals.get(int(ind), float("nan"))),
            "verdict": str(validated._verdicts.get(int(ind), "obeyed"))}


def _antiunitary_settled(group, validated, refuted, square, kept=None, *,
                         tolerances=EXACT) -> _AntiunitaryRound:
    r"""What one round measured of the antiunitary operations it asks last.

    A ``g·T`` is asked last in the round whose descent leaves ``g`` out, since
    the group the descent arrives at does not hold it.  On the group time
    reversal is measured on, every ``g·T`` is asked last: a descent after it,
    by the projective class of spinor states read without spin, can still leave
    ``g`` out.  The round records the measurement and decides nothing; what
    holds is read once, against the group returned (:func:`_antiunitary_held`).

    Parameters
    ----------
    group : GroupData
        The round's group.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    refuted : dict or None
        The verdict stage's refutations on ``group``, by ``.ind``.
    square : _Square or None
        Time reversal's square (:func:`_time_reversal_square`); ``None`` where
        the group has no pure time reversal or the detector refuted it.
    kept : iterable of int, optional
        The unitary operations the round's descent keeps, by ``.ind``.
        ``None`` on the group time reversal is measured on.
    tolerances : Tolerances, default EXACT
        What the call declared, against which a solution space's noise is read.

    Returns
    -------
    _AntiunitaryRound
        For each operation asked, its entry, its matrix and its solution
        space.  The matrix is ``None`` where the operation is refuted by the
        detector or the verdict stage, for time reversal itself where
        ``square`` is not obeyed, and for another operation whose unitary part
        squares to the identity where no member's square ``X X̄`` has the sign
        the group's table gives (:func:`_antiunitary_square`); the member of
        that sign is the matrix kept otherwise.
    """
    antiunitary = {int(s.ind): s for s in group._symmetries if s.time_reversal}
    if kept is None:
        asked = set(antiunitary)
    else:
        kept = {int(i) for i in kept}
        staying = {_operation_key(s)[:2] for s in group._symmetries
                   if not s.time_reversal and int(s.ind) in kept}
        asked = {i for i, s in antiunitary.items() if _operation_key(s)[:2] not in staying}
    detector = {i for i in antiunitary if i in set(validated.broken)}
    reversal = None if square is None else int(square.ind)
    if square is not None and not square.obeyed:
        detector.add(reversal)
    failed = detector | {i for i in antiunitary if i in set(refuted or ())}
    at = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    ops = {}
    for i in sorted(asked):
        member = None if i in failed else validated._intertwiners.get(i)
        if member is not None and i != reversal:
            # its own square, where its unitary part squares to the identity:
            # the member of the sign the table gives it is the matrix kept
            got = _antiunitary_square(group, validated, at[i], tolerances=tolerances)
            if got is not None:
                member = got.member if got.obeyed else None
        if member is None:
            failed.add(i)
        space = None if member is None else (getattr(validated, "_bases", None) or {}).get(i)
        ops[i] = _AntiunitaryOp(
            sym=antiunitary[i], entry=_antiunitary_entry(group, validated, i),
            member=None if member is None else np.asarray(member, dtype=complex),
            space=None if space is None else np.asarray(space, dtype=complex))
    return _AntiunitaryRound(group=group, ops=ops, detector=frozenset(detector),
                             refuted=frozenset(failed), n=len(asked),
                             closed=bool(validated.closed), reversal=kept is None)


def _time_reversal_measured(group, validated, refuted=None, *, tolerances=EXACT) -> tuple:
    """The verdict on time reversal, on the group the unitary operations settle on.

    Time reversal is measured broken where the detector breaks an operation
    combined with it; where no member of its solution space squares,
    ``X X̄``, to the sign the tables ask for, ``+1`` single-valued and ``-1``
    double-valued (:func:`_time_reversal_square`); or where the verdict stage
    refuted it by its spin lift or its site permutation.  Where it is broken,
    what this round measured of every operation combined with time reversal
    is kept (:func:`_antiunitary_settled`), to be read once the group has
    settled (:func:`_beyond_grey`).

    Parameters
    ----------
    group : GroupData
        The group measured, every unitary operation of it obeyed.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    refuted : dict, optional
        The verdict stage's refutations on ``group``, by ``.ind``.
    tolerances : Tolerances, default EXACT
        What the call declared, against which a solution space's noise is read.

    Returns
    -------
    tuple
        ``(record, asked, validated)``.  ``record`` is the time-reversal
        record of the group returned: its ``outcome``, ``"kept"`` or
        ``"measured broken"``, the counts, and the square asked and measured.
        Where time reversal is broken its ``n_obeyed``, ``n_broken`` and
        ``note`` are ``None``, for :func:`_beyond_grey` to write.  ``asked``
        is this round's :class:`_AntiunitaryRound`, its ``itself`` saying in
        words what broke time reversal itself where its square, its spin
        lift or its site permutation did; ``None`` where time reversal is
        kept.  ``validated`` is the detector's record with time reversal's
        member of the sign asked in place (:func:`_time_reversal_member`).
    """
    antiunitary = {int(s.ind): s for s in group._symmetries if s.time_reversal}
    broken = {i for i in validated.broken if i in antiunitary}
    square = (_time_reversal_square(group, validated, tolerances=tolerances)
              if antiunitary else None)
    if square is not None and not square.obeyed:
        broken.add(int(square.ind))
    # ... and time reversal the verdict stage refuted by its tensor structure:
    # its spin lift, i sigma_y on every spin pair, or its site permutation
    lift = (None if square is None
            else (refuted or {}).get(int(square.ind)))
    lifted = (isinstance(lift, dict)
              and str(lift.get("refuted_by", "tensor structure")) == "tensor structure")
    if lifted:
        broken.add(int(square.ind))
    validated = _time_reversal_member(validated, square)
    obeyed = sorted(set(antiunitary) - broken)
    undetermined = tuple(i for i in obeyed
                         if validated._verdicts.get(i) == "undetermined")

    record = {
        "n_antiunitary": len(antiunitary),
        "n_obeyed": len(obeyed), "n_broken": len(broken),
        "undetermined": undetermined, "grey_number_str": str(group.number_str),
        "grey_name": str(group.name),
        "square": (None if square is None else
                   {"asked": int(square.asked), "measured": float(square.measured)}),
    }
    if not antiunitary or not broken:
        record["outcome"] = "kept"
        record["note"] = ("the cell was asked as a grey group and every antiunitary "
                          "operation of it is obeyed, so the realization is grey"
                          if antiunitary else
                          "the cell was asked without time reversal, so there was no "
                          "antiunitary coset to measure")
        return record, None, validated

    record["outcome"] = "measured broken"
    itself = ""
    if lifted and lift.get("own_form") == "site permutation":
        # its matrices carry the sites as another operation does: no square of
        # time reversal's own is read off them
        itself = (" Time reversal itself is refuted by its site permutation: no member of "
                  "its solution space carries each site's states onto its own as time "
                  "reversal does -- the matrices with time reversal's action on k carry "
                  "the sites as time reversal composed with another operation does, a "
                  "translation in an antiferromagnet whose moments, or currents, "
                  "alternate between sites the cell's point set does not tell apart.")
    elif square is not None and not square.obeyed:
        tables = "single-valued" if square.asked > 0 else "double-valued"
        measured = float(square.measured)
        if np.isfinite(measured):
            kind = ("a spinor's time reversal" if measured < 0
                    else "a spin-free time reversal")
            said = (f"with X its intertwiner the square X conj(X) measures "
                    f"{measured:+.6g}, {kind}, where the {tables} tables read here ask "
                    f"for {square.asked:+d} -- a rescaling of X changes the square only "
                    f"by a positive factor, so the sign is the Hamiltonian's own, and no "
                    f"member of its solution space carries the other.")
        else:
            said = (f"no member X of its solution space squares, X conj(X), to the "
                    f"{square.asked:+d} the {tables} tables read here ask for.")
        itself = (
            f" Time reversal itself is broken by its square: {said}"
            + (" A spinless model carrying spinors in its orbital index, as a wannier90 "
               "import does, reads this way; `spin=` declares which orbitals are spin "
               "partners, and the double-valued tables are then read."
               if square.asked > 0 else ""))
    elif lifted:
        itself = (" Time reversal itself is refuted by its spin lift: no member of its "
                  "solution space acts as i sigma_y on every spin pair, as time reversal "
                  "acts on the two states of a spinor. A declared spinor basis whose "
                  "pairs join the states of different Wannier functions reads this way.")
    # What this round measured of every operation combined with time reversal,
    # each asked last here.  The count and the note are written once the group
    # has settled, against the group returned (`_beyond_grey`).
    asked = replace(_antiunitary_settled(group, validated, refuted, square,
                                         tolerances=tolerances),
                    itself=itself)
    record["n_obeyed"] = record["n_broken"] = record["note"] = None
    return record, asked, validated


#: The rule an operation combined with time reversal is held to, as the
#: records' notes say it.
_ANTIUNITARY_RULE = (
    "it holds only where a matrix measured for it and the matrices of the group "
    "returned, up to a one-dimensional character of that group, make a "
    "corepresentation of the magnetic group it generates with that group")


def _carried_onto(group, asked, symprec) -> tuple:
    r"""Each round's record of antiunitary operations, carried onto the group returned.

    Each step keeps a subset of the operations of the round before it, in the
    origin they were measured in (:func:`_shared_origin`), so every unitary
    operation of the group returned is paired with itself in each round's
    group (:func:`_match_operations`).

    Parameters
    ----------
    group : GroupData
        The group returned.
    asked : sequence of _AntiunitaryRound
        What each round measured (:func:`_antiunitary_settled`).
    symprec : float
        The symmetry finder's distance, in the model's units of length, to
        which two operations' translations are matched modulo the lattice.

    Returns
    -------
    tuple of _AntiunitaryRound
        Each record of ``asked`` with ``returned`` set to ``{index in the
        round's group: index in the group returned}``.

    Raises
    ------
    ..refusal.InternalError
        Where a unitary operation of the group returned is not in a round's group.
    """
    unitary = [s for s in group._symmetries if not s.time_reversal]
    out = []
    for rec in asked:
        theirs = [s for s in rec.group._symmetries if not s.time_reversal]
        pairs, lost, _extra = _match_operations(unitary, theirs, symprec, group._lattice)
        if lost:
            raise InternalError(
                f"the group returned, {group.name} #{group.number_str}, holds "
                f"{len(lost)} operation{'' if len(lost) == 1 else 's'} {lost} that "
                f"{rec.group.name} #{rec.group.number_str}, the group of an earlier "
                "round, does not. Report it, with the model and the call.")
        out.append(replace(rec, returned={int(t): int(g) for g, t in pairs}))
    return tuple(out)


def _antiunitary_held(group, matrices, asked, *, tolerances=EXACT) -> _AntiunitaryReading:
    r"""Which antiunitary operations of the cell hold, read against the group returned.

    An operation ``a`` combined with time reversal holds where its matrix
    ``X`` and the matrices ``D`` of the group returned ``G`` make a
    corepresentation of the magnetic group ``M`` that ``a`` generates with
    ``G``: every operation of the coset ``aG`` is kept in its round and ``M``
    holds no unitary operation outside ``G``; ``X X̄ = D(a²)``; and
    ``X D̄(h) X⁻¹ = D(a h a⁻¹)`` for each generator ``h`` of ``G``.  The last
    two carry the signs of the group's multiplication table and are asked up
    to a one-dimensional character of ``G``, each phase to within
    :data:`.._tolerance._CLASS_MARGIN`.  Where the operation's solution space
    holds more than one member, ``X`` is any member.  Read where time reversal
    is broken; nothing is measured here.

    Parameters
    ----------
    group : GroupData
        The group returned, ``G``.
    matrices : dict or None
        The matrices of ``G`` that :func:`.._build._build_rep` assembled, by
        ``.ind``; ``None`` where none were built.
    asked : sequence of _AntiunitaryRound
        What each round measured of the operations combined with time
        reversal it asked last (:func:`_antiunitary_settled`), carried onto
        ``G`` (:func:`_carried_onto`).
    tolerances : Tolerances, default EXACT
        What the call declared, against which a solution space's noise is read.

    Returns
    -------
    _AntiunitaryReading
        For each operation, keyed ``(round, ind)``, what decided it:
        ``"returned"`` where its unitary part is in ``G``; ``"refuted"`` in
        its own round; ``"closure"`` where ``M`` holds a unitary operation
        outside ``G`` or an operation not kept; ``"unread"`` where no matrices
        were built; ``"matrices"`` where they make no corepresentation;
        ``"apart"`` where those that hold are not the operations of one
        ``M``; or ``"held"``.  With it, the ``(round, ind)`` that hold, in
        order.
    """

    # the members of a solution space read off a model symmetric only within
    # its precision: how small a direction of it is none, and how far a unitary
    # read off it may sit outside it
    zero, in_span = tolerances.noise(ZERO), tolerances.noise(IN_SPAN)
    own = group._symmetries
    by_ind = {int(s.ind): s for s in own}
    found = _minimal_generators(range(len(own)), group._get_product_table(get_diff=False),
                                _identity_position(group))
    # the relation asked of every operation, where no small generating set is
    # found, is the one asked of the generators and more
    gens = [int(own[p].ind) for p in (found if found is not None else range(len(own)))]
    characters = None
    verdicts, held, cosets = {}, [], {}
    for r, rec in enumerate(asked):
        cell = rec.group
        syms = cell._symmetries
        at = {int(s.ind): p for p, s in enumerate(syms)}
        prod, _tdiff, spf = cell._get_product_table(get_diff=True)
        ident = _identity_position(cell)
        onto = rec.returned
        inside = {at[i] for i in onto}
        here = {int(g): at[int(i)] for i, g in onto.items()}
        unitary = {_operation_key(s)[:2]: p for p, s in enumerate(syms)
                   if not s.time_reversal}
        # the spin lift each operation of the group returned carries here: the
        # same rotation, its lift the same SU(2) element or its negative
        lift = {}
        for i, g in onto.items():
            q, sign = at[int(i)], 1
            if bool(cell.spinor) and bool(group.spinor):
                mine = np.asarray(syms[q].spinor_rotation, dtype=complex)
                theirs = np.asarray(by_ind[int(g)].spinor_rotation, dtype=complex)
                sign = (1 if np.allclose(mine, theirs, atol=1e-6) else
                        -1 if np.allclose(mine, -theirs, atol=1e-6) else 0)
                if not sign:
                    raise InternalError(
                        f"the operation {g} of {group.name} #{group.number_str} and its "
                        f"pair {i} in {cell.name} #{cell.number_str} are one rotation, and "
                        "their spin lifts are not one SU(2) element up to a sign. Report "
                        "it, with the model and the call.")
            lift[q] = sign

        def _matrix(q):
            """The matrix of the group returned at position ``q`` here, in this group's lift."""
            return lift[q] * np.asarray(matrices[int(onto[int(syms[q].ind)])], dtype=complex)

        def _carried(p, back, X):
            """Whether ``X`` makes a corepresentation with the matrices of ``G``.

            ``X`` is a matrix of the operation at position ``p``, whose inverse
            is at ``back``.
            """
            norm = float(np.linalg.norm(X))
            if not norm > 0.0:
                return False
            X = X * (np.sqrt(X.shape[0]) / norm)
            try:
                inverse = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                return False
            # the square: X conj(X) = D(a^2), with the sign the table gives
            q2 = int(prod[p, p])
            mu, res = _proportionality(X @ X.conj(), int(spf[p, p]) * _matrix(q2))
            if res > _CLASS_MARGIN or abs(mu) <= _CLASS_MARGIN:
                return False
            # how it carries the group returned: X D(h)* X^-1 = D(a h a^-1)
            phases = []
            for g in gens:
                q = here[g]
                pq = int(prod[p, q])
                c = int(prod[pq, back])
                sign = int(spf[p, q]) * int(spf[pq, back]) * int(spf[p, back])
                rho, res = _proportionality(X @ _matrix(q).conj() @ inverse,
                                            sign * _matrix(c))
                if res > _CLASS_MARGIN or abs(rho) <= _CLASS_MARGIN:
                    return False
                phases.append((g, int(onto[int(syms[c].ind)]), rho / abs(rho)))
            square = mu / abs(mu)
            a2 = int(onto[int(syms[q2].ind)])
            # ... up to a one-dimensional character of the group returned
            return any(abs(square - chi[a2]) <= _CLASS_MARGIN
                       and all(abs(lam - chi[h] * chi[c]) <= _CLASS_MARGIN
                               for h, c, lam in phases)
                       for chi in characters)

        def _carrying(p, back, space):
            """The members of ``space`` carrying ``G`` as some one-dimensional character asks.

            ``p`` and ``back`` are the positions of the operation and its
            inverse.  For each character the condition
            ``X D̄(h) = chi(h) chi(c) D(c) X``, ``c = a h a^-1`` with the table's
            sign, on every generator is linear in ``X``.  Returns its solutions
            in the space, a fixed combination of them where there are several,
            and, where ``a²`` is the identity, the unitary nearest the part of
            each that transposition keeps with the table's sign for ``a²``,
            where that is still a member.  Empty where the space has dimension
            below two.
            """
            if space is None or len(space) < 2:
                return []
            space = np.asarray(space, dtype=complex)
            orth, tri = np.linalg.qr(space.reshape(len(space), -1).T)
            span = orth[:, np.abs(np.diag(tri)) > zero * max(1.0, float(np.abs(tri).max()))]
            q2 = int(prod[p, p])
            out = []
            for chi in characters:
                rows = []
                for g in gens:
                    qg = here[g]
                    pq = int(prod[p, qg])
                    c = int(prod[pq, back])
                    sign = int(spf[p, qg]) * int(spf[pq, back]) * int(spf[p, back])
                    lam = sign * chi[g] * chi[int(onto[int(syms[c].ind)])]
                    D, Dc = _matrix(qg).conj(), _matrix(c)
                    rows.append(np.stack([(B @ D - lam * (Dc @ B)).reshape(-1)
                                          for B in space], axis=1))
                _u, sv, vh = np.linalg.svd(np.vstack(rows))
                rank = int(np.sum(sv > zero * max(float(sv[0]), 1.0))) if sv.size else 0
                found = [np.einsum("j,jab->ab", v, space) for v in vh[rank:].conj()]
                if len(found) > 1:
                    weights = np.random.default_rng(17).normal(size=(len(found), 2))
                    found.append(np.einsum("k,kab->ab", weights[:, 0] + 1j * weights[:, 1],
                                           np.asarray(found)))
                for Y in found:
                    out.append(Y)
                    if q2 == ident:
                        U = _unitarized(0.5 * (Y + int(spf[p, p]) * Y.T), tol=zero)
                        if U is not None and float(np.abs(
                                U.reshape(-1) - span @ (span.conj().T @ U.reshape(-1))).max()
                                ) <= in_span:
                            out.append(U)
            return out

        for ind in sorted(rec.ops):
            key = (r, int(ind))
            p = at[int(ind)]
            if unitary.get(_operation_key(syms[p])[:2]) in inside:
                verdicts[key] = "returned"
                continue
            if ind in rec.refuted:
                verdicts[key] = "refuted"
                continue
            # The group it makes with the group returned G is G and the coset aG
            # where a squares into G and carries G onto itself.  Every operation
            # of aG must also be kept in this round, the last that asked it.
            back = next(q for q in range(len(syms)) if int(prod[p, q]) == ident)
            coset = {int(prod[p, q]) for q in inside}
            if (int(prod[p, p]) not in inside
                    or any(int(prod[int(prod[p, here[g]]), back]) not in inside
                           for g in gens)
                    or any(int(syms[q].ind) in rec.refuted
                           or int(syms[q].ind) not in rec.ops for q in coset)):
                verdicts[key] = "closure"
                continue
            if matrices is None or any(int(g) not in matrices for g in onto.values()):
                verdicts[key] = "unread"
                continue
            if characters is None:
                characters = _characters_of_subgroup(group, sorted(by_ind), roots=True)
            op = rec.ops[ind]
            if not (_carried(p, back, op.member)
                    or any(_carried(p, back, X)
                           for X in _carrying(p, back, op.space))):
                verdicts[key] = "matrices"
                continue
            verdicts[key] = "held"
            held.append(key)
            cosets[key] = frozenset(int(syms[q].ind) for q in coset)
    # ... and those that hold are the operations of one magnetic group
    if held and (len({r for r, _i in held}) > 1
                 or {i for _r, i in held} != cosets[held[0]]):
        for key in held:
            verdicts[key] = "apart"
        held = []
    return _AntiunitaryReading(verdicts=verdicts, held=tuple(held))


def _beyond_grey(group, matrices, asked, record, *, tolerances=EXACT) -> tuple:
    r"""The antiunitary operations that hold while time reversal is broken.

    Where ``record`` says time reversal is measured broken, every operation
    combined with time reversal that a round measured is read once against the
    group returned, by the rule of :func:`_antiunitary_held`.  Otherwise
    nothing is read.

    Parameters
    ----------
    group : GroupData
        The group returned.
    matrices : dict or None
        Its matrices from :func:`.._build._build_rep`, by ``.ind``; ``None``
        where none were built.
    asked : sequence of _AntiunitaryRound
        What the rounds measured, carried onto ``group`` (:func:`_carried_onto`).
    record : dict or None
        The time-reversal record :func:`_time_reversal_measured` returned.
    tolerances : Tolerances, default EXACT
        What the call declared, against which a solution space's noise is read.

    Returns
    -------
    tuple
        ``(record, beyond)``.  Where time reversal is broken, ``record`` is
        completed: ``n_obeyed`` and ``n_broken`` count the operations of the
        group time reversal was measured on that hold and that do not, and
        ``note`` says what decided them; otherwise it is returned as handed
        in.  ``beyond`` is what :attr:`RealizedGroup.beyond_grey` is made
        from: the operations that hold, each by its index in the group it was
        measured in, with the magnetic type they make; ``None`` where none
        holds.
    """
    if record is None or record.get("outcome") != "measured broken":
        return record, None
    read = _antiunitary_held(group, matrices, asked, tolerances=tolerances)
    verdicts, held = read.verdicts, read.held
    r_t = next(r for r, rec in enumerate(asked) if rec.reversal)
    reversal = asked[r_t]
    # the count is of the group time reversal was measured on, and the note
    # says what decided the operations of it the detector kept
    mine = [v for (r, i), v in verdicts.items()
            if r == r_t and i not in reversal.detector]
    n_all = int(record["n_antiunitary"])
    n_held = mine.count("held")
    unread = mine.count("unread")
    returned = mine.count("returned")
    failing = len(mine) - n_held - unread
    refuted = n_all - n_held
    # where every one is refuted nothing is counted, and the first count the
    # note does make names the unit, once
    unit = [] if refuted == n_all else [True]

    def named(n):
        if unit:
            return ""
        unit.append(True)
        if n == 1:
            return (", a coset representative of the lattice translations"
                    + (", the single-valued group's, standing for itself and its negative "
                       "in the double group" if group.spinor else "") + ",")
        return (f", {_PER_COSET}" + (f", {_DOUBLE_GROUP}" if group.spinor else "") + ",")

    said = ""
    if failing == refuted:
        said += ((" It has" if failing == 1 else " Every one of them has")
                 + " a symmetry matrix and fails the rule an operation combined "
                 f"with time reversal is held to: {_ANTIUNITARY_RULE}.")
    elif failing:
        said += (f" Of those, {failing}{named(failing)} {'has' if failing == 1 else 'have'} "
                 f"a symmetry matrix and {'fails' if failing == 1 else 'fail'} the rule an "
                 f"operation combined with time reversal is held to: {_ANTIUNITARY_RULE}.")
    if failing and returned == failing:
        said += ((" It combines" if failing == 1 else " Each of them combines")
                 + " time reversal with an operation of the group returned, so it "
                 "generates time reversal with its own unitary part, and time "
                 "reversal is refuted.")
    elif returned:
        said += (f" {returned} of them{named(returned)} "
                 f"combine{'s' if returned == 1 else ''} time "
                 "reversal with an operation of the group returned, so each generates "
                 "time reversal with its own unitary part, and time reversal is "
                 "refuted.")
    if n_held:
        said += (f" The other {n_held} {'holds' if n_held == 1 else 'hold'} by "
                 + ("that rule" if failing else
                    "the rule an operation combined with time reversal is held to: "
                    + _ANTIUNITARY_RULE)
                 + ("; it combines" if n_held == 1 else "; each combines")
                 + " time reversal with an operation outside the group returned, and "
                 f"`beyond_grey` names {'it' if n_held == 1 else 'them'}.")
    if unread:
        said += (f" {unread} more{named(unread)} combine{'s' if unread == 1 else ''} time "
                 "reversal with an operation outside the group returned and "
                 f"{'is' if unread == 1 else 'are'} not read against it: the matrices of "
                 "the group returned to read them with were not built.")
    record = dict(record, n_obeyed=n_held, n_broken=refuted, note=(
        "time reversal measured broken: "
        + (f"every operation of {record['grey_name']} #{record['grey_number_str']} "
           "combined with time reversal is refuted by the Hamiltonian"
           if refuted == n_all else
           f"{refuted} of the {n_all} operation{'' if n_all == 1 else 's'} of "
           f"{record['grey_name']} #{record['grey_number_str']} combined with time "
           f"reversal, {_PER_COSET}"
           + (f" and {_DOUBLE_GROUP}" if group.spinor else "")
           + ", are refuted by the Hamiltonian")
        + ", so the realized group is unitary and the cell was re-identified without "
        "time reversal."
        + said
        + " Labels, EBR rows and the indicator that follow are the unitary group's "
        f"own.{reversal.itself}"))
    if not held:
        return record, None

    rec = asked[held[0][0]]
    inds = [i for _r, i in held]
    entries = tuple(rec.ops[i].entry for i in inds)
    kind = _magnetic_type(_CosetView([rec.ops[i].sym for i in inds]))
    n = len(entries)
    beyond = {
        "operations": entries,
        "n_operations": n,
        "n_measured": sum(int(r.n) for r in asked),
        "magnetic_type": kind if kind in ("III", "IV") else "magnetic, beyond grey",
        "closed": bool(rec.closed),
        "note": (
            f"{n} operation{'' if n == 1 else 's'} of {rec.group.name} "
            f"#{rec.group.number_str} combined with time reversal, "
            + ("a coset representative of the lattice translations" if n == 1
               else _PER_COSET)
            + ((", the single-valued group's, standing for itself and its negative in "
                "the double group" if n == 1 else f", {_DOUBLE_GROUP}")
               if group.spinor else "")
            + f", {'was' if n == 1 else 'were'} measured to hold "
            "while time reversal itself was refuted, read once, against the group "
            "returned, by the rule an operation combined with time reversal is held "
            f"to: {_ANTIUNITARY_RULE}. "
            + ("It combines time reversal with a unitary operation outside the group "
               "returned, which a descent left out, and is named by its index in the group "
               "it was measured in." if n == 1 else
               "All of them combine time reversal with a unitary operation outside the "
               "group returned, which a descent left out, and are named by their index in "
               "the group they were measured in.")
            + " So the "
            "Hamiltonian's magnetic group is larger than the group "
            "returned here. The returned group is the largest one inside this "
            "package's validated universe -- the 230 space groups and their grey "
            "extensions -- and the operations listed above are not represented, "
            "because magnetic groups beyond grey are not validated here. Every "
            "returned label, EBR row and indicator is the unitary group's own and is "
            "therefore in scope; what is out of scope is what those operations would "
            "additionally constrain."),
    }
    return record, beyond


def _precision_holding(steps, scale_of):
    r"""The least precision at which every operation a descent broke would hold.

    An operation broken by its tensor structure has a ``covariance`` in its
    entry: how far the member of its solution space nearest its own form, taken
    in that form, carries ``H(k)`` onto ``H(Rk)`` for the operation ``{R|t}``,
    or ``H(k)*`` onto ``H(-Rk)`` for one combined with time reversal.  Such
    an operation holds
    under a declared precision ``p`` whose gate, ``10 · p · max|H|``, reaches
    that far.  A precision is named only where the detector's matching window
    at it (:func:`.._tolerance.eigen_window`) stays below its ceiling.

    Parameters
    ----------
    steps : sequence of dict
        The descent's steps, each with the entries it ``dropped``.
    scale_of : callable
        Returns ``max|H|``; called only where every operation broken was
        broken by its tensor structure.

    Returns
    -------
    tuple or None
        ``(worst, p)``: the largest such covariance, in the units of ``H``,
        and the least ``p`` whose gate holds it, rounded up to two significant
        figures.  ``None`` where an operation was broken by anything other
        than its tensor structure, the closure and the tie rule aside, or
        carries no ``covariance``; where none was broken; where ``max|H|`` is
        zero; or where the window at ``p`` reaches its ceiling.
    """
    broken = [b for st in steps for b in st["dropped"]
              if b["refuted_by"] not in ("closure", "tie")]
    if not broken or any(b["refuted_by"] != "tensor structure" or b.get("covariance") is None
                         for b in broken):
        return None
    scale = float(scale_of())
    if not scale > 0.0:
        return None
    worst = max(float(b["covariance"]) for b in broken)
    least = worst / (HEADROOM * scale)
    figure = 10.0 ** (math.floor(math.log10(least)) - 1)
    least = math.ceil(least / figure) * figure
    if eigen_window(HEADROOM * least) >= _EIGEN_WINDOW_CEIL:
        return None
    return worst, least


# --------------------------------------------------------------------------
# the unitary descent: the cell lowered to the group the Hamiltonian obeys
# --------------------------------------------------------------------------

#: The generic point whose orbit is added to a cell to lower its group.  Its
#: reduced coordinates are unlike any Wyckoff coordinate, off every axis, plane
#: and diagonal, so that its stabilizer is trivial: an operation that preserves
#: the orbit of such a point is one of the set the orbit was generated from,
#: and the construction is exact.  The third entry is used only along a
#: periodic direction; along a vacuum one the point is placed by the two
#: numbers below.
_DESCENT_POINT = (0.1234, 0.2718, 0.0577)

#: Along a vacuum direction the generic point sits at this fraction of the
#: cell's own half-extent, measured from the cell's own mid-plane.  Inside the
#: extent, :func:`_pad_cell` centers the decorated cell exactly as it centers
#: the parent, its vacuum being the model's own whatever the cell holds, so the
#: descended group is read in the frame the parent was.
_DESCENT_PAD_FRACTION = 0.1234

#: Where the cell has no extent along a vacuum direction, every point in one
#: plane as on a free-standing flat layer, this reduced offset from the plane
#: is used instead: a point in the plane would leave ``σ_h`` unbroken, and the
#: descent could never remove it.
_DESCENT_PAD_OFFSET = 0.0577

#: The species of the added orbit, one ``cell_typat`` value distinct from every
#: orbital and decoration species (a suffix is appended where it collides), so
#: that the orbit is one species and its own orbit under the kept set.
_DESCENT_SPECIES = "descent"

#: The generic point the cell reduction of step 1 builds its orbit from, and
#: its species.  A cell the declaration has reduced is the base every
#: later descent builds on, so the two orbits share a cell and must not share a
#: point: these periodic coordinates are as unlike :data:`_DESCENT_POINT`'s as
#: they are unlike any Wyckoff coordinate.  The coordinate along the vacuum is
#: placed by the same two numbers above, and the vacuum is the model's own
#: (:func:`_pad_cell`), so adding either orbit leaves the lattice, its vacuum
#: included, where it was.
_DECLARED_POINT = (0.3141, 0.1592, 0.0919)

_DECLARED_SPECIES = "declared"

#: The generic points a descent, and the cell reduction of step 1, build their
#: orbit from, in order, the first of each being the point above.  The cell an
#: orbit decorates carries exactly the operations the orbit was generated
#: from, unless the identification reads it as still non-primitive or with
#: other operations, an accident of that point which the next point of the
#: sequence does not share.  All six are distinct from one another and from
#: every Wyckoff coordinate, so no orbit of one sequence meets an orbit of the
#: other.
_DESCENT_POINTS = (_DESCENT_POINT, (0.3719, 0.0857, 0.2461), (0.2087, 0.4421, 0.1693))

_DECLARED_POINTS = (_DECLARED_POINT, (0.0671, 0.3853, 0.2297), (0.4462, 0.2239, 0.0383))


def _descent_frame(cell, group) -> tuple:
    r"""How the cell's own reduced coordinates map onto those of the group's 3D cell.

    :func:`_pad_cell` scales every non-periodic lattice vector of ``cell``
    until the vacuum reaches the vacuum length of ``group``, and centers the
    positions on the origin along it; the operations of ``group`` are written
    in that frame.  The map is ``y = x / scale − offset`` modulo 1, ``scale``
    read off the two lattices and ``offset`` off the first orbital; along a
    periodic direction they are ``1`` and ``0``.

    ``cell`` is the point set ``group`` was identified on, as a model.
    Returns ``(scale, offset, positions, vacuum_dirs)``, the last two as
    :func:`_pad_cell` returns them, from which the cell's own extent along a
    vacuum direction is read.
    """
    lat_pad, pos_pad, padded = _pad_cell(cell, group._vacuum_length)
    lat_cell = np.asarray(cell.lat_vecs, dtype=float)
    scale = np.array([float(np.linalg.norm(lat_pad[d])
                            / max(np.linalg.norm(lat_cell[d]), 1e-30))
                      for d in range(3)])
    offset = np.asarray(cell.orb_vecs, dtype=float)[0] / scale - pos_pad[0]
    return scale, offset - np.rint(offset), pos_pad, tuple(padded)


def _generic_orbit(cell, group, operations, seed=_DESCENT_POINT) -> tuple:
    r"""The orbit of one generic point under ``operations``, in both frames.

    The point is ``seed`` along every periodic direction and sits off the
    cell's own mid-plane along a vacuum one (:data:`_DESCENT_PAD_FRACTION`,
    :data:`_DESCENT_PAD_OFFSET`).  Its stabilizer is trivial, so the cell with
    the orbit added carries exactly the operations the orbit was generated
    from, up to an accident of the point, which :func:`_orbit_cell` checks for.

    Parameters
    ----------
    cell : pythtb.TBModel
        The point set the orbit is added to, as a model.
    group : GroupData
        The group identified on ``cell``.
    operations : sequence
        Operations of ``group``, which generate the orbit.
    seed : tuple of float, default _DESCENT_POINT
        The point's periodic coordinates: :data:`_DESCENT_POINT` for a
        descent, :data:`_DECLARED_POINT` for the cell reduction of step 1.

    Returns
    -------
    tuple
        ``(point, orbit_padded, orbit_cell)``: the point and its orbit in
        reduced coordinates of the group's 3D cell, and the orbit in the
        cell's own, ready for :func:`_decorated_cell`.  Along a vacuum
        direction the orbit is written on the branch about zero, as the cell's
        own positions are.
    """
    scale, offset, pos_pad, padded = _descent_frame(cell, group)
    point = np.array([float(seed[d]) for d in range(3)])
    for d in padded:
        signed = pos_pad[:, d] - np.rint(pos_pad[:, d])
        lo, hi = float(signed.min()), float(signed.max())
        point[d] = 0.5 * (lo + hi) + (_DESCENT_PAD_FRACTION * 0.5 * (hi - lo)
                                      if hi - lo > 1e-9 else _DESCENT_PAD_OFFSET)

    orbit: list = []
    for sym in operations:
        y = (np.asarray(sym.rotation, dtype=float) @ point
             + np.asarray(sym.translation, dtype=float).reshape(-1)) % 1.0
        for d in padded:                       # the cell's own branch, not [0, 1)
            y[d] -= np.rint(y[d])
        if not any(np.abs((y - o + 0.5) % 1.0 - 0.5).max() <= 1e-9 for o in orbit):
            orbit.append(y)
    orbit_padded = np.array(orbit) if orbit else np.zeros((0, 3))
    return point, orbit_padded, (orbit_padded + offset) * scale


def _mirror_partners(group, validated, assignment) -> dict | None:
    r"""Each operation composed with a flat layer's horizontal mirror, and its partner.

    Where every orbital of a layer lies in one plane, the horizontal mirror
    ``σ_h`` moves no orbital and no in-plane ``k``, so an operation ``g`` and
    its product ``σ_h·g`` share an action on ``k`` and a site permutation, and
    the hoppings of a Hamiltonian without spin cannot tell them apart.  The
    sign each orbital carries under ``σ_h`` does (:func:`_mirror_sign`).

    ``group`` is the round's group, ``validated`` the detector's verdicts on
    it, and ``assignment`` the :class:`.._spacegroup.OrbitalAssignment` whose
    site permutations are compared.  Returns ``{ind of σ_h·g: ind of g}``,
    ``g`` keeping the layer's side, one entry for each class of operations
    sharing an action on ``k`` that holds an undetermined operation; the
    identity's class gives the mirror.  Returns ``{}`` where nothing is
    undetermined.  Otherwise returns ``None`` where the group is not set in
    vacuum along exactly one direction, or where such a class is not two
    undetermined operations, one keeping the side and one turning it over,
    with one site permutation.
    """
    undetermined = {int(i) for i in validated.undetermined}
    if not undetermined:
        return {}
    padded = tuple(int(d) for d in getattr(group, "_vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return None
    sides = {int(s.ind): _side_preserving(s, padded[0]) for s in group._symmetries}
    perm = np.asarray(assignment.permutations, dtype=int)
    at = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    partners = {}
    for members in validated._k_action_classes:
        pair = [int(i) for i in members if int(i) in undetermined]
        if not pair:
            continue
        if (len(pair) != 2 or len(members) != 2
                or any(sides.get(i) is None for i in pair) or sides[pair[0]] == sides[pair[1]]):
            return None
        keep, flip = pair if sides[pair[0]] else pair[::-1]
        if not np.array_equal(perm[at[keep]], perm[at[flip]]):
            return None
        partners[flip] = keep
    return partners


def _mirror_sign(group, partners, decls=()) -> dict:
    r"""The sign chosen for the horizontal mirror on a flat layer read without spin.

    ``σ_h`` moves nothing on a flat layer, so its matrix commutes with
    ``H(k)`` and is a multiple of the identity on each coupled part; the
    hoppings leave its sign open.  The sign is the parity a declared site
    states for the mirror (``s`` even, ``p_z`` odd), and otherwise the even
    one, by the lowest-angular-momentum rule.  It is never refused.

    Parameters
    ----------
    group : GroupData
        The round's group.
    partners : dict
        ``{ind of σ_h·g: ind of g}`` (:func:`_mirror_partners`); the mirror
        is the partner of the identity.
    decls : sequence, default ()
        The round's declarations, as :func:`_declared_parts_of` reads them.

    Returns
    -------
    dict
        ``{"sign", "determined_by"}``: ``+1`` or ``-1``, and
        ``"declaration"`` or ``"lowest-l"``.
    """

    identity = int(group._symmetries[_identity_position(group)].ind)
    mirror = next((f for f, g in partners.items() if g == identity), None)
    if mirror is not None:
        for content in _declared_parts_of(decls or ()):
            chi = (getattr(content, "constraints", None) or {}).get(mirror)
            if chi is None:
                continue
            parity = complex(chi).real / max(1, int(content.dim))
            if abs(abs(parity) - 1.0) <= 1e-6:
                return {"sign": 1 if parity > 0 else -1, "determined_by": "declaration"}
    return {"sign": 1, "determined_by": "lowest-l"}


def _class_members(group, validated, partners=None, sign=1):
    r"""One matrix per operation of a spinless reading, to read its projective class.

    The matrices are the detector's own members of the unitary operations,
    read only where each is fixed up to a phase: a single-valued group, every
    operation of it kept, only the identity commuting with ``H``, and each
    operation alone in its action on ``k`` or, on a flat layer, paired with
    its product with the horizontal mirror.  Such a product is given ``sign``
    times its partner's matrix, ``D(σ_h·g) = D(σ_h) D(g)``.

    ``group`` is the round's group, ``validated`` the detector's verdicts on
    it, ``partners`` the pairs of :func:`_mirror_partners` and ``sign`` the
    mirror's (:func:`_mirror_sign`).  Returns ``(D, positions)``, ``D`` each
    unitary operation's matrix by its position in ``group`` and ``positions``
    the positions of the unitary operations; ``None`` where the class is not
    read here.
    """

    syms = group._symmetries
    if bool(group.spinor):
        return None
    inds = [int(s.ind) for s in syms]
    if set(inds) - {int(i) for i in validated._indices}:
        return None
    partners = dict(partners or {})
    if {int(i) for i in validated.undetermined} - set(partners) - set(partners.values()):
        return None
    ident = _identity_position(group)
    if int((validated._nullspace_dims or {}).get(int(syms[ident].ind), 0) or 0) != 1:
        return None
    members = validated._intertwiners or {}
    positions = [p for p, s in enumerate(syms) if not s.time_reversal]
    D = {}
    for p in positions:
        ind = inds[p]
        member = members.get(partners.get(ind, ind))
        if member is None:
            return None
        D[p] = (int(sign) if ind in partners else 1) * np.asarray(member, dtype=complex)
    return D, positions


def _class_kept(group, D, positions, reversed_pairs=()):
    r"""The largest subgroup on which the matrices ``D`` are an ordinary representation.

    ``D`` holds one matrix per unitary position in ``group``, and
    ``positions`` lists the positions it covers (:func:`_class_members`).
    The pairs of commuting operations whose matrices commute only up to a
    phase the multiplication table lacks are read first
    (:func:`.._matrices._commuting_pair_phases`).  ``reversed_pairs`` holds
    the pairs ``(g, t, -1)`` of a group with time reversal, ``t`` time
    reversal's position, for each ``g`` whose product with time reversal
    squares to the other sign than the table's (:func:`_reversed_pairs`).
    The subgroup kept is one of the subgroups of ``group`` among the
    operations at ``positions`` that hold no such pair and no such ``g``
    (:func:`_subgroups_inside`): of the largest order at which ``D`` is a
    representation once rephased (:func:`.._matrices._rephasable`), and the
    first by the tie rule (:func:`_tie_ranked`) where several of that order
    are.  Each phase is read to within :data:`.._tolerance._CLASS_MARGIN`.

    Returns ``(pairs, kept, ranked)``: the pairs ``(a, b, phase)`` by
    position, those with time reversal last, the ``.ind`` of the subgroup
    kept, and the subgroups of the kept order in the rule's order.  Returns
    ``None`` where there is no pair of either kind.
    """

    syms = group._symmetries
    prod, _tdiff, spf = group._get_product_table(get_diff=True)
    commuting = _commuting_pair_phases(D, positions, syms, prod, spf, tol=_CLASS_MARGIN)
    pairs = tuple(commuting) + tuple(reversed_pairs)
    if not pairs:
        return None
    at = {int(s.ind): p for p, s in enumerate(syms)}
    reversed_out = {int(g) for g, _t, _c in reversed_pairs}
    subs = _subgroups_inside(group, [int(syms[p].ind) for p in positions
                                     if int(p) not in reversed_out], avoid={
        frozenset((int(syms[a].ind), int(syms[b].ind))) for a, b, _c in commuting})
    kept, ranked = frozenset({int(syms[_identity_position(group)].ind)}), ()
    for size in sorted({len(s) for s in subs}, reverse=True):
        good = [s for s in subs if len(s) == size
                and _rephasable(D, [at[i] for i in s], group, prod, spf, tol=_CLASS_MARGIN)]
        if good:
            ranked = _tie_ranked(group, good)
            kept = frozenset(ranked[0])
            break
    return pairs, kept, ranked


def _class_record(group, pairs, kept, ranked, *, rule_group=None, rule_ranked=None):
    r"""The operations a class reading leaves out, each with the pair that names it, and the step's record.

    Parameters
    ----------
    group : GroupData
        The group the class was read on.
    pairs : sequence of tuple
        :func:`_class_kept`'s pairs ``(a, b, phase)``, by position in ``group``,
        ``b`` time reversal for an operation whose product with it squares to
        the other sign than the table's.
    kept : set of int
        The ``.ind`` of the subgroup kept.
    ranked : sequence
        The subgroups of the kept order in the tie rule's order, the one kept
        first.
    rule_group : GroupData, default None
        The group the tie was settled in, where that is not ``group``.
        ``None`` is ``group``.
    rule_ranked : sequence, default None
        The tied subgroups as ``rule_group`` holds them.  ``None`` is
        ``ranked``.

    Returns
    -------
    tuple
        ``(refuted, record)``.  ``refuted`` maps the ``.ind`` of each unitary
        operation of ``group`` outside ``kept`` to ``refuted_by = "projective
        class"``, the ``pair`` that names it (one with exactly one operation
        kept or time reversal where there is one), their ``names``, the
        ``commutation_phase``, whether that phase is the sign the two
        operations' spin-1/2 lifts carry (``spin_half``, false for a pair with
        time reversal), and a ``note``; an operation in no pair has ``None``
        for ``pair``, ``names``, ``commutation_phase`` and ``spin_half``.  On
        a group with time reversal each such operation's product with time
        reversal is refuted with it, under the same pair and a note of its
        own, and no other operation combined with time reversal is.
        ``record`` holds every pair (``pairs``), the ``.ind`` kept (``kept``)
        and the tied subgroups (``tied``, empty where one subgroup is of the
        kept order); where several tie, also the rule's keys as
        ``rule_group`` reads them (``rule``) and why it kept the first
        (``reason``).
    """

    syms = group._symmetries
    _prod, _tdiff, spf = group._get_product_table(get_diff=True)
    at = {int(s.ind): p for p, s in enumerate(syms)}
    lattice = np.asarray(group._lattice, dtype=float)

    def _name(i):
        return _record_name(syms[at[int(i)]], lattice)

    witnesses = []
    for a, b, c in pairs:
        if bool(syms[b].time_reversal):
            # g paired with time reversal: g T squares to the other sign than
            # the table's, which no spin-1/2 rotation is asked about
            half = None
        else:
            Sa = np.asarray(syms[a].spinor_rotation, dtype=complex)
            Sb = np.asarray(syms[b].spinor_rotation, dtype=complex)
            half, _res = _proportionality(Sa @ Sb,
                                          int(spf[a, b]) * int(spf[b, a]) * (Sb @ Sa))
        witnesses.append({"pair": (int(syms[a].ind), int(syms[b].ind)),
                          "names": (_name(syms[a].ind), _name(syms[b].ind)),
                          "commutation_phase": complex(c),
                          "spin_half": bool(half is not None
                                            and abs(complex(half) - complex(c))
                                            <= _CLASS_MARGIN)})

    # the unitary operations outside the subgroup kept, each named by a pair;
    # on a group with time reversal, an operation combined with it goes where
    # its unitary part does, below
    reversing = {int(s.ind) for s in syms if s.time_reversal}
    staying = set(kept) | reversing
    refuted = {}
    for i in sorted(set(at) - set(kept) - reversing):
        mine = [w for w in witnesses if i in w["pair"]]
        best = next((w for w in mine if sum(1 for j in w["pair"] if j in staying) == 1),
                    mine[0] if mine else None)
        if best is None:
            refuted[i] = {"refuted_by": "projective class", "pair": None, "names": None,
                          "commutation_phase": None, "spin_half": None, "note": (
                              f"no subgroup holding {_name(i)} together with the operations "
                              "kept carries a single-valued representation on these states")}
            continue
        a, b = best["names"]
        if best["pair"][1] in reversing:
            refuted[i] = dict(best, refuted_by="projective class", note=(
                f"{a} combined with time reversal squares to -1 on these states, a sign no "
                "rescaling of its matrix changes, where a single-valued representation "
                f"gives +1. No single-valued representation of a group holding {a} and "
                f"time reversal exists on these states, so {a} is not an operation of the "
                "spinless crystal: the largest subgroup on which the states carry one, "
                f"time reversal with it, is kept, and it does not hold {a}."))
            continue
        refuted[i] = dict(best, refuted_by="projective class", note=(
            f"{a} and {b} commute, and on these states their matrices do not: "
            f"D({a}) D({b}) = {_fmt_char(best['commutation_phase'])} "
            f"D({b}) D({a}), a phase no rescaling of either matrix removes"
            # said of spinor states read without their spin, which lose time
            # reversal first: on a group that keeps it the states are not spinors
            + (" -- the sign the spin-1/2 lifts of the two operations carry"
               if best["spin_half"] and not reversing else "")
            + f". No single-valued representation of a group holding both exists on "
            f"these states, so {_name(i)} is not an operation of the spinless crystal: "
            "the largest subgroup on which the states carry one is kept, and it does not "
            f"hold {_name(i)}."))
    # on a group with time reversal, each unitary operation left out takes its
    # product with time reversal with it: a group holding that product and time
    # reversal holds the operation
    t = _reversal_position(group)
    if t is not None:
        prod = group._get_product_table(get_diff=False)
        for i, rec in list(refuted.items()):
            mate = int(syms[int(prod[t, at[i]])].ind)
            refuted[mate] = dict(rec, note=(
                f"{_name(i)} combined with time reversal, left out with {_name(i)}: a "
                f"group holding it and time reversal holds {_name(i)}"))
    record = {"pairs": tuple(witnesses), "kept": tuple(sorted(kept)),
              "tied": tuple(ranked) if len(ranked) > 1 else ()}
    if len(ranked) > 1:
        # the rule's keys as the group they were read in reads them, and why it
        # kept the first, for the descent's note
        settled = rule_group if rule_group is not None else group
        record["rule"] = _tie_rule_words(settled)
        record["reason"] = _tie_reason(settled,
                                       rule_ranked if rule_ranked is not None else ranked)
    return refuted, record


def _reversed_pairs(group, validated, positions, partners, *, tolerances=EXACT):
    r"""Each unitary operation whose product with time reversal squares to the other sign.

    For a unitary ``g`` that squares to the identity modulo the lattice,
    ``(g·T)²`` is the identity, and the sign of the square ``X X̄`` of the
    member of ``g·T`` is the Hamiltonian's own (:func:`_antiunitary_square`).
    Where no member has the table's sign and the square is measured with the
    other, ``-1`` on a single-valued
    group, no single-valued representation of a group holding ``g`` and time
    reversal exists on the states: with ``D(g)`` scaled to square to ``+1``,
    ``D(g)`` and time reversal anticommute.  On a flat layer an operation
    composed with the horizontal mirror takes its partner's reading, as its
    matrix does (:func:`_class_members`).

    ``group`` is the round's group, holding time reversal; ``validated`` the
    detector's verdicts on it; ``positions`` the unitary positions the class
    is read on; ``partners`` the pairs of :func:`_mirror_partners`; and
    ``tolerances`` the call's declared tolerances.  Returns the pairs
    ``(g, t, -1)`` by position, ``t`` time reversal's, in the order of
    ``positions``; empty where the group has no pure time reversal.
    """
    syms = group._symmetries
    t = _reversal_position(group)
    if t is None:
        return ()
    prod = group._get_product_table(get_diff=False)
    at = {int(s.ind): p for p, s in enumerate(syms)}
    found = []
    for p in positions:
        # the partner's own g T, where the operation shares its measurement
        source = at[int((partners or {}).get(int(syms[p].ind), int(syms[p].ind)))]
        got = _antiunitary_square(group, validated, int(prod[source, t]),
                                  tolerances=tolerances)
        # a square measured with the other sign, not one left unread by noise
        if (got is not None and not got.obeyed and np.isfinite(got.measured)
                and got.measured * got.asked < 0):
            found.append((int(p), int(t), complex(-1.0)))
    return tuple(found)


def _projective_class_refuted(group, validated, *, assignment=None, decls=(),
                              refuted=None, square=None, tolerances=EXACT):
    r"""The operations of a spinless reading that no single-valued representation carries.

    The detector's members of the unitary operations of a single-valued group
    are the matrices of a projective representation.  Where its class is not
    the multiplication table's, the unitary operations outside the largest
    subgroup that carries an ordinary representation, the first by the tie
    rule, are refuted (:func:`_class_kept`).  The class is read only where
    every member is fixed up to a phase (:func:`_class_members`).  On a flat
    layer each operation composed with the horizontal mirror is given its
    partner's matrix times the mirror's sign, ``D(σ_h·g) = D(σ_h) D(g)``
    (:func:`_mirror_partners`, :func:`_mirror_sign`); the sign commutes with
    every matrix, so the class does not depend on it.

    On a group that holds time reversal the class is read only where time
    reversal's own square has the sign the tables ask for and no operation
    combined with time reversal is broken or refuted.  The subgroup kept then
    also holds no unitary ``g`` whose ``g·T`` squares to the other sign
    (:func:`_reversed_pairs`), and the operations combined with time
    reversal go where their unitary parts do.

    Parameters
    ----------
    group : GroupData
        The round's group.
    validated : ValidatedGroup
        The detector's verdicts on ``group``, whose members are read.
    assignment : OrbitalAssignment, default None
        The orbitals' orbits in ``group``, whose site permutations pair an
        operation with its product with the mirror.  ``None`` pairs nothing,
        and the class is then not read where an operation is undetermined.
    decls : sequence, default ()
        The round's declarations, as :func:`_declared_parts_of` reads them,
        which give the mirror's parity where they state one.
    refuted : dict, default None
        The verdict stage's refutations on ``group``, by ``.ind``.
    square : _Square, default None
        Time reversal's square (:func:`_time_reversal_square`); ``None``
        where ``group`` has no pure time reversal or the detector refuted it.
    tolerances : Tolerances, default EXACT
        The call's declared tolerances, against which each square is read.

    Returns
    -------
    tuple
        ``(refuted, record)`` as :func:`_class_record` returns them, the
        record holding, on a flat layer, the mirror's sign and how it was
        determined (``horizontal_mirror``).  ``({}, None)`` where the class is
        the table's, or is not read here.
    """
    if bool(group.spinor):
        return {}, None
    reversing = {int(s.ind) for s in group._symmetries if s.time_reversal}
    if reversing and (square is None or not square.obeyed
                      or reversing & (set(validated.broken) | set(refuted or ()))):
        return {}, None
    partners = {}
    if validated.undetermined:
        partners = (None if assignment is None
                    else _mirror_partners(group, validated, assignment))
        if partners is None:
            return {}, None
    mirror = _mirror_sign(group, partners, decls) if partners else None
    members = _class_members(group, validated, partners,
                             sign=1 if mirror is None else mirror["sign"])
    if members is None:
        return {}, None
    D, positions = members
    reversed_pairs = (_reversed_pairs(group, validated, positions, partners,
                                      tolerances=tolerances) if reversing else ())
    got = _class_kept(group, D, positions, reversed_pairs)
    if got is None:
        return {}, None
    out, record = _class_record(group, *got)
    if mirror is not None:
        record["horizontal_mirror"] = dict(mirror)
    return out, record


def _free_standing_lift(model, group, free_standing):
    r"""A substrate group with the horizontal mirror adjoined: the free-standing layer's group.

    On a flat layer the free-standing cell's group holds the mirror ``σ_h``,
    which moves no orbital, and every operation of ``group``, matched by
    rotation and by translation modulo the lattice (:func:`_match_operations`).
    The group returned is ``group`` with ``σ_h`` adjoined: the free-standing
    cell's own group where its unitary operations are exactly these, and
    otherwise the group of a cell built to carry exactly those operations
    (:func:`_orbit_cell`).

    Parameters
    ----------
    model : pythtb.TBModel
        The model whose orbitals are assigned to the free-standing cell's
        group (:func:`.._tables._assign_orbitals`).
    group : GroupData
        The substrate group, every operation of which keeps the layer's side.
    free_standing : dict
        The free-standing cell of the same point set
        (:func:`.._realized._layer_realized`): ``"cell"``, ``"cell_typat"``,
        ``"labels"`` (each orbital's species) and ``"symprec"``.  Its group is
        identified the first time it is asked for and kept under ``"group"``.

    Returns
    -------
    tuple or None
        ``(free, of)``: the free-standing group, and for each of its unitary
        operations, by ``.ind``, ``(ind of g in group, whether it is σ_h·g)``.
        ``None`` where the free-standing group is not set in vacuum along
        exactly one direction, where the mirror is not one of its operations
        or moves an orbital, or where an identification or the assignment of
        the orbitals is refused.

    Raises
    ------
    ..refusal.InternalError
        Where an operation of ``group`` is not an operation of the
        free-standing cell, or where no generic point's orbit is identified as
        the operations to carry (:func:`_orbit_cell`).
    """
    parent = free_standing.get("group")
    if parent is None:
        parent = _group_of(free_standing["cell"], typat=free_standing["cell_typat"],
                           spinor=False, include_TR=False, symprec=free_standing["symprec"])
        free_standing["group"] = parent
    if isinstance(parent, Refusal):
        return None
    padded = tuple(int(d) for d in getattr(parent, "_vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return None
    assignment = _assign_orbitals(model, parent, typat=free_standing["labels"])
    if isinstance(assignment, Refusal):
        return None
    reflect = np.eye(3, dtype=int)
    reflect[padded[0], padded[0]] = -1
    psyms = parent._symmetries
    perm = np.asarray(assignment.permutations, dtype=int)
    still = np.arange(perm.shape[1])
    mirror = next((p for p, s in enumerate(psyms) if not s.time_reversal
                   and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int), reflect)
                   and np.array_equal(perm[p], still)), None)
    if mirror is None:
        return None
    pairs, missing, _extra = _match_operations(
        [s for s in group._symmetries if not s.time_reversal],
        [s for s in psyms if not s.time_reversal], free_standing["symprec"],
        group._lattice)
    if missing:
        by_ind = {int(s.ind): s for s in group._symmetries}
        lattice = np.asarray(group._lattice, dtype=float)
        raise InternalError(
            f"{len(missing)} operations of {group.name} #{group.number_str}, read on a "
            "substrate, are not operations of the free-standing cell of the same point "
            f"set ({parent.name} #{parent.number_str}): "
            + ", ".join(_record_name(by_ind[int(i)], lattice) for i in missing)
            + ". Report it, with the model and the call.")
    at = {int(s.ind): p for p, s in enumerate(psyms)}
    prod = parent._get_product_table(get_diff=False)
    of_parent = {}
    for here, there in pairs:
        of_parent[at[int(there)]] = (int(here), False)
        of_parent[int(prod[mirror, at[int(there)]])] = (int(here), True)
    unitary = {p for p, s in enumerate(psyms) if not s.time_reversal}
    if set(of_parent) == unitary:
        return parent, {int(psyms[p].ind): v for p, v in of_parent.items()}
    kept = [psyms[p] for p in sorted(of_parent)]
    species = _DESCENT_SPECIES
    while any(str(t) == species for t in free_standing["cell_typat"]):
        species += "_"

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What the free-standing cell was identified as, with the counts unmatched.

        ``got`` is the group the cell was identified as and ``unitary`` its
        unitary operations; ``unmatched_kept`` and ``unmatched_found`` are
        what :func:`_match_operations` left unmatched.  ``orbit`` is not read.
        """
        return (
            f"the free-standing cell built to carry {group.name} #{group.number_str} with the "
            f"horizontal mirror adjoined ({len(kept)} operations of {parent.name} "
            f"#{parent.number_str}) was identified as {got.name} #{got.number_str}, whose "
            f"{len(unitary)} unitary operations are not that set ({len(unmatched_kept)} "
            f"missing, {len(unmatched_found)} extra).")

    made = _orbit_cell(free_standing["cell"], free_standing["cell_typat"], parent, kept,
                       points=_DESCENT_POINTS, species=species, spinor=False,
                       include_TR=False, symprec=free_standing["symprec"],
                       diagnose=_diagnose)
    if isinstance(made, Refusal):
        return None
    free, _cell, _typat, pairs, _point, _orbit = made
    of_ind = {int(psyms[p].ind): v for p, v in of_parent.items()}
    return free, {int(f): of_ind[int(k)] for k, f in pairs}


def _projective_class_carried(model, group, validated, *, free_standing, refuted=None,
                              square=None, tolerances=EXACT):
    r"""A flat layer's projective class on a substrate, read on the free-standing layer.

    The class is read once on the free-standing group, ``group`` with the
    horizontal mirror ``σ_h`` adjoined (:func:`_free_standing_lift`), each
    operation composed with the mirror given its partner's matrix times
    ``+1``, the even parity the lowest-angular-momentum rule assigns, and a
    tie settled by the tie rule in that group (:func:`_class_kept`).  Every
    operation the class leaves out there is left out here, so the group kept
    on the substrate is the part of the one kept free-standing that preserves
    the layer's normal.  The matrices of ``group`` decide only whether there
    is a class to read, and name the pairs on the record.  Where
    :func:`_free_standing_lift` returns ``None``, the class is read on
    ``group`` itself, with no operation paired with the mirror
    (:func:`_projective_class_refuted`).

    On a group that holds time reversal the class is read under the condition
    :func:`_projective_class_refuted` sets, and each unitary ``g`` whose
    ``g·T`` squares to the other sign (:func:`_reversed_pairs`) is left out of
    every free-standing subgroup with its product with the mirror, which takes
    its reading.

    ``model`` is the model, ``validated`` the detector's verdicts on
    ``group``, ``free_standing`` the free-standing cell
    :func:`_free_standing_lift` reads, and ``refuted``, ``square`` and
    ``tolerances`` as :func:`_projective_class_refuted` takes them.  Returns
    ``(refuted, record)`` as
    :func:`_projective_class_refuted` does, the pairs those of ``group``, the
    record naming the free-standing group the class was read on and how many
    of its operations were kept there (``read_on``).  Returns ``({}, None)``
    where the class is the table's, or is not read here.

    Raises
    ------
    ..refusal.InternalError
        Where ``group`` has a pair of commuting operations whose matrices
        commute only up to a phase and the free-standing group has none, and
        where :func:`_free_standing_lift` raises.
    """

    if bool(group.spinor):
        return {}, None
    reversing = {int(s.ind) for s in group._symmetries if s.time_reversal}
    if reversing and (square is None or not square.obeyed
                      or reversing & (set(validated.broken) | set(refuted or ()))):
        return {}, None
    members = _class_members(group, validated)
    if members is None:
        return {}, None
    D, positions = members
    prod, _tdiff, spf = group._get_product_table(get_diff=True)
    pairs = _commuting_pair_phases(D, positions, group._symmetries, prod, spf,
                                   tol=_CLASS_MARGIN)
    reversed_pairs = (_reversed_pairs(group, validated, positions, {},
                                      tolerances=tolerances) if reversing else ())
    if not pairs and not reversed_pairs:
        # the free-standing pairs are these and their products with the mirror,
        # which commutes with every matrix: no pair here, none there
        return {}, None
    lift = _free_standing_lift(model, group, free_standing)
    if lift is None:
        return _projective_class_refuted(group, validated, refuted=refuted, square=square,
                                         tolerances=tolerances)
    free, of = lift
    # each operation composed with the mirror is given the mirror's sign times
    # its partner's matrix, D(sigma_h g) = D(sigma_h) D(g), with the sign +1,
    # the even parity the lowest-angular-momentum rule assigns
    sign = 1
    at = {int(s.ind): p for p, s in enumerate(group._symmetries)}
    fD = {p: (sign if of[int(s.ind)][1] else 1) * D[at[of[int(s.ind)][0]]]
          for p, s in enumerate(free._symmetries) if not s.time_reversal}
    # g and its product with the mirror both take g's reading with time
    # reversal, since (sigma_h g)^2 = g^2 and the mirror squares to +1 with it;
    # only the first member of each entry is read on the free-standing group
    paired = {int(group._symmetries[g].ind) for g, _t, _c in reversed_pairs}
    freversed = tuple((p, None, complex(-1.0)) for p in sorted(fD)
                      if of[int(free._symmetries[p].ind)][0] in paired)
    got = _class_kept(free, fD, sorted(fD), freversed)
    if got is None:                     # pragma: no cover: the pairs above are its pairs too
        raise InternalError(
            f"the projective class of the states measured in {group.name} "
            f"#{group.number_str} has {len(pairs)} pairs of commuting operations whose "
            "matrices do not commute, and the free-standing group with the horizontal mirror "
            f"adjoined, {free.name} #{free.number_str}, has none. Report it, with the "
            "model and the call.")
    _fpairs, fkept, franked = got

    def _here(subgroup):
        """A subgroup of the free-standing group, by its operations here that preserve the layer's normal.

        ``subgroup`` holds ``.ind`` of the free-standing group.  Returns the
        sorted ``.ind`` in ``group`` of its operations that are not composed
        with the mirror.
        """
        return tuple(sorted(of[int(i)][0] for i in subgroup if not of[int(i)][1]))

    ranked = tuple(_here(c) for c in franked)
    out, record = _class_record(group, tuple(pairs) + tuple(reversed_pairs),
                                frozenset(_here(fkept)), ranked,
                                rule_group=free, rule_ranked=franked)
    record["read_on"] = {"number_str": str(free.number_str), "name": str(free.name),
                         "size": int(free._size), "kept": len(fkept)}
    return out, record


def _match_operations(kept, found, symprec, lattice) -> tuple:
    r"""Each kept operation paired with the identified operation that is the same one.

    Two operations are the same where their rotations are the same integer
    matrix and their translations differ, modulo the lattice, by a Cartesian
    length of at most ``symprec`` on ``lattice`` (rows the cell's vectors, in
    the units ``symprec`` is in).  ``kept`` and ``found`` are sequences of
    operations, and each operation of ``found`` is paired at most once.
    Returns ``(pairs, kept_unmatched, found_unmatched)`` by ``.ind``: the
    pairs ``(kept, found)``; the kept operations paired with none, left
    where the identification moved them; and the identified ones paired with
    none, left where the identification is too large.
    """
    lat = np.asarray(lattice, dtype=float)
    pool = list(found)
    pairs, missing = [], []
    for sym in kept:
        rot = np.rint(np.asarray(sym.rotation, dtype=float)).astype(int)
        tra = np.asarray(sym.translation, dtype=float).reshape(-1)
        for j, other in enumerate(pool):
            if not np.array_equal(
                    rot, np.rint(np.asarray(other.rotation, dtype=float)).astype(int)):
                continue
            d = tra - np.asarray(other.translation, dtype=float).reshape(-1)
            if float(np.linalg.norm((d - np.rint(d)) @ lat)) <= float(symprec):
                pairs.append((int(sym.ind), int(other.ind)))
                pool.pop(j)
                break
        else:
            missing.append(int(sym.ind))
    return (tuple(pairs), tuple(missing), tuple(int(o.ind) for o in pool))


def _shared_origin(found, kept, pairs) -> GroupData:
    r"""The identified group, written in the origin of the operations it was built to carry.

    An identification reads each translation off the cell it is handed, and
    the operations kept are written in the origin the centers satisfy.  The
    two differ by one shift of the origin ``δ``: each kept translation is the
    identified one plus ``(1 − A)·δ``, ``A`` the operation's rotation.  The
    differences ``d_g = t_kept(g) − t_found(g)``, modulo the lattice, are
    averaged over the pairs to ``δ``, and ``found`` is moved by it
    (:func:`_moved_origin`).

    ``found`` is the identified group, ``kept`` the operations it was built to
    carry, each with ``.ind`` and ``.translation``, and ``pairs`` the
    ``(kept, identified)`` pairs of ``.ind`` of :func:`_match_operations`.
    Returns ``found`` itself where every pair's translations agree to
    :data:`.._tolerance._EXACT_CLOSURE`, as on every cell built from exact
    numbers; otherwise a new record, ``found`` left as it was.

    Raises
    ------
    ..refusal.InternalError
        Where no single shift carries the identified translations onto the
        kept ones to within :data:`.._tolerance._EXACT_CLOSURE`.
    """
    by_kept = {int(s.ind): s for s in kept}
    by_found = {int(s.ind): s for s in found._symmetries}
    diffs = []
    for k, f in pairs:
        d = (np.asarray(by_kept[int(k)].translation, dtype=float).reshape(-1)
             - np.asarray(by_found[int(f)].translation, dtype=float).reshape(-1))
        diffs.append((np.asarray(by_found[int(f)].rotation, dtype=float), d - np.rint(d)))
    if not diffs or max(float(np.abs(d).max()) for _r, d in diffs) <= _EXACT_CLOSURE:
        return found
    delta = np.mean([d for _r, d in diffs], axis=0)
    miss = max(float(np.abs((np.eye(3) - r) @ delta - d).max()) for r, d in diffs)
    if miss > _EXACT_CLOSURE:
        raise InternalError(
            f"group #{found.number_str}: the operations identified on the cell and the "
            f"{len(diffs)} it was built to carry are not one group about two origins: "
            f"the best single shift of the origin misses by {miss:.3g} in reduced "
            "coordinates. Report it, with the model and the call.")
    return _moved_origin(found, delta)


def _moved_origin(group, delta) -> GroupData:
    r"""``group`` written about an origin moved by ``delta``.

    ``delta`` is the shift ``δ``, in reduced coordinates of the group's 3D
    cell.  Every translation moves by ``(1 − A)·δ``, ``A`` the operation's
    rotation, and the origin of the tables' setting (``shiftUC``) by ``δ``,
    which leaves every operation's tabulated translation where it was.
    Returns a new record; ``group`` is left as it was.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    moved = []
    for s in group._symmetries:
        new = copy.copy(s)
        for cached in ("translation_cart", "is_identity"):
            new.__dict__.pop(cached, None)
        new.translation = (np.asarray(s.translation, dtype=float)
                           + (np.eye(3) - np.asarray(s.rotation, dtype=float)) @ delta)
        moved.append(new)
    shift = np.asarray(group._shiftUC, dtype=float) + delta
    sg = copy.copy(group._sg)
    sg.__dict__.pop("translations_cart", None)
    sg.symmetries, sg.shiftUC = moved, shift
    return replace(group, _symmetries=list(moved), _shiftUC=shift, _sg=sg)


def _orbit_cell(base_cell, base_typat, group, kept, *, points, species, spinor,
                include_TR, symprec, diagnose):
    r"""The cell that carries exactly ``kept``, and its identified group.

    The cell is ``base_cell``'s own points plus the orbit of one generic point
    under ``kept``, as one extra species (:func:`_generic_orbit`).  It is
    identified (:func:`_group_of`) and checked against ``kept`` operation by
    operation (:func:`_match_operations`).  A cell read as non-primitive, or
    with operations other than ``kept``, is rebuilt from the next point of
    ``points``.  The group returned carries the operations ``kept`` in the
    origin they were measured in (:func:`_shared_origin`).

    Parameters
    ----------
    base_cell : pythtb.TBModel
        The point set ``group`` was identified on, as a model.
    base_typat : sequence
        The species of each point of ``base_cell``.
    group : GroupData
        The group identified on ``base_cell``.
    kept : sequence
        The operations of ``group`` the cell is to carry.
    points : sequence of tuple of float
        The generic points, tried in turn (:data:`_DESCENT_POINTS`,
        :data:`_DECLARED_POINTS`).
    species : str
        The species the orbit is added as, one ``base_typat`` does not hold.
    spinor, include_TR : bool
        Whether the cell is identified in double-valued tables, and with time
        reversal adjoined (:func:`_group_of`).
    symprec : float
        The symmetry finder's distance, in the model's units of length, also
        the one the operations are matched to.
    diagnose : callable
        Writes the reason the error is raised with, from ``(group, unitary
        operations, kept unmatched, found unmatched, orbit)`` of the last
        point.

    Returns
    -------
    tuple or Refusal
        ``(group, cell, cell_typat, pairs, point, orbit)``: the identified
        group, the cell and its species, the pairs of
        :func:`_match_operations`, the point used, and its orbit in reduced
        coordinates of the 3D cell of ``group``.  The
        :class:`..refusal.Refusal` an identification made, where one is
        refused.

    Raises
    ------
    ..refusal.InternalError
        Where no point of ``points`` gives a cell read as exactly ``kept``.
    """
    last = None
    for seed in points:
        point, orbit_padded, orbit_cell = _generic_orbit(base_cell, group, kept, seed=seed)
        cell, cell_typat = _decorated_cell(
            base_cell, base_typat, environment="free-standing",
            decorations=[([float(x) for x in row], species) for row in orbit_cell])
        got = _group_of(cell, typat=cell_typat, spinor=bool(spinor),
                        include_TR=bool(include_TR), symprec=symprec)
        if isinstance(got, Refusal):
            return got
        unitary = [s for s in got._symmetries if not s.time_reversal]
        pairs, unmatched_kept, unmatched_found = _match_operations(kept, unitary, symprec,
                                                                   group._lattice)
        if int(got._lattice_index) <= 1 and not unmatched_kept and not unmatched_found:
            return (_shared_origin(got, kept, pairs), cell, cell_typat, pairs, point,
                    orbit_padded)
        last = (got, unitary, unmatched_kept, unmatched_found, orbit_padded)
    raise InternalError(
        diagnose(*last) + f" It comes from the last of {len(points)} generic points "
        "tried, and none of them gave a cell whose group is exactly the kept operations. "
        "Report it, with the model and the call.")


def _centers_not_exactly_symmetric(model, group, assignment, *,
                                   tol) -> Refusal | None:
    r"""The refusal of orbital centers whose defect alone breaks operations.

    In convention I the Bloch phase carries ``e^{2πi k·(R + τ_j − τ_i)}``, so
    centers that miss the group's action by ``δ``, in reduced coordinates
    (:func:`.._centers._position_defect`), cost ``H(k)`` a covariance residual
    of order ``2π·δ·max|H|`` whatever the hoppings do, ``max|H|`` as
    :func:`.._measure._max_hamiltonian` measures it.  It is read once the
    cell's group is identified, where no tolerance for the centers is
    declared (:func:`.._realized._realized_core`).

    ``model`` is the model, ``group`` the cell's group, ``assignment`` the
    orbitals' orbits in it, and ``tol`` the covariance tolerance, in the units
    of ``H``.  Returns ``positions_not_exactly_symmetric`` where the estimate
    is above ``tol``, naming ``group`` as an upper bound and carrying it in its
    context; ``None`` otherwise.
    """

    delta = float(_position_defect(model, group, assignment,
                                   {int(s.ind): s for s in group._symmetries}))
    if not delta > 0.0:
        return None
    max_h = _max_hamiltonian(model)
    estimate = 2.0 * np.pi * delta * max_h
    if estimate <= float(tol):
        return None
    return Refusal(
        "positions_not_exactly_symmetric",
        f"the orbital centers miss the cell's own operations by delta = {delta:.3g}, in "
        "reduced coordinates, which alone gives H(k) a covariance residual of order "
        f"{estimate:.3g}, above tol = {float(tol):.3g}, in the units of H. The cell's "
        f"group is {group.name} #{group.number_str}, an upper bound on what this "
        "Hamiltonian obeys. Declare the model's precision with `precision=`, or a "
        "tolerance for the centers alone with `position_tol=`.",
        value=delta, threshold=float(tol) / (2.0 * np.pi * max_h),
        context={"group": group, "number_str": str(group.number_str),
                 "position_defect": delta, "max_h": max_h, "tol": float(tol)})


def _reversal_position(group):
    """The position of pure time reversal among the operations of ``group``.

    A position is an index into ``group._symmetries``.  Returns ``None`` where
    ``group`` holds no pure time reversal.
    """
    return next((p for p, s in enumerate(group._symmetries) if s.time_reversal
                 and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                    np.eye(3, dtype=int))
                 and np.abs(((np.asarray(s.translation, dtype=float) + 0.5) % 1.0)
                            - 0.5).max() <= 1e-6), None)


def _time_reversal_square(group, validated, *, tolerances=EXACT):
    r"""The pure time reversal's square, against the sign the tables read here ask for.

    Time reversal ``A = X K`` squares to ``X X̄``, whose sign, where it is a
    multiple of the identity, no rescaling of ``X`` changes.  The tables ask
    for ``+1`` where they are single-valued and ``-1`` where they are
    double-valued.  A member of that sign is looked for as
    :func:`_antiunitary_square` does, and it is the matrix every later step
    reads for time reversal (:func:`_time_reversal_member`).

    ``group`` is the round's group, ``validated`` the detector's verdicts on
    it, and ``tolerances`` the call's declared tolerances
    (:class:`.._tolerance.Tolerances`).  Returns the :class:`_Square`
    reading, or ``None`` where ``group`` has no pure time reversal, or the
    detector refuted it or kept no member for it.
    """
    position = _reversal_position(group)
    if position is None:
        return None
    # Pure time reversal squares to the identity, and the sign the tables ask for
    # is their valuedness: -1 in double-valued tables, +1 in single-valued ones.
    # Nothing else would be read off the multiplication table, so it is not built.
    return _antiunitary_square(group, validated, position,
                               asked=-1 if group.spinor else 1, tolerances=tolerances)


def _antiunitary_square(group, validated, position, *, asked=None, tolerances=EXACT):
    r"""One antiunitary operation's square, against the sign the group's table gives it.

    For an operation ``g·T`` whose unitary part squares to the identity modulo
    the lattice, the square ``X X̄`` of its member ``X`` is a member of the
    identity's, and where it is a multiple of the identity its sign is the
    Hamiltonian's own.  A member of the sign asked, ``s``, is looked for in
    turn: the detector's own; the unitary nearest ``(Y + s·Yᵀ)/2``, for ``Y``
    the detector's own or one of three fixed combinations of the solution
    space, kept where it squares to ``s`` and stays in the space; and the
    detector's own again, where its square reads ``s`` within
    :data:`.._tolerance._TIME_REVERSAL_SQUARE_MARGIN`.  The space is the
    members of the operation's own form, or the detector's whole space where
    none has that form (:func:`.._measure._own_form_read`).

    Parameters
    ----------
    group : GroupData
        The round's group.
    validated : ValidatedGroup
        The detector's verdicts on ``group``, with each operation's members.
    position : int
        The operation's position in ``group``.
    asked : int, default None
        The sign the table gives the square, where it is known without the
        table.  ``None`` reads it off the table.
    tolerances : Tolerances, default EXACT
        The call's declared tolerances: how far from ``s`` a member of a
        model symmetric only within its precision may square, and how far
        outside the space it may lie.

    Returns
    -------
    _Square or None
        The reading, with ``obeyed`` false and ``member`` ``None`` where no
        member of the sign asked is found.  ``None`` where the detector
        refuted the operation or kept no member for it, or where its unitary
        part does not square to the identity, as for a threefold or a fourfold
        with time reversal.
    """

    ind = int(group._symmetries[position].ind)
    member = validated._intertwiners.get(ind)
    if ind in set(validated.broken) or member is None:
        return None
    if asked is None:
        prod, _tdiff, spf = group._get_product_table(get_diff=True)
        if int(prod[position, position]) != _identity_position(group):
            return None                          # no square of its own to read
        asked = int(spf[position, position])
    asked = int(asked)
    zero, snap, in_span = (tolerances.noise(ZERO), tolerances.noise(SNAP),
                           tolerances.noise(IN_SPAN))
    X = np.asarray(member, dtype=complex)
    eye = np.eye(X.shape[0])
    square = X @ X.conj()
    scalar = complex(np.trace(square)) / X.shape[0]
    # the sign, which is what no rescaling of X changes: a member the detector
    # left unnormalized squares to a positive multiple of it
    measured = (float(np.sign(scalar.real))
                if (float(np.abs(square - scalar * eye).max()) <= _TIME_REVERSAL_SQUARE_MARGIN
                    and scalar.real != 0.0)
                else float("nan"))
    out = _Square(ind=ind, asked=asked, measured=measured, obeyed=True, member=X)
    if float(np.abs(square - asked * eye).max()) <= snap:
        return out                               # the detector's own, of the sign asked
    basis = (getattr(validated, "_bases", None) or {}).get(ind)
    if basis is None or not len(np.asarray(basis)):
        # an operation its own form refuted has no matrix of its own: whether
        # any matrix with its action on k squares to the sign asked is read
        # off the detector's whole space
        basis = ((getattr(validated, "_own_form", None) or {}).get(ind) or {}).get("space")
    basis = (None if basis is None or len(np.asarray(basis)) < 2
             else np.asarray(basis, dtype=complex))
    if basis is not None:
        flat = basis.reshape(len(basis), -1)
        q, r = np.linalg.qr(flat.T)                  # the space, orthonormal
        span = q[:, np.abs(np.diag(r)) > zero * max(1.0, float(np.abs(r).max()))]
        rng = np.random.default_rng(17)
        seeds = [X] + [np.einsum("j,jab->ab",
                                 rng.normal(size=len(basis))
                                 + 1j * rng.normal(size=len(basis)), basis)
                       for _ in range(3)]
        for Y in seeds:
            U = _unitarized(0.5 * (Y + asked * Y.T), tol=zero)
            if U is None:
                continue
            inside = U.reshape(-1) - span @ (span.conj().T @ U.reshape(-1))
            if (float(np.abs(inside).max()) <= in_span
                    and float(np.abs(U @ U.conj() - asked * eye).max())
                    <= _TIME_REVERSAL_SQUARE_MARGIN):
                return replace(out, member=U,
                               measured=float(np.sign((complex(np.trace(U @ U.conj()))
                                                       / U.shape[0]).real)))
    if float(np.abs(square - asked * eye).max()) <= _TIME_REVERSAL_SQUARE_MARGIN:
        return out                               # the sign asked, read off a noisy member
    return replace(out, obeyed=False, member=None)


def _time_reversal_member(validated, square):
    """The detector's record with time reversal's member of the sign asked in place.

    ``validated`` is the detector's record and ``square`` the
    :class:`_Square` of :func:`_time_reversal_square`.  Returns a new record
    whose member for time reversal is ``square.member``; ``validated`` itself
    where ``square`` is ``None``, found no member of the sign asked, or kept
    the detector's own.
    """
    if (square is None or not square.obeyed
            or square.member is validated._intertwiners.get(square.ind)):
        return validated
    return replace(validated, _intertwiners={**validated._intertwiners,
                                            int(square.ind): square.member})


def _ansatz_refuted(model, group, assignment, validated, *, layout, spin,
                    blocks=None, max_candidates=2 ** 20, skip=()):
    r"""The operations whose matrices do not have their tensor structure on the multiplets.

    An operation's matrix is ``D(g) = O(g) ⊗ S(g)``, ``O`` carrying each
    multiplet where the operation carries its site and ``S(g)`` its own lift:
    the tables' spin lift on a one-dimensional multiplet, the model's own
    block on a wider one.  The test asks whether a member of the operation's
    own form (:func:`.._measure._own_form_read`) has that structure, on the
    ``g`` branch alone (:func:`.._multiplets._ansatz_defects`).  An operation
    without one, its scaled defect above the gate, is refuted by its tensor
    structure.  It is asked of the operations the detector left undetermined,
    which share their action on ``k`` with another; and, where the multiplets
    are stated by ``site_multiplets=``, of those it found obeyed as well.
    Every obeyed operation carries a measured multiplet, and a stated one only
    where it does not mix it with another.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The round's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    layout : SiteMultiplets or None
        The multiplets; ``layout.determined_by`` is ``"declaration"`` where
        they are stated.  ``None`` has :func:`.._multiplets._ansatz_defects`
        measure them.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    blocks : dict, default None
        The multiplet blocks built once for the round, ``{ind: {channel:
        S}}`` (:func:`.._multiplets._multiplet_blocks`).  ``None`` has
        :func:`.._multiplets._ansatz_defects` build them.
    max_candidates : int, default 2 ** 20
        The most candidates a multiplet's block enumeration may try.
    skip : set of int, default ()
        The ``.ind`` of the operations their own form refuted, which are not
        tested again.

    Returns
    -------
    tuple or Refusal
        ``(tested, refuted)``: the record of the test for each operation
        tested, by ``.ind``, as :func:`.._multiplets._ansatz_defects` returns
        it, and those of them not realizable.  ``({}, {})`` where nothing is
        tested.  The :class:`..refusal.Refusal` the test made: a declared
        spinor pair the operations do not carry onto one pair, or a block
        enumeration past ``max_candidates`` (``rep_search_too_large``) where
        ``blocks`` is ``None``.
    """

    shared = set(validated.undetermined)
    if layout is not None and layout.determined_by == "declaration":
        # stated multiplets are asked of every operation the detector kept:
        # measured ones are carried by each of them by construction, stated
        # ones only where the operation does not mix them
        shared |= set(validated.obeyed)
    shared -= set(skip or ())
    if not shared:
        return {}, {}
    positions = [i for i, s in enumerate(group._symmetries) if int(s.ind) in shared]
    got = _ansatz_defects(model, group, assignment, validated, positions,
                          layout=layout, spin=spin, blocks=blocks,
                          max_candidates=max_candidates, orients=(1,))
    if isinstance(got, Refusal):
        return got
    return got, {i: r for i, r in got.items() if not r["realizable"]}


def _translation_permutation(model, assignment, position):
    r"""The matrix carrying every orbital onto its translate's under one pure translation.

    ``P[σ(i), i] = 1``, ``σ`` carrying each state onto its translate's under
    the operation at ``position`` of the group: each site's orbitals onto its
    image's (``assignment.permutations``), in the order the sites list them
    (``assignment.site_orbitals``), spin by spin.  ``model`` gives the number
    of states.  Returns ``P``, of shape ``(nstate, nstate)``, or ``None``
    where a site and its image hold different numbers of orbitals.
    """
    perm = np.asarray(assignment.permutations, dtype=int)
    site_orbs = tuple(assignment.site_orbitals)
    nspin, nstate = int(model.nspin), int(model.nstate)
    P = np.zeros((nstate, nstate), dtype=complex)
    for i, orbs in enumerate(site_orbs):
        dst = site_orbs[int(perm[int(position), i])]
        if len(dst) != len(orbs):                   # pragma: no cover: defensive
            return None
        for slot, a in enumerate(orbs):
            for s in range(nspin):
                P[int(dst[slot]) * nspin + s, int(a) * nspin + s] = 1.0
    return P


def _translation_held(model, group, assignment, validated, position, *, spin=None,
                      tol=None, commuting=()):
    r"""Whether ``H(k)`` obeys one pure translation, with a change of basis within each site.

    The translation's solution space, the matrices commuting with ``H(k)`` at
    every ``k``, is restricted to the blocks carrying each site's states onto
    its translate's (:func:`.._matrices._block_members`).  The translation
    holds where that restriction has a member nonsingular on every block
    (:func:`.._matrices._filled_check`).  The member may act as any unitary on
    each site's orbitals, the permutation of the sites among them, and leaves
    a spin, native or declared, untouched.  The matrix it holds with is read
    among the members commuting with ``commuting``: the member nearest the
    permutation of the sites (:func:`_translation_permutation`), or the first
    of the fixed generic members (:func:`.._matrices._generic_members`) where
    that one is singular, made unitary block by block.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The cell's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    validated : ValidatedGroup
        The detector's verdicts on ``group``, whose solution spaces are read.
    position : int
        The translation's position in ``group``.
    spin : SpinBasis, default None
        The declared spin partners; ``None`` where none are declared.
    tol : float, default None
        The declared tolerance, relative to ``max|H|``, every reading takes;
        ``None`` where nothing was declared.
    commuting : sequence of ndarray, default ()
        The matrices already read for the other translations.

    Returns
    -------
    tuple
        ``(held, matrix, turn)``: whether the translation holds so; the
        unitary it holds with, ``None`` where no member commuting with
        ``commuting`` is nonsingular; and how far that unitary turns each
        site's orbitals onto its translate's, half the smallest arc of the
        unit circle holding every eigenvalue of its blocks, so that a phase
        common to all of them is no turn, ``0.0`` within the declared
        tolerance and ``nan`` where there is no unitary.  ``(False, None,
        nan)`` where the translation does not hold, the detector kept no space
        for the translation, or its space is past the memory ceiling
        :data:`.._detector._READOFF_MAX_BYTES`.
    """

    nothing = (False, None, float("nan"))
    basis = (getattr(validated, "_bases", None) or {}).get(
        int(group._symmetries[int(position)].ind))
    nstate = int(model.nstate)
    if basis is None:
        return nothing
    basis = np.asarray(basis)
    if (basis.ndim != 3 or not basis.shape[0]
            or basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES):
        return nothing
    # every site one block, carried onto its translate's whole
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    sites = _multiplets_of(model, assignment, spin, tuple((orbs,) for orbs in site_orbs),
                           tol=tol)
    image = _channel_permutation(assignment, int(position), model, sites)
    blocks = _ansatz_blocks(sites._chan_states, image, 1)
    if any(len(rows) != len(cols) for rows, cols in blocks):   # pragma: no cover: defensive
        return nothing
    pieces, lifts = blocks, None
    if int(model.nspin) == 2 or spin is not None:
        # a translation turns no spin: every operation is read with its own
        # spin rotation, the identity here, so the change of basis acts on the
        # orbitals alone, the same on both states of every spinor
        spinors = [[(2 * o, 2 * o + 1) for o in orbs] for orbs in site_orbs]
        if spin is not None:
            spinors = [[tuple(int(x) for x in a) for a in site]
                       for site in _atoms_of_site(site_orbs, spin)]
        pieces = [(np.asarray(to, dtype=int), np.asarray(of, dtype=int))
                  for s in range(len(site_orbs)) for to in spinors[int(image[s])]
                  for of in spinors[s]]
        lifts = [np.eye(2, dtype=complex)] * len(pieces)
    V, _reach = _block_members(basis, pieces, tol=noise_floor(_IN_BLOCKS, tol), lifts=lifts)
    held, _unfilled = _filled_check(V, blocks, tol=tol)
    if not held:
        return nothing
    if commuting:
        # the members commuting with every matrix already read for another
        # translation, to the precision a relation among measured matrices has
        A = np.stack([np.concatenate([(X @ D - D @ X).reshape(-1) for D in commuting])
                      for X in V], axis=1)
        _u, s, vh = np.linalg.svd(A, full_matrices=False)
        gate = noise_floor(ASSEMBLED, tol) * max(float(s[0]), 1.0)
        keep = [i for i in range(len(s)) if float(s[i]) <= gate]
        V = (np.stack([np.einsum("j,jab->ab", vh[i].conj(), V) for i in keep])
             if keep else V[:0])
    if not len(V):
        return True, None, float("nan")
    P = _translation_permutation(model, assignment, position)
    nearest = np.linalg.lstsq(V.reshape(len(V), -1).T, P.reshape(-1), rcond=None)[0]
    singular = noise_floor(SINGULAR, tol)
    member = None
    for X in (np.einsum("j,jab->ab", nearest, V), *_generic_members(V)):
        scale = max(float(np.abs(X).max()), 1.0)
        if all(float(np.linalg.svd(X[np.ix_(rows, cols)], compute_uv=False)[-1])
               > singular * scale for rows, cols in blocks):
            member = X
            break
    if member is None:
        return True, None, float("nan")
    D = np.zeros((nstate, nstate), dtype=complex)
    phases = []
    for rows, cols in blocks:
        u, _s, wh = np.linalg.svd(member[np.ix_(rows, cols)])
        D[np.ix_(rows, cols)] = u @ wh
        phases.append(np.angle(np.linalg.eigvals(u @ wh)))
    around = np.sort(np.mod(np.concatenate(phases), 2.0 * np.pi))
    gap = float(np.max(np.diff(np.concatenate([around, around[:1] + 2.0 * np.pi]))))
    turn = (2.0 * np.pi - gap) / 2.0
    return True, D, (0.0 if turn <= noise_floor(SNAP, tol) else turn)


def _translation_residual(model, assignment, position, k_pts, grid=None):
    r"""How far ``H(k)`` is from obeying one pure translation as written.

    For a model that writes every copy of a site in one basis, in pythtb's
    convention I

    .. math:: T_\tau\,|\varphi_i(k)\rangle = e^{-2\pi i\,k\cdot\tau}\,
              |\varphi_{\sigma(i)}(k)\rangle,

    so ``D(τ)`` is the orbital permutation ``σ`` times a global phase, and
    covariance is ``P H(k) P† = H(k)``, ``P`` the permutation matrix
    (:func:`_translation_permutation`).  A model that writes the copies in
    bases turned from site to site obeys the translation with that change of
    basis beside the permutation, which :func:`_translation_held` reads.

    ``model`` and ``assignment`` give ``P`` for the translation at
    ``position`` of the group.  ``k_pts`` is the grid, and ``grid`` the
    detector's own ``H`` on it (:class:`.._detector._GridHamiltonians`), or
    ``None`` to evaluate ``H`` here.  Returns the largest entry of
    ``|P H(k) P† − H(k)|`` over the grid, in the units of ``H``, or ``None``
    where a site and its image hold different numbers of orbitals.
    """
    P = _translation_permutation(model, assignment, position)
    if P is None:                                   # pragma: no cover: defensive
        return None
    # `grid` is the detector's own evaluation of H on this same mesh, handed in
    # by the measurement context where the two k-sets are byte for byte the
    # same one (checked there, not assumed here).  Both evaluate the model's
    # one Bloch sum on the same points, so the measurement is the same either
    # way.
    H = (np.asarray(grid.full()) if grid is not None
         else np.asarray(_bloch.hamiltonian(model, np.atleast_2d(k_pts))))
    return float(np.abs(np.einsum("ab,kbc,dc->kad", P, H, P.conj()) - H).max())


def _lattice_verdict(model, group, assignment, validated, refuted, tol, k_pts,
                     context=None, *, spin=None, matrix_tol=None) -> dict:
    r"""What the Hamiltonian does with the cell's own pure translations.

    A cell whose point set repeats under ``_lattice_index − 1`` fractional
    translations ``{1|τ}`` has a lattice supergroup.  Each translation is read
    at the sites.  It holds where its permutation of the sites holds as
    written, its residual (:func:`_translation_residual`) at most ``tol``, or
    else where a change of basis within each site carries the sites' orbitals
    onto their translates' (:func:`_translation_held`), read among the
    members commuting with the matrices the translations before it hold
    with.  Where a translation holds and no matrix it holds with commutes
    with those, only the translations obeyed as written are counted.  A
    translation the detector, the declaration or the centers refuted stays
    refuted; one refuted by its tensor structure alone is read again here.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The cell's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    refuted : dict
        The operations refuted past the detector, by ``.ind``, each with its
        record.
    tol : float
        The covariance tolerance, in the units of ``H``.
    k_pts : ndarray
        The grid the residuals are measured on.
    context : _MeasurementContext, default None
        The measurement context, whose grid of ``H`` is reused where it is
        this model's on ``k_pts``.
    spin : SpinBasis, default None
        The declared spin partners.
    matrix_tol : float, default None
        The declared tolerance, relative to ``max|H|``, that
        :func:`_translation_held` reads with; ``None`` where nothing was
        declared.

    Returns
    -------
    dict
        Empty where the cell has no fractional translation.  Otherwise
        ``index``; ``fractional_translations``; ``operations``, one entry per
        translation with its ``ind``, ``translation`` (reduced ``τ``),
        ``cartesian``, ``refuted_by``, the ``scaled`` tensor-structure
        defect, the ``residual`` of the permutation as written, and ``tol``;
        ``realized``, the translations the Hamiltonian obeys; ``dressing``,
        for each of them the ``matrix`` it is obeyed with and the ``turn``
        that matrix makes; ``refuted``, the translations refuted here and by
        nothing before, each with its record; and
        ``coset_partners_refuted``, how many refuted unitary operations are
        not pure translations.  ``refuted_by`` is ``None`` for a translation
        obeyed, the detector's reason where the detector refuted it, the word
        of the verdict that refuted it otherwise (``"tensor structure"``,
        ``"declaration"``, ``"centers"``), and ``"translation"`` where none
        did and the Hamiltonian does not obey it.
    """
    index = int(getattr(group, "_lattice_index", 1) or 1)
    if index <= 1:
        return {}
    broken = set(validated.broken)
    reason_of = {int(d.index): str(d.refuted_by) for d in validated.discrepancies}
    refuted = dict(refuted or {})
    lattice = np.asarray(group._lattice, dtype=float)
    ident = np.eye(3, dtype=int)
    antiunitary = {int(s.ind) for s in group._symmetries if s.time_reversal}
    gone = (broken | set(refuted)) - antiunitary

    found, pure_inds = [], set()
    for position, s in enumerate(group._symmetries):
        if bool(s.time_reversal):
            continue
        if not np.array_equal(np.rint(np.asarray(s.rotation, dtype=float)).astype(int),
                              ident):
            continue
        t = np.asarray(s.translation, dtype=float).reshape(-1)
        t = t - np.rint(t)
        # zero as the identification read it: no longer than the finder's distance
        if float(np.linalg.norm(t @ lattice)) <= float(group._symprec):
            continue
        ind = int(s.ind)
        pure_inds.add(ind)
        # which judge refuted it: the detector's own measurement (its reason),
        # or a verdict read on the detector's spaces, in its own word
        why = (reason_of[ind] if ind in broken
               else str(refuted[ind].get("refuted_by", "tensor structure"))
               if ind in refuted else None)
        scaled = (float(refuted[ind]["scaled"]) if ind in refuted and "scaled" in refuted[ind]
                  else float("nan"))
        residual = _translation_residual(
            model, assignment, position, k_pts,
            grid=None if context is None else context.grid_for(model, k_pts))
        found.append((position, ind, tuple(float(x) for x in t),
                      tuple(float(x) for x in t @ lattice), why, scaled, residual))

    # The translations the detector, the declaration and the centers left
    # standing, for the hoppings to decide.  Those obeyed as written carry the
    # permutation itself; every other one is read at the sites, among the
    # members commuting with the matrices read before it
    standing = [f for f in found if f[1] not in broken and f[4] in (None, "tensor structure")]
    bare = [f for f in standing if f[6] is not None and f[6] <= float(tol)]
    commuting = [_translation_permutation(model, assignment, f[0]) for f in bare]
    dressing = [{"translation": f[2], "turn": 0.0, "matrix": M}
                for f, M in zip(bare, commuting)]
    commute = True
    for position, _ind, tau, _cart, _why, _scaled, residual in standing:
        if residual is not None and residual <= float(tol):
            continue
        held, matrix, turn = _translation_held(model, group, assignment, validated,
                                               position, spin=spin, tol=matrix_tol,
                                               commuting=tuple(commuting))
        if held and matrix is None:
            commute = False
        elif held:
            commuting.append(matrix)
            dressing.append({"translation": tau, "turn": turn, "matrix": matrix})
    if not commute:
        # translations obeyed only with changes of basis that do not commute:
        # a magnetic translation group, which is no crystal's, so only the
        # translations obeyed as written are counted
        dressing = dressing[:len(bare)]
    realized = tuple(d["translation"] for d in dressing)

    entries = tuple(
        {"ind": ind, "translation": tau, "cartesian": cart,
         "refuted_by": (None if tau in realized
                        else "translation" if why is None else why),
         "scaled": scaled, "residual": residual, "tol": float(tol)}
        for _p, ind, tau, cart, why, scaled, residual in found)
    # what the direct measurement refuted and nothing before it: a pure
    # translation is never left to inference, so it joins the refuted set with
    # its own reason and the descent removes it like any other broken operation
    add = {int(e["ind"]): {
        "scaled": float(e["residual"]), "defect": float(e["residual"]),
        "scale": float(e["tol"]), "orientation": 0, "dim": 0,
        "refuted_by": "translation", "residual": float(e["residual"])}
        for e in entries
        if e["refuted_by"] == "translation" and e["residual"] is not None}
    removed = gone | set(add)
    return {"index": index,
            "fractional_translations": tuple(group._fractional_translations),
            "operations": entries,
            "realized": realized,
            "dressing": tuple(dressing),
            "refuted": add,
            "coset_partners_refuted": len(removed - pure_inds)}


def _supercell_refusal(group, validated, lattice_record) -> Refusal:
    r"""The ``supercell`` refusal of a Hamiltonian that obeys the cell's own pure translation.

    ``lattice_record`` is :func:`_lattice_verdict`'s record, with at least one
    translation in ``realized``.  The reason names the first of them, the
    index and the primitive lattice the realized translations generate
    (:func:`.._spacegroup._primitive_lattice`), and how far the matrices in
    ``dressing`` turn each site's orbitals, where they turn them at all.
    ``validated`` is the detector's verdicts, carried in the context with the
    translations, the index, the primitive lattice, the dressing and the
    record.  ``group`` is the cell's group, whose lattice gives the Cartesian
    vectors; it is named in the reason and carried in ``context["group"]`` by
    :func:`.._realized._named_refusal`.
    """
    # the finer lattice is the one the Hamiltonian was measured to have, the
    # one the realized translations generate, and not the one every
    # translation the point set allows would generate
    taus = tuple(lattice_record["realized"])
    lattice = np.asarray(group._lattice, dtype=float)
    prim = _primitive_lattice(lattice, taus)
    tau = np.round(np.asarray(taus[0], dtype=float), 6)
    cart = np.round(np.asarray(taus[0], dtype=float) @ lattice, 6)
    turn = max(float(d["turn"]) for d in lattice_record["dressing"])
    turned = ("" if turn == 0.0 else
              ", once each site's orbitals are carried onto its translate's by a change "
              f"of basis within the site, which turns them by up to {turn:.3g} rad")
    rebuild = ("Rebuild the model on its primitive cell"
               if turn == 0.0 else
               "Rebuild the model on its primitive cell, with every copy's orbitals "
               "written in one basis: `context[\"dressing\"]` holds the matrices that "
               "carry each site's orbitals onto its translate's")
    return Refusal(
        "supercell",
        f"the Hamiltonian obeys the pure translation {tau.tolist()} (reduced) = "
        f"{cart.tolist()} (Cartesian){turned}. The model is written on a supercell: its "
        f"lattice is a sublattice of index {int(prim['index'])} of the crystal's "
        "lattice, whose primitive vectors are "
        f"{np.round(np.asarray(prim['cartesian'], dtype=float), 6).tolist()} "
        f"(Cartesian). {rebuild}. Where the translation is meant to be broken, break it "
        "in the Hamiltonian, or declare the copies different atoms with `typat=`.",
        value=float(lattice_record["index"]), threshold=1.0,
        context={"translations": taus, "index": int(lattice_record["index"]),
                 "primitive_lattice": prim,
                 "dressing": tuple(lattice_record["dressing"]),
                 "validated": validated, "lattice": lattice_record})


def _declared_cell_reduction(model, group, assignment, *, spin, characters):
    r"""The operations of the cell that the declaration leaves, read before any hopping.

    ``characters`` is read in the site groups of ``group``
    (:func:`.._declaration._declared_site_contents`).  An operation is excluded
    where it carries a site onto a site declared with different site irreps,
    or carries the span of a declared function, such as ``px``, out of itself
    (:func:`.._declaration._content_compatible`).  The operations kept are the
    largest subgroup of ``group`` inside those left, the first by the tie rule
    where several of that order exist
    (:func:`.._declaration._largest_compatible_group`).

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The cell's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    characters : sequence or mapping
        The declaration, as ``characters=`` gives it.

    Returns
    -------
    tuple or Refusal
        ``(kept, excluded, read_in)``: the ``.ind`` kept, a frozenset over
        every operation; ``{ind: record}`` for each operation excluded, with
        its reason, sites and orbitals; and, by the declaration's orbitals,
        the site group and group each word was read in.  The
        :class:`..refusal.Refusal` the reading made where a word names nothing
        in these site groups, or ``site_group`` where a site group has no
        table.  The refusal is about this cell's site groups, not the crystal,
        and the loop asks again on the next group.
    """

    try:
        contents, read_in = _declared_site_contents(
            model, group, assignment, characters, spin=spin)
    except _sitechars.DeclarationError as exc:
        return Refusal(exc.gate, exc.message, context=dict(exc.context))
    except Exception as exc:                     # noqa: BLE001 - a site group with no table
        return Refusal(getattr(exc, "gate", "site_group"), str(exc), context={})
    kept, excluded = _content_compatible(group, assignment, contents)
    return kept, excluded, read_in


def _layer_declared_reading(model, labels, *, extras, spinor, spin, symprec, characters):
    r"""The declaration read in the free-standing layer's site groups.

    The layer's cell is built free-standing, with nothing under it, and its
    group identified; the declaration is read there by
    :func:`_declared_cell_reduction`.  On a substrate the operations this
    reading excludes are excluded too (:func:`_carried_reduction`), and words
    the substrate's site groups name nothing for are read here and restricted
    (:func:`_restricted_contents`).

    Parameters
    ----------
    model : pythtb.TBModel
        The layer, presented as a slab.
    labels : sequence
        The species of each orbital.
    extras : sequence of (position, label)
        The declared decorations.
    spinor : bool
        Whether the cell is identified in double-valued tables.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    symprec : float
        The symmetry finder's distance, in the model's units of length.
    characters : sequence or mapping
        The declaration, as ``characters=`` gives it.

    Returns
    -------
    dict or None
        ``{"group", "assignment", "kept", "excluded", "read_in"}``: the
        free-standing group, the orbitals' orbits in it, and
        :func:`_declared_cell_reduction`'s reading there.  ``None`` where the
        identification, the assignment of the orbitals or the reading is
        refused, as it is for words written in a substrate's site irreps.
    """
    cell, cell_typat = _decorated_cell(model, labels, environment="free-standing",
                                       decorations=extras)
    group = _group_of(cell, typat=cell_typat, spinor=bool(spinor), symprec=symprec)
    if isinstance(group, Refusal):
        return None
    assignment = _assign_orbitals(model, group, typat=labels)
    if isinstance(assignment, Refusal):
        return None
    got = _declared_cell_reduction(model, group, assignment, spin=spin,
                                   characters=characters)
    if isinstance(got, Refusal):
        return None
    kept, excluded, read_in = got
    return {"group": group, "assignment": assignment, "kept": kept,
            "excluded": excluded, "read_in": read_in}


def _restricted_contents(model, written, group, layout, *, spin, characters) -> tuple:
    r"""The declaration read in a larger group's site groups, restricted to this group's.

    The words are read in the site groups of ``written["group"]``, a larger
    group of the same point set, on the multiplets ``layout`` holds.  Each
    reading is then re-keyed onto the operations of ``group``, matched by
    rotation and translation modulo the lattice
    (:func:`.._subgroups._operation_key`); the operations ``group`` does not
    have drop out, which is the restriction to its site groups (``A2''`` of
    ``-6m2`` read as the ``A1`` of ``3m``).

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    written : dict
        The declaration read in the larger group, as
        :func:`_layer_declared_reading` returns it.
    group : GroupData
        The group the readings are restricted to.
    layout : SiteMultiplets
        The multiplets the Hamiltonian measures in ``group``.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    characters : sequence or mapping
        The declaration, as ``characters=`` gives it.

    Returns
    -------
    tuple of DeclaredContent
        One for each declaration read, keyed on the operations of ``group``.
        ``()`` where the words name nothing in the larger group's site groups
        either.
    """

    larger, there = written["group"], written["assignment"]
    kept_u = {int(s.ind) for s in larger._symmetries if not s.time_reversal}
    kept_au = {int(s.ind) for s in larger._symmetries if s.time_reversal}
    try:
        tables = _site_tables(larger, there, kept_u, kept_au, range(len(there.site_orbitals)))
        decls = _normalize_declaration(characters, model, there, larger, tables, layout,
                                       spin=spin, read_in=written.get("read_in"))
    except _sitechars.DeclarationError:
        return ()
    theirs = {_operation_key(s): int(s.ind) for s in larger._symmetries}
    pairs = [(theirs[_operation_key(s)], int(s.ind)) for s in group._symmetries
             if _operation_key(s) in theirs]
    return _rekey_contents(tuple(d["content"] for d in decls), pairs, {})


def _carried_reduction(group, assignment, own, wider):
    r"""The operations of the cell the declaration leaves, read here and in a larger group.

    Each operation ``wider`` excludes is excluded in ``group`` too, matched by
    rotation and translation modulo the lattice
    (:func:`.._subgroups._operation_key`), beside the exclusions of ``own``.
    Only exclusions by the declaration itself are carried from either reading.
    The operations kept are then the largest subgroup of ``group`` inside those
    neither excludes, asked here
    (:func:`.._declaration._largest_compatible_group`).

    Parameters
    ----------
    group : GroupData
        The cell's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    own : tuple or Refusal
        :func:`_declared_cell_reduction`'s reading on ``group``, or the
        refusal it made.
    wider : dict
        The declaration read in a larger group of the same point set, as
        :func:`_layer_declared_reading` returns it.

    Returns
    -------
    tuple
        ``(kept, excluded, read_in)`` on ``group``, as
        :func:`_declared_cell_reduction` returns it.  ``read_in`` is the one
        ``own`` holds, or the one of ``wider`` where ``own`` is a refusal.
    """

    if isinstance(own, Refusal):
        excluded, read_in = {}, dict(wider["read_in"])
    else:
        _kept, excluded, read_in = own
        excluded, read_in = ({i: rec for i, rec in excluded.items()
                              if not _outside_the_group(rec)}, dict(read_in))
    lattice = np.asarray(group._lattice, dtype=float)
    theirs = {_operation_key(s): int(s.ind) for s in wider["group"]._symmetries}
    # a site of the larger group's orbits by the one it is here, through an
    # orbital on it: the two groups number their sites each their own way
    there = np.asarray(wider["assignment"].site_of_orbital, dtype=int)
    here = np.asarray(assignment.site_of_orbital, dtype=int)
    site_here = {int(there[o]): int(here[o]) for o in range(len(there))}
    for sym in group._symmetries:
        ind = int(sym.ind)
        rec = wider["excluded"].get(theirs.get(_operation_key(sym)))
        if rec is None or ind in excluded or _outside_the_group(rec):
            continue
        name = _record_name(sym, lattice)
        a, b = (site_here[int(s)] for s in rec["sites"])
        where = next((w for orbs, w in wider["read_in"].items()
                      if set(orbs) & set(rec["orbitals"])), "")
        excluded[ind] = dict(
            rec, ind=ind, name=name, sites=(a, b),
            note=(f"#{ind} {name}: {rec['reason']} ("
                  + (f"site {a}" if a == b else f"site {a} -> site {b}")
                  + f", orbitals {rec['orbitals']})"
                  + (f", as the {where} reads it" if where else "")))
    declared = {int(o) for reading in (read_in, wider["read_in"]) for orbs in reading
                for o in orbs}
    kept, excluded = _largest_compatible_group(group, assignment, declared, excluded)
    return kept, excluded, read_in


def _declared_cell(base_cell, base_typat, group, kept, *, spinor, include_TR, symprec):
    r"""The cell lowered to the operations the declaration leaves, and its group.

    The cell is built and identified as a descent's is (:func:`_orbit_cell`),
    from the generic points :data:`_DECLARED_POINTS`, the orbit added as a
    species ``base_typat`` does not hold.

    Parameters
    ----------
    base_cell : pythtb.TBModel
        The point set ``group`` was identified on, as a model.
    base_typat : sequence
        The species of each point of ``base_cell``.
    group : GroupData
        The cell's group.
    kept : collection of int
        The ``.ind`` the declaration leaves (:func:`_declared_cell_reduction`);
        its unitary operations are the ones carried.
    spinor, include_TR : bool
        Whether the cell is identified in double-valued tables, and with time
        reversal adjoined.
    symprec : float
        The symmetry finder's distance, in the model's units of length.

    Returns
    -------
    tuple or Refusal
        ``(group, cell, cell_typat, pairs, point, orbit, species)``: what
        :func:`_orbit_cell` returns, and the species of the orbit.  The
        :class:`..refusal.Refusal` an identification made, where one is
        refused.

    Raises
    ------
    ..refusal.InternalError
        Where the unitary operations of ``kept`` do not close, or no generic
        point gives a cell read as exactly them.
    """
    kept_syms = [s for s in group._symmetries
                 if int(s.ind) in set(kept) and not s.time_reversal]
    leaks = _descent_closure(group, kept_syms)
    if leaks:                                    # pragma: no cover: defensive
        raise InternalError(
            f"the {len(kept_syms)} operations kept from those that carry the site irreps "
            f"declared with characters= are not a group: {len(leaks)} of their products "
            f"leave the set, the first being #{leaks[0][0]} * #{leaks[0][1]} = "
            f"#{leaks[0][2]}. Report it, with the model and the call.")
    species = _DECLARED_SPECIES
    while any(str(t) == species for t in base_typat):   # pragma: no cover: defensive
        species += "_"
    names = {int(s.ind): _record_name(s, group._lattice) for s in group._symmetries}

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What one reduced cell was identified as, with its orbit and symprec.

        ``got`` is the group the cell was identified as and ``unitary`` its
        unitary operations; ``unmatched_kept`` and ``unmatched_found`` are
        what :func:`_match_operations` left unmatched, and ``orbit`` the orbit
        added to the cell.
        """
        return (
            f"the cell built to carry the {len(kept_syms)} operations kept from those that "
            "carry the site irreps declared with characters= was identified as "
            f"{got.name} #{got.number_str}, whose {len(unitary)} unitary operations are "
            "not that set: "
            + (f"{len(unmatched_kept)} of the operations kept "
               + ", ".join(f"#{i} {names.get(int(i), '?')}" for i in unmatched_kept)
               + " were not found in it" if unmatched_kept else "")
            + (" and " if unmatched_kept and unmatched_found else "")
            + (f"{len(unmatched_found)} of its operations {unmatched_found} are not among "
               "them" if unmatched_found else "")
            + ". The orbit added to the cell is "
            f"{np.asarray(orbit, dtype=float).tolist()}, at symprec {float(symprec):.3g}.")

    made = _orbit_cell(base_cell, base_typat, group, kept_syms, points=_DECLARED_POINTS,
                       species=species, spinor=spinor, include_TR=include_TR,
                       symprec=symprec, diagnose=_diagnose)
    if isinstance(made, Refusal):
        return made
    got, cell, cell_typat, pairs, point, orbit_padded = made
    return got, cell, cell_typat, pairs, point, orbit_padded, species


def _declared_contents(model, group, assignment, validated, layout, *, spin, characters,
                       read_in=None):
    r"""The round's declarations, read on the multiplets the Hamiltonian measures.

    The declaration is read in the site tables of the operations of ``group``
    the detector kept, and each site's declarations are then read on the
    measured multiplets (:func:`.._declaration._declared_blocks`).  The
    multiplets are not changed by the declaration.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The round's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    layout : SiteMultiplets
        The measured multiplets.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    characters : sequence or mapping
        The declaration as ``characters=`` gives it, or
        ``{orbitals: DeclaredContent}`` read on an earlier round, whose
        constraints are carried rather than read again.
    read_in : dict, default None
        :func:`_declared_cell_reduction`'s ``read_in``, so that a declaration
        read in a cell the declaration lowered still says where its word was
        read.

    Returns
    -------
    tuple or Refusal
        ``(decls, layout)``: one record per declaration read, each with its
        :class:`.._sitechars.DeclaredContent` and the channels it lies on,
        and ``layout`` as handed in.  The :class:`..refusal.Refusal` the
        reading made, as :func:`.._build._build_rep` makes it, or
        ``site_group`` where a site group has no table.
    """

    kept_u = {int(s.ind) for s in group._symmetries
              if int(s.ind) in set(validated._indices) and not s.time_reversal}
    kept_au = {int(s.ind) for s in group._symmetries
               if int(s.ind) in set(validated._indices) and s.time_reversal}
    try:
        tables = _site_tables(group, assignment, kept_u, kept_au,
                              range(len(assignment.site_orbitals)))
        decls = _normalize_declaration(characters, model, assignment, group, tables,
                                       layout, spin=spin, read_in=read_in)
    except _sitechars.DeclarationError as exc:
        return Refusal(exc.gate, exc.message, context=dict(exc.context, multiplets=layout))
    except Exception as exc:                     # noqa: BLE001 - a site group that has no table
        return Refusal(getattr(exc, "gate", "site_group"), str(exc),
                       context={"multiplets": layout})
    decls = _declared_blocks(decls, layout, assignment, tables, spin=spin)
    chan_of = layout._chan_of
    for d in decls:
        d["channels"] = tuple(sorted({int(chan_of[int(o)])
                                      for o in d.get("block", d["orbitals"])}))
    return decls, layout


def _declared_parts_of(decls) -> tuple:
    """The declarations a round was read from, as they were written and resolved.

    ``decls`` is the round's declarations, as :func:`_declared_contents`
    returns them.  Where a site's declarations were read as one sum over its
    multiplets, the declarations it was made of stand in its place, so that
    each later round, and the build, reads them on its own multiplets.
    Returns the :class:`.._sitechars.DeclaredContent` of each declaration.
    """
    return tuple(g["content"] for d in decls for g in d.get("groups", (d,)))


def _declared_contents_wider(model, group, assignment, layout, *, spin, characters,
                             read_in=None) -> tuple:
    r"""The declaration read in the site groups of every operation of the round's group.

    :func:`_declared_contents` reads only the operations the detector kept,
    whose site groups can be too small to name a word: ``A1`` of a ``3m`` site
    names nothing in a trivial site group.  Here the words are read in the site
    groups of all of ``group`` and, where it is a layer set in vacuum along one
    direction, of its operations that keep the layer's side.  The two are
    tried in that order, and the first whose site groups name every word is
    returned.  The reading is carried through the descent and restricted, as
    any declaration is.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The round's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    layout : SiteMultiplets
        The measured multiplets.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    characters : sequence or mapping
        The declaration, as ``characters=`` gives it.
    read_in : dict, default None
        :func:`_declared_cell_reduction`'s ``read_in``.

    Returns
    -------
    tuple of DeclaredContent
        One for each declaration read.  ``()`` where the words name nothing
        in either set of site groups.
    """

    readings = [list(group._symmetries)]
    padded = tuple(int(d) for d in getattr(group, "_vacuum_dirs", ()) or ())
    if len(padded) == 1:
        readings.append([s for s in group._symmetries
                         if _side_preserving(s, padded[0])])
    for syms in readings:
        kept_u = {int(s.ind) for s in syms if not s.time_reversal}
        kept_au = {int(s.ind) for s in syms if s.time_reversal}
        try:
            tables = _site_tables(group, assignment, kept_u, kept_au,
                                  range(len(assignment.site_orbitals)))
            decls = _normalize_declaration(characters, model, assignment, group, tables,
                                           layout, spin=spin, read_in=read_in)
        except InternalError:
            raise
        except _sitechars.DeclarationError:
            continue
        except Exception:                        # noqa: BLE001 - a site group that has no table
            continue
        return tuple(d["content"] for d in decls)
    return ()


def _restricts_to(contents, group, assignment) -> bool:
    r"""Whether carried declarations are restrictions to the site groups of ``group``.

    ``contents`` holds :class:`.._sitechars.DeclaredContent` carried to
    ``group``, and ``assignment`` the orbitals' orbits in it.  A declaration
    is a restriction where it states a character for every unitary operation
    of ``group`` that fixes the site of its first orbital.  Returns ``True``
    where every one of ``contents`` is.
    """
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    unitary = {int(s.ind) for s in group._symmetries if not s.time_reversal}
    for content in contents:
        site = int(site_of[int(content.orbitals[0])])
        here = {int(i) for i in assignment.site_group[site]} & unitary
        if not here <= {int(k) for k in content.constraints}:
            return False
    return True


def _rekey_contents(contents, pairs, refuted):
    """The declarations carried onto a re-identified group.

    ``contents`` holds :class:`.._sitechars.DeclaredContent`, ``pairs`` the
    ``(old, new)`` ``.ind`` of the operations the two groups share, and
    ``refuted`` the old ``.ind`` to leave out.  Each constraint is re-keyed
    (:meth:`.._sitechars.DeclaredContent.rekeyed`), and the operations the new
    group does not have drop out, which is the restriction to its site groups.
    A restriction that is reducible is carried as it is.  Returns each
    declaration re-keyed.
    """
    drop = {int(i) for i in (refuted or ())}
    keep = tuple((int(a), int(b)) for a, b in pairs if int(a) not in drop)
    return tuple(c.rekeyed(keep) for c in contents)


def _declaration_refuted(group, validated, *, refuted, verdict, tolerances=EXACT):
    r"""The operations the declaration breaks, and the subgroup it leaves.

    The subgroup is the largest one of the unitary operations still standing on
    which the declaration is covariant with ``H``
    (:func:`.._declaration._covariant_subgroup`).  Where several of that order
    exist, it is the first by the tie rule.  The identity alone is always such
    a subgroup.  Every unitary operation outside it is broken by the
    declaration, and so is its product with time reversal, where that is
    standing.  The subgroup returned holds the product with time reversal of
    each unitary operation it keeps, where that is standing.

    Parameters
    ----------
    group : GroupData
        The round's group.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    refuted : dict
        The operations already refuted this round, by ``.ind``, which are not
        asked.
    verdict : _declaration._Verdict
        The round's one reading of the solution spaces
        (:func:`.._declaration._verdicts_and_choices`), whose requirements
        are read.
    tolerances : Tolerances, default EXACT
        What the call declared, against which a character's distance from the
        value a requirement asks is read.

    Returns
    -------
    _DeclarationVerdict or Refusal
        The verdict, each operation broken recorded with
        ``refuted_by = "declaration"``.  The ``rep_search_too_large``
        :class:`..refusal.Refusal` of a search past its ceiling.
    """

    syms = group._symmetries
    standing = set(validated._indices) - set(refuted)
    kept_u = [int(s.ind) for s in syms
              if int(s.ind) in standing and not s.time_reversal]
    got = verdict
    unitary_only = not any(s.time_reversal and int(s.ind) in standing for s in syms)
    sub = _covariant_subgroup(group, kept_u, got.requirements,
                              unitary_only=unitary_only,
                              tol=tolerances.noise(CHARACTER),
                              joint=got.joint,
                              search=got.search,
                              matrices=got.matrices)
    if isinstance(sub, Refusal):
        return sub
    lattice = np.asarray(group._lattice, dtype=float)
    # the primed partner of a broken unitary operation, through the group's own
    # product table: T * g, with T the pure time reversal
    prod = group._get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(syms)}
    partner = {}
    t_pos = next((p for p, s in enumerate(syms) if s.time_reversal
                  and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                     np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    if t_pos is not None:
        for u in kept_u:
            q = int(syms[int(prod[t_pos, at[int(u)]])].ind)
            if q in standing:
                partner[int(u)] = q
    broken = {}
    # the declarations read together, and those the sectors alone weigh, each
    # named where it is read
    read_together = tuple(got.joint) + tuple(got.by_sectors)
    for ind in sub.broken:
        rows = tuple(r for r in got.rows if int(r.ind) == int(ind))
        culprit = next((r for r in rows if r.required is not None
                        and int(ind) in got.reasons), rows[-1] if rows else None)
        said = None if culprit is None else (culprit.declared, culprit.measured)
        note = got.reasons.get(int(ind)) or _joint_reason(read_together, ind)
        if note is None:
            note = ("the declared content is carried by this operation on its own, but "
                    "no group of operations containing it and those kept carries every "
                    "declared multiplet in its declared irrep at once")
        together = next((j for j in read_together
                         if any(int(ind) in row for row in j.theta.values())), None)
        if culprit is None and together is not None:
            # a declaration read over several multiplets: the declared value
            # there, and what its multiplets carry, each against its own part's
            # scale
            said = (together.decl["constraints"].get(int(ind), 0.0),
                    sum(v for row in together.theta.values()
                        for g, v in row.items() if int(g) == int(ind)))
        orbitals = ({int(o) for r in rows for o in r.orbitals} if rows else
                    {int(o) for o in (together.decl["orbitals"] if together else ())})
        rec = {"ind": int(ind), "refuted_by": "declaration", "note": str(note),
               "orbitals": tuple(sorted(orbitals)),
               "declared": (None if said is None else complex(said[0])),
               "measured": (None if said is None else complex(said[1])),
               "rows": _returned_rows(rows)}
        broken[int(ind)] = rec
        if int(ind) in partner:
            mate = dict(rec, ind=int(partner[int(ind)]))
            mate["note"] = (
                f"the primed copy of #{int(ind)} "
                f"{_record_name(syms[at[int(ind)]], lattice)}, which the "
                "declared orbital content breaks: a site character is a character of the "
                "unitary site group, so the two stand or fall together")
            broken[int(partner[int(ind)])] = mate
    # the subgroup returned holds the primed copy of each unitary operation
    # kept, where it stands: a site character decides about the unitary half,
    # and the primed copy is kept with it
    grey = set(sub.subgroup) | {partner[i] for i in sub.subgroup
                                   if i in partner}
    return _DeclarationVerdict(refuted=broken, subgroup=tuple(sorted(grey)),
                               candidates=sub.candidates,
                               rows=_returned_rows(got.rows))


def _operation_verdicts(model, group, assignment, validated, *, tolerances, nk,
                        context, spin, layout, decls=None,
                        reference_orbital=0, max_candidates=2 ** 20, free_standing=None,
                        centers=None, sectors=None, own_form=None, square=None):
    r"""The verdicts on the round's operations past the detector's, with their reasons.

    The detector's verdicts are in ``validated``, and the centers' and each
    operation's own form are handed in.  Where the own form refutes a unitary
    operation the detector found obeyed, only the pure translations are read
    besides, and the group is lowered on that.  Otherwise, in order: the
    tensor structure of the operations the detector left undetermined, or of
    every one it kept where the multiplets are stated (:func:`_ansatz_refuted`);
    the one reading of the solution spaces, which pins what the hoppings fix
    (:func:`.._declaration._verdicts_and_choices`); the declaration
    (:func:`_declaration_refuted`); the pure translations
    (:func:`_lattice_verdict`); and, where every unitary operation still
    stands, the states' projective class on a spinless reading
    (:func:`_projective_class_refuted`, or on a substrate
    :func:`_projective_class_carried`).

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : GroupData
        The round's group.
    assignment : OrbitalAssignment
        The orbitals' orbits in ``group``.
    validated : ValidatedGroup
        The detector's verdicts on ``group``.
    tolerances : Tolerances
        What the call declared, against which every verdict reads noise.
    nk : int
        The points of the grid along each periodic direction.
    context : _MeasurementContext
        The measurement context, whose grid of ``H`` the translations reuse.
    spin : SpinBasis or None
        The declared spin partners, or ``None``.
    layout : SiteMultiplets
        The measured or stated multiplets.
    decls : sequence of dict, default None
        The round's declarations, as :func:`_declared_contents` returns them;
        ``None`` or empty where nothing is declared.
    reference_orbital : int, default 0
        The orbital the lowest-angular-momentum rule reads the crystal from
        first, the one the build is given.
    max_candidates : int, default 2 ** 20
        The most candidates a block enumeration may try.
    free_standing : dict, default None
        On a substrate, the free-standing cell of the same point set, on
        which the projective class is read.
    centers : dict, default None
        The operations the centers are not carried by within the declared
        precision, by ``.ind``, each with its record
        (:func:`.._realized._centers_carried`).  They stand whatever else is
        read.
    sectors : SectorReading, default None
        The sectors of the Hamiltonian's conserved matrices
        (:func:`.._sectors.read_sectors`).
    own_form : dict, default None
        The operations their own form refutes, by ``.ind``, each with its
        record (:func:`.._measure._own_form_read`).
    square : _Square, default None
        Time reversal's square (:func:`_time_reversal_square`), which decides
        whether the projective class is read on a group holding time reversal.

    Returns
    -------
    tuple or Refusal
        ``(refuted, lattice, declaration, choices, projective)``: the
        operations refuted past the detector, by ``.ind``, each with its
        record; :func:`_lattice_verdict`'s record; the
        :class:`_DeclarationVerdict`, ``None`` where nothing is declared; the
        reading of the solution spaces; and the projective class's record,
        ``None`` where the class is not read or is the table's.  Where the
        own form lowers the group, the last three are ``None``.  The
        :class:`..refusal.Refusal` the multiplet blocks, the tensor test, the
        reading or the declaration made.
    """
    syms = group._symmetries
    centered = dict(centers or {})
    own_form = dict(own_form or {})
    # The own form of each operation the detector found obeyed, first.  A
    # unitary one it refutes lowers the group at once, with the pure
    # translations measured as for any descent, and the next round reads the
    # rest on the group that holds, since the multiplets this round reads were
    # measured with that operation.
    obeyed = set(validated.obeyed)
    lifted = {i: r for i, r in own_form.items() if i in obeyed}
    if any(int(s.ind) in lifted and not s.time_reversal for s in syms):
        lattice = _lattice_verdict(model, group, assignment, validated, lifted,
                                   tolerances.energy,
                                   _detection_grid(int(group._dim_k), int(nk)),
                                   context=context, spin=spin,
                                   matrix_tol=tolerances.matrix or None)
        refuted = dict(lifted)
        if lattice and lattice["refuted"]:
            refuted = {**refuted, **lattice["refuted"]}
        return {**refuted, **centered}, lattice, None, None, None
    # The multiplet blocks, built once per round: the tensor test, the reading
    # of the declaration and the representation all read them.  They are read
    # on the operations whose own form holds, since one refuted by it has no
    # matrix to read a block off.
    blocks = _multiplet_blocks(
        model, group, assignment, spin, validated, layout,
        [p for p, s in enumerate(group._symmetries)
         if int(s.ind) in set(validated._indices) and int(s.ind) not in own_form],
        ref_chan=int(np.asarray(layout._chan_of, dtype=int)[0]),
        max_candidates=max_candidates, sectors=sectors)
    if isinstance(blocks, Refusal):
        return blocks
    # The undetermined operations, each told apart from its partners by its own
    # form, read above, and by its tensor structure on the multiplets.
    shared = {i: r for i, r in own_form.items() if i not in obeyed}
    got = _ansatz_refuted(model, group, assignment, validated,
                          layout=layout, spin=spin, blocks=blocks,
                          max_candidates=max_candidates, skip=set(shared))
    if isinstance(got, Refusal):
        return got
    # the operations their own form or their tensor test refuted, and any
    # antiunitary one found obeyed that its own form refuted: those start no
    # descent, as no antiunitary refutation does; and the centers' verdict,
    # which every reading below takes as given
    refuted = {**lifted, **got[1], **shared, **centered}
    # The one reading of the solution spaces, over the operations still
    # standing: the verdict on the declaration and the choice the
    # representation is built from, together.
    standing = set(validated._indices) - set(refuted)
    positions = [p for p, sym in enumerate(syms)
                 if int(sym.ind) in standing and not sym.time_reversal]
    # the time reversal that anchors the phases a crystal of multiplets is read
    # in, chosen as the build chooses it: the pure one where it stands
    reversing = [p for p, sym in enumerate(syms)
                 if int(sym.ind) in standing and sym.time_reversal]
    pure = [p for p in reversing
            if np.array_equal(np.rint(np.asarray(syms[p].rotation)).astype(int),
                              np.eye(3, dtype=int))
            and np.allclose(np.asarray(syms[p].translation) % 1.0, 0.0, atol=1e-8)]
    choices = _verdicts_and_choices(model, group, assignment, spin, validated,
                                    layout, decls or (), positions, blocks=blocks,
                                    ref_orb=int(reference_orbital),
                                    space_tol=tolerances.noise(IN_SPACE),
                                    tol=tolerances.noise(CHARACTER),
                                    max_candidates=max_candidates,
                                    reversal=(pure or reversing or [None])[0],
                                    sectors=sectors)
    if isinstance(choices, Refusal):
        return choices
    declaration = None
    if decls:
        declaration = _declaration_refuted(group, validated,
                                           refuted=refuted, verdict=choices,
                                           tolerances=tolerances)
        if isinstance(declaration, Refusal):
            return declaration
        # integer operation indices as keys: a mapping merge, not keyword
        # arguments to a call
        refuted = {**refuted, **declaration.refuted}
    # The pure translations, measured on the same group.  A cell whose point set
    # repeats under one is identified as a lattice supergroup.  Where the
    # Hamiltonian obeys the translation the model is a supercell; where it
    # breaks it, the model's own lattice is the translation group, and the
    # first descent is to it.
    lattice = _lattice_verdict(model, group, assignment, validated, refuted,
                               tolerances.energy,
                               _detection_grid(int(group._dim_k), int(nk)),
                               context=context, spin=spin,
                               matrix_tol=tolerances.matrix or None)
    if lattice and lattice["refuted"]:
        refuted = {**refuted, **lattice["refuted"]}
    # The states' projective class, last, read only where every unitary
    # operation still stands.  On a spinless reading whose states carry a class
    # the single-valued tables do not, such as spinors in the orbital index,
    # the operations no single-valued representation carries are refuted, as
    # broken ones are; on a group holding time reversal, only where every
    # operation combined with it stands.  On a substrate the class is read on
    # the free-standing layer, and its part that preserves the layer's normal
    # is carried.
    projective = None
    unitary = {int(s.ind) for s in syms if not s.time_reversal}
    if not unitary & (set(validated.broken) | set(refuted)):
        if free_standing is not None:
            more, projective = _projective_class_carried(model, group, validated,
                                                         free_standing=free_standing,
                                                         refuted=refuted, square=square,
                                                         tolerances=tolerances)
        else:
            more, projective = _projective_class_refuted(group, validated,
                                                         assignment=assignment,
                                                         decls=decls or (),
                                                         refuted=refuted, square=square,
                                                         tolerances=tolerances)
        refuted = {**refuted, **more}
    return {**refuted, **centered}, lattice, declaration, choices, projective


def _descend_once(base_cell, base_typat, group, validated, *, spinor, include_TR,
                  symprec, step, refuted=None, lattice_record=None,
                  declaration=None, projective=None, centers=None) -> tuple | Refusal:
    r"""One descent step: the cell lowered to the unitary operations that hold.

    The operations kept, ``K``, are those of ``validated._indices`` that are
    unitary and not in ``refuted``.  Where they do not close, ``K`` is the
    largest subgroup inside them, the first by the tie rule where several of
    that order exist (:func:`.._subgroups._resolve_closure`).  That subgroup
    can be the identity alone, so a set that does not close is never refused.
    An operation left out is recorded under ``refuted_by = "closure"``, or
    ``"tie"`` where only subgroups the rule did not keep hold it.  The cell is
    ``base_cell`` with the orbit of a generic point under ``K`` added,
    identified and checked by :func:`_orbit_cell` from the points
    :data:`_DESCENT_POINTS`.  The model is not changed.

    Parameters
    ----------
    base_cell : pythtb.TBModel
        The point set ``group`` was identified on, as a model, with any
        species the environment puts under it.
    base_typat : sequence
        The species of each point of ``base_cell``.
    group : GroupData
        The round's group.
    validated : ValidatedGroup
        The detector's verdicts on ``group``; its discrepancies are recorded
        as dropped, except on operations the centers refuted.
    spinor, include_TR : bool
        Whether the cell is identified in double-valued tables, and with time
        reversal adjoined.
    symprec : float
        The symmetry finder's distance, in the model's units of length.
    step : int
        The step's number, from 1.
    refuted : dict, default None
        The verdict stage's refutations, by ``.ind``, each a record with its
        ``refuted_by`` (:func:`_operation_verdicts`).
    lattice_record : dict, default None
        :func:`_lattice_verdict`'s record.  Where it is not empty, it is
        carried onto the step as ``lattice``, with its note.
    declaration : _DeclarationVerdict, default None
        The declaration's verdict, carried onto the step as ``declaration``,
        with the readings that tie, where several do.
    projective : dict, default None
        The projective class's record, carried onto the step as
        ``projective``.
    centers : _CentersVerdict, default None
        The centers' verdict under a declared precision
        (:func:`.._realized._centers_carried`), carried onto the step as
        ``centers``: the precision, the move, the operations kept, the
        subgroups that tie and a note.

    Returns
    -------
    tuple or Refusal
        ``(group, cell, cell_typat, record)``: the group identified, the cell
        and its species, and the step's record.  The record holds ``step``,
        ``from`` and ``to`` (each a group's ``number_str``, ``name`` and
        ``size``), ``dropped`` and ``n_dropped`` (one entry per operation
        dropped, with its ``refuted_by``), ``refuted_by`` (the count of each),
        ``kept``, ``pairs`` (:func:`_match_operations`) and ``cell`` (the
        generic point, its orbit and its species), and ``tie``,
        ``declaration``, ``projective``, ``centers`` and ``lattice`` where
        they apply.  The :class:`..refusal.Refusal` an identification made,
        where one is refused.

    Raises
    ------
    ..refusal.InternalError
        Where no generic point gives a cell read as exactly ``K``.
    """
    refuted = dict(refuted or {})
    antiunitary = {int(s.ind) for s in group._symmetries if s.time_reversal}
    keep = set(validated._indices) - antiunitary - set(refuted)
    kept = [s for s in group._symmetries
            if int(s.ind) in keep and not s.time_reversal]
    lattice = np.asarray(group._lattice, dtype=float)
    by_ind = {int(s.ind): s for s in group._symmetries}

    def _name(ind):
        """The name of operation ``ind``, as the step lists it (:func:`_record_name`)."""
        return _record_name(by_ind[int(ind)], lattice)

    # The operations kept, resolved to a group where they do not close.  An
    # operation the declared tolerance puts on the wrong side of the gate leaves
    # them here rather than being returned or refused; where several subgroups
    # tie, the one kept is the first by the tie rule (`_resolve_closure`), and
    # the step says which and why.
    left_out: tuple = ()
    tie = None
    if _descent_closure(group, kept):
        resolved, dropped, tied = _resolve_closure(group, [int(s.ind) for s in kept])

        def _set(inds):
            return "{" + ", ".join(_name(i) for i in inds) + "}"

        if tied:
            left = [_name(i) for i, why in dropped if why == "tie"]
            tie = {"tied": tied, "kept": tied[0], "note": (
                f"the operations measured to hold in #{group.number_str} do not close "
                "into a group, and the largest subgroups inside them tie: "
                + " and ".join(_set(c) for c in tied)
                + ". The one kept is the first by the tie rule `realized_group` states -- "
                f"{_tie_rule_words(group)} -- since {_tie_reason(group, tied)}; "
                + " and ".join(left)
                + (" is" if len(left) == 1 else " are")
                + " left out, under `refuted_by = \"tie\"`.")}
        left_out = tuple(dict({"ind": int(i), "name": _name(i),
                               "refuted_by": str(why), "residual": float("nan"),
                               "step": int(step), "of": str(group.number_str)},
                              **({"note": (
                                  "the largest subgroups the operations measured to hold "
                                  f"contain tie, and the one the tie rule keeps, "
                                  f"{_set(tied[0])}, does not hold it")}
                                 if why == "tie" else {}))
                         for i, why in dropped)
        keep = {i for i in keep if i in resolved}
        kept = [s for s in kept if int(s.ind) in resolved]

    def _entry(i):
        """The step's entry for the operation ``i`` of ``refuted``.

        The entry holds the fields of the measurement that broke it.  Where
        the centers, the declaration or the projective class broke it, the
        residual is ``nan`` and the entry holds what that verdict rests on:
        for the declaration, the orbitals, what they were declared and what
        they measure, with the declared site irreps checked against it.
        """
        rec = refuted[i]
        why = str(rec.get("refuted_by", "tensor structure"))
        base = {"ind": int(i), "name": _name(i), "refuted_by": why,
                "step": int(step), "of": str(group.number_str)}
        if why == "centers":
            # a verdict about where the centers are carries no residual of H:
            # what it rests on is how far the operation alone would move one
            return dict(base, residual=float("nan"), miss=float(rec["miss"]),
                        precision=float(rec["precision"]), note=str(rec["note"]))
        if why == "declaration":
            return dict(base, residual=float("nan"),
                        orbitals=tuple(int(o) for o in rec.get("orbitals", ())),
                        declared=rec.get("declared"), measured=rec.get("measured"),
                        note=str(rec.get("note", "")), rows=rec.get("rows"))
        if why == "projective class":
            # a verdict about the states' class carries no residual either: the
            # sign it rests on sits a whole unit from the table's
            return dict(base, residual=float("nan"), pair=rec.get("pair"),
                        names=rec.get("names"),
                        commutation_phase=rec.get("commutation_phase"),
                        spin_half=rec.get("spin_half"), note=str(rec.get("note", "")))
        return dict(base, residual=float(rec["scaled"]),
                    # how far its own form carries H, where that was measured
                    **({} if rec.get("covariance") is None else
                       {"covariance": float(rec["covariance"])}))

    # the centers' verdict stands for the operations it refutes, whatever the
    # detector read on them: each broken operation is listed once, with its reason
    centered = {int(i) for i, r in refuted.items()
               if isinstance(r, dict) and r.get("refuted_by") == "centers"}
    broken = tuple(
        {"ind": int(d.index), "name": _name(d.index),
         "refuted_by": str(d.refuted_by),
         "residual": float(d.residual),
         "step": int(step), "of": str(group.number_str)}
        for d in validated.discrepancies if int(d.index) not in centered) + tuple(
        _entry(i) for i in sorted(refuted)) + left_out

    species = _DESCENT_SPECIES
    while any(str(t) == species for t in base_typat):   # pragma: no cover: defensive
        species += "_"

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What one descent cell was identified as, with its orbit and symprec.

        ``got`` is the group the cell was identified as and ``unitary`` its
        unitary operations; ``unmatched_kept`` and ``unmatched_found`` are
        what :func:`_match_operations` left unmatched, and ``orbit`` the orbit
        added to the cell.
        """
        where = np.round(np.asarray(orbit, dtype=float), 6).tolist()
        if int(got._lattice_index) > 1:
            return (
                f"the cell built to carry the {len(kept)} operations the Hamiltonian obeys "
                f"(from {group.name} #{group.number_str}) was identified as {got.name} "
                f"#{got.number_str} and is still not primitive: it carries the pure "
                "translation "
                f"{np.round(np.asarray(got._fractional_translations[0], dtype=float), 6).tolist()} "
                f"(index {int(got._lattice_index)}). The orbit added to the cell is "
                f"{where}, at symprec {float(symprec):.3g}.")
        return (
            f"the cell built to carry the {len(kept)} operations the Hamiltonian obeys (from "
            f"{group.name} #{group.number_str}) was identified as {got.name} "
            f"#{got.number_str}, whose {len(unitary)} unitary operations are not that set: "
            + (f"{len(unmatched_kept)} kept operations {unmatched_kept} were not found in it"
               if unmatched_kept else "")
            + (" and " if unmatched_kept and unmatched_found else "")
            + (f"{len(unmatched_found)} of its operations {unmatched_found} are not among "
               "the kept ones" if unmatched_found else "")
            + f". The orbit added to the cell is {where}, at symprec {float(symprec):.3g}.")

    # The cell handed in already carries the environment asked for, its shadow
    # species included, so nothing is put under it here: the orbit is all this
    # cell adds.
    made = _orbit_cell(base_cell, base_typat, group, kept, points=_DESCENT_POINTS,
                       species=species, spinor=spinor, include_TR=include_TR,
                       symprec=symprec, diagnose=_diagnose)
    if isinstance(made, Refusal):
        return made
    got, cell, cell_typat, pairs, point, orbit_padded = made

    record = {
        "step": int(step),
        "from": {"number_str": str(group.number_str), "name": str(group.name),
                 "size": int(group._size)},
        "to": {"number_str": str(got.number_str), "name": str(got.name),
               "size": int(got._size)},
        "dropped": broken, "n_dropped": len(broken),
        # how each verdict was reached: the detector's own reasons; "tensor
        # structure" for an operation it kept that the tensor structure refuted,
        # an undetermined member of a class or one whose own spin lift its space
        # does not carry; and the rest in the words above
        "refuted_by": {r: sum(1 for b in broken if b["refuted_by"] == r)
                       for r in sorted({b["refuted_by"] for b in broken})},
        "kept": tuple(sorted(int(s.ind) for s in kept)),
        "pairs": pairs,
        "cell": {"generic_point": tuple(float(x) for x in point),
                 "orbit": tuple(tuple(float(x) for x in row) for row in orbit_padded),
                 "species": species},
    }
    if tie is not None:
        record["tie"] = tie
    if declaration is not None:
        readings = tuple(declaration.candidates)
        record["declaration"] = {
            "subgroup": tuple(declaration.subgroup),
            "tied": readings,
            "rows": tuple(declaration.rows),
        }
        if readings:
            # the readings, one by one in the rule's order, by their operations'
            # names, which is how the environments' comparison names them
            record["declaration"]["names"] = tuple(
                tuple(_name(i) for i in r) for r in readings)
            record["declaration"]["note"] = (
                "several readings of the declared content are maximal and none contains "
                "the others: "
                + " and ".join("{" + ", ".join(_name(i) for i in r) + "}" for r in readings)
                + ". The one returned is the first by the tie rule `realized_group` states "
                f"-- {_tie_rule_words(group)} -- since {_tie_reason(group, readings)}; "
                "`descent.tied` lists them in that order.")
    if projective is not None:
        record["projective"] = dict(projective)
    if centers is not None:
        record["centers"] = {"precision": float(centers.precision),
                             "move": float(centers.move),
                             "kept": tuple(int(i) for i in centers.kept),
                             "tied": tuple(tuple(int(i) for i in c)
                                           for c in centers.tied),
                             "note": str(centers.note)}
    if lattice_record:
        record["lattice"] = dict(lattice_record)
        record["lattice"]["note"] = _lattice_note(lattice_record, got)
    return got, cell, cell_typat, record


#: Each cause :func:`_descent_counts` counts, as the record's one-line summary says it.
_DROPPED_BY_WORDS = {"centers": "by the centers", "hoppings": "by the hoppings",
                     "declaration": "by the declared content",
                     "closure": "left out to close a group", "tie": "by the tie rule",
                     "time reversal": "with time reversal",
                     "projective class": "by the states' projective class"}


def _descent_counts(steps, time_reversal) -> dict:
    r"""What a descent dropped, every step together, by cause.

    ``steps`` holds the steps' records (:func:`_descend_once`), and
    ``time_reversal`` the record of time reversal's measurement, or ``None``.
    Returns ``{cause: count}`` in the order the measurement meets the causes:
    ``"centers"``, beyond a declared precision; ``"hoppings"``, every
    operation dropped for no other cause; ``"declaration"``; ``"closure"``,
    operations the Hamiltonian obeys that are left out so that the operations
    kept form a group; ``"tie"``, operations only the subgroups the tie rule
    did not keep hold; ``"time reversal"``, the operations combined with it
    where it is measured broken between two steps; and ``"projective
    class"``.  The counts add up to the number of coset representatives of
    the lattice translations in the group the first step lowered, less that
    in the group the last arrived at, a reduction of the cell by the
    declaration between two steps aside.
    """
    dropped = [str(b["refuted_by"]) for st in steps for b in st["dropped"]]
    named = {r: dropped.count(r)
             for r in ("centers", "declaration", "closure", "tie", "projective class")}
    tr = time_reversal or {}
    between = sum(int(steps[k]["to"]["size"]) - int(steps[k + 1]["from"]["size"])
                  for k in range(len(steps) - 1)
                  if tr.get("outcome") == "measured broken"
                  and steps[k]["to"]["number_str"] == tr.get("grey_number_str"))
    return {"centers": named["centers"], "hoppings": len(dropped) - sum(named.values()),
            "declaration": named["declaration"], "closure": named["closure"],
            "tie": named["tie"], "time reversal": int(between),
            "projective class": named["projective class"]}


def _descent_clauses(counts) -> tuple:
    r"""A descent's counts by cause, and what its operations kept carry, in words.

    ``counts`` is what :func:`_descent_counts` returns.  Returns ``(split,
    lowered)``.  ``split`` says what dropped the cell's operations: each of
    the hoppings and the declaration that did, or, where any other cause did,
    each cause once in the order the measurement meets them.  ``lowered``
    describes the group the cell was lowered to, the largest group of its
    operations that carries what each cause asks, where the centers, the
    closure, the tie rule or the states' projective class dropped any; it is
    ``None`` where none of those four dropped any.
    """
    d_cen, d_hop, d_decl = counts["centers"], counts["hoppings"], counts["declaration"]
    d_closure, d_tie = counts["closure"], counts["tie"]
    d_tr, d_proj = counts["time reversal"], counts["projective class"]
    # Each cause is counted apart.  The hoppings and the declaration are two
    # different statements about the crystal, and the states' projective class
    # and the centers beyond the declared precision are neither, so a sum would
    # hide which one to act on.
    split = (
        f"the Hamiltonian broke {d_hop} of them by its hoppings and the declared "
        f"orbital content broke {d_decl} more" if d_hop and d_decl else
        f"the declared orbital content broke {d_decl} of them" if d_decl else
        f"the Hamiltonian broke {d_hop} of them")
    if d_cen or d_closure or d_tie or d_tr or d_proj:
        said = []
        if d_cen:
            said.append(f"the orbital centers are not carried by {d_cen} of them within "
                        "the declared precision")
        if d_hop:
            said.append(f"the Hamiltonian broke {d_hop} {'more' if said else 'of them'} "
                        "by its hoppings")
        if d_decl:
            said.append(f"the declared orbital content broke {d_decl} "
                        f"{'more' if said else 'of them'}")
        if d_closure:
            said.append(f"{d_closure} {'more' if said else 'of them'} "
                        f"{'is' if d_closure == 1 else 'are'} left out so that the operations "
                        "kept form a group")
        if d_tie:
            said.append(f"{d_tie} {'more' if said else 'of them'} "
                        f"belong{'s' if d_tie == 1 else ''} only to the subgroups the tie "
                        "rule did not keep")
        if d_tr:
            said.append("time reversal is measured broken, which leaves out the "
                        f"{d_tr} operations combined with it")
        if d_proj:
            said.append(f"{d_proj} {'more' if said else 'of them'}, which the Hamiltonian "
                        "obeys, carry no single-valued representation on these states")
        split = ", and ".join(said)
    lowered = None
    if d_cen or d_closure or d_tie or d_proj:
        kept_as = []
        if d_cen:
            kept_as.append("that carry the orbital centers within the declared precision")
        if d_hop or d_closure or d_tie:
            kept_as.append("that the Hamiltonian obeys")
        if d_decl:
            kept_as.append("that carry the declared orbital content")
        if d_proj:
            kept_as.append("on which these states carry a single-valued representation")
        lowered = (f"the largest group of its {'unitary ' if d_tr else ''}operations "
                   + _listed(kept_as) + (", the first by the tie rule" if d_tie else ""))
    return split, lowered


def _lattice_note(lattice_record, got=None) -> str:
    """The lattice descent in words, and the group it arrived at where ``got`` names it.

    ``lattice_record`` is :func:`_lattice_verdict`'s record of a translation
    the Hamiltonian broke.  ``got`` is the group the step arrived at, left out
    where it is not the group that holds: a grey group on which time reversal
    is then measured broken.  Returns the sentence.
    """
    tau = np.round(np.asarray(lattice_record["fractional_translations"][0],
                              dtype=float), 6)
    return (
        f"the cell allowed a lattice supergroup of index "
        f"{int(lattice_record['index'])} -- its point set repeats under the pure "
        f"translation {tau.tolist()} (reduced) -- and the Hamiltonian broke that "
        f"translation, together with "
        f"{int(lattice_record['coset_partners_refuted'])} coset partners of it. So "
        "this cell is "
        "not a supercell after all: the model's own lattice is its translation group"
        + (f", and the group returned is {got.name} #{got.number_str} on it."
           if got is not None else "."))
