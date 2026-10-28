"""The unitary descent: one verdict per operation of the cell, off the solution
spaces, the hoppings, the centres, the declaration, the lattice and the
antiunitary coset; the cell lowered to the group of the operations that hold;
and what each step dropped, by cause.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math

import numpy as np

from . import _bloch
from . import _sitechars
from .refusal import InternalError, Refusal
from ._tolerance import (
    ASSEMBLED, CHARACTER, EXACT, HEADROOM, IN_SPACE, IN_SPAN, SINGULAR, SNAP, ZERO,
    _CLASS_MARGIN, _EIGEN_WINDOW_CEIL, _EXACT_CLOSURE, _IN_BLOCKS,
    _TIME_REVERSAL_SQUARE_MARGIN, eigen_window, noise_floor)
from ._text import _DOUBLE_GROUP, _PER_COSET, _fmt_char, _joined
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
from ._centres import _position_defect
from ._naming import _site_tables
from ._multiplets import _ansatz_defects, _multiplet_blocks
from ._declaration import (
    _content_compatible, _covariant_subgroup, _declared_blocks,
    _declared_site_contents, _joint_reason, _largest_compatible_group,
    _normalize_declaration, _outside_the_group, _returned_rows,
    _verdicts_and_choices)
from ._layers import _decorated_cell
from ._measure import _detection_grid, _max_hamiltonian


def _antiunitary_entry(group, validated, ind) -> dict:
    """One operation combined with time reversal, as :attr:`RealizedGroup.beyond_grey` names it.

    Its index ``ind`` in the group it was measured in, that group's label
    (``of``), its ``name`` in the generator frame ``characters=`` is written
    in, its ``rotation`` and ``translation`` in the cell's reduced
    coordinates, and the ``residual`` and ``verdict`` the detector measured.
    Where the translation is not a lattice vector of the cell the name carries
    it, in Seitz form in the cell's reduced coordinates -- ``{E|1/2,0,0}'``,
    the half translation of an antiferromagnet's magnetic cell combined with
    time reversal, and never time reversal's own ``E'`` (:func:`_record_name`).
    """
    sym = next(s for s in group.symmetries if int(s.ind) == int(ind))
    return {"ind": int(ind),
            "name": _record_name(sym, group.lattice),
            "of": str(group.number_str),
            "rotation": tuple(tuple(int(round(x)) for x in row)
                              for row in np.asarray(sym.rotation, dtype=float)),
            "translation": tuple(float(x) for x in
                                 np.asarray(sym.translation, dtype=float).reshape(-1)),
            "residual": float(validated.residuals.get(int(ind), float("nan"))),
            "verdict": str(validated.verdicts.get(int(ind), "obeyed"))}


def _antiunitary_settled(group, validated, refuted, square, kept=None, *,
                         tolerances=EXACT) -> dict:
    r"""What one round measured of the operations combined with time reversal it asks last.

    A descent keeps the unitary operations measured to hold, ``kept``, and the
    group it arrives at is asked with time reversal adjoined again, so a
    ``g·T`` whose ``g`` is kept is measured again there.  One whose ``g`` is
    left out -- a mirror the Hamiltonian breaks, combined with time reversal
    -- is asked nowhere else, and this round's verdict on it is its last.  So
    is every operation combined with time reversal of the group time reversal
    is measured on (``kept`` left out), the last group that asks any of them:
    a descent after it -- the projective class of spinor states read
    spinless -- can still leave their unitary part out.

    What is kept is the measurement and no conclusion: whether the detector
    kept the operation, whether its tensor structure or its spin lift refuted
    it (``refuted``), its matrix, and, where its unitary part squares to the
    identity, its own square ``X X̄``, a sign no rescaling of ``X`` changes,
    against the sign the group's multiplication table gives it
    (:func:`_antiunitary_square`; time reversal's own is ``square``,
    :func:`_time_reversal_square`), the member of that sign being the matrix
    kept.  The operations are read once, after the group settles, against the
    group returned (:func:`_antiunitary_held`).

    Returns ``{"group", "ops", "detector", "refuted", "n", "closed",
    "reversal"}``: this round's group; per operation asked last here, keyed by
    its ``.ind``, the operation itself (``sym``), its entry as
    :attr:`RealizedGroup.beyond_grey` names it (``entry``,
    :func:`_antiunitary_entry`), its matrix (``member``, ``None`` where it
    is refuted) and the solution space it is a member of (``space``), which
    holds more than it where the Hamiltonian conserves something, a spin
    rotation; the operations the detector refuted, time reversal among them
    where its square is not the sign asked; those refuted here at all, by the
    detector, by their tensor structure or spin lift, or by their own square;
    how many were asked last here; whether the operations the detector kept
    close in this group; and whether it is the group time reversal was
    measured on.  ``tolerances`` is the record of what the call declared, which
    every reading of a solution space here judges noise against.
    """
    antiunitary = {int(s.ind): s for s in group.symmetries if s.time_reversal}
    if kept is None:
        asked = set(antiunitary)
    else:
        kept = {int(i) for i in kept}
        staying = {_operation_key(s)[:2] for s in group.symmetries
                   if not s.time_reversal and int(s.ind) in kept}
        asked = {i for i, s in antiunitary.items() if _operation_key(s)[:2] not in staying}
    detector = {i for i in antiunitary if i in set(validated.broken)}
    reversal = None if square is None else int(square["ind"])
    if square is not None and not square["obeyed"]:
        detector.add(reversal)
    failed = detector | {i for i in antiunitary if i in set(refuted or ())}
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    ops = {}
    for i in sorted(asked):
        member = None if i in failed else validated._intertwiners.get(i)
        if member is not None and i != reversal:
            # its own square, where its unitary part squares to the identity:
            # the member of the sign the table gives it is the matrix kept
            got = _antiunitary_square(group, validated, at[i], tolerances=tolerances)
            if got is not None:
                member = got["member"] if got["obeyed"] else None
        if member is None:
            failed.add(i)
        space = None if member is None else (getattr(validated, "_bases", None) or {}).get(i)
        ops[i] = {"sym": antiunitary[i], "entry": _antiunitary_entry(group, validated, i),
                  "member": None if member is None else np.asarray(member, dtype=complex),
                  "space": None if space is None else np.asarray(space, dtype=complex)}
    return {"group": group, "ops": ops, "detector": frozenset(detector),
            "refuted": frozenset(failed), "n": len(asked),
            "closed": bool(validated.closed), "reversal": kept is None}


def _time_reversal_measured(group, validated, refuted=None, *, tolerances=EXACT) -> tuple:
    """The verdict on the antiunitary coset, and what it costs.

    A cell is asked as a **grey** group — that is the honest allowance, since a
    point set says nothing about a Hamiltonian's time reversal — and the
    detector then measures the coset.  Two things can come back:

    * **every antiunitary operation obeyed.**  The realization is grey; nothing
      changes, and nothing beyond grey is read.
    * **any one refuted.**  Time reversal is measured broken and the
      realization is a *unitary* group.  The same cell is re-identified
      with ``include_TR=False``, so the labels, EBR rows and indicator that
      follow are the unitary group's own — a grey record with rows deleted
      would be neither.

    **Time reversal is obeyed only as the tables read it.**  Its square
    ``X X̄`` is a sign that belongs to the Hamiltonian (:func:`_time_reversal_square`),
    and the tables ask for one: ``+1`` single-valued, ``-1`` double-valued.  A
    time reversal whose solution space holds no member of the sign asked is
    broken, and the realization descends as it does for any Hamiltonian that
    breaks time reversal -- which is what spinor states in the orbital index of
    a spinless model are, read in single-valued tables.  Where a member of the
    sign asked had to be chosen, it is the intertwiner every later step reads.

    **What holds of the rest is read once, against the group returned.**  On
    the group time reversal is measured on every unitary operation holds, and
    a ``g·T`` kept beside its own ``g`` makes time reversal itself,
    ``T = g⁻¹·(g·T)``; but a descent after it can still leave ``g`` out --
    the projective class of spinor states read spinless lowers the unitary
    group once time reversal is measured, and the flattened ``(p_x, p_y)``
    doublet of ``P4/mmm1'`` returns ``P4/m``, beside which the eight
    operations of ``4/mmm`` outside ``4/m`` combined with time reversal make
    ``P4/mm'm'``.  So this round's verdict on every operation combined with
    time reversal is kept (:func:`_antiunitary_settled`) beside each earlier
    descent's on the ones it left behind -- Haldane's six mirrors combined
    with time reversal, which the group time reversal is measured on does not
    hold -- and all of them are read once the group has settled
    (:func:`_beyond_grey`): the record's ``n_obeyed``, ``n_broken`` and
    ``note`` are written there, by the rule :attr:`RealizedGroup.beyond_grey`
    names its operations by, and an operation whose unitary part is in the
    group returned never counts, making time reversal with it.  Where time
    reversal is obeyed, nothing of them is said.

    ``refuted`` is the verdict stage's own refutations on this group: an
    operation combined with time reversal its own form or its tensor structure
    refutes is not one of the Hamiltonian's, and time reversal itself refuted
    there is broken as one of the wrong square is -- by its spin lift, no
    member of its solution space acting as ``iσ_y`` on every spin pair, as on
    spin pairs declared across two Wannier functions of one site, or by its
    site permutation, no member carrying each site's states onto its own, as
    in an antiferromagnet whose moments alternate between sites the cell's
    point set does not tell apart.

    Returns ``(record, asked, validated)``: the narration that always rides
    on the returned group, whose count and note :func:`_beyond_grey` writes
    where time reversal is broken; what this round measured of the operations
    combined with time reversal (:func:`_antiunitary_settled`), ``None``
    where time reversal is obeyed; and the detector's record with time
    reversal's member of the sign asked in place.  ``tolerances`` is the record
    of what the call declared, which every reading of a solution space here
    judges noise against.
    """
    antiunitary = {int(s.ind): s for s in group.symmetries if s.time_reversal}
    broken = {i for i in validated.broken if i in antiunitary}
    square = (_time_reversal_square(group, validated, tolerances=tolerances)
              if antiunitary else None)
    if square is not None and not square["obeyed"]:
        broken.add(int(square["ind"]))
    # ... and time reversal whose own tensor structure the verdict stage refuted:
    # no member of its solution space carries its lift, i sigma_y on every spin
    # pair, as on spin pairs declared across two Wannier functions of one site
    lift = (None if square is None
            else (refuted or {}).get(int(square["ind"])))
    lifted = (isinstance(lift, dict)
              and str(lift.get("refuted_by", "tensor structure")) == "tensor structure")
    if lifted:
        broken.add(int(square["ind"]))
    validated = _time_reversal_member(validated, square)
    obeyed = sorted(set(antiunitary) - broken)
    undetermined = tuple(i for i in obeyed
                         if validated.verdicts.get(i) == "undetermined")

    record = {
        "n_antiunitary": len(antiunitary),
        "n_obeyed": len(obeyed), "n_broken": len(broken),
        "undetermined": undetermined, "grey_number_str": str(group.number_str),
        "grey_name": str(group.name),
        "square": (None if square is None else
                   {"asked": int(square["asked"]), "measured": float(square["measured"])}),
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
    elif square is not None and not square["obeyed"]:
        tables = "single-valued" if square["asked"] > 0 else "double-valued"
        measured = float(square["measured"])
        if np.isfinite(measured):
            kind = ("a spinor's time reversal" if measured < 0
                    else "a spin-free time reversal")
            said = (f"with X its intertwiner the square X conj(X) measures "
                    f"{measured:+.6g}, {kind}, where the {tables} tables read here ask "
                    f"for {square['asked']:+d} -- a rescaling of X changes the square only "
                    f"by a positive factor, so the sign is the Hamiltonian's own, and no "
                    f"member of its solution space carries the other.")
        else:
            said = (f"no member X of its solution space squares, X conj(X), to the "
                    f"{square['asked']:+d} the {tables} tables read here ask for.")
        itself = (
            f" Time reversal itself is broken by its square: {said}"
            + (" A spinless model carrying spinors in its orbital index, as a wannier90 "
               "import does, reads this way; `spin=` declares which orbitals are spin "
               "partners, and the double-valued tables are then read."
               if square["asked"] > 0 else ""))
    elif lifted:
        itself = (" Time reversal itself is refuted by its spin lift: no member of its "
                  "solution space acts as i sigma_y on every spin pair, as time reversal "
                  "acts on the two states of a spinor. A declared spinor basis whose "
                  "pairs join the states of different Wannier functions reads this way.")
    # WHAT THIS ROUND ASKED of every operation combined with time reversal: the
    # last group to ask any of them, a descent after it -- the projective class
    # of spinor states read spinless -- still able to leave a unitary part
    # out.  Its count and its note are written where they are read, once,
    # against the group returned (`_beyond_grey`).
    asked = dict(_antiunitary_settled(group, validated, refuted, square,
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
    r"""What each round measured of the operations combined with time reversal, carried onto the group that settled.

    Every round's group holds the group the loop settles on: each step keeps
    a subset of the operations the round before it held, written in the
    origin they were measured in (:func:`_shared_origin`).  So each operation
    of the group returned is paired with the one it is in each round's group
    (:func:`_match_operations`), and each record of ``asked``
    (:func:`_antiunitary_settled`) is handed back with the pairing as
    ``returned``, ``{index there: index in the group returned}``: what the
    one reading of those operations reads them against.

    Raises
    ------
    ..refusal.InternalError
        Where an operation of the group returned is not in a round's group.
        Every step keeps a subset of what it measured, so this is the
        package's own bookkeeping failing, not a measurement.
    """
    unitary = [s for s in group.symmetries if not s.time_reversal]
    out = []
    for rec in asked:
        theirs = [s for s in rec["group"].symmetries if not s.time_reversal]
        pairs, lost, _extra = _match_operations(unitary, theirs, symprec, group.lattice)
        if lost:
            raise InternalError(
                f"the group returned, {group.name} #{group.number_str}, holds "
                f"{len(lost)} operation{'' if len(lost) == 1 else 's'} {lost} that "
                f"{rec['group'].name} #{rec['group'].number_str}, the group of an earlier "
                "round, does not. Every round keeps a subset of the operations the round "
                "before it measured, so this is this package's own bookkeeping failing, "
                "not a measurement about the model.")
        out.append(dict(rec, returned={int(t): int(g) for g, t in pairs}))
    return tuple(out)


def _antiunitary_held(group, matrices, asked, *, tolerances=EXACT) -> dict:
    r"""Which operations of the cell combined with time reversal hold, read once against the group returned.

    ``asked`` is what the rounds measured of the operations combined with time
    reversal, each in the round that asked it last
    (:func:`_antiunitary_settled`), carried onto the group returned
    (:func:`_carried_onto`); ``matrices`` is the representation of the group
    returned that :func:`.._build._build_rep` assembled, by ``.ind``, or
    ``None`` where none was built.  It is read where time reversal is broken.

    The rule is one question: whether the matrices measured make a
    representation of the magnetic group ``M`` an operation ``a`` combined
    with time reversal generates with the group returned ``G``.  It has three
    parts:

    * **the group it makes.**  Every operation of ``M`` combined with time
      reversal is kept by the detector, by its own tensor structure and spin
      lift, and by its own square where its unitary part squares to the
      identity, and ``M`` holds no unitary operation outside ``G``.  One whose
      unitary part is in ``G`` makes time reversal itself with it,
      ``T = g⁻¹·(g·T)``, and time reversal is refuted, so it never counts: on
      spinors read in single-valued tables a mirror that holds and that mirror
      with time reversal, squaring to ``+1``, together make time reversal,
      whose square is ``-1``.
    * **its square.**  ``X X̄ = D(a²)`` for its matrix ``X`` and ``D`` the
      matrices of ``G``, with the sign the group's multiplication table gives
      ``a²``: for an ``a`` whose unitary part squares to the identity, the
      sign of ``X X̄``, which no rescaling of ``X`` changes.
    * **how it carries the group returned.**  ``X D(h)* X⁻¹ = D(a h a⁻¹)``
      for each generator ``h`` of ``G``, with the signs the table gives.

    The last two are asked up to a one-dimensional character ``χ`` of ``G``,
    the one freedom the matrices of ``G`` are assembled with: one where the
    phase ``λ(h)`` measured on each generator is ``χ(h) χ(a h a⁻¹)`` and the
    phase of the square is ``χ(a²)`` makes them hold exactly, and these are
    Wigner's conditions for ``X`` and ``D`` to be a representation of ``M``.
    Each phase is read at the margin every phase read off the states is read
    at (:data:`_CLASS_MARGIN`).  Nothing is measured here.

    ``X`` is any member of the operation's measured solution space.  Where
    the space holds one, it is the matrix measured; where it holds more -- a
    Hamiltonian that conserves a spin rotation, whose every member times a
    conserved rotation is a member too -- the question is whether some member
    makes the representation, as time reversal's own square asks whether
    some member has its sign (:func:`_time_reversal_square`).  Once the
    character is fixed the third condition is linear in ``X``, so the members
    that carry ``G`` are read off the space, and the square is asked of each,
    and of the unitary nearest the part of it its transposition keeps with
    the sign the table gives ``a²`` where ``a²`` is the identity, wherever
    that is a member too.

    Each operation is asked on its own, and those counted are the operations
    of one magnetic group combined with time reversal: where they are not
    exactly the ones of one ``M``, no single magnetic group holds them with
    ``G``, and none counts.

    Returns ``{"verdicts", "held"}``: per operation, keyed ``(round, ind)``,
    what decided it -- ``"returned"`` where its unitary part is in the group
    returned, ``"refuted"`` in its own round, ``"closure"`` where the group it
    makes holds a refuted operation, ``"unread"`` where no matrices were built
    to read it with, ``"matrices"`` where they make no representation,
    ``"apart"`` where those that hold are not the operations of one magnetic
    group, or ``"held"`` -- and the ``(round, ind)`` that hold, in order.
    """

    # the members of a solution space read off a model symmetric only within
    # its precision: how small a direction of it is none, and how far a unitary
    # read off it may sit outside it
    zero, in_span = tolerances.noise(ZERO), tolerances.noise(IN_SPAN)
    own = group.symmetries
    by_ind = {int(s.ind): s for s in own}
    found = _minimal_generators(range(len(own)), group.get_product_table(get_diff=False),
                                _identity_position(group))
    # the relation asked of every operation, where no small generating set is
    # found, is the one asked of the generators and more
    gens = [int(own[p].ind) for p in (found if found is not None else range(len(own)))]
    characters = None
    verdicts, held, cosets = {}, [], {}
    for r, rec in enumerate(asked):
        cell = rec["group"]
        syms = cell.symmetries
        at = {int(s.ind): p for p, s in enumerate(syms)}
        prod, _tdiff, spf = cell.get_product_table(get_diff=True)
        ident = _identity_position(cell)
        onto = rec["returned"]
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
                        f"pair {i} in {cell.name} #{cell.number_str} carry spin lifts that "
                        "are not one SU(2) element up to a sign, though they are one "
                        "rotation: this is this package's bookkeeping failing, not a "
                        "measurement about the model.")
            lift[q] = sign

        def _matrix(q):
            """The matrix of the group returned at position ``q`` here, in this group's lift."""
            return lift[q] * np.asarray(matrices[int(onto[int(syms[q].ind)])], dtype=complex)

        def _carried(p, back, X):
            """Whether ``X`` and the group returned's matrices are a representation of the group they make."""
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
            """The members of a space of more than one that carry the group returned, as a character asks.

            For each character the condition ``X D(h)* = chi(h) chi(c) D(c) X``,
            ``c = a h a^-1`` with the table's sign, on every generator is linear
            in ``X``: its solutions in the space are read off one SVD, offered
            with a fixed combination of them, and each, where ``a²`` is the
            identity, also as the unitary nearest the part of it its
            transposition keeps with the sign the table gives ``a²``, wherever
            that is still a member of the space.
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

        for ind in sorted(rec["ops"]):
            key = (r, int(ind))
            p = at[int(ind)]
            if unitary.get(_operation_key(syms[p])[:2]) in inside:
                verdicts[key] = "returned"
                continue
            if ind in rec["refuted"]:
                verdicts[key] = "refuted"
                continue
            # THE GROUP IT MAKES with the group returned G: G and the coset aG,
            # where a squares into G and carries G onto itself -- otherwise the
            # group they generate holds a unitary operation outside G -- and every
            # operation of aG kept in this round, the last that asked it
            back = next(q for q in range(len(syms)) if int(prod[p, q]) == ident)
            coset = {int(prod[p, q]) for q in inside}
            if (int(prod[p, p]) not in inside
                    or any(int(prod[int(prod[p, here[g]]), back]) not in inside
                           for g in gens)
                    or any(int(syms[q].ind) in rec["refuted"]
                           or int(syms[q].ind) not in rec["ops"] for q in coset)):
                verdicts[key] = "closure"
                continue
            if matrices is None or any(int(g) not in matrices for g in onto.values()):
                verdicts[key] = "unread"
                continue
            if characters is None:
                characters = _characters_of_subgroup(group, sorted(by_ind), roots=True)
            op = rec["ops"][ind]
            if not (_carried(p, back, op["member"])
                    or any(_carried(p, back, X)
                           for X in _carrying(p, back, op.get("space")))):
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
    return {"verdicts": verdicts, "held": tuple(held)}


def _beyond_grey(group, matrices, asked, record, *, tolerances=EXACT) -> tuple:
    r"""The operations combined with time reversal that hold while it is broken, read once.

    Where time reversal is measured broken, every operation of the cell
    combined with time reversal that a round measured -- each descent the ones
    whose unitary part it left out, the group time reversal is measured on
    all of its own (:func:`_antiunitary_settled`) -- is read here, once the
    group has settled, against the group returned and the matrices
    :func:`.._build._build_rep` assembled for it (``matrices``, ``None``
    where none were built), by the one rule :func:`_antiunitary_held` states.
    Where time reversal is kept nothing is read and nothing is named.

    The time-reversal ``record`` is completed from it: ``n_obeyed`` and
    ``n_broken`` count the operations of the group time reversal was measured
    on by that rule, and the ``note`` says what decided them.  The ones that
    hold are what :attr:`RealizedGroup.beyond_grey` names, each by its index
    in the group it was measured in, with the magnetic type they make.

    ``tolerances`` is the record of what the call declared, which the reading of
    the solution spaces judges noise against.

    Returns ``(record, beyond)``: the record, completed, and the beyond-grey
    record, ``None`` where nothing holds.
    """
    if record is None or record.get("outcome") != "measured broken":
        return record, None
    read = _antiunitary_held(group, matrices, asked, tolerances=tolerances)
    verdicts, held = read["verdicts"], read["held"]
    r_t = next(r for r, rec in enumerate(asked) if rec["reversal"])
    reversal = asked[r_t]
    # the count is of the group time reversal was measured on, and the note
    # says what decided the operations of it the detector kept
    mine = [v for (r, i), v in verdicts.items()
            if r == r_t and i not in reversal["detector"]]
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
        f"own.{reversal['itself']}"))
    if not held:
        return record, None

    rec = asked[held[0][0]]
    inds = [i for _r, i in held]
    entries = tuple(rec["ops"][i]["entry"] for i in inds)
    kind = _magnetic_type(_CosetView([rec["ops"][i]["sym"] for i in inds]))
    n = len(entries)
    beyond = {
        "operations": entries,
        "n_operations": n,
        "n_measured": sum(int(r["n"]) for r in asked),
        "magnetic_type": kind if kind in ("III", "IV") else "magnetic, beyond grey",
        "closed": bool(rec["closed"]),
        "note": (
            f"{n} operation{'' if n == 1 else 's'} of {rec['group'].name} "
            f"#{rec['group'].number_str} combined with time reversal, "
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
    r"""The precision a model's numbers would have to be declared at for every operation a descent broke to hold.

    An operation broken by its own form beyond the covariance gate -- the
    member of its solution space nearest that form, its part in the form,
    carries ``H(k)`` onto ``H(gk)`` only to the entry's ``covariance`` -- holds
    under a declared precision ``p`` whose gate, ``10 · p · max|H|``, reaches
    that far, as spin blocks written in a frame turned against the lattice by
    an angle within the model's own precision do.  ``scale_of`` measures
    ``max|H|``, and is asked only where every operation broken was broken so.

    A precision is named only where the detector can read the model at it: its
    matching window, ``clip(10·sqrt(10·p), 1e-3, 0.3)``
    (:func:`.._tolerance.eigen_window`), below its ceiling.  A defect that
    needs more is no model's numerical precision -- spin blocks written along
    another axis, or turned by a sizeable angle, carry ``H`` that far -- and
    none is named for it.

    Returns ``(worst, p)``: the largest such covariance, in the units of ``H``,
    and the least precision whose gate holds it, rounded up to two figures;
    ``None`` where an operation was broken by anything else, where nothing
    was, or where that precision is past the window's ceiling.
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
#: reduced coordinates are deliberately unlike any Wyckoff coordinate — off
#: every axis, plane and diagonal — so that its **stabilizer is trivial**: an
#: operation preserving the orbit of such a point belongs to the set the orbit
#: was generated from, which is what makes the construction exact rather than
#: approximate.  The third entry is used only along a genuinely periodic
#: direction; a vacuum one is placed by the two numbers below.
_DESCENT_POINT = (0.1234, 0.2718, 0.0577)

#: Along a **vacuum** direction the generic point sits at this fraction of the
#: cell's own half-extent, measured from the cell's own mid-plane.  Staying
#: *inside* the extent is what keeps :func:`_pad_cell`'s re-centring
#: bit-identical to the parent cell's -- its vacuum is the model's own whatever
#: the cell holds -- so the descended group is read in the same frame the parent
#: was.
_DESCENT_PAD_FRACTION = 0.1234

#: ... and where the cell has **no** extent along a vacuum direction (every
#: point in one plane, which is what a free-standing flat layer leaves), this
#: reduced offset is used instead: a point in the plane would leave ``σ_h``
#: unbroken by construction and the descent could then never remove it.
_DESCENT_PAD_OFFSET = 0.0577

#: Species label of the added orbit — one ``cell_typat`` value, distinct from
#: every orbital and decoration species (a suffix is appended if it collides),
#: so the orbit is one species and its own orbit under the kept set.
_DESCENT_SPECIES = "descent"

#: The generic point **step 1** carries its orbit from, and its species.  A cell
#: the declared content has already reduced is the base every later descent
#: builds on, so the two orbits share a cell and must not share a point: these
#: periodic coordinates are as unlike :data:`_DESCENT_POINT`'s as they are
#: unlike any Wyckoff coordinate, and no point of one orbit is a point of the
#: other.  The coordinate along the vacuum is placed by the same two numbers
#: above, and the vacuum is the model's own (:func:`_pad_cell`), so adding
#: either orbit
#: leaves the lattice, its vacuum included, where it was.
_DECLARED_POINT = (0.3141, 0.1592, 0.0919)

_DECLARED_SPECIES = "declared"

#: The generic points a descent, and a step 1 reduction, build their orbit
#: from, in order.  The first of each is the point above.  A generic point's
#: stabilizer is trivial, so the cell its orbit decorates carries exactly the
#: operations the orbit was generated from -- up to an accident of the
#: identification, a cell still read as non-primitive or with operations other
#: than the kept set, which the next point of the sequence does not share.  The
#: coordinates of all six are distinct from one another and from every Wyckoff
#: coordinate, so no orbit of one sequence meets an orbit of the other.
_DESCENT_POINTS = (_DESCENT_POINT, (0.3719, 0.0857, 0.2461), (0.2087, 0.4421, 0.1693))

_DECLARED_POINTS = (_DECLARED_POINT, (0.0671, 0.3853, 0.2297), (0.4462, 0.2239, 0.0383))


def _descent_frame(cell, group) -> tuple:
    r"""How the cell's own reduced coordinates map onto those of the group's 3D cell.

    :func:`_group_of` does not read a cell's positions directly: :func:`_pad_cell`
    first *scales* every non-periodic lattice vector until the vacuum reaches
    ``vacuum_length`` and then re-centres the positions along it on the origin, and
    the operations that come back are written in **that** frame.  A point added
    to the cell therefore has to be placed in the cell's frame while its orbit is
    computed in the group's, and this measures the one map between them:
    ``y = x / scale − offset`` (mod 1), with ``scale`` read off the two lattices
    and ``offset`` off the first orbital.  Both are exactly ``1`` and ``0`` on a
    periodic direction, and on a fully periodic cell the map is the identity.

    Returns ``(scale, offset, positions, vacuum_dirs)`` — the last two as
    :func:`_pad_cell` returns them, so the cell's own extent along a vacuum
    direction can be read off the same numbers.
    """
    lat_pad, pos_pad, padded = _pad_cell(cell, group.vacuum_length)
    lat_cell = np.asarray(cell.lat_vecs, dtype=float)
    scale = np.array([float(np.linalg.norm(lat_pad[d])
                            / max(np.linalg.norm(lat_cell[d]), 1e-30))
                      for d in range(3)])
    offset = np.asarray(cell.orb_vecs, dtype=float)[0] / scale - pos_pad[0]
    return scale, offset - np.rint(offset), pos_pad, tuple(padded)


def _generic_orbit(cell, group, operations, seed=_DESCENT_POINT) -> tuple:
    r"""The orbit of one generic point under ``operations``, in both frames.

    The point is ``seed`` along every periodic direction — :data:`_DESCENT_POINT`
    for a descent and :data:`_DECLARED_POINT` for step 1's cell reduction, two
    points that never coincide, since a reduced cell is the base a later descent
    adds its own orbit to — and sits off the cell's own mid-plane along a vacuum
    one (see :data:`_DESCENT_PAD_FRACTION`).  Its stabilizer in the cell's group
    is trivial, so the point set ``sites ∪ orbit`` is invariant under exactly the
    operations the orbit was generated from — plus, in principle, an accident,
    which is why the identification is **verified** rather than trusted, and
    an accident rebuilt from the next point of a fixed sequence
    (:func:`_orbit_cell`).

    Returns ``(point, orbit_padded, orbit_cell)``: the generic point and the
    orbit in the group's 3D cell, and the same orbit in the cell's own reduced
    coordinates, ready for :func:`_decorated_cell`.  Components along the
    vacuum are carried on the ``(−½, ½]`` branch, which is the branch the
    cell's own
    positions are written in and the one :func:`_pad_cell` measures its extent
    from.
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
    r"""On a flat layer, each operation composed with the horizontal mirror, and its partner.

    Where every orbital of a layer lies in one plane, the horizontal mirror
    ``σ_h`` moves no orbital and no in-plane ``k``, so an operation ``g`` and
    its product ``σ_h·g`` act alike on everything a Hamiltonian without spin
    has: they share an action on ``k`` (:attr:`ValidatedGroup.k_action_classes`)
    and a site permutation, one solution space holds the matrices of both, and
    the hoppings cannot tell them apart.  What does is the sign each orbital
    carries under ``σ_h`` (:func:`_mirror_sign`).

    Returns ``{ind of σ_h·g: ind of g}`` -- ``g`` keeping the layer's side and
    ``σ_h·g`` turning it over, either of them combined with time reversal where
    ``g`` is -- for every class of operations the detector left undetermined,
    the identity and the mirror itself among them; or ``None`` where one of
    those classes has another shape, two members that move the orbitals
    differently among them, which the orbitals' positions tell apart.  Empty
    where nothing is undetermined.
    """
    undetermined = {int(i) for i in validated.undetermined}
    if not undetermined:
        return {}
    padded = tuple(int(d) for d in getattr(group, "vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return None
    sides = {int(s.ind): _side_preserving(s, padded[0]) for s in group.symmetries}
    perm = np.asarray(assignment.permutations, dtype=int)
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    partners = {}
    for members in validated.k_action_classes:
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
    r"""The sign the horizontal mirror carries on a flat layer read without spin, chosen.

    The hoppings fix how a flat layer's orbitals transform under ``σ_h``
    relative to one another and not the sign itself: ``σ_h`` moves nothing, so
    its matrix is one commuting with ``H(k)`` at every ``k``, and on each
    part of the crystal the hoppings couple that is a multiple of the identity
    -- one sign per coupled part, ``D(σ_h)`` and ``−D(σ_h)`` alike.  The sign
    is therefore chosen: the parity the declared content states where it
    states one (``s`` even, ``p_z`` odd), and otherwise the even one the
    lowest-angular-momentum rule assigns, the lowest function of any in-plane
    content lying in the plane.  Fixing it fixes every operation composed with
    the mirror from its partner, ``D(σ_h·g) = D(σ_h) D(g)``, the inversion
    ``i = σ_h·C2(z)`` among them, and it is never refused: a coupled part
    declared two parities is a crystal without the mirror, which the
    declaration's own verdict lowers before this is read.  The states'
    projective class is read on one coupled part, so one sign is returned.

    Returns ``{"sign", "determined_by"}``: ``+1`` or ``-1``, and
    ``"declaration"`` or ``"lowest-l"``.
    """

    identity = int(group.symmetries[_identity_position(group)].ind)
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
    r"""One matrix per operation of a spinless reading, where the class can be read off them.

    The class is read off the detector's own members, and only where a member
    is fixed up to a phase: a unitary, single-valued group whose every
    operation stands, with nothing but the identity commuting with ``H``, each
    operation alone in its action on ``k`` -- or, on a flat layer, sharing it
    with its product with the horizontal mirror (``partners``,
    :func:`_mirror_partners`), which is given the mirror's chosen ``sign``
    times its partner's matrix, ``D(σ_h·g) = D(σ_h) D(g)``.

    Returns ``(D, positions)`` over every operation of the group, or ``None``
    where the class is not read here.
    """

    syms = group.symmetries
    if bool(group.spinor) or any(bool(s.time_reversal) for s in syms):
        return None
    inds = [int(s.ind) for s in syms]
    if set(inds) - {int(i) for i in validated.indices}:
        return None
    partners = dict(partners or {})
    if {int(i) for i in validated.undetermined} - set(partners) - set(partners.values()):
        return None
    ident = _identity_position(group)
    if int((validated._nullspace_dims or {}).get(int(syms[ident].ind), 0) or 0) != 1:
        return None
    members = validated._intertwiners or {}
    D = {}
    for p, ind in enumerate(inds):
        member = members.get(partners.get(ind, ind))
        if member is None:
            return None
        D[p] = (int(sign) if ind in partners else 1) * np.asarray(member, dtype=complex)
    return D, list(range(len(syms)))


def _class_kept(group, D, positions):
    r"""The largest subgroup on which matrices ``D`` are an ordinary representation, by the tie rule.

    Two witnesses, neither of which a choice of phases moves: the ratio of two
    commuting operations' matrices taken in the two orders
    (:func:`.._matrices._commuting_pair_phases`), which names the pairs, and the
    relations a rephasing has to close on a subgroup
    (:func:`.._matrices._rephasable`), which decides it.  The subgroups holding
    no such pair are enumerated (:func:`_subgroups_inside`), the largest order
    at which one of them is rephasable is the order kept, and where several of
    that order are, the one kept is the first by the tie rule
    (:func:`_tie_ranked`), never their intersection.

    Returns ``(pairs, kept, ranked)`` -- the pairs ``(a, b, phase)`` by
    position, the ``.ind`` kept and the subgroups of the kept order in the
    rule's order -- or ``None`` where no pair of commuting operations carries
    a phase the table lacks.
    """

    syms = group.symmetries
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
    pairs = _commuting_pair_phases(D, positions, syms, prod, spf, tol=_CLASS_MARGIN)
    if not pairs:
        return None
    at = {int(s.ind): p for p, s in enumerate(syms)}
    subs = _subgroups_inside(group, [int(syms[p].ind) for p in positions], avoid={
        frozenset((int(syms[a].ind), int(syms[b].ind))) for a, b, _c in pairs})
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

    ``pairs`` are :func:`_class_kept`'s, by position in ``group``; ``kept`` the
    ``.ind`` kept and ``ranked`` the tied subgroups, the one kept first.
    ``rule_group`` is the group the tie was settled in, where that is not
    ``group``, and ``rule_ranked`` the tied subgroups there, so that the rule's
    words, and the reason its first candidate was kept, are the ones it was
    read with.
    """

    syms = group.symmetries
    _prod, _tdiff, spf = group.get_product_table(get_diff=True)
    at = {int(s.ind): p for p, s in enumerate(syms)}
    lattice = np.asarray(group.lattice, dtype=float)

    def _name(i):
        return _record_name(syms[at[int(i)]], lattice)

    witnesses = []
    for a, b, c in pairs:
        Sa = np.asarray(syms[a].spinor_rotation, dtype=complex)
        Sb = np.asarray(syms[b].spinor_rotation, dtype=complex)
        half, _res = _proportionality(Sa @ Sb, int(spf[a, b]) * int(spf[b, a]) * (Sb @ Sa))
        witnesses.append({"pair": (int(syms[a].ind), int(syms[b].ind)),
                          "names": (_name(syms[a].ind), _name(syms[b].ind)),
                          "commutation_phase": complex(c),
                          "spin_half": bool(abs(complex(half) - complex(c)) <= _CLASS_MARGIN)})

    refuted = {}
    for i in sorted(set(at) - set(kept)):
        mine = [w for w in witnesses if i in w["pair"]]
        best = next((w for w in mine if sum(1 for j in w["pair"] if j in kept) == 1),
                    mine[0] if mine else None)
        if best is None:
            refuted[i] = {"refuted_by": "projective class", "pair": None, "names": None,
                          "commutation_phase": None, "spin_half": None, "note": (
                              f"no subgroup holding {_name(i)} together with the operations "
                              "kept carries a single-valued representation on these states")}
            continue
        a, b = best["names"]
        refuted[i] = dict(best, refuted_by="projective class", note=(
            f"{a} and {b} commute, and on these states their matrices do not: "
            f"D({a}) D({b}) = {_fmt_char(best['commutation_phase'])} "
            f"D({b}) D({a}), a phase no rescaling of either matrix removes"
            + (" -- the sign the spin-1/2 lifts of the two operations carry"
               if best["spin_half"] else "")
            + f". No single-valued representation of a group holding both exists on "
            f"these states, so {_name(i)} is not an operation of the spinless crystal: "
            "the largest subgroup on which the states carry one is kept, and it does not "
            f"hold {_name(i)}."))
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


def _projective_class_refuted(group, validated, *, assignment=None, decls=()):
    r"""The operations of a spinless reading that no single-valued representation carries.

    A spinless reading reads the states in single-valued tables, and the matrices
    the Hamiltonian's operations are on them form a projective representation.
    Where its class is not trivial -- spin-½ states in the orbital index of a
    spinless model are the case, where two perpendicular twofolds commute and
    their matrices anticommute -- no single-valued representation of the group
    exists on the states, and one of every such pair is not an operation of the
    spinless crystal.  So the spinless group is the **largest subgroup on which
    the states carry an ordinary representation**, and what is refuted here is
    the rest.

    The class is read off the detector's own members, and only where a member is
    fixed up to a phase (:func:`_class_members`): a unitary, single-valued group
    whose every operation stands, with nothing but the identity commuting with
    ``H``, each operation alone in its action on ``k`` -- or, on a flat layer,
    sharing it with its product with the horizontal mirror, whose matrix is
    then the mirror's chosen sign times its partner's (:func:`_mirror_partners`,
    :func:`_mirror_sign`): ``D(σ_h·g) = D(σ_h) D(g)``, so that every operation
    of the free-standing layer has its matrix and the class is read there as
    anywhere else.  The sign commutes with every matrix, and the class does not
    depend on it.  Two witnesses, neither of which a choice of phases moves,
    decide it (:func:`_class_kept`).

    Where several subgroups of the kept order carry one, one is chosen by the
    tie rule every such choice is made by (:func:`_tie_ranked`), never their
    intersection: on a layer the most operations that keep its side first;
    then the most rotations, so that a subgroup keeping rotations stands
    before one keeping mirrors; then the highest-order rotation; then its axis
    in the conventional cell -- ``c``, then ``a``, then ``b``, then the
    diagonals -- and last every other operation's axis the same way.

    ``assignment`` is the orbitals' orbits in ``group``, which say whether an
    operation and its product with the mirror move the orbitals alike, and
    ``decls`` the declared contents the round read, which say the mirror's
    parity where they state one.

    Returns ``(refuted, record)``: per operation left out, keyed by ``.ind``,
    ``refuted_by = "projective class"`` with the ``pair`` of operations that
    names it -- a commuting partner whose matrix does not commute with its own,
    one that is kept where there is one -- their ``names``, the
    ``commutation_phase``, whether that phase is the sign the two operations'
    spin-½ lifts carry (``spin_half``), and a ``note``; and the step's own
    record, every such pair and the subgroups of the kept order in the rule's
    order, the first of them kept, with the rule's keys as this group reads
    them (``rule``) and why it kept the first (``reason``) where several tie,
    and on a flat layer the mirror's sign and how it was determined
    (``horizontal_mirror``).  ``({}, None)`` where the class is the table's, or
    is not read here.
    """
    if bool(group.spinor) or any(bool(s.time_reversal) for s in group.symmetries):
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
    got = _class_kept(group, *members)
    if got is None:
        return {}, None
    refuted, record = _class_record(group, *got)
    if mirror is not None:
        record["horizontal_mirror"] = dict(mirror)
    return refuted, record


def _free_standing_lift(model, group, free_standing):
    r"""A substrate group with the horizontal mirror adjoined: the free-standing layer's group.

    ``free_standing`` is the free-standing cell of the same point set
    (:func:`.._realized._layer_realized`), whose group is identified the first time it is
    asked for and kept on the record.  On a flat layer that group holds the
    mirror ``σ_h``, which moves no orbital, and every operation of ``group``
    -- each keeps the layer's side -- matched by rotation and by translation
    modulo the lattice (:func:`_match_operations`); the free-standing group of
    ``group`` is ``group``
    with ``σ_h`` adjoined, the whole cell group where ``group`` is the whole
    substrate cell's and otherwise a cell carrying exactly those operations,
    identified as a descent's is (:func:`_orbit_cell`), so that the tie rule
    reads it in its own conventional cell, as the free-standing layer's own
    measurement does.

    Returns ``(free, of)`` -- the free-standing group, and per unitary
    operation of it ``(ind of g in group, whether it is σ_h·g)`` -- or
    ``None`` where the layer is not flat: the mirror is not an operation of
    that cell, or it moves an orbital.

    Raises
    ------
    ..refusal.InternalError
        Where an operation of ``group`` is not one of the free-standing cell's:
        a substrate only removes operations, so this is the package
        disagreeing with itself.
    """
    parent = free_standing.get("group")
    if parent is None:
        parent = _group_of(free_standing["cell"], typat=free_standing["cell_typat"],
                           spinor=False, include_TR=False, symprec=free_standing["symprec"])
        free_standing["group"] = parent
    if isinstance(parent, Refusal):
        return None
    padded = tuple(int(d) for d in getattr(parent, "vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return None
    assignment = _assign_orbitals(model, parent, typat=free_standing["labels"])
    if isinstance(assignment, Refusal):
        return None
    reflect = np.eye(3, dtype=int)
    reflect[padded[0], padded[0]] = -1
    psyms = parent.symmetries
    perm = np.asarray(assignment.permutations, dtype=int)
    still = np.arange(perm.shape[1])
    mirror = next((p for p, s in enumerate(psyms) if not s.time_reversal
                   and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int), reflect)
                   and np.array_equal(perm[p], still)), None)
    if mirror is None:
        return None
    pairs, missing, _extra = _match_operations(
        [s for s in group.symmetries if not s.time_reversal],
        [s for s in psyms if not s.time_reversal], free_standing["symprec"],
        group.lattice)
    if missing:
        by_ind = {int(s.ind): s for s in group.symmetries}
        lattice = np.asarray(group.lattice, dtype=float)
        raise InternalError(
            f"{len(missing)} operations of {group.name} #{group.number_str}, read on a "
            "substrate, are not operations of the free-standing cell of the same point "
            f"set ({parent.name} #{parent.number_str}): "
            + ", ".join(_record_name(by_ind[int(i)], lattice) for i in missing)
            + ". A substrate only removes operations, so this is the package disagreeing "
            "with itself, not a measurement about the model.")
    at = {int(s.ind): p for p, s in enumerate(psyms)}
    prod = parent.get_product_table(get_diff=False)
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
        """What the identification of the free-standing cell was read as, and why that is a defect."""
        return (
            f"the free-standing cell built to carry {group.name} #{group.number_str} with the "
            f"horizontal mirror adjoined ({len(kept)} operations of {parent.name} "
            f"#{parent.number_str}) was identified as {got.name} #{got.number_str}, whose "
            f"{len(unitary)} unitary operations are not that set ({len(unmatched_kept)} "
            f"missing, {len(unmatched_found)} extra), so this is the package's "
            "identification disagreeing with its own construction.")

    made = _orbit_cell(free_standing["cell"], free_standing["cell_typat"], parent, kept,
                       points=_DESCENT_POINTS, species=species, spinor=False,
                       include_TR=False, symprec=free_standing["symprec"],
                       diagnose=_diagnose)
    if isinstance(made, Refusal):
        return None
    free, _cell, _typat, pairs, _point, _orbit = made
    of_ind = {int(psyms[p].ind): v for p, v in of_parent.items()}
    return free, {int(f): of_ind[int(k)] for k, f in pairs}


def _projective_class_carried(model, group, validated, *, free_standing):
    r"""A flat layer's projective class on a substrate: the free-standing layer's reading, carried.

    On a flat layer read without spin the free-standing group is this one with
    the horizontal mirror ``σ_h`` adjoined, and its matrices are these times the
    mirror's sign, every operation composed with the mirror given
    ``D(σ_h) D(g)`` (:func:`_mirror_sign`): nothing a substrate measures is
    missing from it, and nothing is measured again.  So the class is read there
    once (:func:`_free_standing_lift`, :func:`_class_kept`) -- the largest
    subgroup on which the states carry an ordinary representation, by the one
    tie rule in the free-standing layer's own conventional cell -- and the
    substrate carries its part that preserves the layer's normal, as it carries a declaration read
    in the free-standing layer's site groups (:func:`_carried_reduction`):
    every operation the class leaves out of the free-standing layer is left out
    here too.  The two environments then read one class, and the group kept on
    a substrate is the half of the one kept free-standing that preserves the
    layer's normal.  This
    group's own matrices decide only whether there is a class to read -- the
    free-standing pairs are its pairs and their products with the mirror -- and
    name the pairs on the record.  The sign is the lowest-angular-momentum
    rule's: a substrate reads no declared content under a mirror it does not
    have, and the class does not depend on the sign.

    Where the layer is not flat -- the mirror moves an orbital, or is not an
    operation of the free-standing cell -- the orbitals' positions tell an
    operation from its product with the mirror, the free-standing group is not
    this one with the mirror adjoined, and the class is read here
    (:func:`_projective_class_refuted`).

    Returns ``(refuted, record)`` as :func:`_projective_class_refuted` does, the
    pairs those of this group and the record naming the free-standing group the
    class was read on (``read_on``) and how many of its operations were kept
    there.
    """

    members = _class_members(group, validated)
    if members is None:
        return {}, None
    D, positions = members
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
    pairs = _commuting_pair_phases(D, positions, group.symmetries, prod, spf,
                                   tol=_CLASS_MARGIN)
    if not pairs:
        # the free-standing pairs are these and their products with the mirror,
        # which commutes with every matrix: no pair here, none there
        return {}, None
    lift = _free_standing_lift(model, group, free_standing)
    if lift is None:
        return _projective_class_refuted(group, validated)
    free, of = lift
    # each operation composed with the mirror is given the mirror's sign times
    # its partner's matrix, the sign the even one of the lowest-angular-momentum
    # rule: D(sigma_h g) = D(sigma_h) D(g)
    sign = 1
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    fD = {p: (sign if of[int(s.ind)][1] else 1) * D[at[of[int(s.ind)][0]]]
          for p, s in enumerate(free.symmetries) if not s.time_reversal}
    got = _class_kept(free, fD, sorted(fD))
    if got is None:                     # pragma: no cover - the pairs above are its pairs too
        raise InternalError(
            f"the projective class of the states measured in {group.name} "
            f"#{group.number_str} has {len(pairs)} pairs of commuting operations whose "
            "matrices do not commute, and the free-standing group with the horizontal mirror "
            f"adjoined, {free.name} #{free.number_str}, has none: the mirror's matrix "
            "commutes with every other, so this is the package disagreeing with itself.")
    _fpairs, fkept, franked = got

    def _here(subgroup):
        """A subgroup of the free-standing group, by its operations here that preserve the layer's normal."""
        return tuple(sorted(of[int(i)][0] for i in subgroup if not of[int(i)][1]))

    ranked = tuple(_here(c) for c in franked)
    refuted, record = _class_record(group, pairs, frozenset(_here(fkept)), ranked,
                                    rule_group=free, rule_ranked=franked)
    record["read_on"] = {"number_str": str(free.number_str), "name": str(free.name),
                         "size": int(free.size), "kept": len(fkept)}
    return refuted, record


def _match_operations(kept, found, symprec, lattice) -> tuple:
    r"""Pair each kept operation with the identified one that *is* it.

    Two operations are the same when their rotations are the same integer matrix
    and their translations agree **modulo the lattice** -- the sense in which a
    space-group translation is defined -- to the finder's distance ``symprec``,
    their difference read as the length it is on ``lattice`` (rows the cell's
    vectors), in the units the finder's distance is in.  The
    pairing is one-to-one by construction — a matched operation is removed from
    the pool — so an identification that is still too large shows up as an
    unmatched *found* operation and one that has moved as an unmatched *kept*
    one.  Returns ``(pairs, kept_unmatched, found_unmatched)``.
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

    An identification names a group, puts it in the tables' setting and orders
    its operations, and it reads each translation off the cell it is handed.
    The operations kept are the ones every earlier step was measured against,
    written in the origin the centres satisfy.  Where the cell an
    identification reads was built on exact centres -- every cell under a
    declared precision is, the centres made exact before any is identified
    (:func:`.._realized._exact_centres`, :func:`.._realized._position_pass`) -- the two agree to
    rounding and nothing moves.  Where they differ, it is by one shift of the
    origin, ``t' = t − (1 − A)·δ``: the group the cell allows, named on the
    centres projected onto its own positions, carried onto the ones the
    centres were moved to where it does not carry them, or a cell read with
    centres no declaration made exact.  The identified group is carried onto
    the kept operations by that one shift (:func:`_moved_origin`): every
    translation moved by ``(1 − A)·δ`` and the origin of the tables' setting
    (``shiftUC``) by ``δ``, which leaves every operation's tabulated
    translation where it was.

    ``δ`` is read off the pairs.  Two copies of one group, each closing
    exactly, differ by translations ``d_g = t_kept(g) − t_found(g)`` that
    compose as the group does, ``d_{gh} = d_g + A_g·d_h``, and their average
    over the group is the ``δ`` with ``d_g = (1 − A_g)·δ`` for every ``g``.

    ``pairs`` are :func:`_match_operations`' ``(kept, identified)`` pairs of
    ``.ind``, one for every unitary operation of ``found``.  Where every pair's
    translations agree to rounding -- every cell built from exact numbers --
    ``found`` is returned as it is, bit for bit; otherwise a new record is
    returned and the identification's own is left as it was made.

    Raises
    ------
    ..refusal.InternalError
        Where no single shift carries the identified operations onto the kept
        ones.  Two exact copies of one group are always related by one, so
        this is arithmetic in this package failing rather than a measurement.
    """
    by_kept = {int(s.ind): s for s in kept}
    by_found = {int(s.ind): s for s in found.symmetries}
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
            f"{len(diffs)} it was built to carry are not one group about two origins -- "
            f"the best single shift misses by {miss:.3g}. Both close exactly, so this "
            "is arithmetic in this package failing, not a measurement about the model.")
    return _moved_origin(found, delta)


def _moved_origin(group, delta) -> GroupData:
    r"""``group`` written about an origin moved by ``delta``, reduced, in its 3D cell.

    Every translation moves by ``(1 − A)·δ`` and the origin of the tables'
    setting (``shiftUC``) by ``δ``, which leaves every operation's tabulated
    translation where it was: the same group, about another point.  A new
    record is returned and ``group`` is left as it was.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    moved = []
    for s in group.symmetries:
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
    return replace(group, symmetries=list(moved), _shiftUC=shift, _sg=sg)


def _orbit_cell(base_cell, base_typat, group, kept, *, points, species, spinor,
                include_TR, symprec, diagnose):
    r"""The cell that carries exactly ``kept``, identified — from the first point that does.

    The parent cell's own points plus the orbit of one generic point under
    ``kept``, as one extra species (:func:`_generic_orbit`), handed to
    :func:`_group_of`, the one identification that can name the subgroup and
    stamp the ``.ind`` order every table lookup keys on, and then verified
    against ``kept`` operation by operation (:func:`_match_operations`).  A
    generic point's orbit under a group has that group for its stabilizer, so a
    cell read as still non-primitive, or with operations other than ``kept``,
    is an accident of that point and not a statement about the operations: the
    orbit is rebuilt from the next point of ``points``.

    Returns ``(group, cell, cell_typat, pairs, point, orbit)`` — ``orbit`` in the
    identified group's 3D cell — or the :class:`..refusal.Refusal` the
    identification made.  The group carries the operations ``kept``
    themselves, in the origin they were measured in (:func:`_shared_origin`),
    and not the translations the identification reads off the parent cell's
    own points: those are the positions as handed in, and where a declared
    precision has moved the centres it is ``kept`` that they satisfy -- the
    origin they were written in is the one fitted to the moved centres, also
    where the descent is the centres' own, the operations they are not carried
    by within the precision left out.

    Raises
    ------
    ..refusal.InternalError
        When every point of ``points`` is read that way.  ``diagnose`` writes
        what the last one was read as, from ``(group, unitary operations,
        kept unmatched, found unmatched, orbit)``.
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
        unitary = [s for s in got.symmetries if not s.time_reversal]
        pairs, unmatched_kept, unmatched_found = _match_operations(kept, unitary, symprec,
                                                                   group.lattice)
        if int(got._lattice_index) <= 1 and not unmatched_kept and not unmatched_found:
            return (_shared_origin(got, kept, pairs), cell, cell_typat, pairs, point,
                    orbit_padded)
        last = (got, unitary, unmatched_kept, unmatched_found, orbit_padded)
    raise InternalError(
        diagnose(*last) + f" That is the last of {len(points)} generic points tried, "
        "and the orbit of none of them was identified as exactly the kept operations.")


def _centres_not_exactly_symmetric(model, group, assignment, *, tol,
                                   params) -> Refusal | None:
    r"""Centres whose defect alone breaks operations — refused before anything is built.

    In convention I the Bloch phase carries ``e^{2πi k·(R + τ_j − τ_i)}``, so
    centres that miss the group's own action by ``δ`` cost ``H(k)`` a covariance
    residual of order ``2π·δ·max|H|`` whatever the hoppings do.  Where nothing
    declares what noise means for the centres (no ``position_tol``, no
    ``precision``) and that estimate is above the detector's tolerance, the
    defect alone would break operations: which ones the Hamiltonian obeys would
    be the geometry's verdict as much as the hoppings', and everything built on
    it would be built again once the precision is declared.  So it is measured
    as soon as the cell's group is identified, and refused there, loudly, with
    the number to act on.

    Returns the ``positions_not_exactly_symmetric`` refusal, or ``None`` when the
    estimate is within ``tol`` and the detector's verdicts are the hoppings' own.
    """

    delta = float(_position_defect(model, group, assignment,
                                   {int(s.ind): s for s in group.symmetries}))
    if not delta > 0.0:
        return None
    max_h = _max_hamiltonian(model, params=params)
    estimate = 2.0 * np.pi * delta * max_h
    if estimate <= float(tol):
        return None
    return Refusal(
        "positions_not_exactly_symmetric",
        f"the orbital centres miss the cell's own symmetry by {delta:.3g} (reduced): "
        "declare this model's precision -- `realized_group(..., precision=)`, or "
        "`position_tol=` for the centres alone -- and centres within it are moved onto "
        "the positions the group puts them at, a k-dependent gauge transformation "
        "that leaves the spectrum untouched. Undeclared, that defect alone costs H(k) a "
        f"covariance residual of order 2*pi*delta*max|H| = {estimate:.3g}, above the "
        f"tolerance every operation's residual is held to, tol = {float(tol):.3g} "
        f"(max|H| = {max_h:.4g} about the mean on-site energy, measured here), so which "
        "operations this Hamiltonian obeys would be read off the geometry as much as "
        "off the hoppings. Nothing was built. The cell's group is "
        f"{group.name} #{group.number_str}, an upper bound on what this Hamiltonian "
        "obeys.",
        value=delta, threshold=float(tol) / (2.0 * np.pi * max_h),
        context={"group": group, "number_str": str(group.number_str),
                 "position_defect": delta, "max_h": max_h, "tol": float(tol)})


def _time_reversal_square(group, validated, *, tolerances=EXACT):
    r"""The pure time reversal's square, against the one the tables read here ask for.

    Time reversal is an antiunitary operation ``A = X K``, and the detector
    measures its intertwiner ``X`` like any other.  Its square ``A² = X X̄`` is a
    unitary symmetry of ``H(k)``, and rescaling ``X → λX`` multiplies it by
    ``|λ|²``, so where it is a multiple of the identity its **sign** belongs to
    the Hamiltonian: ``+1`` on spin-free states, ``-1`` on spinors.  The tables
    ask for one sign, the one the group's own multiplication table gives
    ``T·T`` -- ``+1`` in single-valued tables, ``-1`` in double-valued ones --
    and a time reversal whose square is the other one is not the time reversal
    those tables describe: a spinless model carrying spinors in its orbital
    index, as a wannier90 import does, read single-valued, or a model read
    double-valued whose only antiunitary symmetry with the k-action of time
    reversal squares to ``+1``.

    Where the solution space is more than one-dimensional -- a commutant, whose
    members can square to either sign -- the question is whether *some* member
    has the sign asked, and it is linear: time reversal's space is closed under
    transposition, and a unitary ``X`` squares to ``s`` exactly where
    ``Xᵀ = s X``.  So the member kept is the detector's own where its square
    already is the sign; otherwise the unitary nearest its part of that
    symmetry (or of one of three fixed combinations of the space), kept only
    where it squares to the sign and stays inside the space; and where neither
    is found, the detector's own again if its square reads the sign within the
    margin a numerical model's noise leaves it.  That member is the
    intertwiner every later step reads.

    Returns ``None`` where the group has no pure time reversal or the detector
    refuted it; otherwise ``ind``, ``asked`` (the sign the tables ask for),
    ``measured`` (the sign of the square of the member the verdict rests on --
    the one kept, or where none has the sign asked the detector's own -- where
    that square is a multiple of the identity, else ``nan``), ``obeyed`` and
    ``member`` (the member of the sign asked, or ``None`` where the space holds
    none).
    """
    position = next((p for p, s in enumerate(group.symmetries) if s.time_reversal
                     and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                        np.eye(3, dtype=int))
                     and np.abs(((np.asarray(s.translation, dtype=float) + 0.5) % 1.0)
                                - 0.5).max() <= 1e-6), None)
    if position is None:
        return None
    # Pure time reversal squares to the identity, and the sign the tables ask for
    # is their valuedness: -1 in double-valued tables, +1 in single-valued ones.
    # Nothing else would be read off the multiplication table, so it is not built.
    return _antiunitary_square(group, validated, position,
                               asked=-1 if group.spinor else 1, tolerances=tolerances)


def _antiunitary_square(group, validated, position, *, asked=None, tolerances=EXACT):
    r"""One antiunitary operation's square, against the sign the group's table gives it.

    :func:`_time_reversal_square`'s reading, for any antiunitary operation
    ``g·T`` whose unitary part squares to the identity modulo the lattice: its
    square ``X X̄`` is then an intertwiner of the identity, rescaling ``X``
    moves it only by ``|λ|²``, and where it is a multiple of the identity its
    sign is the Hamiltonian's own, to be read against the sign the group's
    multiplication table gives ``(g·T)²``.  The solution space of such an
    operation is closed under transposition as time reversal's is, so a member
    of the sign asked is looked for the same way: in the members of its own
    form, or where none has it (:func:`.._measure._own_form_read`) in the detector's
    whole space, which then says whether any matrix with its action on ``k``
    squares to the sign asked.

    Returns ``None`` where the detector refuted the operation, it carries no
    member, or its unitary part does not square to the identity -- a threefold
    or a fourfold with time reversal, whose square is another operation with a
    phase of its own, and so no sign to read; otherwise the record
    :func:`_time_reversal_square` returns.  ``asked`` is the sign the table
    would give, where that is known without it, and the table is then not built.
    ``tolerances`` is the record of what the call declared: a member read off a
    model symmetric only within its precision squares to the sign that far
    off, and its solution space holds a member to that precision.
    """

    ind = int(group.symmetries[position].ind)
    member = validated._intertwiners.get(ind)
    if ind in set(validated.broken) or member is None:
        return None
    if asked is None:
        prod, _tdiff, spf = group.get_product_table(get_diff=True)
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
    out = {"ind": ind, "asked": asked, "measured": measured, "obeyed": True, "member": X}
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
                out["member"] = U
                out["measured"] = float(np.sign((complex(np.trace(U @ U.conj()))
                                                 / U.shape[0]).real))
                return out
    if float(np.abs(square - asked * eye).max()) <= _TIME_REVERSAL_SQUARE_MARGIN:
        return out                               # the sign asked, read off a noisy member
    out.update(obeyed=False, member=None)
    return out


def _time_reversal_member(validated, square):
    """The detector's record with time reversal's member of the sign asked in place.

    ``square`` is :func:`_time_reversal_square`'s reading; where it kept the
    detector's own member, or found none of the sign asked, the record is
    returned as it stands.
    """
    if (square is None or not square["obeyed"]
            or square["member"] is validated._intertwiners.get(square["ind"])):
        return validated
    return replace(validated, _intertwiners={**validated._intertwiners,
                                            int(square["ind"]): square["member"]})


def _ansatz_refuted(model, group, assignment, validated, *, layout, spin,
                    blocks=None, max_candidates=2 ** 20, skip=()):
    r"""The operations the detector left undetermined, resolved by tensor structure.

    The detector's verdicts are per **k-action class**, and a surviving class of
    more than one member proves only that *some* operator with that k-action is
    a symmetry: its members are marked ``"undetermined"``
    (:attr:`ValidatedGroup.verdicts`) and are individually undetermined there.
    On a slab that is the whole horizontal-mirror family — every class is the
    pair ``{g, σ_h·g}``, the in-plane partner satisfies the intertwiner problem,
    and no presentation can change that, because a layer's ``H(k)`` does not
    depend on ``k_z`` and ``σ_h·g`` and ``g`` therefore impose the identical
    constraint.

    What *does* separate them is the **tensor structure** of the representation:
    ``D(g) = O(g) ⊗ S(g)``, with ``O`` fixed by the site permutation (the
    identity on a flat layer's sites) and ``S(g)`` the operation's own spin
    lift, and ``S(σ_h·g)`` is a different SU(2) element from ``S(g)``.  Each
    operation's own form -- each site's states carried onto its image's, with
    its own spin lift -- was read where the detector spoke
    (:func:`.._measure._own_form_read`): the solution space every reading here reads is
    the members of that form, which tells a pair apart by its lift on a
    multiplet of any dimension, and the operations none of whose members has
    it are refuted there and passed in as ``skip``.  What is left is the form
    on the **multiplets** (:func:`.._multiplets._ansatz_defects`): each carried
    where the operation carries its site, with the tables' lift on a
    one-dimensional multiplet and the model's own block on a wider one, which
    on a spinless model is what tells a pair apart.  An operation it refutes
    -- its scaled defect above the gate -- is **broken by tensor structure**, a
    per-operation verdict the descent below can act on where ``_build_rep``'s
    own covariance gate (a *global* "no candidate closes") cannot.

    Applied **exactly to the operations the detector declares undetermined** —
    nowhere else.  Where every k-action class is a singleton (every fully
    periodic model on its primitive cell, and every 2D model on a substrate)
    this runs on nothing and costs nothing: a singleton class carries the
    detector's own ``"obeyed"`` verdict, and its own form is the whole question.

    **One lift per operation, and it is the tables' own.**  An operation is never
    represented through its ``σ_h`` partner's lift: representing ``g``
    by ``S(σ_h)·S(g)`` is representing a *different* operation, and a model
    written for the opposite spin quantization axis reads the group it
    genuinely has in pythtb's convention.

    **So the test asks the operation's own branch, in every environment**, as
    its own form is read.  On the
    ``g⁻¹`` branch a member is read as ``D(g)†``, whose spin factor is
    ``S(g)†``: on a **fourfold** ``S(C4)† = −S(σ_h)·S(C4)``, the ``σ_h``
    partner's lift exactly, and on an antiunitary twofold it can be the
    partner's too, so accepting either branch keeps an operation on a lift the
    tables do not give it.  On a substrate cell the partner is not in the cell
    at all; on a free-standing one it is, and the operation kept on its
    partner's lift reaches the group besides it.  Measured on lin3g
    free-standing, where every multiplet is one spinor orbital: on the ``g``
    branch exactly one member of each of the sixteen ``σ_h`` pairs holds, and
    the two ``-4`` members and the time-reversal partners of the two diagonal
    twofolds, which pass on the ``g⁻¹`` branch alone, are refuted with the
    rest; and on the SRSI paper's ``P41'`` model in the opposite-axis
    transcription the ``g`` branch keeps ``{E, C2(z), m(x), m(y)}`` at ~7e-16
    on each, where both branches together also keep ``C4`` and ``C4⁻¹`` (``g``
    branch 1.0, ``g⁻¹`` branch 6.7e-16) and the survivors are not a group.
    Only the acceptance of an operation reads one branch: which branch the
    representation is built on is :func:`.._build._build_rep`'s convention.

    Returns ``(tested, refuted)`` — the per-operation record of the test, and
    the subset that failed — or the :class:`..refusal.Refusal` a declared
    spinor pair the operations do not carry onto one pair makes, or that a
    multiplet's own block enumeration made past ``max_candidates``
    (``rep_search_too_large``, where ``blocks`` was not already built for it).
    """

    shared = set(validated.undetermined) - set(skip or ())
    if not shared:
        return {}, {}
    positions = [i for i, s in enumerate(group.symmetries) if int(s.ind) in shared]
    got = _ansatz_defects(model, group, assignment, validated, positions,
                          layout=layout, spin=spin, blocks=blocks,
                          max_candidates=max_candidates, orients=(1,))
    if isinstance(got, Refusal):
        return got
    return got, {i: r for i, r in got.items() if not r["realizable"]}


def _translation_permutation(model, assignment, position):
    r"""The matrix carrying every orbital onto its translate's under one pure translation.

    ``P[σ(i), i] = 1`` for the site permutation ``σ`` the operation at
    ``position`` makes, each site's orbitals onto its image's in the order the
    sites list them, spin by spin.  ``None`` where a site and its image hold
    different numbers of orbitals.
    """
    perm = np.asarray(assignment.permutations, dtype=int)
    site_orbs = tuple(assignment.site_orbitals)
    nspin, nstate = int(model.nspin), int(model.nstate)
    P = np.zeros((nstate, nstate), dtype=complex)
    for i, orbs in enumerate(site_orbs):
        dst = site_orbs[int(perm[int(position), i])]
        if len(dst) != len(orbs):                   # pragma: no cover - defensive
            return None
        for slot, a in enumerate(orbs):
            for s in range(nspin):
                P[int(dst[slot]) * nspin + s, int(a) * nspin + s] = 1.0
    return P


def _translation_held(model, group, assignment, validated, position, *, spin=None,
                      tol=None, commuting=()):
    r"""Whether ``H(k)`` obeys one pure translation once each site's orbitals are carried onto its translate's within the site.

    A translation carries every site onto a copy of it, and a model may write
    the copy's orbitals in another basis -- a frame turned from one site to the
    next, a sign or a phase on an orbital -- and still be one crystal written on
    a bigger cell.  So the translation is read as every operation is, on its own
    solution space: the matrices commuting with ``H(k)`` at every ``k``, since a
    translation acts on ``k`` as the identity does, restricted to the blocks
    carrying each site's states onto its translate's
    (:func:`.._matrices._block_members`).  It holds where that restriction has a
    member nonsingular on every block (:func:`.._matrices._filled_check`): any
    unitary on each site's orbitals is allowed, the permutation of the sites
    itself among them, and on states that carry a spin, native or declared,
    the spin untouched -- every operation is read with its own spin rotation,
    and a translation's is the identity.  ``tol`` is the declared tolerance,
    relative to ``max|H|``, every one of those readings takes; ``None`` where
    nothing was declared.

    Translations commute, so the matrix a translation holds with is read among
    the members commuting with ``commuting``, the matrices already read for the
    others: the member nearest the permutation itself
    (:func:`_translation_permutation`), or the first of the fixed generic ones
    the verdict reads (:func:`.._matrices._generic_members`) where that one is
    singular, made unitary block by block.

    Returns ``(held, matrix, turn)``: whether the translation holds so; the
    unitary it holds with, ``None`` where no member commuting with
    ``commuting`` is nonsingular; and how far that unitary turns each site's
    orbitals onto its translate's -- half the smallest arc of the unit circle
    holding every eigenvalue of its blocks, so that one phase common to all of
    them is no turn, and ``0.0`` within the declared tolerance.  A space the
    detector kept none of, or one past the read-off's own ceiling, is read as
    not holding: the permutation alone is measured there
    (:func:`_translation_residual`).
    """

    nothing = (False, None, float("nan"))
    basis = (getattr(validated, "_bases", None) or {}).get(
        int(group.symmetries[int(position)].ind))
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
    if any(len(rows) != len(cols) for rows, cols in blocks):   # pragma: no cover - defensive
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


def _translation_residual(model, assignment, position, k_pts, params, grid=None):
    r"""How far ``H(k)`` is from obeying one **pure translation** as written.

    For a model that writes every copy of a site in one basis, in pythtb's
    convention I

    .. math:: T_\tau\,|\varphi_i(k)\rangle = e^{-2\pi i\,k\cdot\tau}\,
              |\varphi_{\sigma(i)}(k)\rangle,

    so ``D(τ)`` is the orbital permutation ``σ`` times a *global* phase — no
    internal factor (a translation rotates nothing), no SU(2) lift (it acts on
    spin as the identity) — and the covariance statement collapses to
    ``P H(k) P† = H(k)``, ``P`` the permutation matrix
    (:func:`_translation_permutation`); ``max_k`` of its defect is returned.  A
    model that writes the copies in bases turned from site to site obeys the
    translation with that change of basis beside the permutation, which
    :func:`_translation_held` reads.  Returns ``None`` when the permutation
    cannot be built (translationally paired sites carrying different numbers of
    orbitals).
    """
    P = _translation_permutation(model, assignment, position)
    if P is None:                                   # pragma: no cover - defensive
        return None
    # `grid` is the detector's own evaluation of H on this same mesh, handed in
    # by `realized_group`'s measurement context when the two k-sets are byte for
    # byte the same one (checked there, not assumed here).  It is the same array
    # either way -- `_GridHamiltonians` and the call below evaluate the model's one
    # Bloch sum on the same points -- so this measurement does not change, only
    # its cost.
    H = (np.asarray(grid.full()) if grid is not None
         else np.asarray(_bloch.hamiltonian(model, np.atleast_2d(k_pts), params)))
    return float(np.abs(np.einsum("ab,kbc,dc->kad", P, H, P.conj()) - H).max())


def _lattice_verdict(model, group, assignment, validated, refuted, tol, k_pts,
                     params, context=None, *, spin=None, matrix_tol=None) -> dict:
    r"""What the Hamiltonian did with the cell's own pure translations.

    A lattice supergroup carries ``_lattice_index - 1`` fractional translations
    ``{1|τ}`` and, with each of them, a whole coset of partners ``τ·g``.  Every
    one of them sits in the **identity's k-action class** — a pure translation
    acts trivially on ``k``, and ``τ·g`` acts exactly as ``g`` does — so the
    detector marks them ``"undetermined"``, and the tensor structure tells the
    partners apart (:func:`_ansatz_refuted`).

    **A translation is read at the sites.**  It carries every site onto a copy
    of it, and a model may write the copies in bases turned from site to site,
    so it holds where its solution space has a member carrying each site's
    orbitals onto its translate's by any change of basis within the site
    (:func:`_translation_held`); where the copies are written alike that member
    is the permutation of the sites itself, measured directly against ``tol``
    (:func:`_translation_residual`), and nothing is read beyond it.
    Translations commute, so each is read among the members commuting with the
    matrices the others hold with.  A Hamiltonian obeying several translations
    only with changes of basis that do not commute -- magnetic translations,
    a flux threading the cell -- obeys no crystal's translations, and of those
    only the translations obeyed as written are counted.  A translation the
    declaration or the centres refuted stays refuted whatever the hoppings
    obey: the crystal they state has no such translation.

    Returns the record the descent step carries:
    ``index``, ``fractional_translations``, ``operations`` (one entry per
    translation, with the ``.ind``, the reduced and Cartesian ``τ``, the direct
    covariance ``residual`` of the permutation itself, the scaled
    tensor-structure defect, and ``refuted_by`` in the descent's own words: the
    detector's reason, ``"no symmetry matrix"`` or ``"covariance"``, when the
    detector's own measurement refuted the translation, the refuting
    measurement's own word where the verdicts that read the detector's spaces
    refuted it -- ``"tensor structure"``, ``"declaration"``, ``"centres"`` --
    ``"translation"`` when none of them did and the Hamiltonian does not obey
    it, and ``None`` when it does), ``realized`` (the translations the
    Hamiltonian **obeys** — a supercell, if any), ``dressing`` (for each of
    them, the ``matrix`` it is obeyed with, which carries each site's orbitals
    onto its translate's, and the ``turn`` that matrix makes,
    :func:`_translation_held`), ``refuted`` (the translations refuted here and
    by nothing before) and ``coset_partners_refuted`` (how many refuted
    unitary operations are not themselves pure translations).
    """
    index = int(getattr(group, "_lattice_index", 1) or 1)
    if index <= 1:
        return {}
    broken = set(validated.broken)
    reason_of = {int(d["ind"]): str(d["refuted_by"]) for d in validated.discrepancies}
    refuted = dict(refuted or {})
    lattice = np.asarray(group.lattice, dtype=float)
    ident = np.eye(3, dtype=int)
    antiunitary = {int(s.ind) for s in group.symmetries if s.time_reversal}
    gone = (broken | set(refuted)) - antiunitary

    found, pure_inds = [], set()
    for position, s in enumerate(group.symmetries):
        if bool(s.time_reversal):
            continue
        if not np.array_equal(np.rint(np.asarray(s.rotation, dtype=float)).astype(int),
                              ident):
            continue
        t = np.asarray(s.translation, dtype=float).reshape(-1)
        t = t - np.rint(t)
        # zero as the identification read it: no longer than the finder's distance
        if float(np.linalg.norm(t @ lattice)) <= float(group.symprec):
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
            model, assignment, position, k_pts, params,
            grid=None if context is None else context.grid_for(model, k_pts))
        found.append((position, ind, tuple(float(x) for x in t),
                      tuple(float(x) for x in t @ lattice), why, scaled, residual))

    # THE HOPPINGS' TO DECIDE: every translation the detector, the declaration
    # and the centres left standing.  Those obeyed as written carry the
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
    r"""The verdict on a Hamiltonian that obeys the cell's own pure translation.

    Not a tolerance and not a defect in the model: the model **is** a supercell,
    and every k-point label, EBR row and SRSI computed on its folded zone would
    be a statement about the wrong Brillouin zone.  Where the copies are
    written in bases turned from site to site, the message says how far, and
    ``context["dressing"]`` carries the matrices each translation is obeyed
    with.  A falsy :class:`..refusal.Refusal`, never a returned group; the loop
    that measured it names the cell's group in it and carries it in
    ``context["group"]`` (:func:`.._realized._named_refusal`).
    """
    # the finer lattice is the one the Hamiltonian was measured to have, i.e. the
    # one the REALIZED translations generate -- not every translation the point
    # set allows, which is what the cell allows and is a different statement
    taus = tuple(lattice_record["realized"])
    lattice = np.asarray(group.lattice, dtype=float)
    prim = _primitive_lattice(lattice, taus)
    tau = np.round(np.asarray(taus[0], dtype=float), 6)
    cart = np.round(np.asarray(taus[0], dtype=float) @ lattice, 6)
    turn = max(float(d["turn"]) for d in lattice_record["dressing"])
    turned = ("" if turn == 0.0 else
              ", each site's orbitals carried onto its translate's by a change of basis "
              f"within the site that turns them by up to {turn:.3g} rad")
    rebuild = ("Rebuild the model on its primitive cell"
               if turn == 0.0 else
               "Rebuild the model on its primitive cell, every copy's orbitals written in "
               "one basis -- context[\"dressing\"] holds the matrices carrying each site's "
               "orbitals onto its translate's --")
    return Refusal(
        "supercell",
        f"the Hamiltonian obeys the pure translation {tau.tolist()} (reduced) = "
        f"{cart.tolist()} (Cartesian) that this cell's point set allows{turned}, so the "
        f"model is a supercell: its lattice is an index-{int(prim['index'])} "
        "sublattice of the crystal's own, whose primitive vectors are "
        f"{np.round(np.asarray(prim['cartesian'], dtype=float), 6).tolist()} "
        "(Cartesian). Every "
        "k-point label, EBR row and symmetry indicator computed here would be "
        "computed on a folded Brillouin zone -- the bands at one k of this cell are "
        "the crystal's bands at several k of its own zone, and no table names that "
        f"combination -- so no group is returned. {rebuild} and ask again; or, if "
        "the translation is meant to be broken, break it in the Hamiltonian -- a "
        "supercell whose hoppings distinguish the copies is a primitive model of a "
        "bigger cell and is descended to automatically.",
        value=float(lattice_record["index"]), threshold=1.0,
        context={"translations": taus, "index": int(lattice_record["index"]),
                 "primitive_lattice": prim,
                 "dressing": tuple(lattice_record["dressing"]),
                 "validated": validated, "lattice": lattice_record})


def _declared_cell_reduction(model, group, assignment, *, spin, characters):
    r"""Step 1: the cell the **declared content** leaves, before any Hamiltonian.

    The first of the three steps a measurement is: ``typat=`` says
    the atoms are different and ``characters=`` says what they *are*, and both
    are statements about the crystal that are made before a hopping is read.
    An operation the declared content is not carried by is therefore not an
    operation of this cell at all — it is not a symmetry the Hamiltonian
    breaks, and it never reaches a verdict.

    Two reductions, one rule (:func:`.._declaration._content_compatible`): the
    **species split**, where two sites of one orbit were declared different
    content -- the whole content each holds, however the words partition its
    orbitals, and as corep content where a word states that alone -- and the
    operations exchanging them go, and the **stabilizer of a declared
    function**, where a shorthand names a Cartesian polynomial whose span at
    its site is not invariant and the operations carrying it out of itself go.
    A single ``px`` at a ``3m`` site is a crystal with one mirror.  What is
    kept is the largest group inside the operations left, where a declaration
    that leaves sites of an orbit undeclared leaves operations that do not
    compose.

    Returns ``(kept, excluded, read_in)`` — the ``.ind`` kept, one record
    per excluded operation with the reason it was excluded, and where each
    declaration's word was read — or a :class:`..refusal.Refusal` where the
    declaration cannot be read in *this* cell's site groups at all, which is a
    statement about the group the loop is on and not about the crystal: the
    loop carries it and asks again on the next one.
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
    r"""Step 1 read in a layer's own site groups: the free-standing layer group's.

    What a declaration says two sites are is read where its words were
    written, and a layer's site groups hold ``σ_h``: ``E'`` and ``E''`` are
    different irreps of ``-6m2``, as ``s`` and ``p_z`` are different functions
    there, and a site declared the one and a site declared the other are
    different sites in every environment, as two sites of different ``typat``
    are.  On a substrate ``σ_h`` is gone and both restrict to one irrep of
    ``3m``, so the substrate's own site groups cannot say it: this reads the
    declaration in the site groups of the same point set with nothing under
    it, the free-standing cell, and the substrate's step 1 carries what it
    excludes (:func:`_carried_reduction`).  Where the substrate's site groups
    name nothing for the words at all -- a free-standing layer's declaration,
    read on a substrate by the cross-check -- they are read in this cell's and
    restricted (:func:`_restricted_contents`).

    Returns ``{"group", "assignment", "kept", "excluded", "read_in"}`` -- that
    cell's group, the orbitals' orbits in it and :func:`_declared_cell_reduction`'s
    reading there -- or ``None`` where the words name nothing in those site
    groups, written in the in-plane names a substrate's own site groups read,
    or the cell is not identified.
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
    r"""The declared words read where they were written, restricted to this group's site groups.

    ``written`` is the declaration as a larger group of the same point set
    reads it -- a layer's free-standing group, for the layer on a substrate
    (:func:`_layer_declared_reading`) -- whose site irreps the words are:
    ``E'`` and ``A2''`` of a ``-6m2`` site, which a substrate's ``3m`` names
    nothing for.  Read there, on the multiplets this group's Hamiltonian
    measures, each content is re-keyed onto this group's own operations,
    matched by rotation and translation modulo the lattice
    (:func:`_operation_key`), and the operations this group does not have drop
    out: that is the restriction of the declared content to the smaller site
    groups, ``A2''`` read as the ``A1`` of ``3m`` it restricts to, as after any
    descent.

    Returns one :class:`.._sitechars.DeclaredContent` per declared content, or
    ``()`` where the words name nothing in the larger group's site groups
    either.
    """

    larger, there = written["group"], written["assignment"]
    kept_u = {int(s.ind) for s in larger.symmetries if not s.time_reversal}
    kept_au = {int(s.ind) for s in larger.symmetries if s.time_reversal}
    try:
        tables = _site_tables(larger, there, kept_u, kept_au, range(len(there.site_orbitals)))
        decls = _normalize_declaration(characters, model, there, larger, tables, layout,
                                       spin=spin, read_in=written.get("read_in"))
    except _sitechars.DeclarationError:
        return ()
    theirs = {_operation_key(s): int(s.ind) for s in larger.symmetries}
    pairs = [(theirs[_operation_key(s)], int(s.ind)) for s in group.symmetries
             if _operation_key(s) in theirs]
    return _rekey_contents(tuple(d["content"] for d in decls), pairs, {})


def _carried_reduction(group, assignment, own, wider):
    r"""Step 1 on this cell, with the declared content also read in a larger group of its points.

    ``wider`` is the declaration read in the site groups of a group of the same
    point set that holds more operations -- a layer's free-standing group, for
    the layer on a substrate (:func:`_layer_declared_reading`) -- and every
    operation it excludes is excluded here too wherever this cell has it,
    matched by rotation and translation modulo the lattice
    (:func:`_operation_key`): two sites declared different content are
    different sites whatever group is asked about them.  ``own`` is this cell's
    own reading (:func:`_declared_cell_reduction`), whose exclusions stand
    beside those, or the refusal it made where the words name nothing in this
    cell's site groups; where it read them, it also says where they were read.

    What is carried from either reading is what the declared content is not
    carried by.  The operations kept are then the largest group inside the
    operations neither excludes, asked once, here
    (:func:`.._declaration._largest_compatible_group`), so an operation either
    reading left out only because its own largest group did not hold it is
    asked again rather than carried.

    Returns ``(kept, excluded, read_in)`` on this group, as
    :func:`_declared_cell_reduction` does.
    """

    if isinstance(own, Refusal):
        excluded, read_in = {}, dict(wider["read_in"])
    else:
        _kept, excluded, read_in = own
        excluded, read_in = ({i: rec for i, rec in excluded.items()
                              if not _outside_the_group(rec)}, dict(read_in))
    lattice = np.asarray(group.lattice, dtype=float)
    theirs = {_operation_key(s): int(s.ind) for s in wider["group"].symmetries}
    # a site of the larger group's orbits by the one it is here, through an
    # orbital on it: the two groups number their sites each their own way
    there = np.asarray(wider["assignment"].site_of_orbital, dtype=int)
    here = np.asarray(assignment.site_of_orbital, dtype=int)
    site_here = {int(there[o]): int(here[o]) for o in range(len(there))}
    for sym in group.symmetries:
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
    r"""The reduced cell of a step 1 reduction, identified and verified.

    Built exactly as a descent cell is (:func:`_descend_once`) — the parent
    cell's own points plus the orbit of one generic point under the operations
    kept, as one extra species — because only ``_group_of`` can *name* a
    subgroup and stamp the ``.ind`` order every table lookup keys on, and it
    answers about a cell.  What differs is what the cell **is**: not a group
    the Hamiltonian selected out of a larger one, but the declared crystal's
    own cell, so it carries no descent step and its provenance is ``realized``.

    Returns ``(group, cell, cell_typat, pairs, point, orbit, species)``, or the
    :class:`..refusal.Refusal` the identification itself made.

    Raises
    ------
    ..refusal.InternalError
        Where the operations kept do not close, or no generic point of
        :data:`_DECLARED_POINTS` gives a cell identified as them
        (:func:`_orbit_cell`).  The operations kept are a group by
        construction -- the largest subgroup inside the ones the declared
        content is carried by (:func:`.._declaration._largest_compatible_group`)
        -- and the orbit of a generic point under a subgroup has exactly it for
        a stabilizer, so anything else is arithmetic in this package failing
        rather than a measurement about the model: a cell reduction, like a
        descent, can only land on a smaller group, never fail.
    """
    kept_syms = [s for s in group.symmetries
                 if int(s.ind) in set(kept) and not s.time_reversal]
    leaks = _descent_closure(group, kept_syms)
    if leaks:                                    # pragma: no cover - defensive
        raise InternalError(
            f"the {len(kept_syms)} operations kept from those the declared orbital content "
            f"is carried by are not a group: {len(leaks)} of their products leave the set, "
            f"the first being #{leaks[0][0]} * #{leaks[0][1]} = #{leaks[0][2]}. They are the "
            "largest subgroup inside the compatible operations by construction, so this is "
            "a defect here and not a statement about the model.")
    species = _DECLARED_SPECIES
    while any(str(t) == species for t in base_typat):   # pragma: no cover - defensive
        species += "_"
    names = {int(s.ind): _record_name(s, group.lattice) for s in group.symmetries}

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What the identification of one reduced cell was read as, and why that is a defect."""
        return (
            f"the cell built to carry the {len(kept_syms)} operations the declared orbital "
            f"content leaves was identified as {got.name} #{got.number_str}, whose "
            f"{len(unitary)} unitary operations are not that set: "
            + (f"{len(unmatched_kept)} of the operations kept "
               + ", ".join(f"#{i} {names.get(int(i), '?')}" for i in unmatched_kept)
               + " were not found in it" if unmatched_kept else "")
            + (" and " if unmatched_kept and unmatched_found else "")
            + (f"{len(unmatched_found)} of its operations {unmatched_found} are not among "
               "them" if unmatched_found else "")
            + ". The operations kept are a subgroup (the largest inside the compatible "
            "ones) and the orbit of a generic point under a subgroup has exactly that "
            "subgroup for a stabilizer, so this is arithmetic in this package failing "
            f"rather than a measurement about the model. The orbit is "
            f"{np.asarray(orbit, dtype=float).tolist()} at symprec {float(symprec):.3g}.")

    made = _orbit_cell(base_cell, base_typat, group, kept_syms, points=_DECLARED_POINTS,
                       species=species, spinor=spinor, include_TR=include_TR,
                       symprec=symprec, diagnose=_diagnose)
    if isinstance(made, Refusal):
        return made
    got, cell, cell_typat, pairs, point, orbit_padded = made
    return got, cell, cell_typat, pairs, point, orbit_padded, species


def _declared_contents(model, group, assignment, validated, layout, *, spin, characters,
                       read_in=None):
    r"""``(decls, layout)`` — the declared content, and the multiplets it is read on.

    ``characters=`` is resolved in the site tables of the group the loop is
    currently on, into one :class:`.._sitechars.DeclaredContent` per declared
    content, and read on the multiplets the Hamiltonian measures: a
    declaration states what each site holds, and how the orbitals are
    partitioned inside a site is their basis
    (:func:`.._declaration._declared_blocks`).  The multiplets are the measured
    ones, whatever the declaration.

    ``characters`` may be the declaration as written or a mapping of contents
    already resolved on a larger group, which is what a descent hands back
    round: the constraint is then carried and restricted rather than resolved
    again, since a fresh resolution in a smaller site group refuses exactly
    where the restriction is merely reducible.

    ``read_in`` is step 1's own reading of the same words
    (:func:`_declared_cell_reduction`), so that a content resolved in a cell
    the declaration itself reduced still says where the word was read.

    A grammar error here is the same :class:`..refusal.Refusal`
    :func:`.._build._build_rep` would return for it.
    """

    kept_u = {int(s.ind) for s in group.symmetries
              if int(s.ind) in set(validated.indices) and not s.time_reversal}
    kept_au = {int(s.ind) for s in group.symmetries
               if int(s.ind) in set(validated.indices) and s.time_reversal}
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
    """The declared contents a round was read from, a block's in place of its one sum.

    What is carried to the next round and handed to the build: the
    declaration as it was written and resolved, so that every round reads it
    against its own measured multiplets (:func:`.._declaration._declared_blocks`).
    """
    return tuple(g["content"] for d in decls for g in d.get("groups", (d,)))


def _declared_contents_wider(model, group, assignment, layout, *, spin, characters,
                             read_in=None) -> tuple:
    r"""The declared words read in the site groups of the group the loop is on.

    The verdict stage reads a declaration in the site tables of the operations
    the detector kept (:func:`_declared_contents`), and a Hamiltonian that
    breaks most of the cell leaves those too small to name the word: ``A1`` of
    a ``3m`` site names nothing in the trivial site group the surviving
    operations leave.  What the word means is fixed where it was written — the
    site groups of the group the loop is on, or, on a layer, their in-plane
    part, since a free-standing layer's site groups carry ``σ_h`` and so have
    primed names only — and that reading, carried through the descent and
    restricted to the smaller site group, is the declaration there, as it is
    after any other descent.

    Returns one :class:`.._sitechars.DeclaredContent` per declared multiplet,
    or ``()`` where the words name nothing in either.
    """

    readings = [list(group.symmetries)]
    padded = tuple(int(d) for d in getattr(group, "vacuum_dirs", ()) or ())
    if len(padded) == 1:
        readings.append([s for s in group.symmetries
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
    r"""Whether carried contents are **restrictions** to this group's site groups.

    A content read in a larger site group and carried down says what every
    operation of the smaller one does, which is what makes it the restriction
    there.  One read in a *smaller* group than the site group it lands in — the
    in-plane part of a free-standing layer's site group, read before a descent
    that never came — leaves the other operations unsaid, and is not a reading
    of the word in that site group at all.
    """
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    unitary = {int(s.ind) for s in group.symmetries if not s.time_reversal}
    for content in contents:
        site = int(site_of[int(content.orbitals[0])])
        here = {int(i) for i in assignment.site_group[site]} & unitary
        if not here <= {int(k) for k in content.constraints}:
            return False
    return True


def _rekey_contents(contents, pairs, refuted):
    """The declared contents on a re-identified group, through ``(old, new)`` pairs.

    Operations the smaller group no longer has drop out of every constraint,
    and that dropping **is** the restriction of the declared irrep to the
    smaller site group: nothing is re-resolved, so a restriction that has become
    reducible is carried where a fresh resolution would refuse it.
    """
    drop = {int(i) for i in (refuted or ())}
    keep = tuple((int(a), int(b)) for a, b in pairs if int(a) not in drop)
    return tuple(c.rekeyed(keep) for c in contents)


def _declaration_refuted(group, validated, *, refuted, verdict, tolerances=EXACT):
    r"""Which operations the **declared orbital content** breaks, and what is left.

    The fourth per-operation verdict, written from the one reading of the
    solution spaces (:func:`.._declaration._verdicts_and_choices`, made once per
    round for every measurement that needs it).  ``characters=`` is a statement
    about what the orbitals *are*, and the hoppings cannot contradict it — they
    can only be the hoppings of a crystal with less symmetry.

    What is returned is the largest subgroup on which one representation per
    orbit of the sectors of the Hamiltonian's conserved matrices under **that
    subgroup**, of those the hoppings leave open, gives every declared site
    what was declared -- the same content where the whole site was declared, a
    content holding it where part was (:func:`.._declaration._covariant_subgroup`,
    :class:`.._sectors.SectorSearch`) -- the first by the tie rule where
    several of that order exist (:func:`_tie_ranked`); and the identity alone
    always does, which is what makes a descent by declaration land somewhere
    rather than refuse.  A coupled part of the crystal is one orbit of one
    sector, carried by every one-dimensional character, and uncoupled orbitals
    a symmetry exchanges are an orbit of several, which a smaller group splits.
    Where the sectors read nothing, the coupled parts' requirements decide: an
    operation whose declared multiplets of one part need different scalars,
    or whose solution space holds no nonsingular member in the block form of
    the multiplets a ``multiplets=`` states, is *individually* disallowed, and
    the subgroup is the largest on which every part has a one-dimensional
    character of it meeting all of its requirements.  Every declaration is such a
    constraint, wherever it sits, and an undeclared orbital, the reference
    orbital included, imposes none: declaring one sublattice of an orbit is
    the same statement as declaring the other, where the hoppings carry the
    two alike.  Every operation outside the subgroup is refuted the same way,
    and so is the primed partner of each: a site character is a character of
    the *unitary* site group, and the antiunitary copy of a broken operation is
    broken with it.

    Returns ``{"refuted", "subgroup", "candidates", "rows"}``, or the
    ``rep_search_too_large`` :class:`..refusal.Refusal` of a search past its
    ceiling.
    """

    syms = group.symmetries
    standing = set(validated.indices) - set(refuted)
    kept_u = [int(s.ind) for s in syms
              if int(s.ind) in standing and not s.time_reversal]
    got = verdict
    unitary_only = not any(s.time_reversal and int(s.ind) in standing for s in syms)
    sub = _covariant_subgroup(group, kept_u, got["requirements"],
                              unitary_only=unitary_only,
                              tol=tolerances.noise(CHARACTER),
                              joint=got.get("joint", ()),
                              search=got.get("search"),
                              matrices=got.get("matrices", ()))
    if isinstance(sub, Refusal):
        return sub
    lattice = np.asarray(group.lattice, dtype=float)
    # the primed partner of a broken unitary operation, through the group's own
    # product table: T * g, with T the pure time reversal
    prod = group.get_product_table(get_diff=False)
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
    # the contents read together, and those the sectors alone weigh, each named
    # where it is read
    read_together = tuple(got.get("joint", ())) + tuple(got.get("by_sectors", ()))
    for ind in sub["broken"]:
        rows = tuple(r for r in got["rows"] if int(r["ind"]) == int(ind))
        culprit = next((r for r in rows if r["required"] is not None
                        and int(ind) in got["reasons"]), rows[-1] if rows else None)
        note = got["reasons"].get(int(ind)) or _joint_reason(read_together, ind)
        if note is None:
            note = ("the declared content is carried by this operation on its own, but "
                    "no group of operations containing it and those kept carries every "
                    "declared multiplet in its declared irrep at once")
        together = next((j for j in read_together
                         if any(int(ind) in row for row in j["theta"].values())), None)
        if culprit is None and together is not None:
            # a content read over several multiplets: the declared value there,
            # and what its multiplets carry, each against its own part's scale
            culprit = {"declared": together["decl"]["constraints"].get(int(ind), 0.0),
                       "measured": sum(v for row in together["theta"].values()
                                       for g, v in row.items() if int(g) == int(ind))}
        orbitals = ({int(o) for r in rows for o in r["orbitals"]} if rows else
                    {int(o) for o in (together["decl"]["orbitals"] if together else ())})
        rec = {"ind": int(ind), "refuted_by": "declaration", "note": str(note),
               "orbitals": tuple(sorted(orbitals)),
               "declared": (None if culprit is None else complex(culprit["declared"])),
               "measured": (None if culprit is None else complex(culprit["measured"])),
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
    # the returned subgroup is the GREY one the survivors generate: a site
    # character decides about the unitary half, and the primed copy of an
    # operation it keeps is kept with it
    grey = set(sub["subgroup"]) | {partner[i] for i in sub["subgroup"]
                                   if i in partner}
    return {"refuted": broken, "subgroup": tuple(sorted(grey)),
            "candidates": sub["candidates"],
            "rows": _returned_rows(got["rows"])}


def _operation_verdicts(model, group, assignment, validated, *, tolerances, nk, params,
                        context, spin, layout, decls=None,
                        reference_orbital=0, max_candidates=2 ** 20, free_standing=None,
                        centres=None, sectors=None, own_form=None):
    r"""One verdict per operation of the group, each with the reason behind it.

    Each verdict judges noise against the call's own tolerances
    (:class:`.._tolerance.Tolerances`).  The detector has already spoken for most of them:
    :attr:`ValidatedGroup.discrepancies` carries ``"no symmetry matrix"``,
    ``"covariance"`` and ``"out of plane"`` per operation, antiunitary coset
    included.  So have the orbital **centres**, where a precision is declared:
    ``centres`` is their verdict (:func:`.._realized._centres_carried`), the operations they
    are not carried by within the declared precision, ``"centres"``, which
    stands for each of them whatever else reads it and which every verdict below
    reads as already given.  Four measurements finish the verdict where those
    cannot decide.

    The **own form** of every operation the detector kept, read once where the
    detector spoke (:func:`.._measure._own_form_read`) and handed in as ``own_form``, the
    records of the operations it refuted: a matrix with the operation's action
    on ``k`` represents it only where it carries each site's states onto its
    image's, with the operation's own spin lift on states that carry a spin
    (``"tensor structure"``, the record's ``own_form`` naming the condition
    that failed).  Where it refutes a unitary operation the detector found
    obeyed, the group is lowered at once: the multiplets, the declared content
    and everything read on them belong to the group that holds, and are read
    there on the next round.

    The **tensor structure** of the operations the detector leaves undetermined:
    a surviving k-action class of more than one member proves only that *some*
    operator with that k-action is a symmetry, and each member's own form, and
    on the multiplets :func:`_ansatz_refuted`, say which member it is
    (``"tensor structure"``) -- its own spin lift telling a layer's ``σ_h``
    pair apart where the multiplets are wider than one orbital and cannot.

    The **declared orbital content**, where there is one, read as what each
    site holds on the multiplets the Hamiltonian measures: every operation
    outside the largest subgroup the content is covariant on is broken by the
    declaration (:func:`_declaration_refuted`), each subgroup asked on its own
    orbits of the sectors of the Hamiltonian's conserved matrices -- ``sectors``,
    read once where the model entered (:func:`.._sectors.read_sectors`).

    The **pure translations** of a cell whose point set repeats under one, each
    read at the sites, as written or with each site's orbitals carried onto its
    translate's by a change of basis within the site (:func:`_lattice_verdict`).
    Their refutations join the rest under ``"translation"``; a translation the
    Hamiltonian *obeys* means the model is a supercell, which the record carries
    under ``realized`` for the step that owns that verdict.

    And what the same reading **pins**, by the same function: the scalar each
    orbit of multiplets carries relative to the anchor where the hoppings fix
    it, the ratios inside a part they leave free of it, and the coupled parts
    themselves (:func:`.._declaration._verdicts_and_choices`).  What the hoppings
    leave open is no measurement, and nothing here chooses it: the labelling
    is chosen once, by the lowest-angular-momentum rule inside
    :func:`.._build._build_rep`, among the characters the declarations leave.
    The reading is carried out of here so that the build assembles what was
    measured rather than measuring it a second time.  ``reference_orbital``
    is the one that build will be given: the orbital the lowest-angular-momentum
    rule reads the crystal from first, so that the descent and the build read
    one crystal.

    And last, once every other verdict leaves each unitary operation standing,
    the **projective class** of the states on a spinless reading: where the
    matrices the operations are on these states carry a class the single-valued
    tables do not -- two commuting operations whose matrices anticommute, the
    spin-½ sign of spinors stored in the orbital index -- the operations outside
    the largest subgroup that carries an ordinary representation are refuted
    (:func:`_projective_class_refuted`), and a tie between subgroups of that
    order is settled by the tie rule (:func:`_tie_ranked`), never by their
    intersection.  On a flat layer the class is read once, on the free-standing
    layer, every operation composed with the horizontal mirror given the
    mirror's chosen sign times its partner's matrix; a layer on a substrate
    carries the part of that reading that preserves the layer's normal
    (``free_standing``, the
    free-standing cell of its point set, :func:`_projective_class_carried`)
    rather than reading the class itself.

    Returns ``(refuted, lattice, declaration, choices, projective)`` -- the
    operations refuted past the detector, keyed by ``.ind``, the lattice record
    (empty where the cell carries no fractional translation), the declaration's
    own record (``None`` where nothing was declared), the one reading of the
    solution spaces and the projective class's own record (``None`` where
    nothing was refuted by it) -- or the :class:`..refusal.Refusal` the tensor
    test's own multiplet permutation, the declared reading or the block
    enumeration made.
    """
    syms = group.symmetries
    centred = dict(centres or {})
    own_form = dict(own_form or {})
    # THE OWN FORM of every operation the detector found obeyed, first: a
    # unitary operation none of whose matrices carries each site onto its
    # image with its own spin lift is not an operation of this Hamiltonian,
    # and the multiplets the rest of this round reads were measured with it.
    # So the group is lowered on that verdict at once, with the pure
    # translations measured as for any descent, and the next round reads the
    # rest on the group that holds.
    obeyed = set(validated.obeyed)
    lifted = {i: r for i, r in own_form.items() if i in obeyed}
    if any(int(s.ind) in lifted and not s.time_reversal for s in syms):
        lattice = _lattice_verdict(model, group, assignment, validated, lifted,
                                   tolerances.energy,
                                   _detection_grid(int(group.dim_k), int(nk)), params,
                                   context=context, spin=spin,
                                   matrix_tol=tolerances.matrix or None)
        refuted = dict(lifted)
        if lattice and lattice["refuted"]:
            refuted = {**refuted, **lattice["refuted"]}
        return {**refuted, **centred}, lattice, None, None, None
    # THE MULTIPLET BLOCKS, once per round.  The tensor test, the declared
    # reading and the representation built on the group that settles all read
    # them, and they are one object; building them per reader is the one cost a
    # wannier90 Hamiltonian cannot carry.  They are read on the operations that
    # hold their own form: one refuted by it has no matrix to read a block off
    blocks = _multiplet_blocks(
        model, group, assignment, spin, validated, layout,
        [p for p, s in enumerate(group.symmetries)
         if int(s.ind) in set(validated.indices) and int(s.ind) not in own_form],
        ref_chan=int(np.asarray(layout._chan_of, dtype=int)[0]),
        max_candidates=max_candidates, sectors=sectors)
    if isinstance(blocks, Refusal):
        return blocks
    # THE UNDETERMINED OPERATIONS, each told apart from its partners: by its own
    # form, read once above, and on the multiplets by its tensor structure
    shared = {i: r for i, r in own_form.items() if i not in obeyed}
    got = _ansatz_refuted(model, group, assignment, validated,
                          layout=layout, spin=spin, blocks=blocks,
                          max_candidates=max_candidates, skip=set(shared))
    if isinstance(got, Refusal):
        return got
    # the operations their own form or their tensor test refuted, and any
    # antiunitary one found obeyed that its own form refuted: those start no
    # descent, as no antiunitary refutation does; and the centres' verdict,
    # which every reading below takes as given
    refuted = {**lifted, **got[1], **shared, **centred}
    # THE ONE READING of the solution spaces: the verdict on the declared
    # content and the choice the representation is built from, together, over
    # the operations that are still standing.
    standing = set(validated.indices) - set(refuted)
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
        refuted = {**refuted, **declaration["refuted"]}
    # THE LATTICE, measured on the same group and before anything is concluded
    # from the rest.  A cell whose point set repeats under a pure translation was
    # returned as the lattice supergroup, and only the Hamiltonian can say what
    # that means: obeyed, the model IS a supercell; broken, the model's own
    # lattice is its translation group and the first descent is the one to it.
    lattice = _lattice_verdict(model, group, assignment, validated, refuted,
                               tolerances.energy,
                               _detection_grid(int(group.dim_k), int(nk)), params,
                               context=context, spin=spin,
                               matrix_tol=tolerances.matrix or None)
    if lattice and lattice["refuted"]:
        refuted = {**refuted, **lattice["refuted"]}
    # THE STATES' PROJECTIVE CLASS, last, and read on a group: once every other
    # verdict leaves each unitary operation of it standing.  A spinless reading
    # whose states carry a class the single-valued tables do not -- spinors in a
    # spinless model's orbital index -- has operations no single-valued
    # representation carries, and they leave the group here exactly as a broken
    # one does, for the descent to lower.  A layer reads it once, free-standing:
    # there every operation composed with the horizontal mirror has the
    # mirror's chosen sign times its partner's matrix, and a substrate carries
    # the part of that reading that preserves the layer's normal rather than
    # making its own.
    projective = None
    unitary = {int(s.ind) for s in syms if not s.time_reversal}
    if not unitary & (set(validated.broken) | set(refuted)):
        if free_standing is not None:
            more, projective = _projective_class_carried(model, group, validated,
                                                         free_standing=free_standing)
        else:
            more, projective = _projective_class_refuted(group, validated,
                                                         assignment=assignment,
                                                         decls=decls or ())
        refuted = {**refuted, **more}
    return {**refuted, **centred}, lattice, declaration, choices, projective


def _descend_once(base_cell, base_typat, group, validated, *, spinor, include_TR,
                  symprec, step, refuted=None, lattice_record=None,
                  declaration=None, projective=None, centres=None) -> tuple | Refusal:
    r"""Lower one cell to the unitary operations the Hamiltonian was measured to obey.

    The kept set ``K`` is known before this runs — it is
    :attr:`ValidatedGroup.indices` minus the antiunitary operations — so nothing
    here searches for a group.  What the cell is *for* is the identification
    step: only ``_group_of`` (spglib + ``irrep`` on a point set) can **name** the
    subgroup, supply the setting transformation ``refUC``/``shiftUC`` and stamp
    the ``.ind`` order every table lookup downstream keys on, and it answers
    about a cell.  So a cell carrying exactly ``K`` is built — the parent cell's
    own points plus the orbit of one generic point under ``K``, as one extra
    species — and handed back.  **The model is never touched.**

    ``refuted`` is the tensor-structure verdict of :func:`.._measure._own_form_read` — the
    operations none of whose matrices carries each site's states onto its
    image's with the operation's own spin lift — and of :func:`_ansatz_refuted`
    — the operations the detector left undetermined (``"undetermined"``) whose
    tensor structure on the multiplets refutes them — together with the
    operations the **declared orbital content** breaks
    (:func:`_declaration_refuted`, ``refuted_by = "declaration"``).  They leave ``K`` exactly as a detector discrepancy does
    and are recorded beside them, ``refuted_by`` saying which measurement made
    each verdict.

    ``declaration`` is that verdict's own record, carried onto the step so the
    returned group can say which readings of the declared content were possible
    where several tie (``tied``, in the order the tie rule ranks them, the one
    returned first).

    ``centres`` is the orbital centres' verdict under a declared precision
    (:func:`.._realized._centres_carried`), whose refuted operations are among ``refuted``
    under ``refuted_by = "centres"`` and stand for them over the detector's
    reading: carried onto the step as ``centres``, the move the parent group
    asked of the centres, the operations that carry them within the
    precision and a note.

    ``projective`` is the record of the states' **projective class**
    (:func:`_projective_class_refuted`), whose refuted operations are among
    ``refuted`` under ``refuted_by = "projective class"``: carried onto the step
    as ``projective``, the pairs of commuting operations whose matrices do not
    commute and, where subgroups of the kept order tied, all of them in the
    order the rule ranked them; on a layer, the sign the horizontal mirror was
    given free-standing, or on a substrate the free-standing group the class
    was read on (:func:`_projective_class_carried`).

    ``lattice_record`` is :func:`_lattice_verdict`'s record when the parent is a
    **lattice supergroup** — a cell that is a supercell of the crystal's own
    primitive cell whose Hamiltonian was measured to break the translation.  The
    step then *is* the lattice descent, and the identification is asserted to
    come back **primitive** (``_lattice_index == 1``): the generic orbit under a
    group whose translations are the model's own lattice has no finer
    translation, so anything else is a defect here rather than a measurement.

    A kept set that does **not close** is resolved rather than refused
    (:func:`_resolve_closure`): the largest subgroup inside it, or where several
    tie the first by the tie rule (:func:`_tie_ranked`), down to the identity
    alone — every Hamiltonian obeys some space group and the question which one
    is never refused.  The operations that leaves out are recorded beside the
    refuted ones, ``refuted_by = "closure"`` for one no candidate subgroup
    carries — an operation sitting the wrong side of the declared tolerance —
    and ``"tie"``
    for one only the candidates the rule did not keep carry; the step's ``tie``
    record lists the candidates in the rule's order and says why the first was
    kept.

    The cell is identified and verified by :func:`_orbit_cell`: an
    identification that does not reproduce ``K``, or is still not primitive, is
    an accident of the generic point, and the orbit is rebuilt from the next one
    of :data:`_DESCENT_POINTS`.

    Returns ``(group, cell, cell_typat, record)``, or the
    :class:`..refusal.Refusal` the identification itself made.

    Raises
    ------
    ..refusal.InternalError
        When no generic point of the sequence gives a cell identified as ``K``:
        the package's identification disagreeing with its own construction.
    """
    refuted = dict(refuted or {})
    antiunitary = {int(s.ind) for s in group.symmetries if s.time_reversal}
    keep = set(validated.indices) - antiunitary - set(refuted)
    kept = [s for s in group.symmetries
            if int(s.ind) in keep and not s.time_reversal]
    lattice = np.asarray(group.lattice, dtype=float)
    by_ind = {int(s.ind): s for s in group.symmetries}

    def _name(ind):
        """The operation as the descent's record lists it (:func:`_record_name`)."""
        return _record_name(by_ind[int(ind)], lattice)

    # THE SURVIVORS RESOLVED TO A GROUP.  An operation the declared tolerance
    # puts on the wrong side of the gate leaves the kept set here rather than
    # being returned or refused; where several candidate subgroups tie, the one
    # kept is the first by the tie rule (`_resolve_closure`), and the step says
    # which and why.
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
        """One broken operation, in the fields the measurement that broke it has.

        A verdict about the declared content carries no residual: nothing was
        near a gate, the hoppings are not those of the crystal that was
        declared, and what the record has to say is which orbitals, what they
        were declared to be and what they measure.
        """
        rec = refuted[i]
        why = str(rec.get("refuted_by", "tensor structure"))
        base = {"ind": int(i), "name": _name(i), "refuted_by": why,
                "step": int(step), "of": str(group.number_str)}
        if why == "centres":
            # a verdict about where the centres are carries no residual of H:
            # what it rests on is how far the operation alone would move one
            return dict(base, residual=float("nan"), miss=float(rec["miss"]),
                        precision=float(rec["precision"]), note=str(rec["note"]))
        if why == "declaration":
            return dict(base, residual=float("nan"),
                        orbitals=tuple(int(o) for o in rec.get("orbitals", ())),
                        declared=rec.get("declared"), measured=rec.get("measured"),
                        note=str(rec.get("note", "")))
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

    # the centres' verdict stands for the operations it refutes, whatever the
    # detector read on them: each broken operation is listed once, with its reason
    centred = {int(i) for i, r in refuted.items()
               if isinstance(r, dict) and r.get("refuted_by") == "centres"}
    broken = tuple(
        {"ind": int(d["ind"]), "name": _name(d["ind"]),
         "refuted_by": str(d["refuted_by"]),
         "residual": float(d["residual"]),
         "step": int(step), "of": str(group.number_str)}
        for d in validated.discrepancies if int(d["ind"]) not in centred) + tuple(
        _entry(i) for i in sorted(refuted)) + left_out

    species = _DESCENT_SPECIES
    while any(str(t) == species for t in base_typat):   # pragma: no cover - defensive
        species += "_"

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What the identification of one descent cell was read as, and why that is a defect."""
        where = np.round(np.asarray(orbit, dtype=float), 6).tolist()
        if int(got._lattice_index) > 1:
            return (
                f"the cell built to carry the {len(kept)} operations the Hamiltonian obeys "
                f"(from {group.name} #{group.number_str}) was identified as {got.name} "
                f"#{got.number_str} and is still not primitive: it carries the pure "
                "translation "
                f"{np.round(np.asarray(got._fractional_translations[0], dtype=float), 6).tolist()} "
                f"(index {int(got._lattice_index)}). The generic orbit added to it is a free "
                "orbit under a group whose translations are the model's own lattice, and "
                "such an orbit cannot carry a finer translation, so this is arithmetic in "
                "this package failing rather than a measurement about the model. The orbit "
                f"is {where} at symprec {float(symprec):.3g}.")
        return (
            f"the cell built to carry the {len(kept)} operations the Hamiltonian obeys (from "
            f"{group.name} #{group.number_str}) was identified as {got.name} "
            f"#{got.number_str}, whose {len(unitary)} unitary operations are not that set: "
            + (f"{len(unmatched_kept)} kept operations {unmatched_kept} were not found in it"
               if unmatched_kept else "")
            + (" and " if unmatched_kept and unmatched_found else "")
            + (f"{len(unmatched_found)} of its operations {unmatched_found} are not among "
               "the kept ones" if unmatched_found else "")
            + ". Returning this group would label the bands with a group that is either "
            "larger than the one measured or in a different position, so the descent "
            f"stops here. The generic orbit added to the cell is {where}; a symprec "
            f"({float(symprec):.3g}) too tight for it, or a cell whose own points are "
            "closer together than symprec, is what makes the identification miss.")

    # The cell handed in already carries the environment that was asked for --
    # its shadow species, where there is one, are part of the point set the
    # descent is lowering -- so nothing is put under it here: the orbit, and
    # exactly the orbit, is what this cell adds.
    made = _orbit_cell(base_cell, base_typat, group, kept, points=_DESCENT_POINTS,
                       species=species, spinor=spinor, include_TR=include_TR,
                       symprec=symprec, diagnose=_diagnose)
    if isinstance(made, Refusal):
        return made
    got, cell, cell_typat, pairs, point, orbit_padded = made

    record = {
        "step": int(step),
        "from": {"number_str": str(group.number_str), "name": str(group.name),
                 "size": int(group.size)},
        "to": {"number_str": str(got.number_str), "name": str(got.name),
               "size": int(got.size)},
        "dropped": broken, "n_dropped": len(broken),
        # how each verdict was reached: the detector's own reasons, "tensor
        # structure" for the operations it kept and the tensor structure
        # refuted -- an undetermined member of a class, or an operation whose
        # own spin lift its space does not carry -- and the rest in the words
        # above
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
        readings = tuple(declaration.get("candidates", ()))
        record["declaration"] = {
            "subgroup": tuple(declaration.get("subgroup", ())),
            "tied": readings,
            "rows": tuple(declaration.get("rows", ())),
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
                "`descent['tied']` lists them in that order.")
    if projective is not None:
        record["projective"] = dict(projective)
    if centres is not None:
        record["centres"] = {"precision": float(centres["precision"]),
                             "move": float(centres["move"]),
                             "kept": tuple(int(i) for i in centres["kept"]),
                             "tied": tuple(tuple(int(i) for i in c)
                                           for c in centres.get("tied", ())),
                             "note": str(centres["note"])}
    if lattice_record:
        record["lattice"] = dict(lattice_record)
        record["lattice"]["note"] = _lattice_note(lattice_record, got)
    return got, cell, cell_typat, record


#: Each cause :func:`_descent_counts` counts, as the record's one-line summary says it.
_DROPPED_BY_WORDS = {"centres": "by the centres", "hoppings": "by the hoppings",
                     "declaration": "by the declared content",
                     "closure": "left out to close a group", "tie": "by the tie rule",
                     "time reversal": "with time reversal",
                     "projective class": "by the states' projective class"}


def _descent_counts(steps, time_reversal) -> dict:
    r"""What a descent dropped, every step together, by cause.

    ``{cause: count}``, in the order the measurement meets the causes: the
    orbital centres beyond a declared precision; the hoppings; the declared
    orbital content; ``"closure"``, operations the Hamiltonian obeys that are
    left out so that the operations kept form a group; ``"tie"``, operations
    only the subgroups the tie rule did not keep hold; ``"time reversal"``, the
    operations combined with it where it was measured broken between two steps;
    and the states' projective class.  Each step drops operations of the cell
    no earlier step dropped, so the counts add up to the cell's group less the
    group the descent arrives at, a reduction of the cell by the declared
    content between two steps aside.  The descent's ``dropped_by`` holds the
    ones that are not zero, which its note and the record's one-line summary
    read.
    """
    dropped = [str(b["refuted_by"]) for st in steps for b in st["dropped"]]
    named = {r: dropped.count(r)
             for r in ("centres", "declaration", "closure", "tie", "projective class")}
    tr = time_reversal or {}
    between = sum(int(steps[k]["to"]["size"]) - int(steps[k + 1]["from"]["size"])
                  for k in range(len(steps) - 1)
                  if tr.get("outcome") == "measured broken"
                  and steps[k]["to"]["number_str"] == tr.get("grey_number_str"))
    return {"centres": named["centres"], "hoppings": len(dropped) - sum(named.values()),
            "declaration": named["declaration"], "closure": named["closure"],
            "tie": named["tie"], "time reversal": int(between),
            "projective class": named["projective class"]}


def _descent_clauses(counts) -> tuple:
    r"""``(split, lowered)``: a descent's counts by cause, and what its operations kept carry, in words.

    ``counts`` is :func:`_descent_counts`'.  ``split`` says what dropped the
    cell's operations: in the words the note has always had where only the
    hoppings and the declared content did, and otherwise each cause once, in
    the order the measurement meets them.  An operation the states' projective
    class leaves out is one the Hamiltonian obeys, and so is one left out so
    that the operations kept form a group, or by the tie rule.  ``lowered``
    names the group the cell was lowered to where the centres, the closure, the
    tie rule or the states' class dropped some, the largest group of its
    operations that carries what each of them asks, and is ``None`` where the
    hoppings and the declared content alone did.
    """
    d_cen, d_hop, d_decl = counts["centres"], counts["hoppings"], counts["declaration"]
    d_closure, d_tie = counts["closure"], counts["tie"]
    d_tr, d_proj = counts["time reversal"], counts["projective class"]
    # WHO BROKE WHAT.  The hoppings and the declared orbital content are two
    # different statements about the crystal, and a record that added them up
    # would hide which of the two is the one to act on.  So is the states'
    # projective class, which no hopping breaks: the spinless reading of spinor
    # states has no single-valued representation there.  And so are the
    # centres, beyond the declared precision: the geometry, not the hoppings.
    split = (
        f"the Hamiltonian broke {d_hop} of them by its hoppings and the declared "
        f"orbital content broke {d_decl} more" if d_hop and d_decl else
        f"the declared orbital content broke {d_decl} of them" if d_decl else
        f"the Hamiltonian broke {d_hop} of them")
    if d_cen or d_closure or d_tie or d_tr or d_proj:
        said = []
        if d_cen:
            said.append(f"the orbital centres are not carried by {d_cen} of them within "
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
            kept_as.append("that carry the orbital centres within the declared precision")
        if d_hop or d_closure or d_tie:
            kept_as.append("that the Hamiltonian obeys")
        if d_decl:
            kept_as.append("that carry the declared orbital content")
        if d_proj:
            kept_as.append("on which these states carry a single-valued representation")
        lowered = (f"the largest group of its {'unitary ' if d_tr else ''}operations "
                   + _joined(kept_as) + (", the first by the tie rule" if d_tie else ""))
    return split, lowered


def _lattice_note(lattice_record, got=None) -> str:
    """The lattice descent in words, and the group it arrived at where ``got`` names it.

    ``got`` is left out where that group is not what holds: a grey group on
    which time reversal is then measured broken.
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
