"""The multiplets the Hamiltonian binds each site's orbitals into, and the block
of every operation on each multiplet, read off the solution spaces: the finest
partition every validated operation respects, the orbits of sectors built by
induction, the blocks carried round an orbit, and whether each solution space
holds a member in the operation's own form.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import _sectors
from .refusal import InternalError, Refusal
from ._tolerance import (
    ANSATZ_GATE, ASSEMBLED, CHARACTER, FLAT_ZERO, SINGULAR, SNAP, SPREAD, ZERO,
    _IN_BLOCKS, _LAYOUT_MERGE, noise_floor)
from ._detector import _READOFF_MAX_BYTES
from ._matrices import (
    SiteMultiplets, _Union, _ansatz_blocks, _ansatz_member, _ansatz_residual,
    _atom_blocks, _atoms_of_site, _block_members, _block_space, _channel_classes,
    _channel_permutation, _filled_check, _identity_position, _lift_of,
    _multiplets_of, _parts_from_merge, _proportionality, _singleton_parts,
    _site_spaces, _spin_channel_permutation, _unfilled_blocks, _unitarized)
from ._grammar import _multiplets_grammar
from .symmetry import _fold


def _declared_parts(declared, site_orbs, site_of, atoms):
    """Stated multiplets, re-keyed onto this assignment's own sites.

    The multiplets are a partition of the model's **orbitals**, and which site
    index holds which multiplet is the assignment's business rather than the
    declaration's — so the multiplets are read as a flat set, each assigned to
    the site its orbitals sit on, and the sites' own order is applied here.
    That is what carries the multiplets from a model to another presentation of
    it, where the same orbitals are grouped in a different order.

    Everything that can go wrong is grammar, and raises ``ValueError``
    (:func:`.._grammar._declaration_grammar`), which every call that takes a statement
    has asked before anything was measured.
    """
    pairs = tuple(tuple(int(o) for o in atom) for site_atoms in atoms
                  for atom in site_atoms if len(atom) == 2)
    chans = _multiplets_grammar(declared, len(site_of), pairs, site_of)
    return tuple(tuple(sorted((ch for ch in chans if int(site_of[ch[0]]) == s),
                              key=lambda c: c[0]))
                 for s in range(len(site_orbs)))


def _slot_order(model, group, assignment, atoms, spaces, *, tol=ZERO):
    r"""Each site's atoms in the order of the atoms they are images of at its orbit's reference site.

    Which orbital at one site of an orbit answers to which at another is what
    the operations carry, never where the orbitals happen to be listed: an
    operation carrying the orbit's reference site onto another carries each of
    the reference's atoms into the atoms its solution space puts weight on
    there, and those share its slot.  Atoms an operation carries into one
    another's span -- the members of a multiplet, or copies of one irrep
    written in bases turned from site to site -- are paired in the order each
    site lists them, which is all a slot says about them: the multiplet they
    land in is decided by the site's own operations.  Of the operations
    reaching a site, the one that pairs the atoms most finely is read.  Where
    none reaches it, or none pairs them one to one, the site keeps its own
    order.  A weight below ``tol`` of the largest links nothing: ``1e-9``, or
    the declared tolerance where it is larger (:func:`.._tolerance.noise_floor`).

    ``atoms[s]`` is site ``s``'s atoms in its own order; returns them per site
    in slot order.
    """
    nspin = int(model.nspin)
    orbit_of = np.asarray(assignment.orbit_of_site, dtype=int)
    perms = np.asarray(assignment.permutations, dtype=int)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    ref = {}
    for s in range(len(atoms)):
        ref.setdefault(int(orbit_of[s]), s)

    def _states(atom):
        return np.asarray([o * nspin + t for o in atom for t in range(nspin)], dtype=int)

    out = [list(site) for site in atoms]
    for t in range(len(atoms)):
        s0 = ref[int(orbit_of[t])]
        n = len(atoms[t])
        if t == s0 or n <= 1 or n != len(atoms[s0]):
            continue
        cols = [_states(a) for a in atoms[s0]]
        rows = [_states(a) for a in atoms[t]]
        best = None
        for ind in sorted(spaces):
            p = pos_of_ind.get(int(ind))
            if p is None or int(perms[p][s0]) != t:
                continue
            V = np.asarray(spaces[ind])
            W = np.array([[float(np.abs(V[:, r][:, :, c]).max()) for c in cols]
                          for r in rows])
            if float(W.max()) <= 0.0:
                continue
            linked = _Union()
            for i in range(n):
                for j in range(n):
                    if float(W[i, j]) > tol * float(W.max()):
                        linked.union(("s", j), ("t", i))
            comps = {}
            for j in range(n):
                comps.setdefault(linked.find(("s", j)), ([], []))[0].append(j)
            for i in range(n):
                comps.setdefault(linked.find(("t", i)), ([], []))[1].append(i)
            if any(len(a) != len(b) for a, b in comps.values()):
                continue                  # not one to one: this operation says nothing
            if best is None or len(comps) > best[0]:
                best = (len(comps), comps)
        if best is None:
            continue
        slot = [None] * n
        for js, its in best[1].values():
            for j, i in zip(sorted(js), sorted(its)):
                slot[j] = atoms[t][i]
        out[t] = slot
    return out


def _site_multiplets(model, group, assignment, spin, validated, declared=None, tol=None,
                     sectors=None):
    r"""The multiplets this Hamiltonian glues at every site, measured.

    The rule, one sentence: the **finest** partition of each site's orbitals
    into multiplets for which every validated operation's solution space —
    ``validated._bases[ind]``, the detector's whole nullspace for that operation
    — still contains a nonsingular member in **block form**, block-diagonal
    within every site the operation fixes, the geometric permutation between
    the sites of an orbit, and a declared spinor pair never split.  *Finest*
    assumes nothing the hoppings do not force: two decoupled orbitals at one
    site, each with hoppings of its own, are two one-dimensional multiplets and
    never a doublet, because a diagonal member exists.  What those two carry
    *relatively* is a different question, and one a Hamiltonian that neither
    permutes nor couples them cannot answer.  Two identical ones are copies of
    one another, and those are one multiplet (below).

    The search is a refinement from singletons.  For a candidate partition and
    an operation, the members of the solution space that are in block form are
    measured (:func:`.._matrices._in_blocks`, :func:`_unfilled_blocks`); where the space
    cannot fill a block, what the space *does* couple that multiplet to says
    which multiplets merge.  Only operations the **coarsest** partition — the
    whole site as one multiplet, its blocks carried round the orbit as the
    tensor test carries them — makes realizable drive it: an operation not even
    that can carry is refuted by its own tensor structure, and gluing orbitals
    would substitute a partition for that verdict.  Sites in one orbit share one
    partition, so a merge measured at one site is transported to the whole
    orbit rather than measured again there, and the refinement terminates at
    the whole site at worst.

    **Orbitals on a site are glued by the operations that fix it.**  Those
    decide each site's finest multiplets, and every merge they ask for is made
    before any other.  An operation that carries a site onto another asks for
    a merge where the two sites write their orbitals in bases it does not carry
    into one another, and that merge is made only inside one isotypic
    component: the multiplets it would join carry copies of one site irrep, at
    every site of the orbit.  Their characters under the site group say so —
    the traces of their blocks in one member of each operation's solution
    space, so that the member's free phase is common to both, and on every
    member where the space holds more than one, since a relation the hoppings
    leave open is no evidence of one irrep.  Copies of one irrep written in
    bases rotated from site to site (``A'`` and ``A'`` at a mirror site) become
    one multiplet.  A merge across different irreps is not made, and the
    operation that asked for it stays refuted by its own tensor structure: a
    half translation the hoppings obey only dressed by a rotation of one
    site's frame would glue a ``p_x`` even under the site's mirror to a
    ``p_y`` odd under it, and it is no symmetry of the crystal.

    **No basis splits what the conserved matrices join.**  The matrices that
    commute with ``H(k)`` at every ``k`` and act within sites split the states
    into sectors (:mod:`._sectors`) whatever basis the orbitals are written in,
    and two orbitals of one site that each hold part of one sector are one
    multiplet.  A square's ``p_x, p_y`` pair that nothing couples is two
    sectors; written as ``(p_x ± p_y)/√2`` each orbital holds half of each, and
    the pair is one multiplet there as in every other basis, though in that
    one basis the fourfold's solution space holds a diagonal member and no
    operation would bind the two.  These merges are made first and carried
    round the orbit as every merge is.  **Identical copies are one multiplet**:
    where a sector holds several copies of one set of states, each copy written
    as orbitals of its own, the matrices exchanging the copies join every state
    of one copy to the same state of every other, so that a content mixing
    them -- two identical ``s`` orbitals declared the doublet ``E``, with a spin
    or without -- can be read on them.  ``sectors`` is the reading
    (:func:`.._sectors.read_sectors`), made here where none is handed in.

    Two incomparable finest partitions need an accidental degeneracy (a
    decoupled orbital whose bands coincide with a partner's).  The tie is
    resolved deterministically — the lowest-indexed partner — and recorded on
    :attr:`SiteMultiplets.accidental`; nothing is refused.

    **Which orbital answers to which across an orbit** is measured too, never
    read off the order the sites list their orbitals in: the operation carrying
    the orbit's reference site onto another says which of the reference's
    orbitals each one there is the image of (:func:`_slot_order`), and a slot is
    that correspondence.  So a site that lists ``(p_y, p_x)`` where the
    reference lists ``(p_x, p_y)`` has each orbital in its own multiplet, each
    reading its own irrep, and a merge is asked only where the bases genuinely
    turn.

    ``declared`` holds multiplets stated rather than measured, by
    ``multiplets=``: a :class:`SiteMultiplets`, or the multiplets of every
    site, each given as the orbitals it holds.  They are used verbatim, and a
    statement carries the multiplets from one presentation of a model to
    another.

    ``tol`` is the declared symmetry tolerance relative to ``max|H|``
    (:func:`.._tolerance.noise_floor`): every threshold here that measures noise -- how far
    a member sits outside the blocks, when a block is empty or singular, which
    weights link or couple, when two characters agree, which operations the
    whole-site partition carries -- is the larger of its exact default and it,
    and the record carries it for every later test of its blocks.  ``None``,
    nothing declared, is every default exactly.  A Wannier basis whose orbital
    frame is off the ideal one by less than the declared precision then reads
    the multiplets of the ideal one, measured as holding only within the
    declared precision: the Hamiltonian's own symmetry matrices mix those
    multiplets at the size of the turn, which is how far outside their block
    form the members they were read off reach (:attr:`SiteMultiplets._mixing`,
    :func:`_block_members`), and so do its conserved matrices, by how far a
    sector reaches between two of them (:func:`.._sectors.joined`).
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    atoms = _atoms_of_site(site_orbs, spin)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    if declared is not None:
        if (isinstance(declared, SiteMultiplets)
                and tuple(declared._site_orbitals) == site_orbs):
            # the same record, about the same sites: carried, not re-made, so
            # that multiplets measured once keep saying so wherever they are used
            return declared
        parts = _declared_parts(
            declared.by_site if isinstance(declared, SiteMultiplets) else declared,
            site_orbs, np.asarray(assignment.site_of_orbital, dtype=int), atoms)
        spaces = site_mixing = order = None
        if validated is not None and max((len(a) for a in atoms), default=1) > 1:
            spaces, _site, site_mixing = _site_spaces(model, group, assignment, spin,
                                                      validated, tol=tol)
        if spaces is not None:
            order = tuple(tuple(int(o) for a in site for o in a)
                          for site in _slot_order(model, group, assignment, atoms, spaces,
                                                  tol=noise_floor(ZERO, tol)))
        layout = _multiplets_of(model, assignment, spin, parts, source="declared",
                                order=order, tol=tol)
        if not tol or not spaces:
            return layout
        # WHETHER THE DECLARED MULTIPLETS HOLD EXACTLY or only within the
        # declared tolerance: how far outside their block form the members of
        # each operation's space that sit in it reach
        mixing = 0.0
        for ind, space in spaces.items():
            image = _channel_permutation(assignment, pos_of_ind[int(ind)], model, layout)
            got, reach = _block_members(space, _ansatz_blocks(layout._chan_states, image, 1),
                                        tol=noise_floor(_IN_BLOCKS, tol))
            if len(got):
                mixing = max(mixing, float(site_mixing.get(int(ind), 0.0)), float(reach))
        return replace(layout, _mixing=mixing)
    singletons = _singleton_parts(site_orbs, atoms)
    if max((len(a) for a in atoms), default=1) <= 1 or validated is None:
        # nothing at any site to glue, so nothing to measure: the finest
        # partition is the only one, and it is the whole verdict
        return _multiplets_of(model, assignment, spin, singletons, source="singletons",
                              tol=tol)
    bases = getattr(validated, "_bases", None) or {}
    if not bases:
        return _multiplets_of(model, assignment, spin, singletons, source="singletons",
                              tol=tol)

    orbit_of = np.asarray(assignment.orbit_of_site, dtype=int)
    perms = np.asarray(assignment.permutations, dtype=int)
    # THE SITE BLOCKS, once: they are the operation's own permutation of the
    # sites and no partition of a site's orbitals changes them, so the space
    # each operation is asked about is restricted to them before the
    # refinement starts.
    spaces, site_layout, site_mixing = _site_spaces(model, group, assignment, spin,
                                                    validated, tol=tol)
    # THE SLOTS: which atom of every site answers to which of its orbit's
    # reference site, as the operations carry them.  From here on `atoms[s][j]`
    # is the atom of site `s` in slot `j`.
    atoms = _slot_order(model, group, assignment, atoms, spaces,
                        tol=noise_floor(ZERO, tol))
    order = tuple(tuple(int(o) for a in site for o in a) for site in atoms)
    slot_of, orbs_of_slot = {}, {}
    for s, site_atoms in enumerate(atoms):
        for j, a in enumerate(site_atoms):
            for o in a:
                slot_of[int(o)] = (int(orbit_of[s]), j)
            orbs_of_slot.setdefault((int(orbit_of[s]), j), tuple(int(o) for o in a))
    merged, pending = _Union(), []
    # NO BASIS SPLITS WHAT THE CONSERVED MATRICES JOIN.  The sectors of the
    # matrices that commute with H at every k and act within sites are sets of
    # states, whatever basis the orbitals are written in, and two orbitals of a
    # site that each hold part of one sector are one multiplet: a square's
    # p_x, p_y pair nothing couples, written as (p_x +- p_y)/sqrt(2), where the
    # fourfold's solution space holds a diagonal member and no operation would
    # bind the two.  Made first, and carried round the orbit as every merge is.
    if sectors is None:
        sectors = _sectors.read_sectors(model, group, assignment, spin, validated,
                                        tol=tol or None)
    # ... and IDENTICAL COPIES ARE ONE MULTIPLET: where the conserved matrices
    # exchange copies of one set of states, which representation the copies
    # carry together is the freedom the hoppings leave, and a content that mixes
    # them -- two identical s orbitals declared the doublet E -- has to be
    # readable on them, so every state of a copy is one multiplet with the same
    # state of every other.  On states carrying a spin the copies are read in
    # the orbital factor, every copy carrying the operation's own spin lift
    joins, joining = {}, noise_floor(_IN_BLOCKS, tol)
    if sectors is not None:
        for s, site_atoms in enumerate(atoms):
            joins[s] = _sectors.joined(sectors, site_atoms)
            for i in range(len(site_atoms)):
                for j in range(i + 1, len(site_atoms)):
                    if float(joins[s][i, j]) > joining:
                        merged.union((int(orbit_of[s]), i), (int(orbit_of[s]), j))
    # WHICH OPERATIONS THE MULTIPLETS ARE ABOUT.  Gluing a site's orbitals can
    # only ever help an operation that the COARSEST partition -- the whole site
    # as one multiplet -- makes realizable; an operation not even that can carry
    # is refuted by its own tensor structure (a lattice supergroup's coset
    # partner, a sigma_h partner), and answering it by gluing orbitals would
    # substitute a partition for a verdict.  How far a member may sit from that
    # form and still be realizable is noise, so it reads the declared tolerance
    # too: a model symmetric only within its precision has every member that far off.
    positions = [pos_of_ind[int(i)] for i in validated.indices
                 if int(i) in bases and int(i) in pos_of_ind]
    coarse = _ansatz_defects(model, group, assignment, validated, positions,
                             layout=site_layout, spin=spin,
                             tol=noise_floor(ANSATZ_GATE, tol))
    if isinstance(coarse, Refusal):
        return _multiplets_of(model, assignment, spin, singletons, source="singletons",
                              order=order, tol=tol)
    glueable = {int(ind) for ind, rec in coarse.items() if rec["realizable"]}
    # the noise floors of the weights and characters read below
    link, agree = noise_floor(ZERO, tol), noise_floor(CHARACTER, tol)
    syms = group.symmetries
    # the operations a site's own characters are read under: the unitary ones
    # that fix it, among those the multiplets are about
    fixing_of = {int(s): [int(ind) for ind in spaces if int(ind) in glueable
                          and not syms[pos_of_ind[ind]].time_reversal
                          and int(perms[pos_of_ind[ind]][s]) == int(s)]
                 for s in range(len(site_orbs))}
    sites_of_orbit = {}
    for s in range(len(site_orbs)):
        sites_of_orbit.setdefault(int(orbit_of[s]), []).append(int(s))

    def _verdict(layout, asked, ind):
        """``(p, image, unfilled)`` of one operation on one partition.

        ``asked`` holds the partition's verdicts measured so far, each with how
        far outside its blocks the members in block form reach, so that the
        partition the refinement ends on is read once.
        """
        if ind not in asked:
            p = pos_of_ind[ind]
            image = _channel_permutation(assignment, p, model, layout)
            ok, unfilled, reach = _unfilled_blocks(
                spaces[ind], _ansatz_blocks(layout._chan_states, image, 1), tol=tol)
            asked[ind] = (p, image, () if ok else tuple(unfilled), float(reach))
        return asked[ind][:3]

    # the partition last swept, and its verdicts
    last = {}

    def _sweep(parts):
        """One merge, the first a verdict asks for; ``True`` when one ran.

        The merges an operation asks for at a site it fixes are made first;
        only where none is left is a merge asked across the sites of an orbit
        read, and made only between copies of one site irrep.
        """
        layout = _multiplets_of(model, assignment, spin, parts, order=order)
        asked = {}
        last.clear()
        last.update(parts=parts, layout=layout, asked=asked)

        def _one_irrep(orbit, ja, jb):
            """Do the multiplets of slots ``ja`` and ``jb`` carry copies of one site irrep?

            Asked at every site of the orbit, of the characters the operations
            fixing it carry on the two multiplets: the traces of their blocks in
            the same member of the operation's solution space, so that its free
            phase is common to both, and on every member of it.
            """
            for u in sites_of_orbit[int(orbit)]:
                a = int(layout._chan_of[atoms[u][ja][0]])
                b = int(layout._chan_of[atoms[u][jb][0]])
                here = np.asarray(site_layout._chan_states[u], dtype=int)
                for ind in fixing_of[u]:
                    members = spaces[ind]
                    lift = _lift_of(syms[pos_of_ind[ind]], layout._spinful)
                    size = [float(np.linalg.norm(X[np.ix_(here, here)]))
                            / np.sqrt(len(here)) for X in members]
                    for X, scale in zip(members, size):
                        if scale <= agree * max(size):
                            continue          # a member that does not live here
                        chi = [sum(_proportionality(X[np.ix_(at, at)], lift)[0]
                                   for at in _atom_blocks(layout._chan_states[c],
                                                          layout._spinful))
                               / float(layout._dim(c)) for c in (a, b)]
                        if abs(complex(chi[0]) - complex(chi[1])) > agree * scale:
                            return False
            return True

        for across in (False, True):
            for ind in spaces:
                if int(ind) not in glueable:
                    continue
                p, image, unfilled = _verdict(layout, asked, ind)
                space = spaces[ind]
                # A block the space cannot fill says this partition is too
                # fine: the multiplet the operation must carry there is glued to
                # another one of the same site, and WHICH one the space itself
                # says -- the multiplets of the image site it does put weight on.
                floor = link * max(float(np.abs(space).max()), 1.0)
                for c in unfilled:
                    if (int(perms[p][int(layout._chan_site[c])])
                            != int(layout._chan_site[c])) != across:
                        continue
                    cols = np.asarray(layout._chan_states[c], dtype=int)
                    weight = {}
                    for d in range(int(layout._nchan)):
                        if d == int(image[c]) or merged.find(
                                slot_of[layout._chan_orbs[d][0]]) == merged.find(
                                    slot_of[layout._chan_orbs[c][0]]):
                            continue
                        if int(layout._chan_site[d]) != int(perms[p][
                                int(layout._chan_site[c])]):
                            continue
                        rows = np.asarray(layout._chan_states[d], dtype=int)
                        w = float(np.abs(space[:, rows][:, :, cols]).max())
                        if w > floor:
                            weight[d] = w
                    if not weight:
                        continue
                    top = sorted(d for d, w in weight.items()
                                 if w >= _LAYOUT_MERGE * max(weight.values()))
                    mine = slot_of[layout._chan_orbs[c][0]]
                    if across:
                        # an operation carrying the site onto another glues
                        # copies of one site irrep and nothing else
                        top = [d for d in top if _one_irrep(
                            mine[0], mine[1], slot_of[layout._chan_orbs[d][0]][1])]
                        if not top:
                            continue
                    keys = [slot_of[layout._chan_orbs[d][0]] for d in top]
                    if not merged.union(mine, keys[0]):
                        continue
                    if len(top) > 1:
                        # several partners couple equally: the merge taken is
                        # the lowest-indexed one and the rest are recorded, to
                        # be kept as ties only where the multiplets end with
                        # them still apart
                        pending.append({"operation": int(ind), "chosen": (mine, keys[0]),
                                        "alternatives": tuple((mine, k) for k in keys[1:])})
                    return True
        return False

    parts = singletons
    for _ in range(sum(len(a) for a in atoms) + 1):
        parts = _parts_from_merge(site_orbs, atoms, orbit_of, merged)
        if not _sweep(parts):
            break
    parts = _parts_from_merge(site_orbs, atoms, orbit_of, merged)
    # WHETHER THE MULTIPLETS HOLD EXACTLY, or only within the declared
    # tolerance: how far outside the block form of the partition returned the
    # members it was read off reach, at the site blocks and at the multiplets,
    # over the operations it carries.  The last sweep measured that partition,
    # and is read rather than measured again.
    if last.get("parts") == parts:
        final, asked = last["layout"], last["asked"]
    else:
        final, asked = _multiplets_of(model, assignment, spin, parts, order=order), {}
    mixing = 0.0
    for ind in spaces:
        if int(ind) not in glueable or _verdict(final, asked, ind)[2]:
            continue
        mixing = max(mixing, float(site_mixing.get(int(ind), 0.0)), float(asked[ind][3]))
    # ... and how far a sector reaches between two multiplets of one site,
    # below what joins them
    for s, W in joins.items():
        for i in range(W.shape[0]):
            for j in range(i + 1, W.shape[0]):
                if merged.find((int(orbit_of[s]), i)) != merged.find((int(orbit_of[s]), j)):
                    mixing = max(mixing, float(W[i, j]))
    ties = []
    for rec in pending:
        apart = tuple(alt for alt in rec["alternatives"]
                      if merged.find(alt[0]) != merged.find(alt[1]))
        if not apart:
            continue
        ties.append({
            "operation": int(rec["operation"]),
            "orbitals": tuple(sorted({o for pair in (rec["chosen"],) + apart
                                      for key in pair for o in orbs_of_slot[key]})),
            "chosen": tuple(sorted(o for key in rec["chosen"]
                                   for o in orbs_of_slot[key])),
            "note": ("two partitions are equally fine here, an accidental degeneracy, "
                     "and the multiplet returned binds the lowest-indexed partners")})
    return _multiplets_of(model, assignment, spin, parts, ties=ties, order=order, tol=tol,
                          mixing=mixing)


def _channel_perms(model, assignment, syms, positions, layout, spin):
    """``{p: image}`` — where each operation carries each multiplet.

    Nothing is assumed: the image is lifted from ``assignment.permutations`` and
    the slot order of the site's orbitals (:func:`_channel_permutation`), and a
    **declared** spinor pair additionally has to land on one pair, which
    :func:`_spin_channel_permutation` measures, in the same slot order.  Each
    site's orbitals are read pair by pair as the pairs are declared, so every
    operation carries a pair onto a pair, and one that does not is this package
    disagreeing with itself.  That check is about the pair and the image is
    about the **multiplets**, one of which a pair may share with others, so the
    two are asked separately.  ``positions`` are indices into ``syms``.
    """
    if spin is None:
        return {p: _channel_permutation(assignment, p, model, layout)
                for p in positions}
    order = getattr(layout, "_site_order", None)
    cperm = {}
    for p in positions:
        image, split = _spin_channel_permutation(assignment, p, model, spin, order=order)
        if image is None:
            c, cu, cd = split
            up, dn = spin.states(c)
            raise InternalError(
                f"operation #{syms[p].ind} does not carry declared spinor pair {c} = "
                f"(up {up}, dn {dn}) onto one pair: it sends orbital {up} into pair "
                f"{cu} and orbital {dn} into pair {cd}. Each site's orbitals are read "
                f"pair by pair as the pairs are declared, and an operation carries a "
                f"site onto a site, so it carries each pair onto a pair: this is this "
                f"package's own reading failing, not a statement about the model or "
                f"its pairing.")
        cperm[p] = _channel_permutation(assignment, p, model, layout)
    return cperm


#: The roots of unity of order dividing twelve, ``e^{2πik/12}`` for ``k = 0, …, 11``,
#: each to the last bit: the characters of the one-dimensional irreps of an axis
#: up to sixfold, single- and double-valued.
_HALF_SQRT3 = float(np.sqrt(3.0)) / 2.0

_TWELFTH_ROOTS = tuple(complex(re_, im_) for re_, im_ in (
    (1.0, 0.0), (_HALF_SQRT3, 0.5), (0.5, _HALF_SQRT3), (0.0, 1.0),
    (-0.5, _HALF_SQRT3), (-_HALF_SQRT3, 0.5), (-1.0, 0.0), (-_HALF_SQRT3, -0.5),
    (-0.5, -_HALF_SQRT3), (0.0, -1.0), (0.5, -_HALF_SQRT3), (_HALF_SQRT3, -0.5)))


def _orbit_factor(z, noise):
    r"""The factor an orbit of multiplets carries relative to the reference orbit, as read.

    ``z`` is the ratio measured between the two orbits' blocks of one
    operation.  Within ``noise`` of a root of unity of order dividing twelve --
    a sign, a quarter turn, a character of a three- or sixfold axis, or one of
    those times a spin rotation's sign -- it is that root, exactly: the
    orbitals are written as one another's images, and what the measurement
    adds to the root is rounding, or a numerical model's noise.  Any other
    factor of unit modulus is a phase the basis functions carry, an orbital
    written as ``e^{iφ}`` times its partner's image, and it is kept as
    measured.  ``None`` where ``z`` is no phase: its modulus misses one by
    more than ``noise``.  The build's read-off reads every factor this way, and
    the three gates then say whether what it read is a representation.
    """
    z = complex(z)
    if not abs(abs(z) - 1.0) <= float(noise):
        return None
    root = _TWELFTH_ROOTS[int(round(float(np.angle(z)) * 6.0 / np.pi)) % 12]
    return root if abs(z - root) <= float(noise) else z / abs(z)


def _factor_said(z) -> str:
    """A factor between orbits of multiplets in words: a root of unity by name, a phase by value."""
    z = complex(z)
    if z in _TWELFTH_ROOTS:
        k = _TWELFTH_ROOTS.index(z)
        n = int(np.gcd(k, 12))
        return {0: "1", 3: "i", 6: "-1", 9: "-i"}.get(k, f"e^(2πi·{k // n}/{12 // n})")
    return f"{z:.6g}"


def _reversal_scalars(basis, blocks, lifts, classes, ref_chan, *, tol=_IN_BLOCKS):
    r"""The scalar time reversal carries on each orbit of multiplets, as its solution space pins it.

    Time reversal is assembled as its tensor structure: its own internal factor
    on every multiplet, times one scalar per orbit of multiplets.  Where the
    orbitals are real functions every scalar is one.  A basis function that is
    ``i`` times a real one turns its scalar to ``-1`` instead, and the
    Hamiltonian pins that sign wherever such an orbital couples to the
    reference: spinor states read spinless, whose time reversal of square
    ``+1`` the model's commutant supplies, are written that way, the states of
    one ``(p_x, p_y)`` spin sector coupled by an imaginary spin-orbit term.  So
    the scalars are read off the solution space and not assumed.  Its members
    in tensor form (:func:`_ansatz_residual`, one scalar per block) carry a
    vector of scalars on each orbit; two orbits whose vectors are parallel are
    pinned to one another at their ratio, an orbit pinned to the reference's
    takes its ratio to it, and a set of orbits the hoppings leave free of the
    reference is anchored at one on its lowest orbit -- rephasing orbitals the
    Hamiltonian does not couple to the rest changes nothing it measures.

    ``tol`` is how far from tensor form a member may be and still count as in
    it, relative to the space's own scale.  Returns one unit scalar per orbit,
    or ``None`` where the space holds no member in tensor form, or the scalars
    it pins are not of unit modulus -- where time reversal keeps the scalar one
    on every orbit.
    """
    B = np.asarray(basis, dtype=complex)
    if B.ndim != 3 or not len(B):
        return None
    A = np.stack([_ansatz_residual(B[j], blocks, lifts).reshape(-1)
                  for j in range(B.shape[0])], axis=1)
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    top = float(s[0]) if s.size else 0.0
    inside = [k for k in range(len(s)) if float(s[k]) <= tol * max(top, 1.0)]
    if not inside:
        return None
    members = [np.einsum("j,jab->ab", vh[k].conj(), B) for k in inside]
    classes = np.asarray(classes, dtype=int)
    ncls = int(classes.max()) + 1 if classes.size else 0
    ref_cls = int(classes[int(ref_chan)])
    vec = {}
    for cls in range(ncls):
        c0 = int(ref_chan) if cls == ref_cls else int(np.flatnonzero(classes == cls)[0])
        rows, cols = blocks[c0]
        vec[cls] = np.asarray([_proportionality(M[np.ix_(rows, cols)], lifts[c0])[0]
                               for M in members])
        if float(np.linalg.norm(vec[cls])) <= tol:
            return None                  # no member carries this orbit at all
    # which orbits the space pins to one another: parallel vectors of scalars,
    # read within the space's own noise; and a sign read within that noise is
    # that sign exactly, so that the matrices assembled from it carry none of it
    snap = max(SNAP, 10.0 * float(tol))
    anchor = {}
    for cls in [ref_cls] + [c for c in range(ncls) if c != ref_cls]:
        if cls in anchor:
            continue
        anchor[cls] = cls
        for other in range(ncls):
            if other in anchor:
                continue
            a, b = vec[cls], vec[other]
            if abs(complex(np.vdot(a, b))) >= (1.0 - snap) * float(
                    np.linalg.norm(a) * np.linalg.norm(b)):
                anchor[other] = cls
    out = np.ones(ncls, dtype=complex)
    for cls in range(ncls):
        a = vec[anchor[cls]]
        t = complex(np.vdot(a, vec[cls]) / np.vdot(a, a))
        if abs(abs(t) - 1.0) > snap:
            return None
        sign = 1.0 if t.real >= 0.0 else -1.0
        out[cls] = sign if abs(t - sign) <= snap else t / abs(t)
    return out


def _induced_blocks(model, group, assignment, spin, validated, layout, kept, cperm, bases, *,
                    ref_chan=None, sectors=None, max_candidates=2 ** 20):
    r"""Every orbit of sectors built by induction that the sectors read, each as a record.

    The sectors are read where the model entered (``sectors``, read here where
    none is handed in); where the multiplets hold each sector's copies
    together, every orbit of sectors under the operations ``kept`` that holds
    identical copies, or that uncoupled orbitals a symmetry exchanges make one
    the group's one-dimensional characters do not carry
    (:func:`.._sectors.induced_orbits`), is read (:class:`.._sectors.InducedOrbit`):
    its candidates counted before anything is built, each a representation of
    the orbit the hoppings leave open, the first built and checked against the
    group's own table and every operation's solution space.  On states that
    carry a spin the orbits are read on the orbital factor, each matrix
    carrying the operation's own lift on the spin, so identical copies there
    carry an ordinary representation, the double-valued factor system the
    spin's alone.  An orbit the sectors do not read, an orbit of one copy with
    no representation, or one whose first candidate misses those checks, is
    left to the multiplets' own blocks.  Returns the records (:meth:`.._sectors.InducedOrbit.record`), the
    orbit itself under ``"induced"``, or the ``rep_search_too_large`` refusal
    where the candidates exceed ``max_candidates``.
    """

    eps = getattr(layout, "_tol", None) or None

    def _induced(reading):
        return (reading is not None
                and (any(int(s.copies) > 1 for s in reading.sectors)
                     or any(len(h) > 1 for h in _sectors.held_sectors(reading, layout, tol=eps))
                     or bool(_sectors.crossed(reading, layout, group, kept, cperm, tol=eps))))

    reading = (sectors if sectors is not None else
               _sectors.read_sectors(model, group, assignment, spin, validated, tol=eps))
    if not _induced(reading):
        return []
    if sectors is not None and reading.tol is not None:
        # the frames the matrices are built in are the conserved matrices' of
        # THIS model: read where a precision was declared, the model handed in
        # may have been projected since, and its sectors are then exact
        reading = _sectors.read_sectors(model, group, assignment, spin, validated, tol=eps)
        if not _induced(reading):
            return []
    if _sectors.carried_reading(reading, layout, tol=eps) is not reading:
        return []                          # the multiplets keep the copies apart
    action = _sectors.sector_action(reading, model, group, assignment, spin, validated,
                                    tol=eps)
    try:
        orbits = _sectors.induced_orbits(reading, action, group, layout, kept, cperm,
                                         reference=ref_chan, tol=eps,
                                         max_candidates=max_candidates)
    except _sectors._TooMany as many:
        if many.sets is not None:
            return Refusal(
                "rep_search_too_large",
                f"the multiplets of orbitals {many.orbitals} hold {many.sets} sets of states "
                "the Hamiltonian conserves apart and no operation exchanges, and which "
                f"representation each carries is not fixed by it: the {many.count:,} ways of "
                f"choosing one for each are more than max_candidates = "
                f"{int(max_candidates):,}. Declare what the orbitals are (`characters=`), or "
                "raise `max_candidates=` if the enumeration is worth waiting for.",
                value=float(many.count), threshold=float(max_candidates),
                context={"multiplet": tuple(many.orbitals), "sets": int(many.sets),
                         "n_candidates": int(many.count),
                         "max_candidates": int(max_candidates)})
        return Refusal(
            "rep_search_too_large",
            f"the Hamiltonian conserves a {_fold(many.copies)} pseudospin on orbitals "
            f"{many.orbitals}, and which representation its copies carry is not fixed by "
            f"it: the {many.count:,} sums of irreducible representations of dimension "
            f"{many.copies} that could carry it are more than max_candidates = "
            f"{int(max_candidates):,}. Declare what the orbitals are (`characters=`), or "
            "raise `max_candidates=` if the enumeration is worth waiting for.",
            value=float(many.count), threshold=float(max_candidates),
            context={"multiplet": tuple(many.orbitals), "multiplicity": int(many.copies),
                     "n_candidates": int(many.count), "max_candidates": int(max_candidates)})
    gate = noise_floor(ASSEMBLED, eps)
    out = []
    for orbit in orbits:
        if orbit is None:
            continue
        record = orbit.record()
        if not record["n_admitted"] and int(record["dim"]) < 2:
            continue
        if record["n_admitted"] and orbit.check(record["candidates"][0]["W"], bases) > gate:
            continue
        out.append(record)
    return out


def _orbit_blocks(basis, image, chan_states, c0, members, Y, *, identity=False, tol=ZERO):
    r"""``{c: block}`` — one operation's block at every multiplet of a Wyckoff orbit.

    The block at the reference multiplet ``c0`` is ``Y``, a block of the
    operation's solution space at ``(image[c0], c0)``; the block at each other
    multiplet ``c`` is read at ``(image[c], c)`` off the **same member** of that
    space — the one whose reference block is ``Y``.  An intertwiner is one
    matrix, so its blocks at the other sites of the orbit are the operation
    acting in *their* frames: where the orbit permutes the orbital frame they
    are different matrices, the reference's frame carried round the orbit, and
    where every site carries the same functions they are the reference's block
    again.  Where the space is one-dimensional this is the model's own
    intertwiner read at every multiplet, up to one scalar for the operation.

    A multiplet whose block the reference's does not determine — one the
    Hamiltonian decouples from it, so that the member read there vanishes —
    takes the leading block the space carries there.  ``identity`` says the
    operation is the identity, whose reference block ``Y`` is the identity
    matrix in the phase the rest of the build is written in: it is that block
    at every multiplet, since ``D(E) = 1`` is what a representation means,
    whatever conserved operator the identity's solution space holds.  A block
    below ``tol`` of the reference's scale vanishes: :data:`.._tolerance.ZERO`,
    or the declared tolerance the space was measured at where that is larger.
    """
    c0 = int(c0)
    if identity:
        return {int(c): Y for c in members}
    out = {c0: Y}
    others = [int(c) for c in members if int(c) != c0]
    if not others:
        return out
    B = np.asarray(basis)
    r0 = np.asarray(chan_states[int(image[c0])], dtype=int)
    k0 = np.asarray(chan_states[c0], dtype=int)
    A = B[:, r0][:, :, k0].reshape(B.shape[0], -1).T
    coef = np.linalg.lstsq(A, np.asarray(Y, dtype=complex).reshape(-1), rcond=None)[0]
    floor = tol * max(float(np.abs(Y).max()), 1.0)
    for c in others:
        rows = np.asarray(chan_states[int(image[c])], dtype=int)
        cols = np.asarray(chan_states[c], dtype=int)
        blk = B[:, rows][:, :, cols]
        M = np.einsum("j,jab->ab", coef, blk)
        if float(np.abs(M).max()) <= floor:
            lead, _scale = _block_space(B, rows, cols, tol=tol)
            M = lead[0] if lead else M
        U = _unitarized(M, tol=tol)
        out[c] = U if U is not None else np.asarray(M, dtype=complex)
    return out


#: The signs a one-dimensional multiplet's transport between two sites of its
#: orbit is taken exactly at, where the ratio read off a solution space is
#: within :data:`.._tolerance.SNAP` of one, or within the declared tolerance
#: where that is larger: the orbitals written as one another's images, up to a
#: sign or a quarter turn of phase.
_TRANSPORT_SIGNS = (1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j)


def _transported_lifts(syms, kept, cperm, layout, members, c0, bases, intertwiners,
                       prod, ident, *, tol=SPREAD):
    r"""``{p: {c: S}}`` — a class of one-dimensional multiplets, carried round its orbit.

    A ``p`` along each site's own fourfold axis, the axes pointing along
    different cube edges, is a one-dimensional multiplet at every site, and an
    operation carrying one site onto the next carries it onto plus or minus the
    one there: the relative signs inside the orbit are what the model's
    transport says they are.  They are read here off
    **one member** of each operation's solution space -- its block at every
    multiplet of the class relative to its block at the reference multiplet,
    which keeps the tables' own lift -- and only where the space pins them:
    every member of a space with more than one gives the same ratio.

    What is read is taken only where it is a representation, since a sign per
    site that is not one is a gauge the hoppings carry and not a symmetry of
    the crystal (a flux threading the cell): the operations read must be closed
    under the group's product, and any two that commute must carry every
    multiplet of the class the same way in either order.  Otherwise, or where
    nothing is read at all, ``None``, and every multiplet keeps the tables' lift.

    Every threshold here measures the noise of the solution spaces, so each
    reads the declared tolerance the multiplets were measured at where that is
    larger than its exact default (:func:`.._tolerance.noise_floor`): a
    numerical model's transported sign misses the exact one by its own noise.
    """
    su2 = layout._spinful
    eps = getattr(layout, "_tol", None)
    tol = noise_floor(tol, eps)
    zero, snap = noise_floor(ZERO, eps), noise_floor(SNAP, eps)
    rel = {}
    for p in kept:
        ind = int(syms[p].ind)
        if p == ident:
            rel[p] = {int(c): 1.0 + 0.0j for c in members}
            continue
        basis = bases.get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            U = intertwiners.get(ind)
            basis = None if U is None else np.asarray(U)[None, :, :]
        if basis is None:
            continue
        B = np.asarray(basis)
        lift = _lift_of(syms[p], su2)
        floor = zero * max(float(np.abs(B).max()), 1.0)
        scalars = {}
        for c in members:
            rows = np.asarray(layout._chan_states[int(cperm[p][int(c)])], dtype=int)
            cols = np.asarray(layout._chan_states[int(c)], dtype=int)
            v = []
            for j in range(B.shape[0]):
                blk = B[j][np.ix_(rows, cols)]
                rho, res = _proportionality(blk, lift)
                if res > tol * max(float(np.abs(blk).max()), floor):
                    return None           # not the lift times a number: no such class
                v.append(complex(rho))
            scalars[int(c)] = np.asarray(v)
        s0 = scalars[int(c0)]
        n0 = float(np.vdot(s0, s0).real)
        if n0 <= floor ** 2:
            continue                      # the space does not carry the reference here
        got = {}
        for c in members:
            sc = scalars[int(c)]
            sig = complex(np.vdot(s0, sc) / n0)
            spread = float(np.abs(sc - sig * s0).max())
            if abs(sig) < snap or spread > tol * max(float(np.abs(sc).max()),
                                                     float(np.abs(s0).max())):
                got = None                # not pinned: the hoppings leave it open
                break
            # the sign an orbit's sites carry is discrete wherever the orbitals
            # are written as one another's images, and is taken exactly there:
            # the ratio read off a solution space carries its rounding, and a
            # numerical model's its noise.  A phase no such sign is near is a
            # gauge of the orbitals' own, and is kept as measured.
            phase = sig / abs(sig)
            near = min(_TRANSPORT_SIGNS, key=lambda w: abs(phase - w))
            got[int(c)] = near if abs(phase - near) <= snap else phase
        if got is not None:
            rel[p] = got
    unitary = [p for p in kept if not syms[p].time_reversal and p in rel]
    if len(unitary) < 2 or any(int(prod[a, b]) not in rel for a in unitary for b in unitary):
        return None
    for i, a in enumerate(unitary):
        for b in unitary[i + 1:]:
            if int(prod[a, b]) != int(prod[b, a]):
                continue
            for c in members:
                ab = rel[a][int(cperm[b][int(c)])] * rel[b][int(c)]
                ba = rel[b][int(cperm[a][int(c)])] * rel[a][int(c)]
                if abs(ab - ba) > snap:
                    return None           # a projective sign: a gauge, not a symmetry
    out = {}
    for p, got in rel.items():
        lift = _lift_of(syms[p], su2)
        out[p] = {int(c): (got[int(c)] * lift).astype(complex) for c in members}
    return out


def _phase_multiplet(ones, c0, chan_states, commutant, *, tol=SPREAD):
    """The one-dimensional multiplet a measured multiplet's blocks are phased against, or ``None``.

    The first of the one-dimensional multiplets ``ones`` the hoppings couple to
    multiplet ``c0``, directly or through others.  It is read off
    ``commutant``, the identity's solution space -- every matrix commuting with
    ``H`` -- whose members carry one scalar per coupled part of the crystal:
    two multiplets of one part carry proportional scalars on every member, and
    a multiplet the hoppings never reach carries a scalar of its own.  Where
    that space is not at hand the first of ``ones`` is taken.
    """
    if not ones:
        return None
    if commutant is None or len(np.asarray(commutant)) == 0:
        return int(ones[0])
    B = np.asarray(commutant, dtype=complex)

    def _scalar(c):
        st = np.asarray(chan_states[int(c)], dtype=int)
        return np.asarray([complex(np.trace(B[j][np.ix_(st, st)])) / len(st)
                           for j in range(B.shape[0])])

    t = _scalar(c0)
    for o in ones:
        a = _scalar(o)
        base = float(np.abs(a).max()) if a.size else 0.0
        if base <= tol:
            continue
        sig = complex(np.vdot(a, t) / np.vdot(a, a))
        if abs(sig) > tol and float(np.abs(t - sig * a).max()) <= tol * max(
                float(np.abs(t).max()), base, 1.0):
            return int(o)
    return None


def _multiplet_blocks(model, group, assignment, spin, validated, layout, positions, *,
                      ref_chan=None, tol=ZERO, max_candidates=2 ** 20, sectors=None):
    r"""``{ind: {channel: S}}`` — the internal block of every multiplet, the model's own.

    A one-dimensional multiplet's internal factor is the tables' own lift, times
    the sign the model's own transport gives it relative to its class's
    reference multiplet (:func:`_transported_lifts`): what an operation does to
    the ``p`` along a site's own fourfold axis is as much a measurement as what
    it does to a doublet, and it is the relative signs inside the orbit that
    carry it.

    A wider multiplet's block is the **model's own**, whatever basis its
    orbitals are written in, carried round the Wyckoff orbit.  The reference
    multiplet's block is the block of an intertwiner at ``(image channel,
    channel)``, read off the whole solution space rather than off the
    polar-extracted member alone, whose block at the operation's own image can
    vanish while the space still carries one; every other multiplet of the
    class takes its block from the same member of that space
    (:func:`_orbit_blocks`), which is the operation acting in that site's own
    frame.  So the representation of the orbit is the one induced from the
    reference site's in the model's own frames, and where every solution space
    is one-dimensional it is the model's own intertwiner read at every
    multiplet.  Nothing about which orbital functions a multiplet is enters:
    the site irreps are read off these blocks' characters afterwards
    (:func:`.._declaration._multiplet_characters`), and a basis turned inside a site changes
    the blocks and not their characters.

    Where a space is more than one-dimensional the extra members are the
    model's commutant acting inside the multiplet, and where the sectors of the
    conserved matrices read it the block is built by induction (below).  A
    multiplet they leave alone keeps the block its own intertwiner gives,
    carried round the orbit, and :func:`.._build._build_rep` asks of the matrices it
    propagates to, as of every other, whether they represent the group.

    Where the sectors of the conserved matrices (``sectors``,
    :func:`.._sectors.read_sectors`, read here where none is handed in) hold
    identical copies -- a pseudospin ``H`` conserves -- the hoppings do not
    settle what the copies carry; and where uncoupled orbitals a symmetry
    exchanges -- the ``p_x`` of one site and the ``p_y`` a screw carries it
    onto -- lie on sectors whose stabilizer has characters the group's do not
    restrict to, no one-dimensional character of the group carries what they
    can be.  Every representation of each such orbit of sectors the hoppings
    leave open is built from the reading, induced from the stabilizer of one of
    its sectors (:class:`.._sectors.InducedOrbit`), and the orbit's multiplets
    are written in the first; which one they carry is the labelling's choice.
    The sectors are what says whether a multiplet's conserved operators are
    copies or projectors.
    States carrying a spin, native or declared, carry that spin's own lift, as
    a one-dimensional multiplet carries the tables': every solution space read
    here was read in its operation's own form (:func:`_own_form_spaces`), its
    members ``O ⊗ S(g)`` with ``S(g)`` the operation's own lift and ``O`` the
    orbital factor ``H`` fixes, so the spin is never chosen, and an orbit of
    sectors there is built on the orbital factor, identical copies carrying an
    ordinary representation.

    Returns ``{ind: {channel: S}}`` over the multiplets a block was built for,
    with the orbits built by induction on its ``multiplicity``, or a
    :class:`..refusal.Refusal` from the multiplet permutation or from the cap
    on the representations the induction tries.  Every threshold here
    measures the noise of the solution spaces the blocks are read off, so each
    reads the declared tolerance the multiplets were measured at where that is
    larger than its exact default (:attr:`SiteMultiplets._tol`,
    :func:`.._tolerance.noise_floor`).
    """
    syms = group.symmetries
    prod = group.get_product_table(get_diff=False)
    ident = _identity_position(group)
    kept = list(positions)
    eps = getattr(layout, "_tol", None)
    tol, zero = noise_floor(tol, eps), noise_floor(ZERO, eps)
    u_pos = [p for p in kept if not syms[p].time_reversal]
    cperm = _channel_perms(model, assignment, syms, kept, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    classes, ncls = _channel_classes(cperm, u_pos, layout._nchan)
    ones = [c for c in range(int(layout._nchan)) if layout._chan_kind[c] == "tables"]
    if ref_chan is None:
        ref_chan = int(np.asarray(layout._chan_of, dtype=int)[0])
    bases = getattr(validated, "_bases", None) or {}

    out, multiplicity = {}, []
    commutant = bases.get(int(syms[ident].ind))
    # IDENTICAL COPIES, and uncoupled orbitals a symmetry exchanges, read once
    # where the model entered: every representation of each orbit of sectors
    # the one-dimensional characters of the group do not carry that the
    # hoppings leave open, built from the reading and counted before anything
    # is built; the orbit's multiplets are written in the first, and their
    # classes are read no further here.  On states that carry a spin the
    # orbits are read on the orbital factor, each matrix carrying the
    # operation's own lift on the spin
    done = set()
    got = _induced_blocks(model, group, assignment, spin, validated, layout, kept, cperm,
                          bases, ref_chan=ref_chan, sectors=sectors,
                          max_candidates=max_candidates)
    if isinstance(got, Refusal):
        return got
    for record in got:
        multiplicity.append(record)
        if not record["n_admitted"]:
            continue
        W = record["candidates"][0]["W"]
        local = {int(a): i for i, a in enumerate(record["states"])}
        for p in kept:
            for c in record["multiplets"]:
                rows = [local[int(a)] for a in layout._chan_states[int(cperm[p][c])]]
                cols = [local[int(a)] for a in layout._chan_states[int(c)]]
                out.setdefault(int(syms[p].ind), {})[int(c)] = W[p][np.ix_(rows, cols)]
        record["candidates"][0].release()
        done |= {int(classes[int(c)]) for c in record["multiplets"]}
    for cls in range(ncls):
        if cls in done:
            continue
        members = [c for c in range(int(layout._nchan)) if int(classes[c]) == cls]
        c0 = ref_chan if int(classes[ref_chan]) == cls else members[0]
        d = len(layout._chan_states[c0])
        if layout._chan_kind[members[0]] != "measured":
            # A ONE-DIMENSIONAL CLASS keeps the tables' lift at its reference
            # multiplet, and at every other multiplet the sign the model's own
            # transport gives it, read off one member of each solution space --
            # where that is a representation, and the lift everywhere where it
            # is not.
            if len(members) > 1:
                got = _transported_lifts(syms, kept, cperm, layout, members, c0, bases,
                                         validated._intertwiners, prod, ident)
                for p, per in (got or {}).items():
                    for c, S in per.items():
                        out.setdefault(int(syms[p].ind), {})[int(c)] = S
            continue
        one = _phase_multiplet(ones, c0, layout._chan_states, commutant,
                               tol=noise_floor(SPREAD, eps))
        space, polar, basis_of = {}, {}, {}
        for p in kept:
            ind = int(syms[p].ind)
            rows = layout._chan_states[int(cperm[p][c0])]
            cols = layout._chan_states[c0]
            got = validated._intertwiners.get(ind)
            basis = bases.get(ind)
            if basis is None or len(np.asarray(basis)) == 0:
                U = validated._intertwiners.get(ind)
                basis = None if U is None else np.asarray(U)[None, :, :]
            if basis is not None:
                basis_of[p] = np.asarray(basis)
                space[p], _scale = _block_space(basis, rows, cols, tol=tol)
            blk = (None if got is None else
                   np.asarray(got)[np.ix_(np.asarray(rows, dtype=int),
                                          np.asarray(cols, dtype=int))])
            if blk is None or float(np.abs(blk).max()) < zero:
                blk = space.get(p, [None])[0] if space.get(p) else None
            unit = None if blk is None else _unitarized(blk, tol=zero)
            if p == ident:
                # THE IDENTITY'S BLOCK IS THE IDENTITY.  Its solution space is
                # the model's whole commutant, so a polar extraction there
                # returns whichever conserved operator the decomposition
                # happened to hand back -- `-1` on a decoupled doublet -- and
                # that is not a measurement about the operation at all:
                # `D(E) = 1` is what a representation means.
                unit = np.eye(d, dtype=complex)
            if unit is not None and unit.shape == (d, d):
                # THE PHASE A MEASURED BLOCK IS READ IN.  The member this block
                # came from carries one free phase, and a block extracted with
                # it makes the *other* multiplets' scalars carry its inverse --
                # which the read-off then has to find among the two signs the
                # tensor structure offers and cannot.  Where the hoppings couple
                # the multiplet to a one-dimensional one there is a phase
                # convention to hand: the tables' own lift.  So the block is put
                # in it, and the scalars the rest of the build reads are the
                # model's own.  A one-dimensional multiplet the hoppings never
                # reach fixes nothing about this one's phase.
                if one is not None:
                    a_rows = layout._chan_states[int(cperm[p][one])]
                    a_cols = layout._chan_states[one]
                    got_u = got
                    if got_u is not None:
                        alpha, _res = _proportionality(
                            np.asarray(got_u)[np.ix_(np.asarray(a_rows, dtype=int),
                                                     np.asarray(a_cols, dtype=int))],
                            _lift_of(syms[p], layout._spinful))
                        if abs(alpha) > zero:
                            unit = unit * np.conj(alpha) / abs(alpha)
                polar[p] = unit

        def _carried(p, Y):
            """``{c: block}`` -- the member of ``p``'s space with block ``Y`` at ``c0``."""
            return _orbit_blocks(basis_of.get(p), cperm[p], layout._chan_states, c0,
                                 members, Y, identity=(p == ident), tol=tol)

        # CARRIED ROUND THE ORBIT, NEVER COPIED.  The sites of one Wyckoff
        # orbit are one another's images, so the block an operation has at the
        # next site is the operation acting in THAT site's frame, which a copy
        # of the reference's block is only where every site writes its orbitals
        # in the same frame.  The member of the solution space the reference
        # block came from says it at every multiplet at once.
        for p, S in polar.items():
            got = _carried(p, S)
            for c in members:
                out.setdefault(int(syms[p].ind), {})[int(c)] = got[int(c)]
    return _MultipletBlocks(out, multiplicity=multiplicity)


class _MultipletBlocks(dict):
    """``{ind: {multiplet: block}}``, and the orbits of sectors built by induction on the way.

    ``multiplicity`` holds one record per orbit of sectors built by induction
    (:func:`_induced_blocks`): the orbit's orbitals and states, the number of
    identical copies it holds -- one where it holds none -- how many
    representations of it there were and were admitted, and every admitted one
    with the site content it gives each multiplet and its own character.
    """

    def __init__(self, blocks=(), *, multiplicity=()):
        super().__init__(blocks)
        self.multiplicity = tuple(multiplicity)


def _ansatz_defects(model, group, assignment, validated, positions, *,
                    layout=None, spin=None, tol=None, orients=(1, -1),
                    blocks=None, max_candidates=2 ** 20):
    r"""Per-operation: does the detector's solution space contain an ``O ⊗ S``?

    The detector answers about a **k-action class**, not about an operation:
    two operations acting identically on reduced ``k`` pose the identical
    intertwiner problem and share one verdict, and on a slab every class is the
    pair ``{g, σ_h·g}`` (:attr:`..groups.ValidatedGroup.verdicts`).  A surviving
    multi-member class therefore proves only that *some* operator with that
    k-action is a symmetry, and which member it is, is a statement about the
    **tensor structure** of the representation rather than about ``H(k)``'s
    k-dependence: ``D(g) = O(g) ⊗ S(g)`` with ``O`` fixed by the site
    permutation and, on a one-dimensional multiplet, ``S`` the tables' own lift,
    and ``S(σ_h·g)`` is a different SU(2) element from ``S(g)``.

    This measures exactly that, per operation, with the arithmetic
    :func:`.._build._build_rep`'s read-off already uses on a generator: project the
    detector's whole solution space onto the tensor-structure subspace of that
    operation's own blocks and take the closest member (:func:`_ansatz_member`).
    The reported ``scaled`` defect is the norm of what that member could **not**
    put in tensor-structure form, divided by the member's own norm —
    dimensionless, ``0`` when the tensor structure is realizable and ``1`` when
    its component vanishes outright.
    The branches ``orients`` names are measured -- the ``g⁻¹`` branch's
    candidate is the adjoint and carries ``S(g)†`` on the transposed blocks --
    and an operation is realizable when *some* branch measured is, and
    ``orientation`` says which.

    ``positions`` are indices into ``group.symmetries``.  ``S(g)`` on a
    one-dimensional multiplet is that operation's own lift in pythtb's
    convention (:func:`_lift_of`) — there is no override of the lift, because
    an operation represented by some other SU(2) element is a different
    operation — carried round the orbit with the signs the model's own
    transport gives it (:func:`_transported_lifts`); on a wider multiplet it is
    the model's own block for **that** multiplet (:func:`_multiplet_blocks`),
    so that the tensor test asks exactly whether ``P(g) ⊗ S(g)`` is in the
    space the detector solved, and asks it of the frame a Wyckoff orbit
    actually has rather than of a copy of one site's.  A wider multiplet's
    block is read off a solution space both members of a ``σ_h`` fiber share,
    so there it tells them apart by nothing: only a one-dimensional multiplet's
    lift does.  So each operation's own form -- each site's states carried onto
    its image's, with its own lift on the spin -- is read on its whole solution
    space first (:func:`_own_form_spaces`), which tells such a pair apart on a
    multiplet of any dimension, and the spaces this test reads are the members
    of that form.  ``layout`` holds the multiplets to test against, measured
    here when omitted.

    ``tol`` is the gate on the scaled defect: :data:`ANSATZ_GATE` for a model
    built from exact numbers, or the declared tolerance the multiplets were
    measured at where that is larger (:func:`.._tolerance.noise_floor`) -- a model symmetric
    only within its precision has every member of a solution space that far
    from tensor form, while an operation the tensor structure refutes misses by
    order one.  ``None`` reads it off ``layout``.

    ``orients`` says which branches are measured, both by default.  The
    verdict on an operation the detector left undetermined asks the ``g``
    branch alone, ``(1,)``, in every environment
    (:func:`.._descent._ansatz_refuted`): on the ``g⁻¹`` branch the candidate
    carries ``S(g)†``, which for a fourfold **is** the ``σ_h`` partner's lift
    (``S(C4)† = −S(σ_h)·S(C4)``), so asking it would keep an operation on the
    strength of a lift the tables do not give it.  Which operations the
    multiplets are about is asked on both.  Returns
    ``{ind: record}`` with ``dim``, ``defect``, ``scale``, ``norm``, ``scaled``,
    ``orientation``, ``branches`` and ``realizable``.  An operation the
    detector kept no solution space for is reported with ``scaled = nan`` and
    ``realizable = True``: nothing was measured, so nothing is refuted.
    ``spin`` is the declared :class:`..spin.SpinBasis` itself, or ``None``.
    """
    if layout is None:
        layout = _site_multiplets(model, group, assignment, spin, validated)
    eps = getattr(layout, "_tol", None)
    if tol is None:
        tol = noise_floor(ANSATZ_GATE, eps)
    syms = group.symmetries
    positions = list(positions)
    cperm = _channel_perms(model, assignment, syms, positions, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    chan_states, su2 = layout._chan_states, layout._spinful
    ref_chan = int(np.asarray(layout._chan_of, dtype=int)[0])
    # the multiplet orbits of the operations under test, as _build_rep forms them:
    # the tensor structure's freedom is one scalar per orbit, not one per multiplet
    nchan = int(layout._nchan)
    u_pos = [p for p in positions if not bool(syms[p].time_reversal)]
    cls, ncls = _channel_classes(cperm, u_pos, nchan)
    class_blocks = [[c for c in range(nchan) if cls[c] == k] for k in range(ncls)]
    # the multiplets' blocks: the same ones the representation is assembled
    # from, so that the tensor test and the build rest on one tensor structure
    # rather than two.  Measured once per set of multiplets and per round, and
    # handed in where the call already has it: the block IS the same
    # measurement, and re-solving it per test is the one cost this package
    # cannot afford on a wannier90 Hamiltonian.
    got = blocks if blocks is not None else _multiplet_blocks(
        model, group, assignment, spin, validated, layout,
        [p for p, s in enumerate(syms) if int(s.ind) in set(validated.indices)],
        ref_chan=ref_chan, max_candidates=max_candidates)
    if isinstance(got, Refusal):
        return got
    blocks_of = got

    out = {}
    for p in positions:
        sym = syms[p]
        ind = int(sym.ind)
        rec = {"ind": ind, "dim": 0, "defect": float("nan"), "scale": float("nan"),
               "norm": float("nan"), "scaled": float("nan"), "orientation": 0,
               "branches": {}, "realizable": True, "note": None}
        best = None
        basis = getattr(validated, "_bases", {}).get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            member = validated._intertwiners.get(ind)
            if member is None:
                rec["note"] = ("no solution of this operation's condition is carried, "
                               "so its tensor structure was not tested")
                out[ind] = rec
                continue
            # the polar-extracted member IS the space when dim == 1, and where it
            # is not, the projection has nothing to choose from and the defect
            # says so -- a fallback, not a wrong verdict (as in the read-off).
            basis = np.asarray(member)[None, :, :]
            rec["note"] = ("only the unitary member nearest the solution space was "
                           "tested, not the whole space")
        basis = np.asarray(basis)
        rec["dim"] = int(basis.shape[0])
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            # the same ceiling the read-off applies: a projection that would
            # allocate more than this is not run, and an untested operation is
            # reported as such rather than refuted on a measurement not made.
            rec["note"] = (f"the solution space is {basis.shape[0]}-dimensional on "
                           f"{nstate} states, too large to test its tensor structure "
                           f"within {_READOFF_MAX_BYTES / 1e6:.0f} MB")
            out[ind] = rec
            continue
        lift = _lift_of(sym, su2)
        zero = None
        for c in range(nchan):
            if layout._chan_kind[c] != "measured":
                continue
            rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
            cols = np.asarray(chan_states[c], dtype=int)
            if float(np.abs(basis[:, rows][:, :, cols]).max()) < noise_floor(ZERO, eps):
                zero = int(c)
                break
        if zero is not None:
            # No member of the space sends this multiplet where the operation
            # must -- the case a LATTICE SUPERGROUP puts every coset partner
            # in, where the measured operator is the partner's and never this
            # operation's -- so the tensor structure is refuted rather than left
            # untested.
            rec.update(defect=1.0, scale=1.0, norm=1.0, scaled=1.0,
                       orientation=0, realizable=False,
                       note=(f"no member of the measured solution space carries multiplet "
                             f"{zero} to the multiplet the operation carries it to: every "
                             f"member's block there is zero"))
            out[ind] = rec
            continue
        blocks_here = blocks_of.get(ind, {})
        if layout._measured and not blocks_here:
            # Where no block could be read off the model for a multi-dimensional
            # multiplet there is nothing to test the tensor structure against:
            # reported, never refuted on it.
            rec["note"] = ("no internal block could be read for this operation on a "
                           "multi-dimensional multiplet")
            out[ind] = rec
            continue
        S_of_chan = [blocks_here.get(c, None if layout._chan_kind[c] == "measured" else lift)
                     for c in range(nchan)]
        if any(S is None for S in S_of_chan):  # pragma: no cover - a guard, not a path
            rec["note"] = "no internal block of the right size could be read"
            out[ind] = rec
            continue
        for orient in tuple(orients):
            lifts = [S.conj().T if orient < 0 else S for S in S_of_chan]
            blocks = _ansatz_blocks(chan_states, cperm[p], orient)
            try:
                X, defect, scale = _ansatz_member(basis, blocks, lifts, class_blocks)
            except np.linalg.LinAlgError:      # pragma: no cover - not seen on a fixture
                rec["note"] = "the projection onto the tensor-structure form did not converge"
                continue
            norm = float(np.linalg.norm(X))
            scaled = float(defect / norm) if norm > 0 else float("inf")
            rec["branches"][int(orient)] = scaled
            if best is None or scaled < best:
                best = scaled
                rec.update(defect=float(defect), scale=float(scale), norm=norm,
                           scaled=scaled, orientation=int(orient))
        if best is not None:
            rec["realizable"] = bool(best <= float(tol))
        out[ind] = rec
    return out


def _own_form_spaces(model, group, assignment, validated, positions, *, spin=None, tol=None,
                     covariance=None):
    r"""Each operation's solution space restricted to the members of its own form, and the verdict on it.

    The detector finds every matrix carrying ``H(k)`` onto ``H(gk)``, whichever
    operation of the crystal it belongs to: one action on ``k`` poses one
    problem, and where several operations pose it -- a translation and the
    identity, time reversal and time reversal composed with a translation, a
    layer's ``g`` and ``σ_h·g`` -- or where the Hamiltonian conserves more than
    the crystal does, a spin rotation among it, the space holds members of
    other operations and other symmetries.  A matrix represents ``g`` only in
    ``g``'s own form, asked of one member: it carries each site's states onto
    its image's under ``g``, with any change of basis within the site, and on
    states that carry a spin, native or declared, it carries ``g``'s own spin
    lift ``S(g)`` (:func:`_lift_of`) -- ``O ⊗ S(g)``, the orbital factor ``O``
    block by block on the site permutation.  Both conditions are linear, so the
    members of that form are read off the space once: a member is of it where
    what no matrix of the form carries is within :data:`ANSATZ_GATE` of the
    member's own norm, the defect and the gate every test of tensor structure
    reads, so that matrices missing the form by a model's own numerical noise
    are its matrices still.  The operation holds where one of them is
    nonsingular on every site block (:func:`_filled_check`).  Neither condition
    alone is enough: in a
    Néel antiferromagnet time reversal's space holds members keeping every site
    -- time reversal with a spin rotation the exchange field allows -- and
    members carrying ``iσ_y`` -- time reversal with the half translation,
    which exchanges the two sites -- and no member that does both.

    ``tol`` is the declared tolerance, relative to ``max|H|``, every threshold
    here reads (:func:`.._tolerance.noise_floor`); ``None`` keeps the exact
    defaults.  Where ``covariance`` is given, a form missed by more than
    rounding is also held to the detector's own measure: the member nearest
    the form, its part in the form, is held to the covariance gate
    ``validated.tol`` the operation was kept at (``covariance`` is called with
    the operation's position and that matrix, and returns its covariance
    residual over the detector's grid).  A gate on the form alone cannot tell
    a model's noise from a small angle: spin blocks written in a frame turned
    against the lattice by ``δ`` carry every operation whose lift the turn
    changes through a spin factor ``δ`` from its own, inside any such gate,
    and with its own lift they carry ``H(k)`` onto ``H(gk)`` only to what the
    turn does to ``H``.  So where members are of the form within the gate, the
    operation holds where that part is within the covariance gate; where it is
    not, the condition that failed is the spin lift where the member's part on
    the site blocks, with any spin factor, carries ``H(k)`` within it, and the
    site permutation otherwise.  Where no member is of the form within the
    gate, as a model symmetric only within its precision needs, the operation
    holds where that part is within the covariance gate and nonsingular on
    every site block -- a part the form leaves empty, as it leaves a
    ferrimagnet's spin-diagonal members under time reversal's ``iσ_y``, is no
    matrix of the operation.

    Returns ``(spaces, records)``.  ``spaces[ind]`` holds the members of the
    form -- the space itself where every member has it, the one member nearest
    it where the covariance kept an operation no member of which is of the
    form within the gate, none where the operation is refuted -- in the
    model's own state order; it is what every later reading of the operation
    reads.  ``records[ind]`` carries :func:`_ansatz_defects`'s fields
    (``dim``, ``defect``, ``scale``, ``norm``, ``scaled``, ``orientation``,
    ``branches``, ``realizable``, ``note``), the ``covariance`` residual where
    it was measured, and ``own_form``: ``None`` where the operation holds,
    else the condition that failed -- ``"spin lift"`` where members keeping
    every site exist and none carries the lift within the gates, ``"site
    permutation"`` otherwise, the members carrying the lift, if any, carrying
    the sites elsewhere -- with ``spin_lift`` marking the first.  An operation
    with no measured space, or a space past the read-off's own ceiling, is
    absent from both: nothing was measured, and it is read as it stands.
    """
    syms = group.symmetries
    nstate = int(model.nstate)
    spinful = ((int(model.nspin) == 2 or spin is not None)
               and bool(getattr(group, "spinor", True)))
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    sites = _multiplets_of(model, assignment, spin, tuple((orbs,) for orbs in site_orbs),
                           tol=tol)
    spinors = None
    if spinful:
        spinors = ([[(2 * o, 2 * o + 1) for o in orbs] for orbs in site_orbs]
                   if int(model.nspin) == 2 else
                   [[tuple(int(x) for x in a) for a in site]
                    for site in _atoms_of_site(site_orbs, spin)])
    in_blocks = noise_floor(_IN_BLOCKS, tol)
    form_gate = noise_floor(ANSATZ_GATE, tol)
    spaces, records = {}, {}
    for p in positions:
        sym = syms[p]
        ind = int(sym.ind)
        basis = (getattr(validated, "_bases", None) or {}).get(ind)
        if basis is None or not len(np.asarray(basis)):
            continue
        basis = np.asarray(basis, dtype=complex)
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            continue
        image = _channel_permutation(assignment, p, model, sites)
        blocks = _ansatz_blocks(sites._chan_states, image, 1)
        if spinful:
            S = _lift_of(sym, True)
            pieces = [(np.asarray(to, dtype=int), np.asarray(of, dtype=int))
                      for s in range(len(site_orbs)) for to in spinors[int(image[s])]
                      for of in spinors[s]]
            lifts = [S] * len(pieces)
        else:
            pieces, lifts = blocks, [None] * len(blocks)
        # the members of the form: what no member of it carries is the
        # residual, read once over the whole space, and a member is the form's
        # where its residual is within the gate of its own norm -- the defect
        # every test of tensor structure reads, at their gate, so that a model
        # whose matrices miss the form by its own numerical noise keeps them
        A = np.stack([_ansatz_residual(basis[j], pieces, lifts).reshape(-1)
                      for j in range(len(basis))], axis=1)
        _u, s, vh = np.linalg.svd(A, full_matrices=False)
        top = float(s[0]) if s.size else 0.0
        members = np.einsum("ij,jab->iab", vh.conj(), basis)
        norms = np.linalg.norm(members.reshape(len(members), -1), axis=1)
        keep = [i for i in range(len(s)) if float(s[i]) <= form_gate * float(norms[i])]
        if len(keep) == len(s):
            V = basis                         # every member is of the form
        else:
            V = members[keep] if keep else basis[:0]
        held = bool(len(V)) and _filled_check(V, blocks, tol=tol)[0]
        X = members[-1]
        norm = float(norms[-1])
        scaled = float(s[-1]) / norm if norm > 0 else float("inf")
        rec = {"ind": ind, "dim": int(len(V)) if held else 0, "defect": float(s[-1]),
               "scale": top, "norm": norm, "scaled": scaled, "orientation": 1,
               "branches": {1: scaled}, "realizable": held, "note": None,
               "own_form": None, "spin_lift": False}
        records[ind] = rec
        if held and covariance is not None and scaled > FLAT_ZERO:
            # of the form only within the gate: the member nearest it, its part
            # in the form, held to the covariance gate the operation was kept
            # at, as below where no member is within the gate -- spin blocks
            # written in a frame turned against the lattice by a small angle
            # miss the form within any gate on it, and carry H(k) onto H(gk)
            # with the operation's own lift only as far as the turn allows
            part = X - _ansatz_residual(X, pieces, lifts)
            residual = float(covariance(p, part))
            rec["covariance"] = residual
            if residual > float(validated.tol):
                # the condition that fails: the spin lift where the member's
                # part carrying each site's states onto its image's, with any
                # spin factor, carries H(k) within the gate, else the sites
                in_sites = X - _ansatz_residual(X, blocks, [None] * len(blocks))
                lift = spinful and float(covariance(p, in_sites)) <= float(validated.tol)
                rec.update(realizable=False, dim=0,
                           own_form="spin lift" if lift else "site permutation",
                           spin_lift=lift,
                           note=("its own form carries H(k) onto H(gk) only to "
                                 f"{residual:.3g}, beyond the {float(validated.tol):.3g} it was "
                                 "kept at"
                                 + ("; with a spin factor other than its own lift S(g), its "
                                    "matrices carrying each site's states onto its image's "
                                    "carry it within that" if lift else "")))
                spaces[ind] = basis[:0]
                continue
        if held:
            spaces[ind] = V
            continue
        if not len(V) and covariance is not None:
            # no member of the form within the gate: the member nearest it, its
            # part in the form, held to the covariance gate the operation was
            # kept at -- where that part is a matrix of the operation at all,
            # nonsingular on every site block; a part the form leaves empty or
            # singular is no member of it, whatever its nearest unitary carries
            part = X - _ansatz_residual(X, pieces, lifts)
            if _filled_check(part[None], blocks, tol=tol)[0]:
                residual = float(covariance(p, part))
                rec["covariance"] = residual
                if residual <= float(validated.tol):
                    rec.update(realizable=True, dim=1)
                    spaces[ind] = (part * (np.sqrt(nstate)
                                           / float(np.linalg.norm(part))))[None]
                    continue
        # refuted: which condition fails -- the site permutation, unless members
        # keeping every site exist and none carries the operation's own lift
        site_space = _block_members(basis, blocks, tol=in_blocks)[0]
        keeps_sites = bool(len(site_space)) and _filled_check(site_space, blocks, tol=tol)[0]
        carries_lift = False
        if spinful and keeps_sites:
            lifted = _spin_lift_defects(model, group, validated, [p], spin=spin, tol=tol,
                                        orients=(1,))
            carries_lift = bool(lifted.get(ind, {}).get("realizable", False))
        condition = "spin lift" if (spinful and keeps_sites and not carries_lift) \
            else "site permutation"
        if condition == "spin lift":
            note = ("no member of its solution space carries this operation's own spin "
                    "lift S(g)")
        elif spinful:
            note = ("no member of its solution space carries each site's states onto its "
                    "image's under this operation with its own spin lift: the members that "
                    "carry the lift carry the sites elsewhere, as another operation does")
        else:
            note = ("no member of its solution space carries each site's states onto its "
                    "image's under this operation: its matrices are another operation's")
        if len(V):
            note = "every member of its solution space of its own form is singular"
        rec.update(realizable=False, dim=0, own_form=condition,
                   spin_lift=condition == "spin lift", note=note,
                   **({"defect": 1.0, "scaled": 1.0} if len(V) else {}))
        spaces[ind] = basis[:0]
    return spaces, records


def _spin_lift_defects(model, group, validated, positions, *, spin=None, tol=None,
                       orients=(1, -1)):
    r"""Per operation: does its solution space hold a member carrying its own spin lift?

    Every symmetry matrix of a spinful crystal has the form
    ``D(g) = O(g) ⊗ S(g)``: an orbital factor, and on the spin the operation's
    own SU(2) lift in pythtb's convention -- the tables' lift carried onto the
    model's cell by the computed frame rotation (:func:`_lift_of`), the one
    :func:`.._build._build_rep` builds every representation with.  The detector assumes
    no such structure: it finds every matrix that carries ``H(k)`` onto
    ``H(gk)``.  So a Hamiltonian whose spin blocks are written in a frame
    turned against its lattice passes it for operations it obeys only through
    a spin factor other than their lift, and this is where that is measured.

    With the spin pairing explicit the question is linear.  A member ``X`` of
    the solution space has the form ``O ⊗ S(g)``, for some orbital factor
    ``O``, exactly where ``X (1 ⊗ S(g))†`` commutes with ``1 ⊗ σ_a`` for
    ``a = x, y, z``, that is where it equals its own spin trace times the
    identity on the spin.  The space is projected onto that form once, and
    ``scaled`` is the norm of what the closest member could not put in it,
    relative to that member's own norm: ``0`` where the lift is carried and
    of order one where it is not.  No orbital structure is asked for -- no
    multiplets, no site permutation, any orbital factor at all -- so the test
    reads the spin alone, for a multiplet of any dimension, and a model with
    no spin-orbit coupling passes it for every operation it obeys.  That is
    also what tells a layer's ``σ_h`` pair apart where a site's orbitals are
    bound into a multiplet wider than one: the two members share one solution
    space and differ only in their lift.  A member of
    that form is a symmetry matrix only where it is nonsingular, which is
    measured on fixed pseudo-random combinations of the members in the form.

    Both orientation branches are measured where ``orients`` asks for them,
    the ``g⁻¹`` branch carrying ``S(g)†`` as the build reads it, and an
    operation passes where either does, as in :func:`_ansatz_defects`; an
    antiunitary operation's lift is its unitary part's times time reversal's
    ``iσ_y``, as :func:`_lift_of` gives it.  ``positions`` index
    ``group.symmetries``.  The states are paired as pythtb stores a spinful
    model's, ``2·orbital + spin``, or as the declared
    :class:`..spin.SpinBasis` ``spin`` pairs them; a spinless model with no
    declared basis has no spin factor to test, and nothing is returned for it.
    ``tol`` is the declared tolerance, relative to ``max|H|``, that every
    threshold here reads (:func:`.._tolerance.noise_floor`): the gate on ``scaled`` is
    :data:`ANSATZ_GATE` or it, whichever is larger, and ``None`` keeps the
    exact defaults.

    Returns ``{ind: record}`` in the fields :func:`_ansatz_defects` returns --
    ``dim``, ``defect``, ``scale``, ``norm``, ``scaled``, ``orientation``,
    ``branches``, ``realizable`` and ``note`` -- each record marked
    ``spin_lift``.  An
    operation with no measured space, a single member of a larger space, or a
    space too large to project is reported untested and realizable: nothing
    was measured, so nothing is refuted.
    """
    if int(model.nspin) == 2:
        order = np.arange(int(model.nstate))
    elif spin is not None:
        order = np.asarray(spin.interleaved_index, dtype=int)
    else:
        return {}
    if not bool(getattr(group, "spinor", True)):
        return {}
    gate = noise_floor(ANSATZ_GATE, tol)
    singular = noise_floor(SINGULAR, tol)
    npair = int(order.size) // 2
    syms = group.symmetries
    out = {}
    for p in positions:
        sym = syms[p]
        ind = int(sym.ind)
        rec = {"ind": ind, "dim": 0, "defect": float("nan"), "scale": float("nan"),
               "norm": float("nan"), "scaled": float("nan"), "orientation": 0,
               "branches": {}, "realizable": True, "note": None, "spin_lift": True}
        out[ind] = rec
        basis = getattr(validated, "_bases", {}).get(ind)
        if basis is None or len(np.asarray(basis)) == 0:
            member = validated._intertwiners.get(ind)
            if member is None:
                rec["note"] = ("no solution of this operation's condition is carried, "
                               "so its spin lift was not tested")
                continue
            if int((getattr(validated, "_nullspace_dims", None) or {}).get(ind, 1)) != 1:
                rec["note"] = ("only one member of a larger solution space is carried, "
                               "so its spin lift was not tested")
                continue
            basis = np.asarray(member)[None, :, :]
        basis = np.asarray(basis, dtype=complex)
        nstate = int(basis.shape[1])
        if basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            rec["note"] = (f"the solution space is {basis.shape[0]}-dimensional at "
                           f"nstate = {nstate}, whose projection would allocate more "
                           f"than {_READOFF_MAX_BYTES / 1e6:.0f} MB, so its spin lift "
                           "was not tested")
            continue
        # the space in the pairing's own order, (pair, spin) fast, and an
        # orthonormal basis of it: a unit coefficient vector is a unit member
        flat = basis[:, order][:, :, order].reshape(len(basis), -1).T
        u, sv, _vh = np.linalg.svd(flat, full_matrices=False)
        cut = noise_floor(ZERO, tol) * max(float(sv[0]), 1e-300) if sv.size else 0.0
        u = u[:, sv > cut] if sv.size else u[:, :0]
        dim = int(u.shape[1])
        rec["dim"] = dim
        if not dim:                               # pragma: no cover - a guard, not a path
            rec["note"] = "the measured solution space is empty"
            continue
        members = u.T.reshape(dim, npair, 2, npair, 2)
        lift = _lift_of(sym, True)
        best, singular_only = None, False
        for orient in tuple(orients):
            S = lift.conj().T if orient < 0 else lift
            # X (1 (x) S^dagger), its spin trace, and what no orbital factor carries
            W = np.einsum("jasbu,tu->jasbt", members, S.conj())
            O = 0.5 * np.einsum("jasbs->jab", W)
            R = W - np.einsum("jab,st->jasbt", O, np.eye(2))
            _u2, s, vh = np.linalg.svd(R.reshape(dim, -1).T, full_matrices=False)
            scaled = float(s[-1])
            # the members in that form, nonsingular on a pseudo-random combination
            inside = [k for k in range(len(s)) if float(s[k]) <= gate]
            regular = False
            rng = np.random.default_rng(17)
            for _ in range(3 if inside else 0):
                y = (rng.normal(size=len(inside))
                     + 1j * rng.normal(size=len(inside)))
                coef = np.einsum("k,kj->j", y, vh[inside].conj())
                o_sv = np.linalg.svd(np.einsum("j,jab->ab", coef, O), compute_uv=False)
                if float(o_sv[-1]) > singular * float(o_sv[0]):
                    regular = True
                    break
            rec["branches"][int(orient)] = scaled
            key = (not (bool(inside) and regular), scaled)
            if best is None or key < best:
                best = key
                singular_only = bool(inside) and not regular
                rec.update(defect=scaled, scale=float(s[0]), norm=1.0, scaled=scaled,
                           orientation=int(orient))
        if best is None or not best[0]:
            continue
        if singular_only:
            # in the form, and singular in every member that is: no symmetry
            # matrix of this operation carries its lift
            rec.update(realizable=False, defect=1.0, scaled=1.0,
                       note=("every member of the solution space that carries this "
                             "operation's own spin lift is singular"))
        else:
            rec["realizable"] = False
    return out
