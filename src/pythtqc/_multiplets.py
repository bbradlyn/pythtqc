"""The multiplets the Hamiltonian binds each site's orbitals into, and their blocks.

Both are read off the operations' solution spaces.  An operation's solution
space is every matrix ``X`` with ``X H(k) = H(Rk) X`` on the grid, or
``X H(k)* = H(-Rk) X`` for an operation combined with time reversal, as the
detector finds it (``ValidatedGroup._bases``).  A matrix of the operation's
own form carries each site's states onto its image's, with any change of
basis within the site, and on states that carry a spin it is ``O ⊗ S(g)``,
``S(g)`` the operation's own spin lift (:func:`.._matrices._lift_of`).  A
multiplet's block is what such a matrix carries from the multiplet onto its
image.
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
    """Multiplets stated by ``site_multiplets=``, each given to the site its orbitals sit on.

    ``declared`` is the statement, every site's multiplets as their orbitals,
    read as one partition of the model's orbitals whatever site each list
    stands at.  ``site_orbs`` holds the orbitals of each site of this
    assignment, ``site_of`` each orbital's site, and ``atoms`` each site's
    atoms, its orbitals or its declared spinor pairs.  Returns, per site, its
    multiplets in the order of their first orbitals.  A malformed statement,
    or one that splits a spinor pair or spans two sites, raises
    ``ValueError`` (:func:`.._grammar._multiplets_grammar`); the call that took
    it has checked that before measuring anything.
    """
    pairs = tuple(tuple(int(o) for o in atom) for site_atoms in atoms
                  for atom in site_atoms if len(atom) == 2)
    chans = _multiplets_grammar(declared, len(site_of), pairs, site_of)
    return tuple(tuple(sorted((ch for ch in chans if int(site_of[ch[0]]) == s),
                              key=lambda c: c[0]))
                 for s in range(len(site_orbs)))


def _slot_order(model, group, assignment, atoms, spaces, *, tol=ZERO):
    r"""Each site's atoms in the order of the atoms they are images of at its orbit's reference site.

    An orbit's reference site is its lowest-numbered site.  An operation
    carrying the reference site onto site ``t`` links each atom there to the
    atoms of ``t`` its solution space puts weight on, and linked atoms share a
    slot.  Atoms in one linked set are paired in the order each site lists
    them; the multiplet they land in is decided by the operations that fix
    the site.  Of the operations reaching ``t`` whose every set holds as many
    atoms of one site as of the other, the one with the most sets is read.  A
    site no such operation reaches keeps its own order.  A weight at most
    ``tol`` times the largest links nothing.

    ``model``, ``group`` and ``assignment`` are the model, its group and the
    orbitals' assignment to it.  ``atoms[s]`` is site ``s``'s atoms in its own
    order, each an orbital or a declared spinor pair.  ``spaces`` maps an
    operation's index to its solution space within its site blocks
    (:func:`_site_spaces`).  ``tol`` is :data:`.._tolerance.ZERO`, or the
    declared tolerance where larger.  Returns the atoms per site, in slot
    order.
    """
    nspin = int(model.nspin)
    orbit_of = np.asarray(assignment.orbit_of_site, dtype=int)
    perms = np.asarray(assignment.permutations, dtype=int)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group._symmetries)}
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
    r"""The multiplets the Hamiltonian binds each site's orbitals into, measured.

    A site's multiplets are the finest partition of its atoms, its orbitals
    or declared spinor pairs, for which every operation the partition is
    about still has a nonsingular member of its solution space in block form:
    block-diagonal on the multiplets of every site it fixes, and carrying the
    multiplets of a site onto those of its image.  The operations it is about
    are those the whole site as one multiplet makes realizable
    (:func:`_ansatz_defects`).  The partition is refined from singletons, one
    merge at a time: a block an operation's space cannot fill merges that
    multiplet with one the space couples it to at the image site.  The merges
    asked by operations at a site they fix are made first.  An operation that
    moves a site merges only multiplets that carry copies of one site irrep,
    by their characters under the site group.  Atoms of a site that each hold
    part of one sector of the conserved matrices (:mod:`._sectors`) are one
    multiplet, and so are identical copies.  The sites of an orbit share one
    partition, matched slot by slot (:func:`_slot_order`).  Where two
    partitions are equally fine, the lowest-indexed partner is taken and the
    tie recorded in ``accidental``.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : ..groups.GroupData
        The group measured on it.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites and each operation's permutation of the sites.
    spin : ..spin.SpinBasis or None
        The declared spinor pairs, which no multiplet splits.
    validated : ..groups.ValidatedGroup or None
        The verdicts, ``_bases`` holding each kept operation's solution space.
    declared : SiteMultiplets or sequence, default None
        Multiplets stated by ``site_multiplets=``: a :class:`SiteMultiplets`,
        or each site's multiplets as their orbitals.
    tol : float, default None
        The declared tolerance, relative to ``max|H|``.  Every threshold here
        that measures noise is the larger of its exact default and ``tol``
        (:func:`.._tolerance.noise_floor`).  ``None`` keeps the exact defaults.
    sectors : .._sectors.SectorReading, default None
        The sectors of the conserved matrices
        (:func:`.._sectors.read_sectors`), read here where none is given.

    Returns
    -------
    SiteMultiplets
        The multiplets, carrying ``tol`` for every later test of their blocks.
        ``_mixing`` holds how far outside their block form the members they
        were read off reach, and, where the multiplets are measured, how far
        a sector of the conserved matrices reaches between two multiplets of
        one site: rounding where the Hamiltonian holds them exactly, more
        where it holds them only within ``tol``.  With ``declared``, the
        multiplets are the ones stated, and a :class:`SiteMultiplets` about
        the same sites is returned as it is.  Every atom is its own multiplet
        where nothing is declared and ``validated`` is ``None`` or holds no
        space, or where the whole-site test returns a refusal.
    """
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    atoms = _atoms_of_site(site_orbs, spin)
    pos_of_ind = {int(s.ind): i for i, s in enumerate(group._symmetries)}
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
        # Whether the declared multiplets hold exactly or only within the
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
    # The site blocks, once: they are the operation's own permutation of the
    # sites and no partition of a site's orbitals changes them, so the space
    # each operation is asked about is restricted to them before the
    # refinement starts.
    spaces, site_layout, site_mixing = _site_spaces(model, group, assignment, spin,
                                                    validated, tol=tol)
    # The slots: which atom of every site answers to which of its orbit's
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
    # No basis splits what the conserved matrices join.  The sectors of the
    # matrices that commute with H at every k and act within sites are sets of
    # states, whatever basis the orbitals are written in, and two orbitals of a
    # site that each hold part of one sector are one multiplet: a square's
    # p_x, p_y pair nothing couples, written as (p_x +- p_y)/sqrt(2), where the
    # fourfold's solution space holds a diagonal member and no operation would
    # bind the two.  Made first, and carried round the orbit as every merge is.
    if sectors is None:
        sectors = _sectors.read_sectors(model, group, assignment, spin, validated,
                                        tol=tol or None)
    # And identical copies are one multiplet: where the conserved matrices
    # exchange copies of one set of states, which representation the copies
    # carry together is the freedom the hoppings leave, and a declaration that
    # mixes them, as two identical s orbitals declared the doublet E, has to be
    # readable on them.  So every state of a copy is one multiplet with the
    # same state of every other.  On states carrying a spin the copies are read
    # in the orbital factor, every copy carrying the operation's own spin lift.
    joins, joining = {}, noise_floor(_IN_BLOCKS, tol)
    if sectors is not None:
        for s, site_atoms in enumerate(atoms):
            joins[s] = _sectors.joined(sectors, site_atoms)
            for i in range(len(site_atoms)):
                for j in range(i + 1, len(site_atoms)):
                    if float(joins[s][i, j]) > joining:
                        merged.union((int(orbit_of[s]), i), (int(orbit_of[s]), j))
    # Which operations the multiplets are about.  Gluing a site's orbitals can
    # only ever help an operation that the coarsest partition, the whole site
    # as one multiplet, makes realizable.  An operation not even that can carry
    # is refuted by its own tensor structure (a lattice supergroup's coset
    # partner, a sigma_h partner), and answering it by gluing orbitals would
    # put a partition in place of a verdict.  How far a member may sit from
    # that form and still be realizable is noise, so it reads the declared
    # tolerance too: a model symmetric only within its precision has every
    # member that far off.
    positions = [pos_of_ind[int(i)] for i in validated._indices
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
    syms = group._symmetries
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
        """One operation's verdict on one partition, ``(p, image, unfilled)``.

        ``layout`` is the partition and ``ind`` the operation's index; ``p``
        is its position in the group and ``image`` where it carries each
        multiplet.  ``unfilled`` lists the blocks its space cannot fill, empty
        where a nonsingular member in block form exists
        (:func:`_unfilled_blocks`).  ``asked`` holds the verdicts on
        ``layout`` measured so far, each with how far its members in block
        form reach outside the blocks, so that each is measured once.
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
        """Whether any verdict on ``parts`` asks for a merge; the first is made.

        ``parts`` is each site's multiplets.  The merges asked by operations at
        a site they fix are made first; where none is left, a merge asked
        across the sites of an orbit is made, only between multiplets that
        carry copies of one site irrep (``_one_irrep``).  The partner is the
        lowest-indexed multiplet of the image site that the space couples to
        the unfilled one within half the largest such weight; the others so
        coupled are kept as a possible tie.  The partition swept and its
        verdicts are kept in ``last``.
        """
        layout = _multiplets_of(model, assignment, spin, parts, order=order)
        asked = {}
        last.clear()
        last.update(parts=parts, layout=layout, asked=asked)

        def _one_irrep(orbit, ja, jb):
            """Whether the multiplets of slots ``ja`` and ``jb`` carry copies of one site irrep.

            Asked at every site of orbit ``orbit``, under each unitary
            operation that fixes the site: the two multiplets' characters,
            each its block's trace per orbital with the spin lift divided out,
            agree in every member of the operation's solution space that
            reaches the site.  Both are read in the same member, so that its
            free phase is common to both.
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
                # another one, and which one the space itself says: the
                # multiplets of the image site it does put weight on.
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
                        # several partners couple within half the largest
                        # weight: the merge taken is the lowest-indexed one and
                        # the rest are recorded, to be kept as ties only where
                        # the multiplets end with them still apart
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
    # Whether the multiplets hold exactly, or only within the declared
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
    """Where each operation carries each multiplet, ``{p: image}``.

    ``positions`` are positions in ``syms``, the group's operations.
    ``image[c]`` is the multiplet of ``layout`` that operation ``p`` carries
    multiplet ``c`` onto (:func:`_channel_permutation`), read off the sites'
    permutation in ``assignment`` and the slot order of the orbitals of
    ``model``.  With a declared ``spin``, each operation is also checked to
    carry every spinor pair onto a pair (:func:`_spin_channel_permutation`).
    One that does not raises ``InternalError``: each site's orbitals are read
    pair by pair, so a pair carried off a pair is the package's own reading
    failing.
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
                f"{cu} and orbital {dn} into pair {cd}. Report it, with the model and "
                f"the call.")
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
    operation.  Within ``noise`` of a root of unity of order dividing twelve
    (``_TWELFTH_ROOTS``), it is returned as that root exactly.  Any other
    ``z`` whose modulus is within ``noise`` of one is returned as its phase,
    ``z / |z|``: a phase the orbitals themselves carry, an orbital written as
    ``e^{iφ}`` times its partner's image.  Returns ``None`` where the modulus
    of ``z`` misses one by more than ``noise``.
    """
    z = complex(z)
    if not abs(abs(z) - 1.0) <= float(noise):
        return None
    root = _TWELFTH_ROOTS[int(round(float(np.angle(z)) * 6.0 / np.pi)) % 12]
    return root if abs(z - root) <= float(noise) else z / abs(z)


def _factor_said(z) -> str:
    """The factor ``z`` between two orbits of multiplets, in words.

    A root of ``_TWELFTH_ROOTS`` is named, as ``i`` or ``e^(2πi·1/3)``; any
    other value is printed to six significant digits.
    """
    z = complex(z)
    if z in _TWELFTH_ROOTS:
        k = _TWELFTH_ROOTS.index(z)
        n = int(np.gcd(k, 12))
        return {0: "1", 3: "i", 6: "-1", 9: "-i"}.get(k, f"e^(2πi·{k // n}/{12 // n})")
    return f"{z:.6g}"


def _reversal_scalars(basis, blocks, lifts, classes, ref_chan, *, tol=_IN_BLOCKS):
    r"""The scalar time reversal carries on each orbit of multiplets, as its solution space pins it.

    Time reversal's matrix is its own factor ``lifts[c]`` on each multiplet
    ``c``, times one scalar per orbit of multiplets.  Where the orbitals are
    real functions every scalar is one; an orbital written as ``i`` times a
    real function has ``-1``.  The members of ``basis``, time reversal's
    solution space, that are in that form within ``tol`` of the space's scale
    (:func:`_ansatz_residual`) each carry a scalar on every orbit.  Orbits
    whose vectors of scalars over these members are parallel are pinned to
    one another at their ratio.  The reference orbit, and the lowest orbit of
    each set pinned to none before it, take one.  A ratio within
    ``max(SNAP, 10·tol)`` of a sign is that sign.

    ``blocks`` holds the rows and columns of each multiplet's block
    (:func:`_ansatz_blocks`), ``classes`` each multiplet's orbit, and
    ``ref_chan`` a multiplet of the reference orbit.  Returns one unit scalar
    per orbit, indexed by orbit.  Returns ``None`` where no member is in the
    form, the members carry nothing on some orbit, or a pinned ratio is not
    of unit modulus; time reversal is then assembled with one on every orbit.
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
    r"""The orbits of sectors whose representations are built by induction, as records.

    These are the orbits of sectors of the conserved matrices, under the
    operations ``kept``, of three kinds (:func:`.._sectors.induced_orbits`):
    those whose sectors hold identical copies; those of several sectors of
    one copy that one multiplet holds together; and those of several sectors
    of one copy that an orbit of multiplets crosses
    (:func:`.._sectors.crossed`), where a sector's stabilizer has a
    one-dimensional character that no character of the group restricts to.
    They are read only where the multiplets hold each sector's copies
    together.
    An orbit's candidates, the representations of it the hoppings leave open,
    are counted before any is built, and the first is checked against the
    group's multiplication table and every operation's solution space
    (:meth:`.._sectors.InducedOrbit.check`).  On states that carry a spin the
    orbits are read on the orbital factor, each matrix carrying the
    operation's own spin lift, so identical copies there carry an ordinary
    representation.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : ..groups.GroupData
        The group measured on it.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites and each operation's permutation of the sites.
    spin : ..spin.SpinBasis or None
        The declared spinor pairs.
    validated : ..groups.ValidatedGroup
        The verdicts the sectors are read from.
    layout : SiteMultiplets
        The multiplets, carrying the declared tolerance every threshold reads.
    kept : list of int
        The positions in the group of the operations represented.
    cperm : dict
        Where each of them carries each multiplet (:func:`_channel_perms`).
    bases : dict
        Each operation's solution space, by index, which the first candidate
        is checked against.
    ref_chan : int, default None
        The multiplet an orbit is read at, where the orbit holds it.
    sectors : .._sectors.SectorReading, default None
        The sectors read where the model entered, read here where none is
        given.  Sectors read at a declared tolerance are read again on
        ``model``, which may have been projected since.
    max_candidates : int, default 2**20
        The most candidates an orbit of several copies, or the orbits read
        together, may have.

    Returns
    -------
    list of .._sectors._OrbitRecord or Refusal
        One record per orbit, or per set of orbits read together
        (:meth:`.._sectors.InducedOrbit.record`), whose first candidate passes
        the check, or that holds identical copies and admits no candidate.
        Every other orbit is left to the multiplets' own blocks.  Empty where
        no orbit is built by induction.

    Refusals
    --------
    ``rep_search_too_large``
        the candidates exceed ``max_candidates``.
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
        # this model: read where a precision was declared, the model handed in
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
                "that the Hamiltonian does not couple and no operation exchanges. There "
                f"are {many.count:,} ways of choosing a representation for each of these "
                f"sets, more than max_candidates = {int(max_candidates):,}. Declare what "
                "the orbitals are with `characters=`, or pass a larger `max_candidates=`.",
                value=float(many.count), threshold=float(max_candidates),
                context={"multiplet": tuple(many.orbitals), "sets": int(many.sets),
                         "n_candidates": int(many.count),
                         "max_candidates": int(max_candidates)})
        return Refusal(
            "rep_search_too_large",
            f"the Hamiltonian conserves a {_fold(many.copies)} pseudospin on orbitals "
            f"{many.orbitals}. There are {many.count:,} representations its copies "
            f"could carry, more than max_candidates = {int(max_candidates):,}. Declare "
            "what the orbitals are with `characters=`, or pass a larger `max_candidates=`.",
            value=float(many.count), threshold=float(max_candidates),
            context={"multiplet": tuple(many.orbitals), "multiplicity": int(many.copies),
                     "n_candidates": int(many.count), "max_candidates": int(max_candidates)})
    gate = noise_floor(ASSEMBLED, eps)
    out = []
    for orbit in orbits:
        if orbit is None:
            continue
        record = orbit.record()
        if not record.n_admitted and int(record.dim) < 2:
            continue
        if record.n_admitted and orbit.check(record.candidates[0]["W"], bases) > gate:
            continue
        out.append(record)
    return out


def _orbit_blocks(basis, image, chan_states, c0, members, Y, *, identity=False, tol=ZERO):
    r"""One operation's block at every multiplet of a Wyckoff orbit, ``{c: block}``.

    The block at the reference multiplet ``c0`` is ``Y``, a block of the
    operation's solution space at ``(image[c0], c0)``.  The block at each
    other multiplet ``c`` is read at ``(image[c], c)`` off the same member of
    the space, the one whose block at the reference is ``Y``, and made
    unitary where it can be: it is the operation acting in that site's own
    frame.  Where that member's block at ``c`` is at most ``tol`` times the
    scale of ``Y``, as at a multiplet the Hamiltonian decouples from the
    reference, the largest block the space carries there is taken
    (:func:`_block_space`).

    ``basis`` is the operation's solution space, ``image`` where it carries
    each multiplet, ``chan_states`` each multiplet's states, and ``members``
    the multiplets of the orbit, ``c0`` among them.  With ``identity``, the
    operation is the identity and ``Y`` is returned at every multiplet.
    ``tol`` is :data:`.._tolerance.ZERO`, or the declared tolerance where
    larger.
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
    r"""A class of one-dimensional multiplets carried round its orbit, ``{p: {c: S}}``.

    A ``p`` orbital along each site's own fourfold axis, the axes along
    different cube edges, is such a class: an operation carrying one site
    onto the next carries it onto plus or minus the one there.  Each
    operation's block at each multiplet of the class is its spin lift (one,
    spinless) times a scalar, read off its solution space relative to the
    scalar at ``c0``, and taken only where every member of the space gives
    the same ratio.  A ratio within :data:`.._tolerance.SNAP` of ``±1`` or
    ``±i`` (``_TRANSPORT_SIGNS``), or within the declared tolerance where
    larger, is taken as that value, any other as its phase.  What is read is
    taken only where it is a representation: the unitary operations read
    close under the group's product, and any two that commute carry every
    multiplet alike in either order.  A sign per site that is not a
    representation is a gauge, as a flux threading the cell is, and not a
    symmetry.

    Parameters
    ----------
    syms : list
        The group's operations, ``group._symmetries``.
    kept : list of int
        The positions in ``syms`` of the operations represented.
    cperm : dict
        Where each of them carries each multiplet (:func:`_channel_perms`).
    layout : SiteMultiplets
        The multiplets, carrying the declared tolerance every threshold reads.
    members : list of int
        The multiplets of the class.
    c0 : int
        Its reference multiplet, which keeps the tables' lift.
    bases : dict
        Each operation's solution space, by index.
    intertwiners : dict
        One member of each operation's space, by index, read where ``bases``
        holds none.
    prod : ndarray
        The group's multiplication table, by position.
    ident : int
        The position of the identity, which carries every multiplet by its
        lift.
    tol : float, default SPREAD
        How far a block may miss the lift times a number, and the ratios
        spread over the members, relative to their scale; the declared
        tolerance where larger.

    Returns
    -------
    dict or None
        ``{p: {c: S}}``, each block the lift times the scalar read.  ``None``
        where a block is no multiple of the lift, fewer than two unitary
        operations are read, a product of two of them is not read, or what
        is read is no representation; every multiplet then keeps the tables'
        lift.
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

    Returns the first of the one-dimensional multiplets ``ones`` the hoppings
    couple to multiplet ``c0``, directly or through others.  It is read off
    ``commutant``, the identity's solution space, every matrix commuting with
    ``H``, whose members carry one scalar per coupled part of the crystal:
    two multiplets of one part carry proportional scalars on every member,
    within ``tol`` of their scale.  ``chan_states`` holds each multiplet's
    states.  Where ``commutant`` is not at hand the first of ``ones`` is
    returned.
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
    r"""Every operation's block at every multiplet, the model's own, ``{ind: {c: S}}``.

    A one-dimensional multiplet's block is the tables' spin lift (one,
    spinless) times the sign the model's own transport gives it relative to
    its orbit's reference multiplet (:func:`_transported_lifts`).  A wider
    multiplet's block at the reference multiplet of its orbit is the block of
    the operation's stored member (``validated._intertwiners``), or, where
    that block vanishes, the largest block its solution space carries there
    (:func:`_block_space`), made unitary.  It is put in the phase in which
    the stored member carries the tables' lift on a one-dimensional multiplet
    the hoppings couple it to, where there is one (:func:`_phase_multiplet`).
    The blocks at the other multiplets of the orbit are read off the same
    member of the space (:func:`_orbit_blocks`), so the representation on the
    orbit is the one induced from the reference site's, in the model's own
    frames.  The identity's block is the identity.  The orbits of sectors
    :func:`_induced_blocks` builds by induction are written in their first
    candidate, where they admit one.  The spaces read are each operation's
    own form (:func:`_own_form_spaces`), so on states that carry a spin the
    spin factor is the operation's own lift.  No orbital function enters:
    the site irreps are read off these blocks' characters afterwards
    (:func:`.._declaration._multiplet_characters`).  Whether the blocks make
    a representation is checked by :func:`.._build._build_rep`.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : ..groups.GroupData
        The group measured on it.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites and each operation's permutation of the sites.
    spin : ..spin.SpinBasis or None
        The declared spinor pairs.
    validated : ..groups.ValidatedGroup
        The verdicts: ``_bases`` holds each operation's solution space and
        ``_intertwiners`` one member of it.
    layout : SiteMultiplets
        The multiplets, carrying in ``_tol`` the declared tolerance every
        threshold here reads where it is larger than the default.
    positions : sequence of int
        The positions in the group of the operations whose blocks are built.
    ref_chan : int, default None
        The reference multiplet, the first orbital's where ``None``.
    tol : float, default ZERO
        The scale, relative to the largest, at which a block of a solution
        space counts as zero; the declared tolerance where larger.
    max_candidates : int, default 2**20
        As in :func:`_induced_blocks`.
    sectors : .._sectors.SectorReading, default None
        As in :func:`_induced_blocks`.

    Returns
    -------
    _MultipletBlocks or Refusal
        The blocks by operation index and multiplet, with the orbits built by
        induction in ``multiplicity``.  A one-dimensional multiplet with no
        block carries the tables' lift.

    Refusals
    --------
    ``rep_search_too_large``
        from :func:`_induced_blocks`.
    """
    syms = group._symmetries
    prod = group._get_product_table(get_diff=False)
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
    # The orbits of sectors built by induction (_induced_blocks), read once
    # where the model entered: each orbit that admits a candidate has its
    # multiplets written in the first, and their classes read no further
    # here.  On states that carry a spin the orbits are read on the orbital
    # factor, each matrix carrying the operation's own lift on the spin.
    done = set()
    got = _induced_blocks(model, group, assignment, spin, validated, layout, kept, cperm,
                          bases, ref_chan=ref_chan, sectors=sectors,
                          max_candidates=max_candidates)
    if isinstance(got, Refusal):
        return got
    for record in got:
        multiplicity.append(record)
        if not record.n_admitted:
            continue
        W = record.candidates[0]["W"]
        local = {int(a): i for i, a in enumerate(record.states)}
        for p in kept:
            for c in record.multiplets:
                rows = [local[int(a)] for a in layout._chan_states[int(cperm[p][c])]]
                cols = [local[int(a)] for a in layout._chan_states[int(c)]]
                out.setdefault(int(syms[p].ind), {})[int(c)] = W[p][np.ix_(rows, cols)]
        record.candidates[0].release()
        done |= {int(classes[int(c)]) for c in record.multiplets}
    for cls in range(ncls):
        if cls in done:
            continue
        members = [c for c in range(int(layout._nchan)) if int(classes[c]) == cls]
        c0 = ref_chan if int(classes[ref_chan]) == cls else members[0]
        d = len(layout._chan_states[c0])
        if layout._chan_kind[members[0]] != "measured":
            # A one-dimensional class keeps the tables' lift at its reference
            # multiplet, and at every other multiplet the sign the model's own
            # transport gives it, read off one member of each solution space,
            # where that is a representation; it keeps the lift everywhere
            # where it is not.
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
                # The identity's block is the identity.  Its solution space is
                # the model's whole commutant, so a polar extraction there
                # returns whichever conserved operator the decomposition hands
                # back, `-1` on a decoupled doublet, which says nothing about
                # the operation: `D(E) = 1` is what a representation means.
                unit = np.eye(d, dtype=complex)
            if unit is not None and unit.shape == (d, d):
                # The phase a measured block is read in.  The member this block
                # came from carries one free phase, and a block extracted with
                # it makes the other multiplets' scalars carry its inverse,
                # which the read-off then cannot find among the two signs the
                # tensor structure offers.  Where the hoppings couple
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
            """Operation ``p``'s blocks round the orbit, off its member with block ``Y`` at ``c0``."""
            return _orbit_blocks(basis_of.get(p), cperm[p], layout._chan_states, c0,
                                 members, Y, identity=(p == ident), tol=tol)

        # Carried round the orbit, never copied.  The sites of one Wyckoff
        # orbit are one another's images, so the block an operation has at the
        # next site is the operation acting in that site's frame, which a copy
        # of the reference's block is only where every site writes its orbitals
        # in the same frame.  The member of the solution space the reference
        # block came from says it at every multiplet at once.
        for p, S in polar.items():
            got = _carried(p, S)
            for c in members:
                out.setdefault(int(syms[p].ind), {})[int(c)] = got[int(c)]
    return _MultipletBlocks(out, multiplicity=multiplicity)


class _MultipletBlocks(dict):
    """Every operation's block at every multiplet, ``{ind: {c: block}}``, and the orbits built by induction.

    ``multiplicity`` holds one record per orbit of sectors built by induction
    (:class:`.._sectors._OrbitRecord`, from :func:`_induced_blocks`): its
    multiplets, orbitals and states, how many identical copies each sector
    holds, how many representations of it there were and were admitted, and
    each admitted one with its character and the character it gives each
    multiplet.
    """

    def __init__(self, blocks=(), *, multiplicity=()):
        super().__init__(blocks)
        self.multiplicity = tuple(multiplicity)


def _ansatz_defects(model, group, assignment, validated, positions, *,
                    layout=None, spin=None, tol=None, orients=(1, -1),
                    blocks=None, max_candidates=2 ** 20):
    r"""Whether each operation's solution space holds a matrix of tensor form, and how far it misses.

    A matrix of tensor form for operation ``g`` is ``O(g) ⊗ S(g)``: it
    carries each multiplet onto its image, holding there that multiplet's
    block ``S(g)`` times one scalar per orbit of multiplets.  On a
    one-dimensional multiplet ``S(g)`` is ``g``'s own spin lift
    (:func:`_lift_of`), carried round the orbit with the signs the model's
    transport gives it; on a wider one it is the model's own block
    (:func:`_multiplet_blocks`).  The solution space is projected onto that
    form and its closest member taken (:func:`_ansatz_member`).  The scaled
    defect is the norm of what that member does not put in the form, over
    its own norm: zero where the form is realized, one where it is refuted
    outright.  An operation is realizable where some branch of ``orients``
    has a scaled defect at most ``tol``.  The spaces read are each
    operation's own form where :func:`_own_form_spaces` has restricted them;
    on a free-standing layer that is what tells ``g`` from ``σ_h·g`` on a
    multiplet wider than one, whose block the two share.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : ..groups.GroupData
        The group measured on it.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites and each operation's permutation of the sites.
    validated : ..groups.ValidatedGroup
        The verdicts: ``_bases`` holds each operation's solution space, and
        ``_intertwiners`` the one member read where it holds none.
    positions : sequence of int
        The positions in ``group._symmetries`` of the operations tested.
    layout : SiteMultiplets, default None
        The multiplets the form is written on, measured here
        (:func:`_site_multiplets`) where none is given.
    spin : ..spin.SpinBasis or None, default None
        The declared spinor pairs.
    tol : float, default None
        The gate on the scaled defect: :data:`.._tolerance.ANSATZ_GATE`, or
        the declared tolerance the multiplets carry where larger, which
        ``None`` reads off ``layout``.
    orients : tuple of int, default (1, -1)
        The branches measured: ``1`` the operation's own candidate, ``-1``
        the branch of ``g``'s inverse, whose candidate is the adjoint and
        carries ``S(g)†`` on the transposed blocks.  Pass ``(1,)`` for an
        operation the detector left undetermined: for a fourfold, ``S(g)†``
        is, up to sign, the lift of its ``σ_h`` partner.
    blocks : _MultipletBlocks, default None
        The multiplets' blocks, built here (:func:`_multiplet_blocks`) where
        none is given.
    max_candidates : int, default 2**20
        As in :func:`_multiplet_blocks`, where the blocks are built here.

    Returns
    -------
    dict or Refusal
        ``{ind: record}``, each record holding ``ind``, ``dim``, ``defect``,
        ``scale``, ``norm``, ``scaled``, ``orientation``, ``branches`` (the
        scaled defect of each branch), ``realizable`` and ``note``.  An
        operation not tested, for want of a space, a block or memory, is
        realizable with ``scaled`` ``nan``, and its ``note`` says why:
        nothing was measured, so nothing is refuted.  One whose space carries
        a multiplet wider than one to its image only by a zero block is
        refuted with ``scaled`` one.

    Refusals
    --------
    ``rep_search_too_large``
        where the blocks are built here (:func:`_multiplet_blocks`).
    """
    if layout is None:
        layout = _site_multiplets(model, group, assignment, spin, validated)
    eps = getattr(layout, "_tol", None)
    if tol is None:
        tol = noise_floor(ANSATZ_GATE, eps)
    syms = group._symmetries
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
    # rather than two.  They are built here only where the call does not hand
    # them in.
    got = blocks if blocks is not None else _multiplet_blocks(
        model, group, assignment, spin, validated, layout,
        [p for p, s in enumerate(syms) if int(s.ind) in set(validated._indices)],
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
            # the polar-extracted member is the space when dim == 1; where it is
            # not, the projection has nothing to choose from and the defect says
            # so, a fallback and not a wrong verdict, as in the read-off.
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
            # must, as on a lattice supergroup, where every coset partner's
            # measured operator is the partner's and never this operation's.
            # So the tensor structure is refuted rather than left untested.
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

    Where several operations share one action on ``k``, as a layer's ``g``
    and ``σ_h·g`` do, or ``H`` conserves more than the crystal does, the
    detector's space for ``g`` holds matrices of other operations, or of
    symmetries beyond the crystal's, too.  A member is of ``g``'s own form
    where it carries each site's states onto its image's, and on states that
    carry a spin carries ``g``'s own spin lift (:func:`_lift_of`).  Both
    conditions are linear, so the members of the form are read off the space
    once: a member is of it where the part no matrix of the form carries is
    within :data:`.._tolerance.ANSATZ_GATE` of its own norm.  The operation
    holds where one of them is nonsingular on every site block
    (:func:`_filled_check`).  The two conditions are imposed together.  In a
    Néel antiferromagnet, time reversal's space holds the members of time
    reversal with a spin rotation, which keep every site, and those of time
    reversal with the half translation, which carry ``iσ_y`` between the two
    sites; none does both.

    With ``covariance``, the form is also held to the detector's own measure.
    Where the members are of the form only within the gate, and not to
    rounding, the nearest member's part in the form must carry ``H(k)`` onto
    ``H(gk)`` within ``validated.tol``, the gate the operation was kept at,
    or the operation is refuted.  Where no member is of the form within the
    gate, that part is kept as the operation's one member where it is within
    ``validated.tol`` and nonsingular on every site block.

    Parameters
    ----------
    model : pythtb.TBModel
        The model.
    group : ..groups.GroupData
        The group measured on it.
    assignment : ..groups.OrbitalAssignment
        The orbitals' sites and each operation's permutation of the sites.
    validated : ..groups.ValidatedGroup
        The verdicts: ``_bases`` holds each operation's solution space, and
        ``tol`` the gate the operations were kept at.
    positions : sequence of int
        The positions in ``group._symmetries`` of the operations read.
    spin : ..spin.SpinBasis or None, default None
        The declared spinor pairs.
    tol : float, default None
        The declared tolerance, relative to ``max|H|``, every threshold here
        reads (:func:`.._tolerance.noise_floor`).  ``None`` keeps the exact
        defaults.
    covariance : callable, default None
        Called with an operation's position and a matrix, it returns the
        matrix's covariance residual over the detector's grid, in the units
        of ``H``.  ``None`` reads the form alone.

    Returns
    -------
    tuple
        ``(spaces, records)``.  ``spaces[ind]`` holds the members of the form
        in the model's own state order: the whole space where every member is
        of it, those members where only some are, the one member kept by
        ``covariance`` scaled to norm ``sqrt(nstate)``, or none where the
        operation is refuted.  ``records[ind]`` holds the fields
        :func:`_ansatz_defects` returns, the ``covariance`` residual where it
        was measured, ``own_form`` and ``spin_lift``.  ``own_form`` is
        ``None`` where the operation holds, and otherwise the condition that
        failed: ``"spin lift"`` where a matrix keeping every site carries
        ``H(k)`` onto ``H(gk)`` and none with the operation's own lift does,
        ``"site permutation"`` otherwise; ``spin_lift`` is true for the
        first.  An operation with no space, or a space past the read-off's
        memory ceiling, is in neither: nothing was measured.
    """
    syms = group._symmetries
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
        # where its residual is within the gate of its own norm.  That is the
        # defect every test of tensor structure reads, at their gate: a model
        # whose matrices miss the form by its own numerical noise keeps them.
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
            # at, as below where no member is within the gate.  Spin blocks
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
            # kept at, where that part is a matrix of the operation at all,
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
        # refuted: which condition fails, the site permutation unless members
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
    r"""Whether each operation's solution space holds a member carrying its own spin lift.

    A member ``X`` is ``O ⊗ S(g)`` for some orbital factor ``O`` exactly
    where ``X (1 ⊗ S(g))†`` is its own spin trace times the identity on the
    spin, ``S(g)`` the operation's own lift (:func:`_lift_of`); an
    antiunitary operation's lift is its unitary part's times ``iσ_y``.  The
    space is projected onto that form once, and ``scaled`` is the norm of
    what its closest unit member does not put in it: zero where the lift is
    carried, of order one where it is not.  No orbital structure is asked
    for, so a model with no spin-orbit coupling passes for every operation it
    obeys.  A member of the form counts only where it is nonsingular, which
    is measured on fixed pseudo-random combinations of the members in the
    form.

    ``validated`` holds each operation's solution space (``_bases``), or one
    member of it (``_intertwiners``), and ``positions`` index
    ``group._symmetries``.  The states are paired as pythtb stores a spinful
    ``model``, ``2·orbital + spin``, or as the declared
    :class:`..spin.SpinBasis` ``spin`` pairs them.  ``tol`` is the declared
    tolerance relative to ``max|H|``: the gate on ``scaled`` is
    :data:`.._tolerance.ANSATZ_GATE` or ``tol``, whichever is larger, and
    ``None`` keeps the exact defaults.  ``orients`` names the branches
    measured, ``-1`` carrying ``S(g)†``, and an operation passes where either
    does.

    Returns ``{ind: record}`` in the fields :func:`_ansatz_defects` returns,
    each record marked ``spin_lift``.  An operation with no space, one member
    of a larger space, or a space too large to project is untested and
    realizable.  Returns ``{}`` for a spinless model with no declared basis,
    or where ``group.spinor`` is false.
    """
    if int(model.nspin) == 2:
        order = np.arange(int(model.nstate))
    elif spin is not None:
        order = np.asarray(spin._interleaved_index, dtype=int)
    else:
        return {}
    if not bool(getattr(group, "spinor", True)):
        return {}
    gate = noise_floor(ANSATZ_GATE, tol)
    singular = noise_floor(SINGULAR, tol)
    npair = int(order.size) // 2
    syms = group._symmetries
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
