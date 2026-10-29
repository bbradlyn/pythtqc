"""The labelling the Hamiltonian leaves open, chosen once by the lowest-angular-
momentum rule: what each coupled part of the crystal carries, the keys read
first to last, and the atomic-limit table read off the choice.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import _bloch
from . import _sectors
from . import _sitechars
from . import _spacegroup
from . import _tables
from .refusal import InternalError, Refusal
from ._tolerance import ASSEMBLED, CHARACTER
from ._sitechars import _L_MAX, _content_shells
from ._matrices import _atom_blocks, _lift_of, _proportionality, bloch_operator
from .symmetry import AtomicLimitRow, AtomicLimitTable, _NOT_CARRIED
from ._declaration import (
    _copies_of_one_irrep, _declared_atoms, _declared_needs, _distance_from_trivial,
    _holds_part, _lead_of, _meets, _named_content, _one_copy, _returned_rows)
from .sewing import DEGENERACY_TOL, _as_columns, multiplets


#: What each key of the labelling reads, in the order :func:`_labelling` asks
#: them: the site content at the lowest angular momentum (orbits read together
#: read one by one, and identical copies copy by copy, first; then at the lead's
#: site, over the crystal, on the lead's row), the letter the orbitals read as,
#: the band labels, and last the group's order of operations, on which no label
#: rests.
_KEY_NAMES = ("content", "content", "content", "content", "content", "orbitals", "bands",
              "order")


class _LazyW:
    """A candidate's matrices on its states, built the first time one is read."""

    def __init__(self, cand):
        self._cand = cand

    def _all(self):
        return self._cand["W"]

    def __getitem__(self, p):
        return self._all()[p]

    def __contains__(self, p):
        return p in self._all()

    def __iter__(self):
        return iter(self._all())

    def __len__(self):
        return len(self._all())

    def items(self):
        return self._all().items()

    def keys(self):
        return self._all().keys()


class _LazyOption(dict):
    """A labelling of an orbit built by induction, its matrices built only where a key asks for them.

    ``"W"`` is the candidate's own matrices on the orbit's states, built the
    first time one is read, and ``"D"`` the representation with them put in --
    built each time it is asked for, so that what a key reads off it is kept
    and the matrices are not.
    """

    def __init__(self, base, cand, states, **fields):
        super().__init__(fields, W=_LazyW(cand))
        self._base, self._states = base, np.asarray(states, dtype=int)

    def __missing__(self, key):
        if key == "D":
            W, st = self["W"], self._states
            M = {}
            for p, Dp in self._base.items():
                M[p] = np.array(Dp, dtype=complex, copy=True)
                if p in W:
                    M[p][np.ix_(st, st)] = W[p]
            return M
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _choose(options, keys):
    r"""``(kept, fate)`` -- the option the keys keep, first key first.

    Each key scores the options still tied, and only those with its lowest
    score go on to the next; the first of the last ones left is kept.  A key
    with a ``batch`` attribute scores the tied options together, one score
    each.  ``fate[i]`` is the position in ``keys`` of the key that set option
    ``i`` aside, ``None`` for the one kept, and ``len(keys)`` for one no key set
    apart from it.
    """
    left = list(range(len(options)))
    fate = [None] * len(options)
    for k, key in enumerate(keys):
        if len(left) <= 1:
            break
        scores = (list(key.batch([options[i] for i in left])) if hasattr(key, "batch")
                  else [key(options[i]) for i in left])
        best = min(scores)
        keep = [i for i, s in zip(left, scores) if s == best]
        for i in left:
            if i not in keep:
                fate[i] = k
        left = keep
    for i in left[1:]:
        fate[i] = len(keys)
    return left[0], fate


def _band_fibers(model, group, D, syms, *, params=None):
    r"""The energy multiplets at the high-symmetry points, solved once.

    ``H`` is solved once at every high-symmetry point the tables list, in their
    order, and each point's eigenvectors are grouped into energy multiplets,
    lowest first.  A point whose little group is the identity alone, or holds
    an operation ``D`` does not carry, is left out.

    Returns one ``(point, k, little group, multiplets, V)`` per point; empty
    where the tables do not reach the cell.
    """

    if _spacegroup._lattice_not_descended(group) is not None or not group._tables_ready:
        return ()
    points = _tables.high_symmetry_points(group)
    if isinstance(points, Refusal) or not points:
        return ()
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    wanted = []
    for name, k in points.items():
        lg = _tables.little_group(group, k)
        if isinstance(lg, Refusal) or len(lg.unitary) <= 1:
            continue
        if any(int(i) not in pos_of or pos_of[int(i)] not in D for i in lg.indices):
            continue
        wanted.append((str(name), np.asarray(k, dtype=float).reshape(-1), lg))
    if not wanted:
        return ()
    evals, evecs = _bloch.solve(model, np.array([k for _n, k, _lg in wanted]), params)
    evals, evecs = np.asarray(evals), np.asarray(evecs)
    if evals.ndim == 1:                     # solve_ham drops a length-1 k axis
        evals, evecs = evals[None], evecs[None]
    return tuple((name, k, lg, multiplets(np.asarray(evals[i], dtype=float).reshape(-1),
                                          DEGENERACY_TOL), _as_columns(evecs[i]))
                 for i, (name, k, lg) in enumerate(wanted))


def _band_share(model, group, assignment, D, syms, fibers, states):
    r"""One coupled part's share of every energy multiplet's character.

    Per point of ``fibers`` (:func:`_band_fibers`), ``chars[row, column]``: the
    trace, over the multiplet's eigenvectors restricted to the part's
    ``states``, of the Bloch operator of ``D`` there (:func:`bloch_operator`),
    one column per unitary operation of the little group.  ``D`` carries each
    part onto itself, so a multiplet's character is the sum of its parts'
    shares, and a character ``chi`` on the part multiplies its share by
    ``chi(g)``.  ``None`` at a point where the operator's own checks fail.
    """
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    st = np.asarray(states, dtype=int)
    out = []
    for _name, k, lg, rows, V in fibers:
        chars = np.zeros((len(rows), len(lg.indices)), dtype=complex)
        try:
            for col, ind in enumerate(lg.indices):
                p = pos_of[int(ind)]
                U = bloch_operator(model, group, assignment, syms[p], D[p], k,
                                   k_out=k)[np.ix_(st, st)]
                for r, m in enumerate(rows):
                    Vs = V[np.ix_(st, list(m))]
                    chars[r, col] = np.trace(Vs.conj().T @ U @ Vs)
        except ValueError:
            out.append(None)                # the operator's own checks: nothing read here
            continue
        out.append(chars)
    return out


class _BandKey:
    r"""The band labels a candidate gives, comparable in the tables' order.

    For every high-symmetry point in the tables' order and every energy
    multiplet from the lowest up, the corep content the representation gives
    it once the part carries the candidate and every other part the one
    ``held`` holds it at, as the multiplicity of each of the point's coreps in
    the tables' own order.  Candidates compare at the first place their labels
    differ, the one whose content comes first in the tables' order of that
    point's coreps first.  A point where the labels cannot be read is left out
    for every candidate alike.  ``share(q, option)`` is part ``q``'s share of
    the characters under one of its options (:func:`_band_share`), measured
    the first time any part asks.
    """

    def __init__(self, group, fibers, share, held, part, n_parts):
        self.group, self.fibers, self.share = group, fibers, share
        self.held, self.part, self.n_parts = held, int(part), int(n_parts)
        self.orders = {}

    def _labels(self, i, option):

        _name, k, lg, rows, _V = self.fibers()[i]
        total = np.zeros((len(rows), len(lg.indices)), dtype=complex)
        for q in range(self.n_parts):
            got = self.share(q, option if q == self.part else self.held(q))[i]
            if got is None:
                return None
            total = total + got
        found = _tables.identify_from_characters(self.group, k, total,
                                                 multiplet_dims=[len(m) for m in rows])
        if isinstance(found, Refusal):
            return None
        if i not in self.orders:
            tab = _tables.corep_table(self.group, k, kpname=found.kpname)
            self.orders[i] = () if isinstance(tab, Refusal) else tuple(tab)
        return tuple(tuple(-int(row.get(label, 0)) for label in self.orders[i])
                     for row in found.labels)

    def batch(self, options):
        """One comparable key per candidate: its labels, point by point."""
        keys = [[] for _o in options]
        for i in range(len(self.fibers())):
            got = [self._labels(i, o) for o in options]
            if any(g is None for g in got):
                continue
            for j, g in enumerate(got):
                keys[j].append(g)
        return [tuple(k) for k in keys]

    def reading(self, option) -> dict:
        """``{point: ((bands, {corep: m}), ...)}`` -- the labels one candidate gives every energy multiplet.

        Each multiplet is the band indices it holds, from the lowest up, with
        its corep content; a point whose labels cannot be read is left out.
        """

        out = {}
        for i, (name, k, _lg, rows, _V) in enumerate(self.fibers()):
            total = None
            for q in range(self.n_parts):
                got = self.share(q, option if q == self.part else self.held(q))[i]
                if got is None:
                    total = None
                    break
                total = got if total is None else total + got
            if total is None:
                continue
            found = _tables.identify_from_characters(self.group, k, total,
                                                     multiplet_dims=[len(m) for m in rows])
            if isinstance(found, Refusal):
                continue
            out[str(name)] = tuple((tuple(int(b) for b in m), dict(labels))
                                   for m, labels in zip(rows, found.labels))
        return out


def _coupled_parts(classes, unpinned, parts):
    """The coupled parts of the crystal as sets of multiplet orbits, the anchor's first.

    ``parts`` is the measurement's, the anchor's own part first; an orbit the
    hoppings leave free of the anchor and that no part names is a part of its
    own.
    """
    nclass = int(np.max(classes)) + 1 if len(classes) else 0
    unpinned = frozenset(int(c) for c in (unpinned or ()))
    out = [frozenset(c for c in range(nclass) if c not in unpinned)]
    for part in tuple(parts or ())[1:]:
        got = frozenset(int(c) for c in part) & unpinned
        if got:
            out.append(got)
    for cls in sorted(unpinned):
        if not any(cls in got for got in out):
            out.append(frozenset({int(cls)}))
    return [p for p in out if p]


def _parts_apart(verdict, layout, sectors):
    """The orbitals of each part of the crystal a verdict read apart and the sectors keep apart.

    ``verdict`` is :func:`.._declaration._verdicts_and_choices`'s, on the multiplets ``layout``
    holds, and ``sectors`` the reading of the conserved matrices at the
    declared precision (:func:`.._sectors.read_sectors`).  A part the verdict
    leaves free of the others can still be coupled to them, its ratio carried
    by the blocks a wider multiplet reads off the Hamiltonian, so parts are
    held apart only where no sector joins them: the finest partition both
    coarsen into.  ``None`` where that is one part, or nothing was read.
    """
    if not verdict or verdict.get("classes") is None or sectors is None:
        return None
    classes = np.asarray(verdict["classes"], dtype=int)
    parts = _coupled_parts(classes, verdict.get("free_classes"), verdict.get("parts"))
    if len(parts) < 2:
        return None
    joined = [tuple(sorted({int(o) for c in range(len(classes)) if int(classes[c]) in part
                            for o in layout._chan_orbs[c]})) for part in parts]
    joined += [tuple(sorted({int(o) for a in sector.atoms for o in sectors.atoms[int(a)]}))
               for sector in sectors.sectors]
    root = list(range(int(layout._chan_of.shape[0])))

    def _find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for orbs in joined:
        for o in orbs[1:]:
            ra, rb = _find(int(orbs[0])), _find(int(o))
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
    blocks = {}
    for o in range(len(root)):
        blocks.setdefault(_find(o), []).append(int(o))
    if len(blocks) < 2:
        return None
    return tuple(tuple(b) for _r, b in sorted(blocks.items()))


def _held_sets(sectors, layout, model, group, assignment, spin, validated):
    """What the sectors read at the declared precision keep apart that a representation is read on.

    ``sectors`` is the reading of the conserved matrices at the declared
    precision (:func:`.._sectors.read_sectors`), and ``layout`` the multiplets
    held together by them (:func:`.._sectors.carried_reading`).  Returns
    ``{"copies", "shares"}``, or ``None`` where no precision was declared or
    there is neither:

    * ``copies``: every sector of several identical copies, each copy lying
      along orbitals of its own within that precision, as ``(basis, copies,
      dim, orbitals)`` -- its basis in the states, copy-major and held exactly
      along those orbitals (:func:`.._sectors.aligned`), its number of copies,
      the states one copy holds and the orbitals it lies on; on states that
      carry a spin each copy's states are its orbital factor's times the spin;
    * ``shares``, on states read without spin: every set of sectors of one
      copy that one multiplet holds together -- of orbits no operation of the
      group exchanges, the bonding and the antibonding state two orbitals
      make or orbitals nothing couples stated as one multiplet, or of one
      orbit, the square's ``p_x, p_y`` pair nothing couples -- as ``(bases,
      orbitals)``, each sector's basis held to its own share of every site,
      within the multiplets that hold it (:func:`.._sectors.held_shares`).

    What the Hamiltonian holds between such sectors, between copies or
    between them and the rest, and what tells one copy from another, lies
    below that precision, where it counts as absent; and the representations
    are read on them as exact (:func:`.._sectors.induced_orbits`).
    """

    if sectors is None or sectors.tol is None:
        return None
    if _sectors.carried_reading(sectors, layout, tol=sectors.tol) is not sectors:
        return None
    sites = _sectors.atom_sites(sectors, layout)
    nstate = int(model.nstate)

    def _orbitals(ts):
        return tuple(sorted({int(o) for t in ts for a in sectors.sectors[int(t)].atoms
                             for o in sectors.atoms[int(a)]}))

    copies = []
    for t, s in enumerate(sectors.sectors):
        m, d = int(s.copies), int(s.dim)
        if m < 2:
            continue
        Q = _sectors.aligned(s, sectors.tol)
        exact = _sectors._snapped(Q.reshape(int(Q.shape[0]), m, d), sites, sectors.tol)
        if exact is None:
            continue                       # copies that do not lie along orbitals of their own
        basis = exact.reshape(int(Q.shape[0]), m * d)
        if sectors.su2:
            basis, d = _sectors.in_states(sectors, basis, nstate), 2 * d
        copies.append((basis, m, d, _orbitals((t,))))
    shares = []
    held = _sectors.held_sectors(sectors, layout, tol=sectors.tol)
    if not sectors.su2 and any(len(h) > 1 for h in held):
        syms = group.symmetries
        kept = [p for p, s in enumerate(syms) if int(s.ind) in set(validated.indices)]
        action = _sectors.sector_action(sectors, model, group, assignment, spin, validated,
                                        tol=sectors.tol)
        found = _sectors.sector_orbits(sectors, action, group, layout, kept, tol=sectors.tol)
        for roots, _multiplets in (found[1] if found is not None else ()):
            members = [t for r in roots for t in found[0][r]]
            if len(roots) < 2 and not any(len(set(h) & set(members)) > 1 for h in held):
                continue                   # one orbit, no multiplet holding two of its sectors
            if any(int(sectors.sectors[t].copies) != 1 for t in members):
                continue
            bases = _sectors.held_shares(sectors, members, sites, sectors.tol,
                                         on=_sectors.held_on(sectors, layout, held, members))
            if bases is not None:
                shares.append((bases, _orbitals(members)))
    return {"copies": tuple(copies), "shares": tuple(shares)} if copies or shares else None


def _tables_target(group, table, rotations, content, orbitals, words):
    r"""``{ind: matrix}`` -- the tables' matrices of a multiplet's content, along its orbitals; or ``None``.

    ``content`` is ``{ind: chi(g)}`` over the site group ``table`` is the
    table of, and ``rotations`` each operation's Cartesian matrix.  Where every
    orbital was declared a function (``words``, ``{orbital: word}``) and those
    functions carry the content, each orbital is its own function; otherwise
    the content's irreps are laid along the orbitals in order, the lowest
    angular momentum first and then the table's own order, each on the
    functions the tables write it on (:func:`.._sitechars.irrep_functions`):
    ``(p_x, p_y)`` for ``E`` at a ``4mm`` site.  A matrix is the operation's
    action on those functions, ``(g·f)(r) = f(R⁻¹ r)``.  ``None`` where the
    content is no character of the site group, or an irrep in it has no such
    functions.
    """
    if table is None or any(int(cls[0]) not in content for cls in table.classes):
        return None
    indices = [int(i) for i in table.indices]
    orbitals = [int(o) for o in orbitals]
    shells = {}

    def _on(funcs):
        """``{ind: matrix}`` -- every operation on the functions ``[(degree, column)]``."""
        out = {}
        for ind in indices:
            M = np.zeros((len(funcs), len(funcs)), dtype=complex)
            for j, (dj, fj) in enumerate(funcs):
                if (ind, dj) not in shells:
                    shells[(ind, dj)] = _sitechars.shell_operator(group, ind, dj)
                image = shells[(ind, dj)] @ fj
                for i, (di, fi) in enumerate(funcs):
                    if di == dj:
                        M[i, j] = np.vdot(fi, image)
            out[ind] = M
        return out

    def _carries(target):
        return all(float(np.abs(T @ T.conj().T - np.eye(len(T))).max()) <= ASSEMBLED
                   and abs(complex(np.trace(T)) - complex(content[ind])) <= CHARACTER
                   for ind, T in target.items())

    if orbitals and all(o in words for o in orbitals):
        funcs = [_sitechars.function_of(words[o]) for o in orbitals]
        if all(f is not None for f in funcs):
            target = _on(funcs)
            if _carries(target):
                return target
    mult = _sitechars.decompose(table, np.asarray([complex(content[int(cls[0])])
                                                   for cls in table.classes]))
    funcs = []
    for row, n in enumerate(mult):
        k = int(round(float(np.real(n))))
        if abs(complex(n) - k) > CHARACTER or k < 0:
            return None
        got = _sitechars.irrep_functions(group, table, row, rotations) if k else None
        if k and got is None:
            return None
        if k:
            degree, basis = got
            funcs.extend([((degree, row), [(degree, basis[:, j])
                                           for j in range(basis.shape[1])])] * k)
    if sum(len(f) for _key, f in funcs) != len(orbitals):
        return None
    # the lowest angular momentum first, then the table's own order; each
    # irrep's own functions orthogonal to every other's, so the matrices are
    # block-diagonal along the orbitals
    blocks = [_on(f) for _key, f in sorted(funcs, key=lambda item: item[0])]
    target = {}
    for ind in indices:
        T = np.zeros((len(orbitals), len(orbitals)), dtype=complex)
        at = 0
        for on in blocks:
            size = int(on[ind].shape[0])
            T[at:at + size, at:at + size] = on[ind]
            at += size
        target[ind] = T
    return target if _carries(target) else None


def _in_tables_gauge(rec, winner, *, group, layout, syms, u_pos, cperm, site_tables,
                     rotations, words):
    r"""The matrices of the representation chosen for an orbit built by induction, in the tables' own.

    Which of the representations conjugate under what ``H`` conserves on the
    orbit -- a unitary on identical copies, a phase on each of several sectors
    one multiplet holds -- is no measurement, so it is fixed by the convention
    every matrix returned is written in: on the orbit's lead multiplet, the
    tables' matrices of the content it carries, laid along its orbitals in
    order (:func:`_tables_target`), wherever what ``H`` conserves reaches them
    (:func:`.._sectors.tables_gauge`) -- the copies and the sectors ordered as
    their orbitals.  Where it does not, the orbitals of the multiplet each
    holding a share of several irreps, the representation stays as it was
    induced, each sector the image of the first.  On states that carry a spin
    the tables' matrices are those of the orbital content times each
    operation's own lift on the spin.  ``rec`` is the orbit's record
    and ``winner`` the candidate chosen; the matrices are returned unchanged
    where the orbit holds one sector of one copy, which leaves nothing open.
    """

    W = winner["W"]
    induced = rec.get("induced")
    if induced is None:
        return W
    frames = induced.sector_frames()
    if len(frames) < 2 and all(int(f[1]) < 2 for f in frames):
        return W
    lead = int(rec["lead"])
    orbitals, places = layout._chan_orbs[lead], None
    if layout._spinful:
        # the lead's orbitals (or declared pairs) once each, as its states reach them
        reading = getattr(induced, "reading", None) or induced.orbits[0].reading
        places = _sectors.spin_places(reading, layout._chan_states[lead])
        atoms = list(dict.fromkeys(a for a, _s in places))
        orbitals = [int(reading.atoms[a][0]) for a in atoms]
        places = [(atoms.index(a), s) for a, s in places]
    target = _tables_target(group, site_tables.get(int(layout._chan_site[lead])), rotations,
                            winner["contents"][lead], orbitals, words)
    if target is None:
        return W
    pos_of = {int(syms[p].ind): int(p) for p in u_pos}
    target = {pos_of[ind]: T for ind, T in target.items()
              if ind in pos_of and int(cperm[pos_of[ind]][lead]) == lead}
    if places is not None:
        at = [a for a, _s in places]
        spin = [s for _a, s in places]
        target = {p: T[np.ix_(at, at)] * _lift_of(syms[p], True)[np.ix_(spin, spin)]
                  for p, T in target.items()}
    local = {int(a): i for i, a in enumerate(rec["states"])}
    rows = [local[int(a)] for a in layout._chan_states[lead]]
    return _sectors.tables_gauge(frames, W, target, rows, syms,
                                 tol=getattr(layout, "_tol", None) or None)


def _labelling(D, raw, decls, *, pool, model, group, assignment, layout, syms, u_pos,
               cperm, classes, parts, ref_chan, site_tables, measured_at,
               multiplicity=(), params=None):
    r"""The labelling returned: what each coupled part of the crystal carries, chosen once.

    The hoppings fix the representation ``D`` only up to one one-dimensional
    character per coupled part of the crystal -- the multiplets they connect,
    directly or through others (``parts``) -- and, on a part that is an orbit
    of sectors built by induction -- identical copies, a conserved pseudospin,
    or uncoupled orbitals a symmetry exchanges that those characters do not
    carry -- up to the representation induced on it (``multiplicity``,
    :func:`.._multiplets._induced_blocks`).  A part's **candidates** are its characters in
    ``pool``, or the representations induced on it, under which every
    declaration made on it holds
    (:func:`_meets`, :func:`_holds_part`); where none meets them all, those met
    at the operations the measurement read each declaration at
    (``measured_at``).  Declared generator matrices are read as matrices: a
    candidate holds them where its own matrices on the declared orbitals are
    the declared ones; where none does, the characters alone choose, and the
    rows say the matrices are not carried.  A content declared over several parts
    is one requirement on all of them, so a part is chosen only where the
    parts still to be chosen can complete it, each among its own candidates.

    **One key** then chooses among each part's candidates, the same for every
    part, its lead multiplet the reference orbital's where the part holds it and
    its first otherwise (:func:`_lead_of`).  Every step scores the orbital
    factor, the spin-1/2 lift divided out:

    1. the part's content at its lead's site at the lowest angular momentum
       (:func:`_content_shells`), then over the crystal, then on the lead's
       own row;
    2. the letter the orbitals read as: each orbit's frame character -- over
       its sites, the trace of the block carrying each multiplet to its image,
       orbital by orbital -- averaged over the point group's classes, divided
       by the orbit's site count and read on the point group's own table at
       the lowest angular momentum, the lead's orbit first and then the part's
       orbits together;
    3. the band labels: at the high-symmetry points in the tables' order, and
       from the lowest energy multiplet up, the candidate whose content comes
       first in the tables' order of the point's coreps (:class:`_BandKey`),
       measured only where the keys before leave more than one;
    4. the group's own order of operations relative to ``D``
       (:func:`_distance_from_trivial`), or for the copies of a pseudospin the
       representation whose character is nearest the trivial one: where it
       decides, no band label rests on it.

    Returns ``{"of_chan", "swaps", "raw", "chosen", "candidates", "labelled",
    "parts", "part_states", "spaces", "lift_trace", "matrices_carried"}``: the
    character every multiplet is assembled with; the representation chosen on each orbit built
    by induction, as matrices on the orbit's states; every multiplet's
    character under the labelling's representation before its characters are
    multiplied in; per part the character chosen, and for each candidate its
    character with what the multiplets carry before it multiplies them
    (``None`` for a pseudospin's copies, which carry the chosen one's); per part
    whose candidates include one giving every multiplet the site corep the
    chosen one gives it and labelling some band differently, the band labels
    of every energy multiplet under the one chosen and under each such, with
    the key that set it aside; the parts' multiplets and states; and each
    pseudospin's content returned and every content its candidates give, in
    the key's order; and, keyed by each declaration of generator matrices,
    whether the labelling returned gives its orbitals those matrices.
    """
    chan_states, su2 = layout._chan_states, layout._spinful
    nchan = len(chan_states)
    classes = np.asarray(classes, dtype=int)
    lift_trace = {int(syms[p].ind): complex(np.trace(_lift_of(syms[p], su2))) for p in u_pos}
    rotations = {int(syms[p].ind): np.asarray(syms[p].rotation_cart, dtype=float)
                 for p in u_pos}
    unitary = sorted(int(syms[p].ind) for p in u_pos)
    shells, lowest = {}, {}
    chans_of = [[int(c) for c in range(nchan) if int(classes[c]) in part] for part in parts]
    part_states = [sorted(int(a) for c in chans for a in chan_states[c]) for chans in chans_of]
    spaces = {tuple(sorted(int(c) for c in rec["multiplets"])): rec
              for rec in (multiplicity or ()) if rec.get("candidates")}
    names = _KEY_NAMES
    # the declared function of each orbital, where one was written: the tables'
    # matrices a representation the hoppings leave open is written in follow it
    words = {int(o): w for d in (decls or ()) if d.get("content") is not None
             for o, w in d["content"].words}

    def _options(chans):
        """A part's options: its characters, or the representations induced on it."""
        rec = spaces.get(tuple(sorted(chans)))
        if rec is None:
            # A COUPLED PART'S CHARACTERS.  The representations the hoppings
            # leave on a part whose sectors lie on whole multiplets are the
            # group's one-dimensional characters times ``D`` -- the same
            # mathematics as the representation induced on that orbit of one
            # copy, Ind(chi|K (x) w) = chi (x) Ind(w) -- so they are offered as
            # characters on the matrices the search settled: built by induction
            # instead they come back in another gauge, and a content declared on
            # such a part is not carried.
            return [{"chi": chi, "W": None} for chi in pool], None
        st = rec["states"]
        # each a representation induced on the orbit, its matrices built where
        # a key reads them
        return [_LazyOption(D, cand, st, chi={}, contents=cand["contents"],
                            character=cand["character"], k=int(rec["dim"]),
                            parts=cand.get("parts"), key=id(cand))
                for cand in rec["candidates"]], rec

    def _source(opt):
        return opt["contents"] if opt["W"] is not None else raw

    def _content(chans, opt):
        """``{ind: chi(g)}`` -- what a set of multiplets carries under one candidate."""
        out, src, chi = {}, _source(opt), opt["chi"]
        for c in chans:
            for ind, v in src[int(c)].items():
                out[int(ind)] = (out.get(int(ind), 0.0)
                                 + complex(chi.get(int(ind), 1.0)) * complex(v))
        return out

    def _shells(tab, theta):
        return _content_shells(tab, theta, rotations, shells=shells, lowest=lowest)

    def _order(opt):
        if opt["W"] is not None:
            return _distance_from_trivial({int(i): complex(v) / opt["k"]
                                           for i, v in opt["character"].items()})
        return _distance_from_trivial({ind: complex(opt["chi"].get(ind, 1.0))
                                       for ind in unitary})

    # THE LETTER THE ORBITALS READ AS: an orbit's frame character, read on the
    # representation as built, on the point group's own single-valued table
    tables = {}

    def _point():
        if "point" not in tables:
            tables["point"] = _sitechars.site_group_table(group, unitary)
        return tables["point"]

    orbit_of_site = np.asarray(assignment.orbit_of_site, dtype=int)
    atoms_of = {int(c): _atom_blocks(chan_states[c], su2) for c in range(nchan)}
    lift_of = {int(p): _lift_of(syms[p], su2) for p in u_pos}

    def _frames(chans, M):
        """``{orbit: (phi, n_sites, width)}`` -- the frame character of each orbit of a part.

        ``phi(g)`` sums, over the orbit's sites, the trace of the orbital factor
        of ``M(g)``'s block carrying the site's multiplets to their images:
        orbital ``k`` of a multiplet read against orbital ``k`` of its image,
        the spin-1/2 lift divided out atom by atom.
        """
        by_orbit = {}
        for c in chans:
            by_orbit.setdefault(int(orbit_of_site[int(layout._chan_site[int(c)])]),
                                []).append(int(c))
        out = {}
        for orbit, cs in sorted(by_orbit.items()):
            phi = {}
            for p in u_pos:
                total = 0j
                for c in cs:
                    c2 = int(cperm[p][c])
                    total += sum(complex(_proportionality(
                        np.asarray(M[p])[np.ix_(a2, a1)], lift_of[int(p)])[0])
                        for a2, a1 in zip(atoms_of[c2], atoms_of[c]))
                phi[int(syms[p].ind)] = total
            sites = {int(layout._chan_site[c]) for c in cs}
            width = sum(len(atoms_of[c]) for c in cs) // max(len(sites), 1)
            out[orbit] = (phi, len(sites), int(width))
        return out

    def _letter(frame, chi):
        """``(single shell, per orbital)`` of the letter one orbit reads as under ``chi``.

        The frame character times ``chi``, averaged over each class of the point
        group and divided by the orbit's site count, decomposed on the point
        group's own table with its multiplicities rounded: the point-group
        representation the orbitals are written as, at the angular momentum
        :func:`_content_shells` reads it at.  A rounding that is no
        representation -- a negative multiplicity, or not the orbit's width --
        scores as no character of the table does, and so ties.
        """
        phi, m, width = frame
        point = _point()
        theta = [sum(complex(chi.get(int(i), 1.0)) * phi[int(i)] for i in cls)
                 / (len(cls) * m) for cls in point.classes]
        n = [int(round(float(np.real(x)))) for x in _sitechars.decompose(point, theta)]
        if (any(k < 0 for k in n)
                or sum(k * int(point.dims[q]) for q, k in enumerate(n)) != int(width)):
            return (_L_MAX + 1, 10 ** 6)
        content = {int(i): sum(k * complex(point.character_on(q, i))
                               for q, k in enumerate(n) if k) for i in point.indices}
        return _content_shells(point, content, rotations, shells=shells, lowest=lowest)

    def _frame_key(chans, lead_orbit):
        """The letter of the lead's orbit, then the letters summed over the part's orbits."""
        frames = {}

        def key(opt):
            src = opt["key"] if opt["W"] is not None else 0
            if src not in frames:
                frames[src] = _frames(chans, opt["D"] if opt["W"] is not None else D)
            scores = {o: _letter(f, opt["chi"]) for o, f in frames[src].items()}
            return (scores[lead_orbit], tuple(sum(s) for s in zip(*scores.values())))
        return key

    # THE BAND LABELS, measured once and only where the keys before them tie:
    # every part's share of every energy multiplet's character at the
    # high-symmetry points (:func:`_band_fibers`, :func:`_band_share`)
    measured = {}

    def _fibers():
        if "fibers" not in measured:
            measured["fibers"] = _band_fibers(model, group, D, syms, params=params)
        return measured["fibers"]

    def _share(q, opt):
        if opt is not None and opt["W"] is not None:
            key = ("W", q, opt["key"])
            if key not in measured:
                measured[key] = _band_share(model, group, assignment, opt["D"], syms,
                                            _fibers(), part_states[q])
        else:
            key = ("D", q)
            if key not in measured:
                measured[key] = _band_share(model, group, assignment, D, syms, _fibers(),
                                            part_states[q])
        base = measured[key]
        if opt is None or not opt["chi"]:
            return base
        out = []
        for (_name, _k, lg, _rows, _V), chars in zip(_fibers(), base):
            f = np.asarray([complex(opt["chi"].get(int(i), 1.0)) for i in lg.indices],
                           dtype=complex)
            out.append(None if chars is None else chars * f[None, :])
        return out

    held_at = []

    def _held(q):
        """The candidate part ``q`` is held at while another is chosen: its own, once chosen."""
        return held_at[q] if q < len(held_at) else None

    def _site_irrep(theta, chi):
        return tuple(sorted(
            (int(ind),
             round(float(np.real(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6),
             round(float(np.imag(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6))
            for ind, v in theta.items()))

    # WHAT EACH PART MAY BE, before any is chosen: its options, and among them
    # those every declaration made on it alone holds under
    part_of_chan = {int(c): q for q, chans in enumerate(chans_of) for c in chans}
    options_of, records = [], []
    for chans in chans_of:
        options, rec = _options(chans)
        options_of.append(options)
        records.append(rec)
    mine_of = [[d for d in decls
                if d["channels"] and all(int(c) in set(chans) for c in d["channels"])]
               for chans in chans_of]
    # ... and the declarations a part shares with others, one requirement on all
    # of them, read on the parts' options together
    spanned = {id(d): sorted({part_of_chan[int(c)] for c in d["channels"]}) for d in decls
               if d["channels"] and len({part_of_chan[int(c)] for c in d["channels"]}) > 1}
    shared = [d for d in decls if id(d) in spanned]
    needs = {}

    nspin = int(getattr(model, "nspin", 1) or 1)
    pos_of_ind = {int(syms[p].ind): int(p) for p in u_pos}

    def _matrix_needs(d, opt):
        """``{ind: value}`` -- what declared generator matrices ask of their part's character, on one option's matrices.

        Per declared operation, the scalar carrying the orbital factor of the
        option's block on the declaration's atoms onto the declared matrix, or
        ``None`` where no scalar does.
        """
        key = (id(d), "matrices", opt["key"] if opt["W"] is not None else 0)
        if key not in needs:
            matrices = opt["D"] if opt["W"] is not None else D
            atoms = _declared_atoms(d, chan_states, su2, nspin)
            out = {}
            for ind, M in d["matrices"].items():
                p = pos_of_ind.get(int(ind))
                if p is None or p not in matrices:
                    continue
                B = np.asarray([[_proportionality(np.asarray(matrices[p])[np.ix_(a, b)],
                                                  lift_of[p])[0] for b in atoms]
                                for a in atoms])
                lam, gap = _proportionality(np.asarray(M, dtype=complex), B)
                out[int(ind)] = (lam if gap <= CHARACTER * max(1.0, float(np.abs(M).max()))
                                 else None)
            needs[key] = out
        return needs[key]

    def _ok(q, opt, measured_only=False, matrices=True):
        """Whether every declaration made on part ``q`` alone holds under one of its options.

        ``matrices`` false reads declared generator matrices through their
        traces alone, as every other declaration is read.
        """
        src = _source(opt)
        for d in mine_of[q]:
            if d.get("partial"):
                if not _holds_part(d, lambda _c, chi=opt["chi"]: chi, src, site_tables,
                                   lift_trace):
                    return False
                continue
            key = (id(d), id(src))
            if key not in needs:
                needs[key] = _declared_needs(d, src, lift_trace)
            asked = needs[key]
            if matrices and d.get("matrices"):
                asked = {**asked, **_matrix_needs(d, opt)}
            over = (measured_at.get(tuple(int(o) for o in d["orbitals"]), set())
                    if measured_only else None)
            if not _meets(opt["chi"], asked, over=over):
                return False
        return True

    def _together(d, pick, measured_only=False):
        """Whether a declaration spanning several parts holds, each part at its option in ``pick``."""
        def assigned(c):
            return pick[part_of_chan[int(c)]]["chi"]
        src = {int(c): _source(pick[part_of_chan[int(c)]])[int(c)] for c in d["channels"]}
        if d.get("partial"):
            return _holds_part(d, assigned, src, site_tables, lift_trace)
        carried = {c: {g: complex(assigned(c).get(g, 1.0)) * complex(v) for g, v in row.items()}
                   for c, row in src.items()}
        over = (measured_at.get(tuple(int(o) for o in d["orbitals"]), set())
                if measured_only else None)
        return _meets({}, _declared_needs(d, carried, lift_trace), over=over)

    own_of = {}

    def _own(q):
        """Part ``q``'s options under which its own declarations hold."""
        if q not in own_of:
            own_of[q] = ([o for o in options_of[q] if _ok(q, o)]
                         or [o for o in options_of[q] if _ok(q, o, True)] or options_of[q])
        return own_of[q]

    def _completes(q, opt, measured_only=False):
        """Whether ``opt`` for part ``q`` leaves the parts after it a way to meet what they share."""
        live = [d for d in shared if spanned[id(d)][-1] >= q]
        if not live:
            return True
        later = sorted({p for d in live for p in spanned[id(d)] if p > q})
        for combo in itertools.product(*(_own(p) for p in later)):
            pick = dict(enumerate(held_at))
            pick[q] = opt
            pick.update(zip(later, combo))
            if all(_together(d, pick, measured_only) for d in live
                   if all(p in pick for p in spanned[id(d)])):
                return True
        return False

    chosen, candidates, open_of, swaps, spaces_out = [], [], [], [], []
    matrices_carried = {}
    raw_after = {int(c): dict(v) for c, v in raw.items()}
    for i, chans in enumerate(chans_of):
        options, rec = options_of[i], records[i]
        cands = [o for o in options if _ok(i, o) and _completes(i, o)]
        if not cands:
            cands = [o for o in options if _ok(i, o, True) and _completes(i, o, True)]
        if not cands:
            # what the parts share is said on the rows where no labelling carries it
            cands = ([o for o in options if _ok(i, o)]
                     or [o for o in options if _ok(i, o, True)])
        if not cands:
            # ... and so are declared matrices no labelling gives the orbitals:
            # their traces choose
            cands = ([o for o in options
                      if _ok(i, o, matrices=False) and _completes(i, o)]
                     or [o for o in options
                         if _ok(i, o, True, matrices=False) and _completes(i, o, True)]
                     or [o for o in options if _ok(i, o, matrices=False)]
                     or [o for o in options if _ok(i, o, True, matrices=False)])
        if not cands:
            if int(ref_chan) in chans:     # pragma: no cover - the verdict rules it out
                raise InternalError(
                    "the declared orbital content was found covariant on this group, and no "
                    "labelling of the crystal carries every declared multiplet in its declared "
                    "irrep on the built matrices. The verdict and the build are two readings of "
                    "one measurement and have to agree; the two disagreeing is a defect in this "
                    "package, not a tolerance to loosen.")
            cands = list(options)
        lead = _lead_of(chans, ref_chan)
        here = int(layout._chan_site[lead])
        at = {}
        for c in chans:
            at.setdefault(int(layout._chan_site[c]), []).append(int(c))
        def _copy_by_copy(o, at=at, here=here):
            """Identical copies read copy by copy: each holding one content, at the lowest angular momentum.

            A representation that puts the same content on every copy of every
            multiplet -- ``k`` copies of one content -- comes first, ordered by
            what one copy holds at the lead's site; any other after them.
            """
            k = int(o.get("k") or 0)
            if k < 2:
                return (0, (0, 0))
            one = {}
            for s, cs in sorted(at.items()):
                for c in cs:
                    got = _one_copy(site_tables.get(s), _content((c,), o), k)
                    if got is None:
                        return (1, (0, 0))
                    if s == here:
                        for ind, v in got.items():
                            one[int(ind)] = one.get(int(ind), 0.0) + complex(v)
            return (0, _shells(site_tables.get(here), one))

        def _part_by_part(o, at=at, here=here):
            """Orbits read together read one by one, each at the lowest angular momentum.

            Orbits of sectors whose multiplets hold shares of one another's are
            each a choice of their own (:class:`.._sectors.SharedOrbits`), so
            each is read on its own share of the lead's site, in turn; any
            other option reads alike here.
            """
            parts = o.get("parts")
            if not parts:
                return ()
            out = []
            for contents in parts:
                share = {}
                for c in at.get(here, ()):
                    for ind, v in contents.get(int(c), {}).items():
                        share[int(ind)] = share.get(int(ind), 0.0) + complex(v)
                out.append(_shells(site_tables.get(here), share))
            return tuple(out)

        keys = [
            _part_by_part,
            _copy_by_copy,
            lambda o, at=at, here=here: _shells(site_tables.get(here),
                                                _content(at.get(here, ()), o)),
            lambda o, at=at: (tuple(sum(w) for w in zip(*[
                _shells(site_tables.get(s), _content(cs, o)) for s, cs in sorted(at.items())]))
                if at else (0, 0)),
            lambda o, lead=lead: _shells(site_tables.get(int(layout._chan_site[lead])),
                                         _content((lead,), o)),
            _frame_key(tuple(chans), int(orbit_of_site[here])),
            _BandKey(group, _fibers, _share, _held, i, len(chans_of)),
            _order]
        kept, fate = _choose(cands, keys)
        winner = cands[kept]
        held_at.append(winner)
        for d in mine_of[i]:
            if d.get("matrices") and not d.get("partial"):
                matrices_carried[id(d)] = _meets(winner["chi"], _matrix_needs(d, winner))
        # THE LABELLINGS THE SITE IRREPS LEAVE OPEN: every candidate the
        # Hamiltonian and the declarations leave that gives each multiplet of
        # the part the site irrep the one kept gives it, with the key that set
        # it aside -- the chooser's own decision, whatever freedom it was
        want = [_site_irrep(_source(winner)[c], winner["chi"]) for c in chans]
        open_of.append((cands, [(j, names[min(int(fate[j]), len(names) - 1)])
                                for j, o in enumerate(cands)
                                if j != kept and [_site_irrep(_source(o)[c], o["chi"])
                                                  for c in chans] == want]))
        if winner["W"] is None:
            # a character on ``D``
            chosen.append(winner["chi"])
            candidates.append([(o["chi"], _source(o)) for o in cands])
        else:
            # A PSEUDOSPIN'S COPIES, or uncoupled orbitals a symmetry
            # exchanges: the representation chosen is assembled whole
            for c in chans:
                raw_after[int(c)] = dict(winner["contents"][c])
            swaps.append((tuple(int(a) for a in rec["states"]),
                          _in_tables_gauge(rec, winner, group=group, layout=layout,
                                           syms=syms, u_pos=u_pos, cperm=cperm,
                                           site_tables=site_tables, rotations=rotations,
                                           words=words)))
            chosen.append({})
            # what each multiplet holds under the representations the
            # declarations leave, the rows' own reading of how they are known
            candidates.append([({}, o["contents"]) for o in cands])
            if int(rec["dim"]) > 1:
                # a pseudospin's own record; an orbit of one copy holds none
                ranked, lead_c = [], int(rec["lead"])
                for o in sorted(options, key=lambda o: (keys[-7](o), keys[-6](o),
                                                        keys[-5](o), keys[-4](o), _order(o))):
                    content = o["contents"][lead_c]
                    if not any(set(content) == set(r) and all(
                            abs(complex(content[x]) - complex(r[x])) <= 1e-6 for x in content)
                            for r in ranked):
                        ranked.append(content)
                declared = bool(mine_of[i]) or any(i in spanned[id(d)] for d in shared)
                spaces_out.append({"record": rec, "content": dict(winner["contents"][lead_c]),
                                   "contents": tuple(ranked), "declared": declared,
                                   "open": len({_site_irrep(o["contents"][lead_c], {})
                                                for o in cands})})
    # ... AND THE BAND LABELS EACH OF THEM GIVES, read by the chooser's own band
    # key with every other part at the labelling kept: those that label a band
    # differently are the choice the band labels rest on
    labelled = []
    for i, (cands, ties) in enumerate(open_of):
        if not ties:
            continue
        reader = _BandKey(group, _fibers, _share, _held, i, len(chans_of))
        mine = reader.reading(held_at[i])
        others = tuple((name, got) for name, got in
                       ((name, reader.reading(cands[j])) for j, name in ties) if got != mine)
        if others:
            labelled.append({"part": i, "chosen": mine, "others": others})
    # the matrices built for representations no labelling returns are not kept
    winners = {id(getattr(w, "_cand", None)) for w in (o["W"] for o in held_at
                                                       if o["W"] is not None)}
    for rec in records:
        if rec is not None and rec.get("induced") is not None:
            for cand in rec["candidates"]:
                if id(cand) not in winners:
                    cand.release()
    of_chan = {}
    for i, chans in enumerate(chans_of):
        for c in chans:
            of_chan[int(c)] = chosen[i]
    return {"of_chan": of_chan, "swaps": tuple(swaps), "raw": raw_after, "chosen": chosen,
            "candidates": candidates, "labelled": tuple(labelled), "parts": chans_of,
            "part_states": part_states, "spaces": tuple(spaces_out), "lift_trace": lift_trace,
            "matrices_carried": matrices_carried}


def _assembled(D, choice, syms, layout):
    r"""``{position: matrix}`` -- the representation with the labelling in it.

    The representation each pseudospin's copies carry, as chosen, on its
    orbit's states; then every multiplet's columns -- its block, carrying it to
    its image -- times the value its coupled part's character takes on the
    operation: a character per part, since the operations carry each part onto
    itself, and one crystal-wide character multiplies the whole matrix.
    """
    out = {}
    for p, M in D.items():
        ind = int(syms[p].ind)
        N = np.array(M, dtype=complex, copy=True)
        for states, W in choice["swaps"]:
            if p in W:
                st = np.asarray(states, dtype=int)
                N[np.ix_(st, st)] = W[p]
        for c in range(int(layout._nchan)):
            f = complex(choice["of_chan"][int(c)].get(ind, 1.0))
            if abs(f - 1.0) <= 1e-12:
                continue
            N[:, np.asarray(layout._chan_states[int(c)], dtype=int)] *= f
        out[p] = N
    return out


def _atomic_limit_table(raw, worst, decls, choice, *, group, assignment, model, layout,
                        syms, kept, checks, pool, site_tables, ident, note="", spin=None,
                        ref_chan):
    r"""The atomic-limit table of the representation returned, read off it and nothing else.

    Every character here was measured before the labelling was chosen
    (:func:`.._declaration._multiplet_characters`), and the labelling itself is
    :func:`_labelling`'s; this writes them down.  A row's site irrep is what
    its multiplet carries under the labelling returned, named from the
    tables (:func:`_named_content`).  How it is known
    (:attr:`AtomicLimitRow.determined_by`) is ``"declaration"`` for a declared
    row the returned matrices carry, ``"declaration not carried"`` for one they
    do not -- a declaration the measurement could not read, such as a content
    declared on part of what a multiplet holds, which no labelling of these
    blocks carries together with the rest -- and otherwise read off the
    candidates of the row's coupled part: one site irrep across all of them is
    ``"hamiltonian"``, several ``"lowest-l"``.

    Characters live on **multiplets** (``layout._chan_of[a]`` is orbital
    ``a``'s, and ``layout._chan_orbs[c]`` its orbitals); the record is still one
    row per model orbital, so a declared spinor pair's two orbitals report the
    one character of the pair, each flagged ``up``/``down`` in the note and in
    :meth:`AtomicLimitTable.summary`.
    """
    chan_of, chan_orbs = layout._chan_of, layout._chan_orbs
    lattice = np.asarray(group.lattice, dtype=float)
    ind_of = {p: int(syms[p].ind) for p in kept}
    names = {ind_of[p]: _sitechars.operation_name(syms[p], lattice)
                        + ("'" if syms[p].time_reversal else "") for p in kept}
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    norb = int(model.norb)
    of_chan, lift_trace = choice["of_chan"], choice["lift_trace"]
    part_of_chan = {int(c): i for i, chans in enumerate(choice["parts"]) for c in chans}
    final = {c: {ind: complex(of_chan[c].get(int(ind), 1.0)) * v for ind, v in chi.items()}
             for c, chi in raw.items()}

    def _site_irrep(theta, chi):
        return tuple(sorted(
            (int(ind),
             round(float(np.real(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6),
             round(float(np.imag(v * complex(chi.get(int(ind), 1.0))
                                 * lift_trace.get(int(ind), 1.0))), 6))
            for ind, v in theta.items()))

    # WHETHER EACH DECLARATION IS CARRIED by the matrices returned.  Where the
    # verdict stage measured a declaration this is so by construction; where it
    # could not -- a content declared on part of what a multiplet holds, a
    # block the hoppings leave free of the scale it was read against, blocks
    # too noisy to settle -- a declaration no labelling of these blocks
    # carries is said on its rows, never returned as though it held.
    carried = {}
    for d in decls:
        if d.get("partial"):
            carried[id(d)] = _holds_part(d, lambda c: of_chan[int(c)], raw, site_tables,
                                         lift_trace)
        else:
            # read on the characters returned, so it asks the trivial character
            carried[id(d)] = _meets({}, _declared_needs(d, final, lift_trace))
        if d.get("matrices"):
            # ... and declared matrices on the matrices returned, which the
            # labelling read as it chose (:func:`_labelling`)
            carried[id(d)] = (carried[id(d)]
                              and choice.get("matrices_carried", {}).get(id(d), True))

    decl_of = {}
    for d in decls:
        for a in d["orbitals"]:
            decl_of[int(a)] = d
    # WHAT EACH DECLARATION IS RETURNED AS: what the multiplets it was read on
    # carry together under the labelling returned -- its own multiplet's site
    # irrep, or for a content read over several of a site's multiplets their
    # sum (`_declared_blocks`), which no one row names
    returned = {}
    for d in decls:
        a0 = int(d["orbitals"][0])
        ch0, tab0 = int(chan_of[a0]), site_tables[int(site_of[a0])]
        if d.get("block") is not None or layout._chan_kind[ch0] != "measured":
            theta0 = {ind: sum(final[int(c)].get(int(ind), 0.0) for c in d["channels"])
                      for ind in tab0.indices if int(ind) in final[ch0]}
        else:
            theta0 = dict(final[ch0])
        returned[id(d)] = _named_content(tab0, _copies_of_one_irrep(tab0, theta0)[0])[0]
    spin_note = {}
    if spin is not None:
        for c, (up, dn) in enumerate(spin.pairs):
            spin_note[int(up)] = f"spin up of pair {c}, partner orbital {int(dn)}"
            spin_note[int(dn)] = f"spin down of pair {c}, partner orbital {int(up)}"
    site_symbols = {s: _sitechars.point_group_symbol(group, tab.indices)
                    for s, tab in site_tables.items()}
    wyckoff = _wyckoff_sites(group, assignment)
    functions = _declared_functions(decls)
    rows = []
    for a in range(norb):
        s = int(site_of[a])
        tab = site_tables[s]
        ch = int(chan_of[int(a)])
        d = decl_of.get(a)
        # the declaration written for THIS orbital, which a content declared
        # over several multiplets holds among its parts
        part = d if d is None or not d.get("groups") else next(
            g for g in d["groups"] if int(a) in {int(o) for o in g["orbitals"]})
        # EVERY ROW IS ITS OWN MEASURED MULTIPLET'S, whatever the spelling: a
        # content declared over several of a site's multiplets is read against
        # what they carry together, and stays the row's `declared_irrep`, while
        # each row names what its own multiplet carries
        if d is not None and d.get("block") is None:
            members, chans = tuple(d["orbitals"]), tuple(d["channels"])
        else:
            members, chans = tuple(int(o) for o in chan_orbs[ch]), (ch,)
        theta = (dict(final[int(ch)]) if layout._chan_kind[ch] == "measured"
                 else {ind: sum(final[int(c)].get(int(ind), 0.0) for c in chans)
                       for ind in tab.indices if int(ind) in final[ch]})
        # A multiplet holding several copies of one site irrep -- copies written
        # in a basis rotated from site to site, which no orbital subset splits --
        # is an isotypic component, and every vector of it carries that irrep:
        # each of its rows is the irrep, with the irrep's own characters and
        # dimension, and the site holds as many copies as the rows say.  A
        # multiplet the site's own operations glue across different irreps
        # keeps its sum.
        theta = _copies_of_one_irrep(tab, theta)[0]
        # The returned label is what the orbitals carry in the site group of the
        # group that is RETURNED, read off the multiplet's character and nothing
        # else, and named for the site irrep the STATES carry -- the orbital
        # character times the operations' own lift, the double-valued site corep
        # where the states carry spin.
        label, named_by = _named_content(tab, theta)
        note_a = "" if part is None else part["note"]
        if d is not None:
            determined_by = "declaration" if carried[id(d)] else _NOT_CARRIED
            if not carried[id(d)]:
                needs = _declared_needs(d, raw, lift_trace) if not d.get("partial") else None
                alone = any((_holds_part(d, lambda _c, chi=chi: chi, raw, site_tables,
                                         lift_trace) if needs is None
                             else _meets(chi, needs)) for chi in pool)
                said = d["label"] or repr(d["spec"])
                if d.get("matrices"):
                    said = f"{d['label']} as generator matrices" if d["label"] else "matrices"
                note_a = ((note_a + "; " if note_a else "")
                          + f"declared {said}, not carried: no one-dimensional character "
                          "the hoppings leave open "
                          + ("gives these orbitals those matrices" if d.get("matrices")
                             else "carries it")
                          + (" together with the other declarations" if alone else "")
                          + f", and the Hamiltonian's own blocks carry "
                          f"{returned[id(d)] or 'a character no irrep of this site group names'}"
                          " under the one returned")
        else:
            options = choice["candidates"][part_of_chan[ch]]
            seen = {_site_irrep({int(ind): v for ind, v in (raw if src is None else src)[ch].items()
                                 if int(ind) in theta}, chi) for chi, src in options}
            determined_by = "hamiltonian" if len(seen) <= 1 else "lowest-l"
        if a in spin_note:
            note_a = (note_a + "; " if note_a else "") + spin_note[a]
        content = None if part is None else part.get("content")
        written = dict(getattr(content, "written", ()) or ())
        if (named_by == "unnamed") == bool(label):    # pragma: no cover - a defect guard
            raise InternalError(
                f"the site table of site {s} names orbital {int(a)} {label!r}, "
                f"named by {named_by!r}. A label the tables gave is a label and says "
                "where it comes from; only a row nothing named may be blank, and it "
                "reads `named_by = \"unnamed\"`. The two disagreeing is a defect in "
                "this package, not a tolerance to loosen.")
        here = wyckoff.get(s)
        rows.append(AtomicLimitRow(
            orbital=int(a), site=s,
            wyckoff="" if here is None else f"{int(here.multiplicity)}{here.letter}",
            site_symmetry=site_symbols.get(s, ""), function=functions.get(a, ""),
            multiplet=members, irrep=label, named_by=named_by,
            determined_by=determined_by,
            dim=int(round(float(np.real(theta.get(int(syms[ident].ind), 1.0)))
                          * (2 if group.spinor else 1))),
            site_group=tuple(int(i) for i in tab.indices),
            character={int(k): complex(v) for k, v in theta.items()},
            generator_character={tab.names[int(r)]: complex(theta.get(int(r), 0.0))
                                 for r in tab.class_reps},
            declared_as=(None if part is None else written.get(int(a), part["spec"])),
            declared_irrep="" if content is None else str(content.label),
            declared_in="" if content is None else str(content.declared_in),
            reduced_in="" if content is None else str(content.reduced_in),
            declaration_broke=() if content is None else tuple(content.broke),
            note=note_a, residual=float(worst)))

    return AtomicLimitTable(
        orbitals=tuple(rows),
        declarations=tuple({"orbitals": tuple(d["orbitals"]), "declared_as": d["spec"],
                            "declared_irrep": d["label"], "dim": d["dim"],
                            "note": d["note"],
                            "declared_in": (d["content"].declared_in
                                            if d.get("content") is not None else ""),
                            "reduced_in": (d["content"].reduced_in
                                           if d.get("content") is not None else ""),
                            "broke": (tuple(d["content"].broke)
                                      if d.get("content") is not None else ()),
                            "mulliken": (dict(d["content"].mulliken)
                                         if d.get("content") is not None else {}),
                            "returned": returned[id(x)]}
                           for x in decls for d in x.get("groups", (x,))),
        checks=_returned_rows(checks),
        frame=dict(names), residual=float(worst), site_tables=dict(site_tables),
        spin=spin,
        _labellings={"parts": tuple(tuple(int(a) for a in states)
                                    for states in choice["part_states"]),
                     "choices": tuple(choice["labelled"])})


def _wyckoff_sites(group, assignment):
    """``{site: SiteReference}`` — each site's Wyckoff position in the tables' setting.

    Multiplicity and letter, as :func:`..groups.wyckoff_sites_tables` reads
    them; a site is absent where the tables do not reach the cell, which is the
    same condition under which the assignment carries no letter for it.
    """

    if not tuple(getattr(assignment, "wyckoff_tables", ()) or ()):
        return {}
    sites = _tables.wyckoff_sites_tables(group, assignment.site_positions)
    if isinstance(sites, Refusal):
        return {}
    return dict(enumerate(sites))


def _declared_functions(decls) -> dict:
    """``{orbital: word}`` — the orbital function each orbital was declared to be.

    The word ``characters=`` was given, spelled as ``characters=`` spells it,
    on every orbital a function word was written for, whatever the dimension of
    the multiplet it names: a shorthand (``"px"``), or a shell (``"p"``) on
    every orbital it was written on.  An orbital declared by its irrep, a sum
    of irreps or generator characters, or not declared at all, has none: which
    function a Wannier function is inside its site is a choice of basis the
    Hamiltonian does not measure, and its site irrep is what says what it
    carries.
    """
    out = {}
    for d in decls:
        content = d.get("content")
        for a, word in (getattr(content, "words", ()) or ()):
            out[int(a)] = str(word)
    return out
