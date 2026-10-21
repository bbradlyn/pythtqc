"""Site irreps named off the tables: each row of a measured site-group character
table by the elementary band representation it induces from the site's Wyckoff
position, otherwise by the name the tables' origins give its point group; and
the site tables a declaration is read on, the rows named that way.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import itertools
import os
import re

import numpy as np

from . import _sitechars
from . import _spacegroup
from . import _tables
from .refusal import InternalError, Refusal
from ._tolerance import CHARACTER, _INT_TOL, _TABLE_MATRIX_TOL
from ._tables import _letter_only, ebr_rows, kpoint_of_label
from ._matrices import _primed_partners


def _site_tables(group, assignment, kept_u, kept_au, sites):
    """``{site: SiteGroupTable}`` for the measured site groups of ``sites``.

    Sites that share a site group (every site of one orbit does, up to
    conjugation, and symmetry-equivalent sites often literally do) share one
    computed table.  Each row is then named by the band representation it
    induces (:func:`.._naming._site_irrep_labels`), which is a statement about the
    site's Wyckoff position as well as about its group, so two sites share a
    table only when they share both.

    A row is named in the valuedness the model's own states carry: a spinless
    model's site irreps are the single-valued ones the EBR entries at that
    position are keyed by, and a spinful model's are the double-valued site
    coreps the group's own double-valued entries are keyed by
    (:func:`.._naming._site_corep_labels`).
    """
    letters = tuple(getattr(assignment, "wyckoff_tables", ()) or ())
    out, cache = {}, {}
    for s in sites:
        ind_u = tuple(int(i) for i in assignment.site_group[s] if int(i) in kept_u)
        ind_au = tuple(int(i) for i in assignment.site_group[s] if int(i) in kept_au)
        letter = letters[int(s)] if int(s) < len(letters) else ""
        key = (ind_u, ind_au, letter)
        if key not in cache:
            table = _site_group_table(group, ind_u, antiunitary=ind_au)
            cache[key] = _named_from_tables(group, assignment, int(s), table)
        out[int(s)] = cache[key]
    return out


def _site_group_table(group, indices, antiunitary=()):
    """A site group's character table, each row under the tables' name for that irrep.

    :func:`.._sitechars.site_group_table` computes the table, and each row is
    named by one rule.  The tables' name for that irrep of the site's point
    group, read at the origin of a symmorphic group
    (:func:`.._naming._point_group_irreps`), comes first, so that one irrep has one
    name at every site of a model (``named_by = "point group"``).  Where those
    names leave the row open, because they follow an orientation no table fixes
    at this site, the row keeps the name its Mulliken symbol gives it
    (``"Mulliken"``), where that is a name rather than a placeholder
    (:func:`.._sitechars.is_placeholder`).  A row neither names is unnamed.
    """

    table = _sitechars.site_group_table(group, indices, antiunitary=antiunitary)
    names = _point_group_irreps(group, table) or ("",) * len(table.labels)
    taken = {name for name in names if name}
    labels, named_by = [], []
    for name, own in zip(names, table.labels):
        if name:
            labels.append(name)
            named_by.append("point group")
        elif own and not _sitechars.is_placeholder(own) and own not in taken:
            labels.append(own)
            named_by.append("Mulliken")
        else:
            labels.append("")
            named_by.append("unnamed")
    return replace(table, labels=tuple(labels), named_by=tuple(named_by))


class _SiteTableOf:
    """The character table of one site's group inside a subgroup, built once per subgroup.

    What part of a site's content is read against on a smaller group
    (:func:`.._declaration._covariant_subgroup`): a containment is a decomposition on the site
    group the subgroup leaves, which needs only its characters.  A content
    written in the site coreps is compared as coreps, and that needs the rows
    named as the coreps the states carry (:func:`_site_tables`), the
    antiunitary operations the subgroup keeps at the site gluing their
    complex pairs.  Called with the ``.ind`` of the site group's unitary
    operations inside the subgroup; ``None`` where they carry no table.
    """

    def __init__(self, group, assignment, site, *, corep=False):
        self.group, self.assignment = group, assignment
        self.site, self.corep = int(site), bool(corep)
        self._built = {}

    def __call__(self, indices):
        key = frozenset(int(i) for i in indices)
        if key not in self._built:
            own = [int(i) for i in self.assignment.site_group[self.site]]
            try:
                if self.corep:
                    partners = _primed_partners(self.group, own)
                    primed = {partners[i] for i in key if i in partners}
                    self._built[key] = _site_tables(self.group, self.assignment, set(key),
                                                    primed, [self.site])[self.site]
                else:
                    self._built[key] = _site_group_table(
                        self.group, tuple(i for i in own if i in key))
            except _sitechars.DeclarationError:   # pragma: no cover - a site group closes
                self._built[key] = None
        return self._built[key]


def _named_from_tables(group, assignment, site, table):
    """``table`` with the rows the EBR entries at the site's position name.

    A row the tables reach is named by the band representation it induces: the
    entry at the site's own Wyckoff position whose content it reproduces
    (``named_by = "tables"``).  A row those entries do not name keeps the site
    point group's own name, ``named_by = "point group"``: the tables' own name
    for that irrep of the point group, read at the origin of a symmorphic group,
    single-valued for a spinless row (:func:`.._naming._point_group_irreps`) and
    double-valued for a spinful one (:func:`.._naming._point_group_names`).  A
    spinless row those names leave open keeps its Mulliken symbol, ``named_by =
    "Mulliken"``, where that is a name (:func:`_site_group_table`).  A row
    nothing names is left **empty**, ``named_by = "unnamed"``, and its generator
    characters are the whole statement: naming is a reading of the tables and
    never a condition for a build.

    Where the states carry spin the returned name is the double-valued site
    corep, and the row's **orbital** factor — single-valued, and what a
    declaration states — is named at the same position by the same group's
    single-valued entries (:attr:`.._sitechars.SiteGroupTable.declarable`).
    Both come off the tables; the second is never returned.
    """

    if group.spinor:
        # The states at this site carry the row times the operations' spin-1/2
        # lift, and it is that -- the double-valued site corep -- the tables
        # name.  A declaration states the ORBITAL factor instead, which is the
        # row itself: single-valued, and named at the same position by the same
        # group's single-valued entries.  Both names are read from the tables,
        # from the group's two table files; only the corep is ever returned.
        labels, named_by = _site_corep_labels(group, assignment, int(site), table)
        orbital = _site_orbital_labels(group, assignment, int(site), table)
        return replace(table, labels=labels, named_by=named_by,
                       declarable=tuple(name or own for name, own
                                        in zip(orbital, table.labels)))
    # the single-valued names: one tables entry per row, and the tables' own
    # name for the point group's irrep where the entries do not name the row,
    # so a row needs no second spelling
    labels, named_by = _site_irrep_labels(group, assignment, int(site), table)
    return replace(table, labels=labels, named_by=named_by,
                   declarable=("",) * len(labels))


def _contained(tab, held, want, *, corep=False, asked=None):
    """Whether the content ``held`` contains the content ``want``, both given class by class of ``tab``.

    What is left of ``held`` once ``want`` is taken away has to be a content of
    the site group.  A content written in the site coreps (``corep``) is
    compared as the coreps it lifts to, and where a row's lift has no name,
    ``held`` has to contain an orbital content whose characters are
    ``want``'s on the classes ``asked``, those whose spin-1/2 lift has a
    nonzero trace.
    """
    if not corep:
        return _rows_of(tab, np.asarray(held, dtype=complex)
                        - np.asarray(want, dtype=complex)) is not None
    holds, stated = _rows_of(tab, held), _rows_of(tab, want)
    if holds is None or stated is None:
        return False
    have = _sitechars.corep_counts(tab, holds)
    need = _sitechars.corep_counts(tab, stated)
    if have is not None and need is not None:
        return all(have.get(c, 0) >= n for c, n in need.items())
    counts = [holds.get(p, 0) for p in range(len(tab.characters))]
    asked = [True] * len(want) if asked is None else list(asked)
    for part in itertools.product(*(range(k + 1) for k in counts)):
        chi = sum((m * np.asarray(tab.characters[p], dtype=complex)
                   for p, m in enumerate(part) if m), np.zeros(len(want), dtype=complex))
        if all(abs(complex(chi[i]) - want[i]) <= CHARACTER
               for i in range(len(want)) if asked[i]):
            return True
    return False


def _rows_of(tab, chi):
    """``{row: copies}`` of a content of ``tab``, given class by class, or ``None`` where it is none."""
    out = {}
    for p, n in enumerate(_sitechars.decompose(tab, np.asarray(chi, dtype=complex))):
        n = complex(n)
        if (abs(n.imag) > CHARACTER or abs(n.real - round(n.real)) > CHARACTER
                or round(n.real) < 0):
            return None
        if round(n.real):
            out[p] = int(round(n.real))
    return out


# --------------------------------------------------------------------------
# naming a site irrep by the band representation it induces
# --------------------------------------------------------------------------
#
# A site irrep of a measured site group has no name of its own: the names a
# physicist looks up are the tables' names for the irreps of the Wyckoff
# positions of the identified space group.  What connects the two is the band
# representation the site irrep induces, which is exactly what an EBR column
# holds -- so the name is read by inducing and matching, never by a character
# lookup that would need an orientation convention no measurement carries.


def _unitary_frame(group):
    """``(positions, inverse, product table, position of each .ind)``.

    The product table comes back as nested lists: the conjugations below run
    ``|P|^2`` times per site and an element of a numpy array is an object to
    build, while an element of a list is one to read.
    """
    prod, _, _ = group.get_product_table(get_diff=True)
    prod = np.asarray(prod, dtype=int).tolist()
    syms = group.symmetries
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    unitary = [p for p, s in enumerate(syms) if not s.time_reversal]
    ident = next(p for p in unitary
                 if np.array_equal(np.asarray(syms[p].rotation), np.eye(3, dtype=int))
                 and np.allclose(np.asarray(syms[p].translation) % 1.0, 0.0, atol=1e-8))
    inverse = {p: next(q for q in unitary if prod[p][q] == ident) for p in unitary}
    return unitary, inverse, prod, pos_of


def _decompose_against_coreps(table, chi, *, where, tol=_INT_TOL):
    """Multiplicity of every corep of ``table`` in the character ``chi``.

    Orthogonality over the little group's unitary operations, with the
    tabulated characters' own rounding absorbed by ``tol``.  A multiplicity
    that is not an integer is a defect in this package and raises, since the
    character being decomposed is a character of that little group by
    construction.

    Parameters
    ----------
    table : dict
        ``{corep label: {ind: character}}``, as :func:`..groups.corep_table`
        holds it.
    chi : dict
        ``{ind: character}`` over the same operations.
    where : str
        What was being decomposed, for the message of a defect.
    tol : float, optional
        Integrality tolerance.

    Returns
    -------
    dict
        ``{corep label: multiplicity}``, the zeros dropped.
    """
    inds = sorted(next(iter(table.values())))
    out = {}
    for label, char in table.items():
        norm = sum(abs(complex(char[i])) ** 2 for i in inds)
        m = sum(complex(chi[i]) * np.conj(complex(char[i])) for i in inds) / norm
        if abs(m.imag) > tol or abs(m.real - round(m.real)) > tol:
            raise InternalError(
                f"the induced content of {where} contains {label} with multiplicity "
                f"{m:.6g}, which is not an integer: a character induced from a site "
                "group is a character of the little group and decomposes over its "
                "coreps with whole multiplicities. That is arithmetic in this package "
                "failing, not a tolerance to loosen.")
        if round(m.real):
            out[label] = int(round(m.real))
    return out


def _induction_plan(group, assignment, site, fixers, *, k=None, kpname=None):
    r"""Everything the induction at one point needs except the character itself.

    The weight of the site group's operation :math:`h` in the induced character
    of the little group's operation :math:`g` does not depend on which site
    irrep is being induced, so it is built once for a site and used for every
    row of its table — which is what keeps the naming group-sized rather than
    row-times-group-sized.

    Parameters
    ----------
    group : ..groups.GroupData
    assignment : ..groups.OrbitalAssignment
    site : int
    fixers : sequence of int
        ``.ind`` of the site's unitary fixers.
    k : array, optional
        The point in the model's reduced coordinates; ``None`` is
        :math:`\Gamma`.
    kpname : str, optional
        Forced tabulated label, forwarded to :func:`..groups.corep_table`.

    Returns
    -------
    tuple or None
        ``(corep table, {ind of g: {ind of h: weight}})``, and ``None`` where
        the tabulated characters at that point are unavailable or the site's
        fixers are a proper subgroup of what fixes it.
    """
    fixer_inds = {int(i) for i in fixers}
    if not fixer_inds:
        return None
    point = k
    if point is None:
        point = _gamma_point(group)
        if isinstance(point, Refusal):
            return None
        kpname = kpname or "GM"
    table = _tables.corep_table(group, point, kpname=kpname)
    if isinstance(table, Refusal):
        return None
    unitary, inverse, prod, pos_of = _unitary_frame(group)
    syms = group.symmetries
    held = {pos_of[i] for i in fixer_inds}
    weights: dict = {}
    if k is None:
        for i in sorted(next(iter(table.values()))):
            p, row = pos_of[int(i)], {}
            for x in unitary:
                y = prod[inverse[x]][prod[p][x]]
                if y in held:
                    ind = int(syms[y].ind)
                    row[ind] = row.get(ind, 0.0) + 1.0 / len(held)
            weights[int(i)] = row
        return table, weights
    k3 = np.asarray(group.k_to_3d(np.asarray(k, dtype=float)[None, :])[0], dtype=float)
    perm = np.asarray(assignment.permutations, dtype=int)
    shift = np.asarray(assignment.lattice_shift, dtype=float)
    orbit = sorted({int(perm[p, int(site)]) for p in unitary})
    carry = {j: next(p for p in unitary if int(perm[p, int(site)]) == j) for j in orbit}
    for i in sorted(next(iter(table.values()))):
        p, row = pos_of[int(i)], {}
        for j in orbit:
            if int(perm[p, j]) != j:
                continue
            q = carry[j]
            ind = int(syms[prod[inverse[q]][prod[p][q]]].ind)
            if ind not in fixer_inds:
                return None
            row[ind] = row.get(ind, 0j) + np.exp(-2j * np.pi * float(k3 @ shift[p, j]))
        weights[int(i)] = row
    return table, weights


def _induced_content(group, assignment, site, chi, *, k=None, kpname=None,
                     plan=None) -> dict:
    r"""The band representation a site character induces, as corep multiplicities.

    ``chi`` is one character of the site group of ``site`` — the operations of
    ``group`` that fix it, measured by ``.._tables._assign_orbitals`` — and what
    comes back is the content of the band representation it induces, at ``k``.

    At :math:`\Gamma` the induction is the real-space one,
    :math:`\chi_{\rm ind}(g) = |H|^{-1}\sum_{x\in P}[x^{-1}gx\in H]\,
    \chi(x^{-1}gx)` over the unitary operations :math:`P` modulo the lattice,
    taken through the group's own product table.  Away from it the sum is the
    Frobenius one over the sites the operation fixes, each carrying the phase
    :math:`e^{-2\pi i\,k\cdot m}` of its own displacement
    :math:`m = R\tau_j + t - \tau_j`, the integer vector
    ``assignment.lattice_shift`` holds; that phase carries the operation's
    fractional translation and any reciprocal-lattice vector the little group
    needs, so no further factor enters.  The sign is the one the EBR rows are
    transported into (:func:`..groups.ebr_data`) and is what makes an induced
    content comparable with them at a point away from a TRIM.

    Parameters
    ----------
    group : ..groups.GroupData
    assignment : ..groups.OrbitalAssignment
        The orbit and the displacements the phases are read from.
    site : int
        Which site of ``assignment`` the character belongs to.
    chi : dict
        ``{ind: character}`` over the site's **unitary** fixers, which are its
        keys.  For a group with double-valued tables the character already
        carries the operations' spin-½ lift.
    k : array, optional
        The point, in the model's reduced coordinates.  ``None`` is
        :math:`\Gamma`.
    kpname : str, optional
        Forced tabulated label, forwarded to :func:`..groups.corep_table`.
    plan : tuple, optional
        :func:`_induction_plan` for this site and point, to be shared across the
        rows of one table instead of rebuilt for each.

    Returns
    -------
    dict
        ``{corep label: multiplicity}`` at that point, the zeros dropped.
        Empty where the transported character cannot be read on the orbit —
        the site's fixers are a proper subgroup of what fixes it — or where the
        tabulated characters at that point are unavailable.
    """
    if plan is None:
        plan = _induction_plan(group, assignment, int(site), tuple(chi),
                               k=k, kpname=kpname)
    if plan is None:
        return {}
    table, weights = plan
    chi_ind = {ind: sum(w * complex(chi[h]) for h, w in row.items())
               for ind, row in weights.items()}
    return _decompose_against_coreps(
        table, chi_ind, where=f"a site irrep of the site group of site {int(site)}")


def _gamma_point(group):
    """The tabulated :math:`\\Gamma` in the model's own reduced coordinates."""
    points = _tables.high_symmetry_points(group)
    if isinstance(points, Refusal):
        return points
    return points["GM"]


def _rows_at_letter(rows, letter) -> dict:
    """``{EBR name: content}`` of the entries of ``rows`` at one Wyckoff letter."""
    return {name.split("@")[0]: content for name, content in rows.items()
            if _letter_only(name.split("@")[1]) == str(letter)}


def _gamma_ties(group) -> dict | Refusal:
    r"""Which EBRs of one group share their :math:`\Gamma` content at a position.

    Two site irreps at one Wyckoff position can induce the same coreps at
    :math:`\Gamma` and part only away from it — ``B1↑G`` and ``B2↑G`` at ``2c``
    of P4mm are both ``GM5`` there and differ at X, M, A, R and Z — so a
    :math:`\Gamma` content alone names a site irrep at every position except
    these.  The list is a property of the group's EBR file and needs no model.

    Parameters
    ----------
    group : ..groups.GroupData

    Returns
    -------
    dict or Refusal
        ``{Wyckoff position: (tied EBR names, ...)}``, holding only the
        positions where two or more EBRs share a :math:`\Gamma` content.

    Refusals
    --------
    ``"ebr_tables"``, ``"irrep_tables"``, ``"ebr_convention"``, ``"lattice_not_descended"``
        propagated from :func:`ebr_rows`.
    """
    rows = ebr_rows(group)
    if isinstance(rows, Refusal):
        return rows
    return _gamma_ties_of(rows)


def _gamma_ties_of(rows) -> dict:
    """:func:`_gamma_ties` on EBR rows already in hand."""
    by_position: dict = {}
    for name, content in rows.items():
        base, position = name.split("@")
        key = tuple(sorted((lab, mult) for lab, mult in content.items()
                           if kpoint_of_label(lab) == "GM"))
        by_position.setdefault(position, {}).setdefault(key, []).append(base)
    out = {}
    for position, shared in by_position.items():
        tied = tuple(tuple(names) for names in shared.values() if len(names) > 1)
        if tied:
            out[position] = tied
    return out


def _site_irrep_names(group, assignment, site, table) -> tuple:
    r"""The tables' name of every row of one computed site-group character table.

    Each row is induced to the group's high-symmetry points and matched against
    the EBR entries at the site's Wyckoff letter in the tables' setting
    (``assignment.wyckoff_tables``): :math:`\Gamma` first, and, where the
    entries at that position leave more than one candidate there, the remaining
    points in the order :func:`..groups.high_symmetry_points` holds them until
    one entry is left.  Which positions of a group that can happen at is
    :func:`_gamma_ties`, read from the EBR file with no model in hand, and the
    list is checked against the induction: candidates that survive
    :math:`\Gamma` together must be entries that list holds as sharing a
    content, and a disagreement raises rather than naming a row on it.  A row of
    a group with double-valued tables is induced together with the operations'
    spin-½ lift, so what it matches is a double-valued EBR entry.

    Raises
    ------
    ..refusal.InternalError
        The induction leaves two entries at :math:`\Gamma` that the group's own
        tie list does not carry as tied.

    Parameters
    ----------
    group : ..groups.GroupData
    assignment : ..groups.OrbitalAssignment
    site : int
    table : .._sitechars.SiteGroupTable
        The computed character table of that site's measured group.

    Returns
    -------
    tuple
        One entry per row of ``table``: the EBR entry's full name where the
        match is unique, and ``""`` where the position is not tabulated, the
        induced content is reducible, or two rows claim one entry.
    """
    blank = ("",) * len(table.labels)
    letters = tuple(assignment.wyckoff_tables or ())
    if int(site) >= len(letters) or not letters[int(site)]:
        return blank
    rows = ebr_rows(group)
    if isinstance(rows, Refusal):
        return blank
    at_letter = _rows_at_letter(rows, letters[int(site)])
    if not at_letter:
        return blank
    points = _tables.high_symmetry_points(group)
    if isinstance(points, Refusal):
        return blank
    tied_here = ()
    for position, shared in _gamma_ties_of(rows).items():
        if _letter_only(position) == letters[int(site)]:
            tied_here = shared
    lift = None
    if group.spinor:
        lift = {int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
                for s in group.symmetries}
    order = ["GM"] + [name for name in points if name != "GM"]
    plans: dict = {}
    characters = [{int(i): (table.character_on(row, int(i))
                            * (1.0 if lift is None else lift[int(i)]))
                   for i in table.indices}
                  for row in range(len(table.labels))]
    named = []
    for chi in characters:
        left = set(at_letter)
        for point in order:
            if point not in plans:
                plans[point] = _induction_plan(
                    group, assignment, int(site), tuple(table.indices),
                    k=None if point == "GM" else points[point], kpname=point)
            content = ({} if plans[point] is None
                       else _induced_content(group, assignment, int(site), chi,
                                             plan=plans[point]))
            if not content:
                left = set()
                break
            left &= {name for name, full in at_letter.items()
                     if {lab: mult for lab, mult in full.items()
                         if kpoint_of_label(lab) == point} == content}
            if len(left) <= 1:
                break
            if not any(left <= set(names) for names in tied_here):
                raise InternalError(
                    f"the induced content of a site irrep at {letters[int(site)]} of "
                    f"#{group.number_str} matches {sorted(left)} at {point}, which the "
                    "tie list the EBR file itself carries does not hold as sharing a "
                    "content. Those are two readings of one table and have to agree; "
                    "the two disagreeing is a defect in this package.")
        named.append(next(iter(left)) if len(left) == 1 else "")
    claimed = [name for name in named if name]
    return tuple(name if name and claimed.count(name) == 1 else "" for name in named)


def _site_irrep_labels(group, assignment, site, table) -> tuple:
    r"""The site-irrep labels of one computed table, and where each name comes from.

    A name the tables carry at the position is the part of the EBR entry
    before ``↑G``, which is what a Wyckoff position is looked up by; a row the
    entries there do not reach keeps the table's own name and where it came
    from (``.._naming._site_group_table``): the tables' name for that irrep of
    the site's point group, its Mulliken symbol, or none.  Labels have to be
    unique within one table, so a name from the entries that collides with a
    kept one leaves the whole table on its own names.

    Parameters
    ----------
    group : ..groups.GroupData
    assignment : ..groups.OrbitalAssignment
    site : int
    table : .._sitechars.SiteGroupTable

    Returns
    -------
    tuple
        ``(labels, named_by)``, one entry of each per row: the label, and
        ``"tables"``, ``"point group"``, ``"Mulliken"`` or, for a row nothing
        names, ``"unnamed"``.
    """
    names = _site_irrep_names(group, assignment, int(site), table)
    own = tuple(table.labels)
    mine = tuple(table.named_by) if len(table.named_by) == len(own) else (
        tuple("point group" if label else "unnamed" for label in own))
    labels, named_by = [], []
    for row, name in enumerate(names):
        base = str(name).split("↑G")[0] if name else ""
        if base:
            labels.append(base)
            named_by.append("tables")
        else:
            labels.append(own[row])
            named_by.append(mine[row])
    named = [label for label in labels if label]
    if len(set(named)) != len(named):
        return own, mine
    return tuple(labels), tuple(named_by)


# --------------------------------------------------------------------------
# naming a site rep by the entries its induced content is the sum of
# --------------------------------------------------------------------------
#
# A row of the computed table of a spinful model's site group carries the
# *orbital* factor; what the model's states carry there is that row times the
# operations' spin-1/2 lift, which is a double-valued representation of the site
# group and need not be irreducible -- a ``T`` row of a tetrahedral site times
# the lift is a double-valued pair.  Its name is therefore the sum of the
# double-valued site irreps it contains, and those are read the same way a
# single one is: by the band representation the whole thing induces, matched
# against the EBR entries at the position.

#: Largest multiplicity of one EBR entry a site representation is searched for,
#: and the most matching combinations kept before the search gives up.  Both are
#: far above anything a site group of order <= 48 can produce: a row of
#: dimension ``d`` times the lift induces ``2d`` times the position's
#: multiplicity bands, and an entry carries at least two of them.
_MAX_ENTRY_MULTIPLICITY = 4

_MAX_COMBINATIONS = 64


def _entry_combinations(contents, target, *, cap=_MAX_COMBINATIONS):
    """Which multisets of ``contents`` sum to ``target``.

    ``contents`` is ``{entry name: {corep label: multiplicity}}`` at one point
    and ``target`` the induced content there.  What comes back is every multiset
    of entry names that sums to it, each in ``contents``' own order and with its
    repeats written out, and an empty list where nothing does.

    The search is exhaustive with two ceilings (:data:`_MAX_ENTRY_MULTIPLICITY`,
    ``cap``), which is what keeps a pathological table from turning a naming
    into a search: reaching either is reported as "no name", never as a wrong
    one.
    """
    names = list(contents)
    labels = sorted({lab for c in contents.values() for lab in c} | set(target))
    want = tuple(int(target.get(lab, 0)) for lab in labels)
    vecs = [tuple(int(contents[n].get(lab, 0)) for lab in labels) for n in names]
    out: list = []

    def walk(i, rest, acc):
        if not any(rest):
            out.append(tuple(acc))
            return
        if i >= len(names) or len(out) >= cap or not any(rest[j] > 0 for j in range(len(rest))):
            return
        here = vecs[i]
        if not any(here):                     # an entry with no content at this point
            walk(i + 1, rest, acc)
            return
        left = rest
        for n in range(_MAX_ENTRY_MULTIPLICITY + 1):
            walk(i + 1, left, acc + [names[i]] * n)
            left = tuple(a - b for a, b in zip(left, here))
            if any(x < 0 for x in left):
                break
    walk(0, want, [])
    return out


def _spelled(names) -> str:
    """``("-E1", "-E1", "-F") → "2x-E1 + -F"`` — a multiset of entries as one label."""
    order, count = [], {}
    for name in names:
        base = str(name).split("↑G")[0]
        if base not in count:
            order.append(base)
        count[base] = count.get(base, 0) + 1
    return " + ".join(base if count[base] == 1 else f"{count[base]}x{base}"
                      for base in order)


def _names_by_content(group, assignment, site, table, letter) -> tuple:
    r"""Each row of ``table``, named by the EBR entries at ``letter`` it induces.

    The row's character is taken with the operations' spin-1/2 lift where the
    group's tables are double-valued, so that what is matched is what the
    model's states carry at the site.  A site representation that is not
    irreducible is honestly the sum of the site irreps it contains, so a row may
    be named by several entries.  It is induced at :math:`\Gamma` first and at
    the other high-symmetry points while more than one multiset of entries
    reproduces it: the tie-break by which :func:`_site_irrep_names` names a
    single row, in the form a sum needs.

    Parameters
    ----------
    group : ..groups.GroupData
    assignment : ..groups.OrbitalAssignment
    site : int
    table : .._sitechars.SiteGroupTable
    letter : str
        The site's own Wyckoff letter, whose entries are the vocabulary.

    Returns
    -------
    tuple
        One label per row: the entries' names summed (``"-E1 + -F"``), and
        ``""`` where the entries at that letter reproduce the row's content in
        no way or in more than one way.
    """
    blank = ("",) * len(table.labels)
    if not letter:
        return blank
    # The tables' entries at a position are keyed by the irreps of the site
    # group that position HAS.  Where the represented operations are a proper
    # subgroup of what fixes the site -- a free-standing layer, whose
    # Hamiltonian leaves the sigma_h family undetermined -- a row of the
    # computed table is an irrep of the smaller group, and inducing it gives a
    # sum of entries that is a true statement about the band representation and
    # no name for the row.  So the naming declines, rather than spelling the
    # sum.  (:func:`_induction_plan` makes the same check away from Gamma.)
    fixers = {int(i) for i in assignment.site_group[int(site)]}
    unitary = {int(s.ind) for s in group.symmetries if not s.time_reversal}
    if fixers & unitary != {int(i) for i in table.indices}:
        return blank
    rows = ebr_rows(group)
    if isinstance(rows, Refusal):
        return blank
    at_letter = _rows_at_letter(rows, str(letter))
    if not at_letter:
        return blank
    points = _tables.high_symmetry_points(group)
    if isinstance(points, Refusal):
        return blank
    lift = None
    if group.spinor:
        lift = {int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
                for s in group.symmetries}
    order = ["GM"] + [name for name in points if name != "GM"]
    plans: dict = {}
    out = []
    for row in range(len(table.labels)):
        chi = {int(i): (table.character_on(row, int(i))
                        * (1.0 if lift is None else lift[int(i)]))
               for i in table.indices}
        left = None
        for point in order:
            if point not in plans:
                plans[point] = _induction_plan(
                    group, assignment, int(site), tuple(table.indices),
                    k=None if point == "GM" else points[point], kpname=point)
            content = ({} if plans[point] is None
                       else _induced_content(group, assignment, int(site), chi,
                                             plan=plans[point]))
            if not content:
                left = []
                break
            here = {name: {lab: m for lab, m in full.items()
                           if kpoint_of_label(lab) == point}
                    for name, full in at_letter.items()}
            got = _entry_combinations(here, content)
            left = got if left is None else [c for c in left if c in got]
            if len(left) <= 1:
                break
        out.append(_spelled(left[0]) if left and len(left) == 1 else "")
    return tuple(out)


def _induced_bands(group, assignment, site, chi) -> int:
    r"""How many bands the site character ``chi`` induces, or ``0``.

    The dimension of the band representation, read off its own :math:`\Gamma`
    content: the multiplicity of each corep times that corep's dimension.  It
    is the one number a **composite** site irrep — one no EBR column carries,
    so that no name can be matched — still shares with the SRSI tables, which
    carry it as :attr:`..srsi.MaximalKey.dim`.
    """
    plan = _induction_plan(group, assignment, int(site), tuple(chi))
    if plan is None:
        return 0
    table, _weights = plan
    content = _induced_content(group, assignment, int(site), chi, plan=plan)
    ident = next((int(s.ind) for s in group.symmetries
                  if not s.time_reversal
                  and np.array_equal(np.asarray(s.rotation), np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    if ident is None or any(ident not in table[label] for label in content):
        return 0                               # pragma: no cover - a guard
    return int(round(sum(m * float(np.real(table[label][ident]))
                         for label, m in content.items())))


def _site_corep_labels(group, assignment, site, table) -> tuple:
    r"""``(labels, named_by)`` for one site of a model whose states carry spin.

    Each row of the computed (single-valued) table stands for the
    double-valued site representation the model's states carry there — the
    row times the operations' lift — and takes the name the group's own
    double-valued EBR entries give it at the site's Wyckoff position,
    ``named_by = "tables"``: a sum there is a site representation that is not
    irreducible, which a row times the spin-1/2 lift can be.  Two rows can
    legitimately take **one** name, because two orbital characters differing by
    a character the lift's trace vanishes on are one site corep; that is the
    physics and not a collision.

    A row those entries do not name — the tables list none at the position, or
    none whose band representation is the row's, the row's being composite —
    takes the site point group's own double-valued name
    (:func:`_point_group_names`), ``named_by = "point group"``, as a
    single-valued row keeps the point group's own name.  One name for two rows
    whose states carry different characters would be two conventions meeting in
    one table, and the whole table then takes the point group's names.  A row
    nothing names reads ``""`` and ``"unnamed"``, which leaves its generator
    characters as the whole statement.
    """
    letters = tuple(assignment.wyckoff_tables or ())
    letter = letters[int(site)] if int(site) < len(letters) else ""
    blank = ("",) * len(table.labels)
    names = (_names_by_content(group, assignment, int(site), table, letter)
             if letter else blank)
    if all(names):
        return names, ("tables",) * len(names)
    own = _point_group_names(group, table)
    labels = tuple(name or mine for name, mine in zip(names, own))
    lift = {int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
            for s in group.symmetries}
    carried: dict = {}
    for row, label in enumerate(labels):
        chi = tuple(table.character_on(row, int(i)) * lift[int(i)] for i in table.indices)
        if label and any(abs(a - b) > _TABLE_MATRIX_TOL
                         for a, b in zip(carried.setdefault(label, chi), chi)):
            return own, tuple("point group" if mine else "unnamed" for mine in own)
    return labels, tuple("tables" if name else "point group" if mine else "unnamed"
                         for name, mine in zip(names, own))


# --------------------------------------------------------------------------
# the orbital factor's own name, from the single-valued tables
# --------------------------------------------------------------------------
#
# A row of a computed site table is the ORBITAL factor of what sits at the site,
# and it is single-valued whether or not the states carry spin: the spin is the
# other factor.  So a row has a single-valued name at the position too -- the one
# the group's single-valued EBR entries give the band representation the orbital
# alone induces -- and that name is what a declaration states.  It is read here
# exactly as a spinless model's rows are named, against the group's
# single-valued tables instead of its double-valued ones.


def _single_valued_view(group):
    r"""``group`` as its own **single-valued** tables read it.

    Which of the two table files a query loads — the irrep table, the EBR file,
    the transport record — is chosen by one flag, ``GroupData.spinor``, and
    nothing else: the tables are a property of the crystal and of the valuedness
    asked for.  So the view is this record with that flag turned over. Every
    other field is the same object — the same operations in the same order, the
    same lattice, the same ``refUC``/``shiftUC`` — and the multiplication table
    is therefore literally the same table, carried across rather than recomputed.

    **What may be asked of it.** A character and a Wyckoff letter: the view is
    consumed by :func:`_site_orbital_labels` alone, which induces the orbital
    factor of a site's rows and compares the multiplicities that come out with
    the single-valued entries at the letter. No representation is built on it,
    no lift is taken from it, and it is never returned or stored. That is what
    keeps it a reading of the tables rather than a single-valued question put to
    a spinful model, which is a question with no answer: a spinful model's
    *states* carry the electron's own spin-1/2 and its little-group content is
    double-valued, and that is what the returned label stays.
    """
    if not group.spinor:
        return group
    view = replace(group, spinor=False)
    # `replace` starts a fresh product-table memo, deliberately, because a table
    # computed for another `_sg` would be the wrong one.  Here `_sg` is the same
    # object, so the memo is the right one and copying it is what keeps the
    # second reading free.
    view._product_tables.update(group._product_tables)
    return view


def _site_orbital_labels(group, assignment, site, table) -> tuple:
    r"""The single-valued name of every row of one computed site table.

    The row's *orbital* name at the site's Wyckoff position: what the tables
    call the site irrep the orbital factor carries, which is the word a
    declaration of that orbital is written in
    (:attr:`.._sitechars.SiteGroupTable.declarable`). For a model whose states
    carry spin that is a different name from the one the row is returned under
    — the returned one is the double-valued site corep the states carry — and
    the two are read from the two table files of the one group.

    Parameters
    ----------
    group : ..groups.GroupData
    assignment : ..groups.OrbitalAssignment
    site : int
    table : .._sitechars.SiteGroupTable

    Returns
    -------
    tuple
        One entry per row: the single-valued entry's name at the site's own
        Wyckoff position, and ``""`` for a row the single-valued entries there
        do not reach.
    """
    names = _site_irrep_names(_single_valued_view(group), assignment, int(site), table)
    return tuple(str(name).split("↑G")[0] if name else "" for name in names)


# --------------------------------------------------------------------------
# the site point group's own double-valued names, read off the tables
# --------------------------------------------------------------------------
#
# The tables name a point group's double-valued irreps only through the
# elementary band representations those induce, so a site irrep whose band
# representation is composite, or a site at a position the tables list no
# entries at, has no double-valued name there.  Every
# crystallographic point group is, though, the site group of the origin of a
# symmorphic space group, and at a position whose site group is the whole point
# group the band representation a site irrep induces is, at Gamma, that irrep
# itself.  So each symmorphic group's own tables hold its point group's
# double-valued names -- the EBR entries at the origin -- and their characters
# -- the Gamma corep each entry induces -- on every operation, together with the
# SU(2) matrix every character is written against.  That is the point group's
# own vocabulary, read and not typed, in the tables' convention.
#
# A site row is named by comparing characters with it, operation by operation.
# The site group and the reference are one point group in orientations that
# can differ, so the comparison runs through a rotation carrying the one onto
# the other, and each character follows the operation it belongs to -- its axis,
# its sense, and the sign of the SU(2) matrix the tables lift it by -- never its
# place in a list.


#: The order of a proper rotation, from its trace, which is ``1 + 2 cos θ`` and
#: an integer for every crystallographic one.
_ORDER_OF_TRACE = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}


def _orthogonal_of(S, det) -> np.ndarray:
    r"""The Cartesian matrix of an operation, read off its SU(2) matrix.

    :math:`R_{ij} = \tfrac12\,\mathrm{tr}(\sigma_i S \sigma_j S^\dagger)` is the
    rotation ``S`` lifts, in the frame the tables write their SU(2) matrices in,
    and an improper operation is that rotation times ``det = -1``: the tables
    lift an improper operation by its proper part.
    """
    S = np.asarray(S, dtype=complex)
    pauli = _spacegroup._PAULI
    rotation = np.real(np.einsum("iab,bc,jcd,da->ij", pauli, S, pauli, S.conj().T)) / 2.0
    return float(det) * rotation


@lru_cache(maxsize=None)
def _tabulated_operations(number_str: str, magnetic: bool) -> tuple:
    """``(R, O, S)`` per operation of one group's double-valued table.

    ``R`` is the integer matrix the table writes the operation in (its lattice
    basis), ``O`` the Cartesian matrix in the frame of the table's own SU(2)
    matrix ``S``.  The unitary operations come first and the antiunitary ones
    after them, in the table's own order: the order :attr:`..groups.GroupData.symmetries`
    is indexed in, so operation ``ind`` is entry ``ind - 1``.
    """
    from irreptables.irreps import IrrepTable

    return _operations_of(IrrepTable(str(number_str), True, magnetic=bool(magnetic)))


def _operations_of(table) -> tuple:
    """:func:`_tabulated_operations` of a table already read."""
    out = []
    for sym in list(table.u_symmetries) + list(table.au_symmetries):
        R = np.rint(np.asarray(sym.R, dtype=float)).astype(int)
        det = int(round(float(np.linalg.det(R))))
        out.append((R, _orthogonal_of(sym.S, det), np.asarray(sym.S, dtype=complex)))
    return tuple(out)


def _magnetic_numbers() -> dict:
    """``{symbol: number}`` of every magnetic space group, as ``irrep`` itself lists them.

    ``irrep`` numbers a magnetic group by the line of this list the group sits
    on, which is how a grey group read off a model gets its number; the grey
    group of a space group is the one spelled as the space group's own symbol
    with ``1'`` appended.
    """
    import irrep

    path = os.path.join(os.path.dirname(irrep.__file__), "data", "msg_numbers.data")
    named = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2 and "." in parts[0]:
                named[parts[1]] = parts[0]
    return named


def _origin_letter(raw, centering) -> str:
    """The first position of an EBR list whose site group is the whole point group.

    That is the position with one site per primitive cell — conventional
    multiplicity one more than the centering translations — and in a symmorphic
    group the first such is the origin.
    """
    per_cell = 1 + len(tuple(centering))
    for entry in raw["ebrs"]:
        match = re.match(r"^(\d+)([a-zA-Z])\(", str(entry["wyckoff_position"]))
        if match is not None and int(match.group(1)) == per_cell:
            return match.group(2)
    return ""


@lru_cache(maxsize=None)
def _origin_coreps(grey: bool) -> dict:
    """``{point group: references}`` — every symmorphic group's origin, in its tables' words.

    A space group is symmorphic where every operation its table lists is a pure
    rotation, its translation a lattice or centering vector.  For each such
    group, in the order of its number, the reference holds the table's number,
    its unitary operations ``(R, O, S)`` (:func:`_tabulated_operations`), and
    every double-valued EBR entry at its origin: the site irrep's name, and the
    characters on those operations of the one Gamma corep that entry induces.
    ``grey`` reads the grey tables, where a complex-conjugate pair is one corep
    under time reversal.

    Raises
    ------
    ..refusal.InternalError
        An entry at the origin induces anything but one corep at Gamma of the
        entry's own dimension, or a grey group's operations are not the space
        group's own with time reversal: the tables disagreeing with themselves.
    """
    return _origin_references(grey, True)


@lru_cache(maxsize=None)
def _origin_irreps(grey: bool) -> dict:
    """:func:`_origin_coreps`, read in the single-valued tables.

    The same symmorphic origins, each single-valued EBR entry there with the
    characters of the one Gamma irrep it induces, on the same operations: the
    double-valued table of the same group orients them, since a single-valued
    table carries no SU(2) matrix, and it lists them in the same order, which is
    checked.  ``grey`` reads the grey tables, where a complex-conjugate pair is
    one corep under time reversal (``1E2E``).

    Raises
    ------
    ..refusal.InternalError
        As :func:`_origin_coreps`, or the two tables of one group list different
        operations.
    """
    return _origin_references(grey, False)


def _origin_references(grey: bool, double: bool) -> dict:
    """The body of :func:`_origin_coreps` (``double``) and :func:`_origin_irreps`."""
    from irreptables.ebrs import load_ebr_data
    from irreptables.irreps import IrrepTable

    named = _magnetic_numbers() if grey else {}
    out: dict = {}
    for sg in range(1, 231):
        plain = IrrepTable(str(sg), True, magnetic=False)
        centering = _spacegroup._centering(plain)
        if not all(_spacegroup._is_translation(np.asarray(sym.t, dtype=float), centering)
                   for sym in plain.symmetries):
            continue
        number = (named.get(str(plain.name).strip() + "1'", "") if grey else str(sg))
        if not number or number.split(".")[0] != str(sg):
            raise InternalError(
                f"irrep's own list of magnetic groups names no grey group of space group "
                f"{sg} ({plain.name}), so its point-group names cannot be read with time "
                "reversal.")
        table = IrrepTable(number, True, magnetic=grey) if grey else plain
        unitary = [(i + 1, sym) for i, sym in enumerate(table.symmetries)
                   if not sym.time_reversal]
        if grey and (len(table.au_symmetries) != len(unitary)
                     or len(unitary) != len(plain.symmetries)):
            raise InternalError(
                f"#{number} is listed as the grey group of space group {sg}, and its "
                "table does not carry each operation once with time reversal and once "
                "without.")
        ops = tuple(_operations_of(table)[:len(unitary)])
        # the characters: the double-valued table's own, or the single-valued
        # table of the same group, which has to list the same operations
        source = table if double else IrrepTable(number, False, magnetic=grey)
        if not double and (len(source.symmetries) != len(table.symmetries) or any(
                np.abs(np.asarray(a.R, dtype=float)
                       - np.asarray(b.R, dtype=float)).max() > _TABLE_MATRIX_TOL
                or bool(a.time_reversal) != bool(b.time_reversal)
                for a, b in zip(source.symmetries, table.symmetries))):
            raise InternalError(
                f"#{number}'s single- and double-valued tables list different operations "
                "in one order, so a single-valued character cannot be read on the "
                "operations the double-valued table orients.")
        gamma = {str(irr.name): {int(k): complex(v) for k, v in irr.characters.items()}
                 for irr in source.irreps if str(irr.kpname) == "GM"}
        raw = load_ebr_data(number, double)
        letter = _origin_letter(raw, centering)
        labels = [str(label) for label in raw["basis"]["irrep_labels"]]
        degeneracy = dict(zip(labels, (int(d) for d in raw["basis"]["degeneracies"])))
        names = []
        for entry in raw["ebrs"]:
            match = re.match(r"^(\d+)([a-zA-Z])\(", str(entry["wyckoff_position"]))
            if match is None or match.group(2) != letter:
                continue
            name = str(entry["ebr_name"]).split("↑")[0]
            bands = int(re.search(r"\((\d+)\)", str(entry["ebr_name"])).group(1))
            at_gamma = [(label, int(round(float(m)))) for label, m
                        in zip(labels, entry["vector"])
                        if m and kpoint_of_label(label) == "GM"]
            if (len(at_gamma) != 1 or at_gamma[0][1] != 1
                    or degeneracy[at_gamma[0][0]] != bands
                    or at_gamma[0][0] not in gamma):
                raise InternalError(
                    f"at the origin of #{number} the site irrep {name} induces "
                    f"{at_gamma} at Gamma, where a position whose site group is the whole "
                    "point group induces each site irrep as itself: the EBR list and the "
                    "irrep table do not agree.")
            chars = gamma[at_gamma[0][0]]
            names.append((name, tuple(chars[i] for i, _sym in unitary)))
        symbol = _sitechars.point_group_of(R for R, _O, _S in ops)
        out.setdefault(symbol, []).append((number, ops, tuple(names)))
    return {symbol: tuple(refs) for symbol, refs in out.items()}


def _matrix_key(M) -> tuple:
    """A hashable reading of one Cartesian matrix, to the tables' precision."""
    return tuple(float(x) + 0.0 for x in np.round(np.asarray(M, dtype=float), 3).ravel())


def _proper_axis(R, O) -> tuple:
    """``(axis, order, det)`` of one operation's proper rotation; ``axis`` ``None`` for ``±1``.

    The order and the determinant are read off the integer matrix ``R``,
    exactly; the axis off the Cartesian ``O``, oriented so that the rotation
    turns counterclockwise about it, a twofold's either way.
    """
    det = int(round(float(np.linalg.det(np.asarray(R, dtype=float)))))
    order = _ORDER_OF_TRACE.get(int(round(det * float(np.trace(R)))), 0)
    if order <= 1:
        return None, order, det
    P = det * np.asarray(O, dtype=float)
    if order == 2:
        half = (P + np.eye(3)) / 2.0
        col = int(np.argmax(np.diag(half)))
        axis = half[:, col]
    else:
        axis = np.array([P[2, 1] - P[1, 2], P[0, 2] - P[2, 0], P[1, 0] - P[0, 1]])
    return axis / np.linalg.norm(axis), order, det


def _frame(x, y) -> np.ndarray:
    """The right-handed orthonormal frame whose first axis is ``x`` and whose second lies towards ``y``."""
    x = np.asarray(x, dtype=float) / np.linalg.norm(x)
    y = np.asarray(y, dtype=float) - float(np.asarray(y, dtype=float) @ x) * x
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _rotation_between(a, b) -> np.ndarray:
    """The proper rotation of least angle taking the unit vector ``a`` to ``b``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    v, c = np.cross(a, b), float(a @ b)
    if float(np.linalg.norm(v)) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0.0, 0.0])
        if float(np.linalg.norm(p)) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p = p / np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _carries(Q, source, target) -> bool:
    """Whether the rotation ``Q`` carries the operations ``source`` onto ``target``, as sets."""
    Q = np.asarray(Q, dtype=float)
    return ({_matrix_key(Q @ np.asarray(O, dtype=float) @ Q.T) for _R, O in source}
            == {_matrix_key(O) for _R, O in target})


def _conjugators(source, target, tol=_TABLE_MATRIX_TOL) -> tuple:
    """Every proper rotation ``Q`` with ``Q·source·Qᵀ = target``, the two taken as sets.

    ``source`` and ``target`` are ``(R, O)`` per operation, as
    :func:`_proper_axis` reads them.  A candidate carries an axis of the highest
    order in ``source`` onto an axis of the same order and determinant in
    ``target``, and, where ``source`` has a second axis, that one onto one at the
    same angle; each candidate is kept only where it carries the whole set onto
    the other.  A group with one axis admits any turn about it, which commutes
    with every operation and so changes nothing a character sees.
    """
    src = [info for info in (_proper_axis(R, O) for R, O in source) if info[0] is not None]
    dst = [info for info in (_proper_axis(R, O) for R, O in target) if info[0] is not None]

    def carries(Q):
        return _carries(Q, source, target)

    if not src:
        return (np.eye(3),) if carries(np.eye(3)) else ()
    src.sort(key=lambda info: -info[1])
    first = src[0]
    second = next((info for info in src
                   if abs(abs(float(info[0] @ first[0])) - 1.0) > tol), None)
    candidates = []
    for n1, order1, det1 in dst:
        if (order1, det1) != first[1:]:
            continue
        for s1 in (1.0, -1.0):
            if second is None:
                candidates.append(_rotation_between(first[0], s1 * n1))
                continue
            for n2, order2, det2 in dst:
                if (order2, det2) != second[1:]:
                    continue
                for s2 in (1.0, -1.0):
                    if abs(float(first[0] @ second[0]) - s1 * s2 * float(n1 @ n2)) > tol:
                        continue
                    candidates.append(_frame(s1 * n1, s2 * n2)
                                      @ _frame(first[0], second[0]).T)
    found = []
    for Q in candidates:
        if carries(Q) and not any(float(np.abs(Q - P).max()) < tol for P in found):
            found.append(Q)
    return tuple(found)


def _su2_of(Q) -> np.ndarray:
    r"""An SU(2) matrix lifting the proper rotation ``Q``, read off its quaternion.

    ``cos(θ/2) − i sin(θ/2) n·σ`` for a turn by ``θ`` about ``n``, the
    quaternion taken from the largest of its four squares so that no turn —
    a half turn included — divides by a small number.  Which of the two lifts
    comes back does not matter: the matrix is only ever used to conjugate.
    """
    U, _s, Vt = np.linalg.svd(np.asarray(Q, dtype=float))
    Q = U @ Vt
    trace = float(np.trace(Q))
    squares = [1.0 + trace, 1.0 + 2.0 * Q[0, 0] - trace,
               1.0 + 2.0 * Q[1, 1] - trace, 1.0 + 2.0 * Q[2, 2] - trace]
    k = int(np.argmax(squares))
    big = np.sqrt(max(squares[k], 0.0)) / 2.0
    if k == 0:
        w = big
        x, y, z = ((Q[2, 1] - Q[1, 2]) / (4 * w), (Q[0, 2] - Q[2, 0]) / (4 * w),
                   (Q[1, 0] - Q[0, 1]) / (4 * w))
    elif k == 1:
        x = big
        w, y, z = ((Q[2, 1] - Q[1, 2]) / (4 * x), (Q[0, 1] + Q[1, 0]) / (4 * x),
                   (Q[0, 2] + Q[2, 0]) / (4 * x))
    elif k == 2:
        y = big
        w, x, z = ((Q[0, 2] - Q[2, 0]) / (4 * y), (Q[0, 1] + Q[1, 0]) / (4 * y),
                   (Q[1, 2] + Q[2, 1]) / (4 * y))
    else:
        z = big
        w, x, y = ((Q[1, 0] - Q[0, 1]) / (4 * z), (Q[0, 2] + Q[2, 0]) / (4 * z),
                   (Q[1, 2] + Q[2, 1]) / (4 * z))
    pauli = _spacegroup._PAULI
    return (w * np.eye(2, dtype=complex)
            - 1j * (x * pauli[0] + y * pauli[1] + z * pauli[2]))


def _carried_characters(reference, here, Q, tol=_TABLE_MATRIX_TOL, *, double=True):
    """The reference's characters, carried by ``Q`` onto the site's operations.

    ``here`` is ``(R, O, S)`` per operation of the site.  Each is the image under
    ``Q`` of one reference operation, and its SU(2) matrix the image of that
    operation's under the lift of ``Q`` up to a sign: the sign is the tables'
    own choice of lift for the two, and a double-valued character carries it,
    since the other lift is the operation times the rotation by ``2π``.  A
    single-valued character (``double=False``) is the operation's own and
    carries no sign.  Returns ``[(name, [character per site operation])]``, or
    ``None`` where some operation has no image.
    """
    _number, ops, names = reference
    by_key = {_matrix_key(O): i for i, (_R, O, _S) in enumerate(ops)}
    V = _su2_of(Q)
    back = []
    for _R, O, S in here:
        i = by_key.get(_matrix_key(Q.T @ np.asarray(O, dtype=float) @ Q))
        if i is None:
            return None
        if not double:
            back.append((i, 1.0))
            continue
        carried = V @ ops[i][2] @ V.conj().T
        lam = complex(np.trace(np.asarray(S, dtype=complex).conj().T @ carried)) / 2.0
        if abs(abs(lam) - 1.0) > tol or abs(lam.imag) > tol:
            return None
        back.append((i, 1.0 if lam.real > 0 else -1.0))
    return [(name, [sign * chars[i] for i, sign in back]) for name, chars in names]


def _point_group_names(group, table) -> tuple:
    r"""Each row of a site table, with the spin, named by the site point group's own irreps.

    The row times the operations' spin-½ lift is the double-valued site
    representation the states carry; it is decomposed over the double-valued
    irreps of the site's point group as the tables name them at the origin of a
    symmorphic group (:func:`_origin_coreps`), grey where the group carries time
    reversal, and spelled as the sum of what it holds (``"-1E2E + -E1"``).

    **Which reference, and how the characters are carried.**  Where the site's
    operations are, as Cartesian matrices in the tables' own frame, exactly the
    operations of a symmorphic group's origin — the same axes, the same senses —
    the lowest-numbered such group is the reference and each character is read
    off the operation it is of, its sign the tables' own lift.  Otherwise the
    reference is the lowest-numbered group of the point group and the
    characters are carried by every rotation taking its origin's operations
    onto the site's: a row takes a name where all of them agree on it.  They
    agree on every double-valued name of a grey group, and where they part —
    the ``1``/``2`` members of a complex-conjugate pair of a group without time
    reversal, whose names follow a sense of rotation no table fixes at an axis
    no symmorphic origin shares — the row is left unnamed.

    Returns
    -------
    tuple
        One label per row of ``table``, ``""`` where nothing names it.
    """
    names = _point_group_reading(group, table, double=True)
    return ("",) * len(table.labels) if names is None else names


def _point_group_irreps(group, table):
    r"""Each row of a site table named by the site point group's own single-valued irreps.

    :func:`_point_group_names` without the spin: the row itself, the orbital
    character, decomposed over the single-valued irreps of the site's point
    group as the tables name them at the origin of a symmorphic group
    (:func:`_origin_irreps`), grey where the table's rows are glued by time
    reversal.  The same irrep then has the tables' name at every position, the
    ones their entries list and the ones they do not; a ``1``/``2`` member of a
    complex-conjugate pair whose name follows a sense of rotation no table
    fixes at this site is left unnamed.

    Returns
    -------
    tuple or None
        One label per row, ``""`` where the references do not agree on one, and
        ``None`` where no reference reads the site at all, the tables not set up
        for the group.
    """
    return _point_group_reading(group, table, double=False)


def _point_group_reading(group, table, *, double):
    """:func:`_point_group_names` (``double``) or :func:`_point_group_irreps`; ``None`` where unread."""
    if not getattr(group, "_tables_ready", False) or (double and not group.spinor):
        return None
    symbol = _sitechars.point_group_symbol(group, table.indices)
    if double:
        references = _origin_coreps(_spacegroup._magnetic_type(group) == "II").get(symbol, ())
    else:
        references = _origin_irreps(bool(getattr(table, "antiunitary", ()))).get(symbol, ())
    tabulated = _tabulated_operations(str(group.number_str), bool(group.magnetic))
    if not references or len(tabulated) != len(group.symmetries):
        return None
    here = [tabulated[int(ind) - 1] for ind in table.indices]
    target = [(R, O) for R, O, _S in here]
    aligned = next((reference for reference in references
                    if _carries(np.eye(3), [(R, O) for R, O, _S in reference[1]], target)),
                   None)
    if aligned is not None:
        readings = [_carried_characters(aligned, here, np.eye(3), double=double)]
    else:
        readings = [_carried_characters(references[0], here, Q, double=double) for Q in
                    _conjugators([(R, O) for R, O, _S in references[0][1]], target)]
    if not readings or any(reading is None for reading in readings):
        return None
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    lift = ([complex(np.trace(np.asarray(group.symmetries[pos_of[int(ind)]].spinor_rotation)))
             for ind in table.indices] if double else [1.0] * len(table.indices))
    out = []
    for row in range(len(table.labels)):
        chi = np.array([table.character_on(row, int(ind)) * lift[j]
                        for j, ind in enumerate(table.indices)], dtype=complex)
        spelled = set()
        for reading in readings:
            basis = np.array([chars for _name, chars in reading], dtype=complex).T
            n = np.linalg.lstsq(basis, chi, rcond=None)[0]
            whole = np.round(n.real)
            if (float(np.abs(basis @ whole - chi).max()) > _TABLE_MATRIX_TOL
                    or float(np.abs(n - whole).max()) > _TABLE_MATRIX_TOL
                    or float(whole.min()) < 0 or not whole.any()):
                spelled.add("")
                continue
            spelled.add(_spelled([name for (name, _chars), m in zip(reading, whole)
                                  for _ in range(int(m))]))
        out.append(spelled.pop() if len(spelled) == 1 else "")
    return tuple(out)
