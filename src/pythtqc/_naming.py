"""Names of site irreps, read off the tables.

Each row of a measured site group's character table is named by the
elementary band representation it induces from the site's Wyckoff position.
Where the list of elementary band representations has no entry there, a
single-valued row takes the name of the site irrep that the tables of the
stable real-space invariants give the same invariants. Otherwise a row takes
the name the tables give that irrep of the site point group, read at the
origin of a symmorphic group. Failing that, a single-valued row keeps its
Mulliken symbol. The site tables a declaration is read on carry these names.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import itertools
import os
import re

import numpy as np

from . import _site_names
from . import _sitechars
from . import _spacegroup
from . import _tables
from .refusal import InternalError, Refusal
from ._tolerance import CHARACTER, _INT_TOL, _TABLE_MATRIX_TOL
from ._tables import _ebr_rows, _letter_only, kpoint_of_label
from ._matrices import _primed_partners


def _site_tables(group, assignment, kept_u, kept_au, sites):
    """The named table of the measured site group of each of ``sites``.

    ``assignment`` is the orbitals' assignment to ``group``, and ``kept_u`` and
    ``kept_au`` are the ``.ind`` of the unitary and antiunitary operations kept.
    A site's group is the operations of ``assignment.site_group`` among them.
    Each table is computed by :func:`_site_group_table` and named by
    :func:`_named_from_tables`: under single-valued site irreps, or, on a group
    with double-valued tables, under the double-valued site irreps the states
    carry.  Sites with the same operations and the same Wyckoff letter share one
    table.  Where the shipped table of site-irrep names holds the letter, the
    sites must also lie in one stretch of the position and have their
    operations carried onto the same ones at the position's own site
    (:func:`.._site_names.carry_to_own_site`).  Returns a mapping from each site
    to its :class:`.._sitechars.SiteGroupTable`.
    """
    letters = tuple(getattr(assignment, "wyckoff_tables", ()) or ())
    out, cache = {}, {}
    for s in sites:
        ind_u = tuple(int(i) for i in assignment.site_group[s] if int(i) in kept_u)
        ind_au = tuple(int(i) for i in assignment.site_group[s] if int(i) in kept_au)
        letter = letters[int(s)] if int(s) < len(letters) else ""
        key = (ind_u, ind_au, letter)
        if _site_names._holds(group, letter):
            carried = _site_names.carry_to_own_site(
                group, letter, assignment.site_positions[int(s)], ind_u + ind_au)
            key = key + (carried,)
        if key not in cache:
            table = _site_group_table(group, ind_u, antiunitary=ind_au)
            cache[key] = _named_from_tables(group, assignment, int(s), table)
        out[int(s)] = cache[key]
    return out


def _site_group_table(group, indices, antiunitary=()):
    """A site group's character table, each row under its point-group name.

    ``indices`` are the ``.ind`` of the site group's unitary operations in
    ``group``, and ``antiunitary`` those of its antiunitary ones, under which a
    complex-conjugate pair of rows is merged.  The table is computed by
    :func:`.._sitechars.site_group_table`.  A row takes the name the tables give
    that irrep of the site point group, read at the origin of a symmorphic group
    (:func:`_point_group_irreps`), so that one irrep has one name at every site
    (``named_by = "point group"``).  Where those names leave the row open, it
    keeps its Mulliken symbol (``"Mulliken"``), unless that is a placeholder
    (:func:`.._sitechars.is_placeholder`) or another row's name.  A row neither
    names is left with an empty label (``"unnamed"``).  Returns the table with
    ``labels`` and ``named_by`` set.
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
    """One site's character table within any subgroup, built once per subgroup.

    ``site`` is a site of ``assignment``, the orbitals' assignment to ``group``.
    Called with the ``.ind`` of a subgroup's unitary operations, it returns the
    table of the site group's operations among them, or ``None`` where they do
    not close into a group.  With ``corep=False`` the table is a plain character
    table under point-group names (:func:`_site_group_table`), which is all a
    decomposition reads.  With ``corep=True``, for a declaration written in the
    names of double-valued site irreps, the table is named as
    :func:`_site_tables` names it, with the antiunitary partner of each kept
    operation merging complex-conjugate pairs.

    See Also
    --------
    .._declaration._covariant_subgroup : the largest subgroup that carries a
        declaration, which reads a site's multiplicities on these tables.
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
    """``table`` with each of its rows named off the tables at ``site``.

    ``table`` is the site group's table from :func:`_site_group_table`, and
    ``site`` a site of ``assignment``, the orbitals' assignment to ``group``.  A
    single-valued row is named by :func:`_site_irrep_labels`: by the EBR entry
    at the site's Wyckoff position whose band representation it induces
    (``named_by = "tables"``), else by the tables of the stable real-space
    invariants (``"invariants"``), else by the name it has (``"point group"``,
    ``"Mulliken"``, or ``"unnamed"`` with an empty label).  On a group with
    double-valued tables a row is named by :func:`_site_corep_labels` as the
    double-valued representation the states carry, a site irrep or a sum of
    them, which is the row times the operations' spin-1/2 lift.  Its orbital
    factor's single-valued name (:func:`_site_orbital_labels`), or else the
    name the row has in ``table`` (point-group or Mulliken), is then a second
    spelling a declaration may be written in, and is never a label returned.
    A row nothing names keeps its characters as the whole statement, and
    nothing here refuses.  Returns the table with ``labels``, ``named_by`` and
    ``declarable`` set, each entry of ``declarable`` ``""`` on a single-valued
    group.
    """

    if group.spinor:
        # The states at this site carry the row times the operations' spin-1/2
        # lift, a double-valued site irrep or a sum of them, and its name is the
        # one returned.  A declaration states the orbital factor, the row
        # itself, named by the same group's single-valued entries at the same
        # position.
        labels, named_by = _site_corep_labels(group, assignment, int(site), table)
        orbital = _site_orbital_labels(group, assignment, int(site), table)
        return replace(table, labels=labels, named_by=named_by,
                       declarable=tuple(name or own for name, own
                                        in zip(orbital, table.labels)))
    # the single-valued names: the EBR entry, else the name the stable
    # real-space invariants give the row, else the name the row has in table,
    # so a row needs no second spelling
    labels, named_by = _site_irrep_labels(group, assignment, int(site), table)
    return replace(table, labels=labels, named_by=named_by,
                   declarable=("",) * len(labels))


def _contained(tab, held, want, *, corep=False, asked=None):
    """Whether one representation of a site group contains another.

    ``held`` and ``want`` are their characters, given class by class of the site
    table ``tab``.  The one contains the other where ``held`` less ``want`` is a
    representation of the site group, every multiplicity a nonnegative integer.
    With ``corep=True``, ``want`` is declared in the names of double-valued site
    irreps; both must then be representations of the site group, and they are
    compared by the double-valued site irreps their rows lift to
    (:func:`.._sitechars.corep_counts`).  Where a row's lift has no name,
    ``held`` must contain a representation whose characters equal ``want``'s on
    the classes ``asked`` marks, those on which the spin-1/2 lift has a nonzero
    trace; ``asked=None`` marks every class.
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
    """The multiplicity of each row of ``tab`` in the character ``chi``.

    ``chi`` is given class by class.  Returns a mapping from row to
    multiplicity, the zeros dropped, or ``None`` where a multiplicity is not a
    nonnegative integer within ``CHARACTER``.
    """
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
# A row of a measured site group's table is named by the band representation
# it induces, matched against the EBR columns at the site's Wyckoff position of
# the identified space group.  Inducing and matching needs no convention for
# how the site group is oriented.


def _unitary_frame(group):
    """The unitary operations of ``group`` and their products.

    Returns ``(unitary, inverse, prod, pos_of)``: the positions in
    ``group._symmetries`` of the unitary operations; a mapping from each to the
    position of its inverse; the product table of positions, as nested lists;
    and a mapping from each operation's ``.ind`` to its position.
    """
    prod, _, _ = group._get_product_table(get_diff=True)
    prod = np.asarray(prod, dtype=int).tolist()
    syms = group._symmetries
    pos_of = {int(s.ind): p for p, s in enumerate(syms)}
    unitary = [p for p, s in enumerate(syms) if not s.time_reversal]
    ident = next(p for p in unitary
                 if np.array_equal(np.asarray(syms[p].rotation), np.eye(3, dtype=int))
                 and np.allclose(np.asarray(syms[p].translation) % 1.0, 0.0, atol=1e-8))
    inverse = {p: next(q for q in unitary if prod[p][q] == ident) for p in unitary}
    return unitary, inverse, prod, pos_of


def _decompose_against_coreps(table, chi, *, where, tol=_INT_TOL):
    """The multiplicities of the little-group representations in ``chi``.

    ``table`` maps each little-group representation's name to its characters
    by operation index, as :func:`..groups.corep_table` returns it, and ``chi``
    is a character on the same unitary operations.  The multiplicities are
    found by orthogonality, and ``tol`` absorbs the rounding of the tabulated
    characters.  A character induced from a site group is a character of the
    little group, so a multiplicity off an integer by more than ``tol`` raises
    :class:`..refusal.InternalError`.  ``where`` names what is decomposed, for
    its message.  Returns a mapping from each representation's name to its
    multiplicity, the zeros dropped.
    """
    inds = sorted(next(iter(table.values())))
    out = {}
    for label, char in table.items():
        norm = sum(abs(complex(char[i])) ** 2 for i in inds)
        m = sum(complex(chi[i]) * np.conj(complex(char[i])) for i in inds) / norm
        if abs(m.imag) > tol or abs(m.real - round(m.real)) > tol:
            raise InternalError(
                f"{where} induces {label} with multiplicity {m:.6g}, which is not an "
                "integer. Report it, with the model and the call.")
        if round(m.real):
            out[label] = int(round(m.real))
    return out


def _induction_plan(group, assignment, site, fixers, *, k=None, kpname=None):
    r"""The induction's weights at one point, for every row of a site table.

    ``fixers`` are the ``.ind`` of the unitary operations fixing ``site``, a site
    of ``assignment``, the orbitals' assignment to ``group``.  ``k`` is the point
    in the model's reduced coordinates, ``None`` for ``GM``, and ``kpname`` its
    tabulated name, passed to :func:`.._tables._corep_table`.  The weight of the
    site group's operation :math:`h` in the induced character on the little
    group's operation :math:`g` does not depend on the character induced.
    Returns ``(table, weights)``: the little group's character table, and
    ``weights[g][h]``, both keyed by ``.ind``, which :func:`_induced_content`
    reads.  Returns ``None`` where ``fixers`` is empty or the tables at the
    point refuse, and, away from ``GM``, where the induction meets an operation
    fixing the site that ``fixers`` leave out.
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
    table = _tables._corep_table(group, point, kpname=kpname)
    if isinstance(table, Refusal):
        return None
    unitary, inverse, prod, pos_of = _unitary_frame(group)
    syms = group._symmetries
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
    k3 = np.asarray(group._k_to_3d(np.asarray(k, dtype=float)[None, :])[0], dtype=float)
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
    r"""The band representation a site character induces, at one point.

    At ``GM`` the induction is the real-space one,
    :math:`\chi_{\rm ind}(g) = |H|^{-1}\sum_{x\in P}[x^{-1}gx\in H]\,
    \chi(x^{-1}gx)`, over the unitary operations :math:`P` of ``group`` modulo
    the lattice, :math:`H` the site group, through the group's product table.
    Away from ``GM`` it is the Frobenius sum over the sites :math:`j` of the
    orbit that :math:`g` fixes, each weighted by :math:`e^{-2\pi i\,k\cdot m}`.
    For :math:`g = \{R|t\}` and :math:`\tau_j` the position of site :math:`j`,
    :math:`m = R\tau_j + t - \tau_j` is the lattice vector by which :math:`g`
    returns site :math:`j` (``assignment.lattice_shift``).  The phase carries
    the operation's fractional translation, and its sign is the one the rows of
    :func:`..groups.ebr_data` are written in.

    Parameters
    ----------
    group : ..groups.GroupData
        The group the site was measured in.
    assignment : ..groups.OrbitalAssignment
        The orbitals' assignment to ``group``, which holds the orbit and the
        lattice shifts.
    site : int
        The site of ``assignment`` the character belongs to.
    chi : dict
        ``{ind: character}`` on the site's unitary fixers, which are its keys.
        On a group with double-valued tables it includes the operations'
        spin-1/2 lift.
    k : array, default None
        The point, in the model's reduced coordinates.  ``None`` is ``GM``.
    kpname : str, default None
        The point's tabulated name, passed to :func:`.._tables._corep_table`.
    plan : tuple, default None
        :func:`_induction_plan` for this site and point, shared by the rows of
        one table.  ``None`` builds it here.

    Returns
    -------
    dict
        A mapping from the name of each little-group representation at the
        point to its multiplicity, the zeros dropped.  Empty where
        :func:`_induction_plan` returns ``None``.
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
        table, chi_ind, where=f"a site irrep at site {int(site)}")


def _gamma_point(group):
    """The tabulated ``GM`` of ``group``, in the model's reduced coordinates.

    Returns the refusal of :func:`.._tables._high_symmetry_points` where it
    refuses.
    """
    points = _tables._high_symmetry_points(group)
    if isinstance(points, Refusal):
        return points
    return points["GM"]


def _rows_at_letter(rows, letter) -> dict:
    """The EBR entries of ``rows`` at one Wyckoff letter.

    ``rows`` is what :func:`.._tables._ebr_rows` returns, and ``letter`` a
    Wyckoff letter without its multiplicity.  Returns a mapping from the name
    of each entry at that letter, ``"B1↑G(2)"``, to its multiplicities.
    """
    return {name.split("@")[0]: content for name, content in rows.items()
            if _letter_only(name.split("@")[1]) == str(letter)}


def _gamma_ties(group) -> dict | Refusal:
    r"""The positions where two EBRs have equal multiplicities at ``GM``.

    Two site irreps at one Wyckoff position can induce the same little-group
    representations at ``GM`` and differ only away from it: ``B1↑G`` and
    ``B2↑G`` at ``2c`` of ``P4mm1'`` both induce ``GM5`` and differ at ``X``,
    ``M``, ``R`` and ``A``.  The positions are read off the EBR list of
    ``group``, with no model.  Returns a mapping from each such position,
    ``"2c(mm21',mm2)"``, to the sets of EBR names tied there.

    Refusals
    --------
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``lattice_not_descended``
        those of :func:`.._tables._ebr_rows`.
    """
    rows = _ebr_rows(group)
    if isinstance(rows, Refusal):
        return rows
    return _gamma_ties_of(rows)


def _gamma_ties_of(rows) -> dict:
    """:func:`_gamma_ties` of EBR rows already read.

    ``rows`` is what :func:`.._tables._ebr_rows` returns.
    """
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
    r"""The EBR entry that names each row of one site table.

    ``table`` is the character table of the measured group of ``site``, a site
    of ``assignment``, the orbitals' assignment to ``group``.  Each row is
    induced to the high-symmetry points and matched against the EBR entries at
    the site's Wyckoff letter in the tables' setting
    (``assignment.wyckoff_tables``): at ``GM`` first, then at the other points
    in the tables' own order (:func:`.._tables._points_in_table_order`) while
    more than one entry is left.  On a group with double-valued tables the
    row's character times the operations' spin-1/2 lift is induced, and
    matched against the double-valued entries.  Returns, for each row, the
    full name of the one entry it matches, ``"B1↑G(2)"``, and ``""`` where no
    single entry matches, the tables refuse, or two rows match one entry.

    Raises
    ------
    ..refusal.InternalError
        Where entries left together after ``GM`` are not listed as tied by
        :func:`_gamma_ties`.
    """
    blank = ("",) * len(table.labels)
    letters = tuple(assignment.wyckoff_tables or ())
    if int(site) >= len(letters) or not letters[int(site)]:
        return blank
    rows = _ebr_rows(group)
    if isinstance(rows, Refusal):
        return blank
    at_letter = _rows_at_letter(rows, letters[int(site)])
    if not at_letter:
        return blank
    points = _tables._points_in_table_order(group)
    if isinstance(points, Refusal):
        return blank
    tied_here = ()
    for position, shared in _gamma_ties_of(rows).items():
        if _letter_only(position) == letters[int(site)]:
            tied_here = shared
    lift = None
    if group.spinor:
        lift = {int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
                for s in group._symmetries}
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
                    f"{group.name} #{group.number_str}: a site irrep at "
                    f"{letters[int(site)]} induces the multiplicities that each of "
                    f"{sorted(left)} has at {point}, and the EBR list's grouping of its "
                    "entries by their multiplicities at GM does not put them together. "
                    "Report it, with the model and the call.")
        named.append(next(iter(left)) if len(left) == 1 else "")
    claimed = [name for name in named if name]
    return tuple(name if name and claimed.count(name) == 1 else "" for name in named)


def _site_irrep_labels(group, assignment, site, table) -> tuple:
    r"""The single-valued label of each row of one site table, and its source.

    ``table`` is the site group's table from :func:`_site_group_table`, at
    ``site``, a site of ``assignment``, the orbitals' assignment to ``group``.
    A row takes the site irrep of the EBR entry it matches
    (:func:`_site_irrep_names`), the part of the entry's name before ``↑G``
    (``"tables"``).  A row no entry matches takes the name the tables of the
    stable real-space invariants give it at this site
    (:func:`.._site_names.table_names`, ``"invariants"``).  A row neither names
    keeps its label and ``named_by`` from ``table``.  Labels are unique within a
    table: where a name from the invariants repeats another row's, every name
    from the invariants is dropped, and where a name from the entries still
    repeats one, every row keeps its label from ``table``.  Returns
    ``(labels, named_by)``, one entry of each per row.
    """
    names = _site_irrep_names(group, assignment, int(site), table)
    own = tuple(table.labels)
    mine = tuple(table.named_by) if len(table.named_by) == len(own) else (
        tuple("point group" if label else "unnamed" for label in own))
    bases = [str(name).split("↑G")[0] if name else "" for name in names]
    found = (_site_names.table_names(group, assignment, int(site), table)
             if not all(bases) else ("",) * len(own))
    for invariants in (found, ("",) * len(own)):
        labels, named_by = [], []
        for row, base in enumerate(bases):
            if base:
                labels.append(base)
                named_by.append("tables")
            elif invariants[row]:
                labels.append(invariants[row])
                named_by.append("invariants")
            else:
                labels.append(own[row])
                named_by.append(mine[row])
        named = [label for label in labels if label]
        if len(set(named)) == len(named):
            return tuple(labels), tuple(named_by)
    return own, mine


# --------------------------------------------------------------------------
# naming a site representation by the sum of EBR entries it induces
# --------------------------------------------------------------------------
#
# A row of the computed table of a spinful model's site group is the orbital
# factor.  What the model's states carry there is that row times the
# operations' spin-1/2 lift, a double-valued representation of the site group
# that need not be irreducible: a ``T`` row of a tetrahedral site times the
# lift is a sum of two.  Its name is the sum of the double-valued site irreps
# it contains, read as a single one is: by the band representation it induces,
# matched against the EBR entries at the position.

#: Largest multiplicity of one EBR entry a site representation is searched for,
#: and the most matching combinations kept before the search gives up.  Both are
#: far above anything a site group of order <= 48 can produce: a row of
#: dimension ``d`` times the lift induces ``2d`` times the position's
#: multiplicity bands, and an entry carries at least two of them.
_MAX_ENTRY_MULTIPLICITY = 4

_MAX_COMBINATIONS = 64


def _entry_combinations(contents, target, *, cap=_MAX_COMBINATIONS):
    """Every multiset of EBR entries whose multiplicities sum to ``target``.

    ``contents`` maps each entry's name to its multiplicities at one point, and
    ``target`` holds the induced multiplicities there.  The search is
    exhaustive up to :data:`_MAX_ENTRY_MULTIPLICITY` copies of one entry and
    ``cap`` multisets found.  Returns a list of the multisets, each in the order
    of ``contents`` with its repeats written out, empty where none sums to
    ``target``.
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
        if not any(here):                     # an entry with nothing at this point
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
    """A multiset of EBR entry names, spelled as one label.

    The part of each name from ``↑G`` on is dropped, and each name stands in the
    order it first appears in ``names``: ``("-E1", "-E1", "-F")`` returns
    ``"2x-E1 + -F"``.
    """
    order, count = [], {}
    for name in names:
        base = str(name).split("↑G")[0]
        if base not in count:
            order.append(base)
        count[base] = count.get(base, 0) + 1
    return " + ".join(base if count[base] == 1 else f"{count[base]}x{base}"
                      for base in order)


def _names_by_content(group, assignment, site, table, letter) -> tuple:
    r"""Each row's label, as a sum of the EBR entries at ``letter``.

    ``table`` is the computed table of the measured group of ``site``, a site of
    ``assignment``, the orbitals' assignment to ``group``, and ``letter`` the
    site's own Wyckoff letter.  On a group with double-valued tables a row's
    character is taken times the operations' spin-1/2 lift, which can make it
    reducible, so a row may be named by several entries.  The row is induced at
    ``GM`` first, and at the other points in the tables' order while more than
    one multiset of entries reproduces it (:func:`_entry_combinations`).
    Returns one label per row, the entries' site irreps summed
    (``"-1E2E + -E1"``), and ``""`` where no multiset or more than one
    reproduces the row, or where the site's represented operations are only
    part of the operations fixing it.
    """
    blank = ("",) * len(table.labels)
    if not letter:
        return blank
    # The tables' entries at a position are keyed by the irreps of the whole
    # site group of that position.  Where the represented operations are only
    # part of what fixes the site, as on a free-standing layer whose
    # Hamiltonian does not fix how the orbitals transform under the horizontal
    # mirror and its products, a row is an irrep of the smaller group, and the
    # sum of entries it induces names no row.  So every row of the table is
    # left unnamed.  (:func:`_induction_plan` makes the same check away from
    # GM.)
    fixers = {int(i) for i in assignment.site_group[int(site)]}
    unitary = {int(s.ind) for s in group._symmetries if not s.time_reversal}
    if fixers & unitary != {int(i) for i in table.indices}:
        return blank
    rows = _ebr_rows(group)
    if isinstance(rows, Refusal):
        return blank
    at_letter = _rows_at_letter(rows, str(letter))
    if not at_letter:
        return blank
    points = _tables._points_in_table_order(group)
    if isinstance(points, Refusal):
        return blank
    lift = None
    if group.spinor:
        lift = {int(s.ind): complex(np.trace(np.asarray(s.spinor_rotation)))
                for s in group._symmetries}
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
    r"""The number of bands the site character ``chi`` induces, or ``0``.

    ``chi`` maps the ``.ind`` of each unitary operation fixing ``site``, a site
    of ``assignment``, the orbitals' assignment to ``group``, to its character.
    The number is read off the induced multiplicities at ``GM``, each times the
    dimension of its little-group representation.  Returns ``0`` where
    :func:`_induction_plan` returns ``None``.
    """
    plan = _induction_plan(group, assignment, int(site), tuple(chi))
    if plan is None:
        return 0
    table, _weights = plan
    content = _induced_content(group, assignment, int(site), chi, plan=plan)
    ident = next((int(s.ind) for s in group._symmetries
                  if not s.time_reversal
                  and np.array_equal(np.asarray(s.rotation), np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    if ident is None or any(ident not in table[label] for label in content):
        return 0                               # pragma: no cover - a guard
    return int(round(sum(m * float(np.real(table[label][ident]))
                         for label, m in content.items())))


def _site_corep_labels(group, assignment, site, table) -> tuple:
    r"""The double-valued label of each row of one site table, and its source.

    ``table`` is the site group's table from :func:`_site_group_table`, at
    ``site``, a site of ``assignment``, the orbitals' assignment to ``group``.
    Each row stands for the double-valued representation the states carry, the
    row times the operations' spin-1/2 lift.  It is named by the double-valued
    EBR entries at the site's Wyckoff letter whose sum is the band
    representation it induces (:func:`_names_by_content`, ``"tables"``).  Two
    rows may take one name, since two orbital characters that differ only where
    the lift's trace vanishes lift to one site irrep.  A row the entries do not
    name takes the site point group's double-valued name
    (:func:`_point_group_names`, ``"point group"``), or none (``""`` and
    ``"unnamed"``).  Where one name then stands on rows whose lifted characters
    differ, every row takes its point-group name instead.  Returns
    ``(labels, named_by)``, one entry of each per row.
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
            for s in group._symmetries}
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
# A row of a computed site table is the orbital factor of what sits at the
# site, single-valued whether or not the states carry spin.  Its single-valued
# name at the position, the one the group's single-valued EBR entries give the
# band representation the orbital alone induces, is what a declaration states.
# It is read as a spinless model's rows are, against the group's single-valued
# tables.


def _single_valued_view(group):
    r"""``group`` read through its single-valued tables.

    Which table files a query loads, the irrep table, the EBR file and its
    transport record, is chosen by ``group.spinor`` alone.  The view is
    ``group`` with that flag set to ``False``, every other field the same
    object: the same operations in the same order, the same cell and the same
    product table.  It is read only by :func:`_site_orbital_labels`, which
    induces a character and matches it against the single-valued entries at a
    Wyckoff letter.  No representation is built on it, no lift is taken from it,
    and it is never returned or stored.  Returns ``group`` itself where
    ``group.spinor`` is ``False``.
    """
    if not group.spinor:
        return group
    return replace(group, spinor=False)


def _site_orbital_labels(group, assignment, site, table) -> tuple:
    r"""The single-valued name of each row of one site table.

    ``table`` is the site group's table, at ``site``, a site of ``assignment``,
    the orbitals' assignment to ``group``.  A row is the orbital factor of what
    the states carry, and its single-valued name is the word a declaration of
    the orbital is written in (:attr:`.._sitechars.SiteGroupTable.declarable`).
    On a group with double-valued tables it differs from the label returned,
    which names the double-valued representation the states carry, a site
    irrep or a sum of them.  Returns, for each row, the site irrep of the
    single-valued EBR entry it matches at the site's Wyckoff position
    (:func:`_site_irrep_names` on :func:`_single_valued_view`).  A row no entry
    matches takes the name the tables of the stable real-space invariants give
    it (:func:`.._site_names.table_names`), unless one of those names would
    repeat another row's, when none is taken.  A row neither names is ``""``.
    """
    names = _site_irrep_names(_single_valued_view(group), assignment, int(site), table)
    bases = tuple(str(name).split("↑G")[0] if name else "" for name in names)
    if all(bases):
        return bases
    found = _site_names.table_names(group, assignment, int(site), table)
    filled = tuple(base or name for base, name in zip(bases, found))
    words = [word or own for word, own in zip(filled, table.labels)]
    words = [word for word in words if word]
    return filled if len(set(words)) == len(words) else bases


# --------------------------------------------------------------------------
# the site point group's own double-valued names, read off the tables
# --------------------------------------------------------------------------
#
# The tables name a point group's double-valued irreps only through the
# elementary band representations those induce, so a site irrep whose band
# representation is composite, or a site at a position the tables list no
# entries at, has no double-valued name there.  Every crystallographic point
# group is the site group of the origin of a symmorphic space group, and at a
# position whose site group is the whole point group the band representation a
# site irrep induces is, at GM, that irrep itself.  So each symmorphic group's
# tables hold its point group's double-valued names, which are the EBR entries
# at the origin.  The characters of each name on every operation are those of
# the representation its entry induces at GM, and the tables hold them with the
# SU(2) matrix every character is written against.  That is the point group's
# own vocabulary, read and not typed, in the tables' convention.
#
# A site row is named by comparing characters with it, operation by operation.
# The site group and the reference are one point group in orientations that
# can differ, so the comparison runs through a rotation carrying the one onto
# the other.  Each character follows the operation it belongs to, by its axis,
# its sense and the sign of the SU(2) matrix the tables lift it by, and never
# by its place in a list.


#: The order of a proper rotation, from its trace, which is ``1 + 2 cos θ`` and
#: an integer for every crystallographic one.
_ORDER_OF_TRACE = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}


def _orthogonal_of(S, det) -> np.ndarray:
    r"""The Cartesian matrix of an operation, read off its SU(2) matrix ``S``.

    :math:`R_{ij} = \tfrac12\,\mathrm{tr}(\sigma_i S \sigma_j S^\dagger)` is the
    rotation ``S`` lifts, in the frame the tables write their SU(2) matrices in.
    The tables lift an improper operation by its proper part, so the matrix
    returned is that rotation times ``det``, the operation's determinant.
    """
    S = np.asarray(S, dtype=complex)
    pauli = _spacegroup._PAULI
    rotation = np.real(np.einsum("iab,bc,jcd,da->ij", pauli, S, pauli, S.conj().T)) / 2.0
    return float(det) * rotation


@lru_cache(maxsize=None)
def _tabulated_operations(number_str: str, magnetic: bool) -> tuple:
    """``(R, O, S)`` for each operation of one group's double-valued table.

    ``number_str`` is the group's number, and ``magnetic`` whether the table is
    read among the magnetic groups, as a grey group's is.  ``R`` is the integer
    matrix the table writes the operation in, in its lattice basis, ``S`` the
    table's SU(2) matrix, and ``O`` the Cartesian matrix in the frame of ``S``.
    The unitary operations come first and the antiunitary ones after them, each
    in the table's own order.  That is the order of
    :attr:`..groups.GroupData._symmetries`, so operation ``ind`` is entry
    ``ind - 1``.
    """
    return _operations_of(_spacegroup._irrep_table(number_str, True, magnetic))


def _operations_of(table) -> tuple:
    """:func:`_tabulated_operations` of a ``table`` already read."""
    out = []
    for sym in list(table.u_symmetries) + list(table.au_symmetries):
        R = np.rint(np.asarray(sym.R, dtype=float)).astype(int)
        det = int(round(float(np.linalg.det(R))))
        out.append((R, _orthogonal_of(sym.S, det), np.asarray(sym.S, dtype=complex)))
    return tuple(out)


def _magnetic_numbers() -> dict:
    """The BNS number of every magnetic space group, by its symbol.

    The numbers are read from ``irrep``'s own list, ``msg_numbers.data``.  The
    grey group of a space group is the one whose symbol is the space group's
    with ``1'`` appended.  Returns a mapping from symbol to number,
    ``"P4mm1'"`` to ``"99.164"``.
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
    """The first letter in an EBR list with one site per primitive cell.

    ``raw`` is a group's EBR file as ``irreptables`` ships it, and ``centering``
    its centering translations.  The position sought has a conventional
    multiplicity one more than the number of centering translations, and its
    site group is the whole point group.  In a symmorphic group the first such
    position is the origin.  Returns ``""`` where the list has none.
    """
    per_cell = 1 + len(tuple(centering))
    for entry in raw["ebrs"]:
        match = re.match(r"^(\d+)([a-zA-Z])\(", str(entry["wyckoff_position"]))
        if match is not None and int(match.group(1)) == per_cell:
            return match.group(2)
    return ""


@lru_cache(maxsize=None)
def _origin_coreps(grey: bool) -> dict:
    """Each point group's double-valued site irreps, read at symmorphic origins.

    A space group is taken as symmorphic where every translation its table lists
    is a lattice or centering vector.  For each such group, in the order of its
    number, the reference holds the table's number, its unitary operations
    ``(R, O, S)`` (:func:`_tabulated_operations`), and each double-valued EBR
    entry at its origin: the site irrep's name, and the characters on those
    operations of the one representation the entry induces at ``GM``.  With
    ``grey=True`` the grey groups' tables are read, where time reversal joins a
    complex-conjugate pair into one corep.  Returns a mapping from each point
    group's symbol to its references.

    Raises
    ------
    ..refusal.InternalError
        Where ``irrep``'s list names no grey group of a symmorphic group, a grey
        group's table does not carry each operation once with time reversal and
        once without, or an entry at the origin induces anything but one
        representation at ``GM`` of the entry's own dimension.
    """
    return _origin_references(grey, True)


@lru_cache(maxsize=None)
def _origin_irreps(grey: bool) -> dict:
    """Each point group's single-valued site irreps, read at symmorphic origins.

    The references of :func:`_origin_coreps`, each with the single-valued EBR
    entries at the origin and the characters of the one ``GM`` representation
    each induces, on the same operations.  The double-valued table of the same
    group orients the operations, since a single-valued table carries no SU(2)
    matrix.  With ``grey=True`` the grey groups' tables are read, where time
    reversal joins a complex-conjugate pair into one corep (``1E2E``).  Returns
    a mapping from each point group's symbol to its references.

    Raises
    ------
    ..refusal.InternalError
        As :func:`_origin_coreps`, or where the single- and double-valued tables
        of one group list different operations, or the same ones in another
        order.
    """
    return _origin_references(grey, False)


def _origin_references(grey: bool, double: bool) -> dict:
    """The references of :func:`_origin_coreps` or :func:`_origin_irreps`.

    ``double`` chooses the double-valued entries, as :func:`_origin_coreps` reads
    them, and ``grey`` the grey groups' tables.
    """
    named = _magnetic_numbers() if grey else {}
    out: dict = {}
    for sg in range(1, 231):
        plain = _spacegroup._irrep_table(str(sg), True, False)
        centering = _spacegroup._centering(plain)
        if not all(_spacegroup._is_translation(np.asarray(sym.t, dtype=float), centering)
                   for sym in plain.symmetries):
            continue
        number = (named.get(str(plain.name).strip() + "1'", "") if grey else str(sg))
        if not number or number.split(".")[0] != str(sg):
            raise InternalError(
                f"irrep's list of magnetic groups names no grey group of space group "
                f"{sg} ({plain.name}). Report it, with what tables_status() returns.")
        table = _spacegroup._irrep_table(number, True, grey) if grey else plain
        unitary = [(i + 1, sym) for i, sym in enumerate(table.symmetries)
                   if not sym.time_reversal]
        if grey and (len(table.au_symmetries) != len(unitary)
                     or len(unitary) != len(plain.symmetries)):
            raise InternalError(
                f"#{number} is listed as the grey group of space group {sg}, and its "
                "table does not carry each operation once with time reversal and once "
                "without. Report it, with what tables_status() returns.")
        ops = tuple(_operations_of(table)[:len(unitary)])
        # the characters: the double-valued table's own, or the single-valued
        # table of the same group, which has to list the same operations
        source = table if double else _spacegroup._irrep_table(number, False, grey)
        if not double and (len(source.symmetries) != len(table.symmetries) or any(
                np.abs(np.asarray(a.R, dtype=float)
                       - np.asarray(b.R, dtype=float)).max() > _TABLE_MATRIX_TOL
                or bool(a.time_reversal) != bool(b.time_reversal)
                for a, b in zip(source.symmetries, table.symmetries))):
            raise InternalError(
                f"the single- and double-valued tables of #{number} do not list the "
                "same operations in the same order. Report it, with what "
                "tables_status() returns.")
        gamma = {str(irr.name): {int(k): complex(v) for k, v in irr.characters.items()}
                 for irr in source.irreps if str(irr.kpname) == "GM"}
        raw = _spacegroup._ebr_file(number, double)
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
                    f"at the origin of #{number}, the EBR list has the site irrep {name} "
                    f"induce {at_gamma} at GM, which is not one little-group "
                    "representation of its dimension in the irrep table. Report it, with "
                    "what tables_status() returns.")
            chars = gamma[at_gamma[0][0]]
            names.append((name, tuple(chars[i] for i, _sym in unitary)))
        symbol = _sitechars.point_group_of(R for R, _O, _S in ops)
        out.setdefault(symbol, []).append((number, ops, tuple(names)))
    return {symbol: tuple(refs) for symbol, refs in out.items()}


def _matrix_key(M) -> tuple:
    """A matrix ``M``, rounded to three decimals, as a hashable key."""
    return tuple(float(x) + 0.0 for x in np.round(np.asarray(M, dtype=float), 3).ravel())


def _proper_axis(R, O) -> tuple:
    """``(axis, order, det)`` of one operation.

    ``axis`` and ``order`` are those of the operation's proper rotation, and
    ``det`` is its determinant.  ``R`` is the operation's integer matrix and
    ``O`` its Cartesian matrix.  The order and the determinant are read off
    ``R`` exactly, and the unit axis off ``O``, oriented so that the rotation
    turns counterclockwise about it, a twofold's either way.  ``axis`` is
    ``None`` for the identity and inversion.
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
    """The right-handed orthonormal frame whose first axis lies along ``x``.

    Its second axis is the part of ``y`` orthogonal to ``x``, normalized.
    Returns the three axes as the columns of a 3 by 3 matrix.
    """
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
    """Whether the rotation ``Q`` carries ``source`` onto ``target``.

    ``source`` and ``target`` are ``(R, O)`` per operation, and only the
    Cartesian matrices ``O`` are read.  The two are compared as sets, each
    ``Q @ O @ Q.T`` of ``source`` against the ``O`` of ``target``, to three
    decimals (:func:`_matrix_key`).
    """
    Q = np.asarray(Q, dtype=float)
    return ({_matrix_key(Q @ np.asarray(O, dtype=float) @ Q.T) for _R, O in source}
            == {_matrix_key(O) for _R, O in target})


def _conjugators(source, target, tol=_TABLE_MATRIX_TOL) -> tuple:
    """The proper rotations carrying the operations ``source`` onto ``target``.

    ``source`` and ``target`` are ``(R, O)`` per operation, as
    :func:`_proper_axis` reads them, and a rotation carries the one onto the
    other as :func:`_carries` says.  A candidate takes an axis of the highest
    order in ``source`` onto an axis of the same order and determinant in
    ``target``, either way along it.  Where ``source`` has a second axis, the
    candidate also takes it onto one at the same angle to the first.  Where it
    has one axis, one turn about it stands for all, since every turn about that
    axis commutes with every operation.  ``tol`` is how far two cosines, or the
    entries of two rotations, may differ and be equal.  Returns each rotation
    that carries the whole set, once, as a Cartesian matrix: the identity alone
    where ``source`` has no axis and is ``target``, none where nothing carries.
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
    r"""An SU(2) matrix lifting the proper rotation ``Q``.

    ``Q`` is a Cartesian matrix, replaced first by the nearest orthogonal one.
    The matrix returned is ``cos(θ/2) − i sin(θ/2) n·σ`` for a turn by
    ``θ`` about the unit axis ``n``, read off the rotation's quaternion.
    Which of the two lifts is returned is not fixed, so the matrix is used only
    to conjugate.
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
    """The characters of a reference, carried by ``Q`` to a site's operations.

    ``reference`` is one reference of :func:`_origin_coreps` or
    :func:`_origin_irreps`: its table's number, its operations ``(R, O, S)``,
    and its names with their characters.  ``here`` is ``(R, O, S)`` per
    operation of the site, as :func:`_tabulated_operations` gives them, and
    ``Q`` a proper rotation in the tables' frame.  Each operation of the site
    is matched to the reference operation ``Q`` carries onto it, by its
    Cartesian matrix to three decimals.  With ``double=True``, the default, a
    character also takes the sign relating the site operation's SU(2) matrix
    to the reference operation's carried by the lift of ``Q``
    (:func:`_su2_of`).  The two lifts of one operation differ by the rotation
    through 2π, on which a double-valued character changes sign.  With
    ``double=False`` a character is carried as it is.  ``tol`` is how far the
    relating factor may be from 1 or -1.  Returns
    ``[(name, [character per site operation])]``, one pair per name of the
    reference.  Returns ``None`` where an operation of the site is the image of
    none, or its SU(2) matrix is not the carried one up to a sign.
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
    r"""The double-valued point-group name of each row of a site table.

    ``table`` is a site table of ``group``, its rows the orbital characters.
    A row times the operations' spin-1/2 lift is the double-valued site
    representation the states carry.  It is named by the double-valued irreps
    of the site's point group, as the tables name them at the origin of a
    symmorphic group (:func:`_origin_coreps`), grey where ``group`` carries
    time reversal.  A reducible one is spelled as the sum of what it holds
    (``"-1E2E + -E1"``).  The names are carried onto the site by
    :func:`_point_group_reading`.  Returns one name per row of ``table``,
    ``""`` where nothing names it, and every name ``""`` where
    :func:`_point_group_reading` returns ``None``.
    """
    names = _point_group_reading(group, table, double=True)
    return ("",) * len(table.labels) if names is None else names


def _point_group_irreps(group, table):
    r"""The single-valued point-group name of each row of a site table.

    ``table`` is a site table of ``group``.  Each row, an orbital character, is
    named by the single-valued irreps of the site's point group, as the tables
    name them at the origin of a symmorphic group (:func:`_origin_irreps`),
    grey where time reversal joins the table's rows (``table.antiunitary``
    nonempty).  So one irrep has the tables' name at every position, the ones
    their entries list and the ones they do not.  The names are carried onto
    the site by :func:`_point_group_reading`.  Returns one name per row,
    ``""`` where nothing names it, or ``None`` where
    :func:`_point_group_reading` returns ``None``.
    """
    return _point_group_reading(group, table, double=False)


def _point_group_reading(group, table, *, double):
    """The point-group names of a site table's rows, single- or double-valued.

    They are what :func:`_point_group_names` and :func:`_point_group_irreps`
    return.  ``group`` is the measured group and ``table`` one of its site
    tables.  ``double`` chooses the double-valued names, of each row times the
    spin-1/2 lift, over the single-valued ones.  The reference is a symmorphic
    group of the site's point group, read at its origin.  Where the site's
    operations, as Cartesian matrices in the tables' frame, are those of such
    an origin, the lowest-numbered one is the reference and each character is
    read off the operation it belongs to.  Otherwise the reference is the
    lowest-numbered group of the point group, and its characters are carried
    by every rotation taking its origin's operations onto the site's
    (:func:`_conjugators`).  A row is named where it is a sum of the
    reference's irreps with nonnegative integer multiplicities, not all zero,
    and every rotation gives the same sum.  Where they give different sums the
    row is left unnamed.  This happens to the two members of a
    complex-conjugate pair at an axis no symmorphic origin shares, in a group
    without time reversal.  Returns one name per row, ``""`` for a row not
    named.  Returns ``None`` where the tables are not set up for ``group``,
    ``double`` is asked of a group without spin, no reference holds the point
    group, the group's table does not list as many operations as the group
    holds, or the reference's operations are not all carried onto the site's.
    """
    if not getattr(group, "_tables_ready", False) or (double and not group.spinor):
        return None
    symbol = _sitechars.point_group_symbol(group, table.indices)
    if double:
        references = _origin_coreps(_spacegroup._magnetic_type(group) == "II").get(symbol, ())
    else:
        references = _origin_irreps(bool(getattr(table, "antiunitary", ()))).get(symbol, ())
    tabulated = _tabulated_operations(str(group.number_str), bool(group._magnetic))
    if not references or len(tabulated) != len(group._symmetries):
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
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    lift = ([complex(np.trace(np.asarray(group._symmetries[pos_of[int(ind)]].spinor_rotation)))
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
