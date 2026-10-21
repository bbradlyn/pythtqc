r"""The model's representation of its space group, and the atomic-limit table of its orbitals.

:func:`..groups.realized_group` measures, for each operation ``g = {R|t}`` of
the cell, every matrix ``X`` with ``X H(k) = H(Rk) X`` at once on a grid of
``k``, ``X H(k)* = H(−Rk) X`` for one combined with time reversal, assuming
nothing about what ``g`` does to the orbitals: no solution is a proof that
``g`` is broken, and at the identity the solutions are the operators the
model conserves.  From the operations that hold it builds
:class:`SpaceGroupRep`, one matrix per operation on the model's states in the
tables' convention, spin along ``+z`` in a right-handed frame, through which
every band label is read; and :class:`AtomicLimitTable`, what each orbital
carries at its site.

The hoppings fix how the orbitals transform relative to one another, not what
they are: one one-dimensional character of the group is left open on each part
of the crystal the hoppings couple, real where the group has time reversal and
any, complex ones included, where it has none: the spin–orbit ``I4_1`` model in
its unitary group, its ``4a`` orbitals declared ``B``, is returned with
``χ(4⁺) = −i``.  ``characters=`` declares it, and what the
declarations leave open is assigned by the lowest-angular-momentum rule; each
row of the table says which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _bloch
from . import _sitechars
from .refusal import Refusal
from ._tolerance import ANSATZ_GATE, EXACT_TOL
from ._text import (
    _content_terms, _ebrs_of, _fmt_char, _index_ranges, _orbitals_named, _sum_lines,
    _sum_words)
from ._seitz import operations
from ._matrices import (
    NormalizedRep, RepAnchor, SiteMultiplets, bloch_operator, covariance_gate,
    normalize_rep)

__all__ = ["RepAnchor",
           "NormalizedRep", "AtomicLimitRow", "AtomicLimitTable", "BandRepresentation",
           "MultiplicitySpace", "SiteMultiplets", "SpaceGroupRep",
           "normalize_rep",
           "bloch_operator", "ANSATZ_GATE", "EXACT_TOL", "covariance_gate"]

# --------------------------------------------------------------------------
# the atomic-limit table
# --------------------------------------------------------------------------

#: The one sentence :meth:`AtomicLimitTable.summary` prints under the table
#: wherever a row's site irrep rests on the lowest-``l`` rule.
_LOWEST_L_LEGEND = (
    "the hoppings fix how the orbitals transform relative to one another, not what "
    "they are; where several site irreps fit, the one carried by the "
    "lowest-angular-momentum functions is shown, and `characters=` states another.")

#: What :attr:`AtomicLimitRow.determined_by` reads on a declared row whose
#: declared irrep the Hamiltonian's own blocks do not carry.
_NOT_CARRIED = "declaration not carried"

@dataclass(frozen=True)
class AtomicLimitRow:
    """One orbital's row of the atomic-limit table: its Wyckoff position, the site irrep it carries, and how that is known.

    ``rep.characters[i]`` is orbital ``i``'s row, and ``row.summary()`` prints
    it.

    Attributes
    ----------
    orbital, site : int
        The orbital, and the site it sits on.
    wyckoff : str
        The site's Wyckoff position in the tables' setting, multiplicity and
        letter (``"3c"``); ``""`` where the tables do not reach the cell.
    site_symmetry : str
        The Hermann–Mauguin symbol of the site-symmetry group (``"-42m"``).
    function : str
        The orbital function declared, as ``characters=`` was given it
        (``"px"``, ``"dx2-y2"``, the shell ``"p"``); ``""`` where none was.
    multiplet : tuple of int
        The orbitals the Hamiltonian binds this one with, which carry its site
        irrep together.
    irrep : str
        The site irrep the states carry, in the site group of the group
        returned, and with spin the double-valued site corep: ``-E1`` for an
        ``s`` orbital at diamond's ``8a``.  A multiplet holding several irreps
        reads their sum, ``"2x-E1 + -1E2E"`` for a spin–orbit ``p`` shell at a
        ``3m`` site, and one holding copies of one irrep reads it on each row;
        ``""`` where nothing names it.
    named_by : str
        Where the name comes from: ``"tables"``, the tables' band
        representations at this position; ``"point group"``, the site's point
        group at the origin of a symmorphic group; ``"Mulliken"``, the row's
        Mulliken symbol; or ``"unnamed"``.
    determined_by : str
        How the site irrep is known: ``"declaration"``; ``"declaration not
        carried"``, declared and not carried by the Hamiltonian's own blocks;
        ``"hamiltonian"``, the same under every labelling the declarations
        leave open; or ``"lowest-l"``, chosen by the lowest-angular-momentum
        rule, which ``characters=`` overrides.
    dim : int
        The dimension of what ``irrep`` names: twice the orbital factor's with
        spin.
    site_group : tuple of int
        The indices of the unitary operations that fix the site.
    character, generator_character : dict
        The orbital factor's character, single-valued with spin or without: on
        every operation of the site group by index, and on one operation of each
        class by name.
    declared_as : object or None
        Exactly what was written for this orbital.
    declared_irrep, declared_in, reduced_in : str
        The irrep, or sum, that declaration resolved to; the site group and
        group it was read in (``"3m in P3m11' #156.50"``); and, where the
        declared content lowered the cell first, the site group and group it
        was resolved in (``"m in Cm1' #8.33"``).  ``""`` where nothing applies.
    declaration_broke : tuple of str
        The operations the declared content breaks.
    note : str
        How a declaration was read; on a declared spinor basis, which member of
        its pair the row is; for a name Mulliken's convention gives another
        row, which row; what a declaration not carried stated; and where the
        cell could not be matched to the tables' setting, why no name comes
        from them.
    residual : float
        How far the orbital's block misses its character times its spin factor,
        or, on a multiplet of several orbitals, misses multiplying as the site
        group does: a large one says the irrep is not to be trusted.
    """

    orbital: int
    site: int
    wyckoff: str
    site_symmetry: str
    function: str
    multiplet: tuple
    irrep: str
    named_by: str
    determined_by: str
    dim: int
    site_group: tuple
    character: dict
    generator_character: dict
    declared_as: Any = None
    declared_irrep: str = ""
    declared_in: str = ""
    reduced_in: str = ""
    declaration_broke: tuple = ()
    note: str = ""
    residual: float = 0.0

    def summary(self) -> str:
        """The row, a line for each part that applies, ending with the attribute that holds it."""
        named = {"tables": "the tables", "point group": "the site's point group",
                 "Mulliken": "Mulliken's convention"}.get(self.named_by)
        lines = [f"orbital {self.orbital}, site {self.site}, at {self.wyckoff or '?'} "
                 f"({self.site_symmetry}): {self.irrep or 'unnamed'}"
                 + (f", named by {named}" if named else "") + "   (.irrep, .named_by)"]
        echo = ""
        if self.determined_by in ("declaration", _NOT_CARRIED):
            said = repr(self.declared_as) + (
                f" ({self.declared_irrep})" if self.declared_irrep
                and self.declared_irrep != str(self.declared_as) else "")
            lines.append(f"  declared {said}, read at the {self.declared_in}"
                         + ("" if self.determined_by == "declaration" else
                            ", and the Hamiltonian's own blocks do not carry it")
                         + "   (.declared_as, .declared_irrep, .declared_in)")
            echo = f"{self.declared_as!r} resolved to {self.declared_irrep} on this site group"
        else:
            lines.append("  " + ("fixed by the Hamiltonian" if self.determined_by == "hamiltonian"
                                 else "chosen by the lowest-angular-momentum rule among the "
                                      "site irreps the hoppings allow")
                         + "   (.determined_by)")
        if self.reduced_in:
            lines.append("  the declared content lowered the cell before any hopping was read, "
                         f"and resolves in the {self.reduced_in}   (.reduced_in)")
        if self.declaration_broke:
            lines.append("  the declaration breaks " + ", ".join(self.declaration_broke)
                         + "   (.declaration_broke)")
        others = [int(o) for o in self.multiplet if int(o) != int(self.orbital)]
        if others:
            lines.append(f"  one multiplet with {_orbitals_named(others)}   (.multiplet)")
        lines.append("  orbital characters: "
                     + ", ".join(f"{k}={_fmt_char(v)}" for k, v in self.generator_character.items())
                     + "   (.generator_character)")
        said = "; ".join(part for part in str(self.note).split("; ") if part and part != echo)
        if said:
            lines.append(f"  {said}   (.note)")
        return "\n".join(lines)

@dataclass(frozen=True)
class BandRepresentation:
    """The band representation an orbit of orbitals induces, as a sum of elementary band representations.

    The sum is written at a position the orbitals can be moved to along their
    variety, their own first.  Where none carries the content, as for ``E`` at
    ``3c`` of ``P-43m``, it is the sum at other positions with the orbit's own
    stable real-space invariants, or, where the group has none tabulated, the
    sum at the first position on the closure of a variety through the site
    that carries it; on a free-standing layer only positions within the layer
    are read.  It is the band representation and nothing finer: which site
    irrep goes to which term is a subduction the tables do not carry.
    ``br.summary()`` prints it; it is one entry of
    ``rep.characters.band_representations``.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals on the orbit's sites.
    wyckoff, site_symmetry : str
        The orbit's Wyckoff position, multiplicity and letter (``"6c"``), and
        its site-symmetry group (``"3m"``).
    terms : tuple
        ``(multiplicity, elementary band representation, position)`` for each
        term, as :func:`..groups.ebr_data` writes them
        (``(1, "-E1g↑G(2)", "3a(-3m1',-3m)")``); empty where no sum is returned.
    alternatives : tuple
        The other sums with the same content and the same stable real-space
        invariants, which nothing here distinguishes from it, each written as
        ``terms`` is; empty where the group has no tabulated invariants.
    note : str
        Where no sum is returned, why: none carries the content, several do and
        nothing here says which is induced, or the content could not be
        identified.
    """

    orbitals: tuple
    wyckoff: str
    site_symmetry: str
    terms: tuple
    alternatives: tuple = ()
    note: str = ""

    def spelled(self, terms=None) -> str:
        """A sum on one line in the labels :func:`..groups.ebr_data` lists, ``"1 x -E1g↑G(2)@3a(-3m1',-3m) + 1 x -E1u↑G(2)@3a(-3m1',-3m)"``, or ``""``.

        :attr:`terms` by default, or one of :attr:`alternatives`.
        """
        terms = self.terms if terms is None else terms
        return " + ".join(f"{m} x {name}@{position}" for m, name, position in terms)

    def sentence(self) -> str:
        """The sentence the atomic-limit table's summary prints for this orbit, where it prints one.

        Orbits at one Wyckoff position that induce the same sums share one
        line there instead, each orbit's orbitals named.
        """
        one = len(self.orbitals) == 1
        who = _orbitals_named(self.orbitals)
        if not self.terms:
            return f"{who} at {self.wyckoff}: {_no_sum(self)}"
        return (f"{who} at {self.wyckoff} induce{'s' if one else ''} "
                f"{_ebrs_of(self.terms)}{_not_distinguished(self)}")

    def summary(self) -> str:
        """The sum this orbit induces, then those it is not distinguished from, a line each."""
        who = (f"{_orbitals_named(self.orbitals)} at {self.wyckoff or '?'} "
               f"({self.site_symmetry})")
        if not self.terms:
            return f"{who}: {_no_sum(self)}   (.note)"
        lines = [f"{who} induce{'s' if len(self.orbitals) == 1 else ''} "
                 f"{_ebrs_of(self.terms)}   (.terms)"]
        if self.alternatives:
            several = len(self.alternatives) > 1
            lines.append(f"  not distinguished from th{'ese sums' if several else 'is sum'}, "
                         f"which ha{'ve' if several else 's'} the same content and the same "
                         "stable real-space invariants   (.alternatives)")
            lines += _sum_lines(self.alternatives)
        return "\n".join(lines)

def _fold(k) -> str:
    """``"twofold"``, ``"threefold"``, ...: how many identical copies a pseudospin makes."""
    return {2: "twofold", 3: "threefold", 4: "fourfold"}.get(int(k), f"{int(k)}-fold")

def _in_words(n) -> str:
    """A small count in words, ``"two"``; a larger one in figures."""
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(int(n), str(int(n)))

@dataclass(frozen=True)
class MultiplicitySpace:
    """A pseudospin the Hamiltonian conserves on an orbit of orbitals, and the site content assigned its copies.

    Where the Hamiltonian holds identical copies of a part of the crystal that
    nothing joins, every band there is as many times degenerate as there are
    copies, and nothing the hoppings measure says what the copies carry, which
    moves the site content itself.  A declaration on their orbitals states it,
    and the lowest-angular-momentum rule assigns it otherwise, the copies read
    copy by copy first.  ``space.summary()`` prints it; it is one entry of
    ``rep.characters.multiplicity_spaces``.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals the copies are made of.
    wyckoff : str
        The Wyckoff position of the site whose content is read; ``""`` where the
        tables do not reach the cell.
    dim : int
        The number of copies, the degeneracy of every band on these orbitals.
    content : str
        The site content returned.
    contents : tuple of str
        The site contents the representations of the copies the group's own
        table admits give, in the order the lowest-angular-momentum rule ranks
        them; with spin, two can name one content.
    n_representations : int
        How many representations of the copies the group's own table admits.
    determined_by : str
        ``"declaration"``; ``"lowest-l"``, the rule's choice; or
        ``"hamiltonian"``, where the table admits one representation.
    """

    orbitals: tuple
    wyckoff: str
    dim: int
    content: str
    contents: tuple
    n_representations: int = 1
    determined_by: str = "lowest-l"
    # how many Wannier functions of the copies one site holds -- orbitals, or
    # declared spinor pairs where ``_pairs`` -- which says what one copy is
    _per_site: int = field(default=0, repr=False, compare=False)
    _pairs: bool = field(default=False, repr=False, compare=False)

    def _distinct(self) -> tuple:
        """The site contents admitted, each named once, in the rule's order.

        With spin, different orbital contents can lift to one site corep
        content, ``2xA1`` and ``2xA2`` of ``3m`` both to ``2x-E1``, and a sum
        can be spelled in either order of its terms, ``-E1 + -E2`` and
        ``-E2 + -E1``: a content is the irreps it holds and how many of each.
        """
        seen, out = set(), []
        for content in self.contents:
            key = tuple(sorted((name, n) for n, name in _content_terms(content)))
            if key not in seen:
                seen.add(key)
                out.append(content)
        return tuple(out)

    def sentence(self) -> str:
        """The sentence the atomic-limit table's summary prints for this pseudospin."""
        head = (f"orbitals {tuple(self.orbitals)}"
                + (f" at {self.wyckoff}" if self.wyckoff else "")
                + f" carry a {_fold(self.dim)} pseudospin the Hamiltonian conserves")
        content = self.content or "(unnamed)"
        if self.determined_by == "declaration":
            return (f"{head}, and what its copies carry it does not fix: the declaration "
                    f"states the site content {content}, of the {len(self._distinct())} the "
                    f"group's own table admits")
        if len(self._distinct()) > 1:
            return (f"{head}, and what its copies carry it does not fix: the "
                    f"lowest-angular-momentum rule assigns the site content {content}, of "
                    f"the {len(self._distinct())} the group's own table admits")
        if int(self.n_representations) > 1:
            return (f"{head}; the {int(self.n_representations)} representations of its "
                    f"copies the group's own table admits all give the site content "
                    f"{content}, and the lowest-angular-momentum rule chose the one "
                    f"returned among them")
        return (f"{head}; the group's own table admits one representation of its copies, "
                f"giving the site content {content}")

    def summary(self) -> str:
        """The pseudospin, what its copies carry and how that was chosen, a line each."""
        how = {"declaration": "stated by the declaration",
               "lowest-l": "assigned by the lowest-angular-momentum rule",
               "hamiltonian": "the one content the group's own table admits"}.get(
                   self.determined_by, self.determined_by)
        # what one copy is: the Wannier functions it holds at one site, orbitals
        # or declared spinor pairs, and whether the orbit has several sites
        unit = "spinor pair" if self._pairs else "orbital"
        held = len(self.orbitals) // (2 if self._pairs else 1)
        per_site = int(self._per_site) or held
        each = max(1, per_site // max(1, int(self.dim)))
        copy = (f"one {unit}'s states" if each == 1 else f"{_in_words(each)} {unit}s' states")
        lines = [f"{_orbitals_named(self.orbitals)}"
                 + (f" at {self.wyckoff}" if self.wyckoff else "")
                 + f": {_in_words(self.dim)} identical copies of {copy}"
                 + (" at each site" if held > per_site else "")
                 + ", a pseudospin the Hamiltonian conserves   (.dim)",
                 f"  the copies carry {self.content or 'a content no irrep names'}, {how}"
                 "   (.content, .determined_by)"]
        distinct = self._distinct()
        if len(distinct) > 1:
            lines.append("  every content the group's own table admits, in the rule's order: "
                         + ", ".join(distinct) + "   (.contents)")
        if int(self.n_representations) > len(distinct):
            lines.append(f"  {int(self.n_representations)} representations of the copies give "
                         + ("these contents" if len(distinct) > 1 else "that content")
                         + "   (.n_representations)")
        return "\n".join(lines)

@dataclass(frozen=True)
class AtomicLimitTable:
    """What each orbital carries at its site: the table an atomic limit is written up in, a row per orbital.

    ``rep.characters`` is the table; ``table.summary()`` prints it, and
    ``table[i]`` is orbital ``i``'s :class:`AtomicLimitRow`.  The table is where
    a mis-oriented declaration shows, in each row's orbital characters.

    Attributes
    ----------
    orbitals : tuple of AtomicLimitRow
        One row per orbital, in the model's order.
    irreps : tuple of str
        Each orbital's site irrep, in the model's order.
    band_representations : tuple of BandRepresentation
        The band representation each orbit of orbitals induces, a sum of
        elementary band representations, in the order of the orbits; empty
        where the tables do not reach the cell.
    multiplicity_spaces : tuple of MultiplicitySpace
        One per orbit on which the Hamiltonian conserves a pseudospin, with the
        site content assigned its copies; empty where none does.
    declarations : tuple of dict
        Each declaration as it was read: ``orbitals``, ``declared_as`` and the
        ``declared_irrep`` it resolved to, ``dim``, ``note``, ``declared_in``,
        ``reduced_in``, the operations it ``broke``; ``mulliken``, each name
        written that Mulliken's convention gives another row, with that row's
        name; and ``returned``, what its multiplets carry together.
    checks : tuple of dict
        Each declaration against each operation of its site group the hoppings
        say something about, ``orbitals``, ``ind``, ``name``, ``measured`` and
        ``declared``: the table a ``declaration_breaks_group`` refusal prints.
        ``measured`` is read against a multiplet of the declaration's own part
        of the crystal, so the two differ by a one-dimensional character of
        that part.
    frame : dict
        ``{ind: name}``, every operation represented, named in the model's
        frame.
    spin : ..spin.SpinBasis or None
        The declared spinor basis, where one was; the table then has a ``spin``
        column, each row's member of its pair.
    residual : float
        The largest ``residual`` of any row.
    site_tables : dict
        ``{site: table}``, each site group's character table.
    """

    orbitals: tuple
    declarations: tuple
    checks: tuple
    frame: dict
    residual: float
    site_tables: dict = field(default_factory=dict, repr=False)
    spin: Any = None
    band_representations: tuple = ()
    multiplicity_spaces: tuple = ()
    # the chooser's own record of the labellings that give every orbital the
    # site irrep the one returned gives it and label some band differently:
    # each coupled part's states, and per part the band labels of every energy
    # multiplet under the one returned and under each of those, with the key
    # that set it aside -- read by `sewing.hsp_table` to say where a set of
    # bands' labels rest on that choice
    _labellings: dict = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.orbitals)

    def __getitem__(self, orbital) -> AtomicLimitRow:
        return self.orbitals[int(orbital)]

    @property
    def irreps(self) -> tuple:
        """Each orbital's site irrep, in the model's order."""
        return tuple(o.irrep for o in self.orbitals)

    def summary(self) -> str:
        """The table, a row per orbital, and a line under it for each thing the rows do not say.

        The columns are the row's fields; ``orbital characters`` is the orbital
        factor's character on one operation of each class, without spin.  Under
        the rows, each line ending with the attribute that holds it: a
        declaration that lowered the group, that the Hamiltonian's own blocks
        do not carry, or that is written in a name Mulliken's convention gives
        another row, with what was declared, where it was read and what is
        returned in its place; an orbit whose band representation is not its
        rows' own elementary band representations at its own position; and a
        conserved pseudospin, with the site content assigned its copies.  Where
        a row reads ``lowest-l``, a line says what that means.
        """
        head = ["orbital", "Wyckoff", "site symmetry", "function"]
        spins = None
        if self.spin is not None:
            sgn = np.asarray(self.spin.spin_of_orbital, dtype=int)
            spins = ["up" if sgn[o.orbital] > 0 else "down" for o in self.orbitals]
            head.append("spin")
        head += ["irrep", "named by", "determined by"]
        body = []
        for i, o in enumerate(self.orbitals):
            cells = [str(o.orbital), o.wyckoff, o.site_symmetry, o.function]
            if spins is not None:
                cells.append(spins[i])
            cells += [o.irrep, o.named_by, o.determined_by]
            body.append(cells)
        width = [max([len(h)] + [len(r[j]) for r in body]) for j, h in enumerate(head)]
        rows = ["  ".join(f"{h:>{w}s}" for h, w in zip(head, width)) + "  orbital characters"]
        for o, cells in zip(self.orbitals, body):
            chars = ", ".join(f"{k}={_fmt_char(v)}" for k, v in o.generator_character.items())
            rows.append("  ".join(f"{c:>{w}s}" for c, w in zip(cells, width)) + f"  {chars}")
        for d in self.declarations:
            orbs = tuple(int(a) for a in d["orbitals"])
            row = self.orbitals[orbs[0]]
            dropped = row.determined_by == _NOT_CARRIED
            if not (d.get("reduced_in") or d.get("broke") or dropped or d.get("mulliken")):
                continue
            who = f"orbital {orbs[0]}" if len(orbs) == 1 else f"orbitals {orbs}"
            said = f"{d['declared_as']!r}" + (
                f" ({d['declared_irrep']})" if d["declared_irrep"]
                and d["declared_irrep"] != d["declared_as"] else "")
            parts = [f"  {who}: declared {said}, read at the {d['declared_in']}"]
            if d.get("mulliken"):
                parts.append(_sitechars.mulliken_clause(d["mulliken"].items()))
            if d.get("reduced_in"):
                parts.append("the declared content lowered the cell before any hopping "
                             f"was read and resolves as "
                             f"{row.declared_irrep or '(unnamed)'} of the {d['reduced_in']}")
            if d.get("broke"):
                parts.append("the declaration breaks " + ", ".join(d["broke"]))
            if dropped:
                parts.append("the Hamiltonian's own blocks do not carry it under the "
                             "one-dimensional character returned")
            parts.append(f"returned as {d.get('returned', row.irrep) or '(unnamed)'} of "
                         + (row.site_symmetry
                            or f"the site group of order {len(row.site_group)}"))
            rows.append("; ".join(parts) + "   (.declarations)")
        alike: dict = {}
        for orbit in self.band_representations:
            if not _its_rows_own(self, orbit):
                key = (orbit.wyckoff, orbit.terms, orbit.alternatives, orbit.note)
                alike.setdefault(key, []).append(orbit)
        for orbits in alike.values():
            rows.append("  " + _orbits_sentence(orbits) + "   (.band_representations)")
        for space in self.multiplicity_spaces:
            rows.append("  " + space.sentence() + "   (.multiplicity_spaces)")
        if any(o.determined_by == "lowest-l" for o in self.orbitals):
            rows.append("  lowest-l: " + _LOWEST_L_LEGEND)
        return "\n".join(rows)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"AtomicLimitTable({'/'.join(self.irreps)}"
                + (", lowest-l" if any(o.determined_by == "lowest-l"
                                       for o in self.orbitals) else "") + ")")

def _its_rows_own(table, orbit) -> bool:
    """Whether an orbit's band representation is its rows' own elementary ones.

    That is, every row of the orbit named by the tables' entries at its own
    position, every term of the sum at that position, and no other sum at that
    position beside it: the terms are then the elementary band representations
    of the rows' own site irreps, and a line under the table would repeat them.
    A sum at other positions with the same content and the same stable
    real-space invariants is on the record
    (:attr:`BandRepresentation.alternatives`) and prints no line of its own.
    """
    rows = [table.orbitals[int(a)] for a in orbit.orbitals]

    def _here(terms):
        return all(str(position).split("(")[0] == orbit.wyckoff
                   for _m, _name, position in terms)

    return (bool(orbit.terms) and _here(orbit.terms)
            and not any(_here(other) for other in orbit.alternatives)
            and all(o.named_by == "tables" for o in rows))

def _orbits_sentence(orbits) -> str:
    """The line under the atomic-limit table for orbits that induce the same sums.

    One orbit reads as :meth:`BandRepresentation.sentence` writes it; several at
    one Wyckoff position, inducing the same sums, read once, each orbit's
    orbitals named.
    """
    first = orbits[0]
    if len(orbits) == 1:
        return first.sentence()
    names = [_index_ranges(o.orbitals) for o in orbits]
    who = (f"{len(orbits)} orbits at {first.wyckoff}, orbitals "
           f"{', '.join(names[:-1])} and {names[-1]}")
    if not first.terms:
        return f"{who}: {_no_sum(first)}"
    return f"{who}, each induce {_ebrs_of(first.terms)}{_not_distinguished(first)}"

def _no_sum(orbit) -> str:
    """What an orbit's line says in place of a sum: its note, which says it whole."""
    return orbit.note or "no sum of elementary band representations"

def _not_distinguished(orbit) -> str:
    """The orbit's alternatives on its line, ``", not distinguished by ... from the ... of A or of B"``, or ``""``."""
    others = orbit.alternatives
    if not others:
        return ""
    if len(others) == 1:
        joined = _ebrs_of(others[0])
    else:
        words = [_sum_words(terms) for terms in others]
        joined = ("the elementary band representations of " + ", of ".join(words[:-1])
                  + " or of " + words[-1])
    return f", not distinguished by the stable real-space invariants from {joined}"

def _content_by_position(table, states_per_orbital) -> list:
    """The orbitals at each Wyckoff position and what one site there holds, a line each.

    ``"2b (3m): orbitals 0-1, -E1"``.  The content is read at the position's
    first site, each multiplet there once, and a multiplet of several
    identical copies of one irrep counts that irrep as many times, its rows
    each naming it (:attr:`AtomicLimitRow.irrep`).  ``states_per_orbital`` is
    two on a model whose orbitals carry their own spin, where a row's
    dimension counts states and its multiplet orbitals, and one otherwise.
    """
    by_position = {}
    for row in table.orbitals:
        by_position.setdefault((row.wyckoff, row.site_symmetry), []).append(row)
    lines = []
    for (wyckoff, site_symmetry), rows in by_position.items():
        first = min(int(row.site) for row in rows)
        counts, seen = {}, set()
        for row in rows:
            if int(row.site) != first or tuple(row.multiplet) in seen:
                continue
            seen.add(tuple(row.multiplet))
            copies = max(1, len(row.multiplet) * int(states_per_orbital) // max(1, int(row.dim)))
            for n, name in _content_terms(row.irrep or "unnamed"):
                counts[name] = counts.get(name, 0) + n * copies
        content = " + ".join(f"{n}x{name}" if n > 1 else name for name, n in counts.items())
        lines.append(f"{wyckoff or '?'} ({site_symmetry}): "
                     f"{_orbitals_named(row.orbital for row in rows)}, {content}")
    return lines

# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceGroupRep:
    """The model's representation of its space group: one matrix per operation, on the model's states.

    ``r.rep`` of :func:`..groups.realized_group`, through which every band
    label is read: :func:`..sewing.hsp_table` and the calls that label bands
    take it.  ``rep.summary()`` prints what it holds.

    Attributes
    ----------
    group : ..groups.GroupData
        The group represented.
    operations : ..groups.OperationTable
        Its operations by index, each with its Seitz symbol and its name in the
        model's frame.
    indices : tuple of int
        The indices of the operations represented, in the group's order.
    matrices : dict
        ``{ind: D(g)}``, each operation's ``k``-independent matrix on the
        model's states, state ``2 × orbital + spin`` on a spinful model, up
        first: the representation induced from :attr:`characters`, in the
        tables' convention, spin along ``+z`` in a right-handed frame.
    antiunitary : tuple of int
        The operations combined with time reversal, each acting as its matrix
        times complex conjugation.
    characters : AtomicLimitTable
        What each orbital carries at its site, from which the matrices are
        induced.
    multiplets : SiteMultiplets
        The multiplets each site's orbitals form.
    spin : ..spin.SpinBasis or None
        The declared spinor basis of a spinless model carrying spin in its
        orbital index, each pair one spinor Wannier function; ``None`` for
        every other model.
    environment : str or None
        A layer's environment, ``"substrate"`` or ``"free-standing"``; ``None``
        for a crystal periodic in three directions.
    free_standing_group : dict or None
        For a layer on a substrate, the free-standing layer's group as
        :func:`..groups.realized_group` measured it beside this one and found
        it to agree, ``{"number_str", "name", "size"}``, which
        :func:`..ebr.symmetry_indicator` names beside a substrate's class;
        ``None`` otherwise.
    residuals : dict
        ``{ind: max_k ‖D H(k) D† − H(Rk)‖}``, in the units of ``H``: how far each
        matrix misses carrying the Hamiltonian onto itself.
    product_residual, factor_residual : float
        How far the matrices miss multiplying as the group does, ``D(g₁)`` times
        ``D(g₂)``, conjugated where ``g₁`` is antiunitary, against the group's
        spinor factor times ``D(g₁g₂)``; and the same for the Bloch operators,
        whose products carry the translations' phases (:func:`bloch_operator`),
        ``nan`` where not measured.
    model, assignment : pythtb.TBModel, ..groups.OrbitalAssignment
        The model, and the orbitals' Wyckoff orbits, it was built on.
    """

    model: Any
    group: Any
    assignment: Any
    indices: tuple
    # the `irrep` operation objects themselves, in index order
    _operations: tuple
    matrices: dict
    antiunitary: tuple
    residuals: dict
    # `{ind: ‖D^n − ζ·1‖}`: how far each matrix misses its canonical norm, `n` its
    # operation's order and `ζ` the sign the group's own table gives that power
    _norm_residuals: dict
    # `{ind: ‖T conj(D) T† − D‖}`: how far each unitary operation's matrix misses
    # commuting with time reversal, the relation the phases are anchored by;
    # empty where the group has no time reversal
    _anchor_residuals: dict
    product_residual: float
    factor_residual: float
    # the generators of the represented unitary operations, by index, whose
    # matrices the rest are propagated from
    _generators: tuple
    # each unitary operation, by index, as a shortest word in `_generators`, a
    # letter being a generator's place among them: the tree the matrices are
    # propagated along
    _words: dict
    # the discrete choice the build settled on: the orientation branch and the
    # relative signs, {"orientation": ±1, "signs": {(generator ind, orbit): phase}}
    _choice: dict
    # the build's record of its search for that choice: how the candidates were
    # found, read off the detector's solution space or enumerated, what each of
    # the three gates measured on them and which one each failed, the block
    # phases read off, and how far the blocks accepted were from composing by
    # the group's table ("composition", which `symmetrize` reads)
    _search: dict
    # which orbit of multiplets each multiplet belongs to, as the build indexes them
    _classes: np.ndarray
    # `{"orbital", "multiplet", "orbit"}`: the orbital the lowest-l rule reads
    # first, orbital 0 by default, the multiplet it sits in and that multiplet's
    # orbit; on a declared spinor basis also `"spin_pair"`, its `(up, dn)` pair
    _reference: dict
    # the operations the Hamiltonian breaks, one entry each as the measurement
    # refuted it: `{"ind", "refuted_by", "residual"}`
    _failed: tuple
    # the k-points the covariance was measured at, which `covariance()` reads again
    _battery: np.ndarray
    multiplets: Any
    # how each matrix's phase was fixed, operation by operation
    _normalization: dict = field(default_factory=dict, repr=False)
    # the index of the pure time reversal, `None` without one: what
    # `antiunitary_operator` returns with its matrix
    _time_reversal: int | None = None
    characters: Any = None
    spin: Any = None
    environment: str | None = None
    free_standing_group: dict | None = None
    # whether the orientation branch was a convention rather than a measurement:
    # true where g and its inverse closed every gate with the same labels, which
    # only a group of involutions allows (mm2, P-1, a spinful 2/m), and then
    # `_choice["orientation"]` is +1 by convention
    _branch_tie: bool = False

    # -- lookups -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __contains__(self, ind) -> bool:
        return int(ind) in self.matrices

    @property
    def operations(self):
        """The operations represented, by index, each with its Seitz symbol and its name in the model's frame.

        ``rep.operations[5].seitz`` is operation 5's Seitz symbol in the tables'
        setting and ``.name`` its name in the model's frame, one row per key of
        :attr:`matrices`.
        """

        return operations(self)

    @property
    def orbits(self) -> tuple:
        """The sites, grouped into the orbits the operations carry them through: ``assignment.orbits``."""
        return self.assignment.orbits

    @property
    def site_orbitals(self) -> tuple:
        """The orbitals on each site: ``assignment.site_orbitals``."""
        return self.assignment.site_orbitals

    @property
    def antiunitary_operator(self):
        """``(ind, T)``, the index and matrix of pure time reversal; ``(None, None)`` where the group has none.

        Every corep label's phase is fixed against it, and its own phase is a
        convention.
        """
        if self._time_reversal is None:
            return None, None
        return int(self._time_reversal), self.matrices[int(self._time_reversal)]

    def _operation(self, ind):
        """The ``irrep`` operation record for ``ind``."""
        for op in self._operations:
            if int(op.ind) == int(ind):
                return op
        raise KeyError(f"operation #{ind} is not part of this representation")

    def k_action(self, ind):
        """One operation's action on the model's reduced ``k``, ``k → Rk``, as a function: the image not folded back into the zone."""
        return self.group.k_action_of(self._operation(ind))

    def transform_k(self, ind, k):
        """The image of ``k`` under one operation, not folded back into the zone."""
        k = np.atleast_2d(np.asarray(k, dtype=float))
        return self.k_action(ind)(k)

    def in_little_group(self, ind, k, *, atol=1e-8):
        """``(member, G)``: whether the operation carries ``k`` to itself up to a reciprocal lattice vector, and that vector, ``G = Rk − k``.

        ``G`` is the ``k' − k_out`` :meth:`bloch_operator` folds back at ``k``.
        """
        k = np.asarray(k, dtype=float).reshape(-1)
        kp = self.transform_k(ind, k)[0]
        G = kp - k
        return bool(np.abs(G - np.rint(G)).max() <= atol), np.rint(G).astype(int)

    def little_group(self, k, *, atol=1e-8):
        """The operations represented that carry ``k`` to itself up to a reciprocal lattice vector, by index."""
        return tuple(ind for ind in self.indices if self.in_little_group(ind, k, atol=atol)[0])

    def bloch_operator(self, ind, k, *, k_out=None, **kwargs):
        """One operation's Bloch operator at ``k``, by index: :func:`bloch_operator` on this representation."""
        return bloch_operator(self.model, self.group, self.assignment,
                              self._operation(ind), self.matrices[int(ind)], k,
                              k_out=k_out, **kwargs)

    def sewing_operator(self, ind, k, *, atol=1e-8, **kwargs):
        """The Bloch operator of an operation that leaves ``k`` invariant, folded back to ``k``.

        It commutes with ``H(k)``, and between the eigenvectors of a set of
        bands it is the sewing matrix of that set.

        Refusals
        --------
        ``little_group``
            the operation does not carry ``k`` to itself up to a reciprocal
            lattice vector.
        """
        ok, G = self.in_little_group(ind, k, atol=atol)
        if not ok:
            kp = self.transform_k(ind, np.asarray(k, dtype=float).reshape(-1))[0]
            return Refusal(
                "little_group",
                f"operation #{ind} maps k = {np.round(np.asarray(k, float), 6).tolist()} to "
                f"{np.round(kp, 6).tolist()}, which is not k modulo a reciprocal-lattice "
                "vector: it has no sewing operator there.",
                value=float(np.abs(kp - np.asarray(k, float).reshape(-1)
                                   - np.rint(kp - np.asarray(k, float).reshape(-1))).max()),
                threshold=atol, context={"operation_ind": int(ind), "k": np.asarray(k, float)})
        return self.bloch_operator(ind, k, k_out=np.asarray(k, dtype=float).reshape(-1),
                                   **kwargs)

    def covariance(self, k_pts=None, *, params=None):
        """``{ind: max_k ‖D H(k) D† − H(Rk)‖}``, each operation's covariance measured again, in the units of ``H``.

        On ``k_pts``, or by default at the ``k`` the representation was built
        at; ``params`` are the values of a model whose terms are given by
        parameter names.
        """
        k_pts = self._battery if k_pts is None else np.atleast_2d(np.asarray(k_pts, float))
        params = dict(params or {})
        bloch = _bloch.evaluator(self.model, params)
        H = np.asarray(bloch(k_pts))
        out = {}
        for ind in self.indices:
            op = self._operation(ind)
            Hr = np.asarray(bloch(self.transform_k(ind, k_pts)))
            D = self.matrices[int(ind)]
            src = H.conj() if op.time_reversal else H
            out[int(ind)] = float(np.abs(D @ src @ D.conj().T - Hr).max())
        return out

    def summary(self) -> str:
        """What the representation holds, a line for each part, with the attribute that holds it."""
        g = self.group
        where = {"substrate": ", on a substrate", "free-standing": ", free-standing"}.get(
            self.environment, "")
        norb, nspin = int(self.model.norb), int(self.model.nspin)
        on = (f"the {norb * nspin} states, each orbital's spin up then down along z"
              if nspin == 2 else f"the {norb} orbital{'s' if norb != 1 else ''}")
        lines = [f"{g.name} #{g.number_str}{where}   (.group)",
                 "  its operations, by index and Seitz symbol   (.operations)",
                 f"  each one's matrix on {on}   (.matrices)"]
        if self.antiunitary:
            lines.append("  an operation with time reversal is its matrix times complex "
                         "conjugation   (.antiunitary)")
        if self.spin is not None:
            kind = {"pairs": "listed pair by pair"}.get(self.spin.kind, self.spin.kind)
            lines.append(f"  the orbitals in {len(self.spin.pairs)} declared spinor pairs, "
                         f"{kind}   (.spin)")
        if self.characters is not None:
            lines.append("  the orbitals, by Wyckoff position   (.characters)")
            lines += ["    " + line for line in _content_by_position(self.characters, nspin)]
        lines.append("  the multiplets each site's orbitals form   (.multiplets)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        worst = max(self.residuals.values()) if self.residuals else float("nan")
        n_multi = sum(1 for k in self.multiplets._chan_kind if k == "measured")
        wide = f" ({n_multi} multi-dimensional)" if n_multi else ""
        return (f"SpaceGroupRep({self.group.name} #{self.group.number_str}, "
                f"{int(self.multiplets._nchan)} multiplets{wide}, covariance<={worst:.1e}, "
                f"product={self.product_residual:.1e}"
                + (f", spinor basis ({self.spin.kind})" if self.spin is not None else "")
                + ")")
