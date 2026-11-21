r"""The model's representation of its space group, and the atomic-limit table of its orbitals.

:func:`..groups.realized_group` measures, for each operation ``g = {R|t}`` of
the cell, every matrix ``X`` with ``X H(k) = H(Rk) X`` at once on a grid of
``k``, ``X H(k)* = H(−Rk) X`` for one combined with time reversal, assuming
nothing about what ``g`` does to the orbitals: no solution is a proof that
``g`` is broken, and at the identity the solutions are the operators the
model conserves.  From the operations that hold it builds
:class:`SpaceGroupRep`, one matrix per operation on the model's states in the
tables' convention, spin along ``+z`` in a right-handed frame, through which
every band label is read; and :class:`AtomicLimitTable`, what the orbitals
carry at their sites, a row per Wyckoff orbit.

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

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import _bloch
from . import _sitechars
from . import _spacegroup
from . import _tables
from .refusal import InternalError, Refusal
from ._tolerance import ANSATZ_GATE, EXACT_TOL
from ._text import (
    _SHOWN, _content_terms, _ebrs_of, _fmt_char, _index_ranges, _orbitals_named, _sum_lines,
    _sum_words)
from ._seitz import _operation_rows
from ._matrices import (
    NormalizedRep, RepAnchor, SiteMultiplets, bloch_operator, covariance_gate,
    normalize_rep)

__all__ = ["RepAnchor",
           "NormalizedRep", "AtomicLimitTable", "AtomicLimitTableNumerical", "WyckoffOrbit",
           "WyckoffOrbitNumerical", "BandRepresentation", "DeclarationCheck",
           "SiteMultiplets", "SpaceGroupRep", "SpaceGroupRepNumerical",
           "normalize_rep",
           "bloch_operator", "ANSATZ_GATE", "EXACT_TOL", "covariance_gate"]

# --------------------------------------------------------------------------
# the atomic-limit table
# --------------------------------------------------------------------------

#: The one sentence :meth:`_OrbitalTable.summary` prints under the table
#: wherever a row's site irrep rests on the lowest-``l`` rule.
_LOWEST_L_LEGEND = (
    "the hoppings fix how the orbitals transform relative to one another, not what "
    "they are; where several site irreps fit, the one carried by the "
    "lowest-angular-momentum functions is shown, and `characters=` states another.")

#: What :attr:`_OrbitalRow.determined_by` reads on a declared row whose
#: declared irrep the Hamiltonian's own blocks do not carry.
_NOT_CARRIED = "declaration not carried"

@dataclass(frozen=True)
class _OrbitalRow:
    """One orbital's row of the atomic-limit table: its Wyckoff position, the site irrep it carries, and how that is known.

    ``rep._characters[i]`` is orbital ``i``'s row, and ``row.summary()`` prints
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
        representations at this position; ``"invariants"``, for a spinless row
        where the tables list no band representation at this position, the site
        irrep the tables of the stable real-space invariants give the same
        invariants; ``"point group"``, the site's point group at the origin of a
        symmorphic group; ``"Mulliken"``, the row's Mulliken symbol; or
        ``"unnamed"``.
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
        named = {"tables": "the tables",
                 "invariants": "the tables of the stable real-space invariants",
                 "point group": "the site's point group",
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
class _OrbitBandRepresentation:
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
    ``rep._characters.band_representations``.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals on the orbit's sites.
    wyckoff, site_symmetry : str
        The orbit's Wyckoff position, multiplicity and letter (``"6c"``), and
        its site-symmetry group (``"3m"``).
    terms : tuple
        ``(multiplicity, elementary band representation, position)`` for each
        term, the band representation and the position with its site groups
        written as the tables list them (``(1, "-E1g↑G(2)", "3a(-3m1',-3m)")``);
        empty where no sum is returned.
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
        """A sum on one line, written as the tables list it, ``"1 x -E1g↑G(2)@3a(-3m1',-3m) + 1 x -E1u↑G(2)@3a(-3m1',-3m)"``, or ``""``.

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
class _MultiplicitySpace:
    """A pseudospin the Hamiltonian conserves on an orbit of orbitals, and the site content assigned its copies.

    Where the Hamiltonian holds identical copies of a part of the crystal that
    nothing joins, every band there is as many times degenerate as there are
    copies, and nothing the hoppings measure says what the copies carry, which
    moves the site content itself.  A declaration on their orbitals states it,
    and the lowest-angular-momentum rule assigns it otherwise, the copies read
    copy by copy first.  ``space.summary()`` prints it; it is one entry of
    ``rep._characters.multiplicity_spaces``.

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
class _OrbitalTable:
    """What each orbital carries at its site: the table an atomic limit is written up in, a row per orbital.

    ``rep._characters`` is the table; ``table.summary()`` prints it, and
    ``table[i]`` is orbital ``i``'s :class:`_OrbitalRow`.  The table is where
    a mis-oriented declaration shows, in each row's orbital characters.

    Attributes
    ----------
    orbitals : tuple of _OrbitalRow
        One row per orbital, in the model's order.
    irreps : tuple of str
        Each orbital's site irrep, in the model's order.
    band_representations : tuple of _OrbitBandRepresentation
        The band representation each orbit of orbitals induces, a sum of
        elementary band representations, in the order of the orbits; empty
        where the tables do not reach the cell.
    multiplicity_spaces : tuple of _MultiplicitySpace
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

    def __getitem__(self, orbital) -> _OrbitalRow:
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
            ups = {int(up) for up, _dn in self.spin.pairs}
            spins = ["up" if int(o.orbital) in ups else "down" for o in self.orbitals]
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
        return (f"_OrbitalTable({'/'.join(self.irreps)}"
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
    (:attr:`_OrbitBandRepresentation.alternatives`) and prints no line of its own.
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

    One orbit reads as :meth:`_OrbitBandRepresentation.sentence` writes it; several at
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
    each naming it (:attr:`_OrbitalRow.irrep`).  ``states_per_orbital`` is
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
# a sum of site irreps, and the rows by Wyckoff orbit
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BandRepresentation:
    """A sum of site irreps at Wyckoff positions: the band representation they induce, with its printed form.

    Iterating, indexing and ``len()`` go over :attr:`terms`.  Two sums are
    equal where their terms are, whatever printed them.

    Attributes
    ----------
    terms : tuple of ..groups.Term
        The terms, each a whole number of copies of a site irrep at a Wyckoff
        position; empty where no sum carries the content.
    label : str
        The sum as printed, ``"-E1 at 1b (4mm1') + 2x-E2 at 1b (4mm1')"``;
        ``""`` where there are no terms.
    """

    terms: tuple = ()
    label: str = field(default="", compare=False)

    def __iter__(self):
        return iter(self.terms)

    def __getitem__(self, index):
        return self.terms[index]

    def __len__(self) -> int:
        return len(self.terms)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BandRepresentation):
            return NotImplemented
        return tuple(self.terms) == tuple(other.terms)

    def __hash__(self) -> int:
        return hash(tuple(self.terms))

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{type(self).__name__}({self.label})"


def _words_label(terms) -> str:
    """A sum of terms in words with each site-symmetry group, ``-E1 at 1b (4mm1') + 2x-E2 at 1b (4mm1')``."""
    return " + ".join(
        ("" if int(t.multiplicity) == 1 else f"{int(t.multiplicity)}x")
        + f"{t.irrep} at {t.wyckoff} ({t.site_symmetry})" for t in terms)


def _paper_label(terms) -> str:
    """A sum of terms as the tables of the stable real-space invariants print it, ``1 (-E2)_b + 1 (-E)_c``."""
    return " + ".join(f"{int(t.multiplicity)} ({t.irrep})_{t.wyckoff.lstrip('0123456789')}"
                      for t in terms)


def _irreps_label(terms) -> str:
    """The site irreps of one site, as a row prints them: ``-E1``, ``2xA1``, ``-E1u + -E3u``."""
    return " + ".join(("" if int(t.multiplicity) == 1 else f"{int(t.multiplicity)}x") + t.irrep
                      for t in terms)


def _sum_of(terms) -> BandRepresentation:
    """A sum of terms with its label in words."""
    terms = tuple(terms)
    return BandRepresentation(terms=terms, label=_words_label(terms))


@dataclass(frozen=True)
class DeclarationCheck:
    """A declaration held against one operation of its site group: what the hoppings measure there and what was declared.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals declared.
    index : int
        The operation, by index.
    name : str
        Its name in the model's frame.
    measured, declared : complex or None
        The character the hoppings measure for it, read against one multiplet
        of the declaration's own part of the crystal, and the character
        declared.
    """

    orbitals: tuple
    index: int
    name: str
    measured: Any = None
    declared: Any = None


def _declaration_checks(rows) -> tuple:
    """The measured-versus-declared rows, each as a :class:`DeclarationCheck`."""
    return tuple(DeclarationCheck(orbitals=tuple(int(o) for o in r["orbitals"]),
                                  index=int(r["ind"]), name=str(r["name"]),
                                  measured=r.get("measured"), declared=r.get("declared"))
                 for r in rows)


@dataclass(frozen=True)
class WyckoffOrbitNumerical:
    """How a Wyckoff orbit's row was read: its sites, each orbital's characters, and how a declaration was resolved.

    ``o.numerical`` of a row of ``r.wyckoff_orbits``.

    Attributes
    ----------
    positions : tuple
        Each site's position in reduced coordinates, in the order of
        ``o.site_multiplets``.
    residual : float
        How far an orbital's block misses its character times its spin factor,
        or a multiplet misses multiplying as the site group does, the largest
        over the row's orbitals: a large one says the site irreps are not to be
        trusted.
    characters : dict
        For each orbital, its orbital factor's character on every operation of
        its site group, by index, single-valued with spin or without.
    named_by : str
        Where the row's site irreps take their names: ``"tables"``, the
        elementary band representations at this position; ``"invariants"``,
        the tables of the stable real-space invariants; ``"point group"``, the
        site's point group at the origin of a symmorphic group;
        ``"Mulliken"``, the Mulliken symbol; or ``"unnamed"``.
    declared_irrep, declared_in, reduced_in, declaration_broke : dict
        For each declared orbital: the site irreps its declaration resolved
        to; the site group and group it was read in (``"3m in P3m11'
        #156.50"``); where the declared content lowered the cell first, the
        site group and group it was resolved in; and the operations the
        declared content breaks.  Empty where nothing was declared.
    determined_by : dict
        For each orbital, how its site irrep is known, as the row's
        ``determined_by`` reads it.
    alternatives : tuple of BandRepresentation
        The other sums with the same symmetry content and the same stable
        real-space invariants, which nothing measured tells from the row's
        own.
    contents : tuple of BandRepresentation
        Where the Hamiltonian conserves a pseudospin on the row, every set of
        site irreps the representations of its copies the group's own table
        admits give, in the order the lowest-angular-momentum rule ranks them;
        empty otherwise.
    n_representations : int
        How many representations of the copies the group's own table admits;
        ``1`` where no pseudospin is conserved.
    """

    positions: tuple
    residual: float
    characters: dict
    named_by: str
    declared_irrep: dict
    declared_in: dict
    reduced_in: dict
    declaration_broke: dict
    determined_by: dict
    alternatives: tuple = ()
    contents: tuple = ()
    n_representations: int = 1

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"WyckoffOrbitNumerical(residual={self.residual:.1e}, named by "
                f"{self.named_by})")


@dataclass(frozen=True)
class WyckoffOrbit:
    """One Wyckoff orbit of the model's orbitals: the orbitals the operations carry into one another, and the site irreps they carry.

    A row of ``r.wyckoff_orbits``; ``o.summary()`` prints it.  Its site irreps
    are the orbit's, one name for the whole orbit: two of its sites that an
    operation exchanging two site irreps carries into one another can carry
    conjugate ones, ``B1`` at ``(1/2, 0)`` and ``B2`` at ``(0, 1/2)``, and the
    row names the orbit as the tables of elementary band representations do.

    Attributes
    ----------
    wyckoff : str
        The Wyckoff position, multiplicity and letter in the tables' setting,
        ``"1a"``; ``""`` where the tables do not reach the cell.
    site_symmetry : str
        The site-symmetry group, with its ``1'`` where the group returned has
        time reversal, ``"4mm1'"``; ``"3"`` in ``P3``.
    orbitals : tuple of int
        The orbitals of the orbit.
    label : str
        The site irreps one of its sites holds, as printed: ``"-E1"``,
        ``"2xA1"``, ``"-E1u + -E3u"``.
    terms : tuple of ..groups.Term
        The same site irreps as terms, ``Term(1, "-E1", "1a", "4mm1'",
        "4mm")``.
    unitary_site_symmetry : str
        The unitary part of the site-symmetry group, ``"4mm"``.
    determined_by : str
        How the site irreps are known, the weakest ground among the orbitals:
        ``"declaration not carried"``, declared and not carried by the
        Hamiltonian's own blocks; ``"lowest-l"``, chosen by the
        lowest-angular-momentum rule, which ``characters=`` overrides;
        ``"declaration"``; or ``"hamiltonian"``, the same under every labeling
        the declarations leave open.
    declared : str, tuple or None
        What ``characters=`` declared for the row: one word where every
        orbital was declared the same, otherwise one entry per orbital,
        ``None`` for an orbital not declared; ``None`` where nothing was.
    band_representation : BandRepresentation
        The sum of elementary band representations the orbit induces, written
        at a position its orbitals can be moved to along their variety, their
        own first; empty where none carries the content.
    site_multiplets : tuple of tuple of int
        The orbitals the Hamiltonian binds together at each of the orbit's
        sites, ``((2,), (3,))``.
    numerical : WyckoffOrbitNumerical
        How the row was read.
    """

    wyckoff: str
    site_symmetry: str
    orbitals: tuple
    label: str
    terms: tuple
    unitary_site_symmetry: str
    determined_by: str
    declared: Any
    band_representation: BandRepresentation
    site_multiplets: tuple
    numerical: WyckoffOrbitNumerical
    # the orbit's sites, by index, in the order of the orbit of sites
    _sites: tuple = field(default=(), repr=False, compare=False)

    def _line(self) -> str:
        """The row in one line, as ``r.summary()`` prints it: ``1a (4mm1'): orbital 0, -E1``."""
        return (f"{self.wyckoff or '?'} ({self.site_symmetry}): "
                f"{_orbitals_named(self.orbitals)}, {self.label or 'unnamed'}")

    def summary(self) -> str:
        """The row, a line for each part that applies, ending with the attribute that holds it."""
        lines = [self._line() + "   (.wyckoff, .site_symmetry, .orbitals, .label)"]
        if self.declared is not None:
            lines.append(f"  declared {self.declared!r}"
                         + (", and the Hamiltonian's own blocks do not carry it"
                            if self.determined_by == _NOT_CARRIED else "")
                         + "   (.declared)")
        if self.determined_by not in ("declaration", _NOT_CARRIED):
            lines.append("  " + ("fixed by the Hamiltonian"
                                 if self.determined_by == "hamiltonian" else
                                 "chosen by the lowest-angular-momentum rule among the "
                                 "site irreps the hoppings allow")
                         + "   (.determined_by)")
        if self.band_representation.terms:
            lines.append(f"  it induces {self.band_representation.label}"
                         "   (.band_representation)")
        else:
            lines.append("  it induces no sum of elementary band representations that is "
                         "read here   (.band_representation)")
        lines.append("  bound together at each site: "
                     + "; ".join(_orbitals_named(m) for m in self.site_multiplets)
                     + "   (.site_multiplets)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{type(self).__name__}({self._line()})"


@dataclass(frozen=True)
class AtomicLimitTableNumerical:
    """How the atomic-limit table was read: each declaration against the hoppings, and the multiplets.

    ``r.wyckoff_orbits.numerical``.

    Attributes
    ----------
    checks : tuple of DeclarationCheck
        Each declaration against each operation of its site group the
        hoppings say something about: the table a ``declaration_breaks_group``
        refusal prints.  What is measured is read against one multiplet of the
        declaration's own part of the crystal, so the two differ by a
        one-dimensional character of that part.
    residual : float
        The largest residual of any row.
    site_multiplets_determined_by : str
        How the multiplets each site's orbitals form were found:
        ``"hamiltonian"``, measured; ``"declaration"``, stated.
    accidental : tuple
        The multiplets the Hamiltonian binds by an accident of its numbers
        rather than by the site group.
    """

    checks: tuple
    residual: float
    site_multiplets_determined_by: str
    accidental: tuple = ()

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"AtomicLimitTableNumerical({len(self.checks)} checks, "
                f"residual={self.residual:.1e})")


@dataclass(frozen=True)
class AtomicLimitTable:
    """What the model's orbitals carry at their sites, a row per Wyckoff orbit: the table an atomic limit is written up in.

    ``r.wyckoff_orbits``; ``table.summary()`` prints it.  ``table[j]`` is row
    ``j``, a :class:`WyckoffOrbit`, in the order the orbits' first orbitals
    come in the model, and iterating goes over the rows.

    Attributes
    ----------
    numerical : AtomicLimitTableNumerical
        How the table was read.
    """

    _rows: tuple
    numerical: AtomicLimitTableNumerical
    # the per-orbital table the rows are listed from, which the lines under the
    # table read
    _orbital_table: Any = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index) -> WyckoffOrbit:
        return self._rows[index]

    def __iter__(self):
        return iter(self._rows)

    def summary(self) -> str:
        """The table, a row per Wyckoff orbit, and a line under it for each thing the rows do not say.

        The columns are the rows' fields.  Under the rows, each line ending with
        the attribute that holds it: a row whose band representation is not
        its own site irreps at its own position, with the sums nothing measured
        tells from it; a conserved pseudospin, with the site irreps assigned
        its copies; a declaration the Hamiltonian's own blocks do not carry or
        written in a name Mulliken's convention gives another row; and where a
        row reads ``lowest-l``, what that means.
        """
        head = ["Wyckoff", "site symmetry", "orbitals", "label", "determined by", "declared"]
        body = [[o.wyckoff or "?", o.site_symmetry, _index_ranges(o.orbitals),
                 o.label or "unnamed", o.determined_by, _declared_text(o.declared)]
                for o in self._rows]
        width = [max([len(h)] + [len(r[j]) for r in body]) for j, h in enumerate(head)]
        lines = ["  ".join(f"{h:>{w}s}" for h, w in zip(head, width)).rstrip()]
        lines += ["  ".join(f"{c:>{w}s}" for c, w in zip(cells, width)).rstrip()
                  for cells in body]
        table = self._orbital_table
        for o, orbit in zip(self._rows, getattr(table, "band_representations", ()) or ()):
            if not orbit.terms:
                lines.append(f"  {o.wyckoff or '?'}, {_orbitals_named(o.orbitals)}: "
                             f"{_no_sum(orbit)}   (.band_representation)")
                continue
            if _its_rows_own(table, orbit):
                continue
            lines.append(f"  {o.wyckoff or '?'}, {_orbitals_named(o.orbitals)}: they induce "
                         f"{o.band_representation.label}   (.band_representation)")
            others = [f"from {other.label}" for other in o.numerical.alternatives]
            if others:
                more = len(others) - _SHOWN
                said = (", ".join(others[:-1]) + " or " + others[-1] if len(others) > 1
                        else others[0]) if more <= 0 else (
                    ", ".join(others[:_SHOWN]) + f" or from {more} more sums")
                lines.append(f"    not told apart {said} by any stable real-space invariant"
                             "   (.numerical.alternatives)")
        for space in getattr(table, "multiplicity_spaces", ()) or ():
            lines.append("  " + _pseudospin_sentence(space) + "   (.label, .numerical.contents)")
        lines += _declaration_lines(table)
        if any(o.determined_by == "lowest-l" for o in self._rows):
            lines.append("  lowest-l: " + _LOWEST_L_LEGEND)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"AtomicLimitTable({'/'.join(o.label or 'unnamed' for o in self._rows)}"
                + (", lowest-l" if any(o.determined_by == "lowest-l"
                                       for o in self._rows) else "") + ")")


def _declared_text(declared) -> str:
    """A row's ``declared`` in the table's column: the word, or one entry per orbital."""
    if declared is None:
        return ""
    if isinstance(declared, tuple):
        return "/".join("-" if d is None else str(d) for d in declared)
    return str(declared)


def _pseudospin_sentence(space) -> str:
    """The line under the table for a pseudospin the Hamiltonian conserves."""
    head = (f"{_orbitals_named(space.orbitals)}"
            + (f" at {space.wyckoff}" if space.wyckoff else "")
            + f" carry a {_fold(space.dim)} pseudospin the Hamiltonian conserves")
    content = space.content or "(unnamed)"
    n = len(space._distinct())
    if space.determined_by == "declaration":
        return (f"{head}, and what its copies carry it does not fix: the declaration "
                f"states the site irreps {content}, of the {n} sets the group's own table "
                "admits")
    if n > 1:
        return (f"{head}, and what its copies carry it does not fix: the "
                f"lowest-angular-momentum rule assigns the site irreps {content}, of the "
                f"{n} sets the group's own table admits")
    if int(space.n_representations) > 1:
        return (f"{head}; the {int(space.n_representations)} representations of its copies "
                f"the group's own table admits all give the site irreps {content}, and the "
                "lowest-angular-momentum rule chose the one returned among them")
    return (f"{head}; the group's own table admits one representation of its copies, "
            f"giving the site irreps {content}")


def _declaration_lines(table) -> list:
    """The lines under the table for a declaration not carried, or written in a name Mulliken's convention gives another row."""
    if table is None:
        return []
    lines = []
    for d in table.declarations:
        orbs = tuple(int(a) for a in d["orbitals"])
        row = table.orbitals[orbs[0]]
        dropped = row.determined_by == _NOT_CARRIED
        if not (dropped or d.get("mulliken")):
            continue
        who = _orbitals_named(orbs)
        said = f"{d['declared_as']!r}" + (
            f" ({d['declared_irrep']})" if d["declared_irrep"]
            and d["declared_irrep"] != d["declared_as"] else "")
        parts = [f"  {who}: declared {said}, read at the {d['declared_in']}"]
        if d.get("mulliken"):
            parts.append(_sitechars.mulliken_clause(d["mulliken"].items()))
        if dropped:
            parts.append("the Hamiltonian's own blocks do not carry it under the "
                         "one-dimensional character returned")
        parts.append(f"returned as {d.get('returned', row.irrep) or '(unnamed)'} of "
                     + (row.site_symmetry
                        or f"the site group of order {len(row.site_group)}"))
        lines.append("; ".join(parts) + "   (.declared)")
    return lines


#: The order a row's ``determined_by`` reads the grounds its orbitals rest on,
#: the weakest first.
_GROUNDS = (_NOT_CARRIED, "lowest-l", "declaration", "hamiltonian")


def _wyckoff_orbits_of(rep) -> tuple:
    """The Wyckoff orbits of a representation's orbitals: ``(sites, orbitals)`` for each.

    A Wyckoff orbit is the orbit of one multiplet under the group, read off the
    build's orbits of multiplets; its sites are kept in the order of their
    orbit of sites, and the orbits in the order their first orbitals come in
    the model.
    """
    layout, assignment = rep._multiplets, rep._assignment
    classes = np.asarray(rep._classes, dtype=int)
    orbit_of_site = np.asarray(assignment.orbit_of_site, dtype=int)
    out = []
    for c in sorted(set(int(x) for x in classes.tolist())):
        chans = [k for k in range(int(layout._nchan)) if int(classes[k]) == c]
        orbitals = tuple(sorted(int(o) for k in chans for o in layout._chan_orbs[k]))
        sites = {int(layout._chan_site[k]) for k in chans}
        coarse = {int(orbit_of_site[s]) for s in sites}
        if len(coarse) != 1:
            raise InternalError(
                f"the orbit of multiplets {c} spans the orbits of sites {sorted(coarse)}: "
                "an orbit of one multiplet under the group lies in one orbit of sites, and "
                "the two disagreeing is a defect in this package.")
        ordered = tuple(int(s) for s in assignment.orbits[coarse.pop()] if int(s) in sites)
        out.append((ordered, orbitals))
    return tuple(sorted(out, key=lambda w: min(w[1])))


def _terms_of(old_terms) -> tuple:
    """A sum written ``(multiplicity, elementary band representation, position)`` as terms."""
    return tuple(replace(_tables._ebr_term(name, position), multiplicity=int(m))
                 for m, name, position in old_terms)


def _wyckoff_orbit_table(rep) -> AtomicLimitTable:
    """``r.wyckoff_orbits``: the per-orbital table's rows gathered by Wyckoff orbit.

    Each row's site irreps are the orbit's, read where the stable real-space
    invariants read them: the terms of its band representation where that is
    written at its own position, every term there and no other sum there; else
    the site irreps the tables of the invariants give a composite band
    representation; else the names its orbitals take at the site the carrying
    to the position's own site leaves as it is.
    """
    from . import srsi as _srsi
    from . import _site_names

    table, group, assignment = rep._characters, rep._group, rep._assignment
    layout = rep._multiplets
    if table is None:
        return AtomicLimitTable(_rows=(), numerical=AtomicLimitTableNumerical(
            checks=(), residual=0.0,
            site_multiplets_determined_by=str(getattr(layout, "determined_by", "")),
            accidental=tuple(getattr(layout, "accidental", ()) or ())))
    grey = getattr(group, "magnetic_type", "I") == "II"
    orbits = _wyckoff_orbits_of(rep)
    sums = tuple(table.band_representations) or (None,) * len(orbits)
    got = _srsi.srsi_tables(group)
    tables = None if isinstance(got, Refusal) else got
    readings = _tables._sites_in_tables_setting(group, assignment.site_positions)
    readings = None if isinstance(readings, Refusal) else readings
    # the site-symmetry groups the list of elementary band representations
    # writes, by position, with every bar; a position it does not name takes the
    # name read off the operations, with its 1' in a group with time reversal
    data = _tables._ebr_data(group)
    listed = {} if isinstance(data, Refusal) else {
        _tables._wyckoff_letter(position): (term.site_symmetry, term.unitary_site_symmetry)
        for name, position in data["ebrs"]
        for term in (_tables._ebr_term(name, position),)}
    spaces = tuple(getattr(table, "multiplicity_spaces", ()) or ())
    positions = np.asarray(assignment.site_positions, dtype=float)
    rows = []
    for (sites, orbitals), orbit in zip(orbits, sums):
        per = [table.orbitals[int(a)] for a in orbitals]
        first = per[0]
        wyckoff = first.wyckoff
        unitary = first.site_symmetry
        site_symmetry = unitary + ("1'" if grey and unitary else "")
        if wyckoff in listed:
            site_symmetry, unitary = listed[wyckoff]

        def made(m, irrep):
            return _spacegroup.Term(multiplicity=int(m), irrep=str(irrep), wyckoff=wyckoff,
                                    site_symmetry=site_symmetry,
                                    unitary_site_symmetry=unitary)

        own = ()
        alternatives = ()
        if orbit is not None and orbit.terms:
            own = _terms_of(orbit.terms)
            alternatives = tuple(_sum_of(_terms_of(terms)) for terms in orbit.alternatives)
        induced = _sum_of(own)
        named_by = None
        # 1. the band representation written at the row's own position
        here = bool(own) and all(t.wyckoff == wyckoff for t in own) and not any(
            all(t.wyckoff == wyckoff for t in other.terms) for other in alternatives)
        if here:
            terms = tuple(made(t.multiplicity, t.irrep) for t in own)
            named_by = "tables"
        else:
            terms = None
            # 2. a composite band representation's site irreps
            if tables is not None and readings is not None and wyckoff:
                composite = _srsi._composite_localization(
                    tables, group, assignment, rep, sites, readings[int(sites[0])],
                    orbitals=orbitals)
                if composite is not None:
                    terms = tuple(made(m, irrep)
                                  for irrep, m in composite["irreps"].items())
            # 3. the names the orbitals take at the site the carrying leaves as it
            # is; where the tables do not reach the cell there is no position's
            # own site, and the orbit's sites are read as listed
            at = sites[0]
            letter = (wyckoff.lstrip("0123456789") if wyckoff else "")
            if letter:
                at = None
                for s in sites:
                    inds = tuple(int(i) for i in assignment.site_group[int(s)])
                    carried = _site_names.carry_to_own_site(group, letter,
                                                            positions[int(s)], inds)
                    if carried is not None and tuple(carried[1]) == inds:
                        at = s
                        break
                if at is None:
                    raise InternalError(
                        f"no site of the orbit of orbitals {list(orbitals)} at {wyckoff} is "
                        "carried to the position's own site with its operations unchanged: "
                        "one of an orbit's sites lies on the stretch of the position's own "
                        "site, and none being found is a defect in this package.")
            mine = [o for o in table.orbitals if int(o.site) == int(at)
                    and int(o.orbital) in set(orbitals)]
            named_by = mine[0].named_by if mine else first.named_by
            if terms is None:
                counts, seen = {}, set()
                for o in mine:
                    if tuple(o.multiplet) in seen:
                        continue
                    seen.add(tuple(o.multiplet))
                    states = 2 if int(rep._model.nspin) == 2 else 1
                    copies = max(1, len(o.multiplet) * states // max(1, int(o.dim)))
                    for n, name in _content_terms(o.irrep or ""):
                        if name:
                            counts[name] = counts.get(name, 0) + n * copies
                terms = tuple(made(n, name) for name, n in counts.items())
        # what a conserved pseudospin leaves open on the row
        space = next((sp for sp in spaces
                      if set(int(o) for o in sp.orbitals) & set(orbitals)), None)
        contents = ()
        n_representations = 1
        if space is not None:
            contents = tuple(
                _sum_of(made(n, name) for n, name in _content_terms(content) if name)
                for content in space._distinct())
            n_representations = int(space.n_representations)
        words = [(None if o.declared_as is None else o.declared_as) for o in per]
        declared_any = [w for w in words if w is not None]
        if not declared_any:
            declared = None
        elif len(declared_any) == len(words) and all(w == words[0] for w in words):
            declared = words[0]
        else:
            declared = tuple(words)
        grounds = [str(o.determined_by) for o in per]
        determined = next((g for g in _GROUNDS if g in grounds), grounds[0])
        decl = [o for o in per if o.declared_as is not None]
        multiplets = []
        for s in sites:
            here_m = sorted({tuple(int(x) for x in o.multiplet) for o in per
                             if int(o.site) == int(s)})
            multiplets.append(tuple(x for m in here_m for x in m))
        rows.append(WyckoffOrbit(
            wyckoff=wyckoff, site_symmetry=site_symmetry, orbitals=tuple(orbitals),
            label=_irreps_label(terms), terms=tuple(terms),
            unitary_site_symmetry=unitary, determined_by=determined, declared=declared,
            band_representation=induced, site_multiplets=tuple(multiplets),
            numerical=WyckoffOrbitNumerical(
                positions=tuple(tuple(float(x) for x in positions[int(s)]) for s in sites),
                residual=float(max(float(o.residual) for o in per)),
                characters={int(o.orbital): dict(o.character) for o in per},
                named_by=str(named_by),
                declared_irrep={int(o.orbital): o.declared_irrep for o in decl},
                declared_in={int(o.orbital): o.declared_in for o in decl},
                reduced_in={int(o.orbital): o.reduced_in for o in decl},
                declaration_broke={int(o.orbital): tuple(o.declaration_broke) for o in decl},
                determined_by={int(o.orbital): str(o.determined_by) for o in per},
                alternatives=alternatives, contents=contents,
                n_representations=n_representations),
            _sites=tuple(int(s) for s in sites)))
    return AtomicLimitTable(
        _rows=tuple(rows),
        numerical=AtomicLimitTableNumerical(
            checks=_declaration_checks(table.checks), residual=float(table.residual),
            site_multiplets_determined_by=str(getattr(layout, "determined_by", "")),
            accidental=tuple(getattr(layout, "accidental", ()) or ())),
        _orbital_table=table)


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceGroupRepNumerical:
    """How closely the representation's matrices carry the Hamiltonian and multiply as the group does.

    ``r.rep.numerical``; ``summary()`` prints it.

    Attributes
    ----------
    residuals : dict
        ``{index: max_k ‖D H(k) D† − H(Rk)‖}``, in the units of ``H``: how far
        each matrix misses carrying the Hamiltonian onto itself.
    product_residual : float
        How far the matrices miss multiplying as the group does, ``D(g₁)``
        times ``D(g₂)``, conjugated where ``g₁`` is antiunitary, against the
        group's spinor factor times ``D(g₁g₂)``.
    factor_residual : float
        The same for the Bloch operators, whose products carry the
        translations' phases (:func:`bloch_operator`); ``nan`` where not
        measured.
    """

    residuals: dict
    product_residual: float
    factor_residual: float

    def summary(self) -> str:
        """The residuals, a line each, with the attribute that holds them."""
        worst = max(self.residuals.values()) if self.residuals else float("nan")
        return "\n".join([
            f"each matrix's covariance residual at most {worst:.2g}, in the units of H"
            "   (.residuals)",
            f"  the matrices multiply as the group does to {self.product_residual:.2g}, "
            f"the Bloch operators to {self.factor_residual:.2g}"
            "   (.product_residual, .factor_residual)"])

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        worst = max(self.residuals.values()) if self.residuals else float("nan")
        return (f"SpaceGroupRepNumerical(covariance<={worst:.1e}, "
                f"product={self.product_residual:.1e}, factor={self.factor_residual:.1e})")


@dataclass(frozen=True)
class SpaceGroupRep:
    """The model's representation of its space group: one matrix per operation, on the model's states.

    ``r.rep`` of :func:`..groups.realized_group`.  The calls that label bands
    read it through what ``realized_group`` returns, ``hsp_table(r,
    bands=n)``; ``rep.summary()`` prints what it holds, and ``rep.numerical``
    how closely its matrices hold.

    Attributes
    ----------
    matrices : dict
        ``{index: D(g)}``, each operation's ``k``-independent matrix on the
        model's states, state ``2 × orbital + spin`` on a spinful model, up
        first: the representation induced from the site irreps of
        ``r.wyckoff_orbits``, in the tables' convention, spin along ``+z`` in a
        right-handed frame.  Pure time reversal's matrix is
        ``matrices[operation_index(r.group, "{1|0,0,0}'")]``.
    antiunitary : tuple of int
        The operations combined with time reversal, each acting as its matrix
        times complex conjugation.
    numerical : SpaceGroupRepNumerical
        How far each matrix misses carrying the Hamiltonian onto itself, and
        how far the matrices and the Bloch operators miss multiplying as the
        group does.
    """

    # the model, the group and the orbitals' Wyckoff orbits the representation
    # was built on, which every method computes with
    _model: Any
    _group: Any
    _assignment: Any
    # the indices of the operations represented, in the group's order
    _indices: tuple
    # the `irrep` operation objects themselves, in index order
    _operations: tuple
    matrices: dict
    antiunitary: tuple
    # `{ind: ‖D^n − ζ·1‖}`: how far each matrix misses its canonical norm, `n` its
    # operation's order and `ζ` the sign the group's own table gives that power
    _norm_residuals: dict
    # `{ind: ‖T conj(D) T† − D‖}`: how far each unitary operation's matrix misses
    # commuting with time reversal, the relation the phases are anchored by;
    # empty where the group has no time reversal
    _anchor_residuals: dict
    numerical: SpaceGroupRepNumerical
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
    # the group's table ("composition", which the projection reads)
    _search: dict
    # which orbit of multiplets each multiplet belongs to, as the build indexes them
    _classes: np.ndarray
    # `{"orbital", "multiplet", "orbit"}`: the orbital the lowest-l rule reads
    # first, orbital 0 by default, the multiplet it sits in and that multiplet's
    # orbit; on a declared spinor basis also `"spin_pair"`, its `(up, dn)` pair
    _reference: dict
    # the operations the Hamiltonian breaks, one entry each as the measurement
    # refuted it
    _failed: tuple
    # the k-points the covariance was measured at, which `covariance()` reads again
    _battery: np.ndarray
    # the multiplets each site's orbitals form
    _multiplets: Any
    # how each matrix's phase was fixed, operation by operation
    _normalization: dict = field(default_factory=dict, repr=False)
    # the index of the pure time reversal, `None` without one
    _time_reversal: int | None = None
    # what each orbital carries at its site, a row per orbital, from which the
    # matrices are induced and the rows of `r.wyckoff_orbits` are listed
    _characters: Any = None
    # the declared spinor basis, read back as `r.spin`
    _spin: Any = None
    # a layer's environment, "substrate" or "free-standing"; None for a crystal
    # periodic in three directions
    _environment: str | None = None
    # for a layer on a substrate, the free-standing layer's group as the
    # cross-check measured it and found it to agree, `{"number_str", "name",
    # "size"}`, which `ebr.symmetry_indicator` names beside a substrate's class
    _free_standing_group: dict | None = None
    # whether the orientation branch was a convention rather than a measurement:
    # true where g and its inverse closed every gate with the same labels, which
    # only a group of involutions allows (mm2, P-1, a spinful 2/m), and then
    # `_choice["orientation"]` is +1 by convention
    _branch_tie: bool = False
    # the content hash of `model` when the representation was measured on it
    # (`_bloch.model_key`), against which the calls that label bands check that
    # the model is unedited; `None` where nothing is checked: the decoupled stack
    # a layer's indicator is completed on
    _model_key: str | None = field(default=None, repr=False, compare=False)

    # -- lookups -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.matrices)

    def __contains__(self, ind) -> bool:
        return int(ind) in self.matrices

    def _operation(self, ind):
        """The ``irrep`` operation record for ``ind``."""
        for op in self._operations:
            if int(op.ind) == int(ind):
                return op
        raise KeyError(f"operation #{ind} is not part of this representation")

    def _check_unedited(self, bloch) -> None:
        """Raise ``ValueError`` where ``model`` has been edited since it was measured.

        ``bloch`` is the model's Bloch sum, which carries the hash of the
        content it was summed from (:func:`.._bloch.evaluator`).
        """
        if self._model_key is not None and bloch.key != self._model_key:
            raise ValueError(
                "the model has been edited since realized_group measured it, and this "
                "representation describes the model as it was: call realized_group on the "
                "edited model and label its bands through what it returns.")

    def _kpoint(self, call, k):
        """``k`` as coordinates, a tabulated name read as the coordinates :func:`..groups.high_symmetry_points` gives it, or the refusal ``out_of_plane`` for a name out of a layer's plane."""
        return _tables._resolve_kpoint(call, self._group, k)

    def transform_k(self, ind, k):
        """The image of ``k``, a tabulated name or reduced coordinates, under one operation: ``Rk``, not folded back into the zone."""
        k = self._kpoint("transform_k", k)
        if isinstance(k, Refusal):
            return k
        k = np.atleast_2d(np.asarray(k, dtype=float))
        return self._group._k_action_of(self._operation(ind))(k)

    def in_little_group(self, ind, k, *, _atol=1e-8):
        """``(member, G)``: whether the operation carries ``k`` to itself up to a reciprocal lattice vector, and that vector, ``G = Rk − k``.

        ``k`` is a tabulated name or reduced coordinates.  ``G`` is the
        ``k' − k_out`` :meth:`bloch_operator` folds back at ``k``.
        """
        k = self._kpoint("in_little_group", k)
        if isinstance(k, Refusal):
            return k
        k = np.asarray(k, dtype=float).reshape(-1)
        kp = self.transform_k(ind, k)[0]
        G = kp - k
        return bool(np.abs(G - np.rint(G)).max() <= _atol), np.rint(G).astype(int)

    def little_group(self, k, *, _atol=1e-8):
        """The operations represented that carry ``k``, a tabulated name or reduced coordinates, to itself up to a reciprocal lattice vector, by index."""
        k = self._kpoint("little_group", k)
        if isinstance(k, Refusal):
            return k
        return tuple(ind for ind in self._indices
                     if self.in_little_group(ind, k, _atol=_atol)[0])

    def bloch_operator(self, ind, k, *, k_out=None, **kwargs):
        """One operation's Bloch operator at ``k``, by index: :func:`bloch_operator` on this representation.

        ``k`` and ``k_out`` are tabulated names or reduced coordinates.
        """
        k = self._kpoint("bloch_operator", k)
        if isinstance(k, Refusal):
            return k
        if k_out is not None:
            k_out = self._kpoint("bloch_operator", k_out)
            if isinstance(k_out, Refusal):
                return k_out
        return bloch_operator(self._model, self._group, self._assignment,
                              self._operation(ind), self.matrices[int(ind)], k,
                              k_out=k_out, **kwargs)

    def sewing_operator(self, ind, k, *, _atol=1e-8, **kwargs):
        """The Bloch operator of an operation that leaves ``k`` invariant, folded back to ``k``.

        ``k`` is a tabulated name or reduced coordinates.  The operator
        commutes with ``H(k)``, and between the eigenvectors of a set of bands
        it is the sewing matrix of that set.

        Refusals
        --------
        ``little_group``
            the operation does not carry ``k`` to itself up to a reciprocal
            lattice vector.
        ``out_of_plane``
            ``k`` names a tabulated point out of a layer's plane.
        """
        k = self._kpoint("sewing_operator", k)
        if isinstance(k, Refusal):
            return k
        ok, G = self.in_little_group(ind, k, _atol=_atol)
        if not ok:
            kp = self.transform_k(ind, np.asarray(k, dtype=float).reshape(-1))[0]
            return Refusal(
                "little_group",
                f"operation #{ind} maps k = {np.round(np.asarray(k, float), 6).tolist()} to "
                f"{np.round(kp, 6).tolist()}, which is not k modulo a reciprocal-lattice "
                "vector: it has no sewing operator there.",
                value=float(np.abs(kp - np.asarray(k, float).reshape(-1)
                                   - np.rint(kp - np.asarray(k, float).reshape(-1))).max()),
                threshold=_atol, context={"operation_ind": int(ind), "k": np.asarray(k, float)})
        return self.bloch_operator(ind, k, k_out=np.asarray(k, dtype=float).reshape(-1),
                                   **kwargs)

    def covariance(self, kpoints=None):
        """``{index: max_k ‖D H(k) D† − H(Rk)‖}``, each operation's covariance measured again, in the units of ``H``.

        At ``kpoints``, an array of reduced coordinates, a row each, or by
        default at the k-points the representation was built at.

        Raises
        ------
        ValueError
            ``r.model`` edited since the representation was measured on it.
        """
        kpoints = (self._battery if kpoints is None
                   else np.atleast_2d(np.asarray(kpoints, float)))
        bloch = _bloch.evaluator(self._model)
        self._check_unedited(bloch)
        H = np.asarray(bloch(kpoints))
        out = {}
        for ind in self._indices:
            op = self._operation(ind)
            Hr = np.asarray(bloch(self.transform_k(ind, kpoints)))
            D = self.matrices[int(ind)]
            src = H.conj() if op.time_reversal else H
            out[int(ind)] = float(np.abs(D @ src @ D.conj().T - Hr).max())
        return out

    def _states(self) -> str:
        """What the matrices act on, as the summary says it."""
        norb, nspin = int(self._model.norb), int(self._model.nspin)
        return (f"the {norb * nspin} states, by index, each orbital's spin up then down "
                "along z" if nspin == 2 else
                f"the {norb} orbital{'s' if norb != 1 else ''}, by index")

    def summary(self) -> str:
        """What the representation holds, a line for each part, with the attribute that holds it."""
        lines = [f"each operation's matrix on {self._states()}   (.matrices)"]
        if self.antiunitary:
            lines.append("  an operation with time reversal is its matrix times complex "
                         "conjugation   (.antiunitary)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        worst = (max(self.numerical.residuals.values()) if self.numerical.residuals
                 else float("nan"))
        norb, nspin = int(self._model.norb), int(self._model.nspin)
        on = (f"{norb * nspin} states" if nspin == 2
              else f"{norb} orbital{'s' if norb != 1 else ''}")
        return (f"SpaceGroupRep({on}, covariance<={worst:.1e}, "
                f"product={self.numerical.product_residual:.1e})")
