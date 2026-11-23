r"""The model's representation of its space group, and the atomic-limit table.

``r.rep`` of what :func:`..groups.realized_group` returns is a
:class:`SpaceGroupRep`: a matrix for each operation of the group, acting on the
model's states.  The little-group representations a set of bands carries are
read through it.  ``r.wyckoff_orbits`` is an :class:`AtomicLimitTable`: the
site irreps the orbitals carry, a row per Wyckoff orbit.  The matrices are the
representation induced from those site irreps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
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
           "NormalizedRep", "AtomicLimitTable", "AtomicLimitTableNumerical",
           "AccidentalDegeneracy", "WyckoffOrbit",
           "WyckoffOrbitNumerical", "BandRepresentation", "DeclarationCheck",
           "SiteMultiplets", "SpaceGroupRep", "SpaceGroupRepNumerical",
           "normalize_rep",
           "bloch_operator", "ANSATZ_GATE", "EXACT_TOL", "covariance_gate"]

# --------------------------------------------------------------------------
# the atomic-limit table
# --------------------------------------------------------------------------

#: The one sentence the summaries of the per-orbital and the atomic-limit
#: tables print under the rows wherever a row's site irrep rests on the
#: lowest-angular-momentum rule.
_LOWEST_L_LEGEND = (
    "the hoppings fix how the orbitals transform relative to one another, not what "
    "they are; where several site irreps fit, the one carried by the "
    "lowest-angular-momentum functions is shown, and `characters=` states another.")

#: What :attr:`_OrbitalRow.determined_by` reads on a declared row whose
#: declared irrep the Hamiltonian's own blocks do not carry.
_NOT_CARRIED = "declaration not carried"

@dataclass(frozen=True)
class _OrbitalRow:
    """One orbital's row of the per-orbital table, and how its site irrep is known.

    ``rep._characters[i]`` is orbital ``i``'s row, and ``row.summary()`` prints
    it.

    Attributes
    ----------
    orbital, site : int
        The orbital, and the site it sits on, by index.
    wyckoff : str
        The site's Wyckoff position, multiplicity and letter (``"3c"``), in
        the conventional cell used by the irreptables package; ``""`` where
        the model's cell could not be matched to that cell.
    site_symmetry : str
        The Hermann-Mauguin symbol of the unitary site-symmetry group
        (``"-42m"``).
    function : str
        The orbital function ``characters=`` declared (``"px"``,
        ``"dx2-y2"``, the shell ``"p"``); ``""`` where none was.
    multiplet : tuple of int
        The orbitals the Hamiltonian binds this one with, which carry its site
        irrep together.
    irrep : str
        The site irrep the orbital's states carry, in the site group of the
        group returned; double-valued with spin, ``-E1`` for an ``s`` orbital
        at a ``3m`` site.  A multiplet holding several irreps reads their sum,
        ``"2x-E1 + -1E2E"`` for a spin-orbit ``p`` shell at a ``3m`` site, and
        one holding copies of one irrep reads that irrep on each row.  ``""``
        where nothing names it.
    named_by : str
        Where the name comes from: ``"tables"``, the elementary band
        representations the tables list at this position; ``"invariants"``,
        on a spinless row at a position the tables list no elementary band
        representation at, the site irrep the tables of the stable real-space
        invariants single out for the orbital's states; ``"point group"``, the
        name the tables give that irrep of the site point group at the origin
        of a symmorphic group; ``"Mulliken"``, the irrep's Mulliken symbol; or
        ``"unnamed"``.
    determined_by : str
        How the site irrep is known: ``"declaration"``; ``"declaration not
        carried"``, declared and not carried by the Hamiltonian's own blocks;
        ``"hamiltonian"``, the same under every labeling the declarations
        leave open; or ``"lowest-l"``, chosen by the lowest-angular-momentum
        rule.
    dim : int
        The dimension of what ``irrep`` names, counting spin: twice the
        orbital factor's in a spinful group.
    site_group : tuple of int
        The indices of the unitary operations that fix the site.
    character, generator_character : dict
        The orbital factor's character, single-valued with spin or without: on
        every operation of the site group by index, and on one operation of each
        class by name.
    declared_as : object or None
        What ``characters=`` held for this orbital, as written; ``None`` where
        nothing was declared.
    declared_irrep, declared_in, reduced_in : str
        The irrep, or sum, that declaration resolved to; the site group and
        group it was read in (``"3m in P3m11' #156.50"``); and, where the
        declared site irreps lowered the cell first, the site group and group
        it was resolved in (``"m in Cm1' #8.33"``).  ``""`` where nothing
        applies.
    declaration_broke : tuple of str
        If nonempty, the operations the declared site irreps break.
    note : str
        If nonempty, how a declaration was read; on a declared spinor basis,
        which member of its pair the row is; for a name Mulliken's convention
        gives another irrep of the site group, which irrep; what a declaration
        not carried stated; and where the cell could not be matched to the
        tables' cell, why no name comes from them.
    residual : float
        The largest misfit over every multiplet of the table, the same on
        every row.  On a multiplet of one orbital or one declared spinor pair,
        the misfit is how far its block misses its character times its spin
        factor; on a measured multiplet of several, how far its blocks miss
        multiplying as its site group does, its double group with spin.  A
        large one says the site irreps are not to be trusted.
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
    """The sum of elementary band representations an orbit of orbitals induces.

    The sum is the first of these to carry the orbit's symmetry data vector:
    a sum at one position the orbitals can be moved to, their own first; the
    sum with the orbit's own stable real-space invariants; a sum at a
    position on the closure of a variety through the site; the one sum the
    vector has (:func:`..srsi._band_representations`).  It does not say
    which site irrep goes to which term.  It is one entry of
    ``rep._characters.band_representations``, and ``br.summary()`` prints it.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals on the orbit's sites.
    wyckoff, site_symmetry : str
        The orbit's Wyckoff position, multiplicity and letter (``"6c"``), and
        its unitary site-symmetry group (``"3m"``).
    terms : tuple
        ``(multiplicity, elementary band representation, position)`` for each
        term, the band representation and the position with its site groups
        written as the tables list them (``(1, "-E1g↑G(2)", "3a(-3m1',-3m)")``);
        empty where no sum is returned.
    alternatives : tuple
        The other sums with the same symmetry data vector and the same stable
        real-space invariants, each written as ``terms`` is; empty where the
        group has no tabulated invariants.
    note : str
        If nonempty, why no sum is returned: none carries the symmetry data
        vector, several do and nothing read says which is induced, or the
        symmetry data vector could not be identified.
    """

    orbitals: tuple
    wyckoff: str
    site_symmetry: str
    terms: tuple
    alternatives: tuple = ()
    note: str = ""

    def spelled(self, terms=None) -> str:
        """A sum on one line, written as the tables list it.

        ``terms`` is :attr:`terms` where ``None``, or one of
        :attr:`alternatives`.  Returns ``"1 x A1↑G(2)@2b(3m1',3m)"`` for
        graphene's orbit, terms joined by ``" + "``, and ``""`` for no terms.
        """
        terms = self.terms if terms is None else terms
        return " + ".join(f"{m} x {name}@{position}" for m, name, position in terms)

    def sentence(self) -> str:
        """The line the per-orbital table's summary prints for this orbit alone.

        The sum it induces and those it is not distinguished from, or the
        note where there is no sum.  Orbits at one Wyckoff position that
        induce the same sums share one line instead (:func:`_orbits_sentence`).
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
    """The count ``k`` of a pseudospin's copies as a word, ``"twofold"``.

    Above four it is written ``"5-fold"``.
    """
    return {2: "twofold", 3: "threefold", 4: "fourfold"}.get(int(k), f"{int(k)}-fold")

def _in_words(n) -> str:
    """The count ``n`` in words up to six, ``"two"``, and in figures above."""
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(int(n), str(int(n)))

@dataclass(frozen=True)
class _MultiplicitySpace:
    """A pseudospin the Hamiltonian conserves on an orbit, and what its copies carry.

    It is one entry of ``rep._characters.multiplicity_spaces``, and
    ``space.summary()`` prints it.  The site irreps are named at one site of
    the orbit, as a row's ``irrep`` is, a sum written ``"2xA1"``.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals the copies are made of.
    wyckoff : str
        The Wyckoff position of the site the site irreps are read at; ``""``
        where the model's cell could not be matched to the conventional cell
        used by the irreptables package.
    dim : int
        The number of copies, the degeneracy of every band on these orbitals.
    content : str
        The site irreps the copies carry, as returned.
    contents : tuple of str
        The site irreps each representation of the copies that the group's own
        table admits gives, in the order the lowest-angular-momentum rule ranks
        them, ``"(unnamed)"`` where nothing names them; with spin, two can be
        the same.
    n_representations : int
        How many representations of the copies the group's own table admits.
    determined_by : str
        ``"declaration"``; ``"lowest-l"``, the rule's choice among several;
        or ``"hamiltonian"``, where the table admits one representation.
    """

    orbitals: tuple
    wyckoff: str
    dim: int
    content: str
    contents: tuple
    n_representations: int = 1
    determined_by: str = "lowest-l"
    # how many orbitals of the copies one site holds, where ``_pairs`` each
    # declared spinor pair counted once; it says what one copy is
    _per_site: int = field(default=0, repr=False, compare=False)
    _pairs: bool = field(default=False, repr=False, compare=False)

    def _distinct(self) -> tuple:
        """The entries of :attr:`contents`, each set of site irreps once, in order.

        Two entries are one set where they hold the same irreps the same
        number of times, whatever the order of the terms: ``-E1 + -E2`` and
        ``-E2 + -E1``, and with spin ``2x-E1`` lifted from both ``2xA1`` and
        ``2xA2`` of ``3m``.
        """
        seen, out = set(), []
        for content in self.contents:
            key = tuple(sorted((name, n) for n, name in _content_terms(content)))
            if key not in seen:
                seen.add(key)
                out.append(content)
        return tuple(out)

    def sentence(self) -> str:
        """The line the per-orbital table's summary prints for this pseudospin."""
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
        """The pseudospin, what its copies carry and how that is known, a line each."""
        how = {"declaration": "stated by the declaration",
               "lowest-l": "assigned by the lowest-angular-momentum rule",
               "hamiltonian": "the one content the group's own table admits"}.get(
                   self.determined_by, self.determined_by)
        # what one copy is: the orbitals it holds at one site, or the declared
        # spinor pairs, and whether the orbit has several sites
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
    """The site irrep each orbital carries, a row per orbital.

    ``rep._characters``, the per-orbital table, from which the matrices are
    induced and the rows of ``r.wyckoff_orbits`` are gathered.  ``table[i]``
    is orbital ``i``'s :class:`_OrbitalRow`, and ``table.summary()`` prints
    it.

    Attributes
    ----------
    orbitals : tuple of _OrbitalRow
        One row per orbital, in the model's order.
    irreps : tuple of str
        Each orbital's site irrep, in the model's order.
    band_representations : tuple of _OrbitBandRepresentation
        The band representation each orbit of orbitals induces, in the order
        of the orbits; empty where the model's cell could not be matched to
        the conventional cell used by the irreptables package.
    multiplicity_spaces : tuple of _MultiplicitySpace
        One per orbit on which the Hamiltonian conserves a pseudospin, with the
        site irreps its copies carry; empty where none does.
    declarations : tuple of dict
        Each declaration as it was read: ``orbitals``, ``declared_as`` and the
        ``declared_irrep`` it resolved to, ``dim``, ``note``, ``declared_in``,
        ``reduced_in``, the operations it ``broke``; ``mulliken``, each name
        written that Mulliken's convention gives another irrep of the site
        group, with that irrep's Mulliken name; and ``returned``, what its
        multiplets carry together.
    checks : tuple of dict
        Each declaration against each operation of its site group the hoppings
        say something about, ``orbitals``, ``ind``, ``name``, ``measured`` and
        ``declared``: the table a ``declaration_breaks_group`` refusal prints.
        ``measured`` is read against a multiplet of the declaration's own
        coupled part of the crystal, so the two can differ by a
        one-dimensional character of the group on that part.
    frame : dict
        ``{ind: name}``, every operation represented, named in the Cartesian
        axes in which the pythtb Lattice is specified.
    spin : ..spin.SpinBasis or None
        The declared spinor basis, where one was; the summary then has a
        ``spin`` column, each row's member of its pair.
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
    # the chooser's own record of the labelings that give every orbital the
    # site irrep the one returned gives it and label some band differently:
    # each coupled part's states, and per part the band labels of every energy
    # multiplet under the one returned and under each of those, with the key
    # that set it aside; `sewing._labeling_note` reads it to say where a set of
    # bands' labels rest on that choice
    _labelings: dict = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.orbitals)

    def __getitem__(self, orbital) -> _OrbitalRow:
        return self.orbitals[int(orbital)]

    @property
    def irreps(self) -> tuple:
        """Each orbital's site irrep, in the model's order."""
        return tuple(o.irrep for o in self.orbitals)

    def summary(self) -> str:
        """The table, a row per orbital, and a line under it for what the rows omit.

        The columns are the rows' fields; ``orbital characters`` is the orbital
        factor's character on one operation of each class, without spin.  A
        mis-oriented declaration shows in that column.  Each line under the
        rows ends with the attribute that holds it, one for: a declaration that
        lowered the cell, breaks operations, is not carried by the
        Hamiltonian's own blocks, or is written in a name Mulliken's convention
        gives another irrep of the site group; an orbit
        whose band representation is not its rows' own elementary band
        representations at its own position (:func:`_its_rows_own`); and a
        conserved pseudospin.  Where a row reads ``lowest-l``, a last line
        says what that means.
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

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"_OrbitalTable({'/'.join(self.irreps)}"
                + (", lowest-l" if any(o.determined_by == "lowest-l"
                                       for o in self.orbitals) else "") + ")")

def _its_rows_own(table, orbit) -> bool:
    """Whether an orbit's sum is its rows' own elementary band representations.

    ``table`` is the per-orbital table and ``orbit`` one of its
    ``band_representations``.  True where every row of the orbit takes its
    name from the tables (``named_by == "tables"``), the sum is nonempty with
    every term at the orbit's own position, and no alternative is wholly at
    that position.  A summary then prints no line under the table for the
    orbit.
    """
    rows = [table.orbitals[int(a)] for a in orbit.orbitals]

    def _here(terms):
        return all(str(position).split("(")[0] == orbit.wyckoff
                   for _m, _name, position in terms)

    return (bool(orbit.terms) and _here(orbit.terms)
            and not any(_here(other) for other in orbit.alternatives)
            and all(o.named_by == "tables" for o in rows))

def _orbits_sentence(orbits) -> str:
    """The line under a summary's table for orbits that induce the same sums.

    ``orbits`` are entries of ``band_representations`` at one Wyckoff
    position with the same terms, alternatives and note.  One orbit reads as
    :meth:`_OrbitBandRepresentation.sentence` writes it; several read once,
    each orbit's orbitals named.
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
    """What the line of ``orbit``, an orbit with no sum, says in its place.

    Its note, or ``"no sum of elementary band representations"`` where the
    note is empty.
    """
    return orbit.note or "no sum of elementary band representations"

def _not_distinguished(orbit) -> str:
    """The clause naming the alternatives of ``orbit`` on its line, or ``""``.

    The clause reads ``", not distinguished by the stable real-space
    invariants from"`` and the sums.
    """
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

# --------------------------------------------------------------------------
# a sum of site irreps, and the rows by Wyckoff orbit
# --------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BandRepresentation:
    """A sum of elementary band representations, each induced from a site irrep.

    The ``band_representation`` of each row of ``r.wyckoff_orbits`` is one, and
    so is each candidate sum :func:`..ebr.decompose` returns.  Iterating,
    indexing and ``len()`` go over :attr:`terms`.  Two sums are equal where
    their terms are.

    Attributes
    ----------
    terms : tuple of ..groups.Term
        If nonempty, the terms of the sum.  Each term holds an integer
        multiplicity, a site irrep, its Wyckoff position, and its
        site-symmetry group with and without time reversal:
        ``Term(1, "-E1", "2b", "3m1'", "3m")``.
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

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return f"{type(self).__name__}({self.label})"


def _words_label(terms) -> str:
    """The sum of ``terms`` with each site-symmetry group, ``"A1 at 2b (3m1')"``."""
    return " + ".join(
        ("" if int(t.multiplicity) == 1 else f"{int(t.multiplicity)}x")
        + f"{t.irrep} at {t.wyckoff} ({t.site_symmetry})" for t in terms)


def _paper_label(terms) -> str:
    """The sum ``terms`` as the tables of the stable real-space invariants write it.

    Each term is its multiplicity, its site irrep and its Wyckoff letter,
    ``"1 (-E2)_b + 1 (-E)_c"``.
    """
    return " + ".join(f"{int(t.multiplicity)} ({t.irrep})_{t.wyckoff.lstrip('0123456789')}"
                      for t in terms)


def _irreps_label(terms) -> str:
    """The site irreps of ``terms``, at one site, as a row prints them: ``"2xA1"``."""
    return " + ".join(("" if int(t.multiplicity) == 1 else f"{int(t.multiplicity)}x") + t.irrep
                      for t in terms)


def _sum_of(terms) -> BandRepresentation:
    """The :class:`BandRepresentation` of ``terms``, labeled by :func:`_words_label`."""
    terms = tuple(terms)
    return BandRepresentation(terms=terms, label=_words_label(terms))


@dataclass(frozen=True)
class DeclarationCheck:
    """The character a declaration states for one operation, beside the one measured.

    One entry of ``r.wyckoff_orbits.numerical.checks``: an operation of the
    declared orbitals' site-symmetry group on which the hoppings fix their
    character relative to the orbitals they couple them to.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals declared.
    index : int
        The operation, by index.
    name : str
        The operation's name, in the Cartesian axes in which the pythtb
        Lattice is specified.
    measured : complex
        The character of the operation on the declared orbitals, as the
        hoppings give it relative to the orbitals they couple them to.
    declared : complex
        The character the declaration states for the operation.  The hoppings
        fix the representation on the orbitals they couple only up to a
        one-dimensional character of the group.  So where the declaration
        holds, ``declared`` is ``measured`` times the value one such character
        takes on the operation.
    """

    orbitals: tuple
    index: int
    name: str
    measured: Any = None
    declared: Any = None


def _declaration_checks(rows) -> tuple:
    """The declaration checks ``rows``, each as a :class:`DeclarationCheck`.

    Each of ``rows`` is a mapping with the keys of
    :attr:`_OrbitalTable.checks`: a declaration's measured and declared
    character on one operation.
    """
    return tuple(DeclarationCheck(orbitals=tuple(int(o) for o in r["orbitals"]),
                                  index=int(r["ind"]), name=str(r["name"]),
                                  measured=r.get("measured"), declared=r.get("declared"))
                 for r in rows)


@dataclass(frozen=True)
class WyckoffOrbitNumerical:
    """How a row of the atomic-limit table was read.

    ``o.numerical`` of a row ``o`` of ``r.wyckoff_orbits``: its sites, each
    orbital's characters, and how each declaration was resolved.

    Attributes
    ----------
    positions : tuple
        Each site's position in reduced coordinates, in the order of
        ``o.site_multiplets``.
    residual : float
        The largest misfit over every multiplet of the table, the same on
        every row.  An operation that fixes a multiplet of one orbital acts on
        its states as a number, the orbital's character, times the operation's
        matrix on spin, the spin factor, which is ``1`` without spin.  On such
        a multiplet, or on one declared spinor pair, the misfit is how far its
        block misses that product.  On a multiplet of several orbitals it
        is how far its blocks miss multiplying as the site-symmetry group
        does, its double group with spin.  Each misfit is the largest modulus
        of an entry of a difference of unitary blocks, and one that is not
        small against ``1`` says the site irreps are not to be trusted.
    characters : dict
        For each orbital, its orbital factor's character on every operation of
        its site-symmetry group, by index.  It is single-valued, with spin or
        without.  A spinor's character is zero on every mirror and twofold
        rotation, so the parity of an orbital under those operations is read
        here.
    named_by : str
        Where the row's site irreps take their names.  ``"tables"``: the
        tables of elementary band representations list the band representation
        the row's orbitals induce at this Wyckoff position, and the name is the
        site irrep those entries carry.  ``"invariants"``: those tables list
        none at this position, and the tables of the stable real-space
        invariants name the site irrep the row's orbitals carry there, which is
        single-valued.  ``"point group"``: neither names it, and the name is
        the one the tables give that irrep at the origin of a symmorphic space
        group with the same point group, so that one irrep has one name at
        every site.  ``"Mulliken"``: those point-group names rest on an
        orientation no table fixes at this site, and the name is the irrep's
        Mulliken symbol, read off its
        characters on the site's rotations and mirrors in the Cartesian axes
        in which the pythtb Lattice is specified.  ``"unnamed"``: nothing names
        the row, and ``o.label`` is blank.  A sum of site irreps takes the
        value of its term that stands latest in the order above.
    declared_irrep : dict
        If nonempty, for each declared orbital, the site irreps its
        declaration resolved to, ``{0: "A1"}``.
    declared_in : dict
        If nonempty, for each declared orbital, the site-symmetry group and
        the group the declaration was read in,
        ``{0: "3m in P6mm1' #183.186"}``.
    reduced_in : dict
        If nonempty, for each declared orbital, where its declared site irreps
        lowered the group of the cell before any hopping was read, the
        site-symmetry group and the group the declaration was resolved in;
        ``""`` where they did not.
    declaration_broke : dict
        If nonempty, for each declared orbital, the names of the operations
        the declared site irreps break.
    determined_by : dict
        For each orbital, how its site irrep is known, in the values of
        ``o.determined_by``.
    alternatives : tuple of BandRepresentation
        If nonempty, the other sums of elementary band representations with
        the same symmetry data vector and the same stable real-space invariants
        as ``o.band_representation``, which no quantity measured distinguishes
        from it.  Empty where the group has no tabulated invariants.
    contents : tuple of BandRepresentation
        If nonempty, the sums of site irreps at the row's Wyckoff position
        that the copies of a pseudospin the Hamiltonian conserves on the row
        can carry, each sum once, ranked by the lowest-angular-momentum rule.
        The Hamiltonian fixes how each operation acts within one copy, and
        leaves open how it acts on the copies themselves.  Each sum comes from
        one representation of the copies: a sum of irreps of the operations
        that carry one copy's part of the crystal onto itself, whose product
        with the action within a copy multiplies as the group returned does.
    n_representations : int
        How many representations of the copies there are, as ``contents``
        defines them; several can give one sum.  ``1`` where no pseudospin
        is conserved.
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

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"WyckoffOrbitNumerical(residual={self.residual:.1e}, named by "
                f"{self.named_by})")


@dataclass(frozen=True)
class WyckoffOrbit:
    """One Wyckoff orbit of the model's orbitals, and the site irreps they carry.

    A row ``o`` of ``r.wyckoff_orbits``: the orbitals the operations of the
    group carry into one another.  ``o.summary()`` prints it.

    Attributes
    ----------
    wyckoff : str
        The Wyckoff position, multiplicity and letter, ``"2b"``, in the
        conventional cell used by the irreptables package; ``""`` where the
        model's cell could not be matched to that cell.
    site_symmetry : str
        The Hermann-Mauguin symbol of the site-symmetry group, read off the
        operations measured to fix a site, with its ``1'`` where the group
        returned has time reversal: ``"3m1'"``, and ``"3"`` in ``P3``.
    orbitals : tuple of int
        The model's orbitals on the orbit's sites.
    label : str
        The site irreps one site of the orbit holds, in the site-symmetry
        group of the group returned, as printed: ``"-E1"``, ``"2xA1"``,
        ``"-E1u + -E3u"``.  With spin they are double-valued, ``-E1`` for an
        ``s`` orbital at a ``3m`` site.  They are read off the characters, so
        are the same in any basis and whatever the spelling of a declaration.
        Two sites of an orbit can carry conjugate site irreps, as ``B1`` at
        ``(1/2, 0)`` and ``B2`` at ``(0, 1/2)`` in ``P4mm1'``.  The row names
        them once, from the source ``numerical.named_by`` gives.
    terms : tuple of ..groups.Term
        The same site irreps as terms, ``Term(1, "-E1", "2b", "3m1'", "3m")``.
    unitary_site_symmetry : str
        The unitary part of the site-symmetry group, ``"3m"``.
    determined_by : str
        How the site irreps are known: of the four values, listed here weakest
        first, the weakest any of the row's orbitals reads.  The hoppings fix
        the representation only up to a one-dimensional character of the group
        on each set of orbitals they couple, and ``characters=`` can fix it.
        ``"declaration not carried"``: ``characters=`` stated the site irreps,
        and the Hamiltonian's own blocks do not carry them; ``label`` is what
        the blocks carry.  ``"lowest-l"``: the choices the hoppings and the
        declarations leave open give some orbital of the row different site
        irreps, and the lowest-angular-momentum rule chose these.
        ``"declaration"``: ``characters=`` stated them, and the matrices
        returned carry them.  ``"hamiltonian"``: nothing was declared for the
        row, and every choice left open gives the same site irreps.
    declared : str, tuple or None
        ``None`` where nothing was declared for the row.  Otherwise what
        ``characters=`` declared: one word where every orbital was declared
        the same, otherwise one entry per orbital, ``None`` for an orbital
        not declared.
    band_representation : BandRepresentation
        The sum of elementary band representations the orbit induces.  It is
        written at the orbit's own Wyckoff position where a sum there carries
        the orbit's symmetry data vector, and otherwise at other positions.
        There its terms need not correspond one to one with the row's site
        irreps.  Its terms are empty where no sum is found for the orbit: the
        model's cell could not be matched to the conventional cell used by the
        irreptables package, no sum carries the symmetry data vector, several
        do and nothing read says which is induced, or the vector could not be
        identified.
    site_multiplets : tuple of tuple of int
        For each of the orbit's sites, the orbitals of its multiplets there,
        which carry the row's site irreps together: ``((0,), (1,))`` for an
        orbit of two sites with one orbital each.
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
        """The row in one line, as ``r.summary()`` prints it.

        ``"2b (3m1'): orbitals 0-1, A1"`` for graphene.
        """
        return (f"{self.wyckoff or '?'} ({self.site_symmetry}): "
                f"{_orbitals_named(self.orbitals)}, {self.label or 'unnamed'}")

    def summary(self) -> str:
        """The row, a line for each part that applies, with the attribute that holds it."""
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

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return f"{type(self).__name__}({self._line()})"


@dataclass(frozen=True)
class AccidentalDegeneracy:
    """A tie between ways of joining a site's orbitals into multiplets.

    One entry of ``r.wyckoff_orbits.numerical.accidental``.  The multiplets
    are found by joining a multiplet with another where the matrices of an
    operation that carry H onto itself reach from the one to the other.  Here
    they reach from one multiplet to several others, by an accidental
    degeneracy of H.  The weight of each such partner, the largest modulus of
    an entry of those matrices between it and the multiplet, is at least half
    the largest weight.  The multiplet could be joined with any of them.  It
    is joined with the lowest-indexed, and the tie is recorded where some of
    the others stay apart.

    Attributes
    ----------
    operation : int
        The operation at which the tie arose, by index.
    orbitals : tuple of int
        The orbitals of the multiplet, of the partner it was joined with, and
        of the partners that stay apart from it.
    chosen : tuple of int
        The orbitals of the multiplet and the partner it was joined with.
    note : str
        The entry, in a sentence.
    """

    operation: int
    orbitals: tuple
    chosen: tuple
    note: str


def _accidental(layout) -> tuple:
    """Accidental degeneracies of ``layout``, as :class:`AccidentalDegeneracy` rows.

    ``layout`` is a representation's multiplets, ``rep._multiplets``; a
    layout with no ``accidental`` entries gives ``()``.
    """
    from ._realized import _record
    return tuple(_record(AccidentalDegeneracy, entry)
                 for entry in (getattr(layout, "accidental", ()) or ()))


@dataclass(frozen=True)
class AtomicLimitTableNumerical:
    """How the atomic-limit table was read.

    ``r.wyckoff_orbits.numerical``: each declaration against the hoppings, and
    how the multiplets were found.

    Attributes
    ----------
    checks : tuple of DeclarationCheck
        If nonempty, for each declaration and each operation of its
        site-symmetry group on which the hoppings fix the declared orbitals'
        character, the character measured beside the character declared.
    residual : float
        The largest ``o.numerical.residual`` over the rows ``o``.
    site_multiplets_determined_by : str
        How the multiplets each site's orbitals form were found:
        ``"hamiltonian"``, measured; ``"declaration"``, stated with
        ``site_multiplets=``.
    accidental : tuple of AccidentalDegeneracy
        If nonempty, an entry for each tie between two ways of joining a
        site's orbitals into multiplets, by an accidental degeneracy of H.
        Each entry names the multiplet returned.
    """

    checks: tuple
    residual: float
    site_multiplets_determined_by: str
    accidental: tuple = ()

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"AtomicLimitTableNumerical({len(self.checks)} checks, "
                f"residual={self.residual:.1e})")


@dataclass(frozen=True)
class AtomicLimitTable:
    """The site irreps the model's orbitals carry, a row per Wyckoff orbit.

    ``r.wyckoff_orbits`` of what :func:`..groups.realized_group` returns: the
    site irreps at each Wyckoff position that specify an atomic limit.
    ``table[j]`` is row ``j``, a :class:`WyckoffOrbit`, the rows in the order
    the orbits' first orbitals come in the model.  Iterating and ``len()`` go
    over the rows.  ``table.summary()`` prints it.  A declaration the
    Hamiltonian's blocks do not carry shows here, as a row whose
    ``determined_by`` reads ``"declaration not carried"`` and a line under the
    table.

    Attributes
    ----------
    numerical : AtomicLimitTableNumerical
        How the table was read: each declaration against the hoppings, and how
        the multiplets were found.
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
        """The table, a row per Wyckoff orbit, and a line under it for what the rows omit.

        The columns are the rows' fields; ``declared`` prints ``-`` for an
        orbital not declared.  Under the rows, a row with no sum of elementary
        band representations has a line saying why.  A row whose sum is not
        its own site irreps at its own position has a line with that sum, and
        with the sums the stable real-space invariants do not distinguish from
        it.  A conserved pseudospin has a line with the site irreps assigned to
        its copies, and how they were chosen.  A declaration the Hamiltonian's
        own blocks do not carry has a line saying so, and so has a declaration
        written in a name Mulliken's convention gives another irrep of the
        site-symmetry group.  Each of these lines ends with the attribute that
        holds it.  Where a row reads ``lowest-l``, a last line says what that
        means.
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

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"AtomicLimitTable({'/'.join(o.label or 'unnamed' for o in self._rows)}"
                + (", lowest-l" if any(o.determined_by == "lowest-l"
                                       for o in self._rows) else "") + ")")


def _declared_text(declared) -> str:
    """A row's ``declared`` as the summary's column prints it.

    ``declared`` is :attr:`WyckoffOrbit.declared`.  Returns the word, the
    entries joined by ``"/"`` with ``"-"`` for an orbital not declared, or
    ``""`` for ``None``.
    """
    if declared is None:
        return ""
    if isinstance(declared, tuple):
        return "/".join("-" if d is None else str(d) for d in declared)
    return str(declared)


def _pseudospin_sentence(space) -> str:
    """The atomic-limit table's summary line for the pseudospin ``space``.

    ``space`` is a :class:`_MultiplicitySpace`.  The line says what its
    copies carry and how that is known.
    """
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
    """The atomic-limit table's summary lines for the declarations of ``table``.

    ``table`` is the per-orbital table, or ``None``, which gives ``[]``.  A
    line is written for each declaration the Hamiltonian's own blocks do not
    carry, or written in a name Mulliken's convention gives another row.
    """
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


#: The grounds a site irrep can rest on, the weakest first: a row reads the
#: weakest among its orbitals'.
_GROUNDS = (_NOT_CARRIED, "lowest-l", "declaration", "hamiltonian")


def _wyckoff_orbits_of(rep) -> tuple:
    """The Wyckoff orbits of the orbitals of ``rep``, a :class:`SpaceGroupRep`.

    A Wyckoff orbit here is the orbit of one multiplet under the group.
    Returns ``(sites, orbitals)`` for each, the sites in the order of their
    orbit of sites, and the orbits in the order their first orbitals come in
    the model: ``(((0, 1), (0, 1)),)`` for graphene.  Raises
    ``InternalError`` where an orbit of multiplets spans several orbits of
    sites.
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
            def position(orbit):
                first = min(int(s) for s in assignment.orbits[orbit])
                row = next((o for o in getattr(rep._characters, "orbitals", ())
                            if int(o.site) == first), None)
                letter = (getattr(row, "wyckoff", "") or (
                    assignment.wyckoff_tables[first]
                    if first < len(assignment.wyckoff_tables) else "") or "?")
                symmetry = (assignment.site_symmetry[first]
                            if first < len(assignment.site_symmetry) else "") or "?"
                return f"{letter} ({symmetry})"
            raise InternalError(
                f"the group carries {_orbitals_named(orbitals)} onto one another, and "
                f"their sites lie at {len(coarse)} Wyckoff positions, "
                + ", ".join(position(o) for o in sorted(coarse))
                + ". Report it, with the model and the call.")
        ordered = tuple(int(s) for s in assignment.orbits[coarse.pop()] if int(s) in sites)
        out.append((ordered, orbitals))
    return tuple(sorted(out, key=lambda w: min(w[1])))


def _terms_of(old_terms) -> tuple:
    """The sum ``old_terms`` as :class:`..groups.Term` records, one per term.

    ``old_terms`` holds ``(multiplicity, elementary band representation,
    position)`` as the tables list them, ``(1, "A1↑G(2)", "2b(3m1',3m)")``,
    which becomes ``Term(1, "A1", "2b", "3m1'", "3m")``.
    """
    return tuple(replace(_tables._ebr_term(name, position), multiplicity=int(m))
                 for m, name, position in old_terms)


def _wyckoff_orbit_table(rep) -> AtomicLimitTable:
    """The atomic-limit table ``r.wyckoff_orbits`` of the representation ``rep``.

    The rows of ``rep._characters`` are gathered by Wyckoff orbit
    (:func:`_wyckoff_orbits_of`).  A row's site irreps are the first of: the
    terms of the orbit's band representation, where every term is at the
    orbit's own position and no alternative is wholly there; the site irreps
    the tables of the stable real-space invariants give a composite band
    representation (:func:`..srsi._composite_localization`); and what the
    orbitals carry at a site of the orbit that is carried to the position's
    own site with its operations unchanged, or at the orbit's first site
    where the orbit has no Wyckoff letter.  Returns a table with no rows
    where ``rep._characters`` is ``None``.  Raises ``InternalError`` where an
    orbit with a Wyckoff letter has no site carried so.
    """
    from . import srsi as _srsi
    from . import _site_names

    table, group, assignment = rep._characters, rep._group, rep._assignment
    layout = rep._multiplets
    if table is None:
        return AtomicLimitTable(_rows=(), numerical=AtomicLimitTableNumerical(
            checks=(), residual=0.0,
            site_multiplets_determined_by=str(getattr(layout, "determined_by", "")),
            accidental=_accidental(layout)))
    grey = getattr(group, "magnetic_type", "I") == "II"
    orbits = _wyckoff_orbits_of(rep)
    sums = tuple(table.band_representations) or (None,) * len(orbits)
    got = _srsi.srsi_tables(group)
    tables = None if isinstance(got, Refusal) else got
    readings = _tables._sites_in_tables_setting(group, assignment.site_positions)
    readings = None if isinstance(readings, Refusal) else readings
    # the site-symmetry groups the list of elementary band representations
    # writes, by position, both symbols, with and without 1', as the list writes
    # them; a position it does not name takes the name read off the operations,
    # with its 1' in a group with time reversal
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
            # 3. the names the orbitals take at a site carried to the position's
            # own site with its operations unchanged; where the tables do not
            # reach the cell there is no position's own site, and the orbit's
            # first site is read
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
                        f"no site of the orbitals {list(orbitals)} at {wyckoff} is carried "
                        "onto the reference site the tables give that position with the "
                        "same site-symmetry operations. Report it, with the model and the "
                        "call.")
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
            accidental=_accidental(layout)),
        _orbital_table=table)


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceGroupRepNumerical:
    """How closely the matrices carry the Hamiltonian and multiply as the group does.

    ``r.rep.numerical``.  ``summary()`` prints it.  Each residual is the
    largest modulus of an entry of a difference of matrices.

    Attributes
    ----------
    residuals : dict
        ``{index: residual}``, in the units of H: how far each operation's
        matrix ``D`` misses carrying the Hamiltonian onto itself.  The residual
        is the largest over the k-points measured of ``D H(k) D† − H(Rk)``
        for the operation ``{R|t}``, and of ``D H(k)* D† − H(−Rk)`` for
        ``{R|t}`` combined with time reversal.
    product_residual : float
        How far the matrices miss multiplying as the group does: ``D(g1)``
        times ``D(g2)``, the second conjugated where ``g1`` is combined with
        time reversal, against ``D(g1 g2)`` times the sign the double group
        gives the product, ``1`` without spin, the largest over every pair of
        operations.
    factor_residual : float
        The same for the Bloch operators of :func:`bloch_operator`, whose
        products carry the phases of the operations' translations; ``nan``
        where computing it failed with an error.
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

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        worst = max(self.residuals.values()) if self.residuals else float("nan")
        return (f"SpaceGroupRepNumerical(covariance<={worst:.1e}, "
                f"product={self.product_residual:.1e}, factor={self.factor_residual:.1e})")


@dataclass(frozen=True)
class SpaceGroupRep:
    """The model's representation of its space group: a matrix for each operation.

    ``r.rep`` of what :func:`..groups.realized_group` returns.  Each matrix
    acts on the model's states and does not depend on ``k``; the Bloch
    operator of an operation at a ``k`` is built from it.  The calls that label
    bands, as ``hsp_table(r, bands=n)``, take ``r`` and read the
    representation from it.  Its methods take an operation by its index, and
    a k-point as a tabulated name, ``"GM"``, or as reduced coordinates;
    :meth:`covariance` takes reduced coordinates only.  ``rep.summary()``
    prints what it holds.

    Attributes
    ----------
    matrices : dict
        ``{index: D(g)}``, each operation's matrix, keyed on the operation's
        index.  It is the representation induced from the site irreps of
        ``r.wyckoff_orbits``, in the convention of the tables used by the
        irreptables package: spin along ``+z`` in a right-handed frame.  On a
        spinful model the states are each orbital's spin up then down, state
        ``2 * orbital + spin``.  The hoppings can leave a multiplet's block
        free up to a change of basis that leaves them unchanged, as on
        identical copies or on uncoupled orbitals.  Such a block is the
        operation on the lowest-angular-momentum functions of its site irreps,
        one function an orbital in order, wherever such a change of basis
        reaches that form: ``p_x, p_y`` for ``E`` at a ``4mm`` site.  Pure time
        reversal's matrix is
        ``matrices[operation_index(r.group, "{1|0,0,0}'")]``.
    antiunitary : tuple of int
        The indices of the operations combined with time reversal, each of
        which acts as its matrix times complex conjugation.
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
    # `{"orbital", "multiplet", "orbit"}`: the orbital the lowest-angular-momentum
    # rule reads first, orbital 0 by default, the multiplet it sits in and that
    # multiplet's orbit; on a declared spinor basis also `"spin_pair"`, its
    # `(up, dn)` pair
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

    # Lookups
    def __len__(self) -> int:
        return len(self.matrices)

    def __contains__(self, ind) -> bool:
        return int(ind) in self.matrices

    def _operation(self, ind, *, call=None):
        """The ``irrep`` operation of index ``ind``.

        ``call`` is the name of the public call its user made, which the
        error opens with; ``None`` opens it with nothing.  Raises ``KeyError``
        where the representation does not hold it.
        """
        for op in self._operations:
            if int(op.ind) == int(ind):
                return op
        raise KeyError(f"{call + '(): ' if call else ''}operation #{ind} is not an "
                       "operation of the group. Pass an index that r.operations lists.")

    def _check_unedited(self, bloch, *, call=None) -> None:
        """Raise ``ValueError`` where the model was edited after it was measured.

        ``bloch`` is the model's Bloch sum (:func:`.._bloch.evaluator`), which
        carries the hash of the model it was summed from.  ``call`` is the
        name of the public call its user made, which the error opens with;
        ``None`` opens it with nothing.  Nothing is checked where
        ``_model_key`` is ``None``.
        """
        if self._model_key is not None and bloch.key != self._model_key:
            raise ValueError(
                f"{call + '(): ' if call else ''}the model has been edited since "
                "realized_group measured it. Call realized_group on the edited model, "
                "and pass what it returns.")

    def _image(self, call, ind, k):
        """The image of the k-points ``k`` under the operation of index ``ind``.

        ``k`` is reduced coordinates, one point or a row for each; ``call``
        is the name of the public call its user made, for the error of an
        index the representation does not hold.  Returns a row for each point.
        """
        k = self._components(call, k)
        return self._group._k_action_of(self._operation(ind, call=call))(k)

    def _components(self, call, k, *, one=False):
        """``k`` as an array with a row for each point, checked for its length.

        ``call`` is the name of the public call its user made, and ``one``
        whether that call takes one point.  Raises ``ValueError`` where ``one``
        is set and ``k`` holds several points, and where a point does not have
        one component for each periodic direction of the model.
        """
        k = np.atleast_2d(np.asarray(k, dtype=float))
        if one and (k.ndim != 2 or k.shape[0] != 1):
            raise ValueError(f"{call}(): k holds {int(np.prod(k.shape[:-1]))} points; pass "
                             "one.")
        dim = int(self._group._dim_k)
        n = int(k.shape[-1])
        if n != dim:
            raise ValueError(f"{call}(): k has {n} component{'' if n == 1 else 's'}; pass "
                             f"{dim}, one for each periodic direction.")
        return k

    @cached_property
    def _scale(self) -> float:
        """``max|H − μ|`` of the model, ``μ`` its mean on-site energy.

        In the units of H, measured once by :func:`.._measure._max_hamiltonian`.
        """
        from ._measure import _max_hamiltonian
        return _max_hamiltonian(self._model)

    def _kpoint(self, call, k):
        """The k-point ``k`` as reduced coordinates, or a refusal.

        ``k`` is reduced coordinates, returned as they are, or a name of
        :func:`..groups.high_symmetry_points`, returned as its coordinates.
        ``call`` is the calling method's name, for the refusal or the error.
        Returns ``out_of_plane`` for a name out of a layer's plane, and the
        refusal of :func:`..groups.high_symmetry_points` where it refuses;
        raises ``ValueError`` for a name the group does not tabulate
        (:func:`.._tables._resolve_kpoint`).
        """
        return _tables._resolve_kpoint(call, self._group, k)

    def transform_k(self, ind, k):
        """The image of a k-point under one operation, not folded back into the zone.

        Parameters
        ----------
        ind : int
            The operation, by index.
        k : str or array_like
            A tabulated name, ``"GM"``, or reduced coordinates in the model's
            periodic directions: one point, or an array with a row for each
            point.

        Returns
        -------
        numpy.ndarray or Refusal
            The image ``gk``, a row for each point, in reduced coordinates:
            ``Rk`` for an operation ``{R|t}``, and ``−Rk`` for one combined
            with time reversal.

        Refusals
        --------
        ``out_of_plane``
            ``k`` names a tabulated point out of the plane of a layer model's
            zone.
        ``tables_not_ready``
            ``k`` is a name, and the model's cell could not be matched to the
            conventional cell used by the irreptables package, so no tabulated
            point can be carried into the model's.

        Raises
        ------
        KeyError
            ``ind`` is not an operation of the representation.
        ValueError
            ``k`` is a name the group does not tabulate, or does not have one
            component for each periodic direction.
        """
        k = self._kpoint("transform_k", k)
        if isinstance(k, Refusal):
            return k
        return self._image("transform_k", ind, k)

    def in_little_group(self, ind, k, *, _atol=1e-8):
        """Whether an operation fixes ``k``, and the reciprocal lattice vector ``gk − k``.

        The operation fixes ``k`` where it carries it to a point ``gk``, the
        image :meth:`transform_k` returns, that is ``k`` up to a reciprocal
        lattice vector.

        Parameters
        ----------
        ind : int
            The operation, by index.
        k : str or array_like
            One point: a tabulated name, ``"GM"``, or reduced coordinates in
            the model's periodic directions.
        _atol : float, default 1e-8
            How far each component of ``gk − k`` may miss an integer, in
            reduced coordinates.

        Returns
        -------
        tuple or Refusal
            ``(member, G)``.  ``member`` is whether ``gk − k`` is a vector of
            integers.  ``G`` is ``gk − k`` rounded to integers, the reciprocal
            lattice vector :meth:`sewing_operator` folds the image back by:
            ``Rk − k`` for an operation ``{R|t}``, and ``−Rk − k`` for one
            combined with time reversal.

        Refusals
        --------
        ``out_of_plane``
            ``k`` names a tabulated point out of the plane of a layer model's
            zone.
        ``tables_not_ready``
            ``k`` is a name, and the model's cell could not be matched to the
            conventional cell used by the irreptables package, so no tabulated
            point can be carried into the model's.

        Raises
        ------
        KeyError
            ``ind`` is not an operation of the representation.
        ValueError
            ``k`` is a name the group does not tabulate, holds several points,
            or does not have one component for each periodic direction.
        """
        k = self._kpoint("in_little_group", k)
        if isinstance(k, Refusal):
            return k
        k = self._components("in_little_group", k, one=True)[0]
        kp = self._image("in_little_group", ind, k)[0]
        G = kp - k
        return bool(np.abs(G - np.rint(G)).max() <= _atol), np.rint(G).astype(int)

    def little_group(self, k, *, _atol=1e-8):
        """The operations that fix a k-point up to a reciprocal lattice vector.

        Parameters
        ----------
        k : str or array_like
            One point: a tabulated name, ``"GM"``, or reduced coordinates in
            the model's periodic directions.
        _atol : float, default 1e-8
            How far each component of ``gk − k`` may miss an integer, in
            reduced coordinates, ``gk`` being the image :meth:`transform_k`
            returns.

        Returns
        -------
        tuple of int or Refusal
            The indices of the operations represented that carry ``k`` to
            itself up to a reciprocal lattice vector, in the group's order,
            those combined with time reversal included.

        Refusals
        --------
        ``out_of_plane``
            ``k`` names a tabulated point out of the plane of a layer model's
            zone.
        ``tables_not_ready``
            ``k`` is a name, and the model's cell could not be matched to the
            conventional cell used by the irreptables package, so no tabulated
            point can be carried into the model's.

        Raises
        ------
        ValueError
            ``k`` is a name the group does not tabulate, holds several points,
            or does not have one component for each periodic direction.

        See Also
        --------
        ..groups.little_group : the little group of ``k`` read off the group
            alone, as a record.
        """
        k = self._kpoint("little_group", k)
        if isinstance(k, Refusal):
            return k
        k = self._components("little_group", k, one=True)[0]
        return tuple(ind for ind in self._indices
                     if self.in_little_group(ind, k, _atol=_atol)[0])

    def bloch_operator(self, ind, k, *, k_out=None, **kwargs):
        """The Bloch operator ``U(g;k)`` of one operation at a k-point.

        It carries the Hamiltonian at ``k`` onto the Hamiltonian at ``k_out``,
        ``U H(k) U† = H(k_out)``, with ``H(k)*`` in place of ``H(k)`` for an
        operation combined with time reversal.  It is :func:`bloch_operator`
        evaluated on this representation's model, group and matrix
        ``matrices[ind]``.

        Parameters
        ----------
        ind : int
            The operation, by index.
        k : str or array_like
            A tabulated name, ``"GM"``, or reduced coordinates in the model's
            periodic directions: one point, or an array with a row for each
            point.
        k_out : str, array_like or None, default None
            Where the Hamiltonian is read: the image ``gk`` that
            :meth:`transform_k` returns, or a point a reciprocal lattice vector
            from it, as a name or reduced coordinates.  ``None`` reads it at
            ``gk`` itself.
        **kwargs
            ``tol`` and ``check``, passed to :func:`bloch_operator`.

        Returns
        -------
        numpy.ndarray or Refusal
            ``U(g;k)``, of shape ``(nstate, nstate)`` for one point and
            ``(Nk, nstate, nstate)`` for ``Nk`` points, ``nstate`` being the
            number of the model's states.

        Refusals
        --------
        ``out_of_plane``
            ``k`` or ``k_out`` names a tabulated point out of the plane of a
            layer model's zone.
        ``tables_not_ready``
            ``k`` or ``k_out`` is a name, and the model's cell could not be
            matched to the conventional cell used by the irreptables package,
            so no tabulated point can be carried into the model's.

        Raises
        ------
        KeyError
            ``ind`` is not an operation of the representation.
        ValueError
            ``k`` or ``k_out`` is a name the group does not tabulate, or ``k``
            does not have one component for each periodic direction.  Also
            where ``k_out`` is not ``gk`` up to a reciprocal lattice vector.
            With ``check=False``, nothing is raised, and the difference is
            rounded to a reciprocal lattice vector.

        See Also
        --------
        sewing_operator : the Bloch operator folded back to ``k``, at a
            ``k`` the operation fixes.
        """
        k = self._kpoint("bloch_operator", k)
        if isinstance(k, Refusal):
            return k
        if k_out is not None:
            k_out = self._kpoint("bloch_operator", k_out)
            if isinstance(k_out, Refusal):
                return k_out
        return bloch_operator(self._model, self._group, self._assignment,
                              self._operation(ind, call="bloch_operator"),
                              self.matrices[int(ind)], k, k_out=k_out, **kwargs)

    def sewing_operator(self, ind, k, *, _atol=1e-8, _call="sewing_operator", **kwargs):
        """The Bloch operator of an operation that fixes ``k``, folded back to ``k``.

        It is :meth:`bloch_operator` with ``k_out = gk − G``: the image ``gk``
        folded back by ``G``, the reciprocal lattice vector nearest
        ``gk − k``.  It carries the Hamiltonian at ``k`` onto the Hamiltonian
        at that point, ``U H(k) U† = H(gk − G)``, with ``H(k)*`` in place of ``H(k)``
        for an operation combined with time reversal.  ``gk − G`` is ``k``
        where the operation fixes ``k`` exactly, and lies within ``_atol`` of
        it otherwise.  For the eigenvectors ``V`` of a set of bands at ``k``,
        ``V† U V`` is the sewing matrix of that set, with ``V*`` in place of
        the last ``V`` for an operation combined with time reversal.  Where
        the operation fixes ``k`` only within ``_atol``, it is so up to the
        change of ``H`` over that distance in ``k``, and not exactly.

        Parameters
        ----------
        ind : int
            The operation, by index.
        k : str or array_like
            One point: a tabulated name, ``"GM"``, or reduced coordinates in
            the model's periodic directions.
        _atol : float, default 1e-8
            How far each component of ``gk − k`` may miss an integer, in
            reduced coordinates, ``gk`` being the image :meth:`transform_k`
            returns.  It alone decides whether the operation fixes ``k``.
        _call : str, default "sewing_operator"
            The name of the public call its user made, which an error opens
            with.
        **kwargs
            ``tol`` and ``check``, passed to :func:`bloch_operator`.  They
            govern the images of the orbitals and the entries of the matrix
            read as zero, not whether the operation fixes ``k``.

        Returns
        -------
        numpy.ndarray or Refusal
            ``U(g;k)``, of shape ``(nstate, nstate)``, ``nstate`` being the
            number of the model's states.

        Refusals
        --------
        ``little_group``
            the operation does not carry ``k`` to itself up to a reciprocal
            lattice vector.
        ``out_of_plane``
            ``k`` names a tabulated point out of the plane of a layer model's
            zone.
        ``tables_not_ready``
            ``k`` is a name, and the model's cell could not be matched to the
            conventional cell used by the irreptables package, so no tabulated
            point can be carried into the model's.

        Raises
        ------
        KeyError
            ``ind`` is not an operation of the representation.
        ValueError
            ``k`` is a name the group does not tabulate, holds several points,
            or does not have one component for each periodic direction.

        See Also
        --------
        ..sewing.sewing_matrix : the sewing matrix of a set of bands, with its
            eigenvalues.
        """
        k = self._kpoint(_call, k)
        if isinstance(k, Refusal):
            return k
        k = self._components(_call, k, one=True)[0]
        kp = self._image(_call, ind, k)[0]
        G = np.rint(kp - k)
        miss = float(np.abs(kp - k - G).max())
        if not miss <= _atol:
            return Refusal(
                "little_group",
                f"operation #{ind} carries k = {np.round(k, 6).tolist()} "
                f"to {np.round(kp, 6).tolist()}, which is not k up to a reciprocal lattice "
                "vector. Pass an operation of the little group of k, which little_group "
                "lists.",
                value=miss, threshold=_atol, context={"operation_ind": int(ind), "k": k})
        # the image is k plus G within _atol: the Hamiltonian is read at the
        # image folded back by G, which bloch_operator then finds a reciprocal
        # lattice vector from the image to rounding
        return self.bloch_operator(ind, k, k_out=kp - G, **kwargs)

    def covariance(self, kpoints=None):
        """Each operation's covariance residual, measured again, in the units of H.

        The residual of an operation with matrix ``D`` is the largest modulus
        of an entry of ``D H(k) D† − H(Rk)`` over the k-points for the
        operation ``{R|t}``, and of ``D H(k)* D† − H(−Rk)`` for ``{R|t}``
        combined with time reversal.

        Parameters
        ----------
        kpoints : array_like or None, default None
            Reduced coordinates in the model's periodic directions, a row for
            each k-point.  ``None`` measures at the k-points
            ``numerical.residuals`` was measured at.

        Returns
        -------
        dict
            ``{index: residual}``, an entry for each operation represented.

        Raises
        ------
        ValueError
            The model has been edited since ``realized_group`` measured it.
        """
        kpoints = (self._battery if kpoints is None
                   else np.atleast_2d(np.asarray(kpoints, float)))
        bloch = _bloch.evaluator(self._model)
        self._check_unedited(bloch, call="covariance")
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
        """What the matrices act on, in the summary's words.

        ``"the 2 orbitals, by index"`` for graphene; on a spinful model, the
        states, each orbital's spin up then down along z.
        """
        norb, nspin = int(self._model.norb), int(self._model.nspin)
        return (f"the {norb * nspin} states, by index, each orbital's spin up then down "
                "along z" if nspin == 2 else
                f"the {norb} orbital{'s' if norb != 1 else ''}, by index")

    def summary(self) -> str:
        """What the representation holds, a line each, with the attribute that holds it."""
        lines = [f"each operation's matrix on {self._states()}   (.matrices)"]
        if self.antiunitary:
            lines.append("  an operation with time reversal is its matrix times complex "
                         "conjugation   (.antiunitary)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        worst = (max(self.numerical.residuals.values()) if self.numerical.residuals
                 else float("nan"))
        norb, nspin = int(self._model.norb), int(self._model.nspin)
        on = (f"{norb * nspin} states" if nspin == 2
              else f"{norb} orbital{'s' if norb != 1 else ''}")
        return (f"SpaceGroupRep({on}, covariance<={worst:.1e}, "
                f"product={self.numerical.product_residual:.1e})")
