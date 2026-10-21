r"""Whether a band set could be an atomic limit, and its symmetry-indicator class.

An elementary band representation (EBR) is the band set one site irrep induces
from its Wyckoff position, where that band set is no sum of others; an atomic
limit's symmetry content is a sum of EBRs with nonnegative integer
multiplicities.

* :func:`decompose`: every such sum that carries a band set's content, or none,
  with what that rests on;
* :func:`symmetry_indicator`: the class of the content among the band
  structures modulo the atomic limits, nonzero where no band set with this
  content is an atomic limit;
* :func:`completeness_check` and :func:`regular_completion`: the bands each sum
  holds, and the bands that would carry a band set up to whole copies of the
  regular band representation;
* :func:`subtract`: one band set's content less another's;
* :func:`ebr_rows`, :func:`in_plane_restriction`, :func:`content_of`,
  :func:`regular_content` and :func:`kpoint_of_label`: the EBRs as symmetry
  vectors, the coreps a layer measures, a content as counts, one regular copy's
  content, and the point a corep belongs to.

A sum is necessary for an atomic limit and never sufficient: the content at
isolated k-points is blind to what happens between them.  The group's data is
``irrep``'s tables, read through :mod:`._tables` and carried into the convention
in which a lattice translation acts as ``e^{-ik·t}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Mapping

import numpy as np

from . import _bloch
from . import _layers
from . import _spacegroup
from . import _srsitables
from . import _tables
from . import _text
from . import sewing as _sewing
from .refusal import InternalError, Refusal
from ._tolerance import CHARACTERS_TOL, ROUNDING, _INT_TOL, _PRESENTATION_RELATIVE
from ._text import _moduli, _sum_lines, _times
from ._tables import (
    _bands_spanned, _letter_only, _wyckoff_letter, ebr_rows, kpoint_of_label)
from ._matrices import _orb_positions_3d, covariance_gate
from ._smith import (
    _compatibility, _divisors_of, _filled_content, _smith_system, _structure,
    _verify_smith)
from ._srsitables import to_irreptables_name

__all__ = [
    "WannierCaveat", "Restriction", "EBRDecomposition", "CompletenessReport",
    "RegularCompletion", "SymmetryIndicator",
    "content_of", "ebr_rows", "kpoint_of_label", "in_plane_restriction",
    "regular_content", "decompose", "subtract", "completeness_check",
    "regular_completion", "symmetry_indicator",
    "DEFAULT_ENUMERATION_CAP", "CAVEAT_TEXT", "INDICATOR_CONVENTION", "CHARACTERS_TOL",
]

#: How many solutions the enumeration reports before it stops.  A truncated
#: enumeration is recorded as such; it never silently becomes "these are all".
DEFAULT_ENUMERATION_CAP = 200

CAVEAT_TEXT = (
    "symmetry-indicated decomposable does not mean Wannierizable: a nonnegative-integer "
    "EBR decomposition of high-symmetry-point content is necessary for an atomic "
    "limit, never sufficient, so an obstruction beyond symmetry indicators can "
    "survive this verdict. No nonnegative-integer solution means no atomic limit."
)

# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WannierCaveat:
    """What a decomposition does not certify: a sum of elementary band representations is necessary for an atomic limit, never sufficient.

    Attributes
    ----------
    text : str
        :data:`CAVEAT_TEXT`, the caveat in a sentence.
    """

    text: str = CAVEAT_TEXT

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return "WannierCaveat(necessary for an atomic limit, not sufficient)"

@dataclass(frozen=True)
class Restriction:
    """Which coreps of the group's elementary band representations a model measures, and whether each of the others repeats a measured one.

    A corep at a point the model does not sample, as a layer's at
    ``k_z = ½``, that occurs in every elementary band representation as often as
    a measured one adds no condition; where every one does, a verdict read on
    the measured coreps is the full verdict.  ``summary()`` prints it.

    Attributes
    ----------
    applied : bool
        Whether any corep is left out; ``False`` for a crystal periodic in three
        directions.
    points : tuple of str
        The points the model samples.
    retained, dropped : tuple of str
        The coreps kept and left out.
    copies : dict
        ``{corep left out: (kept coreps it repeats, ...)}``: each elementary
        band representation holds it as often as it holds those.
    exact : bool
        Whether every corep left out repeats a kept one, so that the verdict on
        the kept coreps is the full one.
    orphans : tuple of str
        The coreps left out that repeat none: conditions given up.
    require_exact : bool
        Whether an orphan was refused.
    rank, full_rank : int
        The rank of the matrix of elementary band representations on the kept
        coreps and on all of them; equal whenever :attr:`exact`.
    layer_extent : tuple of float or None
        For :func:`decompose` on a layer, the heights its orbitals span along
        its normal, from its middle plane; ``None`` for a crystal.
    outside_layer : dict
        ``{Wyckoff position: heights}``: the positions of a free-standing
        layer's group off the layer, which hold none of its orbitals; their
        band representations are left out.
    """

    applied: bool
    points: tuple
    retained: tuple
    dropped: tuple
    copies: dict
    exact: bool
    rank: int
    full_rank: int
    orphans: tuple = ()
    require_exact: bool = True
    layer_extent: tuple | None = None
    outside_layer: dict = field(default_factory=dict)

    def summary(self) -> str:
        """The coreps left out, a line for each part, with the attribute that holds it."""
        if not self.applied:
            return "every tabulated point is measured, and no corep is left out   (.applied)"
        points = sorted({kpoint_of_label(label) for label in self.dropped})
        lines = [f"the coreps at {_text._listed(points)} are not measured, and are left "
                 "out   (.dropped)"]
        copied = {label: rows for label, rows in self.copies.items() if rows}
        if copied:
            which = "each" if self.exact else "each but " + _text._listed(self.orphans)
            lines.append(f"  {which} occurs in every elementary band representation as often "
                         "as a measured one, and leaving it out changes no verdict   (.copies)")
            lines.append("    " + ", ".join(f"{label} as {rows[0]}"
                                            for label, rows in copied.items()))
        if self.orphans:
            lines.append(f"  {_text._listed(self.orphans)} repeat no measured corep: leaving "
                         "them out gives up a condition, and a verdict is a necessary "
                         "condition only   (.orphans, .exact)")
        outside = sorted(self.outside_layer)
        if outside:
            lowest, highest = self.layer_extent or (0.0, 0.0)
            off = "off the layer" if highest - lowest <= ROUNDING else "outside the slab"
            lines.append(f"  {_text._listed(outside)} lie {off}, so their band "
                         "representations are left out   (.outside_layer)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        if not self.applied:
            return "Restriction(none: every tabulated point is in-plane)"
        return (f"Restriction({len(self.retained)} rows kept, {len(self.dropped)} dropped, "
                f"exact={self.exact}, rank {self.rank}/{self.full_rank})")

@dataclass(frozen=True)
class EBRDecomposition:
    """Every sum of elementary band representations, with nonnegative integer multiplicities, that carries a band set's content, or none.

    ``summary()`` prints the verdict and each sum.  The record is truthy, a
    ``"none"`` included.

    Attributes
    ----------
    verdict : {"symmetry-indicated decomposable", "none"}
        Whether a sum carries the content.  The first is necessary for an atomic
        limit and never sufficient: an obstruction beyond symmetry indicators
        can survive it (:attr:`caveat`).  ``"none"`` says no atomic limit has
        this content, and :attr:`certificate` says what that rests on.
    solutions : tuple of tuple of int
        Every sum, as multiplicities over :attr:`ebrs`: the fewest elementary
        band representations first, then the one with more of those the tables
        list first.
    terms : tuple of tuple
        The same, each sum as its nonzero terms, each a multiplicity, an
        elementary band representation and its Wyckoff position;
        :meth:`solution_strings` writes each sum as a line.
    positions : tuple of tuple of str
        Each sum's Wyckoff positions, which no labelling of the orbitals moves
        (:attr:`site_content`).
    ebrs : tuple
        ``(elementary band representation, Wyckoff position)`` for each of the
        group's, the basis :attr:`solutions` is written in.
    content : dict
        The content decomposed, ``{corep: multiplicity}``.
    labels, y : tuple
        The coreps kept, and the content on them.
    certificate : str or None
        What a ``"none"`` rests on: ``"proof"``, no multiplicity being able to
        exceed :attr:`bound` and no sum within it carrying the content, or
        ``"none within the bound"``.  ``None`` for a decomposition.
    bound : int
        The largest multiplicity searched.
    bound_is_rigorous : bool
        Whether the bound provably leaves out no sum: so wherever every
        elementary band representation holds a kept corep, since then no
        multiplicity exceeds the content's largest entry.
    enumeration : dict
        ``{"cap", "n_solutions", "truncated"}``: the most sums listed, how many
        were, and whether the list stops short of all.
    signed_combinations : dict or None
        For a ``"none"``, the same search with multiplicities from ``-bound`` to
        ``bound``: its ``solutions``, ``terms`` and ``n_solutions``.  One found
        is the signature of a fragile band set, none of a stable obstruction;
        signed terms can cancel without limit, so either is evidence.
    restriction : Restriction
        The coreps measured, and on a free-standing layer the Wyckoff positions
        off it, whose band representations every sum leaves out.
    site_content : dict or None
        From ``rep``: each orbital's site irrep and how it is known
        (``determined_by``).  Where one is ``"lowest-l"``, another declared
        content renames the site irreps in :attr:`terms` while
        :attr:`positions` stay.  ``None`` without ``rep``.
    completeness : dict
        ``claim`` is ``"exact"``, or ``"necessary-condition"`` where coreps
        left out repeat no kept one; ``refused`` names the points left out,
        ``skipped`` those whose little group is the identity alone, read as the
        band count, ``source`` whether the content was measured or handed in as
        counts, and ``note`` says it in words.
    caveat : WannierCaveat
        What the verdict does not certify.
    n_bands : int or None
        The number of bands, where the call was told it.
    """

    verdict: str
    content: dict
    labels: tuple
    y: tuple
    ebrs: tuple
    solutions: tuple
    terms: tuple
    positions: tuple
    bound: int
    bound_is_rigorous: bool
    certificate: str | None
    enumeration: dict
    signed_combinations: dict | None
    restriction: Restriction
    site_content: dict | None
    completeness: dict
    caveat: WannierCaveat = field(default_factory=WannierCaveat)
    n_bands: int | None = None
    _status: str = field(default="", repr=False, compare=False)

    @property
    def decomposable(self) -> bool:
        """Whether a sum of elementary band representations carries the content."""
        return self.verdict == "symmetry-indicated decomposable"

    @property
    def unique(self) -> bool:
        """Whether exactly one sum carries the content."""
        return len(self.solutions) == 1

    def solution_strings(self) -> tuple:
        """Each sum as a line, ``"1 x -E1u↑G(2)@3a(-3m1',-3m)"``, a term per elementary band representation."""
        out = []
        for terms in self.terms:
            out.append(" + ".join(f"{m} x {name}@{wy}" for m, name, wy in terms)
                       or "(empty)")
        return tuple(out)

    def summary(self) -> str:
        """The verdict, a line for each part, with the attribute that holds it."""
        n = len(self.solutions)
        if self.decomposable:
            many = ("at least " if self.enumeration.get("truncated") else "") + str(n)
            lines = [f"symmetry-indicated decomposable: {many} sum{'' if n == 1 else 's'} of "
                     f"elementary band representations carr{'ies' if n == 1 else 'y'} this "
                     "content   (.verdict)"]
        else:
            lines = ["no sum of elementary band representations carries this content, so no "
                     "atomic limit has it   (.verdict)"]
        lines.append("  the coreps at the high-symmetry points: "
                     f"{_sewing.format_counts(self.content)}   (.content)")
        if self.terms:
            lines.append("  every sum, each elementary band representation written as the site "
                         "irrep inducing it at its Wyckoff position (site symmetry)   (.terms)")
            lines += _sum_lines(self.terms)
        times = _times(self.bound)
        if self.certificate == "proof":
            lines.append(f"  proved: no elementary band representation can occur more than "
                         f"{times}, and no sum with each at most {times} carries the "
                         "content   (.certificate, .bound)")
        elif self.certificate is not None:
            lines.append(f"  not proved: no sum with each elementary band representation at "
                         f"most {times} carries the content, and a sum with more is not ruled "
                         "out   (.certificate, .bound)")
        if self.signed_combinations:
            lines.append("  " + _signed_sums_line(self.signed_combinations)
                         + "   (.signed_combinations)")
        lines += self._restriction_lines()
        if self.terms and self.site_content is not None and "lowest-l" in tuple(
                self.site_content.get("determined_by", ())):
            lines.append("  some site irreps were assigned by the lowest-angular-momentum rule: "
                         "a different declared content renames them in these sums, while the "
                         "Wyckoff positions stay   (.site_content, .positions)")
        if self.decomposable:
            lines.append("  necessary for an atomic limit, never sufficient: an obstruction "
                         "beyond symmetry indicators can survive it   (.caveat)")
        return "\n".join(lines)

    def _restriction_lines(self) -> list:
        """What the verdict was read without: points refused or out of the plane, and positions."""
        r, lines = self.restriction, []
        refused = sorted(self.completeness.get("refused") or ())
        away = sorted({kpoint_of_label(label) for label in r.dropped} - set(refused))
        where = []
        if refused:
            where.append(f"{_text._listed(refused)} refused, and "
                         f"{'are' if len(refused) > 1 else 'is'} left out with "
                         f"{'their' if len(refused) > 1 else 'its'} coreps")
        if away:
            where.append(f"{_text._listed(away)} "
                         + ("lie out of the layer's plane" if r.layer_extent is not None
                            else "are not measured"))
        if where:
            lines.append("  " + "; ".join(where) + ": "
                         + ("every corep there occurs in every elementary band representation "
                            "as often as a measured one, so the verdict is exact" if r.exact
                            else f"{_text._listed(r.orphans)} repeat no measured corep, so "
                            "the verdict is a necessary condition only")
                         + "   (.restriction, .completeness)")
        if self.completeness.get("source") == "mapping":
            lines.append("  the content was handed in as counts: its completeness is taken as "
                         "stated   (.completeness)")
        outside = sorted(r.outside_layer)
        if outside:
            lowest, highest = r.layer_extent or (0.0, 0.0)
            off = "off the layer" if highest - lowest <= ROUNDING else "outside the slab"
            lines.append(f"  {_text._listed(outside)} lie {off}, so their band representations "
                         "are left out   (.restriction.outside_layer)")
        return lines

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        n = len(self.solutions)
        return (f"EBRDecomposition({self.verdict}, {n} solution{'' if n == 1 else 's'}, "
                + ("exact)" if self.completeness["claim"] == "exact"
                   else "necessary condition only)"))

def _signed_sums_line(probe) -> str:
    """What the search with negative multiplicities allowed found, in one line.

    Every combination it lists has a negative multiplicity, since the
    nonnegative search within the same bound found none; a list cut at its cap
    counts at least that many; and none within the bound is the signature of
    a stable obstruction.  Signed multiplicities have no rigorous bound, so
    each line is evidence.
    """
    n = int(probe["n_solutions"])
    if n:
        many = (f"{'at least ' if probe.get('truncated') else ''}{n} "
                f"combination{'' if n == 1 else 's'}")
        verb = "reproduces" if n == 1 else "reproduce"
        return (f"{many} of elementary band representations with some negative "
                f"multiplicities {verb} the content (evidence of fragility)")
    bound = int(probe["bound"])
    return (f"no combination of elementary band representations with multiplicities "
            f"between -{bound} and {bound} reproduces the content (evidence of a stable "
            "obstruction)")

@dataclass(frozen=True)
class CompletenessReport:
    """Whether a band set's content is a sum of elementary band representations, and how many bands each sum holds.

    The count behind comparing two models at equal band count, each side of a
    gap holding whole band representations.  It speaks for the energy bands
    alone: a degenerate multiplet split across a gap can make two halves with
    opposite feature-resolved Chern numbers whose total reads trivial, and only
    a feature-resolved measurement tells them apart.  ``summary()`` prints it.

    Attributes
    ----------
    complete : bool
        Whether a sum carries the content, with the reach of
        :attr:`EBRDecomposition.verdict`.
    decomposition : EBRDecomposition
        The sums.
    n_bands : int or None
        The bands of the set, where the call was told them.
    band_counts : tuple of int
        The bands each sum holds: each multiplicity times the bands of its
        elementary band representation.
    band_count_matches : bool or None
        Whether every sum holds :attr:`n_bands`; ``None`` where no set of bands
        was named.  A mismatch says the content counts other bands than the set
        named.
    """

    complete: bool
    decomposition: EBRDecomposition
    n_bands: int | None
    band_counts: tuple
    band_count_matches: bool | None

    def summary(self) -> str:
        """The count, a line for each part, with the attribute that holds it."""
        n = len(self.band_counts)
        if self.complete:
            lines = [f"complete: {n} sum{'' if n == 1 else 's'} of elementary band "
                     f"representations carr{'ies' if n == 1 else 'y'} this content   (.complete)"]
            counts = sorted(set(self.band_counts))
            each = (f"each sum is {counts[0]} band{'' if counts[0] == 1 else 's'}"
                    if len(counts) == 1 else f"the sums are {_text._listed(counts)} bands")
            if self.band_count_matches is None:
                lines.append(f"  {each}   (.band_counts)")
            elif self.band_count_matches:
                lines.append(f"  {each}, as many as were asked for   (.band_counts, .n_bands)")
            else:
                lines.append(f"  {each}, where {self.n_bands} were asked for: the content counts "
                             "other bands than those   (.band_counts, .n_bands)")
            lines.append("  the sums, and what a sum does not certify   (.decomposition)")
        else:
            lines = ["not complete: no sum of elementary band representations carries this "
                     "content   (.complete)",
                     "  what that rests on   (.decomposition)"]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"CompletenessReport(complete={self.complete}, "
                f"n_bands={self.n_bands}, band_counts={self.band_counts})")

@dataclass(frozen=True)
class RegularCompletion:
    r"""The bands that would carry a band set up to whole copies of the regular band representation, and the orbitals they would be.

    The regular band representation is the one the regular representation of a
    site group induces: ``|U|`` bands at every k, ``|U|`` the number of unitary
    operations up to lattice translations, the single-valued group's for a
    spinful model.  Both sides of a gap completed to it make an atomic limit
    together, which lets two models be compared at equal band count.
    ``summary()`` prints it.

    Attributes
    ----------
    copies : int
        How many regular copies the completed bands make.
    bands_per_copy : int
        ``|U|``, the bands of one copy.
    target : dict
        The content of that many copies.
    occupied : dict
        The content handed in.
    padding : dict
        ``target − occupied``: the coreps the missing bands carry.
    decomposition : EBRDecomposition
        The padding's own sums: the orbitals the missing bands would be.
    regular : dict
        One copy's content, from the group's corep characters.
    """

    copies: int
    bands_per_copy: int
    target: dict
    occupied: dict
    padding: dict
    decomposition: EBRDecomposition
    regular: dict

    def summary(self) -> str:
        """The padding, a line for each part, with the attribute that holds it."""
        copies = "one copy" if self.copies == 1 else f"{self.copies} copies"
        lines = [f"{copies} of the regular band representation, the one the regular "
                 f"representation of a site group induces: {self.bands_per_copy} bands at every "
                 f"k{'' if self.copies == 1 else ' each'}   (.copies, .bands_per_copy)"]
        if not self.padding:
            lines.append("  nothing is missing: the bands carry that content already   (.padding)")
            return "\n".join(lines)
        lines.append("  the coreps the missing bands carry: "
                     f"{_sewing.format_counts(self.padding)}   (.padding)")
        if self.decomposition.terms:
            lines.append("  the orbitals they would be, each the site irrep at its Wyckoff "
                         "position (site symmetry)   (.decomposition)")
            lines += _sum_lines(self.decomposition.terms)
        else:
            lines.append("  no sum of elementary band representations carries them   "
                         "(.decomposition)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"RegularCompletion({self.copies} x {self.bands_per_copy} bands, "
                f"padding={self.padding}, {self.decomposition.verdict})")

#: The statement of the convention every indicator record carries.
INDICATOR_CONVENTION = (
    "the indicator is evaluated in this package's physical convention, in which a "
    "lattice translation is represented by e^(-ik.t). The tables, written in the BCS "
    "convention -- and therefore irreptables' EBR rows and its shipped Smith factors "
    "u, v -- use the historical e^(+ik.t), whose labels at k are those of the physical "
    "representations at -k; groups.ebr_data undoes that swap on the rows and on u "
    "and v together, so the linear form u.y is the same functional here as there, "
    "read on relabelled coordinates. The indicator group is therefore convention-"
    "invariant by Smith equivalence, and so is every indicator value: measured over "
    "the 379 unitary and grey systems with a nontrivial indicator group, the two "
    "computations agree exactly -- the automorphism relating them is the identity, "
    "in every group. Wyckoff "
    "positions never move. What is not convention-invariant is a linear form written "
    "in the BCS convention's labels: irreptables' named indicator formulas have to be "
    "re-keyed by the row permutation of the change of convention before they can be "
    "read on a returned symmetry vector."
)

@dataclass(frozen=True)
class SymmetryIndicator:
    r"""The symmetry-indicator class of a band set: the class of its symmetry vector among the group's band structures modulo its atomic limits, ``X_BS = {BS}/{AI}``.

    A nonzero class says no band set with this content is an atomic limit, nor
    any with atomic bands added; a zero class certifies nothing, fragile band
    sets lying in it.  ``summary()`` prints it.  The record is truthy whatever
    its verdict.

    Attributes
    ----------
    verdict : {"indicated nontrivial", "indicator zero", "trivial indicator group"}
        Whether the class is nonzero, zero, or not there, the group having no
        indicator.
    number_str, group_name : str
        The group the class lives in, the one handed in.
    magnetic_type : {"I", "II"}
        A space group, or a grey group, time reversal adjoined.
    divisors : tuple of int
        The elementary divisors above 1 of the group's matrix of elementary band
        representations: ``X_BS`` is ``⊕ᵢ ℤ_{dᵢ}``.
    structure : str
        The same as text, ``"Z2 x Z4"``; ``"trivial"`` where there is none.
    values : tuple of int
        The class, one integer modulo each divisor, in the coordinates the
        Smith normal form of that matrix gives.
    order : int
        The class's order, the fewest copies of the band set whose class is
        zero: ``1`` for the zero class.  Unlike :attr:`values` it does not
        depend on the coordinates.
    nontrivial : bool
        Whether the class is nonzero; :attr:`indicated` is the same.
    canonical : dict or None
        The class in the named indicators of the paper that tabulates the stable
        real-space invariants, Hwang *et al.*, Nat. Commun. **17**, 8564 (2026):
        ``{name: value}``, computed through those invariants.  ``None`` for a
        unitary group, which the paper does not tabulate, for a trivial
        indicator group, and where the converted tables are not installed;
        :attr:`canonical_note` says which.
    canonical_note : str
        Where :attr:`canonical` came from, or why it is ``None``.
    canonical_tex : dict
        ``{name: name in LaTeX}``, the paper's typography, for a plot.
    irreptables_named : dict or None
        The same class in ``irreptables``' named indicators (``z4``, ``z2w,1``,
        ...), its formulas carried into the labels returned here;
        :meth:`as_irreptables` respells :attr:`canonical` in these names.
        ``None`` for a unitary group, and where ``irreptables`` holds no formula
        for the group that can be read here; :attr:`irreptables_note` says why.
    irreptables_note : str
        Where :attr:`irreptables_named` came from, or why it is ``None``.
    irreptables_convention_touched : tuple of str
        The named indicators whose formula the change of convention moved.
    content : dict
        The symmetry vector, ``{corep: multiplicity}``.
    reconstructed : dict
        ``{corep: corep}``: coreps the model does not measure, each given the
        multiplicity of a measured one it repeats in every elementary band
        representation.
    two_dimensional : dict or None
        For a layer, how its class reads: the class of its decoupled stack, the
        layer repeated along its normal with no hopping between copies, every
        component an invariant of that stack.  ``out_of_plane_points`` and
        ``out_of_plane_labels`` name the coreps at ``k_z = ½``; ``components``
        says which Smith coordinates read them, and ``irreptables_support``
        where each named indicator's coefficients sit; ``note`` says it in a
        sentence.  ``None`` for a crystal and for a group with no indicator.
    completion : dict or None
        How a layer's coreps at ``k_z = ½`` that repeat none in its plane were
        measured, on its decoupled stack in its own group: the ``outcome``, the
        layer's group and the coreps measured (``input``), the ``stack``, the
        ``environment``, ``stack_period`` and ``stack_extent``, and in ``gates``
        each check that ties the stack to the layer with its measured number.
        ``None`` where nothing had to be.
    free_standing_group : dict or None
        For a layer on a substrate, the free-standing layer's group,
        ``{"number_str", "name", "size"}``, as :func:`.groups.realized_group`
        measured it, where it is larger than the group the class lives in; its
        class is asked with ``environment="free-standing"``.
    site_content : dict or None
        From ``rep``: each orbital's site irrep and how it is known.  Where one
        is ``"lowest-l"``, the class is that of the labelling the
        lowest-angular-momentum rule makes, and a declaration states another.
    labels, y, y_prime, slots : tuple
        The arithmetic: every corep of the group's elementary band
        representations, the symmetry vector on them, the same in the Smith
        coordinates, ``u·y``, and where the divisors sit among those.
    restriction : Restriction
        The coreps the model measures.
    lattice_check : dict
        The compatibility relations, checked exactly: ``in_lattice``,
        ``residual``, and how many ``relations`` the group has.
    verification : dict
        An independent exact computation of the Smith normal form, which agrees
        or the call raises: the divisors and the class computed again, and
        ``change_of_basis`` between the two coordinates.  ``performed`` is
        ``False`` under ``verify=False``, the tables' form taken as it is.
    convention : str
        :data:`INDICATOR_CONVENTION`: the class does not depend on the
        translation convention the tables are written in, and the named
        indicators are carried into this package's.
    group_provenance : str
        ``"realized"`` or ``"descended"``: the group was measured on ``H``.
    n_bands : int or None
        The number of bands, where the call was told it.
    """

    verdict: str
    number_str: str
    group_name: str
    magnetic_type: str
    divisors: tuple
    values: tuple
    structure: str
    nontrivial: bool
    order: int
    y: tuple
    y_prime: tuple
    labels: tuple
    slots: tuple
    content: dict
    reconstructed: dict
    restriction: Restriction
    irreptables_named: dict | None
    irreptables_note: str
    irreptables_convention_touched: tuple
    two_dimensional: dict | None
    verification: dict
    lattice_check: dict
    completion: dict | None = None
    convention: str = INDICATOR_CONVENTION
    site_content: dict | None = None
    group_provenance: str = "cell"
    n_bands: int | None = None
    canonical: dict | None = None
    canonical_note: str = ""
    canonical_tex: dict = field(default_factory=dict, repr=False)
    free_standing_group: dict | None = None

    @property
    def indicated(self) -> bool:
        """Whether the class is nonzero: no atomic limit has this content."""
        return self.nontrivial

    def as_irreptables(self) -> dict | None:
        """:attr:`canonical` in ``irreptables``' spelling of the same names.

        ``{"z3m,pi": 1}`` comes back as ``{"z3m,π": 1}``, and ``{"delta2+,pi": 1}``
        as ``{"δ2,π+": 1}``; a name ``irreptables`` does not have is left out.
        The values are those of :attr:`irreptables_named`, reached through the
        stable real-space invariants rather than ``irreptables``' own formulas.
        ``None`` where :attr:`canonical` is.
        """
        if self.canonical is None:
            return None

        out = {}
        for name, value in self.canonical.items():
            backend = to_irreptables_name(name)
            if backend is not None:
                out[backend] = value
        return out

    def summary(self) -> str:
        """The class, a line for each part, with the attribute that holds it."""
        group = f"{self.group_name} #{self.number_str}"
        if self.verdict == "indicated nontrivial":
            head = (f"indicated nontrivial, in {group}: no atomic limit has this content, nor "
                    "any with atomic bands added")
        elif self.verdict == "indicator zero":
            head = (f"indicator zero, in {group}: the zero class, which certifies nothing, "
                    "fragile band sets lying in it too")
        else:
            head = f"trivial indicator group, in {group}: there is no class to read"
        lines = [head + "   (.verdict)"]
        if self.divisors:
            lines.append("  the indicator group, the band structures modulo the atomic limits: "
                         f"{self.structure}, a class being {_moduli(self.divisors)}   "
                         "(.structure)")
        if self.canonical:
            lines.append("  canonical: " + ", ".join(f"{k} = {v}" for k, v in
                                                     self.canonical.items())
                         + ", the names of the paper that tabulates the stable real-space "
                         "invariants, Hwang et al. (2026)   (.canonical)")
        if self.divisors:
            lines.append("  indicator: " + ", ".join(
                f"{v} mod {d}" for v, d in zip(self.values, self.divisors))
                + (f", of order {self.order}: {self.order} copies of these bands are in "
                   "the zero class   (.values, .order)" if self.order > 1
                   else "   (.values)"))
        if self.irreptables_named:
            lines.append("  irreptables: " + ", ".join(f"{k} = {v}" for k, v in
                                                       self.irreptables_named.items())
                         + ", the same indicators as the irreptables package names "
                         "them   (.irreptables_named)")
        lines.append("  the coreps at the high-symmetry points: "
                     f"{_sewing.format_counts(self.content)}   (.content)")
        if self.reconstructed:
            points = sorted({kpoint_of_label(label) for label in self.reconstructed})
            lines.append(f"  the coreps at {_text._listed(points)} are not measured: each "
                         "occurs in every elementary band representation as often as a "
                         "measured one, and takes its multiplicity   (.reconstructed)")
            lines.append("    " + ", ".join(f"{label} as {source}" for label, source
                                            in self.reconstructed.items()))
        stack_named = False
        comp = self.completion
        if comp and comp.get("outcome") == "completed":
            points = sorted({kpoint_of_label(label) for label in comp["input"]["orphans"]})
            lines.append(f"  {_text._listed(points)}, at k_z = ½ out of the layer's plane: "
                         "their coreps measured on the layer's decoupled stack, the layer "
                         "repeated along its normal with no hopping between copies, in the "
                         f"layer's own group #{comp['input']['number_str']}   (.completion)")
            stack_named = True
        elif comp:
            lines.append(f"  {str(comp['stop_reason']).rstrip('. ')}; the class above is in "
                         f"the group handed in, #{comp['input']['number_str']}   (.completion)")
        if self.two_dimensional:
            lines.append("  the class of the layer's decoupled stack"
                         + ("" if stack_named else ", the layer repeated along its normal "
                            "with no hopping between copies")
                         + ": every component is an invariant of that stack, fixed by the "
                         "layer's own content   (.two_dimensional)")
        if self.free_standing_group:
            free = self.free_standing_group
            lines.append(
                "  the free-standing layer's group, as realized_group measured it, is "
                f"{free['name']} #{free['number_str']}, larger than this one: the same layer "
                "with vacuum on both sides, whose class is asked with "
                "environment='free-standing'   (.free_standing_group)")
        if self.divisors and "lowest-l" in tuple((self.site_content or {}).get(
                "determined_by", ())):
            lines.append("  a site irrep was assigned by the lowest-angular-momentum rule: "
                         "another declared content can call these bands another "
                         "class   (.site_content)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        body = ", ".join(f"{v}|{d}" for v, d in zip(self.values, self.divisors))
        # the group the class lives in, and, where a layer's rows were measured
        # on its decoupled stack, that they were (see `completion`)
        return (f"SymmetryIndicator({self.verdict}, #{self.number_str}, {self.structure}"
                + (f", [{body}]" if self.divisors else "")
                + (f", {self.completion['outcome']} on the stack" if self.completion
                   and self.completion.get("outcome") != "stopped" else "") + ")")

# --------------------------------------------------------------------------
# content plumbing
# --------------------------------------------------------------------------

def content_of(content) -> dict | Refusal:
    """A band set's content as counts: how many of each corep it carries.

    A measured table and a content written out by hand read the same way, so
    every call here takes either.

    Parameters
    ----------
    content : .sewing.HSPTable or mapping
        A table :func:`.sewing.hsp_table` returned, or ``{corep: multiplicity}``.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``, the multiplicities integers and the zeros
        dropped.  A ``Refusal`` in its place is falsy and names what stopped the
        call.

    Raises
    ------
    TypeError
        Neither a table nor a mapping.

    Refusals
    --------
    ``non_integer_multiplicity``
        an entry is no whole number of coreps.
    ``negative_multiplicity``
        an entry is negative: no band set has it, and a difference of two band
        sets is one only where one contains the other.
    """
    counts = getattr(content, "counts", None)
    src = counts if counts is not None else content
    if not isinstance(src, Mapping):
        raise TypeError("content must be an HSPTable or a {label: multiplicity} mapping, "
                        f"got {type(content).__name__}")
    out = {}
    for lab, v in src.items():
        iv = int(round(float(v)))
        if abs(float(v) - iv) > 1e-9:
            return Refusal("non_integer_multiplicity",
                           f"content['{lab}'] = {v} is not an integer multiplicity",
                           value=float(v), context={"label": str(lab)})
        if iv < 0:
            return Refusal("negative_multiplicity",
                           f"content['{lab}'] = {iv} is negative; a symmetry vector "
                           "counts coreps and cannot be negative. If this is a "
                           "difference of two band sets, that difference is not a "
                           "band set.", value=float(iv), threshold=0.0,
                           context={"label": str(lab)})
        if iv:
            out[str(lab)] = iv
    return out

# --------------------------------------------------------------------------
# the in-plane restriction
# --------------------------------------------------------------------------

def in_plane_restriction(group, *, points=None, require_exact=True,
                         _layer=None) -> Restriction | Refusal:
    """Which coreps of the group's elementary band representations a layer measures, and whether each of the others repeats a measured one.

    A layer's ``H(k)`` has no ``k_z``, so its coreps at ``k_z = ½`` are not
    measured.  Each that occurs in every elementary band representation as
    often as a measured one adds no condition, and where every one does, a
    verdict on the measured coreps is exact.

    Parameters
    ----------
    group : .groups.GroupData
        The group, ``r.group``.
    points : sequence of str, optional
        The points the model samples.  Default: every tabulated point in its
        plane.
    require_exact : bool, optional
        Refuse where a corep left out repeats none measured; ``False`` returns
        those as :attr:`Restriction.orphans`.  Default ``True``.

    Returns
    -------
    Restriction or Refusal
        ``r.summary()`` prints it; ``r.copies`` pairs each corep left out with
        the measured ones it repeats, and ``r.exact`` says whether each does.
        A ``Refusal`` in its place is falsy and names what stopped the call.

    Refusals
    --------
    ``restriction_not_exact``
        a corep left out repeats none measured, with ``require_exact``.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        no elementary band representations of this group, or no tables in this
        cell.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    data = _tables.ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels = list(data["labels_physics_convention"])
    matrix = np.asarray(data["matrix"], dtype=int)
    # A LAYER'S OWN BAND REPRESENTATIONS: where a decomposition runs over the
    # positions within the layer only, the rows are compared on those columns
    if _layer is not None:
        matrix = matrix[:, list(_layer["columns"])]

    if points is None:
        hsp = _tables.high_symmetry_points(group)
        if isinstance(hsp, Refusal):
            return hsp
        points = tuple(hsp)
    points = tuple(str(p) for p in points)

    keep, drop = [], []
    for i, lab in enumerate(labels):
        (keep if kpoint_of_label(lab) in points else drop).append(i)

    copies, missing = {}, []
    for i in drop:
        same = [labels[j] for j in keep if np.array_equal(matrix[i], matrix[j])]
        copies[labels[i]] = tuple(same)
        if not same:
            missing.append(labels[i])
    # ON A LAYER'S OWN BAND REPRESENTATIONS every row out of its plane repeats
    # one in it: the layer's operations carry no translation along its normal
    # and keep the positions within it within it.  That is a statement about
    # the whole plane, so it is checked where every point of it was compared
    if _layer is not None and missing:
        hsp = _tables.high_symmetry_points(group)
        whole = not isinstance(hsp, Refusal) and set(hsp) <= set(points)
        off_plane = set(hsp.missing) if whole else set()
        unrepeated = [label for label in missing if kpoint_of_label(label) in off_plane]
        if unrepeated:
            raise InternalError(
                f"#{group.number_str}: on the layer's own band representations the rows "
                f"{unrepeated}, out of its plane, repeat no row in it. The layer's "
                "operations carry no translation along its normal and keep the positions "
                "within it within it, so every such row of a band representation induced "
                "there repeats one in the plane; that one does not is this package "
                "disagreeing with itself, not a verdict on the layer.")
    if missing and require_exact:
        return Refusal(
            "restriction_not_exact",
            f"the EBR rows {missing} cannot be measured by this model (their k-points "
            f"are out of the periodic plane) and duplicate no retained row, so they "
            f"impose independent constraints, and in-plane data alone reach no verdict "
            f"for group #{group.number_str}.",
            value=float(len(missing)), threshold=0.0,
            context={"missing": tuple(missing), "retained": tuple(labels[j] for j in keep),
                     "points": points, "number_str": group.number_str})

    return Restriction(
        applied=bool(drop), points=points,
        retained=tuple(labels[j] for j in keep), dropped=tuple(labels[i] for i in drop),
        copies=copies, exact=not missing,
        rank=int(np.linalg.matrix_rank(matrix[keep])) if keep else 0,
        full_rank=int(np.linalg.matrix_rank(matrix)),
        orphans=tuple(missing), require_exact=bool(require_exact),
        layer_extent=None if _layer is None else tuple(_layer["extent"]),
        outside_layer={} if _layer is None else dict(_layer["outside"]))

def _layer_reading(group, data, rep=None):
    r"""The band representations a layer's atomic limits are written over, or ``None`` for a crystal.

    A free-standing layer's group is read in a cell periodic along the layer's
    normal, and some of its Wyckoff positions sit at a fixed height off the
    layer, half the cell's length along the normal from its middle plane.  No
    orbital of the layer is there, so the layer's own band representations are
    those induced from the positions within it: each free to move along the
    normal, or at a height between the lowest and the highest of the layer's
    orbitals.  On those every row at ``k_z = ½`` repeats one in the plane,
    which is what makes a decomposition read in the plane alone exact.  On a
    substrate every position is free to move along the normal, and every one
    is kept.

    Heights are measured along the normal ``a1 × a2``, from the layer's middle
    plane: each position's varieties written in the cell the group is read in
    (``GroupData._refUC`` and ``_shiftUC``) against the orbitals --
    ``rep``'s model's, or the group's own sites without one -- with the
    symmetry finder's distance as the margin.  Returns ``{"columns",
    "outside", "extent"}``: the indices of ``data``'s elementary band
    representations within the layer, ``{position: heights}`` for the positions
    outside it, and the orbitals' ``(lowest, highest)`` heights.  ``None`` for
    a crystal periodic in three directions, and where the tables' Wyckoff
    letters cannot be read in the group's cell: every position is then kept,
    and the rows say whether the plane suffices.
    """
    padded = tuple(int(d) for d in (getattr(group, "vacuum_dirs", ()) or ()))
    if int(getattr(group, "dim_k", 3)) >= 3 or len(padded) != 1:
        return None
    if isinstance(_tables.wyckoff_sites_tables(group), Refusal):
        return None              # the letters are keys only once the settings agree
    normal_dir = padded[0]
    lattice = np.asarray(group.lattice, dtype=float)
    plane = [d for d in range(3) if d != normal_dir]
    normal = np.cross(lattice[plane[0]], lattice[plane[1]])
    scale = float(lattice[normal_dir] @ normal) / float(np.linalg.norm(normal))

    def height(z):
        """A reduced coordinate along the vacuum direction as a height from the middle plane."""
        return ((float(z) + 0.5) % 1.0 - 0.5) * scale

    model = getattr(rep, "model", None)
    if model is not None:

        where = _orb_positions_3d(model, group)
    else:
        where = np.asarray(group.site_positions, dtype=float)
    own = [height(z) for z in np.atleast_2d(where)[:, normal_dir]]
    lowest, highest = min(own), max(own)
    margin = float(group.symprec)
    refuc = np.asarray(group._refUC, dtype=float)
    shift = np.asarray(group._shiftUC, dtype=float)
    arms_of = {p["letter"]: p["arms"] for p in _spacegroup._pyxtal_wyckoff(int(group.sg_number))[0]}
    columns, outside = [], {}
    for j, (_name, position) in enumerate(data["ebrs"]):
        arms = arms_of[_letter_only(position)]
        free = any(float(np.abs((refuc @ A)[normal_dir]).max()) > _INT_TOL
                   for A, _b, _P in arms)
        heights = tuple(sorted({round(height((refuc @ b + shift)[normal_dir]), 9)
                                for _A, b, _P in arms}))
        if free or all(lowest - margin <= h <= highest + margin for h in heights):
            columns.append(j)
        else:
            outside[_wyckoff_letter(position)] = heights
    return {"columns": tuple(columns), "outside": outside, "extent": (lowest, highest)}

# --------------------------------------------------------------------------
# the integer solve
# --------------------------------------------------------------------------

_NO_BLOCK = object()

def _same_group(a, b) -> bool:
    """Whether two group records are one group written in one cell.

    The same entry of the tables, the same lattice and the same operations,
    each rotation exactly and each translation modulo the lattice to rounding:
    two records of one measurement are one group, and a group written in
    another cell, or another group, is not.
    """
    if a is b:
        return True
    if ((str(a.number_str), bool(a.spinor), bool(a.magnetic))
            != (str(b.number_str), bool(b.spinor), bool(b.magnetic))):
        return False
    if len(a.symmetries) != len(b.symmetries):
        return False
    if not np.allclose(np.asarray(a.lattice, dtype=float), np.asarray(b.lattice, dtype=float)):
        return False

    def keyed(symmetries):
        out = set()
        for sym in symmetries:
            rot = tuple(int(x) for x in np.asarray(sym.rotation).reshape(-1))
            trans = np.asarray(sym.translation, dtype=float)
            trans = trans - np.floor(trans + ROUNDING)
            out.add((rot, tuple(int(x) for x in np.round(trans / ROUNDING))))
        return out

    return keyed(a.symmetries) == keyed(b.symmetries)

def _same_rep(a, b) -> bool:
    """Whether two representations are one: the same group, the same operations and the same matrices."""
    if a is b:
        return True
    if not _same_group(a.group, b.group):
        return False
    if tuple(a.indices) != tuple(b.indices) or tuple(a.antiunitary) != tuple(b.antiunitary):
        return False
    return all(np.allclose(a.matrices[i], b.matrices[i]) for i in a.indices)

def _table_binding(content, group, rep, *, where):
    """A table is read in one group with one representation, and is read with no other.

    An :class:`~.sewing.HSPTable` carries the representation it was read with
    on ``rep``; a group handed in beside it that is not that representation's,
    or a ``rep=`` that is not it, is a contradiction only the call can resolve,
    as ``bands=`` naming other bands is, and is raised.  A content written out
    by hand carries nothing and takes its group as given.
    """
    own = getattr(content, "rep", None) if hasattr(content, "counts") else None
    if own is None:
        return
    if not _same_group(group, own.group):
        raise ValueError(
            f"{where}(): the table was read in {own.group.name} #{own.group.number_str}, "
            f"and group is {group.name} #{group.number_str}. A table's labels are the "
            "coreps of the group it was read in, so there is nothing to read in another "
            "group; the table carries its own as t.rep.group, and hsp_table(rep, bands=...) "
            "on a representation of the other group reads the bands in that one.")
    if rep is not None and not _same_rep(rep, own):
        raise ValueError(
            f"{where}(): rep= is not the representation the table was read with, which "
            "the table carries as t.rep: its group, its operations or its matrices differ. "
            "The labels were read off t.rep, so another representation says nothing about "
            "them; pass t.rep, or read the bands again with hsp_table(rep, bands=...).")

def _table_bands(table):
    """The band block an :class:`~.sewing.HSPTable` was handed, as it was handed in."""
    block = getattr(table, "bands", _NO_BLOCK)
    return getattr(table, "occ", None) if block is _NO_BLOCK else block

def _table_nstate(table):
    """How many states the fiber a table was measured on carries, if it says.

    Every resolved point kept the full spectrum of its own diagonalization, so
    any one of them is the state count the band block is read against.
    ``None`` when nothing resolved.
    """
    for record in (getattr(table, "records", None) or {}).values():
        spectrum = getattr(getattr(record, "table", None), "spectrum", None)
        if spectrum is not None and len(spectrum):
            return int(len(spectrum))
    return None

def _as_block(bands, nstate):
    """``bands`` as sorted band indices, or ``None`` when it names none.

    An ``int`` is the lowest *n* bands and a sequence is exactly those indices —
    :func:`.sewing._band_block`'s rule, and it is the one normalizer.  With
    no state count in hand the same rule is applied without its bounds check.
    """
    if bands is None:
        return None
    if nstate is not None:
        return _sewing._band_block(int(nstate), bands)
    if np.isscalar(bands):
        return tuple(range(int(bands)))
    return tuple(sorted(int(b) for b in bands))

def _bands_of(content, bands, *, where):
    """Which bands a call is about, table and declaration agreed.

    With an :class:`~.sewing.HSPTable` the table's own block is the truth — it
    is the block the content was measured on — so ``bands=`` naming the same
    set is accepted and one naming a different set is a contradiction only the
    call can resolve.  With a plain mapping there is nothing to measure
    against and ``bands=`` states which bands the content counts.  Sorted band
    indices come back, or ``None`` when neither names a block.
    """
    table = content if hasattr(content, "counts") else None
    if table is None:
        return _as_block(bands, None)
    nstate = _table_nstate(table)
    handed = _table_bands(table)
    own = (tuple(range(nstate)) if handed is None and nstate is not None
           else _as_block(handed, nstate))
    if bands is None:
        return own
    asked = _as_block(bands, nstate)
    if own is not None and set(own) != set(asked):
        raise ValueError(
            f"{where}(): bands={list(asked)} names a different set of bands than the "
            f"{len(own)} the content was measured on, {list(own)}. The table is a "
            "measurement and `bands=` is an assertion about which bands it counts; "
            "they disagree, so there is nothing to report on.")
    return own if own is not None else asked

def _band_count_rows(group, table, data, block, counts, resolved):
    """``(counts, resolved)`` with the points a table skipped read as the band count.

    A point whose little group is the identity alone has one irrep, of
    dimension one, so a band set's content there is its number of bands, and
    :func:`.sewing.hsp_table` reads no characters at it.  The tables' one row
    there is each elementary band representation's own band count, which no
    other point's single row repeats; read as the block's band count it is a
    row like any other, and the decomposition and the indicator keep it.
    """
    skipped = tuple(getattr(table, "skipped", {}) or ()) if table is not None else ()
    if not skipped or block is None:
        return counts, resolved
    labels = list(data["labels_physics_convention"])
    matrix = np.asarray(data["matrix"], dtype=int)
    spanned = np.array([_bands_spanned(name) for name, _at in data["ebrs"]], dtype=int)
    counts = dict(counts)
    for name in skipped:
        here = [i for i, label in enumerate(labels) if kpoint_of_label(label) == name]
        if len(here) != 1 or not np.array_equal(matrix[here[0]], spanned):
            raise InternalError(
                f"#{group.number_str}: the little group at {name} is the identity alone, "
                f"yet the tables' rows there ({[labels[i] for i in here]}) are not one row "
                "holding each elementary band representation's band count. A point with "
                "no operation but the identity has one irrep of dimension one; rows that "
                "say otherwise are not this reading's to reconcile.")
        counts.setdefault(labels[here[0]], len(block))
    if resolved is not None:
        resolved = tuple(resolved) + tuple(name for name in skipped if name not in resolved)
    return counts, resolved

def _resolved_points(table, points, *, where):
    """Which k-point names the restriction runs over, table and declaration agreed.

    A table has *measured* which points it resolved; ``points=`` is an
    assertion of which points the model may be asked about.  With both
    in hand they have to name the same set — an assertion the measurement
    contradicts states something about the measurement that the measurement
    denies, and neither one may silently win.
    """
    resolved = None if table is None else tuple(table.content)
    if points is None:
        return resolved
    asked = tuple(str(p) for p in points)
    if resolved is None:
        return asked
    if set(asked) != set(resolved):
        raise ValueError(
            f"{where}(): points={sorted(set(asked))} names a different set of k-points "
            f"than the content resolved, {sorted(set(resolved))}. The table is a "
            "measurement and `points=` is an assertion about what the model can be "
            "asked; they disagree, so there is nothing to restrict to.")
    return resolved

def _cpsat(matrix, y, lo, hi, cap):
    """Enumerate ``{x integer, lo ≤ x ≤ hi : M x = y}`` — CP-SAT, status reported.

    ``irrep``'s own integer solve (``compute_ebr_decomposition``) is the same
    machinery, but it neither takes a restricted matrix nor reports *why* it
    found nothing, and telling ``INFEASIBLE`` from "hit the bound" is the whole
    point of a "proof of none".  So the model is built here, over whatever
    matrix the restriction produced, and the solver's status enum is returned
    with the solutions.  (Building it here also sidesteps the ortools-9.15
    ``status_name`` breakage ``_tables._ebr_solve_shim`` works around.)
    """
    from ortools.sat.python import cp_model

    n_row, n_col = matrix.shape
    model = cp_model.CpModel()
    x = [model.NewIntVar(int(lo), int(hi), f"x{j}") for j in range(n_col)]
    for i in range(n_row):
        model.Add(sum(int(matrix[i, j]) * x[j] for j in range(n_col)) == int(y[i]))

    found: list = []

    class _Collect(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.truncated = False

        def on_solution_callback(self):
            found.append(tuple(int(self.Value(v)) for v in x))
            if len(found) >= cap:
                self.truncated = True
                self.StopSearch()

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    collector = _Collect()
    status = solver.Solve(model, collector)
    names = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
             cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
             cp_model.UNKNOWN: "UNKNOWN"}
    return (names.get(status, str(status)),
            tuple(sorted(found, key=lambda t: (sum(abs(v) for v in t), t))),
            collector.truncated)

def _content_system(data, counts, retained):
    """``(matrix, y, bound)``: the EBR matrix and the content on the rows ``retained``.

    ``bound`` is what each multiplicity is enumerated up to, the content's
    largest entry: a rigorous bound wherever every column has a positive entry
    among those rows (:attr:`EBRDecomposition.bound_is_rigorous`).
    """
    labels = list(data["labels_physics_convention"])
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.asarray(data["matrix"], dtype=int)[[index[label] for label in retained]]
    y = np.array([int(counts.get(label, 0)) for label in retained], dtype=int)
    return matrix, y, int(max(int(y.max()) if y.size else 0, 1))

def _lifted(solutions, columns, n) -> tuple:
    """Solutions over the band representations ``columns`` as multiplicities over all ``n``."""
    out = []
    for solution in solutions:
        vector = [0] * int(n)
        for j, multiplicity in zip(columns, solution):
            vector[int(j)] = int(multiplicity)
        out.append(tuple(vector))
    return tuple(out)

def _ordered(solutions) -> tuple:
    """Sums of elementary band representations, the fewest in all first, then in the tables' order.

    Of two sums with as many elementary band representations in all, the one
    with more of the first the tables list comes first, so a sum at ``1a``
    reads before the same sum at ``1b``.
    """
    return tuple(sorted(solutions, key=lambda s: (sum(abs(int(m)) for m in s),
                                                  tuple(-int(m) for m in s))))

def _terms(solutions, ebrs):
    out = []
    for s in solutions:
        out.append(tuple((int(m), str(ebrs[j][0]), str(ebrs[j][1]))
                         for j, m in enumerate(s) if m))
    return tuple(out)

def _site_content_record(rep):
    """The site content a record's labels rest on, or ``None`` without a ``rep``."""
    if rep is None:
        return None
    ch = getattr(rep, "characters", None)
    if ch is None:
        return None
    return {"number_str": str(rep.group.number_str),
            "irreps": tuple(str(x) for x in ch.irreps),
            "determined_by": tuple(str(o.determined_by) for o in ch.orbitals),
            "residual": float(ch.residual)}

# --------------------------------------------------------------------------
# the public solvers
# --------------------------------------------------------------------------

def decompose(content, group, *, rep=None, points=None, allow_incomplete=False,
              cap=DEFAULT_ENUMERATION_CAP, signed_combinations=True,
              bands=None) -> EBRDecomposition | Refusal:
    r"""Every sum of elementary band representations, with nonnegative integer multiplicities, that carries a band set's content.

    An atomic limit's content is such a sum, so a band set with none is no
    atomic limit; one with a sum may still not be one, by an obstruction
    beyond what symmetry indicators see.  A layer is decomposed over the band
    representations of the Wyckoff positions within it.

    Parameters
    ----------
    content : .sewing.HSPTable or mapping
        The band set's content: a table :func:`.sewing.hsp_table` returned,
        which must be complete, or ``{corep: multiplicity}`` written out.
    group : .groups.GroupData
        The group the content was read in, ``r.group``.
    rep : .symmetry.SpaceGroupRep, optional
        The representation the content was read with, ``r.rep``: it says how
        each site irrep is known, and on a layer which Wyckoff positions lie
        within it.  Pass it whenever it is at hand.
    bands : int or sequence of int, optional
        Which bands the content counts: ``n`` the lowest ``n``, or the band
        indices.  Default: a table's own.
    points : sequence of str, optional
        The high-symmetry points the model samples.  Default: a table's own, or
        every tabulated point in the model's plane.
    allow_incomplete : bool, optional
        Decompose a table that refused at some points, leaving their coreps
        out: for a point whose tables cannot be read, never one where the
        physics refused.  The verdict is then a necessary condition only.
        Default ``False``.
    cap : int, optional
        The most sums listed.  Default 200.
    signed_combinations : bool, optional
        For a ``"none"``, search again with negative multiplicities allowed: a
        combination found is the signature of a fragile band set.  Default
        ``True``.

    Returns
    -------
    EBRDecomposition or Refusal
        ``d.summary()`` prints the verdict and each sum.  ``d.verdict`` is
        ``"symmetry-indicated decomposable"`` or ``"none"``;
        ``d.solution_strings()`` the sums and ``d.positions`` their Wyckoff
        positions, which no labelling of the orbitals moves; ``d.certificate``
        what a ``"none"`` rests on, and ``d.signed_combinations`` the search
        with negative multiplicities; ``d.caveat`` what a sum does not certify.
        The record is truthy, a ``"none"`` included; a ``Refusal`` in its place
        is falsy and names what stopped the call.

    Raises
    ------
    ValueError
        ``bands=`` or ``points=`` naming other bands or points than the table
        measured; or a group, or ``rep=``, that is not the one the table was
        read in, which it carries as ``t.rep``.
    ..refusal.InternalError
        On a layer's own band representations a corep out of its plane repeats
        none in it: the package disagreeing with itself.

    Refusals
    --------
    ``incomplete_content``
        the table refused at some point, and ``allow_incomplete`` is not set.
    ``not_a_band_structure``
        the content fails the group's compatibility relations: no set of bands
        apart from the rest at every k has it.
    ``unknown_label``
        a corep the group's elementary band representations do not hold.
    ``restriction_not_exact``
        a corep the model does not measure repeats none it does.
    ``non_integer_multiplicity`` / ``negative_multiplicity``
        the content is no count of coreps.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        no elementary band representations of this group to decompose against.

    Examples
    --------
    lin3g's six lowest bands (:func:`.models.lin3g`):

    >>> d = decompose(hsp_table(r.rep, bands=6), r.group, rep=r.rep)  # doctest: +SKIP
    >>> d.verdict, d.positions                                        # doctest: +SKIP
    ('symmetry-indicated decomposable', (('1b', '2c'), ('1a', '1a', '1b'), ('1b', '1b', '1b')))
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    counts = content_of(content)
    if isinstance(counts, Refusal):
        return counts

    table = content if hasattr(content, "counts") else None
    _table_binding(content, group, rep, where="decompose")
    block = _bands_of(content, bands, where="decompose")
    skipped = dict(getattr(table, "skipped", {}) or {}) if table is not None else {}
    refused = ({name: r.gate for name, r in (table.refusals or {}).items()}
               if table is not None else {})
    if table is not None and not table.complete and not allow_incomplete:
        return Refusal(
            "incomplete_content",
            "the measured content is missing "
            f"{sorted(refused)}: {refused}. A missing constraint can only make a "
            "decomposition easier to find, so a partial symmetry vector is not "
            "decomposed. Read the table's refusals; pass allow_incomplete=True only for "
            "a point whose tables cannot be read rather than one where the physics "
            "failed, and then read the verdict as a necessary condition.",
            value=float(len(refused)), threshold=0.0,
            context={"refusals": refused, "skipped": tuple(skipped),
                     "resolved": tuple(getattr(table, "content", {}))})

    data = _tables.ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels_all = list(data["labels_physics_convention"])
    ebrs = tuple((str(n), str(w)) for n, w in data["ebrs"])

    unknown = sorted(set(counts) - set(labels_all))
    if unknown:
        return Refusal(
            "unknown_label",
            f"the measured labels {unknown} are not in the EBR basis of group "
            f"#{group.number_str}; the symmetry vector and the basis are not written in "
            "the same alphabet.",
            value=float(len(unknown)), threshold=0.0,
            context={"unknown": tuple(unknown), "basis": tuple(labels_all)})

    # Only the rows that were measured may constrain the solve.  Dropping a
    # measurement while keeping its equation would *over*-constrain the problem
    # (the unmeasured multiplicities would be forced to zero), which is the
    # opposite of the weaker claim the override is supposed to buy — so a point
    # that refused loses its EBR rows too.  A point whose little group is the
    # identity alone keeps its one row, which holds the band count.
    resolved = _resolved_points(table, points, where="decompose")
    counts, resolved = _band_count_rows(group, table, data, block, counts, resolved)
    exact_needed = table is None or table.complete
    # A LAYER'S OWN BAND REPRESENTATIONS.  A free-standing layer's group has
    # positions off the layer, which carry none of its orbitals: their band
    # representations are left out before anything is solved, so that a capped
    # enumeration lists the layer's own sums, and on the rest every row at
    # k_z = 1/2 repeats one in the plane.
    layer = _layer_reading(group, data, rep)
    restriction = in_plane_restriction(group, points=resolved, require_exact=exact_needed,
                                       _layer=layer)
    if isinstance(restriction, Refusal):
        return restriction
    columns = list(range(len(ebrs))) if layer is None else list(layer["columns"])

    # THE COMPATIBILITY RELATIONS, read before anything is solved: a content no
    # band structure has -- a block that cuts through a crossing between two
    # high-symmetry points -- has no decomposition to look for, and the
    # indicator's own check says so.  Read on the vector the dropped rows
    # complete, which an exact restriction determines.
    if restriction.exact:
        y_all, _sources = _filled_content(counts, restriction, labels_all)
        _smith, _y_prime, wrong = _compatibility(data, group.number_str, labels_all, y_all)
        if wrong is not None:
            return wrong

    matrix, y, bound = _content_system(data, counts, restriction.retained)
    matrix = matrix[:, columns]

    # a rigorous bound: M >= 0 and every column has a positive retained entry
    col_max = matrix.max(axis=0) if matrix.size else np.zeros(len(columns), dtype=int)
    rigorous = bool((matrix >= 0).all() and (y >= 0).all() and (col_max > 0).all())

    status, solutions, truncated = _cpsat(matrix, y, 0, bound, cap)
    solutions = _ordered(_lifted(solutions, columns, len(ebrs)))
    decomposable = bool(solutions)

    probe = None
    if not decomposable and signed_combinations:
        p_status, p_sol, p_trunc = _cpsat(matrix, y, -bound, bound, cap)
        if p_status not in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"):  # pragma: no cover
            # no limit is set, so the search runs to its end: it finds solutions,
            # or proves there are none within the bound
            raise InternalError(
                f"the search with negative multiplicities allowed ended {p_status}, "
                "where with no limit set it finds solutions or proves there are none "
                "within the bound: the package disagreeing with itself.")
        p_sol = _lifted(p_sol, columns, len(ebrs))
        probe = {"n_solutions": len(p_sol), "solutions": p_sol,
                 "terms": _terms(p_sol, ebrs), "bound": bound,
                 "bound_is_rigorous": False, "truncated": p_trunc,
                 "note": ("an integer solution that is not a nonnegative one is the "
                          "arithmetic signature of fragility; the bound is not rigorous "
                          "here because signed terms can cancel without limit")}

    certificate = None
    if not decomposable:
        certificate = ("proof" if (status == "INFEASIBLE" and rigorous)
                       else "none within the bound")

    claim = "exact" if restriction.exact else "necessary-condition"
    note = "every tabulated point in the periodic plane resolved"
    if table is not None and not table.complete:
        note = (f"points {sorted(refused)} were dropped under allow_incomplete=True, and "
                f"their EBR rows {list(restriction.orphans)} with them")
        if restriction.exact:
            note += (" -- but every dropped row duplicates a retained one, so the verdict "
                     "is still exact")
        else:
            note += (" -- those rows are independent constraints, so a "
                     "'symmetry-indicated decomposable' verdict here is a necessary "
                     "condition only")
    elif table is None:
        note = ("content supplied as a plain mapping: its completeness is asserted, "
                "not measured")

    return EBRDecomposition(
        verdict="symmetry-indicated decomposable" if decomposable else "none",
        content=dict(counts), labels=restriction.retained,
        y=tuple(int(v) for v in y), ebrs=ebrs,
        solutions=solutions, terms=_terms(solutions, ebrs),
        positions=tuple(tuple(sorted(_wyckoff_letter(ebrs[j][1])
                                     for j, m in enumerate(s) for _ in range(m)))
                        for s in solutions),
        bound=bound, bound_is_rigorous=rigorous,
        certificate=certificate,
        enumeration={"cap": int(cap), "n_solutions": len(solutions),
                     "truncated": bool(truncated)},
        signed_combinations=probe, restriction=restriction,
        site_content=_site_content_record(rep),
        completeness={"source": "HSPTable" if table is not None else "mapping",
                      "complete": bool(getattr(table, "complete", True)),
                      "claim": claim, "skipped": tuple(skipped),
                      "refused": dict(refused), "note": note},
        caveat=WannierCaveat(),
        n_bands=None if block is None else len(block),
        _status=status)

#: ``"I"`` (unitary), ``"II"`` (grey), ``"III"`` or ``"IV"``, measured from the
#: operations rather than parsed out of a BNS number.  One detector serves the
#: whole package: :func:`.._spacegroup._group_of` gates on it at construction, and
#: :func:`symmetry_indicator` re-checks it here.
_magnetic_type = _spacegroup._magnetic_type

@lru_cache(maxsize=None)
def _indicator_formulas(spinor: bool, magnetic: bool) -> dict:
    """``irreptables``' named indicator formulas for every group of one table, read once.

    ``spinor`` picks the double-valued table and ``magnetic`` the one keyed by
    the magnetic group's number (``166.98``).  Every indicator of a group with
    time reversal looks its formulas up here, and the double-valued file is
    large, so it is read the first time and kept; the formulas are only read,
    never changed.
    """
    from irreptables.symmetry_indicators import load_si_table

    return load_si_table(spinor=spinor, magnetic=magnetic)

def _named_forms(group, data, matrix, y, magnetic_type):
    """``irreptables``' named indicator forms for the group, in this package's convention.

    Returns ``(values, touched, note, coefficients)`` — the literature names
    (``z4``, ``z2w,i``, ``η4I``, …) for the same classes the Smith form counts,
    with the re-keyed coefficient vector of each form, or ``None`` and the
    reason.  The coefficients are what :func:`_two_dimensional_reading` reads a
    form's *support* off.

    Two things have to be true before a shipped form is returned, and both are
    *checked*, not assumed.

    **It has to be an indicator of this group.**  A linear form is one exactly
    when it vanishes, modulo its own modulus, on every EBR — that is what makes
    it descend to ``{BS}/{AI}`` — and that is tested here on the group's own
    matrix.  It is worth testing because the tables are not what their file
    names suggest: ``single_indicators.json`` and ``double_indicators.json`` are
    the **grey (type-II) tables keyed by the bare
    space-group number**, byte for byte, so their ``factors`` are grey corep
    labels (``A1A2``, ``GM2+GM3+``) and only 39 of 70 / 128 of 320 of them even
    vanish on the unitary group's own EBRs.  Nothing is returned for a unitary
    group rather than a partial sum over labels that never match.

    **It has to be read on the right coordinates.**  The shipped coefficients
    are keyed by BCS labels, and the transport permutes the row *positions*
    (``y_BCS[j] = y_phys[P[j]]``), so the coefficient belonging on the returned
    entry ``i`` is the shipped one at position ``P[i]``.  In scope that changes
    exactly one form — grey ``220.90``'s ``z2``, whose coefficients sit on the
    ``P``/``PA`` labels the transport swaps — and the ones it changes are
    reported in ``touched``.
    """
    if magnetic_type != "II":
        return None, (), (
            "irreptables' non-magnetic indicator files are the grey (type-II) tables "
            "keyed by the bare space-group number -- their factors are grey corep "
            "labels and most of them do not even vanish on this group's own EBRs -- so "
            "no named form is returned for a unitary group. The Smith-form indicator "
            "above is unaffected: it is computed from this group's own EBR matrix"), {}
    try:
        entry = _indicator_formulas(bool(group.spinor), bool(group.magnetic)).get(
            str(group.number_str), {})
        forms = dict(entry.get("indicators", {}))
    except Exception as exc:                   # pragma: no cover - table defect
        return None, (), f"irreptables' indicator table could not be read: {exc}", {}
    if not forms:
        return None, (), ("irreptables ships no named indicator formulas for "
                          f"#{group.number_str}"), {}

    labels_bcs = list(data["convention"].get("labels_bcs", ()))
    perm = list(data["convention"].get("row_permutation", ()))
    if len(labels_bcs) != len(y) or len(perm) != len(y):
        return None, (), "the convention record does not carry the row permutation", {}
    spec_labels = set()
    for spec in forms.values():
        spec_labels |= set(spec["factors"])
    if spec_labels != set(labels_bcs):
        return None, (), (
            f"the shipped formulas are written over {len(spec_labels)} labels that are "
            f"not this group's EBR basis ({len(labels_bcs)} labels)"), {}

    values, touched, coeffs = {}, [], {}
    for name, spec in forms.items():
        coef = [float(spec["factors"].get(lab, 0.0)) for lab in labels_bcs]
        moved = [coef[perm[i]] for i in range(len(coef))]
        if moved != coef:
            touched.append(str(name))
        mod = float(spec["mod"])
        if abs(mod - round(mod)) > 1e-9 or round(mod) < 2:
            return None, (), f"the shipped form {name!r} has a non-integer modulus {mod}", {}
        mod = int(round(mod))
        for col in range(len(matrix[0])):
            on_ebr = sum(c * matrix[i][col] for i, c in enumerate(moved))
            if abs(on_ebr - round(on_ebr)) > 1e-9 or int(round(on_ebr)) % mod:
                return None, (), (
                    f"the shipped form {name!r} does not vanish mod {mod} on EBR column "
                    f"{col} of #{group.number_str} ({on_ebr}), so it is not an indicator "
                    "of this group and no value of it is returned"), {}
        total = sum(c * int(m) for c, m in zip(moved, y))
        if abs(total - round(total)) > 1e-9:
            return None, (), (f"the shipped form {name!r} does not take an integer value "
                              f"on this content ({total})"), {}
        values[str(name)] = int(round(total)) % mod
        coeffs[str(name)] = tuple(moved)
    return values, tuple(touched), ("irreptables' named formulas, re-keyed by the row "
                                    "permutation of the change of convention and "
                                    "checked to vanish on every EBR of this group"), coeffs

def _two_dimensional_reading(group, labels, u, slots, divisors, values, coeffs,
                             plane=None):
    """For a layer: the class read in a three-dimensional group, and its rows out of the plane.

    Returns ``None`` for a fully periodic model, and for a group whose indicator
    is trivial — there is nothing to say in either case — and otherwise the
    record :attr:`SymmetryIndicator.two_dimensional` carries.

    **The statement.**  A 2D model's class is read in a three-dimensional
    space group, its slab's, whose EBR basis carries k-points the model's plane
    does not contain, at ``k_z = ½`` (``A``, ``H``, ``L`` for ``P6/mmm1'``).
    The layer's ``H(k)`` has no ``k_z``, so the class is the class of the
    layer's **decoupled stack** — the layer repeated along its normal with no
    hopping between copies — whose rows at ``k_z = ½`` repeat the in-plane
    ones: whatever content sits at ``Γ`` sits at ``A``, and every component is
    an invariant of that stack, fixed by the layer's own content.

    Support is measured, never assumed: a form is classified by which rows its
    (transported) coefficient vector is nonzero on, and a Smith component by
    which rows its row of ``u`` reads.

    ``plane`` is the description of the model's own periodic plane when the
    arithmetic was done in a *different* presentation's group — the decoupled
    stack, which is fully periodic and so has no out-of-plane point of its own.
    The statement still applies, because the model is still one layer: the points
    are then the stack group's own, split by whether their k lies in the
    layer's plane, and the dimensions reported are the layer's.
    """
    if not divisors:
        return None                            # no class to misread
    if plane is None:
        if int(getattr(group, "dim_k", 3)) >= 3:
            return None                        # a fully periodic model
        hsp = _tables.high_symmetry_points(group)
        if isinstance(hsp, Refusal):
            return None
        in_plane = set(hsp)
        dim_k = int(group.dim_k)
        periodic = tuple(int(p) for p in group.periodic_dirs)
        padded = tuple(int(p) for p in group.vacuum_dirs)
        presentation = "SLAB"
    else:
        in_plane = set(plane["points"])
        dim_k, periodic, padded = plane["dim_k"], plane["periodic_dirs"], plane["vacuum_dirs"]
        presentation = plane["presentation"]
    out = tuple(i for i, lab in enumerate(labels) if kpoint_of_label(lab) not in in_plane)
    if not out:
        return None
    out_points = tuple(sorted({kpoint_of_label(labels[i]) for i in out}))

    components = tuple(
        {"slot": int(s), "divisor": int(d), "value": int(v),
         "reads_out_of_plane": bool(any(u[s][i] for i in out))}
        for s, d, v in zip(slots, divisors, values))
    support = {}
    for name, coef in (coeffs or {}).items():
        here = any(coef[i] for i in range(len(coef)) if i not in out)
        there = any(coef[i] for i in out)
        support[name] = ("mixed" if here and there
                         else "out-of-plane only" if there else "in-plane only")

    cell = "slab" if presentation == "SLAB" else "decoupled stack"
    note = (
        f"The layer's class is read in #{group.number_str}, the three-dimensional space "
        f"group of its {cell} (the model's periodic directions {periodic}, its normal "
        f"along {padded}), whose EBR basis has rows at {', '.join(out_points)}, at "
        "k_z = ½, k-points the layer's plane does not contain. The layer's H(k) has no "
        "k_z, so this is the class of the layer's decoupled stack -- the layer repeated "
        "along its normal with no hopping between copies -- whose rows at k_z = ½ repeat "
        "the in-plane ones: every component is an invariant of that stack, fixed by the "
        "layer's own content.")
    return {"applied": True, "dim_k": int(dim_k),
            "periodic_dirs": tuple(periodic), "vacuum_dirs": tuple(padded),
            "presentation": presentation,
            "out_of_plane_points": out_points,
            "out_of_plane_labels": tuple(labels[i] for i in out),
            "components": components, "irreptables_support": support,
            "note": note}

def _block_characters(record) -> dict:
    """``{ind: tr B(g, k)}`` of a whole band block, from a :class:`.sewing.CorepContent`."""
    rows = np.asarray(record.table.characters, dtype=complex)
    total = rows.sum(axis=0) if rows.size else np.zeros(len(record.table.columns),
                                                        dtype=complex)
    return {int(c): complex(v) for c, v in zip(record.table.columns, total)}

def _same_k(k_a, k_b, *, atol=1e-6) -> bool:
    """Equal reduced k in the plane, modulo a reciprocal lattice vector."""
    d = np.asarray(k_a, dtype=float).reshape(-1)[:2] - np.asarray(
        k_b, dtype=float).reshape(-1)[:2]
    return bool(np.allclose(d, np.round(d), atol=atol))

#: The ``k_z`` at which the decoupled stack's Hamiltonian is held to the layer's.
_KZ_PROBE = (0.0, 0.25, 0.5)

def _stack_record(table, group, rep, restriction, *, c, environment, note) -> dict:
    """The record a decoupled stack's measurement starts from: the layer as handed in."""
    return {
        "route": "decoupled stack", "outcome": "stopped",
        "stopped_at": None, "stop_reason": None, "stack_period": float(c),
        "stack_extent": None, "environment": environment,
        "input": {"number_str": str(group.number_str), "name": str(group.name),
                  "size": int(group.size), "dim_k": int(group.dim_k),
                  "periodic_dirs": tuple(int(p) for p in group.periodic_dirs),
                  "vacuum_dirs": tuple(int(p) for p in group.vacuum_dirs),
                  "points": tuple(sorted(table.points)),
                  "orphans": tuple(restriction.orphans),
                  "content": dict(table.counts),
                  "n_represented": len(rep.indices)},
        "stack": {}, "gates": {}, "note": note,
    }

def _stack_hamiltonian(rep, c):
    """The layer's decoupled stack, and how far its ``H(k, k_z)`` is from the layer's ``H(k)``.

    Each orbital sits at its height ``h`` along the normal on the stack's
    period ``C``, so the two are the same Hamiltonian in the gauge
    ``Λ(k_z) = diag(e^{2πi k_z h / C})``: ``H_stack(k, k_z) = Λ† H_layer(k) Λ``.
    Returns ``(stack, identity)``, or ``(refusal, None)`` where
    :func:`.groups.stack_presentation` refuses.  ``identity`` holds the
    largest difference between the two over eight seeded in-plane k at each
    ``k_z`` of :data:`_KZ_PROBE` (``"residual"``), the threshold it is held to
    -- zero for a flat layer, and otherwise the rounding the phases carry
    (``"threshold"``) -- the difference at ``k_z = 0``, which no gauge touches
    and is held to zero (``"at_zero"``), the stack's period (``"period"``) and
    the layer's extent along its normal (``"extent"``).
    """
    stack = _layers.stack_presentation(rep.model, c=c)
    if isinstance(stack, Refusal):
        return stack, None
    period = float(np.linalg.norm(np.asarray(stack.lat_vecs, dtype=float)[2]))
    z = np.asarray(stack.orb_vecs, dtype=float)[:, 2]
    extent = float(z.max() - z.min()) * period if z.size else 0.0
    rng = np.random.default_rng(20240829)
    k2 = rng.uniform(-0.5, 0.5, size=(8, 2))
    flat = np.asarray(_bloch.hamiltonian(rep.model, k2))
    stack_sum = _bloch.evaluator(stack)
    phase = 2j * np.pi * z
    if int(stack.nspin) == 2:
        phase = np.repeat(phase, 2)
    worst = at_zero = 0.0
    for kz in _KZ_PROBE:
        k3 = np.concatenate([k2, np.full((8, 1), kz)], axis=1)
        lam = np.exp(phase * float(kz))
        d = float(np.abs(np.conj(lam)[None, :, None] * flat * lam[None, None, :]
                         - np.asarray(stack_sum(k3))).max())
        worst = max(worst, d)
        if float(kz) == 0.0:
            at_zero = d
    threshold = (0.0 if extent == 0.0 else
                 _PRESENTATION_RELATIVE * float(np.abs(flat).max()))
    return stack, {"residual": worst, "threshold": threshold, "at_zero": at_zero,
                   "period": period, "extent": extent}

def _identity_record(rep, identity) -> dict:
    """The stack's Hamiltonian against the layer's, and the hoppings between copies."""
    return {"hamiltonian_identity": {
                "residual": float(identity["residual"]),
                "threshold": float(identity["threshold"]),
                "at_zero": float(identity["at_zero"]), "n_k": 8, "k_z": _KZ_PROBE,
                "note": ("max |H_stack(k, k_z) - L(k_z)^dagger H_layer(k) L(k_z)|, L the "
                         "phases e^{2 pi i k_z h / C} of the orbitals' heights h on the "
                         "stack's period C: zero at k_z = 0, and at every k_z for a flat "
                         "layer")},
            "no_hopping_across_the_stack": {
                "n_hoppings": int(rep.model.nhops), "worst": 0,
                "note": "measured by stack_presentation, which refuses on any nonzero"}}

def _identity_holds(identity) -> bool:
    """Whether the stack is the layer: equal at ``k_z = 0``, to the gauge's rounding elsewhere."""
    return identity["at_zero"] == 0.0 and identity["residual"] <= identity["threshold"]

def _shared_characters(table, table3, matched) -> tuple:
    """The two presentations' band characters where they meet.

    Every k-point of the layer's table against every point of the stack's in
    the layer's plane at the same reduced k, operation by operation through
    ``matched`` (``{layer ind: stack ind}``).  Returns ``(worst, n, points)``:
    the largest difference, how many characters were compared, and the pairs of
    points compared.
    """
    worst, n, shared = 0.0, 0, []
    for name2, rec2 in table.records.items():
        chi2 = _block_characters(rec2)
        for name3, rec3 in table3.records.items():
            k3 = np.asarray(rec3.k, dtype=float).reshape(-1)
            if len(k3) > 2 and abs(k3[2] - round(k3[2])) > 1e-6:
                continue                       # not in the layer's plane
            if not _same_k(rec2.k, k3):
                continue
            chi3 = _block_characters(rec3)
            shared.append((str(name2), str(name3)))
            for i2, x in chi2.items():
                j3 = matched.get(int(i2))
                if j3 is not None and j3 in chi3:
                    n += 1
                    worst = max(worst, abs(x - chi3[j3]))
    return worst, n, tuple(shared)

def _stack_fibers(table3, tol) -> tuple:
    """Each fiber at ``k_z = ½`` against the fiber in the plane at the same in-plane k.

    On a stack with no hopping between the copies the two are the same states:
    the same energies, the same partition into multiplets, the same characters
    at every operation both little groups hold.  Returns ``(fibers, first)``:
    one record per pair, and the first pair that differs, or ``None``.
    """
    fibers = []
    for name_out, rec_out in table3.records.items():
        k_out = np.asarray(rec_out.k, dtype=float).reshape(-1)
        if len(k_out) < 3 or abs(k_out[2] - round(k_out[2])) < 1e-6:
            continue                            # this one is in the plane
        for name_in, rec_in in table3.records.items():
            k_in = np.asarray(rec_in.k, dtype=float).reshape(-1)
            if (abs(k_in[2] - round(k_in[2])) > 1e-6) or not _same_k(k_out, k_in):
                continue
            chi_a, chi_b = _block_characters(rec_out), _block_characters(rec_in)
            common = sorted(set(chi_a) & set(chi_b))
            defect = max((abs(chi_a[i] - chi_b[i]) for i in common), default=0.0)
            e_a = np.asarray(rec_out.table.energies, dtype=float).reshape(-1)
            e_b = np.asarray(rec_in.table.energies, dtype=float).reshape(-1)
            energy = (float(np.abs(e_a - e_b).max()) if e_a.shape == e_b.shape
                      else float("inf"))
            same_partition = tuple(rec_out.partition) == tuple(rec_in.partition)
            same_multiset = (sorted(rec_out.counts.values())
                             == sorted(rec_in.counts.values()))
            fibers.append({"point": str(name_out), "in_plane_point": str(name_in),
                           "energy_defect": energy, "character_defect": float(defect),
                           "n_shared_operations": len(common),
                           "partition_equal": bool(same_partition),
                           "multiplicities_equal": bool(same_multiset)})
            if not (same_partition and same_multiset and energy < 1e-9
                    and defect < tol):
                return tuple(fibers), fibers[-1]
    return tuple(fibers), None

def _stack_reading(group, g3, indicator3, data3):
    """The two-dimensional reading of a class completed on the stack.

    The model is one layer, whatever the stack's group says, so the statement
    of :func:`_two_dimensional_reading` still applies: the points are the stack
    group's own, split by whether their k lies in the layer's plane, and no
    name is carried across.
    """
    hsp3 = _tables.high_symmetry_points(g3)
    in_plane = () if isinstance(hsp3, Refusal) else tuple(
        n for n, k in hsp3.items()
        if abs(float(np.asarray(k, dtype=float).reshape(-1)[2])
               - round(float(np.asarray(k, dtype=float).reshape(-1)[2]))) < 1e-6)
    if not in_plane:
        return indicator3.two_dimensional
    u3, _v3, _d3, matrix3 = _smith_system(data3, g3.number_str)
    _n, _t, _note, coeffs3 = _named_forms(g3, data3, matrix3, list(indicator3.y),
                                          _magnetic_type(g3))
    return _two_dimensional_reading(
        g3, indicator3.labels, u3, indicator3.slots, indicator3.divisors,
        indicator3.values, coeffs3,
        plane={"points": in_plane, "dim_k": int(group.dim_k),
               "periodic_dirs": tuple(int(p) for p in group.periodic_dirs),
               "vacuum_dirs": tuple(int(p) for p in group.vacuum_dirs),
               "presentation": "decoupled-STACK"})

def _stack_in_group(table, group, rep, restriction, *, c, bands, verify, characters_tol):
    """The rows at ``k_z = ½`` measured on the layer's decoupled stack, in the layer's own group.

    For a layer's representation.  The decoupled stack obeys exactly what the
    layer obeys: with no hopping
    between the copies, an operation of the stack, combined with a stacking
    translation, carries one copy onto itself and is an operation of the
    layer, and an operation of the layer is one of the stack's.  So the stack's
    group is the group handed in, carried onto the stack's cell by each
    operation's Cartesian action (:func:`._layers._layer_on_stack`), and its
    matrices are the layer's own, the orbitals being the same; neither is
    measured again.  What the stack adds is a crystal with the points at
    ``k_z = ½``, and the rows there are measured on it.

    Returns ``(indicator, record)``.  ``indicator`` is ``None`` where the stack
    cannot be built for this layer — one with a hopping along its normal — or
    the content cannot be read on it, the record's ``"outcome"`` then
    ``"stopped"`` and ``"stopped_at"`` saying where.

    Raises
    ------
    ..refusal.InternalError
        Where a check that ties the stack to the layer fails: the stack's
        Hamiltonian against the layer's, the layer's matrices on the stack's
        Hamiltonian, the two presentations' band characters where they meet,
        and each fiber at ``k_z = ½`` against its partner in the plane.  The
        physics guarantees all four, so a failure is the package disagreeing
        with itself, and the message names the check and its number.
    """

    record = _stack_record(
        table, group, rep, restriction, c=c,
        environment=(None if getattr(rep, "environment", None) is None
                     else str(rep.environment)),
        note=("the layer's plane has no coordinate for the k-points at k_z = ½, where its "
              "group's EBR basis has rows that repeat no in-plane row. The same layer "
              "repeated along its normal with no hopping between copies, each orbital at "
              "its own height, is a crystal whose bands do not depend on k_z, its "
              "H(k, k_z) the layer's H(k) up to the phases of those heights, and whose "
              "group is the layer's own, "
              "each operation carried by its Cartesian action and represented by the "
              "layer's own matrix, so those rows are measured on it. Every identity "
              "that ties it to the layer is measured and recorded below."))

    def stop(where, reason, **extra):
        record["stopped_at"] = str(where)
        record["stop_reason"] = str(reason)
        record["gates"].setdefault(str(where), {}).update(extra)
        return None, record

    def defect(check, number, what):
        raise InternalError(
            f"the decoupled stack of this layer, read in its own group "
            f"{group.name} #{group.number_str}, fails the check '{check}': {what} "
            f"({number:.3g}). With no hopping between the copies the stack is the "
            "layer, and obeys exactly what the layer obeys, so this is the package "
            "disagreeing with itself, not a property of the model.")

    # ---- 1. the stack, and the identity that makes it the layer ------------
    stack, identity = _stack_hamiltonian(rep, c)
    if isinstance(stack, Refusal):
        # a layer the stack is not built for (a hopping along the normal)
        return stop("stack_presentation",
                    "the decoupled stack is not built for this layer: " + stack.reason,
                    gate=stack.gate, value=stack.value)
    record["gates"].update(_identity_record(rep, identity))
    record["stack_period"], record["stack_extent"] = identity["period"], identity["extent"]
    if not _identity_holds(identity):
        defect("hamiltonian_identity", identity["residual"],
               "its H(k, k_z) is not the layer's H(k) in the stack's gauge")

    # ---- 2. the layer's group and matrices, on the stack ----------------
    represented = {int(i) for i in rep.indices}
    unrepresented = tuple(sorted(int(s.ind) for s in group.symmetries
                                 if int(s.ind) not in represented))
    if unrepresented:
        return stop("representation",
                    "the representation handed in has no matrix for operations "
                    f"{_text._runs(unrepresented)} of {group.name} #{group.number_str}, "
                    "so the decoupled stack's rows at k_z = ½ cannot be read in that group",
                    unrepresented=unrepresented)
    g3, asg3 = _layers._layer_on_stack(group, rep.assignment, stack)
    by_ind = {int(s.ind): s for s in g3.symmetries}
    rng = np.random.default_rng(20241004)
    battery = rng.uniform(-0.5, 0.5, size=(8, 3))
    battery[:len(_KZ_PROBE), 2] = _KZ_PROBE
    rep3 = replace(rep, model=stack, group=g3, assignment=asg3,
                   _operations=tuple(by_ind[int(i)] for i in rep.indices),
                   _battery=battery)
    residuals = rep3.covariance(battery)
    scale = float(np.abs(np.asarray(_bloch.hamiltonian(stack, battery))).max())
    gate = max(covariance_gate(len(rep.indices), int(stack.nstate), scale),
               float((rep._search or {}).get("gate") or 0.0))
    worst_d = max(residuals.values(), default=0.0)
    record["gates"]["representation"] = {
        "covariance": float(worst_d), "threshold": float(gate),
        "n_operations": len(rep.indices), "n_k": int(battery.shape[0]),
        "note": ("the layer's own matrices on the stack's H(k, k_z): "
                 "max |D H(k) D^-1 - H(Rk)| over every operation")}
    if worst_d > gate:
        defect("representation", float(worst_d),
               "the layer's own matrices do not carry the stack's H(k, k_z) onto itself")
    rep3 = replace(rep3, residuals=residuals)
    typat = tuple(int(group.typat[int(s)]) for s in rep.assignment.site_of_orbital)
    record["stack"] = {"number_str": str(g3.number_str), "name": str(g3.name),
                       "size": int(g3.size), "dim_k": int(g3.dim_k),
                       "periodic_dirs": tuple(int(p) for p in g3.periodic_dirs),
                       "typat": typat}

    # ---- 3. the content on the stack, same band block ----------------------
    within = f"the decoupled stack, read in {group.name} #{group.number_str},"
    table3 = _sewing.hsp_table(rep3, bands=_table_bands(table))
    if isinstance(table3, Refusal):
        return stop("stack_content", f"{within} has content that could not be read: "
                    + table3.reason, gate=table3.gate)
    if not table3.complete:
        return stop("stack_content",
                    f"{within} has content missing at {sorted(table3.refusals)}",
                    refusals={k: r.gate for k, r in table3.refusals.items()})

    # ---- 4. the characters, where the two presentations meet --------------
    worst_chi, n_chi, shared_points = _shared_characters(
        table, table3, {int(i): int(i) for i in rep.indices})
    labels_equal = all(dict(table.content[a]) == dict(table3.content[b])
                       for a, b in shared_points
                       if a in table.content and b in table3.content)
    record["gates"]["characters"] = {
        "note": ("the measured characters of the bands at the operations and k-points the "
                 "two presentations share, and the labels there, the group being one"),
        "worst": worst_chi, "threshold": float(characters_tol), "n_comparisons": n_chi,
        "shared_points": shared_points, "labels_equal": labels_equal}
    if not n_chi:
        defect("characters", 0.0, "the stack and the layer share no k-point in the plane")
    if worst_chi > characters_tol:
        defect("characters", worst_chi,
               "the stack's band characters differ from the layer's where they meet")
    if not labels_equal:
        defect("characters", float(len(shared_points)),
               "the stack labels the layer's bands differently in the same group")

    # ---- 5. the fiber at k_z = ½ is the fiber at k_z = 0 ------------------
    fibers, bad = _stack_fibers(table3, characters_tol)
    record["gates"]["fibers"] = fibers
    if bad is not None:
        defect("fibers", max(bad["energy_defect"], bad["character_defect"]),
               f"the fiber at {bad['point']} is not the fiber at {bad['in_plane_point']}")

    # ---- 6. the class, in the layer's own group ----------------------------
    indicator3 = symmetry_indicator(table3, g3, rep=rep3, bands=bands, verify=verify)
    if isinstance(indicator3, Refusal):
        return stop("stack_indicator", f"{within} has no class there: "
                    + indicator3.reason, gate=indicator3.gate)
    data3 = _tables.ebr_data(g3)
    record["stack"]["divisors"] = _divisors_of(data3, g3.number_str)
    record["outcome"] = "completed"
    return replace(indicator3, completion=record,
                   two_dimensional=_stack_reading(group, g3, indicator3, data3)), record

def symmetry_indicator(content, group, *, rep=None, points=None, bands=None,
                       verify=True,
                       stack_period=_layers.STACK_C) -> SymmetryIndicator | Refusal:
    r"""The symmetry-indicator class of a band set's content: the class of its symmetry vector among the band structures modulo the atomic limits.

    A nonzero class says no band set with this content is an atomic limit, nor
    any with atomic bands added; a zero class certifies nothing.  A layer's
    class is read in its slab's three-dimensional group, as the class of its
    decoupled stack.

    Parameters
    ----------
    content : .sewing.HSPTable or mapping
        The band set's content, as :func:`decompose` takes it; a table must be
        complete.
    group : .groups.GroupData
        The group the content was read in, ``r.group``.
    rep : .symmetry.SpaceGroupRep, optional
        The representation the content was read with, ``r.rep``.  A layer needs
        it, with a table, for its coreps at ``k_z = ½`` that repeat none in its
        plane: they are measured on its decoupled stack, the layer repeated
        along its normal with no hopping between copies, built from it.
    bands : int or sequence of int, optional
        Which bands the content counts, as :func:`decompose` takes it.
    points : sequence of str, optional
        The high-symmetry points the model samples; given, no stack is built.
    verify : bool, optional
        Compute the Smith normal form of the group's matrix of elementary band
        representations again, exactly, and check the tables' against it.
        Default ``True``.
    stack_period : float, optional
        The vacuum between the stack's copies, in the model's length units;
        nothing returned depends on it.  Default ``8.0``.

    Returns
    -------
    SymmetryIndicator or Refusal
        ``si.summary()`` prints the class.  ``si.verdict`` is ``"indicated
        nontrivial"``, ``"indicator zero"`` or ``"trivial indicator group"``;
        ``si.structure`` the indicator group and ``si.values`` the class in it,
        one integer modulo each of ``si.divisors``, of order ``si.order``;
        ``si.canonical`` and ``si.irreptables_named`` the class in named
        indicators, as the paper that tabulates the stable real-space
        invariants and as ``irreptables`` name them; ``si.completion`` how a
        layer's coreps out of its plane were measured.  The record is truthy
        whatever its verdict; a ``Refusal`` in its place is falsy and names
        what stopped the call.

    Raises
    ------
    ValueError
        ``bands=`` or ``points=`` naming other bands or points than the table
        measured; or a group, or ``rep=``, that is not the one the table was
        read in, which it carries as ``t.rep``.
    ..refusal.InternalError
        The tables' Smith form and the one computed again disagree, a check
        that ties the decoupled stack to the layer fails, or two classes share
        every named value: the package disagreeing with itself.

    Refusals
    --------
    ``incomplete_content``
        the table refused at some point; there is no override.
    ``indicator_not_evaluable``
        a layer's coreps at ``k_z = ½`` repeat none in its plane and were not
        measured on its stack: no ``rep=`` or no table, ``points=`` given, or a
        check of the stack that did not complete (``context["stack"]``).
    ``not_a_band_structure``
        the content fails the group's compatibility relations.
    ``unknown_label``
        a corep the group's elementary band representations do not hold.
    ``non_integer_multiplicity`` / ``negative_multiplicity``
        the content is no count of coreps.
    ``magnetic_type_out_of_scope``
        a magnetic group of type III or IV, which
        :func:`.groups.realized_group` never returns.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        no elementary band representations of this group, or no tables in this
        cell.

    Examples
    --------
    Fu, Kane and Mele's diamond with one bond detuned, its two lowest bands:

    >>> si = symmetry_indicator(hsp_table(r.rep, bands=2), r.group, rep=r.rep)  # doctest: +SKIP
    >>> si.structure, si.values, si.canonical                       # doctest: +SKIP
    ('Z2 x Z4', (0, 1), {'z2w,1': 1, 'z2w,2': 1, 'z2w,3': 1, 'z4': 3})
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    counts = content_of(content)
    if isinstance(counts, Refusal):
        return counts

    mtype = _magnetic_type(group)
    out_of_scope = _spacegroup._magnetic_scope_refusal(group)
    if out_of_scope is not None:
        return out_of_scope

    table = content if hasattr(content, "counts") else None
    _table_binding(content, group, rep, where="symmetry_indicator")
    block = _bands_of(content, bands, where="symmetry_indicator")
    if table is not None and not table.complete:
        refused = {name: r.gate for name, r in (table.refusals or {}).items()}
        return Refusal(
            "incomplete_content",
            f"the measured content is missing {sorted(refused)}: {refused}. An indicator "
            "reads the whole symmetry vector, and the class of a vector with a point "
            "missing is the class of a different vector, so this content has no class "
            "here. Read the table's refusals. Completing the missing content from the "
            "stable real-space invariants is future work, and there is no "
            "`allow_incomplete` here.",
            value=float(len(refused)), threshold=0.0,
            context={"refusals": refused, "resolved": tuple(getattr(table, "content", {})),
                     "completion": "stable-rsi"})

    data = _tables.ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels = tuple(str(l) for l in data["labels_physics_convention"])

    unknown = sorted(set(counts) - set(labels))
    if unknown:
        return Refusal(
            "unknown_label",
            f"the measured labels {unknown} are not in the EBR basis of group "
            f"#{group.number_str}; the symmetry vector and the indicator basis are not "
            "written in the same alphabet.",
            value=float(len(unknown)), threshold=0.0,
            context={"unknown": tuple(unknown), "basis": labels})

    resolved = _resolved_points(table, points, where="symmetry_indicator")
    counts, resolved = _band_count_rows(group, table, data, block, counts, resolved)
    restriction = in_plane_restriction(group, points=resolved, require_exact=False)
    if isinstance(restriction, Refusal):
        return restriction

    # ---- the rows out of the layer's own plane, measured on the stack -----
    # Available only when the point set was not overridden (that is an
    # instruction about what the model may sample, and it is obeyed), when a
    # representation is in hand (the stack is built from its model and carries
    # its matrices), and when the content came from an HSPTable (which names
    # the band block the stack has to measure again).  The stack obeys exactly
    # what the layer obeys, so the class is completed in the layer's own group,
    # and only where rows at k_z = ½ repeat no in-plane row.
    completion = None
    if (int(getattr(group, "dim_k", 3)) < 3 and points is None and rep is not None
            and table is not None and restriction.orphans):
        stacked, completion = _stack_in_group(
            table, group, rep, restriction, c=stack_period, bands=block,
            verify=verify, characters_tol=CHARACTERS_TOL)
        if stacked is not None:
            return stacked

    if restriction.orphans:
        return Refusal(
            "indicator_not_evaluable",
            f"the EBR rows {list(restriction.orphans)} of #{group.number_str} sit at "
            "k-points this content does not cover, out of a 2D model's periodic plane "
            "or carrying no information, and they duplicate no row it does cover, so "
            "the symmetry vector cannot be completed and the indicator has no value "
            "here. "
            + (f"The decoupled stack of the layer, the crystal on which those rows are "
               f"measured, did not supply them: "
               f"{str(completion['stop_reason']).rstrip('. ')} (what was measured, and "
               f"where it stopped, is in context['stack'])."
               if completion is not None else
               "The rows can be measured on the decoupled stack of the layer, which "
               "needs the representation, with its model and declared characters, and "
               "the bands the content counts: pass an HSPTable and rep=, and do not "
               "override points=.")
            + " Completing missing content from the stable real-space invariants is "
            "future work.",
            value=float(len(restriction.orphans)), threshold=0.0,
            context={"orphans": tuple(restriction.orphans),
                     "points": tuple(restriction.points),
                     "retained": tuple(restriction.retained),
                     "number_str": group.number_str, "completion": "stable-rsi",
                     "stack": completion})

    # Every dropped row duplicates a retained one, so its multiplicity is
    # determined -- the in-plane restriction's own exactness argument, read in
    # the other direction.  Nothing is guessed and nothing is zeroed, and the
    # compatibility relations are read on the vector completed so
    y, reconstructed = _filled_content(counts, restriction, labels)
    (u, v, d, matrix), y_prime, wrong = _compatibility(data, group.number_str, labels, y)
    if wrong is not None:
        return wrong

    slots = tuple(i for i, di in enumerate(d) if di > 1)
    divisors = tuple(int(d[i]) for i in slots)
    values = tuple(int(y_prime[i]) % int(d[i]) for i in slots)
    order = 1
    for val, div in zip(values, divisors):
        step = div // np.gcd(int(val), int(div)) if val else 1
        order = int(np.lcm(order, int(step)))

    verification = {"performed": False,
                    "note": "verify=False: the shipped Smith form was taken on trust"}
    if verify:
        verification = _verify_smith(group.number_str, matrix, u, v, d, y,
                                     slots, divisors, values, order)

    named, touched, irreptables_note, coeffs = _named_forms(group, data, matrix, y, mtype)
    two_d = _two_dimensional_reading(group, labels, u, slots, divisors, values, coeffs)
    # the canonical vocabulary: the same class, spoken in the SRSI paper's own
    # names.  One rational solve of BR.p = y -- the identical arithmetic
    # `srsi.srsi_of_bands` does on the same y -- and then the tabulated forms.

    canonical, canonical_note, canonical_tex = _srsitables._canonical_reading(
        group, mtype, u, v, d, y, divisors)

    verdict = ("trivial indicator group" if not divisors
               else ("indicated nontrivial" if any(values) else "indicator zero"))
    provenance = (getattr(table, "group_provenance", None)
                  or str(getattr(group, "provenance", "cell")))
    # the free-standing layer's group, as `realized_group` measured it beside a
    # substrate's, named where it is larger than the group the class lives in
    free = getattr(rep, "free_standing_group", None) if rep is not None else None
    free = dict(free) if free and int(free["size"]) > int(group.size) else None
    return SymmetryIndicator(
        verdict=verdict, number_str=str(group.number_str), group_name=str(group.name),
        magnetic_type=mtype, divisors=divisors, values=values,
        structure=_structure(divisors), nontrivial=bool(any(values)), order=int(order),
        y=tuple(int(x) for x in y), y_prime=tuple(int(x) for x in y_prime),
        labels=labels, slots=slots, content=dict(counts), reconstructed=reconstructed,
        restriction=restriction, irreptables_named=named,
        irreptables_note=irreptables_note,
        irreptables_convention_touched=touched, two_dimensional=two_d,
        verification=verification, completion=completion,
        lattice_check={"in_lattice": True, "residual": 0,
                       "relations": int(sum(1 for x in d if x == 0))},
        site_content=_site_content_record(rep), group_provenance=str(provenance),
        n_bands=None if block is None else len(block),
        canonical=canonical, canonical_note=canonical_note,
        canonical_tex=canonical_tex, free_standing_group=free)

def subtract(content_a, content_b) -> dict | Refusal:
    """The content of the bands in one set and not in another: ``a − b`` as a symmetry vector.

    Parameters
    ----------
    content_a, content_b : .sewing.HSPTable or mapping
        The two contents; ``b`` is contained in ``a``.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``, the zeros dropped: empty, and so falsy, where
        the two are equal, so ``isinstance(out, Refusal)`` is what tells a
        refusal apart.

    Raises
    ------
    ValueError
        Two tables read in different groups, whose coreps are not one alphabet.

    Refusals
    --------
    ``negative_multiplicity``
        ``b`` is not contained in ``a``: a difference of two band sets is one
        only where one contains the other.  A signed combination is what
        :attr:`EBRDecomposition.signed_combinations` holds.
    ``non_integer_multiplicity``
        either content is no count of coreps.
    """
    a = content_of(content_a)
    if isinstance(a, Refusal):
        return a
    b = content_of(content_b)
    if isinstance(b, Refusal):
        return b
    own_a = getattr(content_a, "rep", None) if hasattr(content_a, "counts") else None
    own_b = getattr(content_b, "rep", None) if hasattr(content_b, "counts") else None
    if own_a is not None and own_b is not None and not _same_group(own_a.group, own_b.group):
        raise ValueError(
            f"subtract(): the tables were read in different groups, {own_a.group.name} "
            f"#{own_a.group.number_str} and {own_b.group.name} #{own_b.group.number_str}, "
            "whose coreps cannot be subtracted label by label.")

    out, bad = {}, {}
    for lab in set(a) | set(b):
        v = a.get(lab, 0) - b.get(lab, 0)
        if v < 0:
            bad[lab] = v
        elif v:
            out[lab] = v
    if bad:
        return Refusal(
            "negative_multiplicity",
            f"the difference has negative multiplicities {bad}; the second content is "
            "not contained in the first, so their difference is not a band set. A "
            "signed EBR combination is still meaningful (it is what fragility means) "
            "but it is not a symmetry vector and cannot be decomposed as one.",
            value=float(min(bad.values())), threshold=0.0,
            context={"negative": bad, "a": a, "b": b})
    return out

def completeness_check(content, group, *, bands=None,
                       **kwargs) -> CompletenessReport | Refusal:
    """Whether a band set's content is a sum of elementary band representations, and how many bands each sum holds.

    Parameters
    ----------
    content : .sewing.HSPTable or mapping
        The band set's content, as :func:`decompose` takes it.
    group : .groups.GroupData
        The group the content was read in, ``r.group``.
    bands : int or sequence of int, optional
        The set of bands, as :func:`decompose` takes it: the bands each sum is
        held against.
    **kwargs
        Passed to :func:`decompose`: ``rep=``, ``allow_incomplete=``, ...

    Returns
    -------
    CompletenessReport or Refusal
        ``c.summary()`` prints it.  ``c.complete`` is whether a sum carries the
        content, ``c.band_counts`` the bands each sum holds, and
        ``c.band_count_matches`` whether those are the bands asked for;
        ``c.decomposition`` holds the sums.  A ``Refusal`` in its place is
        falsy and names what stopped the call.

    Raises
    ------
    ValueError
        As :func:`decompose` raises: ``bands=`` naming other bands than the
        table measured, or a group or ``rep=`` that is not the table's own.

    Refusals
    --------
    ``incomplete_content``, ``not_a_band_structure``, ``unknown_label``, ``restriction_not_exact``, ...
        those of :func:`decompose`.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    out = decompose(content, group, bands=bands, **kwargs)
    if isinstance(out, Refusal):
        return out

    dims = []
    for name, _wy in out.ebrs:
        n = str(name)
        try:
            dims.append(int(n[n.rindex("(") + 1:n.rindex(")")]))
        except (ValueError, IndexError):          # pragma: no cover - defensive
            dims.append(0)
    dims = np.array(dims, dtype=int)
    band_counts = tuple(int(np.dot(dims, np.asarray(s, dtype=int))) for s in out.solutions)
    matches = (None if out.n_bands is None
               else all(b == int(out.n_bands) for b in band_counts))
    return CompletenessReport(complete=out.decomposable, decomposition=out,
                              n_bands=out.n_bands,
                              band_counts=band_counts, band_count_matches=matches)

def regular_content(group, *, points=None) -> dict | Refusal:
    r"""The content of one copy of the regular band representation of the group: ``|U|`` bands at every k.

    The regular band representation is the one the regular representation of a
    site group induces, ``|U|`` the number of unitary operations up to lattice
    translations, the single-valued group's for a spinful model.  At ``k`` it is
    ``|U|/|U_k|`` copies of the regular representation of the little group
    ``U_k``, so a corep ``ρ`` occurs

    .. math:: m_\rho(k) = \frac{|U|\,\dim\rho}{\sum_{g \in U_k} |\chi_\rho(g)|^2}

    times: a corep that time reversal makes of two irreps, or of one irrep
    twice, occurs half or a quarter as often as an irrep of its dimension.

    Parameters
    ----------
    group : .groups.GroupData
        The group, ``r.group``.
    points : sequence of str, optional
        The high-symmetry points to read it at.  Default: every tabulated point
        in the model's plane.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``.  A ``Refusal`` in its place is falsy and
        names what stopped the call.

    Refusals
    --------
    ``regular_content_not_integral``
        the tables' characters give a multiplicity that is no integer, or a
        content of other than ``|U|`` bands at a point.
    ``tables_not_ready`` / ``out_of_plane``
        no tables in this cell, or a point out of a layer's plane.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    hsp = _tables.high_symmetry_points(group)
    if isinstance(hsp, Refusal):
        return hsp
    wanted = tuple(hsp) if points is None else tuple(str(p) for p in points)

    n_u = len(group.u_symmetries)
    out, bands = {}, {}
    for name in wanted:
        k = hsp[name]
        lg = _tables.little_group(group, k)
        if isinstance(lg, Refusal):
            return lg
        table = _tables.corep_table(group, k)
        if isinstance(table, Refusal):
            return table
        ident = min(lg.indices)
        total = 0.0
        for label, char in table.items():
            dim = abs(complex(char[ident]))
            norm = sum(abs(complex(char[i])) ** 2 for i in lg.indices)
            m = n_u * dim / norm
            if abs(m - round(m)) > _INT_TOL:
                return Refusal(
                    "regular_content_not_integral",
                    f"the regular multiplicity of {label} at {name} came out {m:.6g}, "
                    "which is not an integer; the tabulated characters and the little "
                    "group disagree about this corep.",
                    value=float(m), threshold=float(_INT_TOL),
                    context={"kpname": name, "label": label, "n_unitary": n_u,
                             "little_group": len(lg.indices)})
            out[label] = int(round(m))
            total += round(m) * dim
        bands[name] = total
    off = {n: b for n, b in bands.items() if abs(b - n_u) > _INT_TOL}
    if off:
        return Refusal(
            "regular_content_not_integral",
            f"the regular content does not carry the {n_u} bands of one regular band "
            f"representation at every point: {off}",
            value=float(max(off.values())), threshold=float(n_u),
            context={"bands": bands, "n_unitary": n_u})
    return out

def regular_completion(content, group, *, bands=None, rep=None, points=None,
                       **kwargs) -> RegularCompletion | Refusal:
    """The bands that would carry a band set up to whole copies of the regular band representation, and the orbitals they would be.

    The regular band representation is ``|U|`` bands at every k, ``|U|`` the
    number of unitary operations up to lattice translations, the single-valued
    group's for a spinful model (:func:`regular_content`).  Both sides of a gap
    completed to it make an atomic limit together.

    Parameters
    ----------
    content : .sewing.HSPTable or mapping
        The band set's content.
    group : .groups.GroupData
        The group the content was read in, ``r.group``.
    bands : int or sequence of int
        The bands of the completed system: a number of them, or their indices,
        a multiple of ``|U|`` in all.  Required.
    rep : .symmetry.SpaceGroupRep, optional
        The representation, ``r.rep``, which :func:`decompose` reads.
    points : sequence of str, optional
        The high-symmetry points the model samples.
    **kwargs
        Passed to :func:`decompose`.

    Returns
    -------
    RegularCompletion or Refusal
        ``rc.summary()`` prints it.  ``rc.padding`` holds the coreps the
        missing bands carry, ``rc.decomposition`` the orbitals they would be,
        ``rc.regular`` one copy's content and ``rc.bands_per_copy`` its bands.
        A ``Refusal`` in its place is falsy and names what stopped the call.

    Raises
    ------
    ValueError
        A group, or ``rep=``, that is not the one the table was read in, which
        it carries as ``t.rep``.

    Refusals
    --------
    ``not_a_regular_multiple``
        the number of bands is no multiple of ``|U|``.
    ``negative_multiplicity``
        the band set carries more of a corep than that many regular copies do:
        it is no part of them.
    ``regular_content_not_integral``, ``not_a_band_structure``, ``unknown_label``, ...
        those of :func:`regular_content` and :func:`decompose`.

    Examples
    --------
    lin3g's six lowest bands, completed to its eight, one regular copy: the
    padding ``-GM7 + -M6 + -X5`` is ``-E1`` at ``1b``, the content of the two
    bands above the gap.

    >>> rc = regular_completion(hsp_table(r.rep, bands=6), r.group, bands=8, rep=r.rep)  # doctest: +SKIP
    >>> rc.decomposition.solution_strings()                         # doctest: +SKIP
    ("1 x -E1↑G(2)@1b(4mm1',4mm)",)
    """
    if bands is None:
        raise TypeError("regular_completion() missing required argument 'bands'")
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    _table_binding(content, group, rep, where="regular_completion")
    target_one = regular_content(group, points=points)
    if isinstance(target_one, Refusal):
        return target_one

    n_u = len(group.u_symmetries)
    n_bands = len(_as_block(bands, None))
    if n_bands % n_u:
        return Refusal(
            "not_a_regular_multiple",
            f"{n_bands} bands is not a multiple of {n_u}, the band count of one "
            f"regular band representation of #{group.number_str}. Regular-representation "
            "padding is only defined at multiples of it; pick the next multiple "
            f"({n_u * (n_bands // n_u + 1)}).",
            value=float(n_bands % n_u), threshold=0.0,
            context={"n_bands": n_bands, "bands_per_copy": n_u,
                     "number_str": group.number_str})

    copies = n_bands // n_u
    target = {lab: copies * m for lab, m in target_one.items()}
    occ = content_of(content)
    if isinstance(occ, Refusal):
        return occ

    padding = subtract(target, occ)
    if isinstance(padding, Refusal):
        return padding

    got = decompose(padding, group, rep=rep, points=points, **kwargs)
    if isinstance(got, Refusal):
        return got
    return RegularCompletion(copies=copies, bands_per_copy=n_u, target=target,
                             occupied=occ, padding=padding, decomposition=got,
                             regular=dict(target_one))
