r"""Whether bands could be an atomic limit, and their symmetry-indicator class.

A site irrep at a Wyckoff position induces a band representation.  An
elementary band representation (EBR) is one that is not a sum of others.  The
symmetry data vector of an atomic limit is a sum of the symmetry data vectors
of EBRs with nonnegative integer multiplicities.  The symmetry data vector of
some topological sets of bands is such a sum too.

* :func:`decompose`: every such sum that carries a symmetry data vector;
* :func:`symmetry_indicator`: the class of a symmetry data vector among the
  band structures of the group modulo its atomic limits;
* :func:`regular_completion`: the symmetry data vector of the bands that
  complete a set of bands to copies of the regular band representation;
* :func:`subtract`: the symmetry data vector of the bands in one set and not
  in another;
* :func:`ebr_rows`: the EBRs of a group as symmetry data vectors;
* :func:`in_plane_restriction`: the little-group representations a model
  measures, and what the others repeat;
* :func:`content_of`: a symmetry data vector as integer multiplicities;
* :func:`regular_content`: the symmetry data vector of one copy of the regular
  band representation;
* :func:`kpoint_of_label`: the point a little-group representation is at.

:func:`decompose`, :func:`symmetry_indicator` and :func:`regular_completion`
take the table :func:`.sewing.hsp_table` returns, alone, or a symmetry data
vector written out by hand with what :func:`.groups.realized_group` returned
beside it.  The group's data are the tables of the irreptables package, read
in the convention in which a lattice translation is represented by
``e^(-ik.t)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

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
from ._text import _SHOWN, _moduli, _times
from ._tables import (
    _letter_only, _wyckoff_letter, ebr_rows, kpoint_of_label)
from ._matrices import _orb_positions_3d, covariance_gate
from ._smith import (
    _compatibility, _divisors_of, _filled_content, _smith_system, _structure,
    _verify_smith)
from .symmetry import _sum_of, _terms_of

__all__ = [
    "Restriction", "EBRDecomposition", "EBRDecompositionNumerical", "SignedCombinations",
    "Completeness", "Enumeration", "SiteContent",
    "RegularCompletion", "SymmetryIndicator", "SymmetryIndicatorNumerical",
    "SmithVerification", "LatticeCheck", "TwoDimensionalReading", "IndicatorComponent",
    "StackCompletion", "StackInput", "StackGroup", "StackChecks", "HamiltonianIdentity",
    "StackHoppings", "RepresentationCheck", "CharactersCheck", "StackFiber",
    "content_of", "ebr_rows", "kpoint_of_label", "in_plane_restriction",
    "regular_content", "decompose", "subtract",
    "regular_completion", "symmetry_indicator",
    "DEFAULT_ENUMERATION_CAP", "INDICATOR_CONVENTION", "CHARACTERS_TOL",
]

#: How many solutions the enumeration reports before it stops.  A truncated
#: enumeration is recorded as such; it never silently becomes "these are all".
DEFAULT_ENUMERATION_CAP = 200

# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

def _kind(group) -> str:
    """The word a summary uses for the little-group representations of ``group``.

    Returns ``"irreps"`` for a group with no time reversal, of magnetic type
    I, and ``"coreps"`` for any other group, and for ``None``.
    """
    return "irreps" if getattr(group, "magnetic_type", None) == "I" else "coreps"


def _label_lines(sums) -> list:
    """The lines a summary prints for sums of elementary band representations.

    ``sums`` is a sequence of :class:`.symmetry.BandRepresentation`.  Returns
    an indented line holding the ``label`` of each of the first ``_SHOWN``,
    and a last line counting the rest where there are more.
    """
    lines = ["    " + s.label for s in sums[:_SHOWN]]
    if len(sums) > _SHOWN:
        lines.append(f"    and {len(sums) - _SHOWN} more")
    return lines


@dataclass(frozen=True)
class SiteContent:
    """The site irrep of each orbital, and how each is known.

    Held in :attr:`EBRDecomposition.site_content` and
    :attr:`SymmetryIndicator.site_content`.  The names of the little-group
    representations, and so the sums and the class read from them, depend on
    these site irreps.

    Attributes
    ----------
    number_str : str
        The number of the group the site irreps are named in, such as
        ``"99.164"``.
    irreps : tuple of str
        The site irrep of each orbital, in the order of the orbitals.
    determined_by : tuple of str
        How the site irrep of each orbital is known: ``"hamiltonian"``, fixed
        by the Hamiltonian; ``"declaration"``, declared; ``"declaration not
        carried"``, declared, while the Hamiltonian's own blocks carry another
        site irrep, the one named in :attr:`irreps`; ``"lowest-l"``, assigned
        by the lowest-angular-momentum rule among those the hoppings allow.
    residual : float
        The largest defect in reading the orbitals' characters off the
        matrices of the representation: how far an orbital's block is from
        its character times the operation's spin factor, or, for a multiplet
        of several orbitals, from multiplying as the site group does.  It is
        dimensionless, the entries of the matrices being at most 1 in size.
    """

    number_str: str
    irreps: tuple
    determined_by: tuple
    residual: float


@dataclass(frozen=True)
class Restriction:
    """The little-group representations a model measures, and what the others repeat.

    The record :func:`in_plane_restriction` returns, and the one
    :attr:`EBRDecompositionNumerical.restriction` and
    :attr:`SymmetryIndicatorNumerical.restriction` hold.  A little-group
    representation at a point not measured, such as a layer's at
    ``k_z = 1/2``, may occur in every elementary band representation as often
    as a measured one.  It is then said to repeat that one, and adds no
    condition to a symmetry data vector.  Where every one left out repeats a
    measured one, the decomposition or class read from the measured ones is
    that of the whole symmetry data vector.  ``summary()`` prints those left
    out, and the measured one each repeats.

    Attributes
    ----------
    kpoints : tuple of str
        The tabulated names of the points measured.
    retained : tuple of str
        The little-group representations of the group's elementary band
        representations at those points, the measured ones.
    dropped : tuple of str
        If nonempty, those at the other points, which are left out.  Empty
        where every tabulated point is measured.
    copies : dict
        If nonempty, ``{left out: (measured, ...)}``: for each little-group
        representation left out, the measured ones it repeats.  An empty tuple
        for one that repeats no measured one.
    orphans : tuple of str
        If nonempty, the little-group representations left out that repeat no
        measured one.  The multiplicity of each is an entry of a symmetry data
        vector that the measured ones do not fix.
    exact : bool
        Whether every little-group representation left out repeats a measured
        one.
    outside_layer : dict
        If nonempty, ``{Wyckoff position: heights}``: the Wyckoff positions of
        a layer's group that are fixed along its normal, with a height outside
        the range of the orbitals' heights.  The heights are measured from the
        layer's middle plane along its normal, in the model's units of length.
        The band representations induced from those positions are left out.
    """

    kpoints: tuple
    retained: tuple
    dropped: tuple
    copies: dict
    orphans: tuple
    exact: bool
    outside_layer: dict = field(default_factory=dict)
    # The rank of the matrix of elementary band representations on the kept
    # little-group representations and on all of them; for a layer, the heights
    # its orbitals span from its middle plane; whether the call refuses an orphan;
    # whether the group has no time reversal, for the word a summary prints.
    _rank: int = field(default=0, repr=False)
    _full_rank: int = field(default=0, repr=False)
    _layer_extent: tuple | None = field(default=None, repr=False)
    _require_exact: bool = field(default=True, repr=False)
    _irreps: bool = field(default=False, repr=False)

    def summary(self) -> str:
        """What is left out, a line for each part, with the attribute that holds it."""
        kind = "irreps" if self._irreps else "coreps"
        if not self.dropped:
            return (f"every tabulated point is measured, and no {kind[:-1]} is left out   "
                    "(.dropped)")
        points = sorted({kpoint_of_label(label) for label in self.dropped},
                        key=_tables.point_order)
        lines = [f"the {kind} at {_text._listed(points)} are not measured, and are left "
                 "out   (.dropped)"]
        copied = {label: rows for label, rows in self.copies.items() if rows}
        if copied:
            which = "each" if self.exact else "each but " + _text._listed(self.orphans)
            lines.append(f"  {which} occurs in every elementary band representation as often "
                         "as a measured one, and leaving it out changes no answer   (.copies)")
            lines.append("    " + ", ".join(f"{label} as {rows[0]}"
                                            for label, rows in copied.items()))
        if self.orphans:
            lines.append(f"  {_text._listed(self.orphans)} repeat no measured {kind[:-1]}: "
                         "leaving them out gives up a condition, and an answer is a "
                         "necessary condition only   (.orphans, .exact)")
        outside = sorted(self.outside_layer)
        if outside:
            lowest, highest = self._layer_extent or (0.0, 0.0)
            off = "off the layer" if highest - lowest <= ROUNDING else "outside the slab"
            lines.append(f"  {_text._listed(outside)} lie {off}, so their band "
                         "representations are left out   (.outside_layer)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, convenience only
        kind = "irreps" if self._irreps else "coreps"
        return (f"Restriction({len(self.retained)} {kind} kept, {len(self.dropped)} left out, "
                + ("exact)" if self.exact else "not exact)"))


def _where_line(restriction, completeness, kind) -> str | None:
    """The line of a summary that says which points a decomposition was read without.

    ``restriction`` is the decomposition's :class:`Restriction`,
    ``completeness`` its :class:`Completeness`, and ``kind`` the word for its
    little-group representations (:func:`_kind`).  The line names the points
    at which the table refused and those not measured, or out of a layer's
    plane, and says whether the sums are exact without them.  Returns
    ``None`` where no point was left out.
    """
    r = restriction
    refused = sorted(completeness.refused or (), key=_tables.point_order)
    away = sorted({kpoint_of_label(label) for label in r.dropped} - set(refused),
                  key=_tables.point_order)
    where = []
    if refused:
        where.append(f"{_text._listed(refused)} refused, and "
                     f"{'are' if len(refused) > 1 else 'is'} left out with "
                     f"{'their' if len(refused) > 1 else 'its'} {kind}")
    if away:
        where.append(f"{_text._listed(away)} "
                     + ("lie out of the layer's plane" if r._layer_extent is not None
                        else "are not measured"))
    if not where:
        return None
    one = kind[:-1]
    return ("; ".join(where) + ": "
            + (f"every {one} there occurs in every elementary band representation as often "
               "as a measured one, so the answer is exact" if r.exact
               else f"{_text._listed(r.orphans)} repeat no measured {one}, so the answer is "
               "a necessary condition only"))


def _outside_line(restriction) -> str | None:
    """The line of a summary naming the Wyckoff positions off a free-standing layer.

    ``restriction`` is a :class:`Restriction`.  The line says that the
    positions of its ``outside_layer`` lie off the layer, or outside the slab
    its orbitals span, and that their band representations are left out.
    Returns ``None`` where ``outside_layer`` is empty.
    """
    outside = sorted(restriction.outside_layer)
    if not outside:
        return None
    lowest, highest = restriction._layer_extent or (0.0, 0.0)
    off = "off the layer" if highest - lowest <= ROUNDING else "outside the slab"
    return (f"{_text._listed(outside)} lie {off}, so their band representations are left "
            "out")


@dataclass(frozen=True)
class Completeness:
    """Whether a decomposition's sums are those of the whole symmetry data vector.

    Held in :attr:`EBRDecompositionNumerical.completeness`.

    Attributes
    ----------
    source : {"HSPTable", "mapping"}
        ``"HSPTable"`` for a symmetry data vector measured in a table,
        ``"mapping"`` for one written out by hand.
    complete : bool
        Whether the table refused at no point.  ``True`` for a vector written
        out by hand, which is taken as complete.
    claim : {"exact", "necessary-condition"}
        ``"exact"`` where the sums are those of the whole symmetry data vector.
        ``"necessary-condition"`` where a little-group representation left out
        repeats no measured one.  The sums then carry the symmetry data vector
        on the little-group representations measured, and may not carry the
        whole vector.
    refused : dict
        If nonempty, ``{point: gate}``: each point at which the table refused,
        with the name of its refusal.  The decomposition leaves those points
        out.
    note : str
        What the record holds, in a sentence.
    """

    source: str
    complete: bool
    claim: str
    refused: dict
    note: str


@dataclass(frozen=True)
class Enumeration:
    """The cap on the number of sums listed, and whether the list stops at it.

    Held in :attr:`EBRDecompositionNumerical.enumeration`.

    Attributes
    ----------
    cap : int
        The largest number of sums listed.
    n_solutions : int
        The number of sums listed.
    truncated : bool
        Whether there are more sums than the cap, so that the list stops
        short of them.
    """

    cap: int
    n_solutions: int
    truncated: bool


@dataclass(frozen=True)
class SignedCombinations:
    """The combinations of elementary band representations with signed multiplicities.

    Held in :attr:`EBRDecomposition.signed_combinations`, where no sum with
    nonnegative multiplicities carries the symmetry data vector.  Each
    multiplicity is searched between minus and plus the bound,
    :attr:`EBRDecompositionNumerical.bound`.  No bound holds for a signed
    multiplicity, so a combination may have a multiplicity beyond the bound
    searched.

    Attributes
    ----------
    sums : tuple of ..symmetry.BandRepresentation
        If nonempty, each combination found that carries the symmetry data
        vector, up to the cap on the list.
    truncated : bool
        Whether there are more combinations than the cap, so that the list
        stops short of them.
    """

    sums: tuple
    truncated: bool


@dataclass(frozen=True)
class EBRDecompositionNumerical:
    """How a decomposition was searched, and what it was read on.

    Held in :attr:`EBRDecomposition.numerical`.  ``summary()`` prints the
    points left out, and whether the sums are exact without them.

    Attributes
    ----------
    bound : int
        The largest multiplicity searched: the largest entry of the symmetry
        data vector on the little-group representations read, and at least 1.
    bound_is_rigorous : bool
        Whether every elementary band representation carries a little-group
        representation read.  Where it does, no sum has a multiplicity above
        the bound, so a search that finds no sum proves there is none.
    enumeration : Enumeration
        The cap on the number of sums listed, and whether the list stops at it.
    completeness : Completeness
        Whether the sums are exact or rest on the little-group representations
        kept alone, and why.
    restriction : Restriction
        The little-group representations the vector was read on, and those
        left out.
    """

    bound: int
    bound_is_rigorous: bool
    enumeration: Enumeration
    completeness: Completeness
    restriction: Restriction
    _kind: str = field(default="coreps", repr=False, compare=False)

    def summary(self) -> str:
        """What the decomposition was read on, a line for each part with its attribute."""
        lines = []
        where = _where_line(self.restriction, self.completeness, self._kind)
        if where is not None:
            lines.append(where + "   (.restriction, .completeness)")
        outside = _outside_line(self.restriction)
        if outside is not None:
            lines.append(outside + "   (.restriction.outside_layer)")
        if where is None:
            lines.append(f"every tabulated point is measured, and no {self._kind[:-1]} is "
                         "left out   (.restriction)")
        if self.completeness.source == "mapping":
            lines.append("the symmetry data vector was written out by hand: its completeness "
                         "is taken as stated   (.completeness)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, convenience only
        return (f"EBRDecompositionNumerical(bound {self.bound}"
                + (", rigorous" if self.bound_is_rigorous else "")
                + f", {self.completeness.claim})")


@dataclass(frozen=True)
class EBRDecomposition:
    """The sums of elementary band representations that carry a symmetry data vector.

    The record :func:`decompose` returns.  ``summary()`` prints whether a sum
    carries the vector, and each sum.  The record is truthy even where no sum
    carries the vector.

    Attributes
    ----------
    decomposable : bool
        Whether a sum of elementary band representations with nonnegative
        integer multiplicities carries the vector.
    candidate_sums : tuple of ..symmetry.BandRepresentation
        If nonempty, every sum that carries the vector, up to the cap on the
        list.  The sums with the fewest elementary band representations come
        first.  Of two with as many, the one with more copies of the first
        elementary band representation the tables list comes first, then of
        the second, and so on.  Where the list stops at the cap, the sums
        listed are the first the search found, not the first in this order.
        On a layer, the sums leave out the elementary band representations
        induced from Wyckoff positions off the layer.
    symmetry_data : dict
        The symmetry data vector decomposed, ``{corep: multiplicity}``.
    signed_combinations : SignedCombinations or None
        If not ``None``, the combinations of elementary band representations
        that carry the vector, with each multiplicity of either sign and at
        most the bound in size, up to the cap on the list.  ``None`` where a
        sum with nonnegative multiplicities carries the vector, and with
        ``signed_combinations=False``.
    certificate : {"proof", "none within the bound"} or None
        If not ``None``, what the absence of a sum rests on.  ``"proof"``: no
        multiplicity can exceed the bound searched, and no sum within it
        carries the vector.  ``"none within the bound"``: no sum within the
        bound carries the vector, and a sum with a larger multiplicity is not
        ruled out.  ``None`` where a sum carries the vector.
    site_content : SiteContent or None
        If not ``None``, the site irrep of each orbital, and how each is
        known, on which the names of the sums rest.  ``None`` where the
        representation holds no site irreps.
    numerical : EBRDecompositionNumerical
        The bound searched, the cap on the list, and the little-group
        representations the vector was read on.
    """

    decomposable: bool
    candidate_sums: tuple
    symmetry_data: dict
    signed_combinations: SignedCombinations | None
    certificate: str | None
    site_content: SiteContent | None
    numerical: EBRDecompositionNumerical
    _status: str = field(default="", repr=False, compare=False)
    _group: Any = field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        """Whether a sum carries the vector, a line for each part with its attribute."""
        kind = _kind(self._group)
        n = len(self.candidate_sums)
        numerical = self.numerical
        truncated = numerical.enumeration.truncated
        if self.decomposable:
            many = ("at least " if truncated else "") + str(n)
            lines = [f"decomposable: {many} sum{'' if n == 1 else 's'} of elementary band "
                     f"representations carr{'ies' if n == 1 else 'y'} this symmetry data "
                     "vector   (.decomposable, .candidate_sums"
                     + (", .numerical.enumeration" if truncated else "") + ")"]
        else:
            lines = ["not decomposable: no sum of elementary band representations carries "
                     "this symmetry data vector   (.decomposable)"]
        lines.append(f"  the {kind} at the high-symmetry points: "
                     f"{_sewing.format_coreps(self.symmetry_data)}   (.symmetry_data)")
        if self.candidate_sums:
            lines.append("  every candidate sum, each elementary band representation written as "
                         "the site irrep inducing it at its Wyckoff position (site symmetry)   "
                         "(.candidate_sums)")
            lines += _label_lines(self.candidate_sums)
        times = _times(numerical.bound)
        if self.certificate == "proof":
            lines.append(f"  proved: no elementary band representation can occur more than "
                         f"{times}, and no sum with each at most {times} carries it   "
                         "(.certificate, .numerical.bound)")
        elif self.certificate is not None:
            lines.append(f"  not proved: no sum with each elementary band representation at "
                         f"most {times} carries it, and a sum with more is not ruled out   "
                         "(.certificate, .numerical.bound)")
        if self.signed_combinations is not None:
            lines.append("  " + _signed_sums_line(self.signed_combinations, numerical.bound))
        where = _where_line(numerical.restriction, numerical.completeness, kind)
        if where is not None and (numerical.completeness.refused
                                  or not numerical.restriction.exact):
            lines.append("  " + where + "   (.numerical.restriction, .numerical.completeness)")
        if self.candidate_sums and self.site_content is not None and "lowest-l" in tuple(
                self.site_content.determined_by):
            lines.append("  some site irreps were assigned by the lowest-angular-momentum rule: "
                         "a different declared content renames them in these sums, while the "
                         "Wyckoff positions stay   (.site_content)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, convenience only
        n = len(self.candidate_sums)
        return (f"EBRDecomposition({'decomposable' if self.decomposable else 'not decomposable'}"
                f", {n} candidate sum{'' if n == 1 else 's'}, "
                + ("exact)" if self.numerical.completeness.claim == "exact"
                   else "necessary condition only)"))


def _signed_sums_line(signed, bound) -> str:
    """The line of a summary that says what the signed search found.

    ``signed`` is the :class:`SignedCombinations` found by the search with
    multiplicities of either sign, and ``bound`` the bound each multiplicity
    was searched within, :attr:`EBRDecompositionNumerical.bound`.  Each
    combination found has a negative multiplicity, since the search with
    nonnegative multiplicities within the same bound found none.  The line
    counts them, "at least" where the list stops at its cap, or says that
    none was found with multiplicities between ``-bound`` and ``bound``.
    Returns the line.
    """
    n = len(signed.sums)
    if n:
        many = (f"{'at least ' if signed.truncated else ''}{n} "
                f"combination{'' if n == 1 else 's'}")
        verb = "reproduces" if n == 1 else "reproduce"
        return (f"{many} of elementary band representations with some negative "
                f"multiplicities {verb} it   (.signed_combinations)")
    bound = int(bound)
    return (f"no combination of elementary band representations with multiplicities "
            f"between -{bound} and {bound} reproduces it   "
            "(.signed_combinations, .numerical.bound)")


@dataclass(frozen=True)
class RegularCompletion:
    r"""The bands completing a band set to copies of the regular band representation.

    The record :func:`regular_completion` returns.  ``summary()`` prints the
    symmetry data vector of the missing bands, and the orbitals they would be.

    Attributes
    ----------
    copies : int
        The number of copies of the regular band representation the completed
        bands make.
    bands_per_copy : int
        The number of bands in one copy: the number of unitary operations of
        the group up to lattice translations, or twice that for a spinful
        model in a group with time reversal.  For a spinful model these are
        the operations of the single-valued group.
    padding : dict
        If nonempty, the symmetry data vector of the missing bands,
        ``{corep: multiplicity}``: :attr:`copies` times :attr:`regular`, less
        :attr:`symmetry_data`.
    decomposition : EBRDecomposition
        What :func:`decompose` returns for :attr:`padding`: the orbitals the
        missing bands would be.
    symmetry_data : dict
        The symmetry data vector of the band set handed in.
    regular : dict
        The symmetry data vector of one copy, computed from the characters of
        the little-group representations at each point.
    """

    copies: int
    bands_per_copy: int
    padding: dict
    decomposition: EBRDecomposition
    symmetry_data: dict
    regular: dict
    _group: Any = field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        """The padding, a line for each part, with the attribute that holds it."""
        copies = "one copy" if self.copies == 1 else f"{self.copies} copies"
        lines = [f"{copies} of the regular band representation, the one induced from the "
                 f"general position: {self.bands_per_copy} bands at every "
                 f"k{'' if self.copies == 1 else ' each'}   (.copies, .bands_per_copy)"]
        if not self.padding:
            lines.append(f"  nothing is missing: the bands carry those {_kind(self._group)} already"
                         "   (.padding)")
            return "\n".join(lines)
        lines.append(f"  the {_kind(self._group)} the missing bands carry: "
                     f"{_sewing.format_coreps(self.padding)}   (.padding)")
        if self.decomposition.candidate_sums:
            lines.append("  the orbitals they would be, each the site irrep at its Wyckoff "
                         "position (site symmetry)   (.decomposition.candidate_sums)")
            lines += _label_lines(self.decomposition.candidate_sums)
        else:
            lines.append("  no sum of elementary band representations carries them   "
                         "(.decomposition)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, convenience only
        return (f"RegularCompletion({self.copies} cop{'y' if self.copies == 1 else 'ies'} of "
                f"{self.bands_per_copy} bands, padding "
                f"{_sewing.format_coreps(self.padding) or 'none'}, "
                + ("decomposable)" if self.decomposition.decomposable
                   else "not decomposable)"))


#: The statement of the convention every indicator record carries.
INDICATOR_CONVENTION = (
    "the indicator is evaluated in this package's physical convention, in which a "
    "lattice translation is represented by e^(-ik.t). The tables, written in the BCS "
    "convention -- and therefore irreptables' EBR rows and its shipped Smith factors "
    "u, v -- use the historical e^(+ik.t), whose labels at k are those of the physical "
    "representations at -k; groups.ebr_data undoes that swap on the rows and on u "
    "and v together, so the linear form u.y is the same functional here as there, "
    "read on relabeled coordinates. The indicator group is therefore convention-"
    "invariant by Smith equivalence, and so is every indicator value: measured over "
    "the 379 unitary and grey systems with a nontrivial indicator group, the two "
    "computations agree exactly -- the automorphism relating them is the identity, "
    "in every group. Wyckoff "
    "positions never move. What is not convention-invariant is a linear form written "
    "in the BCS convention's labels: irreptables' named indicator formulas have to be "
    "re-keyed by the row permutation of the change of convention before they can be "
    "read on a returned symmetry data vector."
)


@dataclass(frozen=True)
class SmithVerification:
    """The Smith normal form of the group's matrix, computed again in exact integers.

    Held in :attr:`SymmetryIndicatorNumerical.verification`.  The matrix is
    the group's matrix of elementary band representations, whose columns are
    their symmetry data vectors.  Its Smith normal form is computed again and
    compared with the one the irreptables package ships with it.  A record
    returned holds two forms with the same elementary divisors, in which the
    class has the same order.

    Attributes
    ----------
    divisors_denovo : tuple of int
        The elementary divisors above 1 of the form computed again.
    agrees : bool
        Whether the two forms have the same elementary divisors and give the
        class the same order.  ``True`` in a record returned.
    shipped_is_smith : bool
        Whether the two unimodular matrices the irreptables package ships
        bring the matrix to the diagonal form it ships.  ``True`` in a record
        returned.
    denovo_is_smith : bool
        Whether the form computed again is a Smith normal form of the matrix.
        ``True`` in a record returned.
    values_denovo : tuple of int
        The class read in the form computed again: the entries of its
        ``U y`` (:attr:`SymmetryIndicator.values`) at its elementary divisors
        above 1, each modulo that divisor.
    order_denovo : int
        The order of the class, computed again.
    change_of_basis : tuple
        The automorphism of the indicator group that carries
        :attr:`SymmetryIndicator.values` to :attr:`values_denovo`: a matrix
        with a row for each entry of :attr:`values_denovo` and a column for
        each entry of :attr:`SymmetryIndicator.values`, each element modulo
        the elementary divisor of its row.
    change_of_basis_is_identity : bool
        Whether that automorphism is the identity.
    note : str
        What the record holds, in a sentence.
    """

    divisors_denovo: tuple
    agrees: bool
    shipped_is_smith: bool
    denovo_is_smith: bool
    values_denovo: tuple
    order_denovo: int
    change_of_basis: Any
    change_of_basis_is_identity: bool
    note: str


@dataclass(frozen=True)
class LatticeCheck:
    """The compatibility relations, checked exactly on the symmetry data vector.

    Held in :attr:`SymmetryIndicatorNumerical.lattice_check`.  The
    compatibility relations require the entries of ``U y`` at the rows where
    ``D`` is zero to vanish for every band structure of the group.
    Here ``U M V = D`` is the Smith normal form the irreptables package ships
    for the group's matrix ``M`` of elementary band representations, and
    ``y`` is the symmetry data vector.

    Attributes
    ----------
    in_lattice : bool
        Whether the symmetry data vector satisfies every compatibility
        relation.  ``True`` in a record returned.
    residual : int
        The largest absolute value of an entry of ``U y`` that a compatibility
        relation requires to vanish.  Zero in a record returned.
    relations : int
        The number of compatibility relations of the group.
    """

    in_lattice: bool
    residual: int
    relations: int


@dataclass(frozen=True)
class IndicatorComponent:
    """One component of a layer's symmetry-indicator class.

    Held in :attr:`TwoDimensionalReading.components`.  A component is one
    entry of ``U y``, taken modulo its elementary divisor
    (:attr:`SymmetryIndicator.values`).

    Attributes
    ----------
    slot : int
        The index of the entry of ``U y``.
    divisor : int
        The elementary divisor the component is taken modulo.
    value : int
        The component, modulo :attr:`divisor`.
    reads_out_of_plane : bool
        Whether the entry, a linear form in the symmetry data vector, has a
        nonzero coefficient on a little-group representation at a point out of
        the layer's plane.
    """

    slot: int
    divisor: int
    value: int
    reads_out_of_plane: bool


@dataclass(frozen=True)
class TwoDimensionalReading:
    """How a layer's symmetry-indicator class is read, as the class of its stack.

    Held in :attr:`SymmetryIndicatorNumerical.two_dimensional`.  A layer's
    class is read in a three-dimensional space group, whose tables have points
    out of the layer's plane, at ``k_z = 1/2``.  It is the class of the
    layer's decoupled stack, the layer repeated along its normal with no
    hopping between copies.

    Attributes
    ----------
    applied : bool
        ``True`` in every record returned.
    dim_k : int
        The number of the layer's periodic directions.
    periodic_dirs : tuple of int
        The layer's periodic directions, by lattice vector.
    vacuum_dirs : tuple of int
        The lattice vector along the layer's normal.
    presentation : {"SLAB", "decoupled-STACK"}
        ``"decoupled-STACK"`` where the little-group representations out of
        the plane were measured on the decoupled stack, ``"SLAB"`` where none
        was.  The class is read in the layer's own group in both cases.
    out_of_plane_kpoints : tuple of str
        The tabulated points out of the layer's plane.
    out_of_plane_labels : tuple of str
        The little-group representations at those points.
    components : tuple of IndicatorComponent
        Each component of the class, and whether it has a coefficient on a
        little-group representation out of the plane.
    support : dict
        If nonempty, ``{indicator: where}``: for each named indicator, as
        :attr:`SymmetryIndicator.indicators` names it, the little-group
        representations on which its coefficients are nonzero.
        ``"in-plane only"`` for those in the plane alone,
        ``"out-of-plane only"`` for those out of it alone, ``"mixed"`` for
        both.
    note : str
        What the record holds, in a sentence.
    """

    applied: bool
    dim_k: int
    periodic_dirs: tuple
    vacuum_dirs: tuple
    presentation: str
    out_of_plane_kpoints: tuple
    out_of_plane_labels: tuple
    components: tuple
    support: dict
    note: str


@dataclass(frozen=True)
class HamiltonianIdentity:
    """The decoupled stack's Hamiltonian compared with the layer's.

    Held in :attr:`StackChecks.hamiltonian_identity`.  Each orbital sits at
    its height ``h`` along the normal, and the stack's period is ``C``.  The
    stack's ``H(k, k_z)`` is then the layer's ``H(k)`` in the gauge of the
    phases ``exp(2 pi i k_z h / C)`` of the orbitals.  The two are compared in
    that gauge.

    Attributes
    ----------
    residual : float
        The largest difference between the two over the in-plane k-points and
        the values of ``k_z`` probed, in the units of H.
    threshold : float
        The largest residual allowed, in the units of H: zero for a layer
        whose orbitals lie in one plane, otherwise the rounding the phases
        carry.
    at_zero : float
        The difference at ``k_z = 0``, where every phase is 1.  Zero in a
        record returned.
    n_k : int
        The number of in-plane k-points probed.
    k_z : tuple of float
        The values of ``k_z`` probed, in reduced coordinates.
    note : str
        What the record holds, in a sentence.
    """

    residual: float
    threshold: float
    at_zero: float
    n_k: int
    k_z: tuple
    note: str


@dataclass(frozen=True)
class StackHoppings:
    """The hoppings between the copies of the decoupled stack: none.

    Held in :attr:`StackChecks.no_hopping_across_the_stack`.

    Attributes
    ----------
    n_hoppings : int
        The number of the layer's hoppings.
    worst : float
        The largest hopping between copies.  Zero in a record returned.
    note : str
        What the record holds, in a sentence.
    """

    n_hoppings: int
    worst: float
    note: str


@dataclass(frozen=True)
class RepresentationCheck:
    """The matrices of the layer's representation, checked on the stack's Hamiltonian.

    Held in :attr:`StackChecks.representation`.

    Attributes
    ----------
    covariance : float
        The largest covariance residual, ``max |D H(k) D^-1 - H(Rk)|`` over
        every operation and the k-points probed, in the units of H.  Here
        ``D`` is the operation's matrix on the orbitals and ``R`` its
        rotation.
    threshold : float
        The largest residual allowed, the rounding the arithmetic of the
        matrices carries, in the units of H.
    n_operations : int
        The number of operations checked.
    n_k : int
        The number of k-points probed.
    note : str
        What the record holds, in a sentence.
    """

    covariance: float
    threshold: float
    n_operations: int
    n_k: int
    note: str


@dataclass(frozen=True)
class CharactersCheck:
    """The band characters of the layer and of its stack at the k-points they share.

    Held in :attr:`StackChecks.characters`.  Each point of the layer's table
    is paired with the point of the stack's table in the layer's plane at the
    same reduced k.  The characters of the bands are compared there,
    operation by operation.

    Attributes
    ----------
    worst : float
        The largest difference between two characters compared.
    threshold : float
        The largest difference allowed.
    n_comparisons : int
        The number of characters compared.
    shared_kpoints : tuple
        ``(layer's point, stack's point)`` for each pair compared, by
        tabulated name.
    labels_equal : bool
        Whether the two tables give the same little-group representations,
        with the same multiplicities, at each pair.
    note : str
        What the record holds, in a sentence.
    """

    worst: float
    threshold: float
    n_comparisons: int
    shared_kpoints: tuple
    labels_equal: bool
    note: str


@dataclass(frozen=True)
class StackFiber:
    """The stack's bands at one point at ``k_z = 1/2``, against those in the plane.

    Held in :attr:`StackChecks.fibers`.  The bands at a point at
    ``k_z = 1/2`` are compared with those at the point of the plane with the
    same in-plane k.  With no hopping between the copies, the two are the
    same states, with the same energies, multiplets and characters.

    Attributes
    ----------
    kpoint : str
        The tabulated name of the point at ``k_z = 1/2``.
    in_plane_kpoint : str
        The tabulated name of the point in the plane it is compared with.
    energy_defect : float
        The largest difference between the energies at the two points, in the
        units of H.  Infinite where the two hold different numbers of
        multiplets.
    character_defect : float
        The largest difference between the characters at the two points, over
        the operations both little groups hold.
    n_shared_operations : int
        The number of operations both little groups hold.
    partition_equal : bool
        Whether the bands at the two points split into multiplets alike.
    multiplicities_equal : bool
        Whether the two points carry the same multiplicities, whichever
        little-group representations carry them.
    """

    kpoint: str
    in_plane_kpoint: str
    energy_defect: float
    character_defect: float
    n_shared_operations: int
    partition_equal: bool
    multiplicities_equal: bool


@dataclass(frozen=True)
class StackChecks:
    """Each check that ties the stack to the layer, with the number it measured.

    Held in :attr:`StackCompletion.gates`.

    Attributes
    ----------
    hamiltonian_identity : HamiltonianIdentity
        The stack's Hamiltonian compared with the layer's.
    no_hopping_across_the_stack : StackHoppings
        The hoppings between the copies.
    representation : RepresentationCheck
        The layer's matrices checked on the stack's Hamiltonian.
    characters : CharactersCheck
        The band characters of the layer and of the stack at the k-points
        they share.
    fibers : tuple of StackFiber
        The stack's bands at each point at ``k_z = 1/2``, compared with those
        at the point of the plane with the same in-plane k.
    """

    hamiltonian_identity: HamiltonianIdentity
    no_hopping_across_the_stack: StackHoppings
    representation: RepresentationCheck
    characters: CharactersCheck
    fibers: tuple


@dataclass(frozen=True)
class StackInput:
    """The layer whose decoupled stack was built, as handed in.

    Held in :attr:`StackCompletion.input`.

    Attributes
    ----------
    group : ..groups.NamedGroup
        The layer's group.
    dim_k : int
        The number of the layer's periodic directions.
    periodic_dirs : tuple of int
        The layer's periodic directions, by lattice vector.
    vacuum_dirs : tuple of int
        The lattice vector along the layer's normal.
    kpoints : tuple of str
        The tabulated names of the points the layer's table measured.
    orphans : tuple of str
        The little-group representations out of the layer's plane that repeat
        none in it, as :attr:`Restriction.orphans` holds them.
    symmetry_data : dict
        The layer's symmetry data vector, ``{corep: multiplicity}``.
    n_represented : int
        The number of operations the layer's representation holds a matrix
        for.
    """

    group: Any
    dim_k: int
    periodic_dirs: tuple
    vacuum_dirs: tuple
    kpoints: tuple
    orphans: tuple
    symmetry_data: dict
    n_represented: int


@dataclass(frozen=True)
class StackGroup:
    """The decoupled stack's group: the layer's own, carried onto the stack's cell.

    Held in :attr:`StackCompletion.stack`.

    Attributes
    ----------
    group : ..groups.NamedGroup
        The group.
    dim_k : int
        The number of the stack's periodic directions, three.
    periodic_dirs : tuple of int
        The stack's periodic directions, by lattice vector.
    typat : tuple of int
        The species of each orbital's site.
    divisors : tuple of int
        The elementary divisors above 1 of the group's matrix of elementary
        band representations.
    """

    group: Any
    dim_k: int
    periodic_dirs: tuple
    typat: tuple
    divisors: tuple


@dataclass(frozen=True)
class StackCompletion:
    """The measurement of a layer's little-group representations at ``k_z = 1/2``.

    Held in :attr:`SymmetryIndicatorNumerical.completion`.  The decoupled
    stack is the layer repeated along its normal with no hopping between
    copies, each orbital at its own height.  It obeys the layer's own group,
    and the little-group representations at ``k_z = 1/2`` are measured on it.

    Attributes
    ----------
    route : str
        ``"decoupled stack"``.
    outcome : str
        ``"completed"``.
    stopped_at : None
        Where the measurement stopped.  ``None`` in a record returned.
    stop_reason : None
        Why the measurement stopped.  ``None`` in a record returned.
    stack_period : float
        The stack's period, in the model's units of length: the layer's
        extent along its normal plus the vacuum between the copies.
    stack_extent : float
        The layer's extent along its normal, in the model's units of length.
    environment : str or None
        The layer's environment, ``"free-standing"`` or ``"substrate"``.
    note : str
        What the record holds, in a sentence.
    input : StackInput
        The layer as handed in.
    stack : StackGroup
        The stack's group.
    gates : StackChecks
        Each check that ties the stack to the layer, with the number it
        measured.
    """

    route: str
    outcome: str
    stopped_at: Any
    stop_reason: Any
    stack_period: float
    stack_extent: Any
    environment: Any
    note: str
    input: StackInput
    stack: StackGroup
    gates: StackChecks


def _completion_record(record) -> StackCompletion:
    """The :class:`StackCompletion` of a layer's decoupled stack, from its mapping.

    ``record`` is the mapping :func:`_stack_in_group` builds for a completed
    stack: the record's own fields, with ``"input"``, ``"stack"`` and
    ``"gates"`` as mappings of their own.  Returns the record, each of those
    three parts a record of its class.
    """
    from ._realized import NamedGroup

    def named(part):
        return NamedGroup(name=str(part["name"]), number_str=str(part["number_str"]),
                          size=int(part["size"]))

    given, stack, gates = record["input"], record["stack"], record["gates"]
    identity, hops = gates["hamiltonian_identity"], gates["no_hopping_across_the_stack"]
    rep, chars = gates["representation"], gates["characters"]
    return StackCompletion(
        route=record["route"], outcome=record["outcome"], stopped_at=record["stopped_at"],
        stop_reason=record["stop_reason"], stack_period=record["stack_period"],
        stack_extent=record["stack_extent"], environment=record["environment"],
        note=record["note"],
        input=StackInput(
            group=named(given), dim_k=given["dim_k"], periodic_dirs=given["periodic_dirs"],
            vacuum_dirs=given["vacuum_dirs"], kpoints=given["points"],
            orphans=given["orphans"], symmetry_data=given["content"],
            n_represented=given["n_represented"]),
        stack=StackGroup(group=named(stack), dim_k=stack["dim_k"],
                         periodic_dirs=stack["periodic_dirs"], typat=stack["typat"],
                         divisors=stack["divisors"]),
        gates=StackChecks(
            hamiltonian_identity=HamiltonianIdentity(
                residual=identity["residual"], threshold=identity["threshold"],
                at_zero=identity["at_zero"], n_k=identity["n_k"], k_z=identity["k_z"],
                note=identity["note"]),
            no_hopping_across_the_stack=StackHoppings(
                n_hoppings=hops["n_hoppings"], worst=hops["worst"], note=hops["note"]),
            representation=RepresentationCheck(
                covariance=rep["covariance"], threshold=rep["threshold"],
                n_operations=rep["n_operations"], n_k=rep["n_k"], note=rep["note"]),
            characters=CharactersCheck(
                worst=chars["worst"], threshold=chars["threshold"],
                n_comparisons=chars["n_comparisons"], shared_kpoints=chars["shared_points"],
                labels_equal=chars["labels_equal"], note=chars["note"]),
            fibers=tuple(StackFiber(
                kpoint=f["point"], in_plane_kpoint=f["in_plane_point"],
                energy_defect=f["energy_defect"], character_defect=f["character_defect"],
                n_shared_operations=f["n_shared_operations"],
                partition_equal=f["partition_equal"],
                multiplicities_equal=f["multiplicities_equal"])
                for f in gates["fibers"])))


@dataclass(frozen=True)
class SymmetryIndicatorNumerical:
    """How a symmetry-indicator class was read.

    Held in :attr:`SymmetryIndicator.numerical`.  ``summary()`` prints the
    symmetry data vector on every point of the group's tables, how the
    little-group representations a layer does not measure were obtained, and
    the check of the Smith normal form.

    Attributes
    ----------
    verification : SmithVerification
        The Smith normal form of the group's matrix of elementary band
        representations, computed again in exact integers.
    lattice_check : LatticeCheck
        The compatibility relations, checked on the symmetry data vector.
    restriction : Restriction
        The little-group representations measured, and those left out.
    reconstructed : dict
        If nonempty, ``{left out: measured}``: each little-group
        representation left out that the symmetry data vector does not name,
        and the measured one whose multiplicity it takes.  The two occur as
        often as each other in every elementary band representation the
        vector was read on.
    completed_symmetry_data : dict
        The symmetry data vector on every point of the group's tables,
        ``{corep: multiplicity}``, the zeros dropped.  A layer's little-group
        representations out of its plane are measured on its decoupled stack,
        or take the multiplicity of the one in the plane they repeat.  A
        vector written out by hand is read as given at each point out of the
        plane where it names a little-group representation.
    completion : StackCompletion or None
        If not ``None``, how a layer's little-group representations out of its
        plane were measured on its decoupled stack.  ``None`` where none had
        to be.
    two_dimensional : TwoDimensionalReading or None
        If not ``None``, how a layer's class was read.  ``None`` for a crystal
        periodic in three directions, for a trivial indicator group, and where
        there is no point out of the layer's plane to read.
    free_standing_group : ..groups.NamedGroup or None
        If not ``None``, the group of the same layer free-standing, with
        vacuum on both sides, as :func:`.groups.realized_group` measured it
        beside the group on a substrate.  It is given where it is larger than
        the group in which the class is computed.
    convention : str
        :data:`INDICATOR_CONVENTION`, a statement that the class is computed
        in the convention in which a lattice translation is represented by
        ``e^(-ik.t)``, and is the same in the convention of the tables, in
        which a lattice translation is represented by ``e^(+ik.t)``.
    indicators_note : str
        Where :attr:`SymmetryIndicator.indicators` came from, or why it is
        ``None``.
    group_provenance : str
        ``"realized"`` where the group is that of the model's cell;
        ``"descended"`` where it was lowered from the group of the cell by the
        operations the Hamiltonian breaks.
    """

    verification: SmithVerification
    lattice_check: LatticeCheck
    restriction: Restriction
    reconstructed: dict
    completed_symmetry_data: dict
    completion: StackCompletion | None
    two_dimensional: TwoDimensionalReading | None
    free_standing_group: Any
    convention: str
    indicators_note: str
    group_provenance: str
    _group: Any = field(default=None, repr=False, compare=False)

    def summary(self) -> str:
        """How the class was read, a line for each part, with the attribute that holds it."""
        group = self._group
        kind = _kind(group)
        lines = []
        hsp = (_tables._high_symmetry_points(group)
               if group is not None and int(getattr(group, "_dim_k", 3)) < 3 else None)
        away = () if hsp is None or isinstance(hsp, Refusal) else tuple(hsp.out_of_plane)
        if away:
            lines.append(f"the {kind} at every point of the group's tables, "
                         f"{_text._listed(away)} completed: "
                         f"{_sewing.format_coreps(self.completed_symmetry_data)}   "
                         "(.completed_symmetry_data)")
        if self.reconstructed:
            points = sorted({kpoint_of_label(label) for label in self.reconstructed},
                            key=_tables.point_order)
            lines.append(f"the {kind} at {_text._listed(points)} are not measured: each "
                         "occurs in every elementary band representation as often as a "
                         "measured one, and takes its multiplicity   (.reconstructed)")
            lines.append("  " + ", ".join(f"{label} as {source}" for label, source
                                          in self.reconstructed.items()))
        comp = self.completion
        if comp is not None:
            points = sorted({kpoint_of_label(label) for label in comp.input.orphans},
                            key=_tables.point_order)
            lines.append(f"{_text._listed(points)}, at k_z = 1/2 out of the layer's plane: "
                         f"their {kind} measured on the layer's decoupled stack, the layer "
                         "repeated along its normal with no hopping between copies, in the "
                         f"layer's own group {comp.input.group.name} "
                         f"#{comp.input.group.number_str}   (.completion)")
        if self.two_dimensional is not None:
            lines.append("the class of the layer's decoupled stack"
                         + ("" if comp is not None else ", the layer repeated along its "
                            "normal with no hopping between copies")
                         + ": every component is an invariant of that stack, fixed by the "
                         "layer's own symmetry data vector   (.two_dimensional)")
        if self.free_standing_group is not None:
            free = self.free_standing_group
            lines.append(
                "the free-standing layer's group, as realized_group measured it, is "
                f"{free.name} #{free.number_str}, larger than this one: the same layer with "
                "vacuum on both sides, whose class is asked with "
                "environment='free-standing'   (.free_standing_group)")
        lines.append("the Smith normal form of the group's matrix of elementary band "
                     "representations, computed again exactly, gives the same elementary "
                     "divisors and the same order of the class   (.verification)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, convenience only
        return (f"SymmetryIndicatorNumerical({self.group_provenance}"
                + (", completed on the stack" if self.completion is not None else "") + ")")


@dataclass(frozen=True)
class SymmetryIndicator:
    r"""The symmetry-indicator class of a set of bands.

    The record :func:`symmetry_indicator` returns.  ``summary()`` prints the
    class, its indicator group and its order, the class in the named
    indicators, and the symmetry data vector.  The record is truthy whatever
    its class.

    Attributes
    ----------
    values : tuple of int
        If nonempty, the class, one integer modulo each of :attr:`divisors`:
        the entries of ``U y`` at which ``D`` holds an elementary divisor
        above 1, each modulo that divisor.  Here ``U M V = D`` is the Smith
        normal form the irreptables package ships for the group's matrix
        ``M`` of elementary band representations, and ``y`` is the symmetry
        data vector.  Another Smith normal form of the same matrix can give
        the same class other values, and the same order.  Empty for a trivial
        indicator group.
    divisors : tuple of int
        If nonempty, the elementary divisors above 1 of ``M``.  The indicator
        group is the product of the cyclic groups of these orders.  Empty for
        a trivial indicator group.
    indicator_group : str
        The indicator group as text, ``"Z2 x Z4"``, or ``"trivial"``.
    indicators : dict or None
        If not ``None``, the class in the named indicators of Hwang *et al.*,
        Nat. Commun. **17**, 8564 (2026), ``{name: value}``.  They are
        computed from the integer-valued stable real-space invariants (ZSRSIs)
        of the symmetry data vector.  ``None`` in three cases: a group with
        no time reversal, which the paper does not tabulate; a trivial
        indicator group; and a group for which the paper's tables are not
        installed or cannot be read.
        :attr:`SymmetryIndicatorNumerical.indicators_note` says which.
    order : int
        The order of the class: the smallest number of copies of the set of
        bands that together have the zero class, ``1`` for the zero class.
    symmetry_data : dict
        The symmetry data vector, ``{corep: multiplicity}``.  For a layer, the
        multiplicities at the points of its plane alone.
    site_content : SiteContent or None
        If not ``None``, the site irrep of each orbital, and how each is
        known, on which the names of the little-group representations rest.
        Where one is ``"lowest-l"``, another declaration of the orbitals can
        give the bands another class.  ``None`` where the representation holds
        no site irreps.
    group : ..groups.GroupData
        The group in which the class is computed, ``r.group``.
    numerical : SymmetryIndicatorNumerical
        The Smith normal form computed again, the symmetry data vector on
        every point of the group's tables, and how a layer's little-group
        representations out of its plane were read.
    """

    values: tuple
    divisors: tuple
    indicator_group: str
    indicators: dict | None
    order: int
    symmetry_data: dict
    site_content: SiteContent | None
    group: Any = field(repr=False, compare=False)
    numerical: SymmetryIndicatorNumerical
    # The arithmetic: every little-group representation of the group's elementary
    # band representations, the vector on them, the same as U y, and which
    # entries of U y have an elementary divisor above 1.
    _labels: tuple = field(default=(), repr=False)
    _y: tuple = field(default=(), repr=False)
    _y_prime: tuple = field(default=(), repr=False)
    _slots: tuple = field(default=(), repr=False)

    def summary(self) -> str:
        """The class, a line for each part, with the attribute that holds it."""
        group = f"{self.group.name} #{self.group.number_str}"
        if not self.divisors:
            lines = [f"the indicator group of {group} is trivial   (.indicator_group, .group)"]
        else:
            lines = [f"class {tuple(int(v) for v in self.values)} in {self.indicator_group}, "
                     f"in {group}   (.values, .indicator_group, .group)"]
            if self.order > 1:
                lines.append(f"  of order {self.order}: {self.order} copies of these bands are "
                             "in the zero class   (.order)")
            lines.append(f"  the indicator group: {self.indicator_group}, a class being "
                         f"{_moduli(self.divisors)}   (.indicator_group, .divisors)")
            if self.indicators:
                lines.append("  indicators: " + ", ".join(f"{k} = {v}" for k, v in
                                                          self.indicators.items())
                             + ", the names of the paper that tabulates the stable real-space "
                             "invariants, Hwang et al. (2026)   (.indicators)")
        lines.append(f"  the {_kind(self.group)} at the high-symmetry points: "
                     f"{_sewing.format_coreps(self.symmetry_data)}   (.symmetry_data)")
        if self.divisors and "lowest-l" in tuple(
                getattr(self.site_content, "determined_by", ()) or ()):
            lines.append("  a site irrep was assigned by the lowest-angular-momentum rule: "
                         "another declared content can call these bands another "
                         "class   (.site_content)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover, convenience only
        body = (f"class {tuple(int(v) for v in self.values)} in {self.indicator_group}"
                if self.divisors else "trivial indicator group")
        stacked = (self.numerical is not None and self.numerical.completion is not None)
        return (f"SymmetryIndicator({self.group.name} #{self.group.number_str}, {body}"
                + (", completed on the stack" if stacked else "") + ")")


# --------------------------------------------------------------------------
# the symmetry data vector, and what it is read in
# --------------------------------------------------------------------------

def content_of(content) -> dict | Refusal:
    """The symmetry data vector of a set of bands, as integer multiplicities.

    Parameters
    ----------
    content : .sewing.HSPTable or mapping
        The symmetry data vector.  Pass the table :func:`.sewing.hsp_table`
        returns, or ``{corep: multiplicity}`` written out by hand.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``, each multiplicity an integer, the zeros
        dropped.  A ``Refusal`` in its place is falsy and names what stopped
        the call.

    Refusals
    --------
    ``non_integer_multiplicity``
        an entry is not an integer.
    ``negative_multiplicity``
        an entry is negative.

    Raises
    ------
    TypeError
        ``content`` is neither a table nor a mapping.
    """
    src = content.symmetry_data if isinstance(content, _sewing.HSPTable) else content
    if not isinstance(src, Mapping):
        raise TypeError(f"content_of(): content= is of type {type(content).__name__}; "
                        "pass the table hsp_table returns, or {corep: multiplicity} "
                        "written out by hand.")
    out = {}
    for lab, v in src.items():
        iv = int(round(float(v)))
        if abs(float(v) - iv) > 1e-9:
            return Refusal("non_integer_multiplicity",
                           f"the multiplicity of {lab} is {v}, which is not an integer. "
                           "Pass an integer multiplicity for each little-group "
                           "representation.",
                           value=float(v), context={"label": str(lab)})
        if iv < 0:
            return Refusal("negative_multiplicity",
                           f"the multiplicity of {lab} is {iv}, which is negative. "
                           "Pass a nonnegative multiplicity for each little-group "
                           "representation.",
                           value=float(iv), threshold=0.0,
                           context={"label": str(lab)})
        if iv:
            out[str(lab)] = iv
    return _tables.ordered_content(out)


@dataclass(frozen=True)
class _Source:
    """What a symmetry data vector is read in, and what its measurement carries.

    Attributes
    ----------
    group : ..groups.GroupData
        The group the vector is read in.
    table : ..sewing.HSPTable or None
        The table read; ``None`` for a vector written out by hand.
    rep : ..symmetry.SpaceGroupRep or None
        The representation of the group on the model's orbitals, or ``None``.
        A layer's decoupled stack is built from it.
    site_content : SiteContent or None
        The site irreps the names of the little-group representations rest on.
    heights : tuple of float or None
        Each orbital's height from a layer's middle plane along its normal, in
        the model's units of length.  ``None`` for a crystal, and where the
        group's own sites stand for the orbitals.
    free_standing : dict or None
        If not ``None``, for a layer on a substrate, the group the layer has
        free-standing, ``{"number_str", "name", "size"}``.
    provenance : str or None
        How the table's group was measured, as the table records it; ``None``
        for a vector written out by hand.
    """

    group: Any
    table: Any = None
    rep: Any = None
    site_content: Any = None
    heights: Any = None
    free_standing: Any = None
    provenance: Any = None


def _source_of_table(table) -> _Source:
    """The :class:`_Source` of a vector measured in a table.

    ``table`` is a :class:`.sewing.HSPTable`.  Returns its group, the table,
    its representation and site irreps, a layer's heights, the free-standing
    group its representation carries, and how its group was measured.
    """
    rep = table._rep
    return _Source(group=table.group, table=table, rep=rep,
                   site_content=table._site_content, heights=table._heights,
                   free_standing=getattr(rep, "_free_standing_group", None),
                   provenance=table._group_provenance)


def _source_of_rep(group, rep=None) -> _Source:
    """The :class:`_Source` of a vector written out by hand, read in ``group``.

    ``rep`` is the representation beside the vector, or ``None``.  Returns no
    table and no provenance.  The site irreps, the heights and the
    free-standing group are read off ``rep``, and are ``None`` where it is
    ``None``.
    """
    return _Source(group=group, rep=rep, site_content=_sewing._site_content_record(rep),
                   heights=None if rep is None else _sewing._table_heights(rep),
                   free_standing=getattr(rep, "_free_standing_group", None))


def _source_of(symmetry_data, group, rep=None) -> _Source:
    """The :class:`_Source` of a table, or of a vector handed in with its group.

    ``symmetry_data`` is a :class:`.sewing.HSPTable`, read in what it carries,
    ``group`` and ``rep`` not used; or a vector written out by hand, read
    in ``group`` with ``rep`` beside it (:func:`_source_of_rep`).  Returns the
    :class:`_Source`.
    """
    if isinstance(symmetry_data, _sewing.HSPTable):
        return _source_of_table(symmetry_data)
    return _source_of_rep(group, rep)


def _read(call, symmetry_data, realized) -> _Source | Refusal:
    """The :class:`_Source` a public call reads its first two arguments in.

    ``call`` is the call's name, which each error starts with.
    ``symmetry_data`` is a :class:`.sewing.HSPTable`, passed alone.  Or it is
    a vector written out by hand, ``{corep: multiplicity}``, passed with
    ``realized``, what :func:`.groups.realized_group` returned.  Returns the
    :class:`_Source`, or a refusal handed in as ``realized``, as it is.
    Raises ``TypeError``, saying what to pass, for a table with ``realized``
    beside it, a vector without it, a ``realized`` that is not what
    ``realized_group`` returns, and a ``symmetry_data`` that is neither a
    table nor a mapping.
    """
    from ._realized import RealizedGroup

    if isinstance(symmetry_data, _sewing.HSPTable):
        if realized is not None:
            raise TypeError(f"{call}(): a table is passed with realized= beside it; pass "
                            f"the table alone: {call}(t).")
        return _source_of_table(symmetry_data)
    if not isinstance(symmetry_data, Mapping):
        raise TypeError(f"{call}(): symmetry_data= is of type "
                        f"{type(symmetry_data).__name__}; pass the table hsp_table returns, "
                        "or {corep: multiplicity} written out by hand.")
    if realized is None:
        raise TypeError(f"{call}(): a symmetry data vector written out by hand is passed "
                        "without realized=; pass what realized_group returned for its model "
                        f"second: {call}(symmetry_data, r).")
    if isinstance(realized, Refusal):
        return realized
    if not isinstance(realized, RealizedGroup):
        raise TypeError(f"{call}(): realized= is of type {type(realized).__name__}; pass "
                        f"what realized_group returns second: {call}(symmetry_data, r).")
    return _source_of_rep(realized.group, realized.rep)


def _named_points(group, counts):
    """The points a vector written out by hand is read at.

    ``counts`` is the vector, ``{corep: multiplicity}``, read in ``group``.
    The points are every tabulated point of the model's plane, then each
    tabulated point out of it at which ``counts`` gives a little-group
    representation a nonzero multiplicity.  At such a point, a little-group
    representation ``counts`` does not name is read as zero.  Returns the
    points' names, or the refusal of the group's high-symmetry points.
    """
    hsp = _tables._high_symmetry_points(group)
    if isinstance(hsp, Refusal):
        return hsp
    away = set(hsp.out_of_plane)
    named = {kpoint_of_label(label) for label, m in counts.items() if m} & away
    return tuple(hsp) + tuple(sorted(named, key=_tables.point_order))


def _check_names(call, group, names) -> None:
    """Raise on a point's name in ``names`` that ``group`` does not tabulate.

    ``call`` is the public call's name, which the error starts with.  A name
    of a tabulated point, in the model's plane or out of it, passes.  Raises
    ``ValueError`` (:func:`._tables._resolve_kpoint`) on any other.  Nothing
    is checked where the group's high-symmetry points are refused.
    """
    hsp = _tables._high_symmetry_points(group)
    if isinstance(hsp, Refusal):
        return
    for name in names:
        if name not in hsp and name not in hsp.out_of_plane:
            _tables._resolve_kpoint(call, group, name)


# --------------------------------------------------------------------------
# the in-plane restriction
# --------------------------------------------------------------------------

def in_plane_restriction(group, *, kpoints=None, require_exact=True,
                         _layer=None) -> Restriction | Refusal:
    """The little-group representations a model measures, and what the others repeat.

    The little-group representations of the group's elementary band
    representations are split into those at the points in ``kpoints``, the
    measured ones, and the others, which are left out.  A layer, whose
    ``H(k)`` has no ``k_z``, has no point at ``k_z = 1/2`` measured.  A
    little-group representation left out may occur in every elementary band
    representation as often as a measured one.  It is then said to repeat
    that one, and adds no condition to a symmetry data vector.

    Parameters
    ----------
    group : .groups.GroupData
        The group, ``r.group``.
    kpoints : sequence of str, optional
        The tabulated names of the points measured.  Default: every tabulated
        point, for a layer's group those in its plane.
    require_exact : bool, default True
        Whether the call refuses where a little-group representation left out
        repeats no measured one.

    Returns
    -------
    Restriction or Refusal
        The little-group representations measured and those left out, and for
        each one left out the measured ones it repeats.  The record also says
        whether every one left out repeats a measured one.  Where each does,
        the decomposition or class read from the measured ones is that of the
        whole symmetry data vector.  With ``require_exact=False``, the record
        lists those that repeat none in its ``orphans``.  A ``Refusal`` in its
        place is falsy and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``restriction_not_exact``
        a little-group representation left out repeats no measured one.  With
        ``require_exact=False``, the call does not refuse.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.

    Raises
    ------
    TypeError
        ``group`` is not a group: ``r`` or ``r.rep`` in its place, or
        anything else.
    ValueError
        A name in ``kpoints`` the group does not tabulate.
    """
    _spacegroup._require_group("in_plane_restriction", group)
    if isinstance(group, Refusal):
        return group
    if kpoints is not None:
        kpoints = tuple(str(p) for p in kpoints)
        _check_names("in_plane_restriction", group, kpoints)
    return _restriction(group, kpoints, require_exact=require_exact, layer=_layer)


def _restriction(group, points=None, *, require_exact=True, layer=None) -> Restriction | Refusal:
    """What :func:`in_plane_restriction` returns, the names already checked.

    ``group`` is the group and ``require_exact`` is as
    :func:`in_plane_restriction` takes them.  ``points`` are the names of the
    points the model samples, every tabulated point of the model's plane
    where ``None``.  ``layer`` is the record of :func:`_layer_reading`, or
    ``None``.  Where it is given, the little-group representations are
    compared on the band representations within the layer alone, and the
    positions off it fill ``outside_layer``.  Returns the
    :class:`Restriction`, or a refusal: ``lattice_not_descended``, a refusal
    of the tables or of the group's high-symmetry points, or
    ``restriction_not_exact``.  Raises ``InternalError`` where, on a layer's
    own band representations with every point of its plane compared, a
    little-group representation out of the plane repeats none in it.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    data = _tables._ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels = list(data["labels_physics_convention"])
    matrix = np.asarray(data["matrix"], dtype=int)
    # On a layer, the rows are compared on the columns of its own band
    # representations, those of the positions within it.
    if layer is not None:
        matrix = matrix[:, list(layer["columns"])]

    if points is None:
        hsp = _tables._high_symmetry_points(group)
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
    # On a layer's own band representations every row out of its plane repeats
    # one in it, since the layer's operations carry no translation along its
    # normal and map each position within the layer to one within it.  That
    # holds of the whole plane, so it is checked only where every point of the
    # plane was compared.
    if layer is not None and missing:
        hsp = _tables._high_symmetry_points(group)
        whole = not isinstance(hsp, Refusal) and set(hsp) <= set(points)
        off_plane = set(hsp.out_of_plane) if whole else set()
        unrepeated = [label for label in missing if kpoint_of_label(label) in off_plane]
        if unrepeated:
            raise InternalError(
                f"{group.name} #{group.number_str}: the little-group representations "
                f"{unrepeated} are out of the layer's plane, and none of them occurs in "
                "every one of the layer's own elementary band representations as often as "
                "one in the plane. Report it, with the model and the call.")
    if missing and require_exact:
        listed = sorted(missing, key=_tables.corep_order)
        return Refusal(
            "restriction_not_exact",
            f"the little-group representations {listed} are at points not measured, and "
            "none of them occurs in every elementary band representation of "
            f"{group.name} #{group.number_str} as often as a measured one.",
            value=float(len(missing)), threshold=0.0,
            context={"missing": tuple(listed), "retained": tuple(labels[j] for j in keep),
                     "points": points, "group": group, "number_str": group.number_str})

    return Restriction(
        kpoints=points,
        retained=tuple(labels[j] for j in keep), dropped=tuple(labels[i] for i in drop),
        copies=copies, orphans=tuple(missing), exact=not missing,
        outside_layer={} if layer is None else dict(layer["outside"]),
        _rank=int(np.linalg.matrix_rank(matrix[keep])) if keep else 0,
        _full_rank=int(np.linalg.matrix_rank(matrix)),
        _layer_extent=None if layer is None else tuple(layer["extent"]),
        _require_exact=bool(require_exact),
        _irreps=getattr(group, "magnetic_type", None) == "I")


def _layer_reading(group, data, rep=None, *, heights=None):
    r"""The band representations of the Wyckoff positions within a layer.

    ``group`` is the group, for a layer read in a cell periodic along its
    normal.  ``data`` is the group's elementary band representations, as
    :func:`._tables._ebr_data` gives them.  ``heights`` are the orbitals'
    heights from the layer's middle plane along its normal, in the model's
    units of length.  Without them the heights are those of the orbitals of
    ``rep``'s model, or of the group's own sites where ``rep`` is ``None`` or
    has no model.  A band representation is kept where its Wyckoff position
    is free to move along the normal, or where each of its heights lies
    between the lowest and the highest orbital, within the symmetry finder's
    distance.  Returns
    ``{"columns", "outside", "extent"}``: the indices of the band
    representations of ``data`` kept, ``{Wyckoff position: heights}`` for
    the positions left out, and the orbitals' ``(lowest, highest)`` heights.
    Returns ``None`` for a crystal periodic in three directions, and where
    the tables' Wyckoff letters cannot be read in the group's cell.
    """
    reading = _sewing._layer_height(group)
    if reading is None:
        return None
    if isinstance(_tables._sites_in_tables_setting(group), Refusal):
        return None              # the tables' letters name positions once the settings agree
    normal_dir, height = reading

    if heights is not None:
        own = [float(h) for h in heights]
    else:
        model = getattr(rep, "_model", None)
        if model is not None:
            where = _orb_positions_3d(model, group)
        else:
            where = np.asarray(group._site_positions, dtype=float)
        own = list(_sewing._orbital_heights(group, where))
    lowest, highest = min(own), max(own)
    margin = float(group._symprec)
    refuc = np.asarray(group._refUC, dtype=float)
    shift = np.asarray(group._shiftUC, dtype=float)
    arms_of = {p["letter"]: p["arms"] for p in _spacegroup._pyxtal_wyckoff(int(group._sg_number))[0]}
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

    ``a`` and ``b`` are :class:`.groups.GroupData`.  They are one group where
    they have the same number in the tables, both spinful or both not, both
    magnetic or both not, the same lattice, and the same operations, each
    rotation exactly and each translation modulo the lattice to rounding.
    Two records of one measurement are one group; a group written in another
    cell, or another group, is not.
    """
    if a is b:
        return True
    if ((str(a.number_str), bool(a.spinor), bool(a._magnetic))
            != (str(b.number_str), bool(b.spinor), bool(b._magnetic))):
        return False
    if len(a._symmetries) != len(b._symmetries):
        return False
    if not np.allclose(np.asarray(a._lattice, dtype=float), np.asarray(b._lattice, dtype=float)):
        return False

    def keyed(symmetries):
        out = set()
        for sym in symmetries:
            rot = tuple(int(x) for x in np.asarray(sym.rotation).reshape(-1))
            trans = np.asarray(sym.translation, dtype=float)
            trans = trans - np.floor(trans + ROUNDING)
            out.add((rot, tuple(int(x) for x in np.round(trans / ROUNDING))))
        return out

    return keyed(a._symmetries) == keyed(b._symmetries)


def _table_bands(table):
    """The band indices of the set a table was measured on.

    ``table`` is a :class:`.sewing.HSPTable`.  Returns its ``bands``.  An
    object with no ``bands`` gives its ``occ``, or ``None`` where it has
    neither.
    """
    block = getattr(table, "bands", _NO_BLOCK)
    return getattr(table, "occ", None) if block is _NO_BLOCK else block


def _as_block(bands, nstate):
    """``bands`` as sorted band indices.

    ``bands`` is an integer *n*, the lowest *n* bands, or a sequence of band
    indices.  ``nstate`` is the number of states, or ``None``.  With
    ``nstate``, the indices are those :func:`.sewing._band_block` returns,
    raising ``ValueError`` where it does; without it, the same rule is
    applied unchecked.  Returns the indices, ascending, or ``None`` where
    ``bands`` is ``None``.
    """
    if bands is None:
        return None
    if nstate is not None:
        return _sewing._band_block(int(nstate), bands)
    if np.isscalar(bands):
        return tuple(range(int(bands)))
    return tuple(sorted(int(b) for b in bands))


def _require_cap(call, cap) -> None:
    """Raise ``ValueError`` unless ``cap`` is a whole number of at least 1.

    ``cap`` is the largest number of solutions a call lists, and ``call``
    the name of that call, which opens the message.
    """
    if (isinstance(cap, bool) or not isinstance(cap, (int, np.integer))
            or int(cap) < 1):
        raise ValueError(f"{call}(): cap={cap!r}; a cap is a positive whole number, the "
                         "largest number of solutions listed.")


def _cpsat(matrix, y, lo, hi, cap):
    """Integer solutions of ``M x = y`` within bounds, up to ``cap``, by CP-SAT.

    ``matrix`` is ``M``, an integer array, and ``y`` an integer vector.  Each
    entry of ``x`` lies between ``lo`` and ``hi``, both included.  The search,
    by OR-Tools' CP-SAT solver, stops at the first solution past ``cap``.
    Returns ``(status, solutions, truncated)``: the solver's status by name,
    ``"OPTIMAL"``, ``"FEASIBLE"``, ``"INFEASIBLE"``, ``"MODEL_INVALID"`` or
    ``"UNKNOWN"``; the solutions found, at most ``cap``, the least sum of
    absolute values first, then in lexicographic order; and whether there
    are more than ``cap``.  ``"INFEASIBLE"`` says that no solution lies
    within the bounds.
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
            # one past the cap shows that the list stops short of every solution
            if len(found) >= cap:
                self.truncated = True
                self.StopSearch()
                return
            found.append(tuple(int(self.Value(v)) for v in x))

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
    """The matrix of elementary band representations and a vector, on the rows kept.

    ``data`` is the group's elementary band representations, as
    :func:`._tables._ebr_data` gives them.  ``counts`` is the symmetry data
    vector, ``{corep: multiplicity}``; a little-group representation it does
    not name is read as zero.  ``retained`` holds the little-group
    representations kept.  Returns ``(matrix, y, bound)``: the matrix's rows
    and the vector's entries on ``retained``, in its order, and the largest
    entry of ``y``, at least 1, which each multiplicity is searched up to.
    The bound is rigorous wherever every column has a positive entry among
    those rows (:attr:`EBRDecompositionNumerical.bound_is_rigorous`).
    """
    labels = list(data["labels_physics_convention"])
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.asarray(data["matrix"], dtype=int)[[index[label] for label in retained]]
    y = np.array([int(counts.get(label, 0)) for label in retained], dtype=int)
    return matrix, y, int(max(int(y.max()) if y.size else 0, 1))

def _lifted(solutions, columns, n) -> tuple:
    """Solutions over some band representations, as multiplicities over all of them.

    Each of ``solutions`` holds multiplicities, entry ``i`` that of the band
    representation ``columns[i]``.  ``n`` is the number of the group's band
    representations.  Returns each solution as ``n`` multiplicities, zero on
    the band representations not in ``columns``.
    """
    out = []
    for solution in solutions:
        vector = [0] * int(n)
        for j, multiplicity in zip(columns, solution):
            vector[int(j)] = int(multiplicity)
        out.append(tuple(vector))
    return tuple(out)

def _ordered(solutions) -> tuple:
    """Sums of elementary band representations: fewest first, then the tables' order.

    ``solutions`` holds sums, each a sequence of multiplicities, entry ``j``
    that of the ``j``-th elementary band representation the tables list.
    Returns them sorted, the fewest elementary band representations in all
    first, each multiplicity counted by its size.  Of two sums with as many,
    the one with more of the first elementary band representation where they
    differ comes first, so a sum at ``1a`` reads before the same sum at
    ``1b``.
    """
    return tuple(sorted(solutions, key=lambda s: (sum(abs(int(m)) for m in s),
                                                  tuple(-int(m) for m in s))))

def _terms(solutions, ebrs):
    out = []
    for s in solutions:
        out.append(tuple((int(m), str(ebrs[j][0]), str(ebrs[j][1]))
                         for j, m in enumerate(s) if m))
    return tuple(out)


# --------------------------------------------------------------------------
# the public solvers
# --------------------------------------------------------------------------

def decompose(symmetry_data, realized=None, *, allow_incomplete=False,
              cap=DEFAULT_ENUMERATION_CAP,
              signed_combinations=True) -> EBRDecomposition | Refusal:
    r"""The sums of elementary band representations that carry a symmetry data vector.

    The sums are found by enumerating the solutions :math:`x` in nonnegative
    integers of :math:`M x = y`.  Here :math:`y` is the symmetry data vector
    and the columns of :math:`M` are the elementary band representations, both
    taken on the little-group representations at the high-symmetry points
    the model samples.  Each multiplicity is searched up to the
    largest entry of :math:`y`, the bound.  The symmetry data vector of an
    atomic limit is such a sum.  Some band sets whose vector is such a sum are
    not atomic limits.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The symmetry data vector.  Pass the table :func:`.sewing.hsp_table`
        returns, or ``{corep: multiplicity}`` written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned for the model the
        symmetry data vector was measured on.  Pass it with a vector written
        out by hand, and nothing with a table.
    allow_incomplete : bool, default False
        Whether a table that refused at some points is decomposed.  With
        ``allow_incomplete=True``, the little-group representations at the
        points refused are left out.  Pass ``True`` only where the tables of
        the irreptables package cannot be read at those points.
    cap : int, default 200
        The largest number of sums listed.
    signed_combinations : bool, default True
        Whether a vector no sum carries is searched again with negative
        multiplicities allowed.

    Returns
    -------
    EBRDecomposition or Refusal
        Whether a sum carries the vector, and every sum that does, up to
        ``cap`` of them.  Where none does, the record also holds what the
        absence of a sum rests on, and the combinations that carry the vector
        with each multiplicity of either sign and at most the bound in size,
        up to ``cap`` of them.  The record is truthy even where no sum carries
        the vector.  With ``allow_incomplete=True``, the little-group
        representations at the points refused enter no equation, and the
        record's ``numerical.completeness`` says what the sums listed rest on.
        With ``signed_combinations=False``, no combination with negative
        multiplicities is returned.

    Examples
    --------
    The six lowest bands of lin3g (:func:`.models.lin3g`), with three sums:

    >>> d = decompose(hsp_table(r, bands=6))                      # doctest: +SKIP
    >>> [s.label for s in d.candidate_sums]                           # doctest: +SKIP
    ["-E2 at 1b (4mm1') + -E at 2c (mm21')", ...]

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``incomplete_content``
        the table refused at some point.  With ``allow_incomplete=True``, the
        call does not refuse.
    ``not_a_band_structure``
        the vector fails the compatibility relations of the group.
    ``unknown_label``
        the vector names a little-group representation that the group's
        elementary band representations do not carry.
    ``restriction_not_exact``
        a little-group representation the model does not measure is matched
        by no measured one that occurs as often in every elementary band
        representation.  With ``allow_incomplete=True`` and a table that
        refused at some point, the call does not refuse.
    ``non_integer_multiplicity``
        an entry of the vector is not an integer.
    ``negative_multiplicity``
        an entry of the vector is negative.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.

    Raises
    ------
    TypeError
        A table with ``realized`` beside it, a vector written out by hand
        without it, a ``realized`` that is not what
        :func:`.groups.realized_group` returned, or a ``symmetry_data`` that is
        neither a table nor a mapping.
    ValueError
        A ``cap`` that is not a positive whole number.

    See Also
    --------
    symmetry_indicator : the symmetry-indicator class of a band set.  A
        nonzero class rules out an atomic limit.
    """
    _require_cap("decompose", cap)
    source = _read("decompose", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    return _decompose_in(source, symmetry_data, allow_incomplete=allow_incomplete,
                         cap=cap, signed_combinations=signed_combinations)


def _decompose_in(source, symmetry_data, *, allow_incomplete=False,
                  cap=DEFAULT_ENUMERATION_CAP,
                  signed_combinations=True) -> EBRDecomposition | Refusal:
    """What :func:`decompose` returns, for a vector already read in its group.

    ``source`` is the :class:`_Source` :func:`_read` returns for
    ``symmetry_data``, the table or the vector ``{corep: multiplicity}``
    handed to the public call.  ``allow_incomplete``, ``cap`` and
    ``signed_combinations`` are as :func:`decompose` takes them.
    :func:`regular_completion` hands it the vector of the missing bands, with
    no table in ``source``.  Returns the :class:`EBRDecomposition`, a refusal
    :func:`decompose` names, or ``lattice_not_descended``.  Raises
    ``InternalError`` where the search with negative multiplicities allowed
    ends neither with solutions nor with a proof that there are none within
    the bound, and where the tables' Smith factors do not fit the matrix
    (:func:`._smith._smith_system`).
    """
    group = source.group
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    counts = content_of(symmetry_data)
    if isinstance(counts, Refusal):
        return counts

    table = source.table
    refused = ({name: r.gate for name, r in (table.refusals or {}).items()}
               if table is not None else {})
    if table is not None and not table.complete and not allow_incomplete:
        return Refusal(
            "incomplete_content",
            "the table holds no little-group representations at "
            f"{sorted(refused, key=_tables.point_order)}: {refused}. Read each missing "
            "point's refusal in the table's `refusals`.",
            value=float(len(refused)), threshold=0.0,
            context={"refusals": refused,
                     "resolved": tuple(table.multiplets)})

    data = _tables._ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels_all = list(data["labels_physics_convention"])
    ebrs = tuple((str(n), str(w)) for n, w in data["ebrs"])

    unknown = sorted(set(counts) - set(labels_all), key=_tables.corep_order)
    if unknown:
        return Refusal(
            "unknown_label",
            f"the labels {unknown} are not among those of the elementary band "
            f"representations of {group.name} #{group.number_str}. Pass a symmetry "
            "data vector read in this group.",
            value=float(len(unknown)), threshold=0.0,
            context={"unknown": tuple(unknown), "basis": tuple(labels_all),
                     "group": group})

    # Only the measured rows enter the equations.  A point that refused loses
    # its rows of the matrix too, since keeping them would force the
    # multiplicities there to zero.  A vector written out by hand is read at
    # the points of the plane and at each point out of it that it names.
    resolved = (tuple(table.multiplets) if table is not None
                else _named_points(group, counts))
    if isinstance(resolved, Refusal):
        return resolved
    exact_needed = table is None or table.complete
    # A free-standing layer's group has positions off the layer, which carry
    # none of its orbitals.  Their band representations are left out before
    # anything is solved, so that a capped enumeration lists the layer's own
    # sums.  On the rest every row at k_z = 1/2 repeats one in the plane.
    layer = _layer_reading(group, data, heights=source.heights)
    restriction = _restriction(group, resolved, require_exact=exact_needed, layer=layer)
    if isinstance(restriction, Refusal):
        return restriction
    columns = list(range(len(ebrs))) if layer is None else list(layer["columns"])

    # The compatibility relations are read before anything is solved.  A
    # vector no band structure has, such as that of a set of bands cut through
    # a crossing between two high-symmetry points, has no decomposition, and
    # the indicator's own check refuses it.  The check reads the vector
    # completed on the rows left out, which an exact restriction determines.
    if restriction.exact:
        y_all, _sources = _filled_content(counts, restriction, labels_all)
        _smith, _y_prime, wrong = _compatibility(data, group, y_all)
        if wrong is not None:
            return wrong

    matrix, y, bound = _content_system(data, counts, restriction.retained)
    matrix = matrix[:, columns]

    # A rigorous bound: M >= 0 and every column has a positive retained entry.
    col_max = matrix.max(axis=0) if matrix.size else np.zeros(len(columns), dtype=int)
    rigorous = bool((matrix >= 0).all() and (y >= 0).all() and (col_max > 0).all())

    status, solutions, truncated = _cpsat(matrix, y, 0, bound, cap)
    solutions = _ordered(_lifted(solutions, columns, len(ebrs)))
    decomposable = bool(solutions)

    signed = None
    if not decomposable and signed_combinations:
        p_status, p_sol, p_trunc = _cpsat(matrix, y, -bound, bound, cap)
        if p_status not in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"):  # pragma: no cover
            # No limit is set, so the search runs to its end: it finds solutions,
            # or proves there are none within the bound.
            raise InternalError(
                f"the search with negative multiplicities allowed ended {p_status}, "
                "with no limit set. Report it, with the model and the call.")
        p_sol = _lifted(p_sol, columns, len(ebrs))
        signed = SignedCombinations(
            sums=tuple(_sum_of(_terms_of(terms)) for terms in _terms(p_sol, ebrs)),
            truncated=bool(p_trunc))

    certificate = None
    if not decomposable:
        certificate = ("proof" if (status == "INFEASIBLE" and rigorous)
                       else "none within the bound")

    claim = "exact" if restriction.exact else "necessary-condition"
    note = "every tabulated point in the periodic plane resolved"
    if table is not None and not table.complete:
        note = (f"points {sorted(refused, key=_tables.point_order)} were dropped under "
                "allow_incomplete=True, and "
                f"their EBR rows {list(restriction.orphans)} with them")
        if restriction.exact:
            note += (" -- but every dropped row duplicates a retained one, so the answer "
                     "is still exact")
        else:
            note += (" -- those rows are independent constraints, so a sum found here "
                     "is a necessary condition only")
    elif table is None:
        note = ("the symmetry data vector was written out by hand: its completeness is "
                "asserted, not measured")

    kind = _kind(group)
    return EBRDecomposition(
        decomposable=decomposable,
        candidate_sums=tuple(_sum_of(_terms_of(terms)) for terms in _terms(solutions, ebrs)),
        symmetry_data=_tables.ordered_content(counts),
        signed_combinations=signed, certificate=certificate,
        site_content=_site_content(source.site_content),
        numerical=EBRDecompositionNumerical(
            bound=bound, bound_is_rigorous=rigorous,
            enumeration=Enumeration(cap=int(cap), n_solutions=len(solutions),
                                    truncated=bool(truncated)),
            completeness=Completeness(
                source="HSPTable" if table is not None else "mapping",
                complete=bool(getattr(table, "complete", True)), claim=claim,
                refused=dict(refused), note=note),
            restriction=restriction, _kind=kind),
        _status=status, _group=group)


def _site_content(record) -> SiteContent | None:
    """The site irreps a measurement carries, as a :class:`SiteContent`.

    ``record`` is the mapping a table or a representation carries, its keys
    ``"number_str"``, ``"irreps"``, ``"determined_by"`` and ``"residual"``
    the fields of the same names; or a :class:`SiteContent`, returned as it
    is; or ``None``, returned for a measurement that carries no site irreps.
    """
    if record is None or isinstance(record, SiteContent):
        return record
    return SiteContent(number_str=str(record["number_str"]),
                       irreps=tuple(record["irreps"]),
                       determined_by=tuple(record["determined_by"]),
                       residual=float(record["residual"]))


#: ``"I"`` (unitary), ``"II"`` (grey), ``"III"`` or ``"IV"``, measured from the
#: operations rather than parsed out of a BNS number.  One detector serves the
#: whole package: :func:`.._spacegroup._group_of` gates on it at construction, and
#: :func:`symmetry_indicator` re-checks it here.
_magnetic_type = _spacegroup._magnetic_type

def _named_forms(group, data, matrix, y, magnetic_type):
    """The irreptables package's named indicators of a grey group, and their coefficients.

    ``group`` is the group, ``magnetic_type`` its magnetic type, ``"I"`` to
    ``"IV"``, and ``data`` its elementary band representations, as
    :func:`._tables._ebr_data` gives them.  ``matrix`` is the matrix of
    elementary band representations and ``y`` the symmetry data vector, both
    on every row, in the order of ``data``'s little-group representations.
    Each named indicator is a linear form shipped with its coefficients on the
    Bilbao Crystallographic Server's names of the rows,
    ``data["convention"]["labels_bcs"]``.  Its coefficient on
    row ``i`` is the shipped one at ``P[i]``, ``P`` the convention's row
    permutation.  A form is read only where it vanishes, modulo its modulus,
    on every elementary band representation of the group.  Returns
    ``(values, touched, note, coefficients)``: each form's value on ``y``
    modulo its modulus, by the irreptables package's name (``z4``,
    ``z2w,3``); the names of the forms the row permutation moves; a sentence
    on how the forms were read; and each form's coefficients on the rows.
    For a group that is not grey, and where any form cannot be read or fails
    its check, returns ``None``, ``()``, the reason and ``{}``.  Only the
    coefficients are read in the package, by :func:`_two_dimensional_reading`.
    """
    if magnetic_type != "II":
        return None, (), (
            "irreptables' non-magnetic indicator files are the grey (type-II) tables "
            "keyed by the bare space-group number -- their factors are grey corep "
            "labels and most of them do not even vanish on this group's own EBRs -- so "
            "no named form is returned for a unitary group. The Smith-form indicator "
            "above is unaffected: it is computed from this group's own EBR matrix"), {}
    try:
        entry = _spacegroup._indicator_table(group.spinor, group._magnetic).get(
            str(group.number_str), {})
        forms = dict(entry.get("indicators", {}))
    except Exception as exc:                   # pragma: no cover, table defect
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
        moved = list(_tables._form_on_rows(data["convention"], spec["factors"], len(y)))
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
                              f"on this symmetry data vector ({total})"), {}
        values[str(name)] = int(round(total)) % mod
        coeffs[str(name)] = tuple(moved)
    return values, tuple(touched), ("irreptables' named formulas, re-keyed by the row "
                                    "permutation of the change of convention and "
                                    "checked to vanish on every EBR of this group"), coeffs

def _two_dimensional_reading(group, labels, u, slots, divisors, values, coeffs,
                             plane=None):
    """How a layer's symmetry-indicator class reads, as the class of its decoupled stack.

    ``group`` is the group the class was read in.  ``labels`` are the
    little-group representations of the rows of its matrix ``M`` of
    elementary band representations, and ``u`` the left factor of its Smith
    normal form ``u M v = D``.  ``slots`` are the coordinates at which ``D``
    holds an elementary divisor above 1, ``divisors`` those divisors, and
    ``values`` the class at them.  ``coeffs`` holds each named indicator's
    coefficients on the rows, by the irreptables package's name
    (:func:`_named_forms`).  ``plane`` is ``None`` where ``group`` is the
    layer's own.  Where the class was read in the group of the decoupled
    stack, ``plane`` describes the layer, ``{"points", "dim_k",
    "periodic_dirs", "vacuum_dirs", "presentation"}``, ``"points"`` the
    stack's points in the layer's plane.  A component reads out of the plane
    where its row of ``u`` is nonzero on a row out of the plane.  A named
    indicator's support is where its coefficients are nonzero.  Returns the
    :class:`TwoDimensionalReading`, its support keyed by the paper's names, a
    name the paper does not have left out.  Returns ``None`` for a trivial
    indicator group and for a group whose tables have no row out of the
    plane.  Where ``plane`` is ``None``, returns ``None`` also for a crystal
    periodic in three directions and for a group whose high-symmetry points
    cannot be read.
    """
    if not divisors:
        return None                            # no class to misread
    if plane is None:
        if int(getattr(group, "_dim_k", 3)) >= 3:
            return None                        # a fully periodic model
        hsp = _tables._high_symmetry_points(group)
        if isinstance(hsp, Refusal):
            return None
        in_plane = set(hsp)
        dim_k = int(group._dim_k)
        periodic = tuple(int(p) for p in group.periodic_dirs)
        padded = tuple(int(p) for p in group._vacuum_dirs)
        presentation = "SLAB"
    else:
        in_plane = set(plane["points"])
        dim_k, periodic, padded = plane["dim_k"], plane["periodic_dirs"], plane["vacuum_dirs"]
        presentation = plane["presentation"]
    out = tuple(i for i, lab in enumerate(labels) if kpoint_of_label(lab) not in in_plane)
    if not out:
        return None
    out_points = tuple(sorted({kpoint_of_label(labels[i]) for i in out},
                              key=_tables.point_order))

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
        "k_z = 1/2, k-points the layer's plane does not contain. The layer's H(k) has no "
        "k_z, so this is the class of the layer's decoupled stack, the layer repeated "
        "along its normal with no hopping between copies, whose rows at k_z = 1/2 repeat "
        "the in-plane ones: every component is an invariant of that stack, fixed by the "
        "layer's own symmetry data vector.")
    return TwoDimensionalReading(
        applied=True, dim_k=int(dim_k), periodic_dirs=tuple(periodic),
        vacuum_dirs=tuple(padded), presentation=presentation,
        out_of_plane_kpoints=out_points,
        out_of_plane_labels=tuple(labels[i] for i in out),
        components=tuple(IndicatorComponent(**c) for c in components),
        support={_srsitables.from_irreptables_name(name): where
                 for name, where in support.items()
                 if _srsitables.from_irreptables_name(name) is not None},
        note=note)

def _block_characters(record) -> dict:
    """The characters of a whole set of bands at one point, by operation.

    ``record`` is a :class:`.sewing.CorepsAt` that carries its characters.
    Returns ``{operation index: character}``, each the sum of the multiplets'
    characters at that operation, at the operations the first multiplet
    lists; empty where there is no multiplet.
    """
    ch = record._characters
    columns = tuple(ch.multiplets[0].characters) if ch.multiplets else ()
    rows = np.asarray([[m.characters[i] for i in columns] for m in ch.multiplets],
                      dtype=complex).reshape(len(ch.multiplets), len(columns))
    total = rows.sum(axis=0) if rows.size else np.zeros(len(columns), dtype=complex)
    return {int(c): complex(v) for c, v in zip(columns, total)}

def _same_k(k_a, k_b, *, atol=1e-6) -> bool:
    """Whether two k-points agree in the plane, up to a reciprocal lattice vector.

    ``k_a`` and ``k_b`` are reduced coordinates, a layer's two or a stack's
    three, of which the first two are compared.  They are the same where each
    difference is within ``atol`` of an integer.
    """
    d = np.asarray(k_a, dtype=float).reshape(-1)[:2] - np.asarray(
        k_b, dtype=float).reshape(-1)[:2]
    return bool(np.allclose(d, np.round(d), atol=atol))

#: The ``k_z`` at which the decoupled stack's Hamiltonian is held to the layer's.
_KZ_PROBE = (0.0, 0.25, 0.5)

def _stack_record(table, group, rep, restriction, *, environment, note) -> dict:
    """The mapping a layer's decoupled stack is recorded in, before the stack is built.

    ``table`` is the layer's :class:`.sewing.HSPTable`, ``group`` its group,
    ``rep`` its representation and ``restriction`` the :class:`Restriction`
    read on its points.  ``environment`` is the layer's environment or
    ``None``, and ``note`` the sentence the record carries.  Returns the
    mapping :func:`_completion_record` reads: its ``"outcome"``
    ``"stopped"``, its ``"input"`` the layer as handed in, its ``"stack"``
    and ``"gates"`` empty, and its ``"stack_period"`` and ``"stack_extent"``
    ``None`` until the stack is built.
    """
    return {
        "route": "decoupled stack", "outcome": "stopped",
        "stopped_at": None, "stop_reason": None, "stack_period": None,
        "stack_extent": None, "environment": environment,
        "input": {"number_str": str(group.number_str), "name": str(group.name),
                  "size": int(group._size), "dim_k": int(group._dim_k),
                  "periodic_dirs": tuple(int(p) for p in group.periodic_dirs),
                  "vacuum_dirs": tuple(int(p) for p in group._vacuum_dirs),
                  "points": tuple(sorted(table.kpoints, key=_tables.point_order)),
                  "orphans": tuple(restriction.orphans),
                  "content": _tables.ordered_content(table.symmetry_data),
                  "n_represented": len(rep._indices)},
        "stack": {}, "gates": {}, "note": note,
    }

def _stack_hamiltonian(rep, c):
    """A layer's decoupled stack, and how far its Hamiltonian is from the layer's.

    ``rep`` is the layer's representation, whose model is repeated along its
    normal with the vacuum ``c`` between the copies, in the model's units of
    length (:func:`.groups.stack_presentation`).  Each orbital sits at its
    height ``h`` along the normal on the stack's period ``C``, so
    ``H_stack(k, k_z) = Λ† H_layer(k) Λ`` with ``Λ = diag(e^{2πi k_z h / C})``.
    The two are compared at eight fixed in-plane k-points at each ``k_z`` of
    :data:`_KZ_PROBE`.  Returns ``(stack, identity)``, or ``(refusal, None)``
    where the stack is not built.  ``identity`` holds the largest difference,
    ``"residual"``, and the difference at ``k_z = 0``, ``"at_zero"``, in the
    units of H.  Its ``"threshold"``, the largest residual allowed, is zero
    for a layer whose orbitals lie in one plane and otherwise the rounding the
    phases carry, in the units of H.  Its ``"period"`` is ``C`` and its
    ``"extent"`` the layer's extent along its normal, in the model's units of
    length.
    """
    stack = _layers.stack_presentation(rep._model, c=c)
    if isinstance(stack, Refusal):
        return stack, None
    period = float(np.linalg.norm(np.asarray(stack.lat_vecs, dtype=float)[2]))
    z = np.asarray(stack.orb_vecs, dtype=float)[:, 2]
    extent = float(z.max() - z.min()) * period if z.size else 0.0
    rng = np.random.default_rng(20240829)
    k2 = rng.uniform(-0.5, 0.5, size=(8, 2))
    flat = np.asarray(_bloch.hamiltonian(rep._model, k2))
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
    """The two checks of the stack's Hamiltonian, as the stack's record holds them.

    ``identity`` is the comparison :func:`_stack_hamiltonian` returns, and
    ``rep`` the layer's representation.  Returns
    ``{"hamiltonian_identity", "no_hopping_across_the_stack"}``: the
    comparison, with the number of in-plane k-points and the values of
    ``k_z``; and the number of the layer's hoppings, with zero for the largest
    hopping between the copies, since :func:`.groups.stack_presentation`
    builds no stack with one.
    """
    return {"hamiltonian_identity": {
                "residual": float(identity["residual"]),
                "threshold": float(identity["threshold"]),
                "at_zero": float(identity["at_zero"]), "n_k": 8, "k_z": _KZ_PROBE,
                "note": ("max |H_stack(k, k_z) - L(k_z)^dagger H_layer(k) L(k_z)|, L the "
                         "phases e^{2 pi i k_z h / C} of the orbitals' heights h on the "
                         "stack's period C: zero at k_z = 0, and at every k_z for a flat "
                         "layer")},
            "no_hopping_across_the_stack": {
                "n_hoppings": int(rep._model.nhops), "worst": 0,
                "note": "measured by stack_presentation, which refuses on any nonzero"}}

def _identity_holds(identity) -> bool:
    """Whether the stack's Hamiltonian is the layer's, as compared by :func:`_stack_hamiltonian`.

    ``identity`` is its comparison.  The two agree where they are equal at
    ``k_z = 0`` and differ by at most the threshold at every ``k_z``.
    """
    return identity["at_zero"] == 0.0 and identity["residual"] <= identity["threshold"]

def _shared_characters(table, table3, matched) -> tuple:
    """The band characters of a layer and of its decoupled stack where the two meet.

    ``table`` is the layer's :class:`.sewing.HSPTable` and ``table3`` the
    stack's, of the same bands.  Each point of ``table`` is paired with each
    point of ``table3`` in the layer's plane with the same in-plane k
    (:func:`_same_k`).  Their characters are compared at each operation
    ``matched`` maps, ``{layer's operation index: stack's operation index}``.
    Returns ``(worst, n, points)``: the largest difference, the number of
    characters compared, and the pairs of points' names.
    """
    worst, n, shared = 0.0, 0, []
    for name2, rec2 in table._records.items():
        chi2 = _block_characters(rec2)
        for name3, rec3 in table3._records.items():
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

def _stack_fibers(table3, tol, scale=1.0) -> tuple:
    """The stack's bands at each point out of the layer's plane, against those in it.

    ``table3`` is the decoupled stack's :class:`.sewing.HSPTable`.  Each of
    its points out of the plane is compared with each point in the plane with
    the same in-plane k (:func:`_same_k`).  With no hopping between the
    copies the two hold the same states.  They agree where they split into
    multiplets alike, carry the same multiplicities once sorted, have
    energies within ``1e-9`` times ``scale``, the stack's ``max|H|``, and
    characters within ``tol`` at every operation both little groups hold.
    Returns ``(fibers, first)``: a mapping for each pair compared, which
    :func:`_completion_record` reads into a :class:`StackFiber`, and the
    first pair that does not agree, or ``None``.  The comparison stops at
    that pair.
    """
    fibers = []
    for name_out, rec_out in table3._records.items():
        k_out = np.asarray(rec_out.k, dtype=float).reshape(-1)
        if len(k_out) < 3 or abs(k_out[2] - round(k_out[2])) < 1e-6:
            continue                            # this one is in the plane
        for name_in, rec_in in table3._records.items():
            k_in = np.asarray(rec_in.k, dtype=float).reshape(-1)
            if (abs(k_in[2] - round(k_in[2])) > 1e-6) or not _same_k(k_out, k_in):
                continue
            chi_a, chi_b = _block_characters(rec_out), _block_characters(rec_in)
            common = sorted(set(chi_a) & set(chi_b))
            defect = max((abs(chi_a[i] - chi_b[i]) for i in common), default=0.0)
            e_a = np.asarray([m.energy for m in rec_out.multiplets], dtype=float).reshape(-1)
            e_b = np.asarray([m.energy for m in rec_in.multiplets], dtype=float).reshape(-1)
            energy = (float(np.abs(e_a - e_b).max()) if e_a.shape == e_b.shape
                      else float("inf"))
            same_partition = (tuple(rec_out.numerical.identification._partition)
                              == tuple(rec_in.numerical.identification._partition))
            same_multiset = (sorted(rec_out.coreps.values())
                             == sorted(rec_in.coreps.values()))
            fibers.append({"point": str(name_out), "in_plane_point": str(name_in),
                           "energy_defect": energy, "character_defect": float(defect),
                           "n_shared_operations": len(common),
                           "partition_equal": bool(same_partition),
                           "multiplicities_equal": bool(same_multiset)})
            if not (same_partition and same_multiset
                    and energy < 1e-9 * (float(scale) if scale > 0.0 else 1.0)
                    and defect < tol):
                return tuple(fibers), fibers[-1]
    return tuple(fibers), None

def _stack_reading(group, g3, indicator3, data3):
    """The :class:`TwoDimensionalReading` of a layer's class read on its decoupled stack.

    ``group`` is the layer's group and ``g3`` the same group carried onto the
    stack's cell.  ``indicator3`` is the :class:`SymmetryIndicator` read on
    the stack in ``g3``, and ``data3`` the elementary band representations of
    ``g3``, as :func:`._tables._ebr_data` gives them.  The points of ``g3``
    are split by whether their ``k_z`` is an integer, and the reading is that
    of :func:`_two_dimensional_reading`, with the layer's periodic
    directions.  Returns it, or the reading ``indicator3`` holds where no
    point of ``g3`` is found in the plane.
    """
    hsp3 = _tables._high_symmetry_points(g3)
    in_plane = () if isinstance(hsp3, Refusal) else tuple(
        n for n, k in hsp3.items()
        if abs(float(np.asarray(k, dtype=float).reshape(-1)[2])
               - round(float(np.asarray(k, dtype=float).reshape(-1)[2]))) < 1e-6)
    if not in_plane:
        return indicator3.numerical.two_dimensional
    u3, _v3, _d3, matrix3 = _smith_system(data3, g3.number_str)
    _n, _t, _note, coeffs3 = _named_forms(g3, data3, matrix3, list(indicator3._y),
                                          _magnetic_type(g3))
    return _two_dimensional_reading(
        g3, indicator3._labels, u3, indicator3._slots, indicator3.divisors,
        indicator3.values, coeffs3,
        plane={"points": in_plane, "dim_k": int(group._dim_k),
               "periodic_dirs": tuple(int(p) for p in group.periodic_dirs),
               "vacuum_dirs": tuple(int(p) for p in group._vacuum_dirs),
               "presentation": "decoupled-STACK"})

def _stack_in_group(table, group, rep, restriction, *, c, characters_tol):
    """A layer's class, completed on its decoupled stack in the layer's own group.

    ``table`` is the layer's :class:`.sewing.HSPTable`, ``group`` its group,
    ``rep`` its representation and ``restriction`` the :class:`Restriction`
    read on its points, whose ``orphans`` are not empty.  ``c`` is the vacuum
    between the copies, in the model's units of length, and
    ``characters_tol`` the largest difference allowed between two characters
    compared.  The stack's group is ``group`` carried onto the stack's cell
    (:func:`._layers._layer_on_stack`), and its matrices are those of
    ``rep``.  The bands of ``table`` are measured again on the stack.
    Returns ``(indicator, record)``.  ``indicator`` is the
    :class:`SymmetryIndicator` read on the stack.  Its group is ``group``, its
    symmetry data vector that of ``table`` in the plane, and it holds the
    stack's completion and two-dimensional reading.  ``record`` is the mapping of
    :func:`_stack_record`, its ``"outcome"`` ``"completed"``.  Where the
    stack is not built (a hopping along the normal), ``rep`` has no matrix
    for some operation, or the stack's table or class is refused,
    ``indicator`` is ``None`` and the outcome ``"stopped"``, with
    ``"stopped_at"`` and ``"stop_reason"`` saying where and why.

    Raises
    ------
    ..refusal.InternalError
        Where a check that ties the stack to the layer fails: the stack's
        Hamiltonian against the layer's, the layer's matrices on the stack's
        Hamiltonian, the band characters and little-group representations
        where the two presentations meet, and the stack's bands at each point at
        ``k_z = 1/2`` against those at its partner in the plane.  The message
        names the check and its number.
    """

    record = _stack_record(
        table, group, rep, restriction,
        environment=(None if getattr(rep, "_environment", None) is None
                     else str(rep._environment)),
        note=("the layer's plane has no coordinate for the k-points at k_z = 1/2, where "
              "some little-group representations of its group repeat none in the plane. "
              "The same layer repeated along its normal with no hopping between copies, "
              "each orbital at its own height, is a crystal whose bands do not depend on "
              "k_z, its H(k, k_z) the layer's H(k) up to the phases of those heights, and "
              "whose group is the layer's own, each operation carried by its Cartesian "
              "action and represented by the layer's own matrix, so those little-group "
              "representations are measured on it. Every identity that ties it to the "
              "layer is measured and recorded below."))

    def stop(where, reason, **extra):
        record["stopped_at"] = str(where)
        record["stop_reason"] = str(reason)
        record["gates"].setdefault(str(where), {}).update(extra)
        return None, record

    def defect(check, number, what):
        raise InternalError(
            f"{group.name} #{group.number_str}: the decoupled stack of this layer "
            f"fails the check '{check}': {what} "
            f"({number:.3g}). Report it, with the model "
            "and the call.")

    # ---- 1. the stack, and the identity that makes it the layer ------------
    stack, identity = _stack_hamiltonian(rep, c)
    if isinstance(stack, Refusal):
        # A layer the stack is not built for: one with a hopping along the normal.
        return stop("stack_presentation",
                    "the stack is not built for this layer; " + stack.reason,
                    gate=stack.gate, value=stack.value)
    record["gates"].update(_identity_record(rep, identity))
    record["stack_period"], record["stack_extent"] = identity["period"], identity["extent"]
    if not _identity_holds(identity):
        defect("hamiltonian_identity", identity["residual"],
               "its H(k, k_z) is not the layer's H(k) in the stack's gauge")

    # ---- 2. the layer's group and matrices, on the stack ----------------
    represented = {int(i) for i in rep._indices}
    unrepresented = tuple(sorted(int(s.ind) for s in group._symmetries
                                 if int(s.ind) not in represented))
    if unrepresented:
        noun = "operation" if len(unrepresented) == 1 else "operations"
        return stop("representation",
                    f"the representation the table carries has no matrix for {noun} "
                    f"{_text._index_ranges(unrepresented)} of {group.name} #{group.number_str}",
                    unrepresented=unrepresented)
    g3, asg3 = _layers._layer_on_stack(group, rep._assignment, stack)
    by_ind = {int(s.ind): s for s in g3._symmetries}
    rng = np.random.default_rng(20241004)
    battery = rng.uniform(-0.5, 0.5, size=(8, 3))
    battery[:len(_KZ_PROBE), 2] = _KZ_PROBE
    # The layer's representation, carried onto the stack.  No key is kept against
    # an edit of the stack's model, since nothing else holds it.
    rep3 = replace(rep, _model=stack, _group=g3, _assignment=asg3,
                   _operations=tuple(by_ind[int(i)] for i in rep._indices),
                   _battery=battery, _model_key=None)
    residuals = rep3.covariance(battery)
    scale = float(np.abs(np.asarray(_bloch.hamiltonian(stack, battery))).max())
    gate = max(covariance_gate(len(rep._indices), int(stack.nstate), scale),
               float((rep._search or {}).get("gate") or 0.0))
    worst_d = max(residuals.values(), default=0.0)
    record["gates"]["representation"] = {
        "covariance": float(worst_d), "threshold": float(gate),
        "n_operations": len(rep._indices), "n_k": int(battery.shape[0]),
        "note": ("the layer's own matrices on the stack's H(k, k_z): "
                 "max |D H(k) D^-1 - H(Rk)| over every operation")}
    if worst_d > gate:
        defect("representation", float(worst_d),
               "the layer's matrices do not carry the stack's H(k, k_z) onto itself")
    rep3 = replace(rep3, numerical=replace(rep3.numerical, residuals=residuals))
    typat = tuple(int(group._typat[int(s)]) for s in rep._assignment.site_of_orbital)
    record["stack"] = {"number_str": str(g3.number_str), "name": str(g3.name),
                       "size": int(g3._size), "dim_k": int(g3._dim_k),
                       "periodic_dirs": tuple(int(p) for p in g3.periodic_dirs),
                       "typat": typat}

    # ---- 3. the symmetry data vector on the stack, of the same bands ------
    within = f"the stack's table, read in {group.name} #{group.number_str},"
    table3 = _sewing._hsp_table_rep(rep3, bands=_table_bands(table))
    if isinstance(table3, Refusal):
        return stop("stack_content", f"{within} is refused; "
                    + table3.reason, gate=table3.gate)
    if not table3.complete:
        return stop("stack_content",
                    f"{within} holds no little-group representations at "
                    f"{sorted(table3.refusals, key=_tables.point_order)}",
                    refusals={k: r.gate for k, r in table3.refusals.items()})

    # ---- 4. the characters, where the two presentations meet --------------
    worst_chi, n_chi, shared_points = _shared_characters(
        table, table3, {int(i): int(i) for i in rep._indices})
    labels_equal = all(dict(table._records[a].coreps) == dict(table3._records[b].coreps)
                       for a, b in shared_points
                       if a in table.multiplets and b in table3.multiplets)
    record["gates"]["characters"] = {
        "note": ("the measured characters of the bands at the operations and k-points the "
                 "two presentations share, and the little-group representations there, the "
                 "group being one"),
        "worst": worst_chi, "threshold": float(characters_tol), "n_comparisons": n_chi,
        "shared_points": shared_points, "labels_equal": labels_equal}
    if not n_chi:
        defect("characters", 0.0, "it shares no k-point in the plane with the layer")
    if worst_chi > characters_tol:
        defect("characters", worst_chi,
               "its band characters differ from the layer's at the k-points they share")
    if not labels_equal:
        defect("characters", float(len(shared_points)),
               "its little-group representations differ from the layer's at the k-points "
               "they share")

    # ---- 5. the bands at k_z = 1/2 are the bands at k_z = 0 ---------------
    fibers, bad = _stack_fibers(table3, characters_tol, scale)
    record["gates"]["fibers"] = fibers
    if bad is not None:
        defect("fibers", max(bad["energy_defect"], bad["character_defect"]),
               f"its bands at {bad['point']} are not its bands at {bad['in_plane_point']}")

    # ---- 6. the class, in the layer's own group ----------------------------
    indicator3 = _indicator_in(
        _Source(group=g3, table=table3, rep=rep3, site_content=table3._site_content,
                free_standing=getattr(rep3, "_free_standing_group", None),
                provenance=table3._group_provenance), table3)
    if isinstance(indicator3, Refusal):
        return stop("stack_indicator", f"{within} has no symmetry-indicator class; "
                    + indicator3.reason, gate=indicator3.gate)
    data3 = _tables._ebr_data(g3)
    record["stack"]["divisors"] = _divisors_of(data3, g3.number_str)
    record["outcome"] = "completed"
    # The class is returned in the layer's own group, with the layer's symmetry
    # data vector in its plane; the vector at every point, which the class is
    # read from, is the one measured on the stack.
    numerical = replace(indicator3.numerical, completion=_completion_record(record),
                        two_dimensional=_stack_reading(group, g3, indicator3, data3),
                        _group=group)
    return replace(indicator3, group=group,
                   symmetry_data=_plane_vector(group, table.symmetry_data),
                   numerical=numerical), record


def symmetry_indicator(symmetry_data, realized=None) -> SymmetryIndicator | Refusal:
    r"""The symmetry-indicator class of a set of bands.

    The columns of the group's matrix :math:`M` of elementary band
    representations are their symmetry data vectors.  The irreptables package
    ships its Smith normal form, :math:`U M V = D`.  The class of the symmetry
    data vector :math:`y` of the bands is the entries of :math:`U y` at which
    the diagonal of :math:`D` holds an elementary divisor above 1, each taken
    modulo that divisor.  A layer's class is read in the three-dimensional
    space group of its slab, as the class of its decoupled stack, the layer
    repeated along its normal with no hopping between copies.  Each of its
    little-group representations at ``k_z = 1/2`` that occurs in every
    elementary band representation as often as one in the plane takes that
    one's multiplicity.  For a table, the others are measured on the stack.  A
    vector written out by hand is read as given at each point out of the plane
    where it names a little-group representation.  At the other points out of
    the plane, how often each little-group representation occurs is compared
    with those in the plane on the elementary band representations of the
    Wyckoff positions within the layer.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The symmetry data vector.  Pass the table :func:`.sewing.hsp_table`
        returns, or ``{corep: multiplicity}`` written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned for the model the
        symmetry data vector was measured on.  Pass it with a vector written
        out by hand, and nothing with a table.

    Returns
    -------
    SymmetryIndicator or Refusal
        The class, one integer modulo each elementary divisor, with its
        indicator group and its order.  For a group with time reversal and a
        nontrivial indicator group, the record also holds the class in the
        named indicators of Hwang *et al.*, Nat. Commun. **17**, 8564 (2026),
        where the tables of that paper are installed.
        A class can only be returned for a table that refused at no point.
        The record is truthy whatever its class.

    Examples
    --------
    The two lowest bands of Fu, Kane and Mele's diamond model, pythtb's
    ``fu_kane_mele``, with the bond along ``[111]`` strengthened,
    ``r = realized_group(fu_kane_mele(1.0, 0.125, [0.4, 0.0, 0.0, 0.0]),
    characters={0: "s"})``, and their class as ``values`` and in the named
    indicators:

    >>> si = symmetry_indicator(hsp_table(r, bands=2))          # doctest: +SKIP
    >>> si.indicator_group, si.values, si.indicators                # doctest: +SKIP
    ('Z2 x Z4', (0, 1), {'z2w,1': 1, 'z2w,2': 1, 'z2w,3': 1, 'z4': 3})

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``incomplete_content``
        the table refused at some point.
    ``not_a_band_structure``
        the vector fails the compatibility relations of the group.
    ``unknown_label``
        the vector names a little-group representation that the group's
        elementary band representations do not carry.
    ``indicator_not_evaluable``
        a layer's little-group representations at ``k_z = 1/2``, none of
        which occurs in every elementary band representation as often as one
        in its plane, could not be read.  For a table, the measurement on
        the decoupled stack stopped, and ``context["stack"]`` says where.  For a
        vector written out by hand, the Wyckoff positions of the tables cannot
        be read in the group's cell.
    ``non_integer_multiplicity``
        an entry of the vector is not an integer.
    ``negative_multiplicity``
        an entry of the vector is negative.
    ``magnetic_type_out_of_scope``
        a magnetic group of type III or IV, which
        :func:`.groups.realized_group` never returns.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the tables installed hold no elementary band representations of this
        group that can be read in the model's cell.

    Raises
    ------
    TypeError
        A table with ``realized`` beside it, a vector written out by hand
        without it, a ``realized`` that is not what
        :func:`.groups.realized_group` returned, or a ``symmetry_data`` that is
        neither a table nor a mapping.

    See Also
    --------
    decompose : the sums of elementary band representations that carry a
        symmetry data vector.  A vector with a nonzero class has none.
    """
    source = _read("symmetry_indicator", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    return _indicator_in(source, symmetry_data)


#: The message raised where the restriction refuses on a group whose tables
#: were already read.
_RESTRICTION_REFUSED = (
    "{} #{}: in_plane_restriction refused with {} on a group whose elementary band "
    "representations and high-symmetry points were read. Report it, with the model "
    "and the call.")


def _symmetry_indicator(symmetry_data, group, *, rep=None):
    """What :func:`symmetry_indicator` returns, for a vector with its group handed in.

    ``symmetry_data`` is a :class:`.sewing.HSPTable`, read in what it
    carries, ``group`` and ``rep`` not used; or a vector written out by
    hand, ``{corep: multiplicity}``, read in ``group`` with the site irreps
    and the heights of ``rep`` where it is given (:func:`_source_of`).
    Returns what :func:`_indicator_in` returns.
    """
    return _indicator_in(_source_of(symmetry_data, group, rep), symmetry_data)


def _indicator_in(source, symmetry_data) -> SymmetryIndicator | Refusal:
    """What :func:`symmetry_indicator` returns, for a vector already read in its group.

    ``source`` is the :class:`_Source` ``symmetry_data`` is read in, as
    :func:`_read` or :func:`_source_of` returns it, and ``symmetry_data`` is
    the table or the vector ``{corep: multiplicity}``.  For a layer's table
    whose little-group representations at ``k_z = 1/2`` include some that
    repeat none in the plane, the class is completed on the layer's decoupled
    stack (:func:`_stack_in_group`).  Returns the :class:`SymmetryIndicator`,
    a refusal :func:`symmetry_indicator` names, or ``lattice_not_descended``.
    Raises ``InternalError`` where the restriction refuses after the group's
    tables were read, where the tables' Smith factors do not fit the matrix
    (:func:`._smith._smith_system`), where the Smith normal form computed
    again disagrees with the tables' (:func:`._smith._verify_smith`), where
    the paper's tables leave out a group with a nontrivial indicator group or
    the checks of :func:`._srsitables._canonical_gate` on its forms cannot be
    made or fail (:func:`._srsitables._canonical_reading`), and where a check
    of the decoupled stack fails.
    """
    group = source.group
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    counts = content_of(symmetry_data)
    if isinstance(counts, Refusal):
        return counts

    mtype = _magnetic_type(group)
    out_of_scope = _spacegroup._magnetic_scope_refusal(group)
    if out_of_scope is not None:
        return out_of_scope

    table = source.table
    if table is not None and not table.complete:
        refused = {name: r.gate for name, r in (table.refusals or {}).items()}
        return Refusal(
            "incomplete_content",
            f"the table holds no little-group representations at "
            f"{sorted(refused, key=_tables.point_order)}: {refused}. Read each missing "
            "point's refusal in the table's `refusals`.",
            value=float(len(refused)), threshold=0.0,
            context={"refusals": refused, "resolved": tuple(table.multiplets),
                     "completion": "stable-rsi"})

    data = _tables._ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels = tuple(str(l) for l in data["labels_physics_convention"])

    unknown = sorted(set(counts) - set(labels), key=_tables.corep_order)
    if unknown:
        return Refusal(
            "unknown_label",
            f"the labels {unknown} are not among those of the elementary band "
            f"representations of {group.name} #{group.number_str}. Pass a symmetry "
            "data vector read in this group.",
            value=float(len(unknown)), threshold=0.0,
            context={"unknown": tuple(unknown), "basis": labels, "group": group})

    points = (tuple(table.multiplets) if table is not None
              else _named_points(group, counts))
    if isinstance(points, Refusal):
        return points
    restriction = _restriction(group, points, require_exact=False)
    if isinstance(restriction, Refusal):     # pragma: no cover, its gates were read above
        raise InternalError(_RESTRICTION_REFUSED.format(group.name, group.number_str,
                                                        restriction.gate))
    layered = int(getattr(group, "_dim_k", 3)) < 3

    # ---- the rows out of the layer's own plane, measured on the stack -----
    # Only for a table: it names the bands the stack measures again, and its
    # representation carries the model and the matrices the stack is built
    # from.  The stack obeys exactly what the layer obeys, so the class is
    # completed in the layer's own group, where some rows at k_z = 1/2 repeat
    # no in-plane row.
    completion = None
    if layered and table is not None and source.rep is not None and restriction.orphans:
        stacked, completion = _stack_in_group(
            table, group, source.rep, restriction, c=_layers.STACK_C,
            characters_tol=CHARACTERS_TOL)
        if stacked is not None:
            return stacked

    # ---- a vector written out by hand, read on the layer's own band
    # representations: there every row out of its plane repeats one in it, so a
    # little-group representation there the vector does not name takes its
    # partner's multiplicity
    if layered and table is None and restriction.orphans:
        layer = _layer_reading(group, data, heights=source.heights)
        if layer is not None:
            restriction = _restriction(group, points, require_exact=False, layer=layer)
            if isinstance(restriction, Refusal):     # pragma: no cover, as above
                raise InternalError(_RESTRICTION_REFUSED.format(
                    group.name, group.number_str, restriction.gate))

    if restriction.orphans:
        orphans = sorted(restriction.orphans, key=_tables.corep_order)
        reason = (
            (f"the little-group representations {orphans} of "
             f"{group.name} #{group.number_str} are out of the layer's plane, and none "
             "of them occurs in every elementary band representation as often as one "
             "in the plane. "
             "Their measurement on the layer's decoupled stack stopped: "
             f"{str(completion['stop_reason']).rstrip('. ')}.")
            if completion is not None else
            (f"the little-group representations {orphans} of "
             f"{group.name} #{group.number_str} are out of the layer's plane, and the "
             "vector gives none of them. None of them occurs in every elementary band "
             "representation as often as one in the plane, and the layer's own "
             "elementary band representations cannot be read in this group's cell. "
             "Pass the table hsp_table returns for these bands."))
        return Refusal(
            "indicator_not_evaluable", reason,
            value=float(len(restriction.orphans)), threshold=0.0,
            context={"orphans": tuple(restriction.orphans),
                     "points": tuple(restriction.kpoints),
                     "retained": tuple(restriction.retained),
                     "group": group, "number_str": group.number_str,
                     "completion": "stable-rsi",
                     "stack": completion})

    # Every row left out repeats a measured one, and one the vector does not
    # name takes that one's multiplicity.  The compatibility relations are read
    # on the vector so completed.
    y, reconstructed = _filled_content(counts, restriction, labels)
    (u, v, d, matrix), y_prime, wrong = _compatibility(data, group, y)
    if wrong is not None:
        return wrong

    slots = tuple(i for i, di in enumerate(d) if di > 1)
    divisors = tuple(int(d[i]) for i in slots)
    values = tuple(int(y_prime[i]) % int(d[i]) for i in slots)
    order = 1
    for val, div in zip(values, divisors):
        step = div // np.gcd(int(val), int(div)) if val else 1
        order = int(np.lcm(order, int(step)))

    verification = _verify_smith(group.number_str, matrix, u, v, d, y,
                                 slots, divisors, values, order)

    _named, _touched, _note, coeffs = _named_forms(group, data, matrix, y, mtype)
    two_d = _two_dimensional_reading(group, labels, u, slots, divisors, values, coeffs)
    # The paper's named indicators: one rational solve of BR.p = y, the
    # arithmetic srsi_of_bands does on the same y, and then the tabulated forms.
    indicators, indicators_note, _tex = _srsitables._canonical_reading(
        group, mtype, u, v, d, y, divisors)

    provenance = ((table._group_provenance if table is not None else None)
                  or source.provenance or str(getattr(group, "_provenance", "cell")))
    # The free-standing layer's group, as realized_group measured it beside the
    # group on a substrate, is named where it is larger than the class's group.
    free = source.free_standing
    if free and int(free["size"]) > int(group._size):
        from ._realized import NamedGroup
        free = NamedGroup(name=str(free["name"]), number_str=str(free["number_str"]),
                          size=int(free["size"]))
    else:
        free = None
    return SymmetryIndicator(
        values=values, divisors=divisors, indicator_group=_structure(divisors),
        indicators=indicators, order=int(order),
        symmetry_data=_plane_vector(group, counts),
        site_content=_site_content(source.site_content), group=group,
        numerical=SymmetryIndicatorNumerical(
            verification=SmithVerification(**{key: value for key, value in verification.items()
                                              if key != "performed"}),
            lattice_check=LatticeCheck(in_lattice=True, residual=0,
                                       relations=int(sum(1 for x in d if x == 0))),
            restriction=restriction,
            reconstructed=_tables.ordered_content(reconstructed),
            completed_symmetry_data=_tables.ordered_content(
                {label: int(m) for label, m in zip(labels, y) if m}),
            completion=None, two_dimensional=two_d, free_standing_group=free,
            convention=INDICATOR_CONVENTION, indicators_note=indicators_note,
            group_provenance=str(provenance), _group=group),
        _labels=labels, _y=tuple(int(x) for x in y),
        _y_prime=tuple(int(x) for x in y_prime), _slots=slots)


def _plane_vector(group, counts) -> dict:
    """A symmetry data vector at the points of a layer's plane.

    ``counts`` is the vector, ``{corep: multiplicity}``, read in ``group``.
    Returns it ordered as :func:`._tables.ordered_content` orders it, for a
    layer at the tabulated points of its plane alone.  For a crystal periodic
    in three directions, and where the group's high-symmetry points cannot be
    read, returns the whole vector.
    """
    if int(getattr(group, "_dim_k", 3)) >= 3:
        return _tables.ordered_content(counts)
    hsp = _tables._high_symmetry_points(group)
    if isinstance(hsp, Refusal):
        return _tables.ordered_content(counts)
    plane = set(hsp)
    return _tables.ordered_content({label: m for label, m in counts.items()
                                    if kpoint_of_label(label) in plane})


def _incomplete_table(content):
    """The ``incomplete_content`` refusal for a table that refused at some point.

    ``content`` is a :class:`.sewing.HSPTable` or a vector written out by
    hand.  Returns the refusal, naming each point refused and its refusal's
    name, for a table not complete.  Returns ``None`` for a complete table
    and for a vector written out by hand.
    """
    if not isinstance(content, _sewing.HSPTable) or content.complete:
        return None
    refused = {name: r.gate for name, r in (content.refusals or {}).items()}
    names = sorted(refused, key=_tables.point_order)
    return Refusal(
        "incomplete_content",
        f"the table holds no little-group representations at {names}: {refused}. "
        "Read each missing "
        "point's refusal in the table's `refusals`.",
        value=float(len(refused)), threshold=0.0,
        context={"refusals": refused, "resolved": tuple(content.multiplets)})


def subtract(symmetry_data_a, symmetry_data_b) -> dict | Refusal:
    """The symmetry data vector of the bands in one set and not in another, ``a - b``.

    Parameters
    ----------
    symmetry_data_a, symmetry_data_b : .sewing.HSPTable or mapping
        The two symmetry data vectors, ``a`` and ``b``.  Pass for each the
        table :func:`.sewing.hsp_table` returns, or ``{corep: multiplicity}``
        written out by hand.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``, the zeros dropped.  A difference can only be
        returned where ``a`` carries every little-group representation of
        ``b`` at least as many times as ``b`` does.  Where the two are equal it is empty, and so
        falsy.  ``isinstance(out, Refusal)`` tells a refusal apart.

    Refusals
    --------
    ``incomplete_content``
        either is a table that refused at some point.
    ``negative_multiplicity``
        ``b`` carries some little-group representation more times than ``a``
        does, or an entry of either is negative.
    ``non_integer_multiplicity``
        an entry of either is not an integer.

    Raises
    ------
    TypeError
        Either is neither a table nor a mapping.
    ValueError
        The two are tables read in different groups.

    See Also
    --------
    decompose : the sums of elementary band representations that carry a
        symmetry data vector, such as the difference.
    """
    for name, value in (("symmetry_data_a", symmetry_data_a),
                        ("symmetry_data_b", symmetry_data_b)):
        if not isinstance(value, (_sewing.HSPTable, Mapping)):
            raise TypeError(f"subtract(): {name}= is of type {type(value).__name__}; pass "
                            "the table hsp_table returns, or {corep: multiplicity} written "
                            "out by hand.")
    own_a = symmetry_data_a.group if isinstance(symmetry_data_a, _sewing.HSPTable) else None
    own_b = symmetry_data_b.group if isinstance(symmetry_data_b, _sewing.HSPTable) else None
    if own_a is not None and own_b is not None and not _same_group(own_a, own_b):
        raise ValueError(
            f"subtract(): the tables were read in different groups, {own_a.name} "
            f"#{own_a.number_str} and {own_b.name} #{own_b.number_str}. "
            "Pass two tables read in one group.")
    for content in (symmetry_data_a, symmetry_data_b):
        missing = _incomplete_table(content)
        if missing is not None:
            return missing
    a = content_of(symmetry_data_a)
    if isinstance(a, Refusal):
        return a
    b = content_of(symmetry_data_b)
    if isinstance(b, Refusal):
        return b

    out, bad = {}, {}
    for lab in sorted(set(a) | set(b), key=_tables.corep_order):
        v = a.get(lab, 0) - b.get(lab, 0)
        if v < 0:
            bad[lab] = v
        elif v:
            out[lab] = v
    if bad:
        return Refusal(
            "negative_multiplicity",
            f"the difference has negative multiplicities {bad}. Check that the bands "
            "of the second symmetry data vector are among those of the first.",
            value=float(min(bad.values())), threshold=0.0,
            context={"negative": bad, "a": a, "b": b})
    return out


def regular_content(group, *, kpoints=None) -> dict | Refusal:
    r"""The symmetry data vector of one copy of the regular band representation.

    One copy is the band representation induced from the general Wyckoff
    position.  Its site-symmetry group is trivial, or holds time reversal alone
    in a group with time reversal.  It carries one state, or the Kramers pair
    for a spinful model in a group with time reversal.  So one copy has
    ``n_1`` bands at every k: ``|U|``, or ``2|U|`` for the Kramers pair.
    ``|U|`` is the number of unitary operations up to lattice translations.
    For a spinful model they are counted in the single-valued group.  At
    ``k`` one copy is ``n_1/|U_k|`` copies of the regular representation of
    the little group ``U_k``, its double-valued part for a spinful model, so a
    little-group representation ``ρ`` occurs

    .. math:: m_\rho(k) = \frac{n_1\,\dim\rho}{\sum_{g \in U_k} |\chi_\rho(g)|^2}

    times: a corep that time reversal makes of two irreps, or of one irrep
    twice, occurs half or a quarter as often as an irrep of its dimension.

    Parameters
    ----------
    group : .groups.GroupData
        The group, ``r.group``.
    kpoints : sequence of str, optional
        The tabulated names of the points to read it at.  Default: every
        tabulated point in the model's plane.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``.  A ``Refusal`` in its place is falsy and
        names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``magnetic_type_out_of_scope``
        a magnetic group of type III or IV, which
        :func:`.groups.realized_group` never returns.
    ``out_of_plane``
        a point out of a layer's plane.
    ``tables_not_ready``
        the installed tables cannot be read for this group in this cell.

    Raises
    ------
    TypeError
        ``group`` is not a group: ``r`` or ``r.rep`` in its place, or
        anything else.
    ValueError
        A name in ``kpoints`` the group does not tabulate.
    """
    _spacegroup._require_group("regular_content", group)
    if isinstance(group, Refusal):
        return group
    return _regular_content(group, kpoints, call="regular_content")


def _regular_content(group, kpoints=None, *, call="regular_content") -> dict | Refusal:
    """What :func:`regular_content` returns, for a group already checked to be one.

    ``group`` and ``kpoints`` are as :func:`regular_content` takes them, and
    ``call`` is the public call's name, which an error starts with.  Returns
    ``{corep: multiplicity}`` of one copy of the regular band representation
    at the points asked, a refusal :func:`regular_content` names,
    ``lattice_not_descended``, or a refusal of a little group or of its
    table.  Raises ``ValueError`` (:func:`._tables._resolve_kpoint`) for a
    name in ``kpoints`` the group does not tabulate, and ``InternalError``
    where a multiplicity is not an integer, the multiplicities at a point do
    not give one copy's number of bands, or a point listed out of a layer's
    plane has a little group that is read.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    # The count of one copy below is written for magnetic types I and II.
    out_of_scope = _spacegroup._magnetic_scope_refusal(group)
    if out_of_scope is not None:
        return out_of_scope
    hsp = _tables._points_in_table_order(group)
    if isinstance(hsp, Refusal):
        return hsp
    wanted = tuple(hsp) if kpoints is None else tuple(str(p) for p in kpoints)
    for name in wanted:
        if name not in hsp and name not in hsp.out_of_plane:
            _tables._resolve_kpoint(call, group, name)

    n_copy = _bands_per_copy(group)
    out, bands = {}, {}
    for name in wanted:
        if name in hsp.out_of_plane:
            # A point out of a layer's plane, which its little group refuses
            refused = _tables._little_group(group, _tables._table_kpoints(group)[name])
            if not isinstance(refused, Refusal):
                raise InternalError(
                    f"{group.name} #{group.number_str}: {name} is listed as out of the "
                    "layer's plane, but its little group was read. Report it, with the "
                    "model and the call.")
            return refused
        k = hsp[name]
        lg = _tables._little_group(group, k)
        if isinstance(lg, Refusal):
            return lg
        table = _tables._corep_table(group, k)
        if isinstance(table, Refusal):
            return table
        ident = min(lg.unitary)
        total = 0.0
        for label, char in table.items():
            dim = abs(complex(char[ident]))
            norm = sum(abs(complex(char[i])) ** 2 for i in lg.unitary)
            m = n_copy * dim / norm
            if abs(m - round(m)) > _INT_TOL:
                raise InternalError(
                    f"{group.name} #{group.number_str}: the multiplicity of {label} at {name} "
                    f"in one copy of the regular band representation is {m:.6g}, which is not "
                    "an integer. Report it, with the model and the call.")
            out[label] = int(round(m))
            total += round(m) * dim
        bands[name] = total
    bands = {n: bands[n] for n in sorted(bands, key=_tables.point_order)}
    off = {n: b for n, b in bands.items() if abs(b - n_copy) > _INT_TOL}
    if off:
        raise InternalError(
            f"{group.name} #{group.number_str}: one copy of the regular band representation "
            f"has {n_copy} bands, and the numbers of bands its multiplicities give at these "
            f"points are {off}. Report it, with the model and the call.")
    return _tables.ordered_content(out)


def _bands_per_copy(group) -> int:
    """The number of bands of one copy of the regular band representation of ``group``.

    One state at the general position for each unitary operation up to
    lattice translations, and two, the Kramers pair, for a spinful grey
    group.  Written for magnetic types I and II, where an antiunitary
    operation means time reversal is in the group;
    :func:`_regular_content` refuses the other two types first.
    """
    n_u = len(group._u_symmetries)
    return 2 * n_u if bool(group.spinor) and len(group._au_symmetries) else n_u


def regular_completion(symmetry_data, realized=None, *,
                       bands=None) -> RegularCompletion | Refusal:
    """The bands completing a band set to copies of the regular band representation.

    One copy of the regular band representation is the band representation
    induced from the general Wyckoff position, where it carries one state, or
    the Kramers pair for a spinful model in a group with time reversal.  So it
    has ``n_1`` bands at every k: ``|U|``, or ``2|U|`` for the Kramers pair.
    ``|U|`` is the number of unitary operations up to lattice translations,
    counted in the single-valued group for a spinful model.  The symmetry data
    vector of the missing bands is ``n / n_1`` times that of one copy, less that of the band
    set, where ``n`` is the number of bands in ``bands``.  It is decomposed by
    :func:`decompose`.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The symmetry data vector of the band set.  Pass the table
        :func:`.sewing.hsp_table` returns, or ``{corep: multiplicity}``
        written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned for the model the coreps
        were measured on.  Pass it with a vector written out by hand, and
        nothing with a table.
    bands : int or sequence of int
        The bands of the completed system.  Pass their number, or their
        indices.  Only their number is read.  Required.

    Returns
    -------
    RegularCompletion or Refusal
        The symmetry data vector of the missing bands, and its decomposition
        into elementary band representations.  The record also holds the
        number of copies and the symmetry data vector of one copy.

    Examples
    --------
    The six lowest bands of lin3g, a spinful model of ``P4mm1'``, completed to
    one copy, sixteen bands.

    >>> rc = regular_completion(hsp_table(r, bands=6), bands=16)  # doctest: +SKIP
    >>> rc.bands_per_copy, format_coreps(rc.padding)              # doctest: +SKIP
    (16, '2x-GM6 + 3x-GM7 + 3x-M6 + 2x-M7 + 5x-X5')

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``incomplete_content``
        the table refused at some point.
    ``not_a_regular_multiple``
        the number of bands is not a multiple of ``n_1``.
    ``negative_multiplicity``
        the band set carries some little-group representation more times than
        the copies do, or an entry of the vector is negative.
    ``non_integer_multiplicity``
        an entry of the vector is not an integer.
    ``not_a_band_structure``
        the vector of the missing bands fails the compatibility relations of
        the group.
    ``unknown_label``
        the vector of one copy names a little-group representation that the
        group's elementary band representations do not carry.
    ``restriction_not_exact``
        a little-group representation the model does not measure is matched by
        no measured one that occurs as often in every elementary band
        representation.
    ``magnetic_type_out_of_scope``
        a magnetic group of type III or IV, which
        :func:`.groups.realized_group` never returns.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        the installed tables cannot be read for this group in this cell.

    Raises
    ------
    TypeError
        No ``bands``, a table with ``realized`` beside it, a vector written
        out by hand without it, a ``realized`` that is not what
        :func:`.groups.realized_group` returned, or a ``symmetry_data`` that is
        neither a table nor a mapping.
    """
    if bands is None:
        raise TypeError("regular_completion(): no bands= is passed; pass the bands of the "
                        "completed system, as their number or their indices.")
    source = _read("regular_completion", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    group = source.group
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    missing = _incomplete_table(symmetry_data)
    if missing is not None:
        return missing
    target_one = _regular_content(group, call="regular_completion")
    if isinstance(target_one, Refusal):
        return target_one

    n_copy = _bands_per_copy(group)
    n_bands = len(_as_block(bands, None))
    if n_bands % n_copy:
        return Refusal(
            "not_a_regular_multiple",
            f"{n_bands} bands is not a multiple of {n_copy}, the number of bands in one "
            f"copy of the regular band representation of {group.name} "
            f"#{group.number_str}. Pass a multiple of it to bands=, such as "
            f"{n_copy * (n_bands // n_copy + 1)}.",
            value=float(n_bands % n_copy), threshold=0.0,
            context={"n_bands": n_bands, "bands_per_copy": n_copy,
                     "number_str": group.number_str})

    copies = n_bands // n_copy
    target = {lab: copies * m for lab, m in target_one.items()}
    occ = content_of(symmetry_data)
    if isinstance(occ, Refusal):
        return occ

    # More of a little-group representation than the copies hold is refused, by name.
    excess = {lab: occ.get(lab, 0) - target.get(lab, 0)
              for lab in sorted(set(occ) | set(target), key=_tables.corep_order)
              if occ.get(lab, 0) > target.get(lab, 0)}
    if excess:
        absent = [lab for lab in excess if not target_one.get(lab)]
        steps = ([f"The copies hold none of {absent}: check those names."] if absent else [])
        if len(absent) < len(excess):
            steps.append(f"Pass a larger multiple of {n_copy} to bands=.")
        return Refusal(
            "negative_multiplicity",
            "the set of bands holds more of some little-group representations than "
            f"{'1 copy' if copies == 1 else f'{copies} copies'} of the regular band "
            f"representation of {group.name} #{group.number_str} "
            f"{'holds' if copies == 1 else 'hold'}, by {excess}. " + " ".join(steps),
            value=float(-max(excess.values())), threshold=0.0,
            context={"negative": {lab: -m for lab, m in excess.items()},
                     "a": content_of(target), "b": occ})

    padding = subtract(target, occ)
    if isinstance(padding, Refusal):
        return padding

    # The padding is a vector of the same measurement, read as one written out
    # by hand.
    got = _decompose_in(replace(source, table=None), padding)
    if isinstance(got, Refusal):
        return got
    return RegularCompletion(copies=copies, bands_per_copy=n_copy, padding=padding,
                             decomposition=got, symmetry_data=occ,
                             regular=dict(target_one), _group=group)
