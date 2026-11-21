r"""Whether a band set could be an atomic limit, and its symmetry-indicator class.

An elementary band representation (EBR) is the band set one site irrep induces
from its Wyckoff position, where that band set is no sum of others; an atomic
limit's symmetry data vector is a sum of EBRs with nonnegative integer
multiplicities.

* :func:`decompose`: every such sum that carries a band set's symmetry data
  vector, or none, with what that rests on;
* :func:`symmetry_indicator`: the class of the symmetry data vector among the
  band structures modulo the atomic limits, nonzero where no band set with
  this vector is an atomic limit;
* :func:`regular_completion`: the bands that would carry a band set up to
  whole copies of the regular band representation;
* :func:`subtract`: one band set's symmetry data vector less another's;
* :func:`ebr_rows`, :func:`in_plane_restriction`, :func:`content_of`,
  :func:`regular_content` and :func:`kpoint_of_label`: the EBRs as symmetry data
  vectors, the coreps a layer measures, a vector as multiplicities, one regular
  copy's vector, and the point a corep belongs to.

Each call takes the table :func:`.sewing.hsp_table` returns, alone, or a
symmetry data vector written out by hand with what
:func:`.groups.realized_group` returned beside it.  A sum is necessary for an
atomic limit and never sufficient: the coreps at isolated k-points are blind
to what happens between them.  The group's data is ``irrep``'s tables, read
through :mod:`._tables` and carried into the convention in which a lattice
translation acts as ``e^{-ik·t}``.
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
    """``"irreps"`` for a group with no time reversal, ``"coreps"`` otherwise: what a summary calls the labels."""
    return "irreps" if getattr(group, "magnetic_type", None) == "I" else "coreps"


def _label_lines(sums) -> list:
    """Each sum of elementary band representations on its own line, by its label, the first few."""
    lines = ["    " + s.label for s in sums[:_SHOWN]]
    if len(sums) > _SHOWN:
        lines.append(f"    and {len(sums) - _SHOWN} more")
    return lines


@dataclass(frozen=True)
class SiteContent:
    """The site irreps a symmetry data vector's labels rest on, each orbital's, and how each is known.

    Attributes
    ----------
    number_str : str
        The group the site irreps are named in.
    irreps : tuple of str
        Each orbital's site irrep, by orbital.
    determined_by : tuple of str
        How each orbital's site irrep is known: ``"hamiltonian"``, fixed by
        the Hamiltonian; ``"declaration"``, declared; ``"lowest-l"``, chosen
        by the lowest-angular-momentum rule among those the hoppings allow.
    residual : float
        The largest residual of the site irreps' reading, in the units of H.
    """

    number_str: str
    irreps: tuple
    determined_by: tuple
    residual: float


@dataclass(frozen=True)
class Restriction:
    """Which coreps of the group's elementary band representations a model measures, and whether each of the others repeats a measured one.

    A corep at a point the model does not sample, as a layer's at
    ``k_z = 1/2``, that occurs in every elementary band representation as
    often as a measured one adds no condition; where every one does, what is
    read on the measured coreps is exact.  ``summary()`` prints it.

    Attributes
    ----------
    kpoints : tuple of str
        The k-points read.
    retained, dropped : tuple of str
        The coreps kept, and those left out.
    copies : dict
        ``{corep left out: (kept coreps it repeats, ...)}``: each elementary
        band representation holds it as often as it holds those.
    orphans : tuple of str
        The coreps left out that repeat none: conditions given up.
    exact : bool
        Whether every corep left out repeats a kept one, so that what is read
        on the kept coreps is exact.
    outside_layer : dict
        ``{Wyckoff position: heights}``: the positions of a free-standing
        layer's group off the layer, which hold none of its orbitals; their
        band representations are left out.
    """

    kpoints: tuple
    retained: tuple
    dropped: tuple
    copies: dict
    orphans: tuple
    exact: bool
    outside_layer: dict = field(default_factory=dict)
    # the rank of the matrix of elementary band representations on the kept
    # coreps and on all of them; for a layer, the heights its orbitals span from
    # its middle plane; whether an orphan was refused; whether the group has no
    # time reversal, for the word a summary prints
    _rank: int = field(default=0, repr=False)
    _full_rank: int = field(default=0, repr=False)
    _layer_extent: tuple | None = field(default=None, repr=False)
    _require_exact: bool = field(default=True, repr=False)
    _irreps: bool = field(default=False, repr=False)

    def summary(self) -> str:
        """The coreps left out, a line for each part, with the attribute that holds it."""
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        kind = "irreps" if self._irreps else "coreps"
        return (f"Restriction({len(self.retained)} {kind} kept, {len(self.dropped)} left out, "
                + ("exact)" if self.exact else "not exact)"))


def _where_line(restriction, completeness, kind) -> str | None:
    """What a decomposition was read without, the points refused or out of the plane, in one line; ``None`` where nothing was left out."""
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
    """A free-standing layer's positions off it, whose band representations were left out, in one line; ``None`` where there are none."""
    outside = sorted(restriction.outside_layer)
    if not outside:
        return None
    lowest, highest = restriction._layer_extent or (0.0, 0.0)
    off = "off the layer" if highest - lowest <= ROUNDING else "outside the slab"
    return (f"{_text._listed(outside)} lie {off}, so their band representations are left "
            "out")


@dataclass(frozen=True)
class Completeness:
    """Whether a decomposition is exact or a necessary condition, and why.

    Attributes
    ----------
    source : {"HSPTable", "mapping"}
        Whether the symmetry data vector was measured in a table or written
        out by hand.
    complete : bool
        Whether every point was measured; ``True`` for a vector written out
        by hand, whose completeness is taken as stated.
    claim : {"exact", "necessary-condition"}
        ``"necessary-condition"`` where coreps left out repeat no kept one.
    refused : dict
        ``{kpoint: gate}`` for each point a table refused and the decomposition
        left out.
    note : str
        The same in a sentence.
    """

    source: str
    complete: bool
    claim: str
    refused: dict
    note: str


@dataclass(frozen=True)
class Enumeration:
    """How far the list of sums goes.

    Attributes
    ----------
    cap : int
        The most sums listed.
    n_solutions : int
        How many were listed.
    truncated : bool
        Whether the list stops at the cap, short of every sum.
    """

    cap: int
    n_solutions: int
    truncated: bool


@dataclass(frozen=True)
class SignedCombinations:
    """The sums of elementary band representations with multiplicities of either sign that carry a symmetry data vector no nonnegative sum carries.

    One found is the signature of a fragile band set, none of a stable
    obstruction; signed terms can cancel without limit, so either is evidence.

    Attributes
    ----------
    sums : tuple of ..symmetry.BandRepresentation
        Each combination, with multiplicities between ``-bound`` and
        ``bound``, ``bound`` being :attr:`EBRDecompositionNumerical.bound`;
        empty where none was found.
    truncated : bool
        Whether the list stops at the cap, short of every combination.
    """

    sums: tuple
    truncated: bool


@dataclass(frozen=True)
class EBRDecompositionNumerical:
    """How a decomposition was searched and what it was read on: what :attr:`EBRDecomposition.numerical` holds.

    ``summary()`` prints it.

    Attributes
    ----------
    bound : int
        The largest multiplicity searched.
    bound_is_rigorous : bool
        Whether the bound leaves out no sum: so wherever every elementary band
        representation holds a kept corep, since then no multiplicity exceeds
        the vector's largest entry.
    enumeration : Enumeration
        The cap on the list of sums, and whether the list was cut.
    completeness : Completeness
        Whether the sums are exact or a necessary condition, and why.
    restriction : Restriction
        The coreps measured, and on a free-standing layer the Wyckoff positions
        off it, whose band representations every sum leaves out.
    """

    bound: int
    bound_is_rigorous: bool
    enumeration: Enumeration
    completeness: Completeness
    restriction: Restriction
    _kind: str = field(default="coreps", repr=False, compare=False)

    def summary(self) -> str:
        """What the decomposition was read on, a line for each part, with the attribute that holds it."""
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"EBRDecompositionNumerical(bound {self.bound}"
                + (", rigorous" if self.bound_is_rigorous else "")
                + f", {self.completeness.claim})")


@dataclass(frozen=True)
class EBRDecomposition:
    """Every sum of elementary band representations, with nonnegative integer multiplicities, that carries a band set's symmetry data vector, or none.

    ``summary()`` prints whether one does and each sum.  The record is truthy,
    one with no sum included.

    Attributes
    ----------
    decomposable : bool
        Whether a sum of elementary band representations carries the vector.
    candidate_sums : tuple of ..symmetry.BandRepresentation
        Every sum, each a sum of site irreps at Wyckoff positions: the fewest
        elementary band representations first, then the one with more of
        those the tables list first; empty where none carries the vector.
    symmetry_data : dict
        The symmetry data vector decomposed, ``{corep: multiplicity}``.
    signed_combinations : SignedCombinations or None
        Where no sum carries the vector, the same search with multiplicities
        of either sign; ``None`` where a sum does, and with
        ``signed_combinations=False``.
    certificate : str or None
        What a vector no sum carries rests on: ``"proof"``, no multiplicity
        being able to exceed the bound and no sum within it carrying the
        vector, or ``"none within the bound"``.  ``None`` where a sum carries
        it.
    site_content : SiteContent or None
        The site irreps the sums' names rest on.  Where one is ``"lowest-l"``,
        another declared site irrep renames them in :attr:`candidate_sums`
        while the Wyckoff positions stay.
    numerical : EBRDecompositionNumerical
        The bound searched, the cap on the list, and what the vector was read
        on.
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
        """Whether a sum carries the vector, a line for each part, with the attribute that holds it."""
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        n = len(self.candidate_sums)
        return (f"EBRDecomposition({'decomposable' if self.decomposable else 'not decomposable'}"
                f", {n} candidate sum{'' if n == 1 else 's'}, "
                + ("exact)" if self.numerical.completeness.claim == "exact"
                   else "necessary condition only)"))


def _signed_sums_line(signed, bound) -> str:
    """What the search with negative multiplicities allowed found, in one line.

    Every combination it lists has a negative multiplicity, since the
    nonnegative search within the same bound found none; a list cut at its cap
    counts at least that many.  Signed multiplicities have no rigorous bound.
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
    padding : dict
        The coreps the missing bands carry: :attr:`copies` times
        :attr:`regular`, less :attr:`symmetry_data`.
    decomposition : EBRDecomposition
        The padding's own sums: the orbitals the missing bands would be.
    symmetry_data : dict
        The symmetry data vector handed in.
    regular : dict
        One copy's symmetry data vector, from the group's corep characters.
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
        lines = [f"{copies} of the regular band representation, the one the regular "
                 f"representation of a site group induces: {self.bands_per_copy} bands at every "
                 f"k{'' if self.copies == 1 else ' each'}   (.copies, .bands_per_copy)"]
        if not self.padding:
            lines.append("  nothing is missing: the bands carry that content already   (.padding)")
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
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
    "read on relabelled coordinates. The indicator group is therefore convention-"
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
    """The Smith normal form of the group's matrix of elementary band representations, computed again, exactly.

    A disagreement with the tables' form raises, so a record returned holds
    one that agrees.

    Attributes
    ----------
    divisors_denovo : tuple of int
        The elementary divisors above 1 computed again.
    agrees, shipped_is_smith, denovo_is_smith : bool
        Whether the two forms agree, and whether each is a Smith form of the
        matrix.
    values_denovo : tuple of int
        The class in the coordinates of the form computed again.
    order_denovo : int
        The class's order computed again.
    change_of_basis : tuple
        The automorphism of the indicator group that carries the tables'
        coordinates onto those of the form computed again.
    change_of_basis_is_identity : bool
        Whether that automorphism is the identity.
    note : str
        The same in a sentence.
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

    Attributes
    ----------
    in_lattice : bool
        Whether the vector satisfies every compatibility relation; a record
        returned holds one that does.
    residual : int
        The largest violation.
    relations : int
        How many compatibility relations the group has.
    """

    in_lattice: bool
    residual: int
    relations: int


@dataclass(frozen=True)
class IndicatorComponent:
    """One component of a layer's class, and whether it reads coreps out of the layer's plane.

    Attributes
    ----------
    slot : int
        The Smith coordinate it is.
    divisor : int
        Its modulus.
    value : int
        Its value.
    reads_out_of_plane : bool
        Whether its linear form reads a corep at a point out of the plane.
    """

    slot: int
    divisor: int
    value: int
    reads_out_of_plane: bool


@dataclass(frozen=True)
class TwoDimensionalReading:
    """How a layer's class reads: the class of its decoupled stack, every component an invariant of that stack.

    Attributes
    ----------
    applied : bool
        Whether the reading applies; a record returned holds one that does.
    dim_k : int
        The layer's periodic dimensions.
    periodic_dirs, vacuum_dirs : tuple of int
        Its periodic directions, and its normal.
    presentation : {"SLAB", "decoupled-STACK"}
        Whether the class was read in the slab's group, or in the group of the
        decoupled stack the coreps out of the plane were measured on.
    out_of_plane_kpoints : tuple of str
        The tabulated points out of the layer's plane.
    out_of_plane_labels : tuple of str
        The coreps there.
    components : tuple of IndicatorComponent
        Each component of the class, and whether it reads a corep out of the
        plane.
    support : dict
        ``{indicator: where}``, each of the paper's named indicators and where
        its coefficients sit: ``"in-plane only"``, ``"out-of-plane only"`` or
        ``"mixed"``.
    note : str
        The same in a sentence.
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
    """The decoupled stack's Hamiltonian held to the layer's, in the gauge of the orbitals' heights.

    Attributes
    ----------
    residual : float
        The largest difference over the in-plane k-points and the ``k_z``
        probed, in the units of H.
    threshold : float
        What it is held to: zero for a flat layer, otherwise the rounding the
        phases carry.
    at_zero : float
        The difference at ``k_z = 0``, held to zero.
    n_k : int
        The in-plane k-points probed.
    k_z : tuple of float
        The ``k_z`` probed.
    note : str
        The same in a sentence.
    """

    residual: float
    threshold: float
    at_zero: float
    n_k: int
    k_z: tuple
    note: str


@dataclass(frozen=True)
class StackHoppings:
    """The hoppings between the stack's copies: none.

    Attributes
    ----------
    n_hoppings : int
        The layer's hoppings.
    worst : float
        The largest hopping between copies.
    note : str
        The same in a sentence.
    """

    n_hoppings: int
    worst: float
    note: str


@dataclass(frozen=True)
class RepresentationCheck:
    """The layer's own matrices held against the stack's Hamiltonian.

    Attributes
    ----------
    covariance : float
        The largest covariance residual over every operation, in the units of H.
    threshold : float
        What it is held to.
    n_operations : int
        The operations checked.
    n_k : int
        The k-points probed.
    note : str
        The same in a sentence.
    """

    covariance: float
    threshold: float
    n_operations: int
    n_k: int
    note: str


@dataclass(frozen=True)
class CharactersCheck:
    """The two presentations' band characters and labels where their k-points meet.

    Attributes
    ----------
    worst : float
        The largest difference between two characters compared.
    threshold : float
        What it is held to.
    n_comparisons : int
        How many characters were compared.
    shared_kpoints : tuple
        ``(layer's k-point, stack's k-point)`` for each pair compared.
    labels_equal : bool
        Whether the labels agree at each pair.
    note : str
        The same in a sentence.
    """

    worst: float
    threshold: float
    n_comparisons: int
    shared_kpoints: tuple
    labels_equal: bool
    note: str


@dataclass(frozen=True)
class StackFiber:
    """A fiber of the stack at ``k_z = 1/2`` held against the fiber in the plane at the same in-plane k.

    Attributes
    ----------
    kpoint : str
        The point at ``k_z = 1/2``.
    in_plane_kpoint : str
        The point in the plane it is held against.
    energy_defect : float
        The largest difference between their energies, in the units of H.
    character_defect : float
        The largest difference between their characters.
    n_shared_operations : int
        The operations both little groups hold.
    partition_equal : bool
        Whether the two fibers split into multiplets alike.
    multiplicities_equal : bool
        Whether they carry the same corep multiplicities.
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
    """Each check that ties the decoupled stack to the layer, with its measured number.

    Attributes
    ----------
    hamiltonian_identity : HamiltonianIdentity
        The stack's Hamiltonian against the layer's.
    no_hopping_across_the_stack : StackHoppings
        The hoppings between the copies.
    representation : RepresentationCheck
        The layer's matrices on the stack's Hamiltonian.
    characters : CharactersCheck
        The two presentations' characters where they meet.
    fibers : tuple of StackFiber
        Each fiber at ``k_z = 1/2`` against its partner in the plane.
    """

    hamiltonian_identity: HamiltonianIdentity
    no_hopping_across_the_stack: StackHoppings
    representation: RepresentationCheck
    characters: CharactersCheck
    fibers: tuple


@dataclass(frozen=True)
class StackInput:
    """The layer the decoupled stack was built for, as handed in.

    Attributes
    ----------
    group : ..groups.NamedGroup
        The layer's group.
    dim_k : int
        Its periodic dimensions.
    periodic_dirs, vacuum_dirs : tuple of int
        Its periodic directions, and its normal.
    kpoints : tuple of str
        The points its table measured.
    orphans : tuple of str
        The coreps out of its plane that repeat none in it.
    symmetry_data : dict
        Its symmetry data vector.
    n_represented : int
        The operations its representation holds a matrix for.
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

    Attributes
    ----------
    group : ..groups.NamedGroup
        The group.
    dim_k : int
        Its periodic dimensions, three.
    periodic_dirs : tuple of int
        Its periodic directions.
    typat : tuple of int
        Each orbital's species.
    divisors : tuple of int
        The elementary divisors above 1 of its matrix of elementary band
        representations.
    """

    group: Any
    dim_k: int
    periodic_dirs: tuple
    typat: tuple
    divisors: tuple


@dataclass(frozen=True)
class StackCompletion:
    """How a layer's coreps out of its plane were measured, on its decoupled stack in its own group.

    The decoupled stack is the layer repeated along its normal with no hopping
    between copies.

    Attributes
    ----------
    route : str
        ``"decoupled stack"``.
    outcome : str
        ``"completed"``.
    stopped_at, stop_reason : None
        Where the measurement stopped, and why; ``None`` for one that
        completed.
    stack_period : float
        The stack's period, in the model's units of length.
    stack_extent : float
        The layer's extent along its normal, in the same units.
    environment : str or None
        The layer's environment, ``"free-standing"`` or ``"substrate"``.
    note : str
        The same in a sentence.
    input : StackInput
        The layer as handed in.
    stack : StackGroup
        The stack's group.
    gates : StackChecks
        Each check that ties the stack to the layer.
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
    """A completed stack's mapping as a :class:`StackCompletion`."""
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
    """How an indicator class was read: what :attr:`SymmetryIndicator.numerical` holds.

    ``summary()`` prints it.

    Attributes
    ----------
    verification : SmithVerification
        The Smith normal form computed again, exactly.
    lattice_check : LatticeCheck
        The compatibility relations.
    restriction : Restriction
        The coreps measured.
    reconstructed : dict
        ``{corep: corep}``: each corep out of the layer's plane the symmetry
        data vector does not name, and the corep in the plane whose
        multiplicity it takes, which it repeats in every elementary band
        representation it was read on.  Empty for a crystal and for a table
        whose coreps there were measured.
    completed_symmetry_data : dict
        The symmetry data vector on every point of the group's tables, the
        coreps out of a layer's plane completed or measured on its decoupled
        stack, the zeros dropped.
    completion : StackCompletion or None
        How a layer's coreps out of its plane were measured on its decoupled
        stack; ``None`` where none had to be.
    two_dimensional : TwoDimensionalReading or None
        For a layer with an indicator, how its class reads; ``None`` for a
        crystal and for a group with no indicator.
    free_standing_group : ..groups.NamedGroup or None
        For a layer on a substrate, the free-standing layer's group, as
        :func:`..groups.realized_group` measured it, where it is larger than
        the group the class lives in; its class is asked with
        ``environment="free-standing"``.
    convention : str
        :data:`INDICATOR_CONVENTION`: the class does not depend on the
        translation convention the tables are written in.
    indicators_note : str
        Where :attr:`SymmetryIndicator.indicators` came from, or why it is
        ``None``.
    group_provenance : str
        ``"realized"`` or ``"descended"``: how the group was measured on H.
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SymmetryIndicatorNumerical({self.group_provenance}"
                + (", completed on the stack" if self.completion is not None else "") + ")")


@dataclass(frozen=True)
class SymmetryIndicator:
    r"""The symmetry-indicator class of a band set: the class of its symmetry data vector among the group's band structures modulo its atomic limits, ``X_BS = {BS}/{AI}``.

    A nonzero class says no band set with this symmetry data vector is an
    atomic limit, nor any with atomic bands added; a zero class certifies
    nothing, fragile band sets lying in it.  ``summary()`` prints it.  The
    record is truthy whatever its class.

    Attributes
    ----------
    values : tuple of int
        The class, one integer modulo each of :attr:`divisors`, in the
        coordinates the Smith normal form of the group's matrix of elementary
        band representations gives.
    divisors : tuple of int
        The elementary divisors above 1 of that matrix: ``X_BS`` is
        ``⊕ᵢ ℤ_{dᵢ}``; empty where the indicator group is trivial.
    indicator_group : str
        The same as text, ``"Z2 x Z4"``; ``"trivial"`` where there is none.
    indicators : dict or None
        The class in the named indicators of the paper that tabulates the
        stable real-space invariants, Hwang *et al.*, Nat. Commun. **17**, 8564
        (2026): ``{name: value}``, computed through those invariants.  ``None``
        for a group with no time reversal, which the paper does not tabulate,
        for a trivial indicator group, and where the converted tables are not
        installed.
    order : int
        The class's order, the fewest copies of the band set whose class is
        zero: ``1`` for the zero class.  Unlike :attr:`values` it does not
        depend on the coordinates.
    symmetry_data : dict
        The symmetry data vector, ``{corep: multiplicity}``: for a layer, the
        coreps at the points of its plane.
    site_content : SiteContent or None
        The site irreps the labels rest on.  Where one is ``"lowest-l"``, the
        class is that of the labeling the lowest-angular-momentum rule makes,
        and a declaration states another.
    group : ..groups.GroupData
        The group the class lives in, ``r.group``.
    numerical : SymmetryIndicatorNumerical
        The Smith form computed again, the vector on every point of the
        group's tables, and how a layer's coreps out of its plane were read.
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
    # the arithmetic: every corep of the group's elementary band representations,
    # the vector on them, the same in the Smith coordinates, and where the
    # divisors sit among those
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        body = (f"class {tuple(int(v) for v in self.values)} in {self.indicator_group}"
                if self.divisors else "trivial indicator group")
        stacked = (self.numerical is not None and self.numerical.completion is not None)
        return (f"SymmetryIndicator({self.group.name} #{self.group.number_str}, {body}"
                + (", completed on the stack" if stacked else "") + ")")


# --------------------------------------------------------------------------
# the symmetry data vector, and what it is read in
# --------------------------------------------------------------------------

def content_of(content) -> dict | Refusal:
    """A band set's symmetry data vector as multiplicities: how many of each corep it carries.

    A measured table and a vector written out by hand read the same way.

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
    src = content.symmetry_data if isinstance(content, _sewing.HSPTable) else content
    if not isinstance(src, Mapping):
        raise TypeError("content_of(): content is a table hsp_table returns or a "
                        f"{{corep: multiplicity}} mapping; got {type(content).__name__}.")
    out = {}
    for lab, v in src.items():
        iv = int(round(float(v)))
        if abs(float(v) - iv) > 1e-9:
            return Refusal("non_integer_multiplicity",
                           f"content['{lab}'] = {v} is not an integer multiplicity",
                           value=float(v), context={"label": str(lab)})
        if iv < 0:
            return Refusal("negative_multiplicity",
                           f"content['{lab}'] = {iv} is negative; a symmetry data vector "
                           "counts coreps and cannot be negative. If this is a "
                           "difference of two band sets, that difference is not a "
                           "band set.", value=float(iv), threshold=0.0,
                           context={"label": str(lab)})
        if iv:
            out[str(lab)] = iv
    return _tables.ordered_content(out)


@dataclass(frozen=True)
class _Source:
    """What a symmetry data vector is read in: its group, and what the measurement it came from carries.

    ``table`` is the :class:`.sewing.HSPTable` read, or ``None`` for a vector
    written out by hand; ``rep`` the representation a layer's decoupled stack
    is built from, where there is one; ``site_content`` the
    :class:`SiteContent` the labels rest on; ``heights`` a layer's orbitals'
    heights from its middle plane (``None`` for a crystal, and where the
    group's own sites stand for the orbitals); ``free_standing`` the
    free-standing layer's group as the measurement wrote it,
    ``{"number_str", "name", "size"}``; ``provenance`` how the group was
    measured, where the measurement says.
    """

    group: Any
    table: Any = None
    rep: Any = None
    site_content: Any = None
    heights: Any = None
    free_standing: Any = None
    provenance: Any = None


def _source_of_table(table) -> _Source:
    """What a table carries: its group, its site irreps, a layer's heights and representation."""
    rep = table._rep
    return _Source(group=table.group, table=table, rep=rep,
                   site_content=table._site_content, heights=table._heights,
                   free_standing=getattr(rep, "_free_standing_group", None),
                   provenance=table._group_provenance)


def _source_of_rep(group, rep=None) -> _Source:
    """What a representation carries for a vector read in ``group``, or the group alone where ``rep`` is ``None``."""
    return _Source(group=group, rep=rep, site_content=_sewing._site_content_record(rep),
                   heights=None if rep is None else _sewing._table_heights(rep),
                   free_standing=getattr(rep, "_free_standing_group", None))


def _source_of(symmetry_data, group, rep=None) -> _Source:
    """What a private entry reads: a table's own carrying, or ``group`` and ``rep`` beside a vector."""
    if isinstance(symmetry_data, _sewing.HSPTable):
        return _source_of_table(symmetry_data)
    return _source_of_rep(group, rep)


def _read(call, symmetry_data, realized) -> _Source | Refusal:
    """What ``call`` reads its first two arguments in: a table alone, or a vector with what :func:`.groups.realized_group` returned.

    A refusal handed in as ``realized`` is returned as it is.  Raises
    ``TypeError``, starting with ``call``, saying what to pass: for a table
    with anything beside it, a vector with nothing beside it, a second
    argument that is not what ``realized_group`` returns, and a first that is
    neither a table nor a mapping.
    """
    from ._realized import RealizedGroup

    if isinstance(symmetry_data, _sewing.HSPTable):
        if realized is not None:
            raise TypeError(f"{call}(): a table carries what it was read from; pass the "
                            f"table alone: {call}(t).")
        return _source_of_table(symmetry_data)
    if not isinstance(symmetry_data, Mapping):
        raise TypeError(f"{call}(): symmetry_data is a table hsp_table returns or a "
                        f"{{corep: multiplicity}} mapping; got "
                        f"{type(symmetry_data).__name__}.")
    if realized is None:
        raise TypeError(f"{call}(): a symmetry data vector written out by hand is read in "
                        "the measurement its coreps came from; pass it second: "
                        f"{call}(symmetry_data, r).")
    if isinstance(realized, Refusal):
        return realized
    if not isinstance(realized, RealizedGroup):
        raise TypeError(f"{call}(): pass what realized_group returns second, as in "
                        f"{call}(symmetry_data, r); got {type(realized).__name__}.")
    return _source_of_rep(realized.group, realized.rep)


def _named_points(group, counts):
    """The points a vector written out by hand is read at, or the refusal of the group's points.

    Every tabulated point of the model's plane, and each tabulated point out of
    it at which the vector names a corep; at such a point a corep it does not
    name is read as zero.
    """
    hsp = _tables._high_symmetry_points(group)
    if isinstance(hsp, Refusal):
        return hsp
    away = set(hsp.out_of_plane)
    named = {kpoint_of_label(label) for label, m in counts.items() if m} & away
    return tuple(hsp) + tuple(sorted(named, key=_tables.point_order))


def _check_names(call, group, names) -> None:
    """Raise, starting with ``call``, on a name in ``names`` the group does not tabulate."""
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
    """Which coreps of the group's elementary band representations a layer measures, and whether each of the others repeats a measured one.

    A layer's ``H(k)`` has no ``k_z``, so its coreps at ``k_z = 1/2`` are not
    measured.  Each that occurs in every elementary band representation as
    often as a measured one adds no condition, and where every one does, what
    is read on the measured coreps is exact.

    Parameters
    ----------
    group : .groups.GroupData
        The group, ``r.group``.
    kpoints : sequence of str, optional
        The tabulated names of the points the model samples.  Default: every
        tabulated point in its plane.
    require_exact : bool, default True
        Whether a corep left out that repeats none measured is refused.  With
        ``require_exact=False``, those are returned as
        :attr:`Restriction.orphans`.

    Returns
    -------
    Restriction or Refusal
        ``r.summary()`` prints it; ``r.copies`` pairs each corep left out with
        the measured ones it repeats, and ``r.exact`` says whether each does.
        A ``Refusal`` in its place is falsy and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``restriction_not_exact``
        a corep left out repeats none measured.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        no elementary band representations of this group, or no tables in this
        cell.

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
    """What :func:`in_plane_restriction` returns, the names already read.

    ``points`` are the names of the points read, every tabulated point of the
    plane where ``None``; ``layer`` is :func:`_layer_reading`'s record, whose
    columns alone the rows are compared on.
    """
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    data = _tables._ebr_data(group)
    if isinstance(data, Refusal):
        return data
    labels = list(data["labels_physics_convention"])
    matrix = np.asarray(data["matrix"], dtype=int)
    # A LAYER'S OWN BAND REPRESENTATIONS: where a decomposition runs over the
    # positions within the layer only, the rows are compared on those columns
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
    # ON A LAYER'S OWN BAND REPRESENTATIONS every row out of its plane repeats
    # one in it: the layer's operations carry no translation along its normal
    # and keep the positions within it within it.  That is a statement about
    # the whole plane, so it is checked where every point of it was compared
    if layer is not None and missing:
        hsp = _tables._high_symmetry_points(group)
        whole = not isinstance(hsp, Refusal) and set(hsp) <= set(points)
        off_plane = set(hsp.out_of_plane) if whole else set()
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
    ``heights``, the orbitals' own where they are given, else ``rep``'s
    model's, or the group's own sites without either -- with the symmetry
    finder's distance as the margin.  Returns ``{"columns",
    "outside", "extent"}``: the indices of ``data``'s elementary band
    representations within the layer, ``{position: heights}`` for the positions
    outside it, and the orbitals' ``(lowest, highest)`` heights.  ``None`` for
    a crystal periodic in three directions, and where the tables' Wyckoff
    letters cannot be read in the group's cell: every position is then kept,
    and the rows say whether the plane suffices.
    """
    reading = _sewing._layer_height(group)
    if reading is None:
        return None
    if isinstance(_tables._sites_in_tables_setting(group), Refusal):
        return None              # the letters are keys only once the settings agree
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

    The same entry of the tables, the same lattice and the same operations,
    each rotation exactly and each translation modulo the lattice to rounding:
    two records of one measurement are one group, and a group written in
    another cell, or another group, is not.
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
    """The band block an :class:`~.sewing.HSPTable` was handed, as it was handed in."""
    block = getattr(table, "bands", _NO_BLOCK)
    return getattr(table, "occ", None) if block is _NO_BLOCK else block


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


# --------------------------------------------------------------------------
# the public solvers
# --------------------------------------------------------------------------

def decompose(symmetry_data, realized=None, *, allow_incomplete=False,
              cap=DEFAULT_ENUMERATION_CAP,
              signed_combinations=True) -> EBRDecomposition | Refusal:
    r"""Every sum of elementary band representations, with nonnegative integer multiplicities, that carries a band set's symmetry data vector.

    An atomic limit's symmetry data vector is such a sum, so a band set with
    none is no atomic limit; one with a sum may still not be one, by an
    obstruction beyond what symmetry indicators see.  A layer is decomposed
    over the band representations of the Wyckoff positions within it.  A
    vector written out by hand is read at every tabulated point of the
    model's plane and at each point out of it where it names a corep.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The table :func:`.sewing.hsp_table` returns, which must be complete,
        or ``{corep: multiplicity}`` written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned, ``r``, the measurement
        the coreps came from.  Pass it beside a vector written out by hand,
        and with a table nothing.
    allow_incomplete : bool, default False
        Whether a table that refused at some points is decomposed, their
        coreps left out.  With ``allow_incomplete=True``, a point whose tables
        cannot be read is left out, and the sums are then a necessary
        condition only; never pass it for a point where the physics refused.
    cap : int, default 200
        The most sums listed.
    signed_combinations : bool, default True
        Whether a vector no sum carries is searched again with negative
        multiplicities allowed.  With ``signed_combinations=False``, it is
        not.

    Returns
    -------
    EBRDecomposition or Refusal
        ``d.summary()`` prints whether a sum carries the vector and each sum.
        ``d.decomposable`` says whether one does; ``d.candidate_sums`` holds
        each, a sum of site irreps at Wyckoff positions; ``d.certificate``
        what a vector no sum carries rests on, and ``d.signed_combinations``
        the search with negative multiplicities.  The record is truthy, one
        with no sum included; a ``Refusal`` in its place is falsy and names
        what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``incomplete_content``
        the table refused at some point.
    ``not_a_band_structure``
        the vector fails the group's compatibility relations: no set of bands
        apart from the rest at every k has it.
    ``unknown_label``
        a corep the group's elementary band representations do not hold.
    ``restriction_not_exact``
        a corep the model does not measure repeats none it does.
    ``non_integer_multiplicity`` / ``negative_multiplicity``
        the vector is no count of coreps.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention``
        no elementary band representations of this group to decompose against.

    Raises
    ------
    TypeError
        A table with anything beside it, a vector written out by hand with
        nothing beside it, a second argument that is not what
        :func:`.groups.realized_group` returned, or a first that is neither a
        table nor a mapping.
    ..refusal.InternalError
        On a layer's own band representations a corep out of its plane repeats
        none in it: the package disagreeing with itself.

    Examples
    --------
    lin3g's six lowest bands (:func:`.models.lin3g`):

    >>> d = decompose(hsp_table(r, bands=6))                      # doctest: +SKIP
    >>> [s.label for s in d.candidate_sums]                           # doctest: +SKIP
    ["-E2 at 1b (4mm1') + -E at 2c (mm21')", ...]
    """
    source = _read("decompose", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    return _decompose_in(source, symmetry_data, allow_incomplete=allow_incomplete,
                         cap=cap, signed_combinations=signed_combinations)


def _decompose(symmetry_data, group, *, rep=None, allow_incomplete=False,
               cap=DEFAULT_ENUMERATION_CAP, signed_combinations=True):
    """What :func:`decompose` returns for a vector read in ``group``, with the site irreps and heights of ``rep`` where it is given; a table is read in what it carries."""
    return _decompose_in(_source_of(symmetry_data, group, rep), symmetry_data,
                         allow_incomplete=allow_incomplete, cap=cap,
                         signed_combinations=signed_combinations)


def _decompose_in(source, symmetry_data, *, allow_incomplete=False,
                  cap=DEFAULT_ENUMERATION_CAP,
                  signed_combinations=True) -> EBRDecomposition | Refusal:
    """What :func:`decompose` returns for ``symmetry_data`` read in ``source``."""
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
            "the measured content is missing "
            f"{sorted(refused, key=_tables.point_order)}: {refused}. A missing "
            "constraint can only make a "
            "decomposition easier to find, so a partial symmetry data vector is not "
            "decomposed. Read the table's refusals; pass allow_incomplete=True only for "
            "a point whose tables cannot be read rather than one where the physics "
            "failed, and then read the sums as a necessary condition.",
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
            f"the measured labels {unknown} are not in the EBR basis of group "
            f"#{group.number_str}; the symmetry data vector and the basis are not written in "
            "the same alphabet.",
            value=float(len(unknown)), threshold=0.0,
            context={"unknown": tuple(unknown), "basis": tuple(labels_all)})

    # Only the rows that were measured may constrain the solve.  Dropping a
    # measurement while keeping its equation would *over*-constrain the problem
    # (the unmeasured multiplicities would be forced to zero), which is the
    # opposite of the weaker claim the override is supposed to buy -- so a point
    # that refused loses its EBR rows too.  A vector written out by hand is read
    # at the points of the plane and at each point out of it that it names.
    resolved = (tuple(table.multiplets) if table is not None
                else _named_points(group, counts))
    if isinstance(resolved, Refusal):
        return resolved
    exact_needed = table is None or table.complete
    # A LAYER'S OWN BAND REPRESENTATIONS.  A free-standing layer's group has
    # positions off the layer, which carry none of its orbitals: their band
    # representations are left out before anything is solved, so that a capped
    # enumeration lists the layer's own sums, and on the rest every row at
    # k_z = 1/2 repeats one in the plane.
    layer = _layer_reading(group, data, heights=source.heights)
    restriction = _restriction(group, resolved, require_exact=exact_needed, layer=layer)
    if isinstance(restriction, Refusal):
        return restriction
    columns = list(range(len(ebrs))) if layer is None else list(layer["columns"])

    # THE COMPATIBILITY RELATIONS, read before anything is solved: a vector no
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

    signed = None
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
    """The site irreps a measurement carries, as a :class:`SiteContent`; ``None`` where it carries none."""
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
        entry = _spacegroup._indicator_table(group.spinor, group._magnetic).get(
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
    :class:`TwoDimensionalReading` :attr:`SymmetryIndicatorNumerical.two_dimensional`
    holds, its support keyed by the paper's names of the indicators.

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
        "k_z = ½, k-points the layer's plane does not contain. The layer's H(k) has no "
        "k_z, so this is the class of the layer's decoupled stack -- the layer repeated "
        "along its normal with no hopping between copies -- whose rows at k_z = ½ repeat "
        "the in-plane ones: every component is an invariant of that stack, fixed by the "
        "layer's own content.")
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
    """``{ind: tr B(g, k)}`` of a whole band block, from a :class:`.sewing.CorepsAt`."""
    ch = record._characters
    columns = tuple(ch.multiplets[0].characters) if ch.multiplets else ()
    rows = np.asarray([[m.characters[i] for i in columns] for m in ch.multiplets],
                      dtype=complex).reshape(len(ch.multiplets), len(columns))
    total = rows.sum(axis=0) if rows.size else np.zeros(len(columns), dtype=complex)
    return {int(c): complex(v) for c, v in zip(columns, total)}

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
                "n_hoppings": int(rep._model.nhops), "worst": 0,
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

def _stack_fibers(table3, tol) -> tuple:
    """Each fiber at ``k_z = ½`` against the fiber in the plane at the same in-plane k.

    On a stack with no hopping between the copies the two are the same states:
    the same energies, the same partition into multiplets, the same characters
    at every operation both little groups hold.  Returns ``(fibers, first)``:
    one record per pair, and the first pair that differs, or ``None``.
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
        environment=(None if getattr(rep, "_environment", None) is None
                     else str(rep._environment)),
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
    represented = {int(i) for i in rep._indices}
    unrepresented = tuple(sorted(int(s.ind) for s in group._symmetries
                                 if int(s.ind) not in represented))
    if unrepresented:
        return stop("representation",
                    "the representation handed in has no matrix for operations "
                    f"{_text._runs(unrepresented)} of {group.name} #{group.number_str}, "
                    "so the decoupled stack's rows at k_z = ½ cannot be read in that group",
                    unrepresented=unrepresented)
    g3, asg3 = _layers._layer_on_stack(group, rep._assignment, stack)
    by_ind = {int(s.ind): s for s in g3._symmetries}
    rng = np.random.default_rng(20241004)
    battery = rng.uniform(-0.5, 0.5, size=(8, 3))
    battery[:len(_KZ_PROBE), 2] = _KZ_PROBE
    # the stack is built here and held by nothing else, so no edit of it is checked for
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
               "the layer's own matrices do not carry the stack's H(k, k_z) onto itself")
    rep3 = replace(rep3, numerical=replace(rep3.numerical, residuals=residuals))
    typat = tuple(int(group._typat[int(s)]) for s in rep._assignment.site_of_orbital)
    record["stack"] = {"number_str": str(g3.number_str), "name": str(g3.name),
                       "size": int(g3._size), "dim_k": int(g3._dim_k),
                       "periodic_dirs": tuple(int(p) for p in g3.periodic_dirs),
                       "typat": typat}

    # ---- 3. the content on the stack, same band block ----------------------
    within = f"the decoupled stack, read in {group.name} #{group.number_str},"
    table3 = _sewing._hsp_table_rep(rep3, bands=_table_bands(table))
    if isinstance(table3, Refusal):
        return stop("stack_content", f"{within} has content that could not be read: "
                    + table3.reason, gate=table3.gate)
    if not table3.complete:
        return stop("stack_content",
                    f"{within} has content missing at "
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
    indicator3 = _indicator_in(
        _Source(group=g3, table=table3, rep=rep3, site_content=table3._site_content,
                free_standing=getattr(rep3, "_free_standing_group", None),
                provenance=table3._group_provenance), table3)
    if isinstance(indicator3, Refusal):
        return stop("stack_indicator", f"{within} has no class there: "
                    + indicator3.reason, gate=indicator3.gate)
    data3 = _tables._ebr_data(g3)
    record["stack"]["divisors"] = _divisors_of(data3, g3.number_str)
    record["outcome"] = "completed"
    # the class lives in the layer's own group; what is printed of the vector is
    # its plane's, the vector on every point being the stack's measured one
    numerical = replace(indicator3.numerical, completion=_completion_record(record),
                        two_dimensional=_stack_reading(group, g3, indicator3, data3),
                        _group=group)
    return replace(indicator3, group=group,
                   symmetry_data=_plane_vector(group, table.symmetry_data),
                   numerical=numerical), record


def symmetry_indicator(symmetry_data, realized=None) -> SymmetryIndicator | Refusal:
    r"""The symmetry-indicator class of a band set: the class of its symmetry data vector among the band structures modulo the atomic limits.

    A nonzero class says no band set with this symmetry data vector is an
    atomic limit, nor any with atomic bands added; a zero class certifies
    nothing.  A layer's class is read in its slab's three-dimensional group,
    as the class of its decoupled stack, the layer repeated along its normal
    with no hopping between copies.  A table's coreps at points out of the
    layer's plane that repeat none in it are measured on that stack.  A vector
    written out by hand is read at the points of the plane and at each point
    out of it where it names a corep; a corep out of the plane it does not
    name takes the multiplicity of the corep in the plane it repeats in every
    elementary band representation of the layer's own Wyckoff positions.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The table :func:`.sewing.hsp_table` returns, which must be complete,
        or ``{corep: multiplicity}`` written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned, ``r``, the measurement
        the coreps came from.  Pass it beside a vector written out by hand,
        and with a table nothing.

    Returns
    -------
    SymmetryIndicator or Refusal
        ``si.summary()`` prints the class.  ``si.values`` is the class, one
        integer modulo each of ``si.divisors``, in the indicator group
        ``si.indicator_group``, of order ``si.order``; ``si.indicators`` the
        class in the named indicators of the paper that tabulates the stable
        real-space invariants; ``si.group`` the group it lives in.  The record
        is truthy whatever its class; a ``Refusal`` in its place is falsy and
        names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``incomplete_content``
        the table refused at some point; there is no override.
    ``indicator_not_evaluable``
        a layer's coreps at ``k_z = 1/2`` repeat none in its plane and could
        not be read: a vector whose layer's own band representations cannot
        be read in the group's cell, or a table whose decoupled stack stopped
        (``context["stack"]``).
    ``not_a_band_structure``
        the vector fails the group's compatibility relations.
    ``unknown_label``
        a corep the group's elementary band representations do not hold.
    ``non_integer_multiplicity`` / ``negative_multiplicity``
        the vector is no count of coreps.
    ``magnetic_type_out_of_scope``
        a magnetic group of type III or IV, which
        :func:`.groups.realized_group` never returns.
    ``ebr_tables`` / ``irrep_tables`` / ``ebr_convention`` / ``tables_not_ready``
        no elementary band representations of this group, or no tables in this
        cell.

    Raises
    ------
    TypeError
        A table with anything beside it, a vector written out by hand with
        nothing beside it, a second argument that is not what
        :func:`.groups.realized_group` returned, or a first that is neither a
        table nor a mapping.
    ..refusal.InternalError
        The tables' Smith form and the one computed again disagree, a check
        that ties the decoupled stack to the layer fails, or two classes share
        every named value: the package disagreeing with itself.

    Examples
    --------
    Fu, Kane and Mele's diamond with one bond detuned, its two lowest bands:

    >>> si = symmetry_indicator(hsp_table(r, bands=2))          # doctest: +SKIP
    >>> si.indicator_group, si.values, si.indicators                # doctest: +SKIP
    ('Z2 x Z4', (0, 1), {'z2w,1': 1, 'z2w,2': 1, 'z2w,3': 1, 'z4': 3})
    """
    source = _read("symmetry_indicator", symmetry_data, realized)
    if isinstance(source, Refusal):
        return source
    return _indicator_in(source, symmetry_data)


#: The restriction read on a group whose tables were read, refusing.
_RESTRICTION_REFUSED = (
    "#{}: the coreps the model measures could not be read ({}) after the group's "
    "elementary band representations and its points were: the package disagreeing "
    "with itself, not a property of the model.")


def _symmetry_indicator(symmetry_data, group, *, rep=None):
    """What :func:`symmetry_indicator` returns for a vector read in ``group``, with the site irreps and heights of ``rep`` where it is given; a table is read in what it carries."""
    return _indicator_in(_source_of(symmetry_data, group, rep), symmetry_data)


def _indicator_in(source, symmetry_data) -> SymmetryIndicator | Refusal:
    """What :func:`symmetry_indicator` returns for ``symmetry_data`` read in ``source``."""
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
            f"the measured content is missing {sorted(refused, key=_tables.point_order)}: "
            f"{refused}. An indicator "
            "reads the whole symmetry data vector, and the class of a vector with a point "
            "missing is the class of a different vector, so this content has no class "
            "here. Read the table's refusals. Completing the missing content from the "
            "stable real-space invariants is future work, and there is no "
            "`allow_incomplete` here.",
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
            f"the measured labels {unknown} are not in the EBR basis of group "
            f"#{group.number_str}; the symmetry data vector and the indicator basis are not "
            "written in the same alphabet.",
            value=float(len(unknown)), threshold=0.0,
            context={"unknown": tuple(unknown), "basis": labels})

    points = (tuple(table.multiplets) if table is not None
              else _named_points(group, counts))
    if isinstance(points, Refusal):
        return points
    restriction = _restriction(group, points, require_exact=False)
    if isinstance(restriction, Refusal):     # pragma: no cover - its gates were read above
        raise InternalError(_RESTRICTION_REFUSED.format(group.number_str, restriction.gate))
    layered = int(getattr(group, "_dim_k", 3)) < 3

    # ---- the rows out of the layer's own plane, measured on the stack -----
    # For a table, which names the band block the stack has to measure again,
    # and whose representation carries the model and the matrices the stack is
    # built from.  The stack obeys exactly what the layer obeys, so the class is
    # completed in the layer's own group, and only where rows at k_z = 1/2
    # repeat no in-plane row.
    completion = None
    if layered and table is not None and source.rep is not None and restriction.orphans:
        stacked, completion = _stack_in_group(
            table, group, source.rep, restriction, c=_layers.STACK_C,
            characters_tol=CHARACTERS_TOL)
        if stacked is not None:
            return stacked

    # ---- a vector written out by hand, read on the layer's own band
    # representations: there every row out of its plane repeats one in it, so a
    # corep there the vector does not name takes its partner's multiplicity
    if layered and table is None and restriction.orphans:
        layer = _layer_reading(group, data, heights=source.heights)
        if layer is not None:
            restriction = _restriction(group, points, require_exact=False, layer=layer)
            if isinstance(restriction, Refusal):     # pragma: no cover - as above
                raise InternalError(_RESTRICTION_REFUSED.format(group.number_str,
                                                                restriction.gate))

    if restriction.orphans:
        reason = (
            (f"the EBR rows {list(restriction.orphans)} of #{group.number_str} sit at "
             "k-points this content does not cover, out of a 2D model's periodic plane "
             "or carrying no information, and they duplicate no row it does cover, so "
             "the symmetry data vector cannot be completed and the indicator has no value "
             "here. The decoupled stack of the layer, the crystal on which those rows are "
             f"measured, did not supply them: {str(completion['stop_reason']).rstrip('. ')} "
             "(what was measured, and where it stopped, is in context['stack']).")
            if completion is not None else
            (f"the coreps {list(restriction.orphans)} of #{group.number_str} sit at "
             "k-points out of the layer's plane, repeat none in it and cannot be read off "
             "it in this group's cell; pass the table tqc.hsp_table(r, bands=...) returns, "
             "whose coreps there are measured on the layer's decoupled stack."))
        return Refusal(
            "indicator_not_evaluable", reason,
            value=float(len(restriction.orphans)), threshold=0.0,
            context={"orphans": tuple(restriction.orphans),
                     "points": tuple(restriction.kpoints),
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

    verification = _verify_smith(group.number_str, matrix, u, v, d, y,
                                 slots, divisors, values, order)

    _named, _touched, _note, coeffs = _named_forms(group, data, matrix, y, mtype)
    two_d = _two_dimensional_reading(group, labels, u, slots, divisors, values, coeffs)
    # the paper's named indicators: one rational solve of BR.p = y -- the
    # identical arithmetic `srsi.srsi_of_bands` does on the same y -- and then
    # the tabulated forms
    indicators, indicators_note, _tex = _srsitables._canonical_reading(
        group, mtype, u, v, d, y, divisors)

    provenance = ((table._group_provenance if table is not None else None)
                  or source.provenance or str(getattr(group, "_provenance", "cell")))
    # the free-standing layer's group, as `realized_group` measured it beside a
    # substrate's, named where it is larger than the group the class lives in
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
    """The coreps of a symmetry data vector at the points of a layer's plane; the whole vector for a crystal."""
    if int(getattr(group, "_dim_k", 3)) >= 3:
        return _tables.ordered_content(counts)
    hsp = _tables._high_symmetry_points(group)
    if isinstance(hsp, Refusal):
        return _tables.ordered_content(counts)
    plane = set(hsp)
    return _tables.ordered_content({label: m for label, m in counts.items()
                                    if kpoint_of_label(label) in plane})


def _incomplete_table(content):
    """The ``incomplete_content`` refusal for a table that refused at some point, or ``None``.

    A mapping, and a complete table, return ``None``.
    """
    if not isinstance(content, _sewing.HSPTable) or content.complete:
        return None
    refused = {name: r.gate for name, r in (content.refusals or {}).items()}
    names = sorted(refused, key=_tables.point_order)
    return Refusal(
        "incomplete_content",
        f"the table is missing the coreps at {names}: {refused}. Its symmetry data "
        "vector holds nothing at those points, and a difference taken with it would "
        "count them as empty. Read each missing point's refusal in the table's "
        "`refusals`.",
        value=float(len(refused)), threshold=0.0,
        context={"refusals": refused, "resolved": tuple(content.multiplets)})


def subtract(symmetry_data_a, symmetry_data_b) -> dict | Refusal:
    """The symmetry data vector of the bands in one set and not in another: ``a − b``.

    Parameters
    ----------
    symmetry_data_a, symmetry_data_b : .sewing.HSPTable or mapping
        The two vectors, each a table :func:`.sewing.hsp_table` returned or
        ``{corep: multiplicity}`` written out by hand; ``b`` is contained in
        ``a``.

    Returns
    -------
    dict or Refusal
        ``{corep: multiplicity}``, the zeros dropped: empty, and so falsy, where
        the two are equal, so ``isinstance(out, Refusal)`` is what tells a
        refusal apart.

    Refusals
    --------
    ``incomplete_content``
        either is a table that refused at some point.
    ``negative_multiplicity``
        ``b`` is not contained in ``a``: a difference of two band sets is one
        only where one contains the other.  A signed combination is what
        :attr:`EBRDecomposition.signed_combinations` holds.
    ``non_integer_multiplicity``
        either is no count of coreps.

    Raises
    ------
    ValueError
        Two tables read in different groups, whose coreps are not one alphabet.
    """
    own_a = symmetry_data_a.group if isinstance(symmetry_data_a, _sewing.HSPTable) else None
    own_b = symmetry_data_b.group if isinstance(symmetry_data_b, _sewing.HSPTable) else None
    if own_a is not None and own_b is not None and not _same_group(own_a, own_b):
        raise ValueError(
            f"subtract(): the tables were read in different groups, {own_a.name} "
            f"#{own_a.number_str} and {own_b.name} #{own_b.number_str}, "
            "whose coreps cannot be subtracted label by label.")
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
            f"the difference has negative multiplicities {bad}; the second content is "
            "not contained in the first, so their difference is not a band set. A "
            "signed EBR combination is still meaningful (it is what fragility means) "
            "but it is not a symmetry data vector and cannot be decomposed as one.",
            value=float(min(bad.values())), threshold=0.0,
            context={"negative": bad, "a": a, "b": b})
    return out


def regular_content(group, *, kpoints=None) -> dict | Refusal:
    r"""The symmetry data vector of one copy of the regular band representation of the group: ``|U|`` bands at every k.

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

    ``regular_content_not_integral``
        the tables' characters give a multiplicity that is no integer, or a
        vector of other than ``|U|`` bands at a point.
    ``tables_not_ready`` / ``out_of_plane``
        no tables in this cell, or a point out of a layer's plane.

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
    """What :func:`regular_content` returns; a name the group does not tabulate raises, starting with ``call``."""
    gate = _spacegroup._lattice_not_descended(group)
    if gate is not None:
        return gate
    hsp = _tables._points_in_table_order(group)
    if isinstance(hsp, Refusal):
        return hsp
    wanted = tuple(hsp) if kpoints is None else tuple(str(p) for p in kpoints)
    for name in wanted:
        if name not in hsp and name not in hsp.out_of_plane:
            _tables._resolve_kpoint(call, group, name)

    n_u = len(group._u_symmetries)
    out, bands = {}, {}
    for name in wanted:
        if name in hsp.out_of_plane:
            # A point out of a layer's plane, which its little group refuses
            refused = _tables._little_group(group, _tables._table_kpoints(group)[name])
            if not isinstance(refused, Refusal):
                raise InternalError(
                    f"#{group.number_str}: {name} is listed as out of the layer's plane, "
                    "yet its little group was read.")
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
            m = n_u * dim / norm
            if abs(m - round(m)) > _INT_TOL:
                return Refusal(
                    "regular_content_not_integral",
                    f"the regular multiplicity of {label} at {name} came out {m:.6g}, "
                    "which is not an integer; the tabulated characters and the little "
                    "group disagree about this corep.",
                    value=float(m), threshold=float(_INT_TOL),
                    context={"kpname": name, "label": label, "n_unitary": n_u,
                             "little_group": len(lg.unitary)})
            out[label] = int(round(m))
            total += round(m) * dim
        bands[name] = total
    bands = {n: bands[n] for n in sorted(bands, key=_tables.point_order)}
    off = {n: b for n, b in bands.items() if abs(b - n_u) > _INT_TOL}
    if off:
        return Refusal(
            "regular_content_not_integral",
            f"the regular content does not carry the {n_u} bands of one regular band "
            f"representation at every point: {off}",
            value=float(max(off.values())), threshold=float(n_u),
            context={"bands": bands, "n_unitary": n_u})
    return _tables.ordered_content(out)


def regular_completion(symmetry_data, realized=None, *,
                       bands=None) -> RegularCompletion | Refusal:
    """The bands that would carry a band set up to whole copies of the regular band representation, and the orbitals they would be.

    The regular band representation is ``|U|`` bands at every k, ``|U|`` the
    number of unitary operations up to lattice translations, the single-valued
    group's for a spinful model (:func:`regular_content`).  Both sides of a gap
    completed to it make an atomic limit together.

    Parameters
    ----------
    symmetry_data : .sewing.HSPTable or mapping
        The table :func:`.sewing.hsp_table` returns, or ``{corep:
        multiplicity}`` written out by hand.
    realized : .groups.RealizedGroup, optional
        What :func:`.groups.realized_group` returned, ``r``.  Pass it beside a
        vector written out by hand, and with a table nothing.
    bands : int or sequence of int
        The bands of the completed system: a number of them, or their indices,
        a multiple of ``|U|`` in all.  Required.

    Returns
    -------
    RegularCompletion or Refusal
        ``rc.summary()`` prints it.  ``rc.padding`` holds the coreps the
        missing bands carry, ``rc.decomposition`` the orbitals they would be,
        what :func:`decompose` returns for the padding; ``rc.regular`` one
        copy's symmetry data vector and ``rc.bands_per_copy`` its bands.  A
        ``Refusal`` in its place is falsy and names what stopped the call.

    Refusals
    --------
    A refusal handed in as ``realized`` is returned as it is.

    ``incomplete_content``
        the table refused at some point.
    ``not_a_regular_multiple``
        the number of bands is no multiple of ``|U|``.
    ``negative_multiplicity``
        the band set carries more of a corep than that many regular copies do:
        it is no part of them.
    ``restriction_not_exact``
        a corep of the padding the model does not measure repeats none it does.
    ``regular_content_not_integral``, ``not_a_band_structure``, ``unknown_label``, ...
        those of :func:`regular_content` and :func:`decompose`.

    Raises
    ------
    TypeError
        No ``bands``; or a table with anything beside it, a vector written
        out by hand with nothing beside it, or a second argument that is not
        what :func:`.groups.realized_group` returned.

    Examples
    --------
    lin3g's six lowest bands, completed to its eight, one regular copy: the
    padding ``-GM7 + -M6 + -X5`` is ``-E1`` at ``1b``, the coreps of the two
    bands above the gap.

    >>> rc = regular_completion(hsp_table(r, bands=6), bands=8) # doctest: +SKIP
    >>> [s.label for s in rc.decomposition.candidate_sums]         # doctest: +SKIP
    ["-E1 at 1b (4mm1')"]
    """
    if bands is None:
        raise TypeError("regular_completion(): missing required argument 'bands'")
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

    n_u = len(group._u_symmetries)
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
    occ = content_of(symmetry_data)
    if isinstance(occ, Refusal):
        return occ

    padding = subtract(target, occ)
    if isinstance(padding, Refusal):
        return padding

    # the padding is a vector of the same measurement, read as one written out
    got = _decompose_in(replace(source, table=None), padding)
    if isinstance(got, Refusal):
        return got
    return RegularCompletion(copies=copies, bands_per_copy=n_u, padding=padding,
                             decomposition=got, symmetry_data=occ,
                             regular=dict(target_one), _group=group)
