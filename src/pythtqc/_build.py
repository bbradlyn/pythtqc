"""The representation of a space group on a model's states.

The representation has one matrix per operation, built on what the
measurement of the group read, in the convention of the irreptables package's
tables.  With it come the atomic-limit table of the orbitals, the site irreps
chosen for the copies of a conserved pseudospin, and the band representation
each Wyckoff orbit of orbitals induces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import itertools

import numpy as np

from . import _bloch
from . import _measure
from . import _sitechars
from . import srsi as _srsi
from .refusal import InternalError, Refusal
from ._tolerance import CHARACTER, IN_SPACE, SNAP, ZERO, noise_floor
from ._text import _counted_as, _fmt_char
from ._seitz import _record_name
from ._detector import _READOFF_MAX_BYTES
from ._matrices import (
    RepAnchor, SiteMultiplets, _ansatz_blocks, _ansatz_member, _bfs_words,
    _channel_classes, _characters_of_subgroup, _closure, _commuting_pair_phase,
    _covariance_battery, _identity_position, _lift_of, _minimal_generators,
    _moves_padded, _order_and_zeta, _power, _product_table_residual, _propagated,
    _proportionality, _real_characters, _resolve_spin, _settle_unit,
    _phase_rounding, _spin_pair_split, _translation_phase, bloch_operator,
    covariance_gate)
from ._grammar import _declaration_grammar
from ._naming import _site_tables
from .symmetry import _MultiplicitySpace, SpaceGroupRep, SpaceGroupRepNumerical, _fold
from ._multiplets import (
    _channel_perms, _factor_said, _multiplet_blocks, _orbit_factor, _site_multiplets)
from ._declaration import (
    _covariant_subgroup, _declaration_breaks, _declared_blocks, _declared_key,
    _lead_of, _multiplet_characters, _named_content, _normalize_declaration,
    _verdicts_and_choices)
from ._labeling import (
    _assembled, _atomic_limit_table, _coupled_parts, _labeling, _wyckoff_sites)


@dataclass(frozen=True)
class _ReadOff:
    """The discrete choices of a build, measured off the solution spaces of the generators.

    Attributes
    ----------
    choices : list or tuple
        The ``(orientation, factors)`` each branch read cleanly.
    branches : dict
        The record of each branch read cleanly, by orientation.
    reason : str or None
        Why no branch was read; ``None`` where one was.
    record : dict
        The record of the read-off as the build returns it: every branch, and
        the size of the enumeration behind it.
    read : dict
        The factors read on each branch, ``{orientation: {(ind, orbit): factor}}``.
    """

    choices: object
    branches: dict
    reason: object
    record: dict
    read: dict


@dataclass(frozen=True)
class _GateOutcome:
    """The candidate representations of a build, through the three checks.

    Attributes
    ----------
    records : list
        Every candidate built.
    distinct : list
        The candidates grouped where they build the identical representation.
    covariant : list
        The groups that carry ``H(k)`` onto ``H(Rk)`` within the covariance check's
        tolerance.
    consistent : list
        Those that also factorize on every site into a character times the
        group's own internal factor.
    survivors : list
        Those that also reproduce the multiplication table.
    search : dict
        The record of the search as the build returns it.
    """

    records: list
    distinct: list
    covariant: list
    consistent: list
    survivors: list
    search: dict


@dataclass(frozen=True)
class _PinnedBranch:
    """The survivors of the checks once a tie between the two orientation branches is settled.

    Attributes
    ----------
    survivors : list
        The candidates left.
    tie : bool
        Whether the branch was pinned by convention, to ``+1``.
    note : str or None
        The sentence saying why the branch was pinned; ``None`` where it was not.
    """

    survivors: list
    tie: bool
    note: object


@dataclass(frozen=True, eq=False)
class _PartUnit:
    """The coupled parts of the crystal settled together, as one block of states.

    Attributes
    ----------
    states : ndarray of int
        Their states.
    lead : int
        The multiplet they are settled at.
    local : dict
        Each state's position in the block, ``{state: position}``.
    anchor : bool
        Whether the block holds the anchor's part.
    orbitals : tuple of int
        Their orbitals.
    """

    states: object
    lead: int
    local: dict
    anchor: bool
    orbitals: tuple


@dataclass(frozen=True, eq=False)
class _Operations:
    """The operations a build represents, with their generators and the word of each.

    Attributes
    ----------
    group : ..groups.GroupData
        As in :func:`_build_rep`.
    syms : list
        The group's operations, in the group's order.
    kept : list of int
        The positions in ``syms`` of the operations represented.
    u_pos : list of int
        Those of them that are unitary.
    au_pos : list of int
        Those of them that carry time reversal.
    gens : tuple of int
        The generators of the unitary operations, by position.
    words : dict
        Each unitary operation's word in the generators, by position.
    parent : dict
        Each unitary operation's edge in the tree of words.
    T_pos : int or None
        The antiunitary operation the others are written from: a pure time reversal where
        one is represented.
    au_of : dict
        For each antiunitary operation, the unitary one whose product with ``T_pos`` it is.
    prod : ndarray of int
        The group's multiplication table, by position.
    spf : ndarray of int
        The sign the spin-1/2 lifts carry in each product.
    ident : int
        The position of the identity.
    nstate : int
        The model's number of states.
    spin : ..spin.SpinBasis or None
        The declared spinor basis, resolved; ``None`` where none is declared.
    validated : ..groups.ValidatedGroup
        The measurement of the operations, each read in its own form.
    """

    group: object
    syms: object
    kept: object
    u_pos: object
    au_pos: object
    gens: object
    words: object
    parent: object
    T_pos: object
    au_of: object
    prod: object
    spf: object
    ident: object
    nstate: object
    spin: object
    validated: object


@dataclass(frozen=True, eq=False)
class _Multiplets:
    """The multiplets of a build, their orbits under the operations, and the anchor.

    Attributes
    ----------
    layout : SiteMultiplets
        The multiplets.
    su2 : bool
        Whether the states carry the spin-1/2 lift.
    nchan : int
        The number of multiplets.
    chan_states : tuple
        The states of each multiplet.
    cperm : dict
        Where each operation carries each multiplet, by position.
    classes : ndarray of int
        The orbit each multiplet belongs to.
    ncls : int
        The number of orbits.
    anchor : int
        Orbital 0's multiplet, which the representation is written against.
    anchor_cls : int
        Its orbit.
    free_cls : list of int
        The other orbits.
    site_tables : dict
        The character tables of the sites, empty where they are not read or not available.
    decls : list
        The declarations, read on the multiplets.
    char_note : str
        If nonempty, why the site tables are unavailable.
    ref_orb : int
        The orbital the lowest-angular-momentum rule reads first,
        ``reference_orbital`` of :func:`_build_rep`.
    ref_chan : int
        Its multiplet.
    """

    layout: object
    su2: object
    nchan: object
    chan_states: object
    cperm: object
    classes: object
    ncls: object
    anchor: object
    anchor_cls: object
    free_cls: object
    site_tables: object
    decls: object
    char_note: object
    ref_orb: object
    ref_chan: object


@dataclass(frozen=True, eq=False)
class _Measured:
    """What the measurement of a build read: the blocks, the scalars and the coupled parts.

    Attributes
    ----------
    verdict : .._declaration._Verdict
        The measurement, handed in or made once.
    measured_of : dict
        Each operation's internal block on each multiplet.
    zero : float
        The size below which a block or a ratio vanishes.
    snap : float
        The distance within which a factor is read as a root of unity.
    S_gen : dict
        The internal blocks of each generator and of the antiunitary operation the others
        are written from, one for each multiplet.
    reversal : ndarray of complex
        The scalar time reversal carries on each orbit.
    pool : list of dict
        The one-dimensional characters the labeling is chosen among.
    unpinned : frozenset of int
        The orbits the hoppings leave free of the anchor.
    coupled_parts : list
        The orbits of each coupled part of the crystal, the anchor's first.
    part_of_class : dict
        The coupled part of each orbit.
    free_lead : dict
        The lead orbit of the part of each orbit outside the anchor's.
    """

    verdict: object
    measured_of: object
    zero: object
    snap: object
    S_gen: object
    reversal: object
    pool: object
    unpinned: object
    coupled_parts: object
    part_of_class: object
    free_lead: object


@dataclass(frozen=True, eq=False)
class _Battery:
    """What every candidate of a build is measured against, and the choices on offer.

    Attributes
    ----------
    tree : dict
        The tree of words, as the generators are propagated along it.
    k_pts : ndarray
        The k-points of the covariance battery, in reduced coordinates.
    bloch : .._bloch._BlochSum
        The model's Bloch Hamiltonian, evaluated by k-point.
    H : ndarray
        ``H(k)`` on the battery.
    Hrot : dict
        ``H`` at each operation's image of the battery.
    kact : dict
        Each operation's action on k.
    tol : float
        The covariance check's tolerance, in the units of ``H``.
    internal_tol : float
        As in :func:`_build_rep`.
    product_tol : float
        As in :func:`_build_rep`.
    _gate_lifts : dict
        The block each multiplet carries under each unitary operation, which the
        internal-factor check measures against.
    reach : float
        The largest defect of the product table, as the largest entry of a difference of
        matrices, that is replaced by the nearest exact representation.
    units : list of _PartUnit
        The coupled parts, as they are settled together.
    phases : tuple of complex
        The phases the enumeration tries on each free orbit.
    n_free : int
        The number of phases a candidate chooses.
    orients : tuple of int
        The orientation branches tried.
    all_blocks : list
        The multiplets one choice applies to at once.
    part_chans : list of list of int
        The multiplets of each coupled part.
    """

    tree: object
    k_pts: object
    bloch: object
    H: object
    Hrot: object
    kact: object
    tol: object
    internal_tol: object
    product_tol: object
    _gate_lifts: object
    reach: object
    units: object
    phases: object
    n_free: object
    orients: object
    all_blocks: object
    part_chans: object


@dataclass(frozen=True, eq=False)
class _Accepted:
    """The candidate representation the checks accept, with the record of the search.

    Attributes
    ----------
    winner : dict
        The candidate, its settled matrices with it.
    search : dict
        The record of the search, as the representation returns it.
    branch_tie : bool
        Whether the orientation branch was pinned by convention.
    """

    winner: object
    search: object
    branch_tie: object


def _no_multiplicity_representation(group, record, layout, kept, syms) -> Refusal:
    r"""The ``rep_not_multiplicative`` of a pseudospin whose copies no representation fits.

    ``record`` is the :class:`.._sectors._OrbitRecord` of an orbit of sectors,
    each holding ``record.dim`` identical copies, which admit no
    representation.  On the copies, a matrix of the group is ``B ⊗ B'``, with
    ``B`` the factor ``H`` fixes and ``B'`` a matrix on the copies.  None of
    the ``record.n_candidates`` sums of that dimension of the irreducible
    representations of the stabilizer of one sector makes up the factor system
    ``B`` leaves, time reversal's square included.  ``group`` is named in the
    reason.  ``layout`` holds the
    multiplets, and it is carried in the context with the indices of the
    operations at positions ``kept`` of ``syms``.  Returns the refusal.
    """
    k, orbitals = int(record.dim), tuple(record.orbitals)
    named = f"{group.name} #{group.number_str}"
    return Refusal(
        "rep_not_multiplicative",
        f"the Hamiltonian conserves a {_fold(k)} pseudospin on orbitals {orbitals}, held "
        f"as sets of {k} identical copies that the operations carry onto one another, "
        f"and none of the {int(record.n_candidates)} sums of dimension {k} of the "
        f"irreducible representations of the unitary operations that map one set onto "
        f"itself gives its copies the factor system they need, time reversal's square "
        f"included: no representation of {named} with its own factor system exists on "
        f"these states.",
        value=0.0, threshold=1.0,
        context={"multiplets": layout, "multiplet": orbitals, "multiplicity": k,
                 "n_candidates": int(record.n_candidates),
                 "indices": tuple(int(syms[p].ind) for p in kept)})


def _raise_unformed(group, unformed, n_cand, gate, *, spinful):
    r"""Raise ``InternalError`` for operations with no matrix of their own form.

    ``group`` holds the operations.  ``unformed`` is ``{ind: {"own_form":
    condition, "covariance": residual, ...}}``, read off the record each
    operation's own form was read on (:func:`.._measure._own_form_read`).
    The condition is ``"site permutation"``, where no matrix carries each
    site's states onto its image's, or ``"spin lift"``, where none carries
    the operation's own spin rotation.  The covariance is that of the
    matrix of that form nearest the operation's solution space, where one
    was measured.  ``n_cand`` is how many operations were to be represented,
    and ``gate`` the covariance tolerance they were kept at.  ``spinful`` says
    whether the states carry a spin, the model's own or declared; only then
    is the spin rotation named in the message.
    """

    syms = {int(s.ind): s for s in group._symmetries}

    def _named(ind):
        return (f"{_record_name(syms[ind], group._lattice)} "
                f"(#{ind}, its {unformed[ind]['own_form']})")

    named = ", ".join(_named(ind) for ind in sorted(unformed))
    measured = [float(r["covariance"]) for r in unformed.values()
                if r.get("covariance") is not None]
    closest = min(measured) if measured else None
    reach = ("" if closest is None else
             f" The nearest matrix of their own form carries H(k) onto H(Rk) to "
             f"{closest:.3g}, against the {float(gate):.3g} they were kept at.")
    carried = ("every site's states onto its image's with the operation's own spin "
               "rotation" if spinful else
               "every site's states onto its image's")
    raise InternalError(
        f"{len(unformed)} of the {int(n_cand)} operations to be represented, "
        f"{_counted_as(spinful)}, are not realized on these orbitals: {named}. "
        f"Matrices with each one's action on k carry H(k) onto H(Rk), but none carries "
        f"{carried}.{reach} Report it, with the model and the call.")


def _build_rep(model, group, assignment, *, validated=None, multiplets=None,
               reference_orbital=0, characters=None, spin=None,
               generators=None, operations=None,
               phases=(1, -1), orientation_branch=None, include_vacuum_ops=False,
               strategy="readoff", max_candidates=2 ** 20,
               k_pts=None, n_battery=6, tol=None, internal_tol=1e-9,
               product_tol=1e-9, seed=101, _choices=None,
               _returned=True, _labeled=True,
               _search_record=None) -> SpaceGroupRep | Refusal:
    r"""The representation of ``group`` on the model's states: a matrix for each operation.

    Each matrix is ``D(g) = O(g) ⊗ S(g)``.  ``O(g)`` carries each multiplet onto
    its image, as the operation carries the sites, with a factor on each orbit
    of multiplets relative to the orbit of the anchor, orbital 0's multiplet.
    ``S(g)``, the internal factor, is the operation's own matrix on a
    multiplet: its spin rotation, or ``1`` without spin, on a one-dimensional
    multiplet, and the block of the model's own solution space on a wider one.
    An antiunitary operation's matrix carries time reversal's internal factor
    on each multiplet times the scalar the measurement read on each orbit.  The
    factors on the orbits, one per generator and orbit, and the orientation
    branch, ``D(g)`` or ``D(g⁻¹)``, are read off the solution spaces or
    enumerated.  Each candidate is held to three checks:
    ``max_k ‖D H(k) D† − H(Rk)‖ ≤ tol``; every block a multiple of the group's
    own internal factor, within ``internal_tol``; and the group's
    multiplication table, the signs of the spin-1/2 lifts included, within
    ``product_tol``.  A coupled part of the crystal is a set of multiplets the
    hoppings connect, directly or through others.  What the
    declarations leave open of the one-dimensional character the hoppings leave
    on each coupled part, and of the representation a conserved pseudospin's
    copies carry, is then chosen by the lowest-angular-momentum rule, so that
    the matrices are the representation induced from the site irreps returned,
    in the convention of the irreptables package's tables.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, with the values of its parameters in place.
    group : ..groups.GroupData
        The group, as ``realized_group(model).group`` holds it.
    assignment : ..groups.OrbitalAssignment
        Each orbital's site in ``group``, from :func:`.._tables._assign_orbitals`
        on this model and group.
    validated : ..groups.ValidatedGroup, default None
        Which operations ``H`` obeys, as :func:`.._measure._validate_group`
        measures it.  A record not read in each operation's own form is read so
        first, once.  ``None`` measures it here at the defaults.
    multiplets : list or SiteMultiplets, default None
        The multiplets each site's orbitals are bound into, stated: one list per
        site of the multiplets it holds, each written as its orbitals, or a
        :class:`SiteMultiplets`.  ``None`` measures them.
    reference_orbital : int, default 0
        The orbital the lowest-angular-momentum rule reads first.
    characters : sequence or mapping, default None
        The declared site irreps, in the forms ``characters=`` of
        :func:`..groups.realized_group` takes.  ``None`` declares nothing.
    spin : ..spin.SpinBasis, str or dict, default None
        The spinor basis of a spinless model: a :class:`..spin.SpinBasis`,
        ``"interleaved"`` or ``"blocked"``, or the keyword arguments of
        :func:`..spin.spin_basis`.  ``None`` declares none.
    generators : sequence of int, default None
        The indices of generators of the represented unitary operations.
        ``None`` finds a set of at most three from the product table.
    operations : sequence of int, default None
        The indices of the operations to represent: a subgroup holding the
        identity, each antiunitary operation in it the product of one of them,
        time reversal where it is represented, with a represented unitary one.
        ``None`` takes every operation ``validated`` holds.
    phases : sequence of complex, default (1, -1)
        The factors the enumeration tries on each orbit of multiplets, per
        generator, relative to the anchor's orbit.
    orientation_branch : {None, 1, -1}, default None
        The orientation branches tried: ``None`` both, ``1`` only ``D(g)``,
        ``-1`` only ``D(g⁻¹)``.
    include_vacuum_ops : bool, default False
        Whether operations that move a slab cell's vacuum direction are
        represented.
    strategy : {"readoff", "enumerate"}, default "readoff"
        How candidates are found.  ``"readoff"`` reads the factors off the
        solution spaces, one candidate a branch.  It enumerates where it reads
        nothing or what it read does not give one representation; where none of
        what it read passes the covariance check and the best lies between
        ``tol`` and a larger declared tolerance, nothing is enumerated and the
        call refuses ``rep_not_covariant``.  ``"enumerate"`` builds every
        combination of ``phases``.
    max_candidates : int, default 2**20
        The most candidates an enumeration may run: of the factors on the
        orbits, or of the representations that sets of states the Hamiltonian
        does not couple and no operation exchanges, or a conserved pseudospin's
        copies, may carry.
    k_pts : ndarray, shape (n, dim_k), default None
        The k-points of the covariance check, in reduced coordinates.  ``None``
        draws ``n_battery`` points uniformly from ``[-0.5, 0.5)``.  At
        high-symmetry points wrong candidates can pass the covariance check.
    n_battery : int, default 6
        How many random k-points are drawn where ``k_pts`` is ``None``.
    tol : float, default None
        The covariance check's tolerance, in the units of ``H``.  ``None`` is
        :func:`.._matrices.covariance_gate` on the k-points: the rounding of
        the representation's own arithmetic and of the Bloch phases, and never
        below ``5e-14·max|H|``.
    internal_tol : float, default 1e-9
        The internal-factor check's tolerance on ``max ‖B(g) − χ S(g)‖`` over
        every block ``B(g)`` of every unitary operation, ``χ`` free per block.
        Dimensionless.  It also bounds from below how far outside a solution
        space counts as inside it, and a product-table miss of at most
        ``100·internal_tol``, capped at ``1e-2``, is replaced by the nearest
        exact representation.
    product_tol : float, default 1e-9
        The product-table check's tolerance on
        ``max ‖D(g1) D(g2) − spf·D(g1 g2)‖``, ``spf`` the sign of the spin-1/2
        lifts.  Dimensionless.
    seed : int, default 101
        The seed of the random k-points.
    _choices : .._declaration._Verdict, default None
        A measurement on the multiplets made earlier, as
        :func:`..groups.realized_group` makes it.  It is used where it was made
        on the same unitary operations, declarations, tolerance, orbits and
        anchor, and the measurement is made here otherwise.
    _returned : bool, default True
        Whether this is the representation returned: only then are the Bloch
        operators' factor system and each orbit's band representation measured.
    _labeled : bool, default True
        Whether the labeling is chosen.  With ``_labeled=False`` and nothing
        declared, the site tables are not read and no atomic-limit table is
        made.
    _search_record : dict, default None
        Where given, the record of the search is put under its ``"search"`` key,
        beside a refusal as well.

    Returns
    -------
    SpaceGroupRep or Refusal
        The one representation that passes the three checks, its labeling
        chosen: a matrix for each represented operation, the atomic-limit table
        of the orbitals, and the record of the search.  Candidates that build the
        same matrices count once.  Where the two orientation branches both pass
        and are one representation up to a one-dimensional character on each
        coupled part, the branch ``+1`` is returned with ``_branch_tie`` set.
        The covariance, product-table and factor-system residuals are measured on
        the matrices returned.  No environment is recorded: ``_environment`` is
        ``None``.

    Examples
    --------
    The build ``realized_group`` made for lin3g (:func:`.models.lin3g`), made
    again with each site's orbitals stated as one multiplet:

    >>> r = realized_group(models.lin3g())
    >>> rep = _build_rep(r.model, r.group, r._assignment, validated=r.numerical.validated,
    ...                  multiplets=[[orbs] for orbs in r._assignment.site_orbitals])
    >>> print(rep.summary().splitlines()[0])
    each operation's matrix on the 8 states, by index, each orbital's spin up then down along z   (.matrices)

    Refusals
    --------
    ``rep_double_valued_spinless``
        a spinless model in double-valued tables, with no ``spin=``.
    ``spin_pair_split``
        a declared spin pair whose two members sit on different sites.
    ``detector_too_large``
        ``validated`` is ``None``, and the measurement made here needs more
        memory than its limit.
    ``unknown_label`` / ``ambiguous_multiplet`` / ``site_group``
        a declaration its site cannot read, or, where something is declared, a
        site's character table that cannot be made.
    ``declaration_breaks_group``
        the declared site irreps are not covariant under every operation of
        ``group``.  The context names the subgroup they are covariant under.
    ``rep_not_covariant``
        no candidate passes the covariance check.
    ``rep_internal_factor``
        covariant candidates exist, and on none is every block a multiple of the
        group's own internal factor.
    ``rep_not_multiplicative``
        candidates pass both, and none reproduces the multiplication table.  The
        context names two commuting operations whose matrices commute only up to
        a phase, where there are such, or the copies of a conserved pseudospin
        that admit no representation.
    ``canonical_norm`` / ``antiunitary_anchor``
        no candidate that passes the first two checks can be brought to its
        canonical norm and antiunitary anchor.
    ``rep_not_unique``
        more than one representation passes all three checks, other than the two
        orientation branches pinned above.
    ``rep_search_too_large``
        an enumeration of more than ``max_candidates`` was needed.  Under
        ``strategy="readoff"`` the reason says why the read-off enumerated.

    Raises
    ------
    ValueError
        A malformed ``characters=``, ``multiplets=`` or ``spin=``, a ``spin=``
        on a spinful model or written for another model, a ``strategy=`` or
        ``orientation_branch=`` that names nothing, ``operations=`` that are not
        a subgroup holding the identity, ``generators=`` that do not generate
        the represented unitary operations, or single-valued tables for states
        that carry spin 1/2.
    ..refusal.InternalError
        An operation to be represented with no matrix of its own form, operations
        measured to hold that do not close into a group, no generating set of
        three or fewer, an internal block the measurement does not reach, a
        one-dimensional lead multiplet in a group with no antiunitary operation
        whose generator blocks are not real multiples of the tables' lift, or a
        representation with its labeling in it that misses the covariance check.

    See Also
    --------
    ..groups.realized_group : the space group a Hamiltonian obeys, this
        representation built on it.
    """
    ops = _represented_operations(model, group, assignment, validated=validated,
                                  multiplets=multiplets, characters=characters, spin=spin,
                                  generators=generators, operations=operations,
                                  include_vacuum_ops=include_vacuum_ops, strategy=strategy)
    if not isinstance(ops, _Operations):
        return ops
    mul = _multiplet_layout(model, group, assignment, ops, multiplets=multiplets,
                            characters=characters, reference_orbital=reference_orbital,
                            _labeled=_labeled)
    if not isinstance(mul, _Multiplets):
        return mul
    meas = _measurement_read(model, group, assignment, ops, mul, internal_tol=internal_tol,
                             max_candidates=max_candidates, _choices=_choices)
    if not isinstance(meas, _Measured):
        return meas
    bat = _candidate_battery(model, group, ops, mul, meas, k_pts=k_pts, n_battery=n_battery,
                             seed=seed, tol=tol, internal_tol=internal_tol,
                             product_tol=product_tol, phases=phases,
                             orientation_branch=orientation_branch)
    acc = _accepted_candidate(group, ops, mul, meas, bat, strategy=strategy,
                              max_candidates=max_candidates, internal_tol=internal_tol,
                              product_tol=product_tol, _search_record=_search_record)
    if not isinstance(acc, _Accepted):
        return acc
    return _returned_representation(model, group, assignment, ops, mul, meas, bat, acc,
                                    _returned=_returned)


def _represented_operations(model, group, assignment, *, validated, multiplets, characters,
                            spin, generators, operations, include_vacuum_ops, strategy):
    """The operations to be represented, with their generators and the word of each.

    ``model``, ``group``, ``assignment``, ``validated``, ``multiplets``,
    ``characters``, ``spin``, ``generators``, ``operations``,
    ``include_vacuum_ops`` and ``strategy`` are as in :func:`_build_rep`.
    ``validated`` is measured here where it is ``None``, and read in each
    operation's own form where it was not.  Returns the :class:`_Operations`,
    or the refusal that ends the build: ``rep_double_valued_spinless``,
    ``spin_pair_split`` or ``detector_too_large``.

    Raises
    ------
    ValueError
        A ``strategy=`` that names nothing, a malformed declaration or one
        written for another model, a ``spin=`` on a spinful model,
        single-valued tables for states that carry spin 1/2, ``operations=``
        that are not a subgroup holding the identity, or ``generators=`` that
        do not generate the represented unitary operations.
    ..refusal.InternalError
        An operation to be represented with no matrix of its own form,
        operations measured to hold that do not close into a group, or no
        generating set of three operations or fewer for the unitary ones.
    """
    if strategy not in ("readoff", "enumerate"):
        raise ValueError(f"_build_rep(): strategy={strategy!r} names no strategy. Pass "
                         "'readoff' or 'enumerate'.")
    nspin, norb, nstate = int(model.nspin), int(model.norb), int(model.nstate)

    # 0. The spin the tables assume and the model does not carry.
    # Double-valued tables are read as the orbitals carrying the spin-1/2 lift.
    # A spinless model carries it only where `spin=` declares a spinor basis;
    # without one, the build is refused here, before any matrix is built.
    spin = _resolve_spin(spin, model, call="_build_rep")
    # The declarations, read as grammar before anything is measured: a malformed
    # `characters=` or `multiplets=` raises here.
    _declaration_grammar(norb, characters=characters, multiplets=multiplets, spin=spin,
                         site_of=assignment.site_of_orbital)

    if spin is not None and not bool(group.spinor):
        raise ValueError(
            f"_build_rep(): spin= declares the {spin._kind!r} spinor basis on {spin._norb} "
            f"orbitals, and #{group.number_str} is a group of single-valued tables. Pass "
            f"the group realized_group finds with spinor=True, or leave out spin=.")

    if spin is None and nspin == 1 and bool(group.spinor):
        return Refusal(
            "rep_double_valued_spinless",
            f"the model is spinless (nspin=1, spinful=False) and #{group.number_str} "
            f"was asked for in its double-valued tables, with nothing said about where "
            f"the spin is. If the orbitals are the two spin states of each Wannier "
            f"function, as in a spinor Wannier90 import, declare the pairs with "
            f"`spin=\"interleaved\"`, `spin=\"blocked\"` or `spin={{\"pairs\": ...}}`. "
            f"Otherwise ask for the single-valued group with `spinor=False`, or build "
            f"the model with spin if the half-integer spin is the electron's own.",
            context={"nspin": nspin, "spinful": bool(getattr(model, "spinful", nspin == 2)),
                     "number_str": str(group.number_str), "spinor": bool(group.spinor),
                     "include_TR": bool(getattr(group, "_include_TR", False))})

    # A spinful model in single-valued tables is a malformed argument, and it
    # raises before anything is measured: the electron's own spin 1/2 gives its
    # states double-valued representations, which those tables do not describe.
    if nspin == 2 and not bool(group.spinor):
        raise ValueError(
            f"_build_rep(): the model is spinful (nspin=2) and #{group.number_str} is a "
            f"group of single-valued tables. Pass the group realized_group finds with "
            f"spinor=True.")

    # 0b. A declared spinor basis: the two orbitals of a declared (up, dn) pair
    # are one spinful orbital and share a center, which `realized_group` checks
    # before anything is identified, and the build checks on the sites handed in.
    if spin is not None:
        split = _spin_pair_split(spin, assignment.site_orbitals,
                                 assignment.site_positions, group._lattice)
        if split is not None:
            return split

    syms = group._symmetries
    pos_of_ind = {int(s.ind): i for i, s in enumerate(syms)}
    prod, tdiff, spf = group._get_product_table(get_diff=True)
    ident = _identity_position(group)

    if validated is None:
        validated = _measure._validate_group(model, group, assignment=assignment)
        if isinstance(validated, Refusal):
            return validated                # the detector's memory limit, as it came

    # 0c. Each operation's own form, read once.
    # The detector's record holds, for each operation, every matrix with its
    # action on k, another operation's among them, or one composed with a spin
    # rotation the Hamiltonian conserves.  Each operation is held to its own form
    # once, before anything reads a solution space; a record already read, as
    # realized_group's is, is used as it stands.
    validated = _measure._own_form_once(
        model, group, assignment, spin, validated,
        tol=(multiplets._tol if isinstance(multiplets, SiteMultiplets) else None))

    # 1. Which operations are candidates for representation.
    if operations is None:
        cand = [pos_of_ind[int(i)] for i in validated._indices]
    else:
        cand = [pos_of_ind[int(i)] for i in operations]
    if ident not in cand:
        raise ValueError("_build_rep(): operations= leaves out the identity operation. "
                         "Add it.")

    # an operation acting on a vacuum direction of a slab cell is left out
    # unless asked for: the model carries no out-of-plane orbital structure for
    # it to move, and the detector cannot refute it, its in-plane k-action
    # duplicating an in-plane partner's
    kept = [p for p in cand
            if include_vacuum_ops or not _moves_padded(syms[p], group._vacuum_dirs)]
    # an operation to be represented none of whose matrices has its own form
    # is represented by none: its matrices are another operation's, and the
    # operations handed to the build are the ones measured to hold in their
    # own form, so one without it is the package disagreeing with itself
    unformed = {int(syms[p].ind): got for p in kept
                for got in [(validated._own_form or {}).get(int(syms[p].ind))]
                if got is not None}
    if unformed:
        _raise_unformed(group, unformed, len(kept), validated.tol,
                        spinful=nspin == 2 or spin is not None)

    u_pos = [p for p in kept if not syms[p].time_reversal]
    au_pos = [p for p in kept if syms[p].time_reversal]
    if set(_closure(u_pos, prod, ident)) != set(u_pos):
        said = (f"the unitary operations {tuple(int(syms[p].ind) for p in u_pos)} to be "
                f"represented are not closed under the group's product table")
        if operations is not None:
            raise ValueError(f"_build_rep(): {said}. Pass operations= that form a subgroup.")
        raise InternalError(f"{said}. Report it, with the model and the call.")

    # 2. Generators and the word tree.
    if generators is None:
        gens = _minimal_generators(u_pos, prod, ident)
        if gens is None:
            raise InternalError(
                f"no generating set of at most three operations was found for the "
                f"represented unitary operations {tuple(int(syms[p].ind) for p in u_pos)}. "
                f"Report it, with the model and the call.")
    else:
        gens = tuple(pos_of_ind[int(i)] for i in generators)
        if set(_closure(gens, prod, ident)) != set(u_pos):
            raise ValueError(
                f"_build_rep(): generators= {tuple(int(syms[p].ind) for p in gens)} do not "
                f"generate the represented unitary operations "
                f"{tuple(int(syms[p].ind) for p in u_pos)}. Pass generators= that generate "
                f"them, or leave it out.")
    words, parent = _bfs_words(gens, prod, ident, u_pos)
    if set(words) != set(u_pos):  # pragma: no cover: guarded by the closure test
        raise InternalError(
            f"the words in the generators do not reach the operations "
            f"{tuple(int(syms[p].ind) for p in set(u_pos) - set(words))}. Report it, with "
            f"the model and the call.")

    T_pos, au_of = None, {}
    if au_pos:
        pure = [p for p in au_pos
                if np.all(np.asarray(syms[p].rotation, dtype=int) == np.eye(3, dtype=int))
                and np.allclose(np.asarray(syms[p].translation) % 1.0, 0.0, atol=1e-8)]
        T_pos = pure[0] if pure else au_pos[0]
        for p in au_pos:
            match = [q for q in u_pos if int(prod[T_pos, q]) == p]
            if not match:
                said = (f"antiunitary operation #{syms[p].ind} is not the product of "
                        f"#{syms[T_pos].ind} with a represented unitary operation")
                if operations is not None:
                    raise ValueError(f"_build_rep(): {said}. Pass operations= that form a "
                                     "subgroup.")
                raise InternalError(f"{said}. Report it, with the model and the call.")
            au_of[p] = match[0]
    return _Operations(group=group, syms=syms, kept=kept, u_pos=u_pos, au_pos=au_pos,
                       gens=gens, words=words, parent=parent, T_pos=T_pos, au_of=au_of,
                       prod=prod, spf=spf, ident=ident, nstate=nstate, spin=spin,
                       validated=validated)


def _multiplet_layout(model, group, assignment, ops, *, multiplets, characters,
                      reference_orbital, _labeled):
    """The multiplets, their orbits under the operations, and the anchor.

    ``model``, ``group``, ``assignment``, ``multiplets``, ``characters`` and
    ``reference_orbital`` are as in :func:`_build_rep`, and ``ops`` is the
    :class:`_Operations` of :func:`_represented_operations`.  ``_labeled`` says
    whether the labeling is chosen; with ``_labeled=False`` and nothing
    declared, the site tables are not read.  The declarations are read on the
    multiplets, which they never change.  Returns the :class:`_Multiplets`, or
    the refusal that ends the build: a declaration its site cannot read, or,
    where something is declared, a site table that cannot be made.
    """
    au_pos = ops.au_pos
    kept = ops.kept
    spin = ops.spin
    syms = ops.syms
    u_pos = ops.u_pos
    validated = ops.validated

    # 3. Multiplets, their permutations and their orbits.
    # The multiplets are measured, or stated by `multiplets=`; `characters=`
    # never changes them: it states what each site holds, and step 3b reads it
    # on these multiplets.
    layout = _site_multiplets(model, group, assignment, spin, validated,
                             declared=multiplets)
    su2, site_orbs = layout._spinful, layout._site_orbitals
    nchan = int(layout._nchan)
    chan_states, chan_of = layout._chan_states, layout._chan_of

    # 3b. The declared characters, parsed before anything expensive runs.
    kept_u_ind = {int(syms[p].ind) for p in u_pos}
    kept_au_ind = {int(syms[p].ind) for p in au_pos}
    site_tables, decls, char_note = {}, [], ""
    try:
        # the site tables read a declaration and name the labeling; a build whose
        # matrices are wanted only up to that labeling, with nothing declared,
        # needs neither (`_labeled=False`)
        if _labeled or characters is not None:
            site_tables = _site_tables(group, assignment, kept_u_ind, kept_au_ind,
                                       range(len(site_orbs)))
    except Exception as exc:                     # noqa: BLE001 - see below
        if characters is not None:
            gate = getattr(exc, "gate", "site_group")
            return Refusal(gate, str(exc), context={"multiplets": layout})
        # nothing was declared: the site tables only name what was read, and their
        # failure never stops a build that would otherwise succeed
        site_tables, char_note = {}, f"site character tables unavailable: {exc}"
    if characters is not None:
        try:
            decls = _normalize_declaration(characters, model, assignment, group,
                                           site_tables, layout, spin=spin)
        except _sitechars.DeclarationError as exc:
            return Refusal(exc.gate, exc.message,
                           context=dict(exc.context, multiplets=layout))
        # A declaration states what each site holds, read on the multiplets above,
        # which it never changes.
        decls = _declared_blocks(decls, layout, assignment, site_tables, spin=spin)
        for d in decls:
            d["channels"] = tuple(sorted({int(chan_of[int(o)])
                                          for o in d.get("block", d["orbitals"])}))

    cperm = _channel_perms(model, assignment, syms, kept, layout, spin)
    classes, ncls = _channel_classes(cperm, u_pos, nchan)
    # The anchor the representation is written against before any labeling is
    # chosen: orbital 0's multiplet, whichever orbital the lowest-angular-momentum
    # rule reads first (``reference_orbital``, the lead of its own coupled part
    # when the labeling is chosen).
    ref_orb = int(reference_orbital)
    ref_chan = int(chan_of[ref_orb])
    anchor = int(chan_of[0])
    anchor_cls = int(classes[anchor])
    free_cls = [c for c in range(ncls) if c != anchor_cls]
    return _Multiplets(layout=layout, su2=su2, nchan=nchan, chan_states=chan_states,
                       cperm=cperm, classes=classes, ncls=ncls, anchor=anchor,
                       anchor_cls=anchor_cls, free_cls=free_cls, site_tables=site_tables,
                       decls=decls, char_note=char_note, ref_orb=ref_orb, ref_chan=ref_chan)


def _measurement_read(model, group, assignment, ops, mul, *, internal_tol, max_candidates,
                      _choices):
    """What the measurement reads on the multiplets, made once or handed in.

    ``model``, ``group``, ``assignment``, ``internal_tol`` and ``max_candidates``
    are as in :func:`_build_rep`; ``ops`` and ``mul`` are what
    :func:`_represented_operations` and :func:`_multiplet_layout` return.
    ``_choices`` is a measurement made earlier on this group, used where it was
    made on the same unitary operations, declarations, tolerance, orbits and
    anchor; the measurement is made here otherwise.  Returns the
    :class:`_Measured`, or the refusal that ends the build:
    ``declaration_breaks_group`` where the declared site irreps are not
    covariant under every operation, ``rep_not_multiplicative`` where the copies
    of a conserved pseudospin admit no representation, or
    ``rep_search_too_large``, where the representations to weigh exceed
    ``max_candidates``.

    Raises
    ------
    ..refusal.InternalError
        An internal block of a generator the measurement does not reach.
    """
    T_pos = ops.T_pos
    au_of = ops.au_of
    au_pos = ops.au_pos
    gens = ops.gens
    ident = ops.ident
    kept = ops.kept
    parent = ops.parent
    prod = ops.prod
    spin = ops.spin
    syms = ops.syms
    u_pos = ops.u_pos
    validated = ops.validated
    words = ops.words
    anchor = mul.anchor
    chan_states = mul.chan_states
    classes = mul.classes
    decls = mul.decls
    layout = mul.layout
    nchan = mul.nchan
    ncls = mul.ncls
    ref_orb = mul.ref_orb
    site_tables = mul.site_tables

    # 4. The measurement, made once.
    # What each operation's solution space says about the orbitals: the internal
    # matrix S(g) of every multiplet (the tables' lift on a one-dimensional
    # multiplet, with the sign the model's transport gives it from site to site,
    # and the block of the model's own solution space on a wider one, carried
    # round the orbit), what the declared site irreps do to each operation, what
    # each orbit of multiplets carries relative to the anchor where the hoppings
    # pin it, and the scalar time reversal carries on each orbit.  A reading
    # `..groups.realized_group` made on this same group is handed in; otherwise
    # it is made here, and nothing below reads a solution space for it again.
    # How far outside a solution space still counts as inside it: the largest of
    # 1e-8, the declared tolerance the multiplets were measured at, and
    # `internal_tol`.  The same declared tolerance sets every threshold below
    # that judges a solution space: a vanishing block or ratio, a sign read off
    # one.
    eps = layout._tol
    zero, snap = noise_floor(ZERO, eps), noise_floor(SNAP, eps)
    space_tol = max(noise_floor(IN_SPACE, eps), float(internal_tol))
    u_inds = [int(syms[p].ind) for p in u_pos]
    verdict = None
    if _choices is not None and (set(_choices.indices) == set(u_inds)
                                 and _choices.declared == _declared_key(decls)
                                 and float(_choices.space_tol) == space_tol
                                 and np.array_equal(np.asarray(_choices.classes),
                                                    np.asarray(classes))
                                 and int(_choices.anchor) == anchor):
        verdict = _choices
    if verdict is None:
        got = _multiplet_blocks(model, group, assignment, spin, validated, layout,
                                kept, ref_chan=anchor, max_candidates=int(max_candidates))
        if isinstance(got, Refusal):
            return got
        verdict = _verdicts_and_choices(model, group, assignment, spin, validated,
                                        layout, decls, list(u_pos), classes=classes,
                                        ref_orb=ref_orb, space_tol=space_tol,
                                        tol=noise_floor(CHARACTER, eps),
                                        blocks=got, max_candidates=int(max_candidates),
                                        reversal=T_pos)
        if isinstance(verdict, Refusal):
            return verdict
    measured_of = verdict.blocks
    for record in measured_of.multiplicity:
        if not record.n_admitted:
            return _no_multiplicity_representation(group, record, layout, kept, syms)

    S_gen = {}
    for p in list(gens) + ([T_pos] if T_pos is not None else []):
        S = _internal(ops, mul, measured_of, p)
        if S is None or any(float(np.abs(M).max()) < zero for M in S):
            asked = sorted({len(chan_states[c]) for c in range(nchan)})
            raise InternalError(
                f"no internal block of size {asked} could be obtained for operation "
                f"#{syms[p].ind}: the record of the operations measured holds none that "
                f"reaches the multiplet's image under the operation. Report it, with the "
                f"model and the call.")
        S_gen[p] = S

    # What time reversal carries on each orbit of multiplets, as the measurement
    # read it: its internal factor on every multiplet, times one scalar per
    # orbit, one on orbitals that are real functions.
    reversal = np.ones(ncls, dtype=complex)
    if T_pos is not None and verdict.reversal is not None:
        reversal = np.asarray(verdict.reversal, dtype=complex)

    # The candidate characters: the one-dimensional characters of the group the
    # representation is blind to, single-valued on the orbital factor.  With
    # time reversal they are the real ones, each primed operation carrying its
    # unitary part's value; without it, every one-dimensional character of the
    # unitary group, the complex ones included.
    ind_of_pos = {p: int(syms[p].ind) for p in kept}
    admissible = [{ind_of_pos[p]: v for p, v in chi.items() if p in ind_of_pos}
                  for chi in _real_characters(kept, gens, words, parent,
                                              (au_of if T_pos is not None else {}),
                                              prod, ident)]
    if not admissible:            # pragma: no cover: the trivial character always is
        admissible = [{ind_of_pos[p]: 1.0 + 0.0j for p in kept}]
    pool = (admissible if au_pos else
            [{int(k): complex(v) for k, v in chi.items()}
             for chi in _characters_of_subgroup(group, u_inds, roots=True)] or admissible)

    unpinned = verdict.free_classes
    # What the declarations ask, as the measurement read it: one character per
    # coupled part of the crystal, each part's declarations constraining its
    # own.  Declared site irreps not covariant on all of this group break
    # operations of it.
    if decls:
        sub = _covariant_subgroup(group, u_inds, verdict.requirements,
                                  unitary_only=not au_pos,
                                  tol=noise_floor(CHARACTER, eps),
                                  joint=verdict.joint,
                                  search=verdict.search,
                                  matrices=verdict.matrices)
        if isinstance(sub, Refusal):
            return sub
        if sub.broken:
            # The declared site irreps are not covariant on all of this group, which
            # this call cannot lower: the refusal names the subgroup they are covariant
            # on, the group `realized_group` descends to, made by the same function.
            return _declaration_breaks(
                group, assignment, decls, verdict,
                standing=[int(syms[p].ind) for p in kept], subgroup=sub.subgroup,
                broken=sub.broken, readings=sub.candidates,
                not_lowered=(f"The representation was asked for on {group.name} "
                             f"#{group.number_str}, which it cannot lower."),
                site_tables=site_tables, admissible=admissible)

    # 4c. The coupled parts of the crystal, as the measurement read them.
    # The hoppings fix every multiplet's block up to one one-dimensional
    # character per coupled part, the multiplets they connect directly or
    # through others: the anchor's first, then the orbits the hoppings leave
    # free of it, grouped where they pin one another.  Inside a part the orbits
    # carry the ratios the hoppings pin between them.
    coupled_parts = _coupled_parts(classes, unpinned, verdict.parts)
    part_of_class = {int(cls): i for i, part in enumerate(coupled_parts) for cls in part}
    free_lead = {int(cls): int(min(part)) for part in coupled_parts[1:] for cls in part}
    return _Measured(verdict=verdict, measured_of=measured_of, zero=zero, snap=snap,
                     S_gen=S_gen, reversal=reversal, pool=pool, unpinned=unpinned,
                     coupled_parts=coupled_parts, part_of_class=part_of_class,
                     free_lead=free_lead)


def _candidate_battery(model, group, ops, mul, meas, *, k_pts, n_battery, seed, tol,
                       internal_tol, product_tol, phases, orientation_branch):
    """What each candidate representation is measured against, and the choices on offer.

    ``model``, ``group``, ``k_pts``, ``n_battery``, ``seed``, ``tol``,
    ``internal_tol``, ``product_tol``, ``phases`` and ``orientation_branch`` are
    as in :func:`_build_rep`; ``ops``, ``mul`` and ``meas`` are what
    :func:`_represented_operations`, :func:`_multiplet_layout` and
    :func:`_measurement_read` return.  The k-points and ``tol`` are filled
    where they are ``None``.  Returns the :class:`_Battery`.

    Raises
    ------
    ValueError
        An ``orientation_branch=`` that names nothing, a boolean among them.
    """
    T_pos = ops.T_pos
    au_of = ops.au_of
    gens = ops.gens
    ident = ops.ident
    kept = ops.kept
    nstate = ops.nstate
    parent = ops.parent
    spf = ops.spf
    syms = ops.syms
    u_pos = ops.u_pos
    words = ops.words
    anchor = mul.anchor
    chan_states = mul.chan_states
    classes = mul.classes
    free_cls = mul.free_cls
    layout = mul.layout
    nchan = mul.nchan
    coupled_parts = meas.coupled_parts
    part_of_class = meas.part_of_class

    tree = {"gens": gens, "words": words, "parent": parent, "spf": spf, "ident": ident,
            "nstate": nstate, "T_pos": T_pos, "au_of": au_of}

    # 5. The covariance battery.
    k_pts = _covariance_battery(group, k_pts, n_battery, seed)
    bloch = _bloch.evaluator(model)
    H = np.asarray(bloch(k_pts))
    if tol is None:
        # the rounding this representation's own arithmetic and the Bloch
        # phases carry, measured on the battery it is checked on, and never
        # below the exact floor
        tol = covariance_gate(len(kept), nstate, float(np.abs(H).max()),
                              _phase_rounding(bloch, group, [syms[p] for p in kept], k_pts))
    Hrot, kact = {}, {}
    for p in kept:
        act = group._k_action_of(syms[p])
        kact[p] = act
        Hrot[p] = np.asarray(bloch(act(k_pts)))

    _gate_lifts = {p: _gate_lift(ops, mul, meas, p) for p in u_pos}

    # 5b. Normalization, as a function of one candidate.
    # On each candidate that passed the covariance and internal-factor checks,
    # the generators are brought to their canonical norm, antiunitary anchor and
    # positivity, and the rest propagated along the same tree, so that the
    # product-table check runs on the matrices returned.
    #
    # The exact representation nearest the model's own blocks.  A wider
    # multiplet's block is the model's own and carries a near-symmetric model's
    # noise.  Where a candidate misses the product table by more than the
    # canonical norm's tolerance and by at most `reach`, a hundred times
    # `internal_tol` and at most 1e-2, its matrices are replaced by the nearest
    # exact representation (:func:`.._matrices._settled`).  A candidate that
    # misses the table by a sign misses it by order one, and the product-table
    # check refuses it.
    reach = min(1e-2, 100.0 * float(internal_tol))

    # 5c. The settling, one coupled part at a time.
    # The parts that hold a one-dimensional multiplet are written in the tables'
    # own lift, and are settled together, by one canonical norm.  A part of wider
    # multiplets alone carries the free phase of one member of a solution space,
    # which no other part shares, and is settled on its own, at its own lead
    # multiplet.  Each unit's generators are brought to their canonical norm and
    # anchor and the rest propagated; where the signs positivity picks do not
    # multiply, the multiplication table chooses among the roots each canonical
    # norm admits (:func:`.._matrices._table_phases`).  What is left on each part
    # is its one-dimensional character, which the labeling chooses.
    n_parts = len(coupled_parts)
    lifted = [any(layout._chan_kind[c] == "tables" for c in range(nchan)
                  if part_of_class.get(int(classes[c]), 0) == i) for i in range(n_parts)]
    held_together = ([tuple(range(n_parts))] if n_parts == 1 or all(lifted) else
                     ([tuple(i for i in range(n_parts) if lifted[i])] if any(lifted) else [])
                     + [(i,) for i in range(n_parts) if not lifted[i]])
    units = []
    for held in held_together:
        its = [c for c in range(nchan) if part_of_class.get(int(classes[c]), 0) in held]
        where = sorted(int(a) for c in its for a in chan_states[c])
        units.append(_PartUnit(states=np.asarray(where, dtype=int),
                               lead=_lead_of(its, anchor),
                               local={a: i for i, a in enumerate(where)},
                               anchor=0 in held,
                               orbitals=tuple(sorted(int(o) for c in its
                                                     for o in layout._chan_orbs[c]))))

    # 6. The search.
    phases = tuple(complex(z) for z in phases)
    n_free = len(gens) * len(free_cls)
    if orientation_branch is None:
        orients = (1, -1)
    elif isinstance(orientation_branch, bool):
        # a boolean is not a branch: True would read as +1 through int(), silently
        raise ValueError(f"_build_rep(): orientation_branch={orientation_branch!r} is a "
                         "boolean. Pass None, +1 or -1.")
    elif int(orientation_branch) == 1:
        orients = (1,)
    elif int(orientation_branch) == -1:
        orients = (-1,)
    else:
        raise ValueError(f"_build_rep(): orientation_branch={orientation_branch!r} names "
                         "no branch. Pass None, +1 or -1.")

    # Where the choice applies: every block of the operation at once.  A
    # relative character the Hamiltonian leaves open is assembled here as the
    # trivial one (`_relative`), the labeling choosing its part's own in step 7b,
    # and that is one statement about the whole matrix, not one per site.  Two
    # sites the Hamiltonian couples to each other not at all are as free as two
    # multiplets of one site, and `RealizedGroup.conserved` is where the
    # decoupling is said.
    all_blocks = [list(range(nchan))]

    # 6b'. The orientation branch, pinned by convention when the checks tie.
    # The internal-factor check tells the two branches apart wherever
    # `S(g)^dag` is not a multiple of `S(g)`, the factor being free on each
    # block.  On a group of involutions alone `S(g)^2` is a multiple of the
    # identity, so `S(g)^dag` is a multiple of `S(g)`, and all three checks are
    # blind to the branch.  Where the two branches' candidates both pass all
    # three checks and are one representation up to a one-dimensional
    # character on each coupled part (`_one_labeling_apart`, on the
    # multiplets of each part), the branch `+1`, the intertwiner that
    # carries `k -> gk`, is returned with `branch_tie` set (`_pin_branch`).  Any
    # other set of more than one survivor is refused `rep_not_unique`.  The tie
    # is settled on the candidate set, before the strategy dispatch, so the
    # read-off's two candidates are accepted where it resolves.
    part_chans = [[int(c) for c in range(nchan) if int(classes[c]) in part]
                  for part in coupled_parts]
    return _Battery(tree=tree, k_pts=k_pts, bloch=bloch, H=H, Hrot=Hrot, kact=kact, tol=tol,
                    internal_tol=internal_tol, product_tol=product_tol,
                    _gate_lifts=_gate_lifts, reach=reach, units=units, phases=phases,
                    n_free=n_free, orients=orients, all_blocks=all_blocks,
                    part_chans=part_chans)


def _accepted_candidate(group, ops, mul, meas, bat, *, strategy, max_candidates,
                        internal_tol, product_tol, _search_record):
    """The candidate representation the three checks accept.

    ``group``, ``strategy``, ``max_candidates``, ``internal_tol`` and
    ``product_tol`` are as in :func:`_build_rep`; ``ops``, ``mul``, ``meas`` and
    ``bat`` are what :func:`_represented_operations`, :func:`_multiplet_layout`,
    :func:`_measurement_read` and :func:`_candidate_battery` return.
    ``_search_record``, where given, receives the record of the search under
    its ``"search"`` key, beside a refusal as well.  Under
    ``strategy="readoff"`` the read-off runs first, and the enumeration runs
    where it reads nothing or what it read does not give one representation,
    unless none of what it read passes the covariance check and the best lies
    between that check's tolerance and a larger declared tolerance, where the
    call refuses ``rep_not_covariant``.  Returns the :class:`_Accepted`, or the refusal that
    ends the build: ``rep_search_too_large``, ``rep_not_covariant``,
    ``rep_internal_factor``, ``canonical_norm``, ``antiunitary_anchor``,
    ``rep_not_multiplicative`` or ``rep_not_unique``.
    """
    gens = ops.gens
    kept = ops.kept
    prod = ops.prod
    spf = ops.spf
    spin = ops.spin
    syms = ops.syms
    u_pos = ops.u_pos
    validated = ops.validated
    layout = mul.layout
    ncls = mul.ncls
    k_pts = bat.k_pts
    n_free = bat.n_free
    orients = bat.orients
    phases = bat.phases
    tol = bat.tol

    # 6c. The read-off, and the enumeration behind it.
    n_enumeration = int(len(phases) ** n_free * len(orients))
    read = _ReadOff(choices=(), branches={}, reason=None,
                    record={"n_enumeration": n_enumeration, "branches": {}, "reason": None,
                            "notes": []},
                    read={})
    if strategy == "readoff":
        try:
            read = _read_off(ops, mul, meas, bat)
        except np.linalg.LinAlgError as exc:  # pragma: no cover: not seen on any fixture
            # The read-off is a way to find the candidate faster; it must never
            # be the reason a build refuses.  An SVD that will not converge is a
            # reason to enumerate, not an error to return.
            read = replace(read, reason=(f"the projection onto the tensor structure did "
                                         f"not converge ({exc})"))
    read.record["n_enumeration"] = n_enumeration
    outcome, strategy_used, fallback, pin = None, None, None, None
    declared_tol = float(getattr(validated, "tol", tol))
    if strategy == "readoff" and read.choices:
        outcome = _gate(ops, mul, bat, _build(ops, mul, meas, bat, read.choices))
        pin = _pin_branch(ops, mul, bat, outcome.survivors)
        best = float(outcome.search["best"])
        if len(pin.survivors) == 1:
            strategy_used = "readoff"
        elif not outcome.covariant and declared_tol > tol and tol < best <= declared_tol:
            # What was read is near-symmetric: the representation assembled from what
            # the solution spaces measured passes the covariance check at the declared
            # tolerance and misses it only at the exact one.  No other choice of signs,
            # which the measurement never pinned, stands in for it, so nothing is
            # enumerated and the refusal below says so.
            strategy_used = "readoff"
        else:
            # The read-off finds the candidate and never accepts one: a read the checks
            # do not confirm is handed to the enumeration of the phases on offer, whose
            # refusal or survivor is the one returned, and what was read is said with it.
            said = "; ".join(
                f"branch {orient:+d}: " + (", ".join(
                    f"generator #{ind} reads {_factor_said(z)} on orbit {c}"
                    for (ind, c), z in sorted(got.items())) or "no orbit read")
                for orient, got in sorted(read.read.items(), reverse=True))
            fallback = (f"the factors measured between orbits of multiplets ({said}) "
                        f"do not give one representation: {len(outcome.survivors)} of "
                        f"the {outcome.search['n_distinct']} distinct candidates built "
                        "from them carry H(k) onto itself, factorize on every site into "
                        "(character) x (the group's own internal factor) and reproduce the "
                        "multiplication table (the candidates each test removed: "
                        f"{outcome.search['killed_by']})")
            outcome, pin = None, None
    elif strategy == "readoff":
        fallback = read.reason

    if outcome is None:
        if n_enumeration > max_candidates:
            try:
                measured = float(n_enumeration)
            except OverflowError:  # pragma: no cover: past 2**1024 choices
                measured = float("inf")
            hint = ("" if fallback is None else
                    f" The choices were not read off the measurement because {fallback}.")
            return Refusal(
                "rep_search_too_large",
                f"the enumeration of the representation's discrete choices would run "
                f"{n_enumeration:,} candidates ({len(phases)} phases for each of "
                f"{len(gens)} generator(s) on each of {ncls - 1} free orbit(s) of "
                f"multiplets out of {ncls}, on {len(orients)} orientation branch(es)) "
                f"against max_candidates = {max_candidates:,}.{hint} State a site of "
                f"several orbitals as one multiplet with `site_multiplets=`, or pass a "
                f"larger `max_candidates=` if the enumeration is worth waiting for.",
                value=measured, threshold=float(max_candidates),
                context={"n_generators": len(gens), "n_classes": int(ncls),
                         "n_free": int(n_free), "n_candidates": n_enumeration,
                         "max_candidates": int(max_candidates),
                         "multiplets": layout, "fallback_reason": fallback})
        outcome = _gate(ops, mul, bat,
                        _build(ops, mul, meas, bat,
                               ((orient, combo) for orient in orients
                                for combo in itertools.product(phases, repeat=n_free))))
        strategy_used = "enumerate"

    records, distinct = outcome.records, outcome.distinct
    covariant, consistent = outcome.covariant, outcome.consistent
    survivors = outcome.survivors
    spectrum = outcome.search["residuals"]
    int_spectrum = outcome.search["internal_residuals"]
    prod_spectrum = outcome.search["product_residuals"]

    # the branch tie of step 6b', measured on the candidate set that was accepted
    if pin is None:
        pin = _pin_branch(ops, mul, bat, survivors)
    survivors = pin.survivors
    branch_tie, branch_note = pin.tie, pin.note

    search_record = dict(outcome.search)
    # The read-off numbers at the head of the record describe the read that
    # produced the accepted representation, the surviving branch.  The other
    # branch's numbers stay in `search["readoff"]["branches"]`.
    accepted = {}
    if strategy_used == "readoff" and len(survivors) == 1:
        won = int(survivors[0][0]["orientation"])
        accepted = {won: read.branches[won]} if won in read.branches else {}
    search_record.update({
        "strategy": strategy_used,
        "requested": str(strategy),
        "fallback_reason": fallback,
        "n_enumeration": n_enumeration,
        "max_candidates": int(max_candidates),
        "branch_tie": bool(branch_tie),
        "branch_note": branch_note,
        "ansatz_defect": max((float(g["ansatz_defect"])
                              for b in accepted.values()
                              for g in b["generators"].values()), default=float("nan")),
        "ratios": {k: v for b in accepted.values() for k, v in b["ratios"].items()},
        "defects": {k: v for b in accepted.values() for k, v in b["defects"].items()},
        "readoff": read.record})
    search = search_record
    if _search_record is not None:
        # the record of the search, handed back where asked beside a refusal,
        # which carries only the number of candidates tried
        _search_record["search"] = search
    # A declared spinor basis is named beside a refusal of the checks below.
    spin_hint = ("" if spin is None else
                 f" `spin=` declares the {spin._kind!r} spinor basis on {spin._norb} "
                 f"orbitals.")
    # What the read-off could not use, said beside the enumeration's verdict:
    # what it read and why that was not a representation, or why it read nothing.
    n_ph = len(phases)
    read_hint = ("" if fallback is None else
                 f" The enumeration ran because {fallback}. The candidates counted above "
                 f"try {n_ph} phase{'' if n_ph == 1 else 's'} on each free orbit.")

    if not covariant:
        # Where the tolerance declared on the detector's record lies above this
        # check's and the best candidate lands between the two, the operations are
        # near-symmetric, obeyed within the declared tolerance and not to machine
        # precision, and the refusal says so.  Neither tolerance moves.
        near_hint = ""
        declared = float(getattr(validated, "tol", tol))
        if declared > tol and spectrum[0] <= declared:
            near_hint = (
                f" The operations were kept at the declared tolerance {declared:.3g}, and "
                f"the best candidate's {spectrum[0]:.3g} lies between it and the "
                f"{tol:.3g} the representation is held to: these operations are "
                f"near-symmetric, obeyed within the declared tolerance and not exactly.")
        # how far the best candidate carries H(k) under each operation, so that
        # an operation it misses can be read beside what it is held at alone
        nearest = min(distinct, key=lambda grp: float(grp[0]["residual"]))[0]["matrices"]
        by_operation = {int(ops.syms[p].ind): _covariance_of(ops, bat, nearest, p)
                        for p in ops.kept}
        return Refusal(
            "rep_not_covariant",
            f"none of the {len(records)} candidate representations ({len(distinct)} distinct) "
            f"carries H(k) onto H(Rk) within the tolerance: the best reaches "
            f"{spectrum[0]:.3g} against {tol:.3g}.{read_hint}{near_hint}{spin_hint} Where "
            f"`site_multiplets=` states the multiplets, state each site whose orbitals mix "
            f"under its site-symmetry group as one multiplet.",
            value=float(spectrum[0]), threshold=float(tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept),
                     "covariance_by_operation": by_operation,
                     **({} if spin is None else {"spin_kind": str(spin._kind)})})
    if not consistent:
        return Refusal(
            "rep_internal_factor",
            f"{len(covariant)} of the {len(distinct)} distinct candidates carry H(k) onto "
            f"H(Rk) within {tol:.3g}, but on none of them does a site block factorize into "
            f"(character) x (the group's own internal factor): the best defect is "
            f"{int_spectrum[0]:.3g} against {internal_tol:.3g}.{read_hint}{spin_hint} Where "
            f"`site_multiplets=` states the multiplets, check the statement.",
            value=float(int_spectrum[0]), threshold=float(internal_tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept),
                     **({} if spin is None else {"spin_kind": str(spin._kind)})})
    if not survivors:
        stuck = [grp[0]["refusal"] for grp in consistent if "refusal" in grp[0]]
        if len(stuck) == len(consistent):
            # every surviving candidate failed to phase-canonicalize at all;
            # normalize_rep's own diagnosis is sharper than anything said here.
            return replace(stuck[0], reason=stuck[0].reason + read_hint) if read_hint \
                else stuck[0]
        # Why no phase helps, where the matrices say so.  The candidate that came
        # closest carries, where nothing anchored its phases, whether its matrices
        # form a projective representation at all and whether any roots the
        # canonical norms admit made it multiply.  Projective, and none did: two
        # commuting operations whose matrices commute only up to a phase the table
        # does not carry are the statement of it that no gauge changes.
        closest = min((grp[0] for grp in consistent if "normalized" in grp[0]),
                      key=lambda r: float(r["product"]))
        found = closest.get("roots")
        pair = None
        if found is not None and found["projective"] and found["roots"] is None:
            # on the part whose roots did not multiply, where the parts were
            # settled one by one
            on = closest.get("roots_matrices", closest["normalized"][0])
            pair = _commuting_pair_phase(on, u_pos, syms, prod, spf)
        if pair is not None:
            a, b, c = pair

            name_a = _record_name(syms[a], group._lattice)
            name_b = _record_name(syms[b], group._lattice)
            Sa = np.asarray(syms[a].spinor_rotation, dtype=complex)
            Sb = np.asarray(syms[b].spinor_rotation, dtype=complex)
            half, _res = _proportionality(Sa @ Sb,
                                          int(spf[a, b]) * int(spf[b, a]) * (Sb @ Sa))
            half = complex(half)
            spin_half = (not bool(group.spinor)) and abs(half - c) <= 1e-6
            asked = int(spf[a, b]) * int(spf[b, a])
            named = f"{group.name} #{group.number_str}"
            return Refusal(
                "rep_not_multiplicative",
                f"{len(consistent)} of the {len(distinct)} distinct candidates carry H(k) onto "
                f"H(Rk) within the tolerance and factorize on every site into (character) x "
                f"(the group's own internal factor), and their matrices form a projective "
                f"representation of {named} of a class the multiplication table does not "
                f"carry: the best product-table defect is {prod_spectrum[0]:.3g} against "
                f"{product_tol:.3g}. Operations #{syms[a].ind} ({name_a}) and "
                f"#{syms[b].ind} ({name_b}) commute, the table asks "
                f"D(a)D(b) = {asked:+d} D(b)D(a)"
                + (" (spinor factors included)" if bool(group.spinor) else "")
                + f", and on these states D(a)D(b) = {_fmt_char(asked * c)} D(b)D(a). No "
                f"representation of {named} with its own factor system exists on these states."
                + (f" It is the sign the spin-1/2 lifts of these two operations carry: the "
                   f"states carry half-integer spin. Where the orbital index holds the spin, "
                   f"as in a spinor Wannier90 import, declare the spin partners with `spin=`, "
                   f"and the states are read in the double group." if spin_half else "")
                + read_hint,
                value=float(prod_spectrum[0]), threshold=float(product_tol),
                context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                         "indices": tuple(int(syms[p].ind) for p in kept),
                         "commuting_operations": (int(syms[a].ind), int(syms[b].ind)),
                         "commutation_phase": c,
                         "spin_half": bool(spin_half)})
        return Refusal(
            "rep_not_multiplicative",
            f"{len(consistent)} of the {len(distinct)} distinct candidates carry H(k) onto "
            f"H(Rk) within the tolerance and factorize on every site into (character) x "
            f"(the group's own internal factor), but none of them reproduces the group's "
            f"multiplication table (spinor factors included): the best product-table defect is "
            f"{prod_spectrum[0]:.3g} against {product_tol:.3g}.{read_hint}",
            value=float(prod_spectrum[0]), threshold=float(product_tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept)})
    if len(survivors) > 1:
        branches = sorted({int(g[0]["orientation"]) for g in survivors})
        orient_hint = (
            "" if branches != [-1, 1] or len(survivors) != 2 else
            " The two are the two orientation branches, and they give different site "
            "labels under the same labeling.")
        return Refusal(
            "rep_not_unique",
            f"{len(survivors)} distinct representations of {len(records)} candidates carry "
            f"H(k) onto H(Rk) within {tol:.3g} on {k_pts.shape[0]} k-points, factorize on "
            f"every site into (character) x (the group's own internal factor) within "
            f"{internal_tol:.3g} and reproduce the multiplication table within "
            f"{product_tol:.3g}."
            f"{orient_hint}{read_hint} Report it, with the model and the call.",
            value=float(len(survivors)), threshold=1.0,
            context={"n_candidates": int(search["n_candidates"])})
    winner = survivors[0][0]
    search["internal"] = float(winner["internal"])
    search["product"] = float(winner["product"])
    search["composition"] = float(winner.get("composition", 0.0))
    return _Accepted(winner=winner, search=search, branch_tie=branch_tie)


def _returned_representation(model, group, assignment, ops, mul, meas, bat, acc, *,
                             _returned):
    """The representation returned, its labeling chosen and what it claims measured on it.

    ``model``, ``group`` and ``assignment`` are as in :func:`_build_rep`;
    ``ops``, ``mul``, ``meas``, ``bat`` and ``acc`` are what
    :func:`_represented_operations`, :func:`_multiplet_layout`,
    :func:`_measurement_read`, :func:`_candidate_battery` and
    :func:`_accepted_candidate` return.  ``_returned`` says whether this is the
    representation returned; only then are the Bloch operators' factor system
    and each orbit's band representation measured.  The labeling is chosen
    where the site tables were read.  Returns the
    :class:`..symmetry.SpaceGroupRep`.

    Raises
    ------
    ..refusal.InternalError
        The representation with its labeling in it misses the covariance check
        its candidates passed.
    """
    T_pos = ops.T_pos
    au_pos = ops.au_pos
    gens = ops.gens
    ident = ops.ident
    kept = ops.kept
    nstate = ops.nstate
    prod = ops.prod
    spf = ops.spf
    spin = ops.spin
    syms = ops.syms
    u_pos = ops.u_pos
    validated = ops.validated
    words = ops.words
    char_note = mul.char_note
    classes = mul.classes
    cperm = mul.cperm
    decls = mul.decls
    free_cls = mul.free_cls
    layout = mul.layout
    ref_chan = mul.ref_chan
    ref_orb = mul.ref_orb
    site_tables = mul.site_tables
    coupled_parts = meas.coupled_parts
    measured_of = meas.measured_of
    pool = meas.pool
    verdict = meas.verdict
    H = bat.H
    Hrot = bat.Hrot
    bloch = bat.bloch
    k_pts = bat.k_pts
    kact = bat.kact
    tol = bat.tol
    branch_tie = acc.branch_tie
    search = acc.search
    winner = acc.winner

    # 7. The winner's normalization, already measured by the checks.
    D, normalization, anchor_T, DT_n = winner["normalized"]

    # 7b. The labeling, chosen once, and put in.
    # Everything above is blind to D(g) -> chi(g) D(g) on a coupled part, chi a
    # one-dimensional character, and to the representation a conserved
    # pseudospin's copies carry.  The multiplets' characters are read once on the
    # representation as settled (:func:`_multiplet_characters`); one chooser
    # takes each part's candidates, what its declarations leave, to one labeling
    # (:func:`_labeling`); and it is put in here (:func:`_assembled`), so that
    # the matrices returned are the representation induced from the site irreps
    # returned, in the tables' convention, spin up along +z in a right-handed
    # frame.  The atomic-limit table is then written off what was read and
    # chosen (:func:`_atomic_limit_table`), nothing measured again.
    limits = None
    if site_tables:
        raw, worst = _multiplet_characters(D, syms, u_pos, cperm, layout, prod, spf)
        measured_at = {}
        for r in verdict.rows:
            measured_at.setdefault(tuple(int(o) for o in r.orbitals), set()).add(
                int(r.ind))
        for j in tuple(verdict.joint) + tuple(verdict.by_sectors):
            measured_at.setdefault(tuple(int(o) for o in j.decl["orbitals"]), set()).update(
                int(g) for row in j.theta.values() for g in row)
        choice = _labeling(D, raw, decls, pool=pool, model=model, group=group,
                            assignment=assignment, layout=layout,
                            syms=syms, u_pos=u_pos, cperm=cperm, classes=classes,
                            parts=coupled_parts, ref_chan=ref_chan,
                            site_tables=site_tables, measured_at=measured_at,
                            multiplicity=measured_of.multiplicity)
        D = _assembled(D, choice, syms, layout)
        limits = _atomic_limit_table(choice.raw, worst, decls, choice, group=group,
                                     assignment=assignment, model=model, layout=layout,
                                     syms=syms, kept=kept, checks=verdict.rows,
                                     pool=pool, site_tables=site_tables, ident=ident,
                                     spin=spin)
        # A conserved pseudospin's choice, on the record: the site irreps chosen for
        # its copies, and every choice the group's table admits beside it, in the
        # order the key ranks them.
        if choice.spaces:
            limits = _with_multiplicity_spaces(limits, choice.spaces, group, assignment)

    # 8. Everything the representation claims, measured again.
    residuals, norm_res, anchor_res = {}, {}, {}
    for p in kept:
        ind = int(syms[p].ind)
        U = D[p]
        src = H.conj() if syms[p].time_reversal else H
        residuals[ind] = float(np.abs(U @ src @ U.conj().T - Hrot[p]).max())
        n, zeta = _order_and_zeta(group, p, prod=prod, spf=spf)
        norm_res[ind] = float(np.abs(_power(U, bool(syms[p].time_reversal), n)
                                     - zeta * np.eye(nstate)).max())
        if anchor_T is not None and not syms[p].time_reversal:
            # the time reversal returned, the labeling's own included
            Tn = D[T_pos] if T_pos in D else (DT_n if DT_n is not None else anchor_T)
            anchor_res[ind] = float(np.abs(Tn @ np.conj(U) @ Tn.conj().T - U).max())
    # The representation returned passes the covariance check its candidates
    # passed: every labeling it is chosen among is one the hoppings leave open, a
    # symmetry of H, so one that fails it is the package disagreeing with itself.
    worst_ind = max(residuals, key=residuals.get) if residuals else None
    if worst_ind is not None and residuals[worst_ind] > tol:
        raise InternalError(
            f"the representation of {group.name} #{group.number_str} with its labeling "
            f"does not carry H(k) onto H(Rk) within the tolerance its candidates met: "
            f"{residuals[worst_ind]:.3g} against {tol:.3g}, on operation {worst_ind}. "
            "Report it, with the model and the call.")

    product_residual = _product_table_residual(D, kept, syms, prod, spf)

    # The Bloch operators' factor system and each orbit's band representation
    # are what the representation returned carries, measured once its matrices
    # are final.  A representation built as the operators of a projection is
    # never returned, and carries neither (`_returned=False`).
    factor_residual = float("nan")
    try:
        if _returned:
            factor_residual = 0.0
            per = list(group.periodic_dirs)
            U_k = {p: bloch_operator(model, group, assignment, syms[p], D[p], k_pts)
                   for p in kept}
            k_img = {p: kact[p](k_pts) for p in kept}
            for a in kept:
                D_a = np.asarray(D[a], dtype=complex)
                for b in kept:
                    c = int(prod[a, b])
                    if c not in D:
                        continue
                    kb = k_img[b]
                    # the operator of `a` at `kb`, its image not folded back:
                    # the translation's phase times the matrix
                    Ua =_translation_phase(group, syms[a], kb)[:, None, None] * D_a[None, :, :]
                    Ub = U_k[b]
                    Uc = U_k[c]
                    lhs = Ua @ (np.conj(Ub) if syms[a].time_reversal else Ub)
                    delta = (np.asarray(syms[a].translation)
                             + np.asarray(syms[a].rotation, dtype=float)
                             @ np.asarray(syms[b].translation)
                             - np.asarray(syms[c].translation))
                    omega = int(spf[a, b]) * np.exp(-2j * np.pi
                                                    * (kact[c](k_pts) @ delta[per]))
                    factor_residual = max(factor_residual, float(
                        np.abs(lhs - omega[:, None, None] * Uc).max()))
    except Exception:  # pragma: no cover: the check must never mask what was built
        factor_residual = float("nan")

    indices = tuple(int(syms[p].ind) for p in sorted(kept))
    choice = {"orientation": int(winner["orientation"]),
              "signs": {(int(syms[gens[gi]].ind), int(c)):
                        complex(winner["combo"][gi * len(free_cls) + j])
                        for gi in range(len(gens)) for j, c in enumerate(free_cls)}}
    built = SpaceGroupRep(
        _model=model, _group=group, _assignment=assignment,
        _indices=indices,
        _operations=tuple(syms[p] for p in sorted(kept)),
        matrices={int(syms[p].ind): D[p] for p in kept},
        antiunitary=tuple(int(syms[p].ind) for p in sorted(au_pos)),
        _norm_residuals=norm_res, _anchor_residuals=anchor_res,
        numerical=SpaceGroupRepNumerical(
            residuals=residuals, product_residual=float(product_residual),
            factor_residual=float(factor_residual)),
        _generators=tuple(int(syms[p].ind) for p in gens),
        _words={int(syms[p].ind): w for p, w in words.items()},
        _choice=choice, _search=search, _classes=classes,
        _reference=dict({"orbital": ref_orb, "multiplet": ref_chan,
                         "orbit": int(classes[ref_chan])},
                        **({} if spin is None else
                           {"spin_pair": int(spin._pair_of_orbital[ref_orb])})),
        _failed=tuple(validated.discrepancies),
        _battery=k_pts, _multiplets=layout, _normalization=normalization,
        _time_reversal=None if T_pos is None else int(syms[T_pos].ind),
        _characters=limits, _spin=spin, _branch_tie=bool(branch_tie),
        _model_key=bloch.key)
    return _with_tables_unavailable(_with_band_representations(built) if _returned
                                    else built)


def _internal(ops, mul, measured_of, p):
    """The internal block ``S(g)`` of the operation at position ``p``, on each multiplet.

    ``ops`` and ``mul`` are the build's :class:`_Operations` and
    :class:`_Multiplets`, and ``p`` is the operation's position in ``ops.syms``.
    ``measured_of`` holds each operation's measured block on each multiplet, by
    the operation's index and the multiplet's.  A one-dimensional multiplet with
    no block measured takes the operation's own lift
    (:func:`.._matrices._lift_of`).  Returns one block per multiplet, or ``None``
    where a wider multiplet has no block, or a block's size is not its
    multiplet's number of states.
    """
    syms = ops.syms
    chan_states = mul.chan_states
    layout = mul.layout
    nchan = mul.nchan
    su2 = mul.su2

    lift = _lift_of(syms[p], su2)
    blocks = measured_of.get(int(syms[p].ind), {})
    out = []
    for c in range(nchan):
        S = blocks.get(int(c))
        if S is None and layout._chan_kind[c] != "measured":
            out.append(lift)
            continue
        if S is None or S.shape != (len(chan_states[c]),) * 2:
            return None
        out.append(S)
    return out


def _relative(ops, meas, p_gen, cls):
    """The factor on an orbit of multiplets that the build assembles a generator with.

    ``ops`` and ``meas`` are the build's :class:`_Operations` and
    :class:`_Measured`; ``p_gen`` is the generator's position in ``ops.syms``
    and ``cls`` the orbit.  Returns the factor the measurement pinned on the
    orbit relative to the anchor's orbit, or ``None`` where it pinned none.  On
    an orbit the hoppings leave free of the anchor it returns ``1``, the trivial
    relative character; the labeling chooses that part's own character later.
    """
    syms = ops.syms
    unpinned = meas.unpinned
    verdict = meas.verdict

    if int(cls) in unpinned:
        return 1.0 + 0.0j
    got = verdict.chosen.get((int(syms[p_gen].ind), int(cls)))
    return None if got is None else complex(got)


def _assemble(ops, mul, p, S, sign_of_class):
    nstate = ops.nstate
    chan_states = mul.chan_states
    classes = mul.classes
    cperm = mul.cperm
    nchan = mul.nchan

    D = np.zeros((nstate, nstate), dtype=complex)
    for c in range(nchan):
        rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
        cols = np.asarray(chan_states[c], dtype=int)
        D[np.ix_(rows, cols)] = sign_of_class[int(classes[c])] * S[c]
    return D


def _propagate(bat, Dgen, DT):
    """Every represented operation's matrix, propagated from the generators'.

    ``bat`` is the build's :class:`_Battery`, whose ``tree`` holds the words.
    ``Dgen`` holds the generators' matrices, in the order of the build's
    generators, and ``DT`` the matrix of its antiunitary operation, or ``None``
    where it has none.  Returns each operation's matrix by its position
    (:func:`.._matrices._propagated`).
    """
    tree = bat.tree

    return _propagated(Dgen, DT, **tree)


def _covariance(ops, bat, D):
    return max((_covariance_of(ops, bat, D, p) for p in ops.kept), default=0.0)


def _covariance_of(ops, bat, D, p):
    """The covariance residual of the operation at position ``p``, over the battery.

    ``ops`` and ``bat`` are the build's :class:`_Operations` and
    :class:`_Battery`, and ``D`` maps each operation's position to its matrix.
    Returns ``max_k |D(g) H(k) D(g)† − H(Rk)|`` for ``g = {R|t}``, the largest
    modulus of an entry over the battery's k-points, with ``H(k)*`` in place
    of ``H(k)`` and ``H(−Rk)`` in place of ``H(Rk)`` for ``g`` combined with
    time reversal, in the units of ``H``.
    """
    U = D[p]
    src = bat.H.conj() if ops.syms[p].time_reversal else bat.H
    return float(np.abs(U @ src @ U.conj().T - bat.Hrot[p]).max())


def _gate_lift(ops, mul, meas, p):
    r"""The blocks the internal-factor check holds the operation at position ``p`` to.

    ``ops``, ``mul`` and ``meas`` are the build's :class:`_Operations`,
    :class:`_Multiplets` and :class:`_Measured`, and ``p`` is the operation's
    position in ``ops.syms``.  A multiplet's block is the one
    :func:`.._multiplets._multiplet_blocks` measured, from which the matrices
    are assembled.  A one-dimensional multiplet with no block measured takes the
    operation's own lift (:func:`.._matrices._lift_of`).  Returns one block per
    multiplet, ``None`` for a wider multiplet with no block measured, which the
    check skips.
    """
    syms = ops.syms
    layout = mul.layout
    nchan = mul.nchan
    su2 = mul.su2
    measured_of = meas.measured_of

    lift = _lift_of(syms[p], su2)
    blocks = measured_of.get(int(syms[p].ind), {})
    return [blocks.get(int(c),
                       None if layout._chan_kind[c] == "measured" else lift)
            for c in range(nchan)]


def _internal_defect(ops, mul, bat, D):
    r"""The internal-factor defect of a candidate: the largest ``‖B(g) − χ S(g)‖``.

    ``ops``, ``mul`` and ``bat`` are the build's :class:`_Operations`,
    :class:`_Multiplets` and :class:`_Battery`, and ``D`` maps each
    operation's position to the candidate's matrix.  For every unitary
    operation ``g`` and every multiplet, ``B(g)`` is the block of ``D(g)``
    carrying the multiplet to its image, ``S(g)`` the block :func:`_gate_lift`
    gives, and ``χ`` is fitted to each block.  A multiplet with no block
    measured is skipped.  Returns the largest modulus of an entry of
    ``B(g) − χ S(g)``.  The defect is blind to the one-dimensional character
    the labeling puts in later, and unchanged under ``D → λD``.  On a
    one-dimensional multiplet of a spinless model with no declared spinor
    basis ``S(g)`` is ``[[1]]``, and the defect there vanishes by arithmetic.
    """
    u_pos = ops.u_pos
    chan_states = mul.chan_states
    cperm = mul.cperm
    nchan = mul.nchan
    _gate_lifts = bat._gate_lifts

    worst = 0.0
    for p in u_pos:
        lifts = _gate_lifts.get(p) or []
        for c in range(nchan):
            S = lifts[c] if c < len(lifts) else None
            if S is None:
                continue
            rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
            cols = np.asarray(chan_states[c], dtype=int)
            _, res = _proportionality(D[p][np.ix_(rows, cols)], S)
            worst = max(worst, float(res))
    return worst


def _part_anchor(ops, mul, unit):
    """The anchor of one part's operations, at the part's own lead multiplet.

    ``ops`` and ``mul`` are the build's :class:`_Operations` and
    :class:`_Multiplets`, and ``unit`` is the :class:`_PartUnit` settled.
    Returns a function of an operation's position ``p`` and the antiunitary
    operation's matrix ``anchor_T`` (``None`` without one), which gives the
    operation's :class:`.._matrices.RepAnchor` on the unit's states.  Its
    reference block carries the lead multiplet to its image.  Where there is no
    ``anchor_T`` and the lead is a one-dimensional multiplet, the anchor also
    holds the operation's spin rotation in the tables, or ``[[1]]`` without
    spin.
    """
    syms = ops.syms
    chan_states = mul.chan_states
    cperm = mul.cperm
    layout = mul.layout
    su2 = mul.su2

    lead, local = int(unit.lead), unit.local

    def _anchor(p, anchor_T):
        table = None
        if anchor_T is None and layout._chan_kind[lead] == "tables":
            table = (np.asarray(syms[p].spinor_rotation, dtype=complex) if su2
                     else np.eye(1, dtype=complex))
        return RepAnchor(antiunitary=anchor_T,
                         rows=tuple(local[int(a)] for a in chan_states[int(cperm[p][lead])]),
                         cols=tuple(local[int(a)] for a in chan_states[lead]),
                         internal=table)
    return _anchor


def _part_unsettled(ops, bat, bad, unit):
    """The refusal of a unit whose matrices form no representation.

    ``ops`` and ``bat`` are the build's :class:`_Operations` and
    :class:`_Battery`.  ``unit`` is the :class:`_PartUnit` settled, one or
    more coupled parts, and ``bad`` the refusal its settling returned.  Returns
    ``bad`` itself where the crystal is settled as one unit.  Otherwise returns
    a refusal of the same check, value and threshold, whose reason names the
    unit's orbitals and the group before ``bad``'s own, and whose context adds
    the unit's orbitals as ``part``.
    """
    group = ops.group
    units = bat.units

    if len(units) == 1:
        return bad
    return Refusal(
        bad.gate,
        f"the matrices built for orbitals {unit.orbitals}, apart from the rest of the "
        f"crystal, form no representation of {group.name} #{group.number_str}: "
        f"{bad.reason}",
        value=bad.value, threshold=bad.threshold,
        context=dict(bad.context or {}, part=unit.orbitals))


def _settled_by_part(ops, mul, bat, Dsearch):
    """A candidate's matrices, settled one unit of coupled parts at a time.

    ``ops``, ``mul`` and ``bat`` are the build's :class:`_Operations`,
    :class:`_Multiplets` and :class:`_Battery`, and ``Dsearch`` maps each
    represented operation's position to the candidate's matrix.  Each unit of
    ``bat.units`` is settled on its own states by
    :func:`.._matrices._settle_unit`, anchored at its own lead multiplet
    (:func:`_part_anchor`).  Returns ``(refusal, D, normalization, anchor_T,
    DT_n, defect, roots)``.  ``refusal`` is ``None``, or the refusal of the
    first unit that settles into no representation (:func:`_part_unsettled`),
    with ``None`` for every other entry but ``defect``.  ``D`` holds the
    settled matrices by position.  ``normalization`` is the anchor's unit's
    record for each operation it normalized, by index, written on the whole
    matrices.  ``anchor_T`` is the antiunitary operation's matrix the anchors
    were read with and ``DT_n`` that matrix in its canonical phase, both
    ``None`` without one.  ``defect`` is the largest composition defect of the
    units settled, how far their matrices are from composing by the
    multiplication table up to a phase (:func:`.._matrices._composition_defect`).
    ``roots`` is ``(found, matrices)`` for the first unit whose matrices no
    choice of the roots the canonical norms allow brings onto the
    multiplication table, and ``None`` where there is no such unit.
    """
    T_pos = ops.T_pos
    group = ops.group
    kept = ops.kept
    nstate = ops.nstate
    prod = ops.prod
    syms = ops.syms
    u_pos = ops.u_pos
    product_tol = bat.product_tol
    reach = bat.reach
    tree = bat.tree
    units = bat.units

    eye_n = np.eye(nstate, dtype=complex)
    Dn = {p: np.zeros((nstate, nstate), dtype=complex) for p in Dsearch}
    anchor_n = None if T_pos is None else np.zeros((nstate, nstate), dtype=complex)
    DT_all = None if T_pos is None else np.zeros((nstate, nstate), dtype=complex)
    worst, norm_ref, roots = 0.0, {}, None
    for unit in units:
        idx = unit.states
        here = dict(tree, nstate=int(len(idx)))
        sub = {p: np.asarray(M, dtype=complex)[np.ix_(idx, idx)] for p, M in Dsearch.items()}
        bad, D_u, norm_u, anchor_u, DT_u, defect_u, roots_u = _settle_unit(
            sub, group=group, tree=here, kept=kept, u_pos=u_pos, prod=prod,
            anchor_of=_part_anchor(ops, mul, unit), reach=reach, product_tol=product_tol)
        worst = max(worst, float(defect_u))
        if bad is not None:
            return _part_unsettled(ops, bat, bad, unit), None, None, None, None, worst, None
        if roots is None:
            roots = roots_u
        for p, M in D_u.items():
            Dn[p][np.ix_(idx, idx)] = M
        if T_pos is not None:
            anchor_n[np.ix_(idx, idx)] = anchor_u
            DT_all[np.ix_(idx, idx)] = DT_u
        if unit.anchor:
            norm_ref = norm_u
    at = {int(syms[p].ind): p for p in kept}
    normalization = {}
    for ind, rec in norm_ref.items():
        M = Dn[at[int(ind)]]
        normalization[int(ind)] = replace(
            rec, matrix=M,
            norm_residual=float(np.abs(_power(M, bool(syms[at[int(ind)]].time_reversal),
                                              rec.order) - rec.required * eye_n).max()))
    return None, Dn, normalization, anchor_n, DT_all, worst, roots


# 6a. The read-off.
# The detector has solved `X H(k) = H(Rk) X` for every operation the build
# represents, and `validated._bases` holds the whole solution space.  A
# representation of the form `D(g) = O(g) (x) S(g)` is a member of it, so the
# factors are measured on that space and not searched through.  Only the
# generators are read.  The antiunitary operation carries the scalars step 4
# read, with no sign left to search under either strategy, and every other
# operation is propagated along the tree of words.
def _solution_space(ops, p):
    """The detector's solution space for the operation at position ``p``, and a note.

    ``ops`` is the build's :class:`_Operations`, whose ``validated`` holds the
    spaces, and ``p`` is the operation's position in ``ops.syms``.  Returns
    ``(basis, note)``: the space's basis, an array of shape
    ``(dim, nstate, nstate)``, and a ``note`` of ``None``.  Where ``validated``
    holds only the operation's measured intertwiner, ``basis`` is that one
    matrix and ``note`` says so.  ``basis`` is ``None`` and ``note`` says why
    where there is no intertwiner either, or where projecting the space onto
    the tensor structure would allocate more than ``_READOFF_MAX_BYTES``.
    """
    nstate = ops.nstate
    syms = ops.syms
    validated = ops.validated

    ind = int(syms[p].ind)
    B = getattr(validated, "_bases", {}).get(ind)
    if B is None or len(np.asarray(B)) == 0:
        U = validated._intertwiners.get(ind)
        if U is None:
            return None, (f"the validated group carries no measured intertwiner for "
                          f"generator #{ind}")
        # A hand-built ValidatedGroup may carry only the measured intertwiner.
        # That is the whole space where the space is one-dimensional; where it
        # is not, the read-off's projection has nothing to choose from, and its
        # block defects say so.
        return np.asarray(U)[None, :, :], f"only `intertwiners[{ind}]`, not the whole space"
    B = np.asarray(B)
    if B.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
        return None, (f"the solution space of generator #{ind} is "
                      f"{B.shape[0]}-dimensional at nstate = {nstate}, whose "
                      f"projection onto the tensor structure would allocate more "
                      f"than {_READOFF_MAX_BYTES / 1e6:.0f} MB")
    return B, None


def _read_off(ops, mul, meas, bat):
    """The orbits' factors, read off the generators' solution spaces.

    ``ops``, ``mul``, ``meas`` and ``bat`` are the build's
    :class:`_Operations`, :class:`_Multiplets`, :class:`_Measured` and
    :class:`_Battery`.  On each orientation branch of ``bat.orients``, each
    generator's solution space (:func:`_solution_space`) is projected onto the
    form ``D(g) = O(g) ⊗ S(g)``, and the ratio of each orbit's block to
    ``S(g)``, relative to the anchor's orbit, is the orbit's factor.  On the
    branch ``+1``, where the measurement pinned a factor, its factor is taken
    instead.  A factor within ``meas.snap`` of a root of unity of order dividing
    twelve is read as that root.  A branch is not read where a space is
    missing, where the anchor's block vanishes or a block misses a multiple of
    ``S(g)`` by more than ``bat.internal_tol``, or where a factor is not a
    phase.  Returns the :class:`_ReadOff`: the factors of each branch read, and
    why none was where none was.
    """
    gens = ops.gens
    syms = ops.syms
    anchor = mul.anchor
    anchor_cls = mul.anchor_cls
    chan_states = mul.chan_states
    classes = mul.classes
    cperm = mul.cperm
    free_cls = mul.free_cls
    nchan = mul.nchan
    ncls = mul.ncls
    S_gen = meas.S_gen
    free_lead = meas.free_lead
    snap = meas.snap
    unpinned = meas.unpinned
    verdict = meas.verdict
    zero = meas.zero
    all_blocks = bat.all_blocks
    internal_tol = bat.internal_tol
    orients = bat.orients

    rec = {"n_enumeration": None, "branches": {}, "reason": None, "notes": []}
    branches, choices, reasons, read_of = {}, [], [], {}
    for orient in orients:
        per_gen, ratios, defects, combo, why = {}, {}, {}, [], None
        read_as = {}
        for gi, gp in enumerate(gens):
            ind = int(syms[gp].ind)
            basis, note = _solution_space(ops, gp)
            if note and note not in rec["notes"]:
                rec["notes"].append(note)
            if basis is None:
                why = note
                break
            lifts = S_gen[gp]
            if orient < 0:
                # the g^-1 branch's candidate is the adjoint, so its blocks
                # are the transposed ones and carry S(g)^dagger
                lifts = [M.conj().T for M in lifts]
            blocks = _ansatz_blocks(chan_states, cperm[gp], orient)
            # The member chosen.  Where the space leaves a block open, the
            # member taken is the one carrying the factor `_relative` gives
            # there: what the measurement pinned, or the trivial relative
            # character, the labeling being chosen in step 7b.
            want = {cls: _relative(ops, meas, gp, cls) for cls in range(ncls)}
            w_of = [(1.0 + 0.0j) if want[int(classes[c])] is None else
                    complex(np.conj(want[int(classes[c])]) if orient < 0
                            else want[int(classes[c])])
                    for c in range(nchan)]
            X, defect, scale = _ansatz_member(basis, blocks, lifts,
                                              prefer=all_blocks, weights=w_of,
                                              tol=zero)
            w_, sv_, vh_ = np.linalg.svd(X)
            entry = {"ind": ind, "dim": int(np.asarray(basis).shape[0]),
                     "ansatz_defect": float(defect), "ansatz_scale": float(scale),
                     "conditioning": float(sv_[-1] / sv_[0]) if sv_[0] > 0 else 0.0}
            per_gen[ind] = entry
            if float(sv_[0]) <= 1e-12:  # pragma: no cover: a guard, not a path
                why = (f"the projection of generator #{ind}'s solution space onto "
                       f"the tensor structure is the zero matrix")
                break
            U = w_ @ vh_
            cls_ratio, cls_defect = {}, {}
            for cls in range(ncls):
                c0 = (anchor if cls == anchor_cls
                      else int(np.flatnonzero(classes == cls)[0]))
                rows, cols = blocks[c0]
                rho, res = _proportionality(U[np.ix_(rows, cols)], lifts[c0])
                cls_ratio[cls], cls_defect[cls] = complex(rho), float(res)
            entry["block_defect"] = max(cls_defect.values())
            lam = cls_ratio[anchor_cls]
            entry["lambda"] = complex(lam)
            if abs(lam) < snap:
                why = (f"generator #{ind}'s block on orbital 0's multiplet is not a "
                       f"multiple of S(g): |lambda| = {abs(lam):.3g}")
                break
            signs = {cls: (np.conj(r / lam) if orient < 0 else r / lam)
                     for cls, r in cls_ratio.items()}
            # What the measurement pinned is taken from it, on the branch it
            # read: every orbit's ratio to the anchor.  The member above says
            # only whether the tensor structure holds.
            if orient > 0:
                for cls in range(ncls):
                    got = verdict.chosen.get((ind, int(cls)))
                    if (int(cls) != anchor_cls and int(cls) not in unpinned
                            and got is not None):
                        signs[int(cls)] = complex(got)
            # An orbit the hoppings leave free of the anchor carries the
            # trivial relative character (`_relative`), the labeling choosing
            # its part's own in step 7b.  The other orbits of its part are
            # pinned to the part's lead by the hoppings, at the ratio the
            # measurement read between them.
            led = sorted(unpinned, key=lambda c: (free_lead.get(int(c), int(c)) != int(c),
                                                  int(c)))
            for cls in led:
                lead_cls = free_lead.get(int(cls), int(cls))
                rel = (verdict.within.get((ind, int(cls))) if orient > 0
                       else None)
                if rel is None and lead_cls != int(cls) and abs(cls_ratio[lead_cls]) > zero:
                    rel = cls_ratio[int(cls)] / cls_ratio[lead_cls]
                    rel = np.conj(rel) if orient < 0 else rel
                if lead_cls != int(cls) and rel is not None:
                    signs[int(cls)] = complex(signs[lead_cls]) * complex(rel)
                elif want.get(int(cls)) is not None:
                    signs[int(cls)] = complex(want[int(cls)])
            entry["signs"] = {int(c): complex(z) for c, z in signs.items()}
            for cls in range(ncls):
                ratios[(ind, int(cls))] = complex(signs[cls])
                defects[(ind, int(cls))] = float(cls_defect[cls])
            bad = max(cls_defect, key=lambda c: cls_defect[c])
            if cls_defect[bad] > internal_tol:
                why = (f"generator #{ind}'s block on orbit {bad} of the multiplets "
                       f"is not a multiple of S(g), the operation's own internal "
                       f"factor: it misses one by "
                       f"{cls_defect[bad]:.3g} > {internal_tol:.3g}")
                break
            factors = []
            for c in free_cls:
                if int(c) in unpinned:
                    # settled, not read against the anchor: the hoppings
                    # leave this orbit free of it, and it carries the factor
                    # its part is settled with above
                    factors.append(complex(signs[c]))
                    continue
                # The factor as measured: a root of unity where it is one
                # within the noise, the measured phase otherwise.  The three
                # checks say whether it builds a representation.
                z = _orbit_factor(signs[c], snap)
                if z is None:
                    why = (f"generator #{ind} reads {signs[c]:.6g} on orbit {c} of "
                           f"the multiplets, which is no phase: its modulus misses one "
                           f"by {abs(abs(signs[c]) - 1.0):.3g}")
                    break
                factors.append(z)
                read_as[(ind, int(c))] = z
            if why is not None:
                break
            combo.extend(factors)
        branch = {"orientation": int(orient), "generators": per_gen,
                  "ratios": ratios, "defects": defects,
                  "ok": why is None, "reason": why}
        rec["branches"][int(orient)] = branch
        if why is None:
            branches[int(orient)] = branch
            choices.append((orient, tuple(combo)))
            read_of[int(orient)] = read_as
        else:
            reasons.append(f"branch {orient:+d}: {why}")
    if not choices:
        rec["reason"] = "; ".join(reasons) if reasons else "no branch was read"
    return _ReadOff(choices=choices, branches=branches,
                    reason=rec["reason"], record=rec, read=read_of)


def _build(ops, mul, meas, bat, choices):
    """One candidate for each choice of branch and factors, its covariance measured.

    ``ops``, ``mul``, ``meas`` and ``bat`` are the build's
    :class:`_Operations`, :class:`_Multiplets`, :class:`_Measured` and
    :class:`_Battery`.  ``choices`` yields ``(orientation, combo)``: the
    branch, ``1`` or ``-1``, and the factors on the orbits other than the
    anchor's, generator by generator.  Each generator's matrix carries each
    multiplet onto its image with its orbit's factor times its internal block.
    The antiunitary operation's matrix does the same with the scalar
    ``meas.reversal`` holds on each orbit in place of the factor.  On the
    branch ``-1`` their adjoints are taken.  Every other operation's matrix is
    propagated from them.  Returns a list of records, each holding
    ``"orientation"``, ``"combo"``, ``"residual"``, the largest covariance
    residual over the operations (:func:`_covariance_of`), and ``"matrices"``,
    each operation's matrix by position.
    """
    T_pos = ops.T_pos
    gens = ops.gens
    free_cls = mul.free_cls
    ncls = mul.ncls
    S_gen = meas.S_gen
    reversal = meas.reversal

    out = []
    for orient, combo in choices:
        sign = {}
        for gi in range(len(gens)):
            sgn = np.ones(ncls, dtype=complex)
            for j, c in enumerate(free_cls):
                sgn[c] = combo[gi * len(free_cls) + j]
            sign[gi] = sgn
        Dgen = [_assemble(ops, mul, gens[gi], S_gen[gens[gi]], sign[gi])
                for gi in range(len(gens))]
        DT = None
        if T_pos is not None:
            DT = _assemble(ops, mul, T_pos, S_gen[T_pos], reversal)
        if orient < 0:
            Dgen = [d.conj().T for d in Dgen]
            DT = None if DT is None else DT.conj().T
        D = _propagate(bat, Dgen, DT)
        out.append({"orientation": int(orient), "combo": tuple(combo),
                    "residual": _covariance(ops, bat, D), "matrices": D})
    return out


# 6b. The three checks together.
# Three independent checks, each of which a candidate must pass, run cheapest
# first, so that the costlier ones see the fewest candidates:
#
#   covariance       |kept| x n_battery triple products of size nstate;
#                    removes a wrong relative sign inside a Wyckoff orbit.
#   internal factor  |kept_u| x nchan block proportionalities, each of a
#                    multiplet's size, on every block, those that swap sites
#                    included; removes the wrong orientation branch, g against
#                    g^-1, wherever S(g)^dag is not a multiple of S(g), a
#                    branch that is covariant and multiplicative, and a build
#                    whose swapped blocks carry the lift S(sigma_h) S(g) of the
#                    operation's horizontal-mirror partner, covariant and
#                    multiplicative too where the layer group and the group
#                    named are isomorphic.
#   product table    one normalization and |kept|^2 products of size nstate;
#                    removes a set that is not a representation.
#
# The checks are independent, so their order is one of cost, and the check
# that removed each candidate is recorded with it.  The checks take the set of
# candidates, so the read-off's one or two and the enumeration's thousands go
# through the same arithmetic.
def _gate(ops, mul, bat, records):
    kept = ops.kept
    prod = ops.prod
    spf = ops.spf
    syms = ops.syms
    internal_tol = bat.internal_tol
    product_tol = bat.product_tol
    tol = bat.tol
    units = bat.units

    # merge candidates that build the identical representation
    distinct = []
    for rec in records:
        for grp in distinct:
            ref = grp[0]["matrices"]
            if all(np.abs(rec["matrices"][p] - ref[p]).max() < 1e-12 for p in kept):
                grp.append(rec)
                break
        else:
            distinct.append([rec])

    covariant = [grp for grp in distinct if grp[0]["residual"] <= tol]
    for grp in covariant:
        grp[0]["internal"] = _internal_defect(ops, mul, bat, grp[0]["matrices"])
    consistent = [grp for grp in covariant if grp[0]["internal"] <= internal_tol]
    for grp in consistent:
        rec = grp[0]
        # the coupled parts are settled unit by unit (step 5c); a crystal of
        # one part is one unit
        bad, Dn, norm, anchor_T, DT_n, defect, roots = _settled_by_part(
            ops, mul, bat, rec["matrices"])
        rec["composition"] = float(defect)
        if bad is not None:
            rec["product"], rec["refusal"] = float("inf"), bad
            continue
        rec["normalized"] = (Dn, norm, anchor_T, DT_n)
        rec["product"] = _product_table_residual(Dn, kept, syms, prod, spf)
        if roots is not None:
            rec["roots"], rec["roots_matrices"] = roots
    survivors = [grp for grp in consistent if grp[0]["product"] <= product_tol]
    if len(units) > 1 and len(survivors) > 1:
        # A part's own phase is set by its settling and is not a choice:
        # candidates that differ only in the phase a part is settled out of
        # are one representation once each part is settled, and are merged as
        # identical matrices are above.
        merged = []
        for grp in survivors:
            Dn = grp[0]["normalized"][0]
            for keep in merged:
                same = keep[0]["normalized"][0]
                if all(np.abs(Dn[p] - same[p]).max() < 1e-12 for p in kept):
                    keep.extend(grp)
                    break
            else:
                merged.append(grp)
        gone = {id(grp) for grp in survivors} - {id(grp) for grp in merged}
        distinct = [grp for grp in distinct if id(grp) not in gone]
        covariant = [grp for grp in covariant if id(grp) not in gone]
        consistent = [grp for grp in consistent if id(grp) not in gone]
        survivors = merged
    alive = {id(grp) for grp in survivors}

    def _killed_by(grp):
        if id(grp) in alive:
            return None
        if grp[0]["residual"] > tol:
            return "covariance"
        if grp[0]["internal"] > internal_tol:
            return "internal_factor"
        return "product_table"

    spectrum = tuple(sorted(float(grp[0]["residual"]) for grp in distinct))
    int_spectrum = tuple(sorted(float(grp[0]["internal"]) for grp in covariant))
    prod_spectrum = tuple(sorted(float(grp[0]["product"]) for grp in consistent))
    candidates = tuple(
        {"orientation": int(grp[0]["orientation"]), "combo": grp[0]["combo"],
         "n_merged": len(grp), "covariance": float(grp[0]["residual"]),
         "internal": float(grp[0].get("internal", float("nan"))),
         "product": float(grp[0].get("product", float("nan"))),
         "killed_by": _killed_by(grp)}
        for grp in distinct)

    return _GateOutcome(
        records=records, distinct=distinct, covariant=covariant,
        consistent=consistent, survivors=survivors,
        search={"n_candidates": len(records), "n_distinct": len(distinct),
                   "n_covariant": len(covariant), "n_consistent": len(consistent),
                   "n_survivors": len(survivors),
                   "gate": float(tol), "internal_gate": float(internal_tol),
                   "product_gate": float(product_tol),
                   "best": spectrum[0] if spectrum else float("nan"),
                   "runner_up": spectrum[1] if len(spectrum) > 1 else float("nan"),
                   "residuals": spectrum,
                   "internal_best": (int_spectrum[0] if int_spectrum
                                     else float("nan")),
                   "internal_runner_up": (int_spectrum[1] if len(int_spectrum) > 1
                                          else float("nan")),
                   "internal_residuals": int_spectrum,
                   "product_best": (prod_spectrum[0] if prod_spectrum
                                    else float("nan")),
                   "product_runner_up": (prod_spectrum[1] if len(prod_spectrum) > 1
                                         else float("nan")),
                   "product_residuals": prod_spectrum,
                   "killed_by": {"covariance": len(distinct) - len(covariant),
                                 "internal_factor": len(covariant) - len(consistent),
                                 "product_table": len(consistent) - len(survivors)},
                   "candidates": candidates})


def _one_labeling_apart(ops, mul, bat, A, B, tol=1e-8):
    """Whether two settled candidates differ by one character on each coupled part.

    ``ops``, ``mul`` and ``bat`` are the build's :class:`_Operations`,
    :class:`_Multiplets` and :class:`_Battery`, and ``A`` and ``B`` map each
    operation's position to the two candidates' settled matrices.  Returns
    ``True`` where, for every unitary operation and every coupled part, each
    block of ``B`` carrying a multiplet of the part to its image is one phase
    times ``A``'s block, the same phase over the part, within ``tol``; and
    ``False`` otherwise.  Only the unitary operations are compared, since the
    site irreps and the band labels are characters of them.
    """
    u_pos = ops.u_pos
    chan_states = mul.chan_states
    cperm = mul.cperm
    part_chans = bat.part_chans

    for p in u_pos:
        for chans in part_chans:
            ratio = None
            for c in chans:
                rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
                cols = np.asarray(chan_states[c], dtype=int)
                a, b = A[p][np.ix_(rows, cols)], B[p][np.ix_(rows, cols)]
                rho, res = _proportionality(b, a)
                if res > tol or abs(abs(rho) - 1.0) > tol:
                    return False
                if ratio is None:
                    ratio = rho
                elif abs(rho - ratio) > tol:
                    return False
    return True


def _pin_branch(ops, mul, bat, survivors):
    """The survivors of the checks, a tie between the two orientation branches settled.

    ``ops``, ``mul`` and ``bat`` are the build's :class:`_Operations`,
    :class:`_Multiplets` and :class:`_Battery`, and ``survivors`` holds the
    candidates that pass the three checks, each a list of the records that build
    the same matrices.  Where they are two, one on each branch, and, where the
    site tables were read, one labeling apart (:func:`_one_labeling_apart`),
    the branch ``+1`` is kept and the tie recorded with its note.  Returns the
    :class:`_PinnedBranch`, holding ``survivors`` as handed in and no tie
    where there is none to settle.
    """
    site_tables = mul.site_tables

    out = _PinnedBranch(survivors=survivors, tie=False, note=None)
    if len(survivors) != 2:
        return out
    per = {}
    for grp in survivors:
        per.setdefault(int(grp[0]["orientation"]), []).append(grp)
    if sorted(per) != [-1, 1] or any(len(v) != 1 for v in per.values()):
        return out
    if site_tables and not _one_labeling_apart(ops, mul, bat, per[1][0][0]["normalized"][0],
                                                per[-1][0][0]["normalized"][0]):
        return out
    return replace(out, survivors=[per[1][0]], tie=True, note=(
        "the two orientation branches both carry H(k) onto itself, factorize on every "
        "site into (character) x (the group's own internal factor) and reproduce the "
        "multiplication table, and are one "
        "representation up to a one-dimensional character on each coupled part, so "
        "the same labelings and site labels are open on both and nothing "
        "measures between them; the branch was pinned by convention to +1, "
        "the intertwiner that carries k -> gk."))



def _with_multiplicity_spaces(table, chosen, group, assignment):
    """``table`` with each conserved pseudospin's chosen site irreps, its rows marked.

    ``table`` is the atomic-limit table, a :class:`..symmetry._OrbitalTable`,
    and ``chosen`` holds the chooser's :class:`.._labeling._PseudospinChoice`
    for each pseudospin (:func:`.._labeling._labeling`).  ``group`` and
    ``assignment`` give the Wyckoff position of each pseudospin's site.  Each
    pseudospin becomes a :class:`..symmetry._MultiplicitySpace`, its chosen
    site irreps and every other choice named against its site's table.  A row
    of a pseudospin's orbitals that reads ``hamiltonian`` reads ``lowest-l``
    instead where the representations a declaration leaves give more than one
    site irrep there.  Returns the table with its ``multiplicity_spaces`` and
    these rows.
    """
    wyckoff = _wyckoff_sites(group, assignment)
    spaces, marked = [], set()
    for got in chosen:
        rec = got.record
        site = int(rec.site)
        tab = table.site_tables.get(site)
        here = wyckoff.get(site)
        # how many orbitals of the copies one site holds, a declared spinor
        # pair counted once, as one spinful orbital
        on_site = set(int(o) for o in assignment.site_orbitals[site])
        held = sum(1 for o in rec.orbitals if int(o) in on_site)
        spaces.append(_MultiplicitySpace(
            orbitals=tuple(int(o) for o in rec.orbitals),
            wyckoff="" if here is None else f"{int(here.multiplicity)}{here.letter}",
            dim=int(rec.dim), content=_named_content(tab, got.content)[0],
            contents=tuple(_named_content(tab, other)[0] or "(unnamed)"
                           for other in got.contents),
            n_representations=int(rec.n_admitted),
            determined_by=("declaration" if got.declared else
                           "lowest-l" if int(rec.n_admitted) > 1 else "hamiltonian"),
            _per_site=held // (2 if table.spin is not None else 1),
            _pairs=table.spin is not None))
        if int(got.open) > 1:
            marked |= {int(o) for o in rec.orbitals}
    rows = tuple(replace(o, determined_by="lowest-l")
                 if o.orbital in marked and o.determined_by == "hamiltonian" else o
                 for o in table.orbitals)
    return replace(table, orbitals=rows, multiplicity_spaces=tuple(spaces))


def _with_band_representations(rep):
    """``rep`` with each orbit's band representation on its atomic-limit table.

    ``rep`` is a :class:`..symmetry.SpaceGroupRep` whose matrices are final:
    each orbit's band representation is read off them, from the little-group
    representations its states carry (:func:`..srsi._band_representations`).
    Returns ``rep`` as it is where it has no atomic-limit table or no orbit's
    band representation is found.
    """
    table = rep._characters
    if table is None:
        return rep

    orbits = _srsi._band_representations(rep)
    if not orbits:
        return rep
    return replace(rep, _characters=replace(table, band_representations=orbits))


def _with_tables_unavailable(rep):
    """``rep``, where its group has no tables, with each row saying so.

    ``rep`` is a :class:`..symmetry.SpaceGroupRep`.  A group whose cell
    ``irrep`` could not match to the standard setting of the tables the
    irreptables package ships is returned without tables
    (``GroupData._tables_ready`` false), so no row takes a name from them.
    Each row's :attr:`..symmetry._OrbitalRow.note` then says so, with the
    group's own note, which carries ``irrep``'s message.  Returns ``rep`` as it
    is where the tables are available, where the cell is a supercell of the
    crystal's primitive cell (``_lattice_index`` above one), or where ``rep``
    has no atomic-limit table.
    """
    table, group = rep._characters, rep._group
    if (table is None or bool(group._tables_ready)
            or int(getattr(group, "_lattice_index", 1) or 1) > 1):
        return rep
    why = (f"no site irrep here is named from the tables: they are unavailable for "
           f"{group.name} #{group.number_str} -- "
           + (group._setup_note or "irrep could not match this cell to the BCS standard "
                                  "setting"))
    rows = tuple(replace(o, note=(o.note + "; " if o.note else "") + why)
                 for o in table.orbitals)
    return replace(rep, _characters=replace(table, orbitals=rows))


#: The defaults of :func:`_build_rep`, read off its own signature: among them
#: the battery its covariance check is measured on and the operations it
#: represents.
_BUILD_DEFAULTS = {name: p.default
                   for name, p in inspect.signature(_build_rep).parameters.items()
                   if p.default is not inspect.Parameter.empty}

#: Every keyword :func:`_build_rep` takes, read off its own signature once, so that
#: the words handed to the build by name are checked against the build itself.
_BUILD_KEYWORDS = frozenset(name for name, p in inspect.signature(_build_rep).parameters.items()
                            if p.kind is p.KEYWORD_ONLY)


def _exact_gate(model, group, validated, build_kw=None) -> float:
    r"""The tolerance of :func:`_build_rep`'s covariance check for ``model`` on ``group``.

    ``validated`` is the measurement of ``group``'s operations, whose
    operations are represented unless ``build_kw`` names others.
    ``build_kw`` holds the build's own keywords; ``operations=``,
    ``include_vacuum_ops=``, ``k_pts=``, ``n_battery=`` and ``seed=`` bear on
    the tolerance, and the build's default stands for each one left out.
    Returns :func:`.._matrices.covariance_gate` for ``|G|`` the number of
    operations represented, ``N`` the number of the model's states, and
    ``max|H|`` and the rounding of the Bloch phases on the build's battery, in
    the units of ``H``.  A detection record carries neither ``N`` nor
    ``max|H|``, so :func:`..groups.realized_group` and
    :func:`..symmetrization._symmetrized` measure the tolerance here and
    classify operations against it.
    """
    kw = dict(build_kw or {})

    def _given(name):
        return kw[name] if name in kw else _BUILD_DEFAULTS[name]

    syms = {int(s.ind): s for s in group._symmetries}
    ops = _given("operations")
    indices = [int(i) for i in (validated._indices if ops is None else ops)]
    padded = bool(_given("include_vacuum_ops"))
    kept = [syms[i] for i in indices
            if padded or not _moves_padded(syms[i], group._vacuum_dirs)]
    k_pts = _covariance_battery(group, _given("k_pts"), _given("n_battery"), _given("seed"))
    bloch = _bloch.evaluator(model)
    H = np.asarray(bloch(k_pts))
    return covariance_gate(len(kept), int(model.nstate), float(np.abs(H).max()),
                           _phase_rounding(bloch, group, kept, k_pts))
