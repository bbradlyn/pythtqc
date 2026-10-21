"""The representation, built once on what the verdict stage measured: one matrix
per operation on the model's states, in the tables' convention, with the
atomic-limit table of its orbitals, the content assigned to a conserved
pseudospin's copies and the band representation each orbit induces.
"""

from __future__ import annotations

from dataclasses import replace
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
    _spin_pair_split, bloch_operator, covariance_gate)
from ._grammar import _declaration_grammar
from ._naming import _site_tables
from .symmetry import MultiplicitySpace, SpaceGroupRep, _fold
from ._multiplets import (
    _channel_perms, _factor_said, _multiplet_blocks, _orbit_factor, _site_multiplets)
from ._declaration import (
    _covariant_subgroup, _declaration_breaks, _declared_blocks, _declared_key,
    _lead_of, _multiplet_characters, _named_content, _normalize_declaration,
    _verdicts_and_choices)
from ._labelling import (
    _assembled, _atomic_limit_table, _coupled_parts, _labelling, _wyckoff_sites)


def _no_multiplicity_representation(group, record, layout, kept, syms) -> Refusal:
    r"""``rep_not_multiplicative`` — a conserved pseudospin whose copies no representation fits.

    The operators ``H`` conserves on the orbit are a whole matrix algebra on
    ``record["dim"]`` copies, so every representation of the group on its
    states is ``B ⊗ B'`` with ``B`` the factor ``H`` fixes and ``B'`` a
    representation on the copies making up the factor system ``B`` leaves
    them (:class:`.._sectors.InducedOrbit`); no sum of the group's irreducible
    representations with that factor system, of that dimension and extended to
    time reversal with the square the tables read, is admitted.  So no
    representation of the group with its own factor system exists on these
    states: the projective class they carry is one the multiplicity space
    cannot make up.
    """
    k, orbitals = int(record["dim"]), tuple(record["orbitals"])
    named = f"{group.name} #{group.number_str}"
    return Refusal(
        "rep_not_multiplicative",
        f"the Hamiltonian conserves a {_fold(k)} pseudospin on orbitals {orbitals}: the "
        f"operators it conserves there are every {k} x {k} matrix on {k} identical "
        f"copies of the states it acts on, so every operation's matrix on them is the "
        f"one factor the hoppings fix times a matrix on the copies. A representation "
        f"of {named} with its own factor system needs the copies to carry a "
        f"{k}-dimensional representation making up the factor system the fixed "
        f"factor leaves them, time reversal's square included, and of the "
        f"{int(record['n_candidates'])} sums of the group's irreducible representations "
        f"with that factor system that have dimension {k}, none does: no "
        f"representation of {named} with its own factor system exists on these "
        f"states.",
        value=0.0, threshold=1.0,
        context={"multiplets": layout, "multiplet": orbitals, "multiplicity": k,
                 "n_candidates": int(record["n_candidates"]),
                 "indices": tuple(int(syms[p].ind) for p in kept)})


def _raise_unformed(group, unformed, n_cand, gate, *, spinful):
    r"""Raise for operations handed to the build none of whose matrices has its own form.

    The operations a representation is built for are the ones the measurement
    read in their own form, so one that has none is the package disagreeing
    with itself: :class:`..refusal.InternalError`, naming each such operation
    and the condition that failed.

    ``unformed`` is ``{ind: {"own_form": condition, "covariance": residual,
    ...}}``, read off the record each operation's own form was read on
    (:func:`.._measure._own_form_read`): the condition that failed, ``"site
    permutation"`` -- no matrix carries each site's states onto its image's --
    or ``"spin lift"`` -- none carries the operation's own spin rotation --
    and the covariance of the matrix of that form nearest its space, where
    one was measured.  ``n_cand`` is how many operations were to be
    represented, and ``gate`` the covariance gate they were kept at.
    ``spinful`` says whether the states carry a spin, the model's own or
    declared: only then is a spin rotation asked of them, and named.
    """

    syms = {int(s.ind): s for s in group.symmetries}

    def _named(ind):
        return (f"{_record_name(syms[ind], group.lattice)} "
                f"(#{ind}, its {unformed[ind]['own_form']})")

    named = ", ".join(_named(ind) for ind in sorted(unformed))
    measured = [float(r["covariance"]) for r in unformed.values()
                if r.get("covariance") is not None]
    closest = min(measured) if measured else None
    reach = ("" if closest is None else
             f" The matrix of its own form nearest any of their spaces carries H(k) onto "
             f"H(Rk) to {closest:.3g}, against the {float(gate):.3g} they were kept at.")
    carried = ("every site's states onto its image's with the operation's own spin "
               "rotation, the condition named failing" if spinful else
               "every site's states onto its image's")
    raise InternalError(
        f"{len(unformed)} of the {int(n_cand)} operations handed to the build, "
        f"{_counted_as(spinful)}, are not "
        f"realized on this orbital content: {named}. Matrices with each one's action on k "
        f"carry H(k) onto H(Rk), but none carries {carried}: they are other operations' "
        f"matrices, not these operations'.{reach} The operations a representation is "
        f"built for are the ones measured to hold in their own form, so this is the "
        f"package disagreeing with itself.")


def _build_rep(model, group, assignment, *, validated=None, multiplets=None,
               reference_orbital=0, characters=None, spin=None,
               generators=None, operations=None,
               phases=(1, -1), orientation_branch=None, include_vacuum_ops=False,
               strategy="readoff", max_candidates=2 ** 20,
               k_pts=None, n_battery=6, tol=None, internal_tol=1e-9,
               product_tol=1e-9, seed=101, _choices=None,
               _returned=True, _labelled=True, _search_record=None,
               params=None) -> SpaceGroupRep | Refusal:
    r"""The matrix each symmetry operation is, on this model's orbitals.

    Returns one matrix ``D(g)`` per operation the Hamiltonian obeys: the
    representation induced from the site content
    :attr:`SpaceGroupRep.characters` returns — each orbital's Wyckoff position,
    site symmetry and site irrep — in the tables' convention (spin along
    ``+z``, a right-handed frame).  It is the object every corep label is read
    through.  The construction is the tensor

    .. math:: D(g) = O(g) \otimes S(g),

    with ``O(g)`` a signed permutation of the **multiplets** of
    :class:`SiteMultiplets`, whose permutation comes from
    :attr:`..groups.OrbitalAssignment.permutations` (never typed), and
    ``S(g)`` the operation's own matrix on each multiplet.  Which orbitals a
    multiplet holds is **measured** — one per orbital where nothing mixes, the
    multiplet where the Hamiltonian binds one — and its matrix is the model's
    own, read off its solution spaces and carried round the Wyckoff orbit, in
    whatever basis the orbitals are written.  What is left free is a finite
    set of signs — one phase per generator and orbit of multiplets, plus the
    orientation branch, ``D`` versus ``D†``, i.e. ``g`` versus ``g⁻¹`` — and
    those are pinned by measurement, never by hand.  Time reversal carries its
    own internal factor on every multiplet times the scalar its solution space
    pins on each orbit of multiplets: one where the orbitals are real
    functions, and ``-1`` where an orbital is written as ``i`` times one
    relative to the first orbital's.  Every candidate goes
    through **three independent gates**, and uniqueness is asserted on their
    joint survivor set:

    1. **Hamiltonian covariance** — ``max_k ‖D H(k) D† − H(Rk)‖ ≤ tol``.  A
       candidate that fails this is not a symmetry of *this Hamiltonian*: a
       wrong relative sign inside a permuted Wyckoff orbit dies here, by orders
       of magnitude.
    2. **the group's own internal factor** — on every multiplet an operation
       fixes, the block must be ``χ(g)·S(g)`` with ``S(g)`` the operation's own
       ``sym.spinor_rotation``: ``max ‖B(g) − χ(g)S(g)‖ ≤ internal_tol``, the
       same defect the site characters are read off
       (:attr:`AtomicLimitTable.residual`).  A candidate that fails this
       commutes with ``H`` exactly as the group demands but is **not the matrix
       set of these operations**, and no orbital of it has a site character to
       declare or to label.
    3. **the group's own multiplication table**, spinor factors included —
       ``max ‖D(g₁)∘D(g₂) − spf(g₁,g₂)·D(g₁g₂)‖ ≤ product_tol``, measured after
       :func:`.._matrices.normalize_rep` has phase-canonicalized the generators.  The
       canonical norm fixes a generator's phase up to a root of unity; where
       nothing else narrows the roots — no antiunitary operation, and a
       multi-dimensional lead multiplet with no table lift behind it — the
       table itself chooses among them wherever positivity's own pick does not
       multiply.  A candidate that fails this is
       covariant but is *not a representation of the group* at all: the
       operation set, the generators or the internal blocks are wrong, or its
       matrices are a projective representation whose class the table does
       not carry — spin-½ states read in single-valued tables, where two
       commuting operations' matrices anticommute — and no choice of phases
       will fix it.

    **No two of the three suffice.**  A ``g⁻¹`` branch is a perfectly good
    representation of the abstract group, only of the elements relabelled by an
    automorphism, so it is exactly covariant *and* exactly multiplicative; what
    separates it is gate 2, since it carries ``S(g)†`` where the group says
    ``S(g)``.  Where every operation is an involution that measurement does not
    exist: ``S(g)† = −S(g)``, the internal-factor gate's per-block coefficient
    absorbs the sign, and every gate is blind.  The branch is then settled by
    the convention ``+1``, the intertwiner that carries ``k → gk``, with
    ``SpaceGroupRep._branch_tie`` set and ``_search["branch_note"]`` saying
    so.  ``rep_not_unique`` still fires when the two survivors are not one
    representation up to a one-dimensional character on each coupled part of
    the crystal.  Which gate killed a
    candidate is recorded per candidate in
    ``_search["candidates"][i]["killed_by"]``, and an empty survivor set is a
    refusal carrying all three residual spectra.

    All three gates are blind to multiplying the matrices of a coupled part of
    the crystal -- the orbitals the hoppings connect, directly or through
    others -- by a one-dimensional character of the group: the hoppings fix how
    the orbitals transform relative to one another, not what they are.  The
    phases a canonical norm leaves are narrowed by time reversal in a grey
    group, and where the group ships none by the tables' own lift on a
    one-dimensional lead multiplet (:attr:`RepAnchor.internal`): a
    normalization of the matrices before the labelling is chosen, not a
    fourth gate.  The **labelling** is then chosen once, the same way for
    every part: its candidates are the characters its declarations leave --
    with time reversal the real ones, without it every one, the complex ones
    included -- and one key orders them, read on the orbital factor.  First
    the part's site content at the lowest angular momentum, at the reference
    orbital's site, then over the crystal, then on the reference orbital's own
    row: a character, the same in any basis the orbitals are written in.  Then
    the letter the orbitals read as: the point-group representation their frame
    character -- the trace of each block carrying a site's orbitals onto its
    image -- averages to, at the lowest angular momentum.  Then the band labels
    themselves: the candidate whose labels come first in the tables' order at
    the high-symmetry points.  Last the group's own order of operations, on
    which no band label rests (:attr:`AtomicLimitRow.determined_by`, and
    :func:`..sewing.hsp_table`'s note).  A part the hoppings leave free of
    the rest is brought to its own canonical norm, and the key reads it from
    its first orbital unless it holds the reference one.  Where the
    Hamiltonian conserves a pseudospin on states read spinless -- ``k``
    identical copies of a part of the crystal that nothing joins, every
    ``k × k`` matrix on the copies commuting with it -- the representation the
    copies carry is left open as well, and it moves the site content: the same
    key chooses it, the copies read copy by copy first, among the
    representations the group's own table admits
    (:attr:`AtomicLimitTable.multiplicity_spaces`); so it chooses among every
    representation induced on uncoupled orbitals a symmetry exchanges, where
    those carry what no one-dimensional character gives.  Where the
    model's own blocks carry noise within a hundred times ``internal_tol`` the
    matrices are the exact representation nearest them, which a near-symmetric
    model's symmetrizing pass averages with.  The conventions the tables fix are
    stated in ``docs/background/conventions.md``.

    This is one of the pieces :func:`..groups.realized_group` already ran, and
    it is re-runnable with different arguments on a group already in hand —
    ``_build_rep(r.model, r.group, r.assignment, validated=r.validated,
    multiplets=...)``, or with a different ``characters=`` or ``generators=``.

    Parameters
    ----------
    model : pythtb.TBModel
    group : ..groups.GroupData
        A group comes from :func:`..groups.realized_group`:
        ``realized_group(model, ...).group``.
    assignment : ..groups.OrbitalAssignment
        From :func:`.._tables._assign_orbitals` on **this** model and group.
    validated : ..groups.ValidatedGroup, optional
        Which operations the Hamiltonian obeys.  Computed here when omitted,
        by :func:`.._measure._validate_group` at its defaults, whose
        ``detector_too_large`` is then handed back as it comes.  Only its
        surviving operations are candidates for representation, and its
        discrepancy list is carried into :attr:`SpaceGroupRep._failed`.  A
        record as the detector measured it, :func:`.._measure._validate_group`'s
        own, is read here once, first, as :func:`..groups.realized_group` reads
        its own: each operation's solution space held to the matrices of its
        own form, each site's states carried onto its image's with the
        operation's own spin rotation on states that carry a spin.  A record
        already read is read as it stands.
    multiplets : SiteMultiplets, optional
        The multiplets each site's orbitals are bound into, **stated** instead
        of measured: the multiplets of every site, each given as the orbitals it
        holds, or a :class:`SiteMultiplets` record.  It is a partition of the
        model's orbitals, so which site index holds which multiplet is this
        assignment's business and a statement is re-keyed onto it — which is
        what carries one partition to another presentation of the same model.
        A statement that is not such a partition raises ``ValueError`` before
        anything is measured: an orbital named twice or not at all, a multiplet
        spread over two sites, a declared spinor pair split.  Omitted — the
        default — the multiplets are
        measured, by the rule :class:`SiteMultiplets` states: a site's own
        operations bind its orbitals, and copies of one site irrep turned from
        site to site of an orbit are one multiplet.  A one-dimensional multiplet
        takes ``S(g)`` to be the spin factor from the group
        (``sym.spinor_rotation``, raw, and ``iσ_y conj(·)`` for antiunitary
        operations), with the extraction *checked* against the detector's own
        intertwiner; a multi-dimensional one — a site whose orbitals form an
        irreducible multiplet of the site-symmetry group, a
        ``(d_z², d_x²₋y²)`` doublet at a ``32`` site, which no signed
        permutation of orbitals can represent — takes the model's own
        intertwiner's block, in whatever basis the orbitals are written.
    reference_orbital : int, optional
        The orbital the lowest-angular-momentum rule reads the crystal from
        first: among the labellings the declarations leave, its site's
        content is put at the lowest angular momentum it allows, and every
        orbital the hoppings couple to it follows; of two rows one site content
        leaves open, the reference orbital's is the one put at the lower.
        Naming a different orbital starts the rule there instead.  It
        constrains nothing a declaration does not.
    characters : sequence or mapping, optional
        **A statement of what the orbitals are** — the physics input the
        Hamiltonian cannot supply, since the hoppings do not measure what an
        orbital is: they measure which operations are symmetries once the
        orbitals are stated.  A list with one entry per orbital, each a word or
        ``None``, or a mapping from an orbital, or from several orbitals of
        one site together, to a word; a mapping from operations to characters
        stands for a word in either (``docs/guide/orbitals.md``).  A declaration in
        none of these forms raises ``ValueError`` before anything is measured,
        and a word that names nothing at its site is refused
        ``unknown_label``.  A declaration states
        each site's content and is read on the measured multiplets, however it
        partitions them: a content the site does not hold breaks the operations
        it is not carried by — declaring ``s`` and ``d`` where the hoppings
        realize a ``p`` doublet states a crystal whose fourfold axis is not a
        symmetry, and declaring them where nothing couples the pair's two
        orbitals says which of the two the pair is.  ``None`` (the default)
        assigns every orbital by the lowest-angular-momentum rule.  A declaration the
        hoppings are not covariant under is not a wrong declaration: it is a
        smaller group, refused
        ``declaration_breaks_group`` here — this call is handed a **fixed**
        group and cannot lower it — and descended to by
        :func:`..groups.realized_group`.

        **What a declaration speaks for.**  The hoppings fix how the orbitals
        they connect transform relative to one another, so a declaration on
        any orbital is a statement about every orbital coupled to it, directly
        or through others: they are read relative to it, which relabels them
        all at once, wherever the declaration sits.  Declaring one sublattice
        of an orbit is the same statement as declaring the other.  Where no
        labelling carries every declaration of such a coupled part at once,
        the operations they disagree on are broken and the group is smaller.
    spin : ..spin.SpinBasis, str or dict, optional
        **The spinor basis of a spinless model**, declared: which orbital is the
        ``↑`` partner of which ``↓``.  A :class:`..spin.SpinBasis` (from
        :func:`..spin.spin_basis`), the shorthand ``"interleaved"`` /
        ``"blocked"``, or a mapping of :func:`..spin.spin_basis` keyword
        arguments.  This is the one piece of physics a ``wannier90`` import
        cannot supply — the importer returns ``nspin = 1`` whatever the
        calculation did, and ``H(k)`` is blind to the names of its ``2N``
        indices.  A one-dimensional multiplet is then a declared pair carrying
        the tables' own SU(2) lift, and everything downstream runs as for a
        native spinful model; see
        ``docs/guide/spin.md#a-wannier90-spinor-import``.  Only for a spinless
        model asked in double-valued tables; a native spinful model carries its
        own spin axis and raises.
    generators : sequence of int, optional
        ``.ind`` of the operations to enumerate signs on.  A minimal generating
        set of the represented unitary operations is found from the product
        table when omitted.
    operations : sequence of int, optional
        Restrict the represented set (default: everything ``validated`` kept).
    phases : sequence of complex, optional
        The factors the enumeration tries on each orbit of multiplets, per
        generator, relative to the reference orbit.  The default ``(1, -1)``
        covers real relative signs; a richer set such as ``(1, 1j, -1, -1j)``
        covers orbits related by roots of unity, at
        ``len(phases)**(n_gen·(n_class−1))`` candidates.  The read-off does not
        consult it: it takes each factor as the solution spaces measure it, a
        root of unity of order dividing twelve where the measured factor is one
        to within the noise and the measured phase otherwise, so ``phases``
        matters only where the enumeration runs.
    orientation_branch : None or {+1, -1}, optional
        Which orientation branches enter the candidate set: ``None`` (default)
        both, ``+1`` only ``g``, ``-1`` only ``g⁻¹``.  The branch is degenerate
        whenever every generator is an involution with a Hermitian
        representative, and the deduplication step then merges it.  This is a
        **restriction of the search space**, not a tolerance: a ``+1`` build
        never learns whether ``g⁻¹`` would also have passed, so it never reports
        ``SpaceGroupRep._branch_tie``.  It is not needed to settle the
        branch; keep it for speed on a large group, to state a convention
        deliberately, or (with ``-1``) to measure what the other branch does.
    include_vacuum_ops : bool, optional
        Represent operations that act non-trivially on a slab's vacuum
        directions.  Off by default: a 2D pythtb model's orbitals carry no
        out-of-plane structure for such an operation to move, so the *geometry*
        does not fix their orbital action, and — because their in-plane k-action
        is degenerate with an in-plane partner's — the *detector* cannot refute
        them either.  Whether a flat layer's own orbital content nevertheless
        determines the action is a question only a built representation answers,
        by covariance.  Left out, they are not in the representation's
        :attr:`SpaceGroupRep.indices`.
    strategy : {"readoff", "enumerate"}, optional
        How the candidates are **found**.  ``"readoff"`` (the default) measures
        the free factors off the detector's own solution space — when the tensor
        structure is realizable ``D(g)`` is an element of it, so the space is
        projected onto tensor-structure form, polar-extracted, and the phases of
        the multiplet orbits read off the block ratios against ``S(g)`` — and
        builds one candidate per orientation branch.  ``"enumerate"`` builds
        every combination of the factors ``phases`` offers instead, which is the
        instrument for a full kill record and is what the read-off falls back to
        when what it read is not clean or is not a representation; a refusal
        after such a fallback says what the read-off read.  **Neither is a
        tolerance**: both answer to the same three gates, so where the measured
        factors are among those ``phases`` offers the two return the same
        representation and differ only in what they cost.
        ``_search["strategy"]`` says which ran.
    max_candidates : int, optional
        Refuse (``gate="rep_search_too_large"``) rather than run an enumeration
        of more than this many candidates.  Default ``2**20``.  Reached only
        when an enumeration is needed, so under ``strategy="readoff"`` the
        refusal also carries the reason the read-off deferred.
    k_pts : array, optional
        Covariance battery points.  Random generic k by default: high-symmetry
        points make far too many wrong candidates look right.
    n_battery : int, optional
        How many random points to use when ``k_pts`` is omitted.
    tol : float, optional
        The covariance gate, absolute and in the units of ``H``.  Omitted, it
        is :func:`covariance_gate` of this build: the larger of the exact floor
        :data:`.._tolerance.EXACT_TOL`, ``5e-14``, and ``|G|·N·ε·max|H|``, the rounding a
        representation of the ``|G|`` operations represented, on ``N`` states,
        carries, with ``max|H|`` the largest entry of ``H(k)`` on the battery.
        Rounding never refuses a valid model; above the gate the verdict is the
        verdict, and nothing is weakened to make a candidate pass.  A number
        given here is the gate.
    internal_tol : float, optional
        The internal-factor gate: ``max ‖B(g) − χ·S(g)‖`` over **every** block
        of every represented unitary operation — the block carrying a
        multiplet to its image as much as the block of a multiplet the
        operation fixes, with the per-block coefficient free.  Dimensionless
        (the blocks are unitary).  Default ``1e-9``.  Vacuous for a spinless
        model with no declared spinor basis (``S`` is ``1×1``).  On a
        multi-dimensional multiplet ``S(g)`` is not a table's but is
        **measured**, off the whole detector solution space, and the gate is
        the statement that the built blocks are multiples of what was measured.
    product_tol : float, optional
        The product-table gate: ``max ‖D(g₁)∘D(g₂) − spf·D(g₁g₂)‖`` over the
        represented pairs.  The matrices are unitary, so unlike ``tol`` this
        number is **dimensionless** and carries no scale from ``H`` — which is
        why it is a separate knob rather than ``tol`` re-used.  Default
        ``1e-9``; tighten it only after reading
        ``_search["product_residuals"]``.
    seed : int, optional
        Seed of the battery's RNG, so the search is reproducible.
    params : dict, optional
        Model parameter values, as ``model.hamiltonian`` takes them.

    Returns
    -------
    SpaceGroupRep or Refusal
        The single surviving candidate: one matrix per represented operation,
        the record of what pinned every discrete choice, and the resolved site
        character of every orbital.  It records no environment
        (:attr:`SpaceGroupRep.environment` is ``None``); the representation
        :func:`..groups.realized_group` returns keeps the environment it was
        read in.

    Refusals
    --------
    ``"rep_not_covariant"``
        no candidate closes the covariance gate — read it as *this operation
        set is not realized on this orbital content*, e.g. a slab operation
        the model cannot represent on the multiplets the Hamiltonian binds.
    ``"rep_internal_factor"``
        covariant candidates exist but none carries the group's own internal
        factor on the multiplets its operations fix — they are somebody else's
        matrices, not these operations'.
    ``"rep_not_multiplicative"``
        candidates pass both of those but none reproduces the multiplication
        table, and no choice of phases will fix it: the operation set, the
        generators or the internal blocks are wrong, or the matrices form a
        projective representation of a class the table does not carry.  The
        second is measured and named — two commuting operations whose matrices
        commute only up to a phase no rephasing removes
        (``context["commuting_operations"]``, ``context["commutation_phase"]``)
        — and where that phase is the one the operations' spin-½ lifts carry
        (``context["spin_half"]``) the states carry half-integer spin: no
        single-valued representation exists on them, and a spinor basis in the
        orbital index is read in the double group once ``spin=`` declares it.
        This build is handed a fixed group; :func:`..groups.realized_group`
        measures the same class before it builds and descends to the largest
        subgroup that carries an ordinary representation.  On a conserved
        pseudospin the class is not the states' alone -- the copies' own
        representation can make it up -- and the refusal says where no
        representation of the copies' space does (``context["multiplicity"]``).
    ``"rep_not_unique"``
        more than one distinct representation survives **all three** gates,
        which means the battery is too small or the model is more symmetric
        than the group says.  All four of these carry the number of candidates
        tried (``context["n_candidates"]``).  The one survivor pair it does
        **not** fire on is the two orientation branches returning the same site
        labels under the same labelling: that is a convention
        rather than a second representation, and it is pinned to ``+1`` with
        ``SpaceGroupRep._branch_tie`` set.
    ``"rep_search_too_large"``
        not about the model at all: an enumeration was needed (asked for, or
        fallen back to) and would have run more candidates than
        ``max_candidates``.  Under ``strategy="readoff"`` the message also says
        why the read-off deferred, which is the thing to act on — a read that is
        not clean says the tensor structure ``O(g) ⊗ S(g)`` does not describe
        this operation set on this orbital content.  The same ceiling bounds
        the enumeration that grows with the *model*: where the Hamiltonian
        conserves sets of states apart that no operation exchanges, or a
        pseudospin, the hoppings do not fix which representation those states
        carry, and the finite set to try is every choice of one for each set
        -- or, on the pseudospin, every sum of the group's irreducible
        representations the copies' space could carry.
    ``"detector_too_large"``
        only where ``validated`` is omitted and measured here: the detector's
        own memory limit on a spectrum with exactly degenerate levels
        (:func:`.._detector._find_intertwiners`), handed back as it comes.
    ``"rep_double_valued_spinless"``
        a structural gate on the valuedness the model and the tables disagree
        about: the model is spinless, the tables asked for are double-valued,
        and nothing said where the spin is.  For a ``wannier90`` import declare
        the spinor basis (``spin="interleaved"``, ``"blocked"``, or explicit
        ``pairs=``) and the build runs.  What stays unsupported is an
        **unpaired** spinor orbital — a single Wannier function carrying
        half-integer spin with no partner, the locally time-reversal-broken
        magnetic case — which needs double-group *site* declarations that
        single-valued site tables cannot express.
    ``"spin_pair_split"``
        a declared spin pair whose two members sit on different sites, where
        the two spin states of one Wannier function share a centre: a
        mislabelled pairing, or orbitals each carrying one spin state with no
        partner at their own centre, a magnetic configuration outside this
        package's present scope.  :func:`..groups.realized_group` refuses it
        from the centres alone before anything is identified, a declared
        ``precision`` reading two centres closer than it as one.  A pairing
        that is well formed and wrong is a different tensor structure, and it
        fails the covariance or the internal-factor gate like any other, where
        :func:`..groups.realized_group` descends instead.
    ``"unknown_label"``, ``"ambiguous_multiplet"``
        what a well-formed ``characters=`` can say that its site cannot read:
        a label that is not an irrep of *that orbital's measured site group*
        (the message names the site group, lists its valid labels and prints
        the generator frame); a flat list that cannot say which orbitals form
        a multiplet; a joint declaration whose irrep is not the dimension of
        the multiplet it names (``{(0, 1): "A"}``, where ``{0: "A", 1: "A"}``
        is the spelling for two ``A`` orbitals).
    ``"declaration_breaks_group"``
        the declared content is not covariant under every operation of the
        group handed in.  Not a wrong declaration and not a tolerance: the
        *positions* allow both of lin3g's ``1a`` orbitals to be ``s`` and the
        Hamiltonian does not, so what was declared is a crystal with less
        symmetry.  This call is handed a fixed group and cannot lower it, so it
        names the subgroup the content **is** covariant on
        (``context["subgroup"]``), the operations it breaks with their rows
        (``context["broken"]``), the maximal readings where several tie, in the
        order the tie rule ranks them with that subgroup first
        (``context["tied"]``), what the declaration that fails measured, in
        irrep names, and the complete ``checks`` table; ``context["group"]`` is
        the group handed in.
        :func:`..groups.realized_group` descends to that subgroup instead of
        refusing.  In a group with **no antiunitary operation** more
        declarations stand: one the hoppings carry up to a one-dimensional
        character of the group breaks nothing, and the representation returned
        is induced from the content as declared.

    A unitary-only build that cannot be anchored to the tables does **not**
    come back as a refusal: see *Raises*.

    Raises
    ------
    ..refusal.InternalError
        A unitary-only build with a one-dimensional reference multiplet (no
        antiunitary operation in the group) whose generator blocks are not real
        multiples of the operations' own ``sym.spinor_rotation``.  The tables fix that gauge
        outright, so this is a bug in the application of the convention and not
        a measurement about the model, and it is deliberately **not** a
        ``Refusal``: no gauge is left to supply and there is nothing to report.
        A model and a table that disagree about *valuedness* falsify the
        anchor's premise rather than its application, and are refused above
        or raise ``ValueError`` below instead.  Also, the package disagreeing
        with itself in what it hands the build: an operation to be represented
        none of whose matrices has its own form, each site's states carried
        onto its image's with its own spin rotation where the states carry a
        spin; operations measured to hold that do not close into a group; no
        generating set of three operations or fewer for the represented
        unitary ones; an internal block the measurement does not reach.
    ValueError
        Grammar in the arguments: a ``characters=`` that is not a list with
        one entry per orbital or a mapping from orbitals to words, that names
        an orbital twice or out of range, that writes orbitals of two sites as
        one key, or that gives the two orbitals of a declared spin pair
        different words; a ``multiplets=`` that is not a
        partition of the model's orbitals into multiplets of one site each; a
        ``spin=`` declaration written for another model; a ``strategy=`` or an
        ``orientation_branch=`` that names nothing; an ``operations=`` that is
        not closed under the group's product, or holds an antiunitary
        operation that is not time reversal times a represented one;
        ``generators=`` that do not generate the represented unitary
        operations; a group of single-valued tables for states that carry
        spin-1/2, natively or by declaration.  The declarations are read
        before anything is measured.

    Notes
    -----
    **Validity conditions.**

    * The search is only as sharp as the battery.  Generic k-points are used
      deliberately; a battery of high-symmetry points would let wrong signs
      through.
    * "Unique" is asserted on *distinct representations*, not on parameter
      choices: candidates that build the identical matrices (an involutive
      generator under the orientation branch) are merged first.
    * The representation's phase is anchored at orbital 0's multiplet, and on
      a coupled part without it at the part's first, whatever
      ``reference_orbital`` names (see :class:`RepAnchor`); the labelling is
      put on top of it, and ``reference_orbital`` picks only the lead the
      labelling's keys read first.
    * The product-table gate runs on the **normalized** matrices — the ones the
      returned object carries — so ``_search["product"]`` and
      ``product_residual`` are the same measurement.  The internal-factor gate
      needs no normalization pass: its defect is invariant under ``D → λD``.
    * ``factor_residual`` — the Bloch operators' factor system, translation
      differences included — stays a **post-hoc** cross-check and is not a
      gate: for matrices that already reproduce the product table it follows
      from the candidate-independent position identity
      ``A τ_a + t = τ_{g(a)} + s``.
    * **The environment is not set here.**  Nothing in this construction knows
      whether a 2D model's layer is free-standing or substrate-bound — that is
      :func:`..groups.realized_group`'s resolved verdict, stamped on the record
      it returns — so :attr:`SpaceGroupRep.environment` and
      :attr:`SpaceGroupRep.free_standing_group` are ``None`` on a
      representation built here.  :func:`..ebr.symmetry_indicator` reads a
      layer's class in the group handed in either way.
    * ``SpaceGroupRep.residuals`` (covariance), ``._norm_residuals``
      (``D^n = ζ``), ``._anchor_residuals`` (``T conj(D) T† = D``),
      ``.product_residual`` and ``.factor_residual`` are all re-measured on the
      returned matrices.

    Examples
    --------
    Redo the build :func:`..groups.realized_group` already made, with each
    site's orbitals stated as one multiplet instead of measured::

        r = tqc.realized_group(model, characters={0: "s"})
        rep = _build_rep(r.model, r.group, r.assignment,
                         validated=r.validated,
                         multiplets=[[orbs] for orbs in r.assignment.site_orbitals])
    """

    if strategy not in ("readoff", "enumerate"):
        raise ValueError("strategy must be 'readoff' (read the choices off the measured "
                         "solution spaces) or 'enumerate' (the exhaustive search), got "
                         f"{strategy!r}")
    params = dict(params or {})
    nspin, norb, nstate = int(model.nspin), int(model.norb), int(model.nstate)

    # ---- 0. the spin the tables assume and the model does not carry ---------
    # Everything below reads `group.spinor` as "these orbitals carry the SU(2)
    # lift": the internal factor is the operations' own `sym.spinor_rotation`,
    # the canonical norms are the double cover's, and the table anchor of
    # `RepAnchor.internal` rests on that lift being pseudo-real.  A spinless
    # model's orbitals carry no such factor *unless a declaration says where
    # it is*, so the premise is either false before a single matrix is built --
    # and the anchor's own diagnosis would then be wrong, `_table_anchored`
    # raising `InternalError` to say *this package misapplied the tables*
    # where the truth is that the physics asked for is unsupported -- or it is
    # restored verbatim by a `spin=` declaration, whose multiplet is a declared
    # (up, dn) pair carrying exactly that lift.  Both settled here,
    # structurally, ahead of all of it.
    spin = _resolve_spin(spin, model)
    # ... and what the declarations say, read as grammar before anything is
    # measured: a malformed `characters=` or `multiplets=` raises here
    _declaration_grammar(norb, characters=characters, multiplets=multiplets, spin=spin,
                         site_of=assignment.site_of_orbital)

    if spin is not None and not bool(group.spinor):
        raise ValueError(
            f"the orbitals were declared to carry spin (`spin=`, a "
            f"{spin.kind!r} spinor basis on {spin.norb} orbitals) and #{group.number_str} "
            f"is a group of single-valued tables: states that carry spin-1/2 have a "
            f"double-valued content, which those tables do not describe.")

    if spin is None and nspin == 1 and bool(group.spinor):
        return Refusal(
            "rep_double_valued_spinless",
            f"the model is spinless (nspin=1, spinful=False) and #{group.number_str} "
            f"was asked for its double-valued tables, with nothing said about where the "
            f"spin is. A double-valued representation of a spin-free basis says the "
            f"orbitals themselves carry half-integer local spin -- a spinor Wannier "
            f"basis, the spin inside each Wannier function -- so the internal factor is "
            f"not the SU(2) lift the tables are written in, and the tables cannot fix "
            f"its gauge. "
            f"If the orbitals are the two spin states of N Wannier functions -- the "
            f"wannier90 case, where the importer returns nspin=1 whatever the "
            f"calculation did -- then declare the spinor basis with `spin=` "
            f"(`order=\"interleaved\"`, `order=\"blocked\"`, or explicit `pairs=`) and "
            f"this build runs: a multiplet becomes a declared (up, dn) pair and the "
            f"internal factor is the tables' own SU(2) lift on it. "
            f"What remains unsupported is an unpaired spinor orbital -- a single "
            f"Wannier function carrying half-integer spin with no partner, which is the "
            f"locally time-reversal-broken magnetic case -- because that needs "
            f"double-group site declarations and the site tables of `characters=` are "
            f"single-valued. "
            f"Otherwise ask this model in its single-valued group "
            f"(`realized_group(model, spinor=False)`), or give the orbitals a spin axis "
            f"(`spinful=True`) if the half-integer content is the electron's own.",
            context={"nspin": nspin, "spinful": bool(getattr(model, "spinful", nspin == 2)),
                     "number_str": str(group.number_str), "spinor": bool(group.spinor),
                     "include_TR": bool(getattr(group, "include_TR", False))})

    # The mirror of it, and this one is not a deferred feature but an invalid
    # ask: ``nspin == 2`` always means double-valued tables, at every level.  A
    # spinful model's states carry the electron's own spin-1/2, so the tensor
    # structure below assembles SU(2) blocks (`_internal_from_table`, taken whenever
    # `nspin == 2`) whatever the group says.  Against a single-valued table
    # those blocks miss the canonical norm, so the anchor would blame this
    # package for it (`InternalError`) or -- in the grey presentation --
    # answer with a true `antiunitary_anchor` measurement of a false setup.
    # Neither is a question worth asking: the pair is a malformed argument, and
    # it raises before anything is diagnosed.
    if nspin == 2 and not bool(group.spinor):
        raise ValueError(
            f"the model is spinful (nspin=2) and #{group.number_str} is a group of "
            f"single-valued tables: the electron's own spin-1/2 gives its states a "
            f"double-valued content, which those tables do not describe.")

    # ---- 0b. what a declared spinor basis has to be, to be one --------------
    # A declared (up, dn) pair is one Wannier function, which no multiplet
    # splits, so its two orbitals share a centre: the check `realized_group`
    # asks before anything is identified, here on the sites handed in.
    if spin is not None:
        split = _spin_pair_split(spin, assignment.site_orbitals,
                                 assignment.site_positions, group.lattice)
        if split is not None:
            return split

    syms = group.symmetries
    pos_of_ind = {int(s.ind): i for i, s in enumerate(syms)}
    prod, tdiff, spf = group.get_product_table(get_diff=True)
    ident = _identity_position(group)

    if validated is None:
        validated = _measure._validate_group(model, group, params=params)
        if isinstance(validated, Refusal):
            return validated                # the detector's memory limit, as it came

    # ---- 0c. EACH OPERATION'S OWN FORM, read once ---------------------------
    # The detector's record holds, for each operation, every matrix with its
    # action on k: another operation's among them, or one composed with a spin
    # rotation the Hamiltonian conserves.  Each operation is held to its own
    # form once, as the verdict stage holds it, before anything reads a
    # solution space; a record already read -- realized_group's -- is read as
    # it stands, and the build assembles what was read.
    validated = _measure._own_form_once(
        model, group, assignment, spin, validated,
        tol=(multiplets._tol if isinstance(multiplets, SiteMultiplets) else None),
        params=params)

    # ---- 1. which operations are candidates for representation -------------
    if operations is None:
        cand = [pos_of_ind[int(i)] for i in validated.indices]
    else:
        cand = [pos_of_ind[int(i)] for i in operations]
    if ident not in cand:
        raise ValueError("the identity operation is not among the operations to "
                         "represent: `operations=` leaves it out, and every "
                         "Hamiltonian obeys it")

    # an operation acting on a vacuum direction of a slab cell is left out
    # unless asked for: the model carries no out-of-plane orbital structure for
    # it to move, and the detector cannot refute it, its in-plane k-action
    # duplicating an in-plane partner's
    kept = [p for p in cand
            if include_vacuum_ops or not _moves_padded(syms[p], group.vacuum_dirs)]
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
                f"represented are not closed under the group's own product table, so no "
                f"representation of them exists")
        if operations is not None:
            raise ValueError(f"{said}; `operations=` names a subgroup.")
        raise InternalError(f"{said}, and they are the operations measured to hold.")

    # ---- 2. generators and the word tree ----------------------------------
    if generators is None:
        gens = _minimal_generators(u_pos, prod, ident)
        if gens is None:
            raise InternalError(
                f"no generating set of size <= 3 was found for the represented unitary "
                f"operations {tuple(int(syms[p].ind) for p in u_pos)}.")
    else:
        gens = tuple(pos_of_ind[int(i)] for i in generators)
        if set(_closure(gens, prod, ident)) != set(u_pos):
            raise ValueError(
                f"`generators=` {tuple(int(syms[p].ind) for p in gens)} do not generate the "
                f"represented unitary operations {tuple(int(syms[p].ind) for p in u_pos)}.")
    words, parent = _bfs_words(gens, prod, ident, u_pos)
    if set(words) != set(u_pos):  # pragma: no cover - guarded by the closure test
        raise InternalError(
            f"the generator tree does not reach the operations "
            f"{tuple(int(syms[p].ind) for p in set(u_pos) - set(words))}.")

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
                    raise ValueError(f"{said}; `operations=` names a subgroup.")
                raise InternalError(f"{said}, and they are the operations measured to hold.")
            au_of[p] = match[0]

    # ---- 3. multiplets, their permutations and their orbits ----------------
    # The multiplets are measured, or stated by `multiplets=`; `characters=`
    # never changes them: it states what each site holds, and step 3b reads it
    # on these multiplets.
    layout = _site_multiplets(model, group, assignment, spin, validated,
                             declared=multiplets)
    su2, site_orbs = layout._spinful, layout._site_orbitals
    nchan = int(layout._nchan)
    chan_states, chan_of = layout._chan_states, layout._chan_of

    # ---- 3b. the declared characters, parsed before anything expensive runs --
    kept_u_ind = {int(syms[p].ind) for p in u_pos}
    kept_au_ind = {int(syms[p].ind) for p in au_pos}
    site_tables, decls, char_note = {}, [], ""
    try:
        # the site tables read a declaration and name the labelling; a build
        # whose matrices are wanted only up to that labelling, with nothing
        # declared, needs neither (`_labelled=False`)
        if _labelled or characters is not None:
            site_tables = _site_tables(group, assignment, kept_u_ind, kept_au_ind,
                                       range(len(site_orbs)))
    except Exception as exc:                     # noqa: BLE001 - see below
        if characters is not None:
            gate = getattr(exc, "gate", "site_group")
            return Refusal(gate, str(exc), context={"multiplets": layout})
        # nothing was declared: the echo record is a courtesy and must never
        # break a build that would otherwise succeed.
        site_tables, char_note = {}, f"site character tables unavailable: {exc}"
    if characters is not None:
        try:
            decls = _normalize_declaration(characters, model, assignment, group,
                                           site_tables, layout, spin=spin)
        except _sitechars.DeclarationError as exc:
            return Refusal(exc.gate, exc.message,
                           context=dict(exc.context, multiplets=layout))
        # A DECLARATION STATES WHAT EACH SITE HOLDS, read on the multiplets
        # above, which it never changes
        decls = _declared_blocks(decls, layout, assignment, site_tables, spin=spin)
        for d in decls:
            d["channels"] = tuple(sorted({int(chan_of[int(o)])
                                          for o in d.get("block", d["orbitals"])}))

    cperm = _channel_perms(model, assignment, syms, kept, layout, spin)
    if isinstance(cperm, Refusal):
        return cperm
    classes, ncls = _channel_classes(cperm, u_pos, nchan)
    # THE ANCHOR the representation is written against before any labelling is
    # chosen: orbital 0's multiplet, whichever orbital the lowest-angular-momentum
    # rule reads the crystal from first (``reference_orbital``, the lead of its
    # own coupled part when the labelling is chosen)
    ref_orb = int(reference_orbital)
    ref_chan = int(chan_of[ref_orb])
    anchor = int(chan_of[0])
    anchor_cls = int(classes[anchor])
    free_cls = [c for c in range(ncls) if c != anchor_cls]

    # ---- 4. THE MEASUREMENT, made once ----------------------------------------
    # What each operation's solution space says about the orbitals, read once:
    # the internal matrix S(g) of every multiplet -- the tables' lift on a
    # one-dimensional multiplet, with the sign the model's transport gives it
    # from site to site, and the block of the model's own intertwiner on a wider
    # one, carried round the orbit -- what the declared content does to each
    # operation, what each multiplet orbit carries relative to the anchor where
    # the hoppings pin it, and the scalar time reversal carries on each orbit.
    # Where `..groups.realized_group` made this reading on this same group it is
    # handed in; otherwise it is made here, once, and nothing below reads a
    # solution space for it again.
    # HOW FAR OUTSIDE A SOLUTION SPACE still counts as inside it: the model's
    # own defect, which for a numerical Hamiltonian is what `internal_tol`
    # already states, and the declared tolerance the multiplets were measured
    # at where that is larger.  An exact model leaves it at the read-off's own
    # 1e-8.  The same declared tolerance reaches every threshold below that
    # judges a solution space: a vanishing block or ratio, a sign read off one.
    eps = layout._tol
    zero, snap = noise_floor(ZERO, eps), noise_floor(SNAP, eps)
    space_tol = max(noise_floor(IN_SPACE, eps), float(internal_tol))
    u_inds = [int(syms[p].ind) for p in u_pos]
    verdict = None
    if _choices is not None and (set(_choices.get("indices", ())) == set(u_inds)
                                 and _choices.get("declared") == _declared_key(decls)
                                 and float(_choices.get("space_tol", -1.0)) == space_tol
                                 and np.array_equal(np.asarray(_choices.get("classes")),
                                                    np.asarray(classes))
                                 and int(_choices.get("anchor", -1)) == anchor
                                 and "blocks" in _choices):
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
    measured_of = verdict["blocks"]
    for record in measured_of.multiplicity:
        if not record["n_admitted"]:
            return _no_multiplicity_representation(group, record, layout, kept, syms)

    def _internal(p):
        """``[S per channel]`` for one operation, or ``None`` where a block is missing."""
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

    S_gen = {}
    for p in list(gens) + ([T_pos] if T_pos is not None else []):
        S = _internal(p)
        if S is None or any(float(np.abs(M).max()) < zero for M in S):
            asked = sorted({len(chan_states[c]) for c in range(nchan)})
            raise InternalError(
                f"no internal block of size {asked} could be obtained for operation "
                f"#{syms[p].ind}: a measured multiplet's block comes from the model's own "
                f"solution space, and the record of the operations measured holds none "
                f"that reaches that multiplet's image.")
        S_gen[p] = S

    # WHAT TIME REVERSAL CARRIES ON EACH ORBIT OF MULTIPLETS, as the measurement
    # read it: its internal factor on every multiplet, times one scalar per
    # orbit -- one on orbitals that are real functions
    reversal = np.ones(ncls, dtype=complex)
    if T_pos is not None and verdict.get("reversal") is not None:
        reversal = np.asarray(verdict["reversal"], dtype=complex)

    # THE CANDIDATE CHARACTERS: the one-dimensional characters of the group the
    # representation is blind to, single-valued on the orbital factor.  With
    # time reversal they are the real ones, each primed operation carrying its
    # unitary part's value; without it, every one-dimensional character of the
    # unitary group, the complex ones included.
    ind_of_pos = {p: int(syms[p].ind) for p in kept}
    admissible = [{ind_of_pos[p]: v for p, v in chi.items() if p in ind_of_pos}
                  for chi in _real_characters(kept, gens, words, parent,
                                              (au_of if T_pos is not None else {}),
                                              prod, ident)]
    if not admissible:            # pragma: no cover - the trivial character always is
        admissible = [{ind_of_pos[p]: 1.0 + 0.0j for p in kept}]
    pool = (admissible if au_pos else
            [{int(k): complex(v) for k, v in chi.items()}
             for chi in _characters_of_subgroup(group, u_inds, roots=True)] or admissible)

    unpinned = verdict["free_classes"]
    # WHAT THE DECLARATIONS ASK, as the verdict stage read it: one character per
    # coupled part of the crystal, each part's declarations constraining its
    # own.  A declared content not covariant on all of this group breaks
    # operations of it.
    if decls:
        sub = _covariant_subgroup(group, u_inds, verdict["requirements"],
                                  unitary_only=not au_pos,
                                  tol=noise_floor(CHARACTER, eps),
                                  joint=verdict.get("joint", ()),
                                  search=verdict.get("search"))
        if isinstance(sub, Refusal):
            return sub
        if sub["broken"]:
            # THE DECLARED CONTENT IS NOT COVARIANT ON ALL OF THIS GROUP, and
            # this call was handed a fixed group and cannot lower it.  So the
            # honest answer is the subgroup it IS covariant on -- the group
            # `realized_group` descends to, made by the same function there.
            return _declaration_breaks(
                group, assignment, decls, verdict,
                standing=[int(syms[p].ind) for p in kept], subgroup=sub["subgroup"],
                broken=sub["broken"], readings=sub["candidates"],
                not_lowered=("The representation was asked for on this group, which it "
                             "cannot lower."),
                site_tables=site_tables, admissible=admissible)

    def _relative(p_gen, cls):
        """The class scalar the build assembles with, at one generator.

        What the measurement pinned relative to the anchor; and on an orbit the
        hoppings leave free of it, the trivial relative character -- the
        labelling chooses its part's own afterwards.
        """
        if int(cls) in unpinned:
            return 1.0 + 0.0j
        got = verdict["chosen"].get((int(syms[p_gen].ind), int(cls)))
        return None if got is None else complex(got)

    # ---- 4c. THE COUPLED PARTS OF THE CRYSTAL, as the measurement read them --
    # The hoppings fix every multiplet's block up to one one-dimensional
    # character per coupled part -- the multiplets they connect, directly or
    # through others: the anchor's first, then the orbits the hoppings leave
    # free of it, grouped where they pin one another.  Inside a part the orbits
    # carry the ratios the hoppings pin between them.  Derived once, here.
    coupled_parts = _coupled_parts(classes, unpinned, verdict.get("parts"))
    part_of_class = {int(cls): i for i, part in enumerate(coupled_parts) for cls in part}
    free_lead = {int(cls): int(min(part)) for part in coupled_parts[1:] for cls in part}

    def _assemble(p, S, sign_of_class):
        D = np.zeros((nstate, nstate), dtype=complex)
        for c in range(nchan):
            rows = np.asarray(chan_states[int(cperm[p][c])], dtype=int)
            cols = np.asarray(chan_states[c], dtype=int)
            D[np.ix_(rows, cols)] = sign_of_class[int(classes[c])] * S[c]
        return D

    tree = {"gens": gens, "words": words, "parent": parent, "spf": spf, "ident": ident,
            "nstate": nstate, "T_pos": T_pos, "au_of": au_of}

    def _propagate(Dgen, DT):
        """Propagate the generators along the word tree (:func:`_propagated`)."""
        return _propagated(Dgen, DT, **tree)

    # ---- 5. the covariance battery ----------------------------------------
    k_pts = _covariance_battery(group, k_pts, n_battery, seed)
    bloch = _bloch.evaluator(model, params)
    H = np.asarray(bloch(k_pts))
    if tol is None:
        # the rounding this representation's own arithmetic carries, measured
        # on the battery it is gated on, and never below the exact floor
        tol = covariance_gate(len(kept), nstate, float(np.abs(H).max()))
    Hrot, kact = {}, {}
    for p in kept:
        act = group.k_action_of(syms[p])
        kact[p] = act
        Hrot[p] = np.asarray(bloch(act(k_pts)))

    def _covariance(D):
        worst = 0.0
        for p in kept:
            U = D[p]
            src = H.conj() if syms[p].time_reversal else H
            worst = max(worst, float(np.abs(U @ src @ U.conj().T - Hrot[p]).max()))
        return worst

    def _gate_lift(p):
        r"""``[S per channel]`` the closed gate below measures against.

        It is the block :func:`_multiplet_blocks` settled on — the one the
        matrices were assembled from, so that the gate is the statement that
        every multiplet carries its own block and not a comparison against
        whichever member of the solution space a polar extraction happened to
        return — and the operation's own lift, the tables'
        (``sym.spinor_rotation``), on a one-dimensional multiplet no block was
        read for.  An entry is ``None`` where no block could be measured, and
        that multiplet is skipped: there is nothing to test it against.
        """
        lift = _lift_of(syms[p], su2)
        blocks = measured_of.get(int(syms[p].ind), {})
        return [blocks.get(int(c),
                           None if layout._chan_kind[c] == "measured" else lift)
                for c in range(nchan)]

    _gate_lifts = {p: _gate_lift(p) for p in u_pos}

    def _internal_defect(D):
        r"""Worst factorization defect ``max ‖B(g) − χ·S(g)‖``, over **every** block.

        The tensor structure behind the build is ``D(g) = O(g) ⊗ S(g)``, and
        that is a statement about every multiplet block of ``D(g)`` — the block
        carrying a multiplet to its *image* as much as the block of a multiplet
        the operation fixes.  This measures it on every one of them, with the
        per-block coefficient free (the tensor-structure subspace), so it is
        blind to the one-dimensional character the labelling puts in
        afterwards.  A candidate with a large defect carries no site
        characters at all: its blocks are some *other* SU(2) element, so "the
        ``d`` orbital at ``1a``" has nothing to be measured against.

        **Why every block and not only the fixed ones.**  Tested on the fixed
        multiplets alone the gate has a hole, and it is not a small one: the
        ``σ_h``-partner ``σ_h·g`` of an operation has a *different* lift, so a
        Hamiltonian written in the opposite spin convention can carry the
        partner's lift on every site-swapping block and still close a gate that
        never looks at one.  Measured: the SRSI paper's ``P41'`` model
        transcribed raw admits a faithful representation of ``P-4m2`` under the
        name ``P4mm1' #99.164`` — the two are isomorphic, so the product table
        closes at ``5e-16`` too — with ``C4``, ``C4⁻¹`` and both diagonal
        mirrors carrying ``S(σ_h)·S(g)``.  Closing the gate over every block
        refuses that build instead of returning it.

        This is also what separates the ``g``-versus-``g⁻¹`` orientation branch
        on a spinful model, and neither of the other two gates can see it — the
        covariance battery and the product table are both statements about the
        ``D``'s among themselves, while this one is the statement that they are
        the ``D``'s of *these* operations.  Measured on the search matrices: it
        is invariant under ``D → λD``, so normalization cannot change it.

        Vacuous where there is no internal factor to check: a spinless model
        with no declared spinor basis, whose ``S`` is ``1×1`` and every block a
        multiple of it by arithmetic.
        """
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

    # ---- 5b. normalization, as a function of one candidate ------------------
    # Phase-canonicalize the generators (canonical norm + antiunitary anchor +
    # positivity) and propagate along the same BFS tree.  Written as a function
    # of the candidate rather than of the winner so that the product-table gate
    # can run on the matrices the object will carry: `search["product"]`
    # and `SpaceGroupRep.product_residual` are then the same measurement rather
    # than two drafts of it.  Cheap enough because it runs only on candidates
    # that already closed the covariance and internal-factor gates.
    #
    # THE EXACT REPRESENTATION NEAREST THE MODEL'S OWN BLOCKS.  A wider
    # multiplet's block is the model's own, so a near-symmetric model's carries
    # its noise and the candidate composes by the table only to that noise --
    # no canonical norm can be read off it.  Where it misses by more than the
    # norm's own gate it is replaced by the exact representation nearest it
    # (:func:`.._matrices._settled`).  Only a miss the model's own noise explains is
    # projected -- within a hundred times the internal factor's tolerance,
    # which is the noise the build was told to expect: a candidate that misses
    # the table by a sign misses it by order one, and the product-table gate is
    # what refuses it.
    reach = min(1e-2, 100.0 * float(internal_tol))

    # ---- 5c. THE SETTLING, one coupled part at a time --------------------------
    # A part holding a one-dimensional multiplet is written in the tables' own
    # lift: its blocks are that lift, carried round each orbit with the signs
    # the hoppings give it, and a wider multiplet the hoppings couple to it is
    # put in the same phase (`_multiplet_blocks`), so the parts that hold one
    # are settled together, by one canonical norm.  A part of wider multiplets
    # alone carries its blocks as the model's own, read off one member of a
    # solution space, and so carries that member's free phase, which no other
    # part shares.  So each such part is settled on its own, and a crystal of
    # one part is one of them: its generators in their canonical norm and
    # anchor at its own lead multiplet (the anchor's for the anchor's own part),
    # the rest propagated, and -- wherever the signs positivity picks there do
    # not multiply: no antiunitary operation and a multi-dimensional lead with
    # no table lift behind it, or a one-dimensional lead whose character on its
    # site group no one-dimensional character of the group restricts to -- the
    # multiplication table choosing among the roots each canonical norm admits
    # (:func:`.._matrices._table_phases`).  Spin-1/2 states read in
    # single-valued tables arrive there: their matrices are a projective
    # representation, and where its class is trivial the phases that make it a
    # representation of the group are among these roots; a group with two
    # perpendicular twofolds, whose spinor matrices anticommute, has none, and
    # the refusal below says so.  The table anchor applies to a one-dimensional
    # lead in a group with no antiunitary operation (:attr:`RepAnchor.internal`):
    # there `sym.spinor_rotation` IS the internal factor, read from the shipped
    # datum rather than from what was assembled.  What is left on each part is
    # its one-dimensional character, which the labelling chooses.
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
        units.append({"states": np.asarray(where, dtype=int),
                      "lead": _lead_of(its, anchor),
                      "local": {a: i for i, a in enumerate(where)},
                      "anchor": 0 in held,
                      "orbitals": tuple(sorted(int(o) for c in its
                                               for o in layout._chan_orbs[c]))})

    def _part_anchor(unit):
        """The :class:`RepAnchor` of one part's generators, at the part's own lead."""
        lead, local = int(unit["lead"]), unit["local"]

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

    def _part_unsettled(bad, unit):
        """The refusal of a part whose own blocks settle into no representation, said as one."""
        if len(units) == 1:
            return bad
        return Refusal(
            bad.gate,
            f"the coupled part of the crystal holding orbitals {unit['orbitals']} -- the "
            f"multiplets the hoppings connect to one another and to nothing else -- was "
            f"settled on its own in {group.name} #{group.number_str}, and its own blocks do "
            f"not settle into a representation of the group: {bad.reason}",
            value=bad.value, threshold=bad.threshold,
            context=dict(bad.context or {}, part=unit["orbitals"]))

    def _settled_by_part(Dsearch):
        """``(refusal, D, normalization, anchor_T, DT_n, defect, roots)``, part by part.

        Each part is settled on its own by :func:`.._matrices._settled`, and by
        :func:`.._matrices._table_phases` where nothing anchors its generators' phases; the
        matrices are then the parts' side by side.  ``normalization`` is the
        anchor's part's, written on the whole matrices, and ``roots`` is
        ``(found, matrices)`` for the first part whose roots do not multiply.
        """
        eye_n = np.eye(nstate, dtype=complex)
        Dn = {p: np.zeros((nstate, nstate), dtype=complex) for p in Dsearch}
        anchor_n = None if T_pos is None else np.zeros((nstate, nstate), dtype=complex)
        DT_all = None if T_pos is None else np.zeros((nstate, nstate), dtype=complex)
        worst, norm_ref, roots = 0.0, {}, None
        for unit in units:
            idx = unit["states"]
            here = dict(tree, nstate=int(len(idx)))
            sub = {p: np.asarray(M, dtype=complex)[np.ix_(idx, idx)] for p, M in Dsearch.items()}
            bad, D_u, norm_u, anchor_u, DT_u, defect_u, roots_u = _settle_unit(
                sub, group=group, tree=here, kept=kept, u_pos=u_pos, prod=prod,
                anchor_of=_part_anchor(unit), reach=reach, product_tol=product_tol)
            worst = max(worst, float(defect_u))
            if bad is not None:
                return _part_unsettled(bad, unit), None, None, None, None, worst, None
            if roots is None:
                roots = roots_u
            for p, M in D_u.items():
                Dn[p][np.ix_(idx, idx)] = M
            if T_pos is not None:
                anchor_n[np.ix_(idx, idx)] = anchor_u
                DT_all[np.ix_(idx, idx)] = DT_u
            if unit["anchor"]:
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

    # ---- 6. the search ----------------------------------------------------
    phases = tuple(complex(z) for z in phases)
    n_free = len(gens) * len(free_cls)
    if orientation_branch is None:
        orients = (1, -1)
    elif isinstance(orientation_branch, bool):
        # a boolean is not a branch: True would read as +1 through int(), silently
        raise ValueError("orientation_branch must be None (both branches), +1 (the g "
                         "branch) or -1 (the g^-1 branch), not a boolean; got "
                         f"{orientation_branch!r}")
    elif int(orientation_branch) == 1:
        orients = (1,)
    elif int(orientation_branch) == -1:
        orients = (-1,)
    else:
        raise ValueError("orientation_branch must be None (both branches), +1 (the g "
                         "branch) or -1 (the g^-1 branch), got "
                         f"{orientation_branch!r}")

    # WHERE THE CHOICE APPLIES: every block of the operation at once.  A
    # relative character the Hamiltonian leaves open is assembled here as the
    # trivial one (`_relative`), the labelling choosing its part's own in
    # step 7b, and that is one statement about the whole matrix, not one per
    # site.  Two sites the Hamiltonian couples to each other not at all are as
    # free as two multiplets of one site, and `RealizedGroup.conserved` is
    # where the decoupling is said.
    all_blocks = [list(range(nchan))]

    # ---- 6a. the read-off ---------------------------------------------------
    # The detector has already solved `X H(k) = H(Rk) X` for every operation
    # this build represents, and `validated._bases` carries the whole solution
    # space.  A realizable tensor structure `D(g) = O(g) (x) S(g)` is an element
    # of it, so the discrete choices are a *measurement* on that space rather than a
    # search through their product set.  Only the generators are read: the
    # antiunitary operator carries the scalars the measurement of step 4 read
    # once, and no sign left to search, in both strategies, and every other
    # operation is propagated along the BFS tree.
    def _solution_space(p):
        """``(basis, note)`` — the detector's solution space for operation ``p``."""
        ind = int(syms[p].ind)
        B = getattr(validated, "_bases", {}).get(ind)
        if B is None or len(np.asarray(B)) == 0:
            U = validated._intertwiners.get(ind)
            if U is None:
                return None, (f"the validated group carries no measured intertwiner for "
                              f"generator #{ind}")
            # A hand-built ValidatedGroup may carry only the polar-extracted
            # member.  That IS the space when dim == 1, and when it is not the
            # projection below has nothing to choose from and the block defects
            # say so -- which is a fallback, not a wrong verdict.
            return np.asarray(U)[None, :, :], f"only `intertwiners[{ind}]`, not the whole space"
        B = np.asarray(B)
        if B.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES:
            return None, (f"the solution space of generator #{ind} is "
                          f"{B.shape[0]}-dimensional at nstate = {nstate}, whose "
                          f"projection onto the tensor structure would allocate more "
                          f"than {_READOFF_MAX_BYTES / 1e6:.0f} MB")
        return B, None

    def _read_off():
        """The discrete choices, measured off the solution spaces of the generators."""
        rec = {"n_enumeration": None, "branches": {}, "reason": None, "notes": []}
        branches, choices, reasons, read_of = {}, [], [], {}
        for orient in orients:
            per_gen, ratios, defects, combo, why = {}, {}, {}, [], None
            read_as = {}
            for gi, gp in enumerate(gens):
                ind = int(syms[gp].ind)
                basis, note = _solution_space(gp)
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
                # THE CHOSEN MEMBER.  Where the space leaves a block open, the
                # member taken is the one carrying the scalar `_relative` gives
                # there -- what the measurement pinned, or the trivial relative
                # character, the labelling being chosen in step 7b -- rather
                # than whichever direction a decomposition happened to return.
                want = {cls: _relative(gp, cls) for cls in range(ncls)}
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
                if float(sv_[0]) <= 1e-12:  # pragma: no cover - a guard, not a path
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
                # WHAT THE MEASUREMENT PINNED is taken from it: on the branch it
                # read, every orbit's ratio to the anchor, rather than read again
                # off the member above, which says only whether the tensor
                # structure holds
                if orient > 0:
                    for cls in range(ncls):
                        got = verdict["chosen"].get((ind, int(cls)))
                        if (int(cls) != anchor_cls and int(cls) not in unpinned
                                and got is not None):
                            signs[int(cls)] = complex(got)
                # An orbit the hoppings leave free of the anchor carries the
                # trivial relative character (`_relative`; the labelling
                # chooses its part's own in step 7b); the other orbits of its
                # part are pinned to its first by the hoppings, at the ratio the
                # measurement read between them.
                led = sorted(unpinned, key=lambda c: (free_lead.get(int(c), int(c)) != int(c),
                                                      int(c)))
                for cls in led:
                    lead_cls = free_lead.get(int(cls), int(cls))
                    rel = (verdict.get("within", {}).get((ind, int(cls))) if orient > 0
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
                    # THE FACTOR AS MEASURED, whatever it is: a root of unity
                    # where it is one within the noise, the measured phase
                    # otherwise; the three gates say whether it builds a
                    # representation
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
        return {"choices": choices, "branches": branches,
                "reason": rec["reason"], "record": rec, "read": read_of}

    def _build(choices):
        """One candidate record per ``(orientation, combo)``, covariance measured."""
        out = []
        for orient, combo in choices:
            sign = {}
            for gi in range(len(gens)):
                sgn = np.ones(ncls, dtype=complex)
                for j, c in enumerate(free_cls):
                    sgn[c] = combo[gi * len(free_cls) + j]
                sign[gi] = sgn
            Dgen = [_assemble(gens[gi], S_gen[gens[gi]], sign[gi]) for gi in range(len(gens))]
            DT = None
            if T_pos is not None:
                DT = _assemble(T_pos, S_gen[T_pos], reversal)
            if orient < 0:
                Dgen = [d.conj().T for d in Dgen]
                DT = None if DT is None else DT.conj().T
            D = _propagate(Dgen, DT)
            out.append({"orientation": int(orient), "combo": tuple(combo),
                        "residual": _covariance(D), "matrices": D})
        return out

    # ---- 6b. the joint gate -------------------------------------------------
    # Three independent measurements, each of which a candidate must pass, run
    # cheapest first so the expensive ones see the fewest candidates:
    #
    #   covariance      |kept| * n_battery triple products of size nstate;
    #                   kills a wrong relative sign inside a Wyckoff orbit.
    #   internal factor |kept_u| * nchan block proportionalities of size nspin
    #                   -- EVERY block, the site-swapping ones too; kills the
    #                   g-versus-g^-1 orientation branch, which is exactly
    #                   covariant AND exactly multiplicative, and a build whose
    #                   swapped blocks carry the sigma_h-partner's lift, which
    #                   is exactly covariant and multiplicative too whenever the
    #                   layer group and the one being named are isomorphic.
    #   product table   one normalize_rep pass + |kept|**2 products of size
    #                   nstate; kills a set that is not a representation.
    #
    # The ordering is a cost choice only -- the gates are independent, and which
    # one fired is recorded per candidate, which is physics rather than
    # accounting.  Written as a function of the candidate *set* so that the
    # read-off's one-or-two candidates and the enumeration's thousands go
    # through the identical arithmetic: the read-off finds the candidate, this
    # accepts it, and nothing here knows which produced the input.
    def _gate(records):
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
            grp[0]["internal"] = _internal_defect(grp[0]["matrices"])
        consistent = [grp for grp in covariant if grp[0]["internal"] <= internal_tol]
        for grp in consistent:
            rec = grp[0]
            # each coupled part settled on its own (step 5c), a crystal of one
            # part being one
            bad, Dn, norm, anchor_T, DT_n, defect, roots = _settled_by_part(rec["matrices"])
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
            # A PART'S OWN PHASE IS ITS SETTLING'S, not a choice: candidates
            # that differ only in the phase a part is settled out of are one
            # representation once each part is settled, and are merged as the
            # identical matrices are above
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

        return {
            "records": records, "distinct": distinct, "covariant": covariant,
            "consistent": consistent, "survivors": survivors,
            "search": {"n_candidates": len(records), "n_distinct": len(distinct),
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
                       "candidates": candidates}}

    # ---- 6b'. the orientation branch, pinned by convention when the gates tie
    # The internal-factor gate separates the two branches whenever
    # `S(g)^dag != +-S(g)` -- every rotation of order three or more -- because
    # the `g^-1` candidate carries `S(g)^dag` where the group says `S(g)`.  For a
    # group of involutions only (`mm2`, `P-1`, a `2/m` on a spinful model) the
    # lift satisfies `S(g)^dag = -S(g)`, the free per-block coefficient of that
    # gate absorbs the sign, and all three gates are blind to the branch.  There
    # is then nothing left to measure, and refusing would only demand a
    # convention the build is in a better position to state: when the two
    # branches' candidates both close all three gates AND are one representation
    # up to a one-dimensional character on each coupled part of the crystal --
    # so that the same labellings, and the same site irreps, are open on both --
    # the read-off's own branch `+1` -- the intertwiner that carries `k -> gk` --
    # is returned, and `branch_tie` says the choice was a convention rather than
    # a measurement.  `rep_not_unique` remains the verdict for everything else:
    # survivors that are not one representation up to such a character, more
    # than two of them, or two on the same branch.
    #
    # It runs on the candidate SET, before the strategy dispatch, for the same
    # reason `_gate` does: the read-off's two candidates are then accepted where
    # the tie resolves, instead of being handed to an enumeration that would
    # rediscover the identical pair -- and on a model whose enumeration is
    # refused (`rep_search_too_large`) that is the difference between a returned
    # representation and none at all.
    part_chans = [[int(c) for c in range(nchan) if int(classes[c]) in part]
                  for part in coupled_parts]

    def _one_labelling_apart(A, B, tol=1e-8):
        """Whether two settled candidates differ by one character on each coupled part.

        Asked of the unitary operations, which the site irreps and the band
        labels are characters of.
        """
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

    def _pin_branch(survivors):
        """``{survivors, tie, note}`` — the tie settled, or left alone."""
        out = {"survivors": survivors, "tie": False, "note": None}
        if len(survivors) != 2:
            return out
        per = {}
        for grp in survivors:
            per.setdefault(int(grp[0]["orientation"]), []).append(grp)
        if sorted(per) != [-1, 1] or any(len(v) != 1 for v in per.values()):
            return out
        if site_tables and not _one_labelling_apart(per[1][0][0]["normalized"][0],
                                                    per[-1][0][0]["normalized"][0]):
            return out
        out.update(survivors=[per[1][0]], tie=True, note=(
            "the two orientation branches both carry H(k) onto itself, factorize on every "
            "site into (character) x (the group's own internal factor) and reproduce the "
            "multiplication table, and are one "
            "representation up to a one-dimensional character on each coupled part, so "
            "the same labellings and site labels are open on both and nothing "
            "measures between them; the branch was pinned by convention to +1, "
            "the intertwiner that carries k -> gk."))
        return out

    # ---- 6c. the read-off, and the enumeration behind it --------------------
    n_enumeration = int(len(phases) ** n_free * len(orients))
    read = {"choices": (), "branches": {}, "reason": None, "read": {},
            "record": {"n_enumeration": n_enumeration, "branches": {}, "reason": None,
                       "notes": []}}
    if strategy == "readoff":
        try:
            read = _read_off()
        except np.linalg.LinAlgError as exc:  # pragma: no cover - not seen on any fixture
            # The read-off is a way to find the candidate faster; it must never
            # be the reason a build refuses.  An SVD that will not converge is a
            # reason to enumerate, not an error to return.
            read["reason"] = (f"the projection onto the tensor structure did not "
                              f"converge ({exc})")
    read["record"]["n_enumeration"] = n_enumeration
    outcome, strategy_used, fallback, pin = None, None, None, None
    declared_tol = float(getattr(validated, "tol", tol))
    if strategy == "readoff" and read["choices"]:
        outcome = _gate(_build(read["choices"]))
        pin = _pin_branch(outcome["survivors"])
        best = float(outcome["search"]["best"])
        if len(pin["survivors"]) == 1:
            strategy_used = "readoff"
        elif not outcome["covariant"] and declared_tol > tol and tol < best <= declared_tol:
            # WHAT WAS READ IS NEAR-SYMMETRIC: the representation assembled from
            # what the solution spaces measured at the declared tolerance closes
            # it and misses only the exact gate.  That is the verdict -- the
            # operations, or parts of the crystal coupled below the precision,
            # obeyed within what was declared and not exactly -- and no other
            # choice of signs, which the measurement never pinned, stands in for
            # it; the projection is the route.
            strategy_used = "readoff"
        else:
            # The read-off is a way to FIND the candidate, never to accept one:
            # a read that the gates do not confirm is handed back to the
            # enumeration of the phases on offer, whose refusal (or survivor) is
            # the one returned, and what was read is said with it.
            said = "; ".join(
                f"branch {orient:+d}: " + (", ".join(
                    f"generator #{ind} reads {_factor_said(z)} on orbit {c}"
                    for (ind, c), z in sorted(got.items())) or "no orbit read")
                for orient, got in sorted(read["read"].items(), reverse=True))
            fallback = (f"the factors measured between orbits of multiplets ({said}) "
                        f"do not give one representation: {len(outcome['survivors'])} of "
                        f"the {outcome['search']['n_distinct']} distinct candidates built "
                        "from them carry H(k) onto itself, factorize on every site into "
                        "(character) x (the group's own internal factor) and reproduce the "
                        "multiplication table (the candidates each test removed: "
                        f"{outcome['search']['killed_by']})")
            outcome, pin = None, None
    elif strategy == "readoff":
        fallback = read["reason"]

    if outcome is None:
        if n_enumeration > max_candidates:
            try:
                measured = float(n_enumeration)
            except OverflowError:  # pragma: no cover - past 2**1024 choices
                measured = float("inf")
            hint = ("" if fallback is None else
                    f" The read-off deferred to it because {fallback}, which is the "
                    f"thing to act on: a read that is not clean says the tensor "
                    f"structure D(g) = O(g) (x) S(g) does not describe this operation "
                    f"set on this orbital content, and enumerating it would not make it "
                    f"one.")
            return Refusal(
                "rep_search_too_large",
                f"the enumeration of this build's discrete choices would run "
                f"{n_enumeration:,} candidates -- {len(phases)} phases on "
                f"{len(gens)} generator(s) x {ncls - 1} free orbit(s) of multiplets of "
                f"{ncls}, times {len(orients)} orientation branch(es) -- against "
                f"max_candidates = {max_candidates:,}. At the few milliseconds a "
                f"candidate costs that is hours, so it is refused rather than run."
                f"{hint} A `multiplets=` stating a multi-orbital site as one multiplet "
                f"cuts the class count sharply, and `realized_group(..., "
                f"max_candidates=...)` with a larger ceiling runs the enumeration if it "
                f"is worth waiting for.",
                value=measured, threshold=float(max_candidates),
                context={"n_generators": len(gens), "n_classes": int(ncls),
                         "n_free": int(n_free), "n_candidates": n_enumeration,
                         "max_candidates": int(max_candidates),
                         "multiplets": layout, "fallback_reason": fallback})
        outcome = _gate(_build((orient, combo) for orient in orients
                               for combo in itertools.product(phases, repeat=n_free)))
        strategy_used = "enumerate"

    records, distinct = outcome["records"], outcome["distinct"]
    covariant, consistent = outcome["covariant"], outcome["consistent"]
    survivors = outcome["survivors"]
    spectrum = outcome["search"]["residuals"]
    int_spectrum = outcome["search"]["internal_residuals"]
    prod_spectrum = outcome["search"]["product_residuals"]

    # the branch tie of step 6b', measured on the candidate set that was accepted
    if pin is None:
        pin = _pin_branch(survivors)
    survivors = pin["survivors"]
    branch_tie, branch_note = pin["tie"], pin["note"]

    search_record = dict(outcome["search"])
    # The headline read-off numbers describe **the read that produced the
    # accepted representation** -- the surviving branch.  The losing branch's
    # numbers are a measurement of a candidate nobody kept (on a generator whose
    # multiplet permutation is not an involution the `g^-1` pattern is disjoint
    # from the `g` one, and its tensor-structure defect is the whole norm of the
    # solution),
    # so they stay in `search["readoff"]["branches"]` rather than standing for
    # the build.
    accepted = {}
    if strategy_used == "readoff" and len(survivors) == 1:
        won = int(survivors[0][0]["orientation"])
        accepted = {won: read["branches"][won]} if won in read["branches"] else {}
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
        "readoff": read["record"]})
    search = search_record
    if _search_record is not None:
        # the record of the search, handed back where asked beside a refusal,
        # which carries only the number of candidates tried
        _search_record["search"] = search
    # A declared spinor basis is a tensor structure the gates below test: a pairing that
    # is well-formed but wrong (partners swapped, up and down exchanged, the
    # wrong layout for this w90 file) is a different SU(2) placement, and it
    # dies here rather than in a dedicated gate.  Say so, or the model gets
    # debugged instead of the declaration.
    spin_hint = ("" if spin is None else
                 f" A `spin=` declaration is in force ({spin.kind!r} spinor basis on "
                 f"{spin.norb} orbitals); a wrong pairing -- partners swapped, up and down "
                 f"exchanged, or the wrong layout for this wannier90 file -- fails exactly "
                 f"here, so check the declaration against the file before the model.")
    # WHAT THE READ-OFF COULD NOT USE, said beside the enumeration's verdict:
    # what it read and why that was not a representation, or why it read
    # nothing, which is the measurement to act on
    n_ph = len(phases)
    read_hint = ("" if fallback is None else
                 f" The enumeration ran because {fallback}. The candidates counted above "
                 f"try {n_ph} phase{'' if n_ph == 1 else 's'} on each free orbit.")

    if not covariant:
        # The detector's own declared tolerance is on its record.  When it lies
        # ABOVE this gate and the best candidate lands between the two, the
        # operations are *near*-symmetric -- obeyed within the declared
        # tolerance, not to machine precision -- and that is the symmetrizing
        # signal rather than a modelling error.  Stating it is factual: both
        # numbers were measured, and neither gate moves.
        near_hint = ""
        declared = float(getattr(validated, "tol", tol))
        if declared > tol and spectrum[0] <= declared:
            near_hint = (
                f" The operations were measured at a declared tol = {declared:.3g}, above "
                f"the {tol:.3g} the representation is held to, and the best candidate's "
                f"{spectrum[0]:.3g} lies between the two: these "
                f"operations are near-symmetric -- obeyed within what was declared and not "
                f"to machine precision, or on parts of the crystal the measurement reads "
                f"apart that the hoppings couple only below it -- as a numerical "
                f"(wannier90) Hamiltonian gives. The route is to project H onto its "
                f"group-invariant part and build the representation on the symmetrized "
                f"model: `pythtqc.symmetrize` (in module "
                f"`pythtqc.symmetrization`), or "
                f"`realized_group(..., tol={declared:.3g})`, whose default "
                f"`symmetrize=\"auto\"` does it for you.")
        return Refusal(
            "rep_not_covariant",
            f"none of the {len(records)} candidate representations ({len(distinct)} distinct) "
            f"carries H(k) onto itself within the tolerance: the best reaches "
            f"{spectrum[0]:.3g} against {tol:.3g}. Either the operation set is not realized "
            f"on this orbital content (a slab operation, or one the Hamiltonian breaks "
            f"though a matrix with its action on k carries H(k) onto itself) or the tensor "
            f"structure is too narrow for the multiplets it was given — "
            f"a site whose orbitals mix under its site-symmetry group is one multiplet, "
            f"which is what the measured multiplets read off the solution spaces and "
            f"what a `multiplets=` states.{read_hint}{near_hint}{spin_hint}",
            value=float(spectrum[0]), threshold=float(tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept),
                     **({} if spin is None else {"spin_kind": str(spin.kind)})})
    if not consistent:
        return Refusal(
            "rep_internal_factor",
            f"{len(covariant)} of the {len(distinct)} distinct candidates carry H(k) onto "
            f"itself within {tol:.3g}, but on none of them does a site block factorize into "
            f"(character) x (the group's own internal factor): the best defect is "
            f"{int_spectrum[0]:.3g} against {internal_tol:.3g}. These matrices commute with H "
            f"the way the group says, but they are not the matrices of *these* operations — "
            f"nothing here has a site character to be read off it, so no orbital could be "
            f"declared or labelled.{read_hint}{spin_hint}",
            value=float(int_spectrum[0]), threshold=float(internal_tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept),
                     **({} if spin is None else {"spin_kind": str(spin.kind)})})
    if not survivors:
        stuck = [grp[0]["refusal"] for grp in consistent if "refusal" in grp[0]]
        if len(stuck) == len(consistent):
            # every surviving candidate failed to phase-canonicalize at all;
            # normalize_rep's own diagnosis is sharper than anything said here.
            return replace(stuck[0], reason=stuck[0].reason + read_hint) if read_hint \
                else stuck[0]
        # WHY NO PHASE HELPS, where the matrices say so.  The candidate that came
        # closest carries, where nothing anchored its phases, whether its matrices
        # form a projective representation at all and whether any roots the
        # canonical norms admit made it multiply.  Projective, and none did: two
        # commuting operations whose matrices commute only up to a phase the
        # table does not carry are the gauge-free statement of it.
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

            name_a = _record_name(syms[a], group.lattice)
            name_b = _record_name(syms[b], group.lattice)
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
                f"itself within the tolerance and factorize on every site into (character) x "
                f"(the group's own internal factor), and their matrices form a projective "
                f"representation of {named}: every product D(g1)D(g2) is a phase times "
                f"D(g1 g2). No choice among the phases the canonical norms admit turns those "
                f"phases into the group's multiplication table -- the best product-table "
                f"defect is {prod_spectrum[0]:.3g} against {product_tol:.3g} -- because the "
                f"class is not the table's. Operations #{syms[a].ind} ({name_a}) and "
                f"#{syms[b].ind} ({name_b}) commute, the table asks "
                f"D(a)D(b) = {asked:+d} D(b)D(a)"
                + (" (spinor factors included)" if bool(group.spinor) else "")
                + f", and on these states D(a)D(b) = {_fmt_char(asked * c)} D(b)D(a). A phase "
                f"on either matrix cancels from that, so no rephasing removes it, and no "
                f"representation of {named} with its own factor system exists on these states."
                + (f" It is the sign the spin-1/2 lifts of these two operations carry: the "
                   f"states carry half-integer spin, and no single-valued representation of "
                   f"this group exists on them. A spinor basis stored in the orbital index -- a "
                   f"wannier90 import -- is read in the double group once its spin partners are "
                   f"declared (`spin=`)." if spin_half else "") + read_hint,
                value=float(prod_spectrum[0]), threshold=float(product_tol),
                context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                         "indices": tuple(int(syms[p].ind) for p in kept),
                         "commuting_operations": (int(syms[a].ind), int(syms[b].ind)),
                         "commutation_phase": c,
                         "spin_half": bool(spin_half)})
        return Refusal(
            "rep_not_multiplicative",
            f"{len(consistent)} of the {len(distinct)} distinct candidates carry H(k) onto "
            f"itself within the tolerance and factorize on every site into (character) x "
            f"(the group's own internal factor), but none of them reproduces the group's "
            f"multiplication table (spinor factors included): the best product-table defect is "
            f"{prod_spectrum[0]:.3g} against {product_tol:.3g}. A Hamiltonian-covariant set of "
            f"matrices that does not multiply is not a representation of this group — the "
            f"operation set, the generators or the internal blocks are wrong, not the phases."
            f"{read_hint}",
            value=float(prod_spectrum[0]), threshold=float(product_tol),
            context={"n_candidates": int(search["n_candidates"]), "multiplets": layout,
                     "indices": tuple(int(syms[p].ind) for p in kept)})
    if len(survivors) > 1:
        branches = sorted({int(g[0]["orientation"]) for g in survivors})
        orient_hint = (
            "" if branches != [-1, 1] or len(survivors) != 2 else
            " The two are the two orientation branches, which is the one tie this "
            "function settles by convention when it can — but only when both return "
            "the same site labels under the same labelling, and these "
            "do not, so the choice is a genuine second representation rather than a "
            "gauge.")
        return Refusal(
            "rep_not_unique",
            f"{len(survivors)} distinct representations of {len(records)} candidates carry "
            f"H(k) onto itself within {tol:.3g}, factorize on every site into (character) x "
            f"(the group's own internal factor) within {internal_tol:.3g} and reproduce the "
            f"multiplication table within {product_tol:.3g}; the discrete choice is not "
            f"pinned. Either the "
            f"k-point grid the covariance is measured on is too small (it has "
            f"{k_pts.shape[0]} k-points) or the model carries more symmetry than the group does."
            f"{orient_hint}{read_hint} Open an issue on pythtqc's GitHub with the model "
            "and the call: the measurement should have fixed this choice.",
            value=float(len(survivors)), threshold=1.0,
            context={"n_candidates": int(search["n_candidates"])})
    winner = survivors[0][0]
    search["internal"] = float(winner["internal"])
    search["product"] = float(winner["product"])
    search["composition"] = float(winner.get("composition", 0.0))

    # ---- 7. the winner's normalization, already measured by the gate -------
    D, normalization, anchor_T, DT_n = winner["normalized"]

    # ---- 7b. the labelling, chosen once, and put in --------------------------
    # Everything above is blind to D(g) -> chi(g) D(g) on a coupled part, with
    # chi a one-dimensional character, and to the representation a conserved
    # pseudospin's copies carry: the hoppings fix how the orbitals transform
    # relative to one another, not what they are.  The multiplets' characters
    # are read once on the representation as settled
    # (:func:`_multiplet_characters`); one chooser takes each part's candidates
    # -- what its declarations leave -- to one labelling (:func:`_labelling`);
    # and it is put in here (:func:`_assembled`), so that the matrices returned
    # are the representation induced from the returned site content in the
    # tables' convention, spin up along +z in a right-handed frame.  The record
    # is then written off what was read and chosen (:func:`_atomic_limit_table`),
    # nothing measured again.
    limits = None
    if site_tables:
        raw, worst = _multiplet_characters(D, syms, u_pos, cperm, layout, prod)
        measured_at = {}
        for r in verdict["rows"]:
            measured_at.setdefault(tuple(int(o) for o in r["orbitals"]), set()).add(
                int(r["ind"]))
        for j in tuple(verdict.get("joint", ())) + tuple(verdict.get("by_sectors", ())):
            measured_at.setdefault(tuple(int(o) for o in j["decl"]["orbitals"]), set()).update(
                int(g) for row in j["theta"].values() for g in row)
        choice = _labelling(D, raw, decls, pool=pool, model=model, group=group,
                            assignment=assignment, layout=layout, params=params,
                            syms=syms, u_pos=u_pos, cperm=cperm, classes=classes,
                            parts=coupled_parts, ref_chan=ref_chan,
                            site_tables=site_tables, measured_at=measured_at,
                            multiplicity=measured_of.multiplicity)
        D = _assembled(D, choice, syms, layout)
        limits = _atomic_limit_table(choice["raw"], worst, decls, choice, group=group,
                                     assignment=assignment, model=model, layout=layout,
                                     syms=syms, kept=kept, checks=verdict["rows"],
                                     pool=pool, site_tables=site_tables, ident=ident,
                                     note=char_note, spin=spin, ref_chan=ref_chan)
        # A CONSERVED PSEUDOSPIN'S CHOICE, on the record: the site content
        # chosen for its copies, and every content the group's table admits
        # beside it, in the order the key ranks them.
        if choice["spaces"]:
            limits = _with_multiplicity_spaces(limits, choice["spaces"], group, assignment)

    # ---- 8. everything the object claims, re-measured ----------------------
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
            # the time reversal returned, the labelling's own included
            Tn = D[T_pos] if T_pos in D else (DT_n if DT_n is not None else anchor_T)
            anchor_res[ind] = float(np.abs(Tn @ np.conj(U) @ Tn.conj().T - U).max())
    # THE REPRESENTATION RETURNED closes the gate its candidates closed: every
    # labelling it is chosen among is one the hoppings leave open, a symmetry of
    # H, so one that misses the gate is the package disagreeing with itself
    worst_ind = max(residuals, key=residuals.get) if residuals else None
    if worst_ind is not None and residuals[worst_ind] > tol:
        raise InternalError(
            f"the representation of {group.name} #{group.number_str} with its labelling in "
            f"it does not carry H(k) onto itself within the tolerance its candidates met: "
            f"{residuals[worst_ind]:.3g} against {tol:.3g}, on operation {worst_ind}. Every "
            "labelling chosen among is a "
            "representation the hoppings leave open, a symmetry of the Hamiltonian; one "
            "that is not is a defect in this package, not a tolerance to loosen.")

    product_residual = _product_table_residual(D, kept, syms, prod, spf)

    # The Bloch operators' factor system and each orbit's band representation
    # are what the representation RETURNED carries, measured once its matrices
    # are final.  A representation built as the operators of a projection is
    # never returned, and carries neither (`_returned=False`).
    factor_residual = float("nan")
    try:
        if _returned:
            factor_residual = 0.0
            per = list(group.periodic_dirs)
            for a in kept:
                for b in kept:
                    c = int(prod[a, b])
                    if c not in D:
                        continue
                    kb = kact[b](k_pts)
                    Ua = bloch_operator(model, group, assignment, syms[a], D[a], kb)
                    Ub = bloch_operator(model, group, assignment, syms[b], D[b], k_pts)
                    Uc = bloch_operator(model, group, assignment, syms[c], D[c], k_pts)
                    lhs = Ua @ (np.conj(Ub) if syms[a].time_reversal else Ub)
                    delta = (np.asarray(syms[a].translation)
                             + np.asarray(syms[a].rotation, dtype=float)
                             @ np.asarray(syms[b].translation)
                             - np.asarray(syms[c].translation))
                    omega = int(spf[a, b]) * np.exp(-2j * np.pi
                                                    * (kact[c](k_pts) @ delta[per]))
                    factor_residual = max(factor_residual, float(
                        np.abs(lhs - omega[:, None, None] * Uc).max()))
    except Exception:  # pragma: no cover - the check must never mask what was built
        factor_residual = float("nan")

    indices = tuple(int(syms[p].ind) for p in sorted(kept))
    choice = {"orientation": int(winner["orientation"]),
              "signs": {(int(syms[gens[gi]].ind), int(c)):
                        complex(winner["combo"][gi * len(free_cls) + j])
                        for gi in range(len(gens)) for j, c in enumerate(free_cls)}}
    built = SpaceGroupRep(
        model=model, group=group, assignment=assignment,
        indices=indices,
        _operations=tuple(syms[p] for p in sorted(kept)),
        matrices={int(syms[p].ind): D[p] for p in kept},
        antiunitary=tuple(int(syms[p].ind) for p in sorted(au_pos)),
        residuals=residuals, _norm_residuals=norm_res, _anchor_residuals=anchor_res,
        product_residual=float(product_residual), factor_residual=float(factor_residual),
        _generators=tuple(int(syms[p].ind) for p in gens),
        _words={int(syms[p].ind): w for p, w in words.items()},
        _choice=choice, _search=search, _classes=classes,
        _reference=dict({"orbital": ref_orb, "multiplet": ref_chan,
                         "orbit": int(classes[ref_chan])},
                        **({} if spin is None else
                           {"spin_pair": int(spin.pair_of_orbital[ref_orb])})),
        _failed=tuple(validated.discrepancies),
        _battery=k_pts, multiplets=layout, _normalization=normalization,
        _time_reversal=None if T_pos is None else int(syms[T_pos].ind),
        characters=limits, spin=spin, _branch_tie=bool(branch_tie))
    return _with_tables_unavailable(_with_band_representations(built) if _returned
                                    else built)


def _with_multiplicity_spaces(table, chosen, group, assignment):
    """``table`` with each conserved pseudospin's chosen content on it, and its rows marked.

    ``chosen`` is the chooser's record of each pseudospin (:func:`_labelling`):
    the site content returned at the orbit's lead multiplet, and every content
    the representations of its copies give there, in the order the key ranked
    them, each named against the same site table.  A row of such an orbit that
    nothing else left open reads ``lowest-l`` wherever the representations a
    declaration leaves gave more than one content: its site irrep rests on the
    choice.  A pseudospin whose copies a declaration speaks for is determined
    by it.
    """
    wyckoff = _wyckoff_sites(group, assignment)
    spaces, marked = [], set()
    for got in chosen:
        rec = got["record"]
        site = int(rec["site"])
        tab = table.site_tables.get(site)
        here = wyckoff.get(site)
        # how many Wannier functions of the copies one site holds: its orbitals,
        # or its declared spinor pairs
        on_site = set(int(o) for o in assignment.site_orbitals[site])
        held = sum(1 for o in rec["orbitals"] if int(o) in on_site)
        spaces.append(MultiplicitySpace(
            orbitals=tuple(int(o) for o in rec["orbitals"]),
            wyckoff="" if here is None else f"{int(here.multiplicity)}{here.letter}",
            dim=int(rec["dim"]), content=_named_content(tab, got["content"])[0],
            contents=tuple(_named_content(tab, other)[0] or "(unnamed)"
                           for other in got["contents"]),
            n_representations=int(rec["n_admitted"]),
            determined_by=("declaration" if got.get("declared") else
                           "lowest-l" if int(rec["n_admitted"]) > 1 else "hamiltonian"),
            _per_site=held // (2 if table.spin is not None else 1),
            _pairs=table.spin is not None))
        if int(got.get("open", len(got["contents"]))) > 1:
            marked |= {int(o) for o in rec["orbitals"]}
    rows = tuple(replace(o, determined_by="lowest-l")
                 if o.orbital in marked and o.determined_by == "hamiltonian" else o
                 for o in table.orbitals)
    return replace(table, orbitals=rows, multiplicity_spaces=tuple(spaces))


def _with_band_representations(rep):
    """``rep`` with each orbit's band representation on its atomic-limit table.

    That band representation is read off the representation itself — the
    content its states carry — so it is written once the matrices are
    (:func:`..srsi._band_representations`).
    """
    table = rep.characters
    if table is None:
        return rep

    orbits = _srsi._band_representations(rep)
    if not orbits:
        return rep
    return replace(rep, characters=replace(table, band_representations=orbits))


def _with_tables_unavailable(rep):
    """``rep`` with every row saying the tables were unavailable, where they were.

    A group whose cell ``irrep`` could not match to the tables' standard
    setting is returned without tables (``GroupData._tables_ready``
    false on a primitive cell), so no row of it takes a name from them.  Each
    row's :attr:`AtomicLimitRow.note` then says so, and why, in the words of the
    group's own note, which carries ``irrep``'s message.
    """
    table, group = rep.characters, rep.group
    if (table is None or bool(group._tables_ready)
            or int(getattr(group, "_lattice_index", 1) or 1) > 1):
        return rep
    why = (f"no site irrep here is named from the tables: they are unavailable for "
           f"{group.name} #{group.number_str} -- "
           + (group._setup_note or "irrep could not match this cell to the BCS standard "
                                  "setting"))
    rows = tuple(replace(o, note=(o.note + "; " if o.note else "") + why)
                 for o in table.orbitals)
    return replace(rep, characters=replace(table, orbitals=rows))


#: What :func:`_build_rep` measures its covariance gate with when it is not told
#: otherwise -- its battery, and the operations it represents -- read off its
#: own signature rather than retyped.
_BUILD_DEFAULTS = {name: p.default
                   for name, p in inspect.signature(_build_rep).parameters.items()
                   if p.default is not inspect.Parameter.empty}

#: Every keyword :func:`_build_rep` takes, read off its own signature once, so that
#: the words handed to the build by name are checked against the build itself.
_BUILD_KEYWORDS = frozenset(name for name, p in inspect.signature(_build_rep).parameters.items()
                            if p.kind is p.KEYWORD_ONLY)


def _exact_gate(model, group, validated, build_kw=None, params=None) -> float:
    r"""The covariance gate :func:`_build_rep` applies to this model and operation set.

    The same :func:`covariance_gate`, measured the way the build measures it:
    ``|G|`` the operations it represents, ``N`` the model's states and
    ``max|H|`` on its battery.  ``build_kw`` are the build's own keywords, of
    which ``operations=``, ``include_vacuum_ops=``, ``k_pts=``, ``n_battery=``
    and ``seed=`` bear on the gate, and ``params`` the model's parameter values.
    A detection record carries neither ``N`` nor ``max|H|``, so
    :func:`..groups.realized_group` and :func:`..symmetrization.symmetrize`
    measure the gate here and classify operations against it.
    """
    kw = dict(build_kw or {})

    def _given(name):
        return kw[name] if name in kw else _BUILD_DEFAULTS[name]

    syms = {int(s.ind): s for s in group.symmetries}
    ops = _given("operations")
    indices = [int(i) for i in (validated.indices if ops is None else ops)]
    padded = bool(_given("include_vacuum_ops"))
    n_operations = sum(1 for i in indices
                       if padded or not _moves_padded(syms[i], group.vacuum_dirs))
    k_pts = _covariance_battery(group, _given("k_pts"), _given("n_battery"), _given("seed"))
    H = np.asarray(_bloch.hamiltonian(model, k_pts, params))
    return covariance_gate(n_operations, int(model.nstate), float(np.abs(H).max()))
