"""The space group a Hamiltonian obeys: the group of the cell, a verdict on each
of its operations, and the largest consistent subgroup of the survivors,
carried by a generic-orbit cell and identified again, round after round until
it settles, with the centres made exact under a declared precision; and the
record :func:`realized_group` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
import warnings

import numpy as np

from . import _bloch
from .refusal import InternalError, MagneticScopeWarning, Refusal
from ._tolerance import (
    DEFAULT_FINDER, Tolerances, _CLASS_MARGIN, _EXACT_CLOSURE, finder_distance,
    within)
from ._text import _DOUBLE_GROUP, _PER_COSET
from ._spacegroup import (
    GroupData, OrbitalAssignment, _OWN_ORBITALS, _PACKAGE_DIR, _group_of, _pad_cell,
    _sites_and_typat)
from ._seitz import OperationTable, _record_name, operations
from ._tables import _assign_orbitals
from ._subgroups import (
    _descent_closure, _operation_key, _resolve_closure, _subgroups_inside,
    _tie_ranked, _tie_reason, _tie_rule_words)
from ._matrices import _resolve_spin, _spin_pair_split
from ._grammar import _declaration_grammar, _names_joint_keys
from ._centres import (
    _cell_sites, _frame_defect, _geometry, _moved_extras, _moves_of,
    _rational_offset, _site_action, _symmetric_sites)
from .symmetry import _content_by_position
from ._sectors import read_sectors, sector_action
from ._multiplets import _site_multiplets
from ._declaration import _declaration_breaks
from ._layers import (
    DECORATION_DZ, SLAB_C, _decorated_cell, _environment_record,
    _is_presented_as_a_slab, _raise_environment_contradiction, _slab_presentation)
from ._labelling import _held_sets, _parts_apart
from ._measure import (
    ValidatedGroup, _MeasurementContext, _grid_covariance, _max_hamiltonian,
    _own_form_read, _validate_group)
from ._build import _BUILD_KEYWORDS, _build_rep, _exact_gate
from .symmetrization import (
    EXACT_GATE, _held_off, _memory_budget, _parts_coupling, _projection_operators,
    _search_ceiling, _symmetrized as _symmetrize, _with_centres, classify_operations)
from ._descent import (
    _DROPPED_BY_WORDS, _antiunitary_settled, _beyond_grey, _carried_onto,
    _carried_reduction, _centres_not_exactly_symmetric, _declared_cell,
    _declared_cell_reduction, _declared_contents, _declared_contents_wider,
    _declared_parts_of, _descend_once, _descent_clauses, _descent_counts,
    _lattice_note, _layer_declared_reading, _match_operations, _operation_verdicts,
    _precision_holding, _rekey_contents, _restricted_contents, _restricts_to,
    _shared_origin, _supercell_refusal, _time_reversal_measured,
    _time_reversal_member, _time_reversal_square)

__all__ = ["RealizedGroup", "realized_group"]


@dataclass(frozen=True)
class RealizedGroup:
    """The space group a Hamiltonian obeys, with each orbital's Wyckoff position and the representation that labels the bands.

    What :func:`realized_group` returns.  ``summary()`` prints what was
    measured, each line ending with the attribute that holds it, and
    ``model, group, assignment, validated, rep = r`` unpacks the five fields
    every later call takes; the rest are read by name.

    Attributes
    ----------
    model : pythtb.TBModel
        The model every later call takes: a layer's slab, or ``H`` symmetrized.
    source : pythtb.TBModel
        The model as it was handed in.
    group : GroupData
        The space group the Hamiltonian obeys, every operation measured to hold.
    operations : OperationTable
        The group's operations by index, with their Seitz symbols and names.
    assignment : OrbitalAssignment
        Each orbital's site and Wyckoff position, and how operations move sites.
    validated : ValidatedGroup
        The verdict on each operation of the group, and how well it holds.
    rep : SpaceGroupRep
        The matrices that label the bands; ``rep.characters``, the site irreps.
    conserved : dict
        Symmetry beyond the crystal's: ``dim`` more matrices commute with ``H``.
    descent : dict or None
        The operations of the cell ``H`` did not keep, and what dropped each.
    time_reversal : dict
        How time reversal was measured: ``"kept"`` or ``"measured broken"``.
    beyond_grey : dict or None
        Operations with time reversal that hold while time reversal is broken.
    environment, environment_declared : str or None, and bool
        A layer's ``"substrate"`` or ``"free-standing"``, and whether declared.
    cross_check : dict or None
        A layer's other environment, measured and compared with this one.
    precision : dict or None
        The tolerances a declared ``precision`` set against ``H``'s magnitude.
    positions : dict or None
        How far the orbital centres were moved onto the group's positions.
    symmetrization : Symmetrization or None
        How far ``H`` was moved onto its symmetric part, and why.
    decoration : dict or None
        What was added to the cell; in ``declared``, its reduction by content.
    presentation : dict
        How a layer was set in a slab, and the check that ``H(k)`` is unchanged.
    cell, typat, cell_typat : pythtb.TBModel and list
        The point set the group was found on; the orbitals' and its species.
    bundle : tuple
        ``(model, group, assignment, validated, rep)``, what unpacking gives.
    """

    model: Any
    source: Any
    cell: Any
    typat: list
    cell_typat: list
    group: GroupData
    assignment: OrbitalAssignment
    validated: ValidatedGroup
    rep: Any
    presentation: dict = field(default_factory=dict)
    decoration: dict | None = field(default_factory=dict)
    time_reversal: dict = field(default_factory=dict)
    descent: dict | None = None
    beyond_grey: dict | None = None
    symmetrization: Any = None
    positions: dict | None = None
    precision: dict | None = None
    environment: str | None = None
    environment_declared: bool = False
    cross_check: dict | None = None
    # the sectors the conserved matrices acting within sites split the states
    # into, read once where the model entered, and what the operations of the
    # group returned do to them: read on `conserved`
    _sectors: Any = field(default=None, repr=False, compare=False)

    @property
    def conserved(self) -> dict:
        """Symmetry beyond the crystal's own: the matrices that commute with ``H(k)`` at every ``k``.

        Returns
        -------
        dict
            ``dim``: how many such matrices there are beyond the identity, ``0``
            for a generic model and ``1`` for Kane–Mele without its Rashba
            term, which conserves ``s_z``; the labels, elementary band
            representations and indicators returned are the crystalline
            group's all the same, as the ``note`` says.  ``sectors``: how the
            conserved matrices acting within sites split the states, each
            sector naming its ``orbitals``, the ``sites`` they sit at, and its
            ``copies`` and ``dim``, that many identical copies of a set of that
            many orbitals, every matrix on the copies conserved; ``images``: per
            operation by index, the sector it carries each sector onto.  Both
            are ``None`` where the conserved matrices were too many to read.
            Where ``dim`` is above ``0`` a decoupled multiplet's content can be
            a choice, and the atomic-limit table's ``determined_by`` says
            whether a declaration or the lowest-angular-momentum rule made it.
        """
        ident = next((int(s.ind) for s in self.group.symmetries
                      if not s.time_reversal
                      and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                         np.eye(3, dtype=int))
                      and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)),
                     None)
        n = int((self.validated._nullspace_dims or {}).get(int(ident), 1) or 1) \
            if ident is not None else 1
        dim = max(0, n - 1)
        reading, action = self._sectors if self._sectors is not None else (None, None)
        sectors = images = None
        if reading is not None:
            site_of = np.asarray(self.assignment.site_of_orbital, dtype=int)
            sectors = []
            for sector in reading.sectors:
                orbitals = tuple(sorted(int(o) for a in sector.atoms
                                        for o in reading.atoms[int(a)]))
                sectors.append({"orbitals": orbitals,
                                "sites": tuple(sorted({int(site_of[o]) for o in orbitals})),
                                "copies": int(sector.copies), "dim": int(sector.dim)})
            sectors = tuple(sectors)
            images = ({} if action is None else
                      {int(i): tuple(int(j) for j in got)
                       for i, got in sorted(action.images.items())})
        return {"dim": int(dim),
                "note": (f"additional commuting matrices: {dim} -- this Hamiltonian "
                         f"commutes with {dim} matrix{'' if dim == 1 else 'es'} beyond "
                         "the identity at every k, so it has that much symmetry beyond "
                         "the crystal's own; the labels, EBR rows and indicators "
                         "returned here are the crystalline group's."
                         if dim else
                         "additional commuting matrices: 0 -- no matrix beyond the "
                         "identity commutes with this Hamiltonian at every k, so its "
                         "symmetry is the crystal's own."),
                "sectors": sectors, "images": images}

    @property
    def operations(self) -> OperationTable:
        """Every operation of :attr:`group` by index, with its Seitz symbol and name: ``r.operations[5].seitz``."""
        return operations(self)

    @property
    def bundle(self) -> tuple:
        """``(model, group, assignment, validated, rep)``, what unpacking the record gives."""
        return (self.model, self.group, self.assignment, self.validated, self.rep)

    def __iter__(self):
        """Unpack as ``model, group, assignment, validated, rep``."""
        return iter(self.bundle)

    def summary(self) -> str:
        """What was measured, a line for each part, with the attribute that holds it."""
        g = self.group
        where = {"substrate": ", on a substrate", "free-standing": ", free-standing"}.get(
            self.environment, "")
        lines = [f"{g.name} #{g.number_str}{where}   (.group)",
                 "  its operations, by index and Seitz symbol   (.operations)"]

        table = getattr(self.rep, "characters", None)
        by_position = (_content_by_position(table, int(self.model.nspin))
                       if table is not None else [])
        if by_position:
            lines.append("  the orbitals, by Wyckoff position   (.rep.characters)")
            lines += ["    " + line for line in by_position]

        tr = self.time_reversal or {}
        if tr.get("outcome") == "measured broken":
            square = tr.get("square") or {}
            if (square.get("asked", 0) > 0 and np.isfinite(square.get("measured", np.nan))
                    and square["measured"] < 0):
                lines.append("  time reversal broken: its square is -1, so the states carry "
                             "half-integer spin; spin= declares the spin pairs   "
                             "(.time_reversal)")
        if self.beyond_grey:
            names = [str(o.get("name", "")) for o in self.beyond_grey.get("operations", ())]
            shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6
                                            else "")
            lines.append(f"  also obeyed: {shown}, making a magnetic group of type "
                         f"{self.beyond_grey['magnetic_type']}, which is not returned   "
                         "(.beyond_grey)")

        moved = []
        if (self.positions or {}).get("adjusted") and not within(
                self.positions.get("max_move", 0.0), 0.0):
            moved.append(f"centres moved by {self.positions['max_move']:.2g}")
        if self.symmetrization is not None:
            moved.append(f"H symmetrized by {self.symmetrization.defect:.2g}")
        if self.precision:
            lines.append(f"  precision {self.precision['precision']:.3g}"
                         + (": " + ", ".join(moved) if moved else "")
                         + "   (.precision, .positions, .symmetrization)")
        elif moved:
            lines.append("  " + ", ".join(moved) + "   (.positions, .symmetrization)")

        dim = int((self.conserved or {}).get("dim", 0))
        if dim:
            lines.append(("  one more matrix commutes" if dim == 1
                          else f"  {dim} more matrices commute")
                         + " with H(k) at every k   (.conserved)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        tr = self.time_reversal or {}
        pr = self.precision or {}
        red = (self.decoration or {}).get("declared")
        # time reversal read single-valued, its square measured -1: the states
        # carry half-integer spin, and the line names what reads them so
        square = tr.get("square") or {}
        spinor = (square.get("asked", 0) > 0 and np.isfinite(square.get("measured", np.nan))
                  and square["measured"] < 0)
        # the other environment, as the cross-check left it
        cc = self.cross_check or {}
        if self.cross_check is None:
            compared = ""
        elif cc.get("agrees"):
            compared = (", the free-standing group #"
                        + str((cc.get("free_standing") or {}).get("number_str"))
                        if self.environment == "substrate" else
                        ", the substrate group #"
                        + str((cc.get("substrate") or {}).get("number_str"))) + " agrees"
        elif cc.get("outcome") == "not compared":
            compared = (", not compared with the free-standing layer: the declaration is "
                        "written in the substrate's site irreps"
                        if self.environment == "substrate" else
                        ", not compared with the substrate: the declaration is written in "
                        "the free-standing layer's site irreps")
        else:
            compared = ", the other environment " + str(cc.get("outcome"))
        # the operations a clause counts are coset representatives of the
        # lattice translations, and a spinful group's are the single-valued
        # group's: the first count of the line says so, once
        double = bool(getattr(self.group, "spinor", False))
        named = []

        def unit():
            if named:
                return "", ""
            named.append(True)
            return f" {_PER_COSET}", (f"; these are {_DOUBLE_GROUP}" if double else "")

        declared = ""
        if red:
            u, d = unit()
            declared = (f"; cell lowered by the declared content from "
                        f"#{red['from']['number_str']} ({red['n_excluded']} of "
                        f"{red['from']['size']}{u} excluded{d})")
        descended = ""
        if self.descent and len(self.descent.get("steps", ())) > 1:
            # a descent of several steps is counted whole, by cause, as its
            # note counts it
            counts = self.descent["dropped_by"]
            u, d = unit()
            descended = (f"; descended from #{self.descent['from']['number_str']} in "
                         f"{len(self.descent['steps'])} steps ({sum(counts.values())} of "
                         f"{self.descent['from']['size']}{u} dropped: "
                         + ", ".join(f"{n} {_DROPPED_BY_WORDS[cause]}"
                                     for cause, n in counts.items())
                         + f"{d})")
        elif self.descent:
            counts = self.descent["dropped_by"]
            n_all = sum(counts.values())
            u, d = unit()
            if set(counts) <= {"hoppings", "declaration", "centres"}:
                # every operation dropped was broken -- by the hoppings, the
                # declared content or the centres -- and the clause says so
                n_h, n_c, n_d = (int(counts.get(cause, 0))
                                 for cause in ("hoppings", "centres", "declaration"))
                parts = ([f"{n_h} by the hoppings"] if n_h else []) + (
                    [f"{n_c} by the centres"] if n_c else []) + (
                    [f"{n_d} by the declared content"] if n_d else [])
                why = ("" if not n_d and not n_c else
                       " by the declared content" if n_d == n_all else
                       " by the centres" if n_c == n_all else
                       ": " + ", ".join(parts))
                descended = (f"; descended from #{self.descent['from']['number_str']} "
                             f"({n_all} of {self.descent['from']['size']}{u} broken"
                             f"{why}{d})")
            else:
                # operations the Hamiltonian obeys are among them -- the states'
                # projective class, a closure, the tie rule -- so they were
                # dropped, not broken, and each cause is counted
                descended = (f"; descended from #{self.descent['from']['number_str']} "
                             f"({n_all} of {self.descent['from']['size']}{u} dropped: "
                             + ", ".join(f"{n} {_DROPPED_BY_WORDS[cause]}"
                                         for cause, n in counts.items())
                             + f"{d})")
        beyond = ""
        if self.beyond_grey:
            n_b = int(self.beyond_grey["n_operations"])
            u, d = unit()
            beyond = (f" -- a larger magnetic group, of type "
                      f"{self.beyond_grey['magnetic_type']}, is realized ("
                      + (f"{n_b} antiunitary operation{'' if n_b == 1 else 's'}" if not u
                         else f"{n_b} operation{'' if n_b == 1 else 's'} combined with "
                         f"time reversal,{u},")
                      + f" measured to hold{d}) but is beyond the validated scope")
        return (f"RealizedGroup({self.group.name} #{self.group.number_str}"
                + (f"; precision {pr['precision']:.3g} declared -> tol "
                   f"{pr['tol']:.3g} (cut {pr['cut']:.3g}, window "
                   f"{pr['window']:.3g}), "
                   f"position_tol {pr['position_tol']:.3g} "
                   f"(measured max|H| {pr['max_h']:.4g})"
                   + (f", symprec raised to {pr['symprec']:.3g}"
                      if pr.get("symprec_raised") else "")
                   if pr else "")
                + (("; time reversal measured broken"
                    # the grey group of the unitary operations kept is named only where
                    # it was the candidate: where antiunitary operations of the cell held
                    # with time reversal broken, they are what held, on `beyond_grey`
                    + ("" if self.beyond_grey else
                       f", descended from {tr['grey_name']} #{tr['grey_number_str']}")
                    + (": its square is -1, so the states carry half-integer spin, and "
                       "spin= declares which orbitals are spin partners" if spinor else ""))
                   if tr.get("outcome") == "measured broken" else "")
                + declared
                + descended
                + (f"; additional commuting matrices: {self.conserved['dim']}"
                   if self.conserved.get("dim") else "")
                + (f"; centres moved {self.positions['max_move']:.3g}"
                   if (self.positions or {}).get("adjusted")
                   and not within(self.positions.get("max_move", 0.0), 0.0)
                   else "")
                + (f"; symmetrized (defect {self.symmetrization.defect:.3g})"
                   if self.symmetrization is not None else "")
                + (("; on a substrate" if self.environment == "substrate"
                    else f"; {self.environment}")
                   + ("" if self.environment_declared else " (the default)")
                   + compared
                   if self.environment is not None else "")
                + beyond + ")")


def realized_group(model, *, typat=None, environment=None, cross_check=True,
                   decorations=None, dz=DECORATION_DZ,
                   c=SLAB_C, nk=6, tol=None, cut=None, window=None,
                   symmetrize="auto", max_bytes=1e9, position_tol=None,
                   precision=None, spinor=None, spin=None, descend=True,
                   symprec=DEFAULT_FINDER, params=None,
                   characters=None, multiplets=None, max_candidates=2 ** 20):
    r"""The space group obeyed by H, obtained by descent from the space group of the lattice of occupied sites.

    The call also places each orbital on its Wyckoff position and builds the
    representation of the group that labels the bands.

    Parameters
    ----------
    model : pythtb.TBModel
        A crystal periodic in three directions, or a layer periodic in two; a
        chain or a molecule is refused.
    typat : sequence, one entry per orbital, optional
        A species label per orbital: orbitals on atoms of different species
        get different labels.  ``None`` reads sites with the same orbital
        content as one species.
    characters : sequence or mapping, optional
        What the orbitals are: a function (``"s"``, ``"px"``, ``"dx2-y2"``), a
        shell (``"p"``), a site irrep (``"A1"``), with spin a site corep
        (``"-E1"``), or a sum of them (``"A1 + E"``), one per orbital, or a
        mapping from an orbital, or from several orbitals of one site, to such
        a word.  In place of a word, a mapping from operations of the site
        group to their characters, or to the matrices they act with on the
        orbitals named, in the order written: matrices say which basis of its
        site irrep a multiplet is.  The group loses the operations that do not
        carry the declared content.  An orbital left undeclared takes the
        lowest angular momentum the hoppings allow.
    environment : {"substrate", "free-standing"}, optional
        For a layer: on a substrate, no operation turns the layer over;
        free-standing, the whole layer group applies.  Default: the substrate
        for a flat layer, free-standing for a slab whose orbitals have heights.
    cross_check : bool, optional
        For a layer, also read the other environment and check that the two
        agree.  Default ``True``.
    decorations : sequence of (position, label), optional
        Non-orbital species added to the cell before its group is found: a
        crystal field the orbital positions do not show.
    precision : float, optional
        For a numerical model such as a wannier90 import, the relative
        precision of its numbers.  It sets the tolerances below; centres and
        hoppings symmetric within it are made exactly symmetric.
    spin : str, mapping or SpinBasis, optional
        For a spinless model whose orbital index carries spin, as a wannier90
        import's does: which two orbitals are the spin states of one Wannier
        function, ``"interleaved"``, ``"blocked"`` or pair by pair.  Left out,
        such a model is read spinless.
    spinor : bool, optional
        Whether the states carry spin one-half.  Default: from the model, and
        ``True`` with ``spin``.
    descend : bool, optional
        ``False`` returns the group of the lattice of occupied sites itself, and
        refuses ``cell_group_not_realized`` where H does not obey it.  Default
        ``True``.
    symmetrize : {"auto", True, False}, optional
        For operations or multiplets that hold only within the tolerance:
        ``"auto"`` (default) projects H onto its symmetric part where there are
        any, ``True`` always, ``False`` never, and such a model is then refused.
    tol, cut, window, position_tol, symprec : float, optional
        The tolerances ``precision`` sets, one by one: ``tol`` absolute, in the
        units of H; ``cut`` and ``window`` relative; ``position_tol`` in reduced
        coordinates; ``symprec`` Cartesian, default ``1e-5``.  Left out, each
        is exact.
    nk : int, optional
        k-points per periodic direction of the grid the operations are measured
        on.  Default 6.
    dz, c : float, optional
        A layer's presentation: the depth of the substrate's shadow sites and
        the height of the slab's cell.  Neither changes the group.
    multiplets : list or SiteMultiplets, optional
        Each site's multiplets, stated instead of measured: one list per site,
        each multiplet written as its orbitals.
    max_candidates, max_bytes : int and float, optional
        Limits on the search for the representation (default ``2**20``
        candidates) and on the memory of the symmetric projection (default
        ``1e9`` bytes).
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    RealizedGroup or Refusal
        ``r.summary()`` prints what was measured.  The record unpacks as
        ``model, group, assignment, validated, rep = r``: the group, its
        operations by Seitz symbol in ``r.operations``; each orbital's Wyckoff
        position; the verdict on each operation; and the representation, which
        labels bands (``hsp_table(r.rep, bands=n)``) and holds each orbital's
        site irrep (``r.rep.characters``).  By name: ``r.descent``, what was
        broken; ``r.time_reversal``; ``r.environment`` and ``r.cross_check``,
        for a layer; ``r.beyond_grey``, operations combined with time reversal
        that hold where time reversal does not; ``r.precision``,
        ``r.positions`` and ``r.symmetrization``, for a numerical model;
        ``r.conserved``, symmetry beyond the crystal's.  A ``Refusal`` in its
        place is falsy and names what stopped the call.

    Raises
    ------
    ..refusal.InternalError
        Two of the package's own measurements of one H disagree: a defect to
        report, with the model and the call.
    ValueError
        A malformed argument, before anything is measured.

    Warns
    -----
    MagneticScopeWarning
        Time reversal is broken; the group returned is the unitary one.

    Refusals
    --------
    Once the group is found, a refusal names it and carries it in
    ``context["group"]``.

    ``fewer_than_two_periodic_directions``
        a chain or a molecule.
    ``irrep_tables`` / ``single_valued_spinful``
        ``irreptables`` is not installed, or ``spinor=False`` was asked of
        states with spin one-half.
    ``spin_pair_split``
        the two orbitals of a declared spin pair sit at different centres.
    ``wyckoff_orbit``
        an orbital sits on no Wyckoff position of the cell.
    ``positions_not_exactly_symmetric``
        centres off symmetric positions by enough to break operations, with no
        ``precision`` declared.
    ``supercell``
        H obeys a translation shorter than the cell's: the model is written on
        a supercell.
    ``operations_not_nearly_consistent`` / ``operations_not_consistent``
        the operations found on a numerical cell do not close; ``precision``
        makes the centres exact first.
    ``cell_group_not_realized`` / ``declaration_breaks_group``
        with ``descend=False``, operations broken by the hoppings or by the
        declared content.
    ``unknown_label`` / ``ambiguous_multiplet``
        a declared word the site does not name, or a flat list that cannot say
        which orbitals form one multiplet.
    ``rep_not_covariant``
        with ``symmetrize=False``, operations that hold only within the
        tolerance.
    ``rep_internal_factor`` / ``rep_not_multiplicative``
        no candidate representation carries the group's phases, or multiplies
        as the group does.
    ``rep_not_unique``
        more than one representation fits; report it, with the model and the
        call.
    ``rep_double_valued_spinless``
        double-valued tables asked of a spinless model; ``spin=`` declares its
        spin pairs.
    ``rep_search_too_large`` / ``projection_too_large`` / ``detector_too_large``
        past ``max_candidates`` or ``max_bytes``; the reason gives the value
        that runs it.
    ``symmetrize_broken_operation`` / ``symmetrize_defect_exceeds_tol`` /
    ``symmetrize_range_not_closed``
        the symmetric projection's own (:func:`..symmetrization.symmetrize`).

    Examples
    --------
    >>> from pythtb.models import kane_mele                        # doctest: +SKIP
    >>> r = realized_group(kane_mele(1.0, -1.0, 0.3, 0.25), typat=[0, 1],
    ...                    characters={0: "A1"})                    # doctest: +SKIP
    >>> print(r.summary())                                          # doctest: +SKIP
    P3m11' #156.50, on a substrate   (.group)
      its operations, by index and Seitz symbol   (.operations)
      the orbitals, by Wyckoff position   (.rep.characters)
        1c (3m): orbital 0, -E1
        1b (3m): orbital 1, -E1
    """
    return _realize(model, typat=typat, environment=environment,
                    cross_check=cross_check, decorations=decorations, dz=dz, c=c,
                    nk=nk, tol=tol, cut=cut, window=window,
                    symmetrize=symmetrize, max_bytes=max_bytes, position_tol=position_tol,
                    precision=precision, spinor=spinor, spin=spin, descend=descend,
                    symprec=symprec, params=params, selection_only=False, context=None,
                    characters=characters, multiplets=multiplets,
                    max_candidates=max_candidates)


def _realized_selection(model, *, context=None, **kw):
    """The selection pass of :func:`realized_group`, without the representation.

    Everything up to and including the descent — the presentation, the cell, the
    identification, the orbital assignment, the detector and the symmetrizing
    pass — and then the record of it, keyed ``environment``, ``group``,
    ``validated``, ``assignment``, ``model``, ``descent``, ``parents`` (the
    group each descent step lowered), ``held`` (the operations each step's
    round measured to hold), ``positions``, ``time_reversal`` and
    ``unread`` (the refusal a declaration this environment's site groups name
    nothing for would have met, or ``None``).
    No representation is built and no cross-check is run.

    This is how :func:`realized_group` computes a 2D model's *other*
    environment: the two selections are independent, but they pose the same
    in-plane k-actions, so ``context`` hands the first selection's
    :class:`_MeasurementContext` down and the second reuses the
    Hamiltonian and the solution spaces the first one solved for, and
    ``tolerances`` hands down the :class:`.._tolerance.Tolerances` the first
    resolved, so that both judge the same Hamiltonian's noise against the same
    numbers.  Every other keyword is :func:`realized_group`'s own.
    """
    return _realize(model, selection_only=True, context=context, **kw)


def _raise_not_covariant(rep, group, validated, steps, *, where=""):
    r"""The consistency guard: a representation of the identified group that misses covariance.

    The group this loop settles on -- lowered by a descent, or the cell's own
    group realized outright -- rests on three measurements of what this
    Hamiltonian obeys: the detector's per-k-action-class verdict, the tensor
    structure of the operations it keeps -- which member of its class each
    undetermined one is, and on a spinful model whether each one it found
    obeyed carries its own spin lift -- and the covariance gate of the
    representation built on it.  So that gate has to close on the first
    attempt: the three disagreeing beyond the tolerance they were measured at
    is a defect in here, not a tolerance to loosen, and running a second descent
    on the gate's global verdict would be a search by trial rather than a
    measurement.  ``rep`` is the ``rep_not_covariant`` refusal, ``steps`` the
    descent that ran, and ``where`` says which build it came from where that is
    not the representation returned.
    """
    descent_clause = ""
    if steps:
        chain = " -> ".join([f"#{steps[0]['from']['number_str']}"]
                            + [f"#{st['to']['number_str']}" for st in steps])
        descent_clause = f", by the descent {chain}"
    raise InternalError(
        f"the group identified is {group.name} #{group.number_str}{descent_clause}, "
        f"every operation of it obeyed, and its representation{where} still does not "
        "carry H(k) onto itself within the tolerance: the best of "
        f"{rep.context.get('n_candidates', '?')} candidates "
        f"reaches {float(rep.value):.4g} against {float(rep.threshold):.3g}. "
        "Every operation of the group was measured to hold, each by a matrix of its "
        "own form, so a representation of them that misses is the package "
        "disagreeing with itself: a defect to report, not a tolerance to loosen. "
        "`realized_group(..., descend=False)` shows the group the cell allows.")


def _named_refusal(rep, group, validated=None) -> Refusal:
    r"""``rep``, with the group it was reached on named in its reason and carried in context.

    Every refusal :func:`realized_group` returns once the cell's group is
    identified opens with a group, so the one fact every later question needs
    is never lost in a refusal.  Once the measurement has settled it is the
    group identified -- its symbol and number, every operation of it measured
    to hold (``validated``); before that, the
    group the cell has at that point, an upper bound on what the Hamiltonian
    obeys (``validated`` left ``None``), since further rounds may still lower
    it.  ``context["group"]`` carries the group itself, ready for
    :func:`.._build._build_rep` to take by hand.

    A refusal made with its group in hand already states it and already carries
    it in ``context["group"]`` -- the cell's group a position defect is quoted
    against, the operations a descent was declined on, the subgroup a
    declaration is covariant on -- and comes back as it is, named once.
    """
    if "group" in rep.context:
        return rep
    head = (f"The cell's group at this point is {group.name} #{group.number_str}, an "
            "upper bound on what this Hamiltonian obeys"
            if validated is None else
            f"The group identified is {group.name} #{group.number_str}, every operation "
            "of it obeyed")
    return replace(rep, reason=f"{head}; its representation was not built: {rep.reason}",
                   context=dict(rep.context, group=group))


def _realize(model, *, typat=None, environment=None, cross_check=True,
             decorations=None, dz=DECORATION_DZ,
             c=SLAB_C, nk=6, tol=None, cut=None, window=None,
             symmetrize="auto", max_bytes=1e9, position_tol=None,
             precision=None, spinor=None, spin=None, descend=True,
             symprec=DEFAULT_FINDER, check=True, n_check=_bloch.CHECK_K,
             seed=_bloch.CHECK_SEED, params=None,
             selection_only=False, context=None, tolerances=None, **build_kw):
    """The whole pipeline, with the knobs the public signature does not carry.

    ``selection_only`` stops after the descent and returns the selection record
    alone (:func:`_realized_selection`); ``context`` is the
    :class:`_MeasurementContext` to measure in, ``None`` asking for a fresh one;
    ``tolerances`` is the :class:`.._tolerance.Tolerances` a parent call
    resolved, ``None`` asking for them to be resolved here from ``precision``,
    ``tol``, ``cut``, ``window``, ``position_tol`` and ``symprec``, which are
    then not read.  ``build_kw`` is handed to the build of the representation,
    :func:`.._build._build_rep`, and to the measurement before it where a
    keyword is read there: the declarations ``characters`` and ``multiplets``
    and the ceiling ``max_candidates``, which :func:`realized_group` takes by
    name, and every other keyword of the build.  :func:`realized_group` is the
    same call with the first three at their defaults, and carries the
    documentation of every keyword it takes.

    What stands here is the declarations and the record they end in: the
    tolerances one ``precision`` resolves into, the environment a layer's model
    is silent about, the species the cell is identified with, and -- once the
    group has settled -- the representation, built once and on that group alone.
    The measurement itself is :func:`_realized_core`, run on the cell as it
    stands for a fully periodic model and through :func:`_layer_realized` for a
    layer.
    """

    if symmetrize not in ("auto", True, False):
        raise ValueError("symmetrize must be \"auto\" (project when an operation is "
                         "measured near-symmetric), True (always project) or False "
                         f"(never), got {symmetrize!r}")
    # everything else goes to `_build_rep` by name: a word it does not take -- a
    # misspelling, or a model parameter, which travels in `params=` -- is refused
    # here, before the identification runs, rather than by `_build_rep` after it
    unknown = sorted(set(build_kw) - _BUILD_KEYWORDS)
    if unknown:
        raise TypeError("_realize() got an unexpected keyword argument "
                        f"{unknown[0]!r}")
    if environment not in (None, "substrate", "free-standing"):
        raise ValueError(
            "environment must be \"substrate\" (one side of the layer is different, so "
            "every operation with R[2,2] = -1 is absent and the group is the one-sided "
            "layer group), \"free-standing\" (suspended or symmetrically encapsulated, "
            "so the full layer group applies and the Hamiltonian selects among it), or "
            f"None (the default: substrate for a dim_r = 2 model, free-standing for a "
            f"dim_r = 3 slab), got {environment!r}")
    if environment is not None and int(model.lattice.dim_k) == 3:
        raise ValueError(
            f"environment={environment!r} is a question about a layer -- what is on the "
            "other side of it -- and this model is fully periodic (dim_k = 3), so it has "
            "no other side and no operation that could flip one. Drop the "
            "argument.")
    if precision is not None:
        try:
            _prec = float(precision)
        except (TypeError, ValueError):
            _prec = float("nan")
        if not (_prec > 0.0 and np.isfinite(_prec)):
            raise ValueError(
                "precision must be a positive number or None, got "
                f"{precision!r}. It is the relative precision of this model's own "
                "numbers -- how far they miss the symmetry they have, not how many "
                "decimals its files print -- and None means nothing is derived from "
                "it and the individual tolerances stand as given.")
        twice = [n for n, v in (("tol", tol), ("cut", cut), ("window", window),
                                ("position_tol", position_tol))
                 if v is not None]
        if twice:
            raise ValueError(
                f"precision={precision!r} and {' and '.join(twice)} declare the "
                "same thing twice: precision is the declaration those are "
                "derived from (position_tol = precision; tol = "
                "10*precision*max|H| with max|H| measured here; cut = "
                "tol/max|H| = 10*precision; window = clip(10*sqrt(cut), 1e-3, "
                "0.3)). Give the one "
                "number, or give the individual tolerances -- not both. "
                "(symprec may be given alongside precision; it is then the floor "
                "the finder is run at.)")
    if position_tol is not None:
        try:
            _ptol = float(position_tol)
        except (TypeError, ValueError):
            _ptol = float("nan")
        if not (_ptol > 0.0 and np.isfinite(_ptol)):
            raise ValueError(
                "position_tol must be a positive number or None, got "
                f"{position_tol!r}. It is the declared tolerance for the orbital "
                "centres -- absolute, in reduced coordinates -- and None means the "
                "position defect is neither measured nor removed.")
    # the memory budget, read here as grammar: the hopping projection's at its
    # peak, and the ceiling of the detector's own on an exactly degenerate spectrum
    max_bytes = _memory_budget(max_bytes)
    # ... and the ceiling on a search the measurement does not settle
    if "max_candidates" in build_kw:
        build_kw["max_candidates"] = _search_ceiling(build_kw["max_candidates"])
    if spin is not None and spinor is False:
        raise ValueError(
            "spin= declares that the orbital index of this spinless model carries "
            "spin-1/2, so its little-group content is double-valued; spinor=False "
            "contradicts it. Drop one of the two.")
    # THE SPINOR BASIS, resolved once and here, before anything is measured: a
    # word or a mapping becomes the record it names, and that record is what
    # every step below is handed -- the cell reduction, the multiplets, the
    # verdicts, the projection and the build alike.  Every presentation of a
    # layer keeps the orbitals' order and count, so the one record describes
    # them all.  What can go wrong here is grammar, and it raises.
    spin = _resolve_spin(spin, model)
    # THE SITES, read off the centres alone where a declaration needs them -- a
    # spinor basis, stated multiplets, or a key of several orbitals: the orbitals
    # at one centre are one site at the tolerance the cell's group is found at,
    # as every later step groups them
    site_of = sites = site_orbs = lattice = None
    if (spin is not None or build_kw.get("multiplets") is not None
            or _names_joint_keys(build_kw.get("characters"))):
        lattice, centres, _ = _pad_cell(model, c)
        sites, site_orbs, _ = _sites_and_typat(
            centres, None,
            (finder_distance(precision, model, symprec) if tolerances is None
             else tolerances.finder), lattice)
        site_of = np.zeros(int(model.norb), dtype=int)
        for s, orbs in enumerate(site_orbs):
            site_of[list(orbs)] = s
    # WHAT THE DECLARATIONS SAY, read as grammar before anything is measured: a
    # malformed `characters=` or `multiplets=` raises here, and which words name
    # something at a site is read in the site group the measurement reaches
    _declaration_grammar(int(model.norb), characters=build_kw.get("characters"),
                         multiplets=build_kw.get("multiplets"), spin=spin,
                         site_of=site_of)
    # ... and the one thing the centres alone say about a spinor basis: the two
    # spin states of one Wannier function share its centre.  A pair across two
    # centres is a mislabelled pairing or a magnetic configuration this package
    # does not read, and neither needs a group to say so, so it is refused here,
    # before anything is identified or measured, naming none.
    if spin is not None:
        split = _spin_pair_split(spin, site_orbs, sites, lattice)
        if split is not None:
            return split
    # WHAT THE MODEL IS, asked once and before anything is measured: a crystal
    # periodic in three directions, or a layer periodic in two, whose symmetry
    # is a space group.  A model periodic in fewer directions is a molecule or a
    # chain, whose symmetry is a point, rod or frieze group, and is refused
    # whatever the dimension of the cell it is written in.
    per = tuple(int(d) for d in model.periodic_dirs)
    if len(per) < 2:
        dims = {1: "one", 2: "two", 3: "three"}.get(int(model.lattice.dim_r),
                                                    str(int(model.lattice.dim_r)))
        what = (f"no periodic direction in its {dims}-dimensional cell: it is a "
                "molecule, whose symmetry is a point group" if not per else
                f"one periodic direction in its {dims}-dimensional cell, along lattice "
                f"vector {per[0]}: it is a chain or a ribbon, whose symmetry is a rod or "
                "frieze group")
        return Refusal(
            "fewer_than_two_periodic_directions",
            f"this model has {what}. Neither a molecule nor a chain is in this "
            "package's scope yet: it reads a crystal periodic in three directions and a "
            "layer periodic in two, whose symmetry is a space group, and a model "
            "periodic in fewer is refused whatever the dimension of its cell.",
            value=float(len(per)), threshold=2.0,
            context={"periodic_dirs": per, "dim_r": int(model.lattice.dim_r)})
    params = dict(params or {})
    # ... and a copy that survives the symmetrizing pass, which replaces `params`
    # with an empty mapping because what it returns is a model built from numbers.
    # A layer's other environment is a statement about the model as handed in,
    # so it is measured with the parameters as handed in.
    embed_params = dict(params)
    # ONE declaration, resolved once into the tolerances every step judges noise
    # against: the centres', in reduced coordinates, where a relative precision
    # is already absolute; the Hamiltonian's, where it is not, because the
    # covariance gate is in the units of H, so `max|H|` is measured rather than
    # asked for; the matrices read off H, relative to their own scale; and the
    # symmetry finder's distance, in the model's length units.  Under a
    # precision the detector's `cut` and `window` are relative, read against the
    # same `max|H|`, and its nullspace keeps whatever the gate would; declaring
    # either of them beside it says the thing `precision` would have said, which
    # is the double declaration refused above.  A parent call hands its own down,
    # and the other environment of a layer judges the same Hamiltonian against
    # the same numbers.
    if tolerances is None:
        tolerances = Tolerances.declared(
            model, precision=precision, tol=tol, cut=cut, window=window,
            position_tol=position_tol, symprec=symprec,
            scale_of=lambda: _max_hamiltonian(model, params=params))
    # A fully periodic model is already its own presentation and its own cell:
    # both preparation steps exist to make a 2D model answerable by a 3D
    # symmetry finder, and there is nothing here for them to do.
    three_d = int(model.lattice.dim_k) == 3
    # THE ENVIRONMENT, resolved once.  A dim_r = 2 model has no out-of-plane
    # coordinate and is silent about what is under it, so the default is the
    # environment that is always well defined: the substrate.  A dim_r = 3 slab is
    # not silent -- its geometry says which -- so its default is
    # free-standing, and a substrate there is a declaration like any other.
    environment_declared = environment is not None
    if three_d:
        environment = None
    elif environment is None:
        environment = "substrate" if int(model.lattice.dim_r) == 2 else "free-standing"

    # THE MEASUREMENT CONTEXT, one per call.  H(k) on the detection mesh and the
    # detector's solution space per in-plane k-action, shared by every
    # measurement this pipeline makes: every round of the loop below, the
    # re-measurement after a hopping projection, and -- passed down through
    # `_layer_realized` -- the cross-check's other environment, which poses the
    # identical in-plane actions.  It carries the model in its key, so the two
    # steps that change the Hamiltonian (the position pass and the projection)
    # invalidate it by construction.  `context` is the parent's, when this call
    # IS the other environment.
    context = _MeasurementContext() if context is None else context

    labels = [0] * int(model.norb) if typat is None else list(typat)
    extras = () if decorations is None else tuple(decorations)
    if spinor is None:
        # a declared spinor basis says the states carry spin-1/2 as surely as
        # `nspin == 2` does, so the tables to ask for are the double-valued ones
        spinor = int(model.nspin) == 2 or spin is not None

    measured = dict(labels=labels, spinor=bool(spinor), nk=nk, tolerances=tolerances,
                    params=params, context=context, spin=spin,
                    descend=descend, symmetrize=symmetrize, max_bytes=max_bytes,
                    build_kw=build_kw, returned=not selection_only)
    if three_d:
        m3 = model
        if not extras:
            if len(labels) != int(model.norb):
                raise ValueError(f"typat must have one entry per orbital "
                                 f"({int(model.norb)}), got {len(labels)}")

            def rebuild(m, _extras):
                """A fully periodic model is its own cell."""
                return m, list(labels)
        else:
            def rebuild(m, ex):
                """The model's cell with its decorations."""
                return _decorated_cell(m, labels, dz=dz, decorations=ex)
        cell, cell_typat = rebuild(m3, extras)
        # THE CENTRES MADE EXACT, where a precision for them is declared, before
        # the cell's setting is read: the cell is then identified on centres
        # that are exact, in the model's own frame wherever noise is all that
        # kept them from it.
        read = None
        if tolerances.centre is not None:
            got = _exact_centres(m3, cell, cell_typat, extras=extras, rebuild=rebuild,
                                 tolerances=tolerances, params=params)
            if isinstance(got, Refusal):
                return got
            m3, cell, cell_typat, read = got
        got = _realized_core(m3, cell, cell_typat, fully_periodic=True,
                             centres_read=read, **measured)
        if isinstance(got, Refusal):
            return got
        sel, cross = got, None
        presented = {"residual": 0.0, "tilted": None}
    else:
        got = _layer_realized(model, environment=environment, cross_check=cross_check,
                              typat=typat, decorations=decorations, extras=extras,
                              dz=dz, c=c, check=check, n_check=n_check, seed=seed,
                              embed_params=embed_params, **measured)
        if isinstance(got, Refusal):
            return got
        sel, cross, presented = got

    m3, group, assignment = sel["model"], sel["group"], sel["assignment"]
    validated, cell, cell_typat = sel["validated"], sel["cell"], sel["cell_typat"]
    steps, positions = sel["descent"], sel["positions"]
    symmetrization, params = sel["symmetrization"], sel["params"]
    time_reversal, asked = sel["time_reversal"], sel["antiunitary"]

    if selection_only:
        # a comparison, whose representation is never built: its operations
        # combined with time reversal are read without matrices
        time_reversal, _beyond = _beyond_grey(group, None, asked, time_reversal,
                                              tolerances=tolerances)
        return {"environment": environment, "group": group, "validated": validated,
                "assignment": assignment, "model": m3,
                "descent": steps, "parents": sel.get("parents", ()),
                "held": sel.get("held", ()),
                "positions": positions,
                "time_reversal": time_reversal, "unread": sel.get("unread")}

    # The representation is built on what the loop settled on, and with the
    # DECLARED CONTENTS as they stand there rather than with the declaration's
    # words: a label read in the cell's group need not name anything in the
    # returned one (`A2` of `3m` is `A` of `3`, and `E` of `3m` is the merged pair
    # `1E2E`), so what is carried is the constraint, restricted.
    build_now = {**build_kw, "multiplets": sel["layout"]}
    if sel.get("declared"):
        build_now["characters"] = sel["declared"]
    # WHAT THE VERDICT STAGE READ on this same group -- the scalars the hoppings
    # pin -- handed down rather than read again: one reading of one
    # measurement, carried from the per-operation verdict into the
    # representation, whose labelling the build chooses once.  `_build_rep`
    # checks that the reading is about the operation set it represents and
    # makes its own where it is not.
    rep = _build_rep(m3, group, assignment, validated=validated, spin=spin,
                     _choices=sel.get("choices"), params=params, **build_now)
    if isinstance(rep, Refusal):
        if (rep.gate == "rep_not_covariant"
                and not (rep.value is not None
                         and float(rep.value) <= float(tolerances.energy))):
            # The consistency guard, whether or not a descent ran.  The one case
            # it is NOT about: a covariance residual inside the DECLARED
            # tolerance is the near-symmetric signal, not three measurements
            # disagreeing -- the detector was right to keep the operation and the
            # exact gate is right to refuse it, and `symmetrize=` is what the
            # message names.  A descent runs on every free-standing 2D model (the
            # cell keeps the whole sigma_h family and the Hamiltonian selects),
            # so without excluding it the guard would hide that message behind
            # itself.
            _raise_not_covariant(rep, group, validated, steps)
        rep = _named_refusal(rep, group, validated)
        square = (time_reversal or {}).get("square")
        if (square is not None and time_reversal.get("outcome") == "measured broken"
                and not (np.isfinite(square["measured"])
                         and square["measured"] * square["asked"] > 0)):
            # the group identified is unitary because time reversal's own square
            # broke it, and a refusal of its representation says so -- with the
            # operations combined with time reversal read without matrices, none
            # having been built
            time_reversal, _beyond = _beyond_grey(group, None, asked, time_reversal,
                                                  tolerances=tolerances)
            tables = "single-valued" if square["asked"] > 0 else "double-valued"
            rep = replace(rep, reason=rep.reason + (
                f" Time reversal was measured broken first: its square is "
                f"{square['measured']:+.6g} where the {tables} tables read here ask for "
                f"{square['asked']:+d} (context['time_reversal']), so the group "
                f"identified is the unitary one."),
                context=dict(rep.context, time_reversal=time_reversal))
        return _quote_position_defect(rep, positions, symmetrization)

    # THE ENVIRONMENT, stamped onto the representation, and on a substrate the
    # free-standing layer's group the cross-check measured beside it, read once
    # here.  `ebr.symmetry_indicator` completes a layer's class on its decoupled
    # stack in the group the environment measured, and names the free-standing
    # group beside a substrate's class where it is larger, without measuring it
    # again.  A representation built by `_build_rep` alone carries None for both,
    # which the indicator reads as unstated rather than as either environment.
    free_standing_group = None
    if environment == "substrate" and cross and cross.get("outcome") == "agree":
        free_standing_group = dict(cross["free_standing"])
    rep = replace(rep, environment=(None if environment is None else str(environment)),
                  free_standing_group=free_standing_group)

    # THE OPERATIONS COMBINED WITH TIME REVERSAL, read once, here: where time
    # reversal is broken, every one of the cell's that a round measured -- each
    # with that round's verdict and matrix -- against the group returned and
    # the matrices the build assembled for it, the labelling in them the one it
    # chose, which the rule reads up to a one-dimensional character
    time_reversal, beyond_grey = _beyond_grey(group, rep.matrices, asked, time_reversal,
                                              tolerances=tolerances)

    # THE SECTORS, as the measurement read them where the model entered, and
    # what the operations of the group returned do to them, for the record
    sectors = sel.get("sectors")
    if sectors is not None:
        sectors = (sectors, sector_action(sectors, m3, group, assignment, spin, validated,
                                          tol=tolerances.matrix or None))

    # the DECLARED decoration, which is what this record is about, together with
    # the orbit step 1 added where the declared content reduced the cell: the
    # descent's own species are the descent's business and are on
    # `descent["cell"]`.
    reduction = sel.get("reduction")
    bare_3d = three_d and not extras and reduction is None
    base_typat = sel["base_typat"]
    n_extra = len(base_typat) - len(labels)
    # ... and the ENVIRONMENT's own base underneath it: a substrate
    # builds the shadow species, at the height `dz`, whether or not anything was put
    # extras on top of them, so what ran is the environment's question and not
    # the extras'.
    shadow_recipe = not three_d and environment == "substrate"
    descent = None
    if steps:
        lattice_step = next((s["lattice"] for s in steps if "lattice" in s), None)
        # WHAT THE WHOLE DESCENT DROPPED, every step's together and by cause,
        # and what the operations kept carry: the note's first sentence, its
        # counts against the group the cell allowed adding up to the group it
        # arrived at
        counts = _descent_counts(steps, time_reversal)
        split, lowered = _descent_clauses(counts)
        # THE STATES' CLASS, said where it lowered the group: the pair that
        # names it and, where subgroups of the kept order tied, the rule; on a
        # layer, the free-standing layer it was read on, and the sign the
        # horizontal mirror was given there
        projective = ""
        for st in steps:
            rec = st.get("projective")
            if not rec or not rec.get("pairs"):
                continue
            first = rec["pairs"][0]
            a, b = first["names"]
            phase = complex(first["commutation_phase"])
            tied = (f", the first of {len(rec['tied'])} of that order by the tie rule "
                    f"`realized_group` states -- {rec.get('rule')} -- since "
                    f"{rec.get('reason')}" if rec.get("tied") else "")
            projective += (
                f" On the states of this spinless reading the matrices of {a} and {b}, "
                "two operations that commute, "
                + ("anticommute" if abs(phase + 1.0) <= _CLASS_MARGIN else
                   f"commute only up to the phase {phase.real:+.3g}{phase.imag:+.3g}j")
                + (" -- the sign their spin-1/2 lifts carry, stored in the orbital index"
                   if first["spin_half"] else "")
                + f", and {len(rec['pairs'])} such pairs were measured in "
                f"#{st['from']['number_str']}: no single-valued representation of a "
                "group holding one exists on these states")
            free, mirror = rec.get("read_on"), rec.get("horizontal_mirror")
            if free:
                projective += (
                    ". The class is read on the free-standing layer, "
                    f"{free['name']} #{free['number_str']}, this group with the horizontal "
                    "mirror adjoined, where each operation composed with the mirror has "
                    "its partner's matrix times the mirror's sign: the largest subgroup "
                    "there on which the states carry an ordinary representation is kept"
                    + tied + ", and the operations of it that keep the layer's side are "
                    f"#{st['to']['number_str']}, the group kept here")
            else:
                projective += (
                    ", so the group kept is the largest subgroup on which the states "
                    f"carry an ordinary representation, #{st['to']['number_str']}" + tied)
            if mirror:
                projective += (
                    ". Each operation composed with the horizontal mirror acts on these "
                    "orbitals and on in-plane k as its partner does, and has its "
                    "partner's matrix times the mirror's sign, "
                    + (f"{int(mirror['sign']):+d}, the parity the declared content states"
                       if mirror.get("determined_by") == "declaration" else
                       "+1, the even one the lowest-angular-momentum rule assigns")
                    + ", on which the class does not depend")
            projective += ". `spin=` reads the same states in the double group."
        readings = tuple(c for step in steps
                         for c in (step.get("declaration") or {}).get("tied", ()))
        # THE TIES, said where the tie rule chose: which subgroups tied, which
        # one was kept and why -- among the survivors that did not close, and
        # among the readings of a declaration
        ties = "".join(
            " " + note[:1].upper() + note[1:]
            for st in steps
            for note in ((st.get("centres") or {}).get("note"),
                         (st.get("tie") or {}).get("note"),
                         (st.get("declaration") or {}).get("note"))
            if note)
        # WHAT HOLDS, where time reversal was measured broken on the group the
        # last descent arrived at: that group, time reversal adjoined, is not
        # what the Hamiltonian obeys, and is not named as if it were.  What
        # holds is its unitary half, returned, and the operations combined with
        # time reversal that `beyond_grey` names.
        tr = time_reversal or {}
        unitary_after = (tr.get("outcome") == "measured broken"
                         and tr.get("grey_number_str") == steps[-1]["to"]["number_str"])
        if unitary_after:
            n_au = int((beyond_grey or {}).get("n_operations", 0))
            mtype = (beyond_grey or {}).get("magnetic_type")
            arrived = (
                "so the cell was lowered to "
                + (lowered or "the unitary operations measured to hold")
                + ", and time reversal, measured with them, is broken: what holds is the "
                f"unitary operations of {group.name} #{group.number_str}"
                + ((f" and the {n_au} combined with time reversal that `beyond_grey` names"
                    + (f", together a magnetic group of type {mtype}"
                       if mtype in ("III", "IV") else ""))
                   if n_au else ", and no operation combined with time reversal")
                + f". Only {group.name} #{group.number_str}, the unitary group, is "
                "returned: the labels, EBR rows, indicators and SRSIs below are its own.")
        else:
            last = steps[-1]["to"]
            arrived = (
                "so the cell was lowered to "
                + ("the ones measured to hold and" if lowered is None else f"{lowered}, and")
                + f" re-identified as {last['name']} #{last['number_str']}. "
                "Only that group is returned: the labels, EBR rows, indicators and SRSIs "
                "below are its own.")
        lattice_said = ("" if not lattice_step else
                        (_lattice_note(lattice_step) + " ") if unitary_after else
                        (lattice_step["note"] + " "))
        # A PRECISION THAT HOLDS THEM, said where every operation broken here was
        # broken by its own form beyond the covariance gate: the least the
        # model's numbers would be declared at for the cell's group to hold
        holding = _precision_holding(
            steps, lambda: (tolerances.scale if tolerances.scale is not None
                            else _max_hamiltonian(model, params=embed_params)))
        precise = ("" if holding is None else
                   " Each operation broken here carries H(k) onto H(gk) in its own form to "
                   f"{holding[0]:.3g} at most, beyond the tolerance "
                   f"{float(tolerances.energy):.3g} this call holds each operation to: a "
                   f"precision of {holding[1]:.2g} or coarser, declared with `realized_group(..., "
                   "precision=)`, holds every one of them within it, and "
                   f"{steps[0]['from']['name']} #{steps[0]['from']['number_str']} with them.")
        descent = {
            "from": dict(steps[0]["from"]), "to": dict(steps[-1]["to"]),
            "dropped": tuple(b for step in steps for b in step["dropped"]),
            "dropped_by": {cause: int(n) for cause, n in counts.items() if n},
            "tied": readings,
            "cell": dict(steps[-1]["cell"]),
            "iterations": len(steps), "steps": tuple(steps),
            "lattice": lattice_step,
            "note": lattice_said + (
                f"{'The' if lattice_said else 'the'} cell allowed {steps[0]['from']['name']} "
                f"#{steps[0]['from']['number_str']} ({steps[0]['from']['size']} "
                f"{_PER_COSET}"
                + (f"; these are {_DOUBLE_GROUP}" if group.spinor else "")
                + f") and {split}, "
                + arrived
                + " The model handed back is "
                "the original presentation -- the generic orbit that carried the descent "
                "lives on the cell, which is geometry, and never on it.")
            + ((f" Between the two, the declared orbital content lowered the cell itself "
                f"from #{reduction['from']['number_str']} to "
                f"#{reduction['to']['number_str']}: {reduction['n_excluded']} operations "
                "the crystal that was declared does not have, which is not a descent and "
                "is on `decoration['declared']`.") if reduction is not None else "")
            + ties
            + projective
            + precise,
        }
    return RealizedGroup(
        model=m3, source=model, cell=cell, typat=labels, cell_typat=list(cell_typat),
        group=group, assignment=assignment, validated=validated, rep=rep,
        time_reversal=time_reversal, beyond_grey=beyond_grey, descent=descent,
        symmetrization=symmetrization, positions=positions,
        precision=tolerances.returned(),
        environment=(None if environment is None else str(environment)),
        environment_declared=bool(environment_declared), cross_check=cross,
        _sectors=sectors,
        # what the slab presentation measured, by its own identity check: the
        # layer's H(k) against the slab's, carried back through the move where
        # a tilted non-periodic vector was put on the layer's normal
        presentation={"c": None if three_d else float(c),
                      "residual": float(presented["residual"]),
                      "n_check": 0 if three_d else int(n_check),
                      "tilted": presented["tilted"]},
        # `dz` is the sigma_h recipe's own parameter: it is recorded when that
        # recipe ran and `None` when it did not, rather than reporting a number
        # that had no part in the group.  `typat` and `n_atoms` count the
        # whole addition, base and extras alike, so the recipe names both when
        # both are there.
        decoration=None if bare_3d else
        {"dz": (float(dz) if shadow_recipe else None),
         "typat": tuple(base_typat[len(labels):]),
         "n_atoms": int(n_extra),
         "declared": reduction,
         "recipe":
         (("one non-orbital shadow species under every site, at that site's "
           "own height minus dz -- the substrate, as geometry, on the point "
           "set as it stands: removes every operation with R[2,2] = -1 "
           "(sigma_h and everything composed with it, 3D inversion included) "
           "and leaves the subgroup that preserves the layer's normal, its fractional "
           "translations and the layer's own plane alone"
           + (f", with {len(extras)} caller-supplied species on top" if extras else ""))
          if shadow_recipe else
          ("none: the layer is free-standing, so the cell is the point set "
           "as given and every plane-preserving operation is left for the "
           "Hamiltonian to select from rather than removed by construction")
          if not extras and not three_d else
          "caller-supplied decoration" if extras else
          "none: the cell is the model's own point set")
         + ("" if reduction is None else _declared_recipe(reduction, group.spinor))})


def _declared_recipe(reduction, spinor=False) -> str:
    """What the step 1 orbit is, said on :attr:`RealizedGroup.decoration`."""
    return (
        f"; and one free orbit of {len(reduction['orbit'])} points under species "
        f"{reduction['species']!r}, which carries what the declared content leaves of "
        f"the cell: {reduction['n_excluded']} of "
        f"{reduction['from']['name']} #{reduction['from']['number_str']}'s "
        f"{reduction['from']['size']} {_PER_COSET}"
        + (f", {_DOUBLE_GROUP}," if spinor else "")
        + " are not symmetries of the crystal that "
        f"was declared and are gone from the cell before any hopping is read, so the "
        f"cell was identified as {reduction['to']['name']} "
        f"#{reduction['to']['number_str']} instead. Each excluded operation says why "
        "under `declared['excluded']`. This is a statement about what the orbitals are, "
        "not a descent: `descent` carries what the Hamiltonian broke.")


def _layer_realized(model, *, environment, cross_check, typat, decorations, extras,
                    dz, c, check, n_check, seed, embed_params, labels, spinor, nk,
                    tolerances, params, context, spin, descend, symmetrize,
                    max_bytes, build_kw, returned):
    r"""A layer's realization: its presentation, its environment's cell, both sides.

    Everything a **layer** has that a fully periodic crystal does not, in one
    place, so that :func:`_realized_core` knows nothing about layers.

    The **slab presentation**: a ``dim_r = 2`` model carries no third coordinate
    for a 3D symmetry finder to read, a slab whose non-periodic vector is tilted
    off the layer's normal has it put on the normal, every orbital where it was,
    and a slab whose vector lies along the normal with a vacuum already reaching
    ``c`` is its own presentation (:func:`_slab_presentation`).

    The **environment's cell**.  Free-standing, it is the point set as given --
    every plane-preserving operation kept and the Hamiltonian left to select --
    with ``include_vacuum_ops`` on, because the operations that move the vacuum
    direction are the ones being identified.  On a substrate it is the same point
    set with one shadow species under every site, so that every operation with
    ``R[2,2] = -1`` is gone before the Hamiltonian is read.  ``decorations=``
    adds to either, which is what lets a crystal field mean the same thing in
    both and the two be compared at all.

    The **closure resolution** of a free-standing cell, where both members of
    every fiber are present and the survivors need not compose
    (:func:`_resolve_closure`).  The tensor test that tells a fiber's members
    apart holds each operation to its own lift in both environments
    (:func:`.._descent._ansatz_refuted`).

    **What the declared content says the sites are**, read in the layer's own
    site groups on a substrate too (:func:`_layer_declared_reading`): a site
    declared ``E'`` and one declared ``E''`` of ``-6m2`` are different sites in
    both environments, as two sites of different ``typat`` are, although
    without ``σ_h`` both restrict to one ``E`` of ``3m``.

    **The states' projective class**, read once, on the free-standing layer:
    there each operation composed with ``σ_h`` has the mirror's chosen sign
    times its partner's matrix (:func:`.._descent._projective_class_refuted`), and on a
    substrate the free-standing cell of the same point set is handed down so
    that the class is read on the substrate's group with ``σ_h`` adjoined and
    its part that preserves the layer's normal carried
    (:func:`.._descent._projective_class_carried`).

    The **cross-check**: the same Hamiltonian selected again in the other
    environment -- an independent identification and an independent descent,
    through :func:`_realized_selection` so that one measurement context serves
    both -- and held to the subgroup statement (:func:`_environment_record`).
    Where the two disagree no group is returned and the package raises
    (:func:`_raise_environment_contradiction`).  A refusal over there is a
    different thing entirely and is not a mismatch: it is recorded as the
    outcome and the group asked for is returned.  Nor is a disagreement where an
    environment's group is one of several subgroups its measurement could not
    choose between -- among its survivors, the readings of its declared
    content, the subgroups carrying the centres within a declared precision,
    or those carrying the states' projective class -- kept by the tie rule,
    and the statement holds for
    another pair of the two environments' candidates: a model written for a
    substrate need not lift back to a unique layer group, and the lift the rule
    keeps need not be the one the other environment measures
    (:func:`_environment_ties`, :func:`_environment_candidates`).  Where it
    holds for no pair, the package raises as for any other disagreement.
    A free-standing layer's declaration is read on a substrate where its words
    were written, in the free-standing layer's site groups, and restricted to
    the substrate's (:func:`_restricted_contents`): ``A2''`` of ``-6m2`` is
    the ``A1`` of ``3m`` it restricts to, and the substrate reads the same
    crystal the layer was declared to be.  A declaration written in a
    substrate's site irreps, which a free-standing layer's site groups name
    nothing for -- ``A1`` and ``A2`` of ``3m``, which ``-6m2`` has no word for
    -- is not compared: it says which environment is meant, the free-standing
    layer read without it would be another crystal, and nothing lifts it into
    that layer's site irreps.

    Returns ``(record, cross, presentation)`` -- the measurement's own record,
    the cross-check, and what the presentation measured of itself as built,
    before any centre is moved and any hopping projected (``residual`` and
    ``tilted``, :func:`_slab_presentation`) -- or a :class:`..refusal.Refusal`.
    """
    # THE PRESENTATION, and the one decision that belongs here rather than in
    # `_slab_presentation`: whether there is anything to present at all.  A slab
    # whose vector lies along the layer's normal with a vacuum already reaching
    # `c` is its own presentation, and `_slab_presentation` builds a new model in
    # every case it is asked, so asking it that question would make `model is
    # source` depend on the vacuum.
    # Its identity with the model is a statement about the model as handed in,
    # so the presentation measures it on itself as *built* -- before the centres
    # are moved by a declared precision and before the hoppings are projected,
    # exactly as `presentation` claims -- and this is the record of it.
    presentation = {"residual": 0.0, "tilted": None}
    if _is_presented_as_a_slab(model, c):
        m3 = model
    else:
        m3 = _slab_presentation(model, c=c, check=check, n_check=n_check, seed=seed,
                                params=params, record=presentation)

    # THE ENVIRONMENT'S CELL.  A free-standing layer's cell is its point set as
    # given, so nothing is decorated and `include_vacuum_ops` is on.  A layer on
    # a substrate is the other question and needs a different cell: the layer
    # pressed onto its own plane (H(k) bit-identical), with the substrate's
    # shadow species under every projected site.  The SAME point set either way,
    # which is what makes the substrate group a subgroup of the free-standing one
    # rather than the image of some other crystal.  `_decorated_cell` builds that
    # base, so it is built in one place and the environment is passed on rather
    # than acted on here.
    if environment == "free-standing":
        build_kw.setdefault("include_vacuum_ops", True)

    def rebuild(m, ex):
        """The environment's cell on a model and the decorations beside it."""
        return _decorated_cell(m, labels, dz=dz, environment=environment, decorations=ex)

    cell, cell_typat = rebuild(m3, extras)

    # THE CENTRES MADE EXACT, where a precision for them is declared, before the
    # cell's setting is read, on the environment's own cell: every reading of
    # the layer below, the free-standing one a substrate carries included, is
    # then made on centres that are exact.
    read = None
    if tolerances.centre is not None:
        got = _exact_centres(m3, cell, cell_typat, extras=extras, rebuild=rebuild,
                             tolerances=tolerances, params=params)
        if isinstance(got, Refusal):
            return got
        m3, cell, cell_typat, read = got
        extras = read["extras"]

    # WHAT THE DECLARED CONTENT SAYS THE SITES ARE, read in the layer's own site
    # groups.  Two sites declared different content are different sites in
    # every environment, as two sites of different `typat` are -- `E'` and
    # `E''` of a `-6m2` site are different irreps, though on a substrate both
    # restrict to one `E` of `3m` -- so the substrate's cell is lowered by what
    # the declaration says in the free-standing layer's site groups, where
    # its words name something there, as well as by what they say in its own.
    wider = None
    if environment == "substrate" and build_kw.get("characters") is not None:
        wider = _layer_declared_reading(m3, labels, extras=extras, spinor=spinor,
                                        spin=spin, symprec=tolerances.finder,
                                        characters=build_kw["characters"])

    # THE FREE-STANDING LAYER, on which a substrate's projective class is read.
    # The states' class is read once, on the free-standing layer, where each
    # operation composed with the horizontal mirror has the mirror's chosen
    # sign times its partner's matrix; on a flat layer that group is the
    # substrate's with the mirror adjoined, so a substrate carries the
    # part of that reading that preserves the layer's normal rather than making
    # its own.  The cell
    # is geometry and costs nothing; its group is identified only where a class
    # is to be read.
    free_standing = None
    if environment == "substrate":
        fs_cell, fs_typat = _decorated_cell(m3, labels, environment="free-standing",
                                            decorations=extras)
        free_standing = {"cell": fs_cell, "cell_typat": fs_typat, "labels": labels,
                         "symprec": tolerances.finder}

    got = _realized_core(m3, cell, cell_typat, tolerances=tolerances,
                         fully_periodic=False, labels=labels, spinor=spinor,
                         nk=nk, params=params,
                         context=context, spin=spin, descend=descend,
                         symmetrize=symmetrize, max_bytes=max_bytes,
                         build_kw=build_kw, returned=returned,
                         declared_wider=wider, free_standing=free_standing,
                         centres_read=read)
    if isinstance(got, Refusal):
        return got
    if not returned or not cross_check:
        return got, None, presentation

    # THE CONSISTENCY GATE.  The same model is selected again in the other
    # environment, and the substrate group must be the subgroup of the
    # free-standing one that preserves the layer's normal, operation for operation and in both cosets.
    # Where it is not, no group is returned and the package raises: two
    # selections of one Hamiltonian disagreeing about which operations it obeys
    # is this package disagreeing with itself, and nothing is returned past it.
    #
    # Where an environment did not separate its own candidate subgroups and the
    # tie rule kept one of them, whatever made the tie -- survivors that do not
    # close (`_resolve_closure`), a free-standing layer whose lift a flat cell
    # cannot pin, the readings of a declared content
    # (`_declaration._covariant_subgroup`), the subgroups carrying the centres
    # within a declared precision (`_centres_carried`), or those carrying the
    # states' projective class (`_projective_class_refuted`) -- its measurement
    # determined a set of candidate groups, not one, and the two environments'
    # candidate sets are held against each other (`_environment_candidates`).
    # Where the statement holds for the pair returned, the two agree, the tie
    # named beside the agreement; where it fails for that pair and holds for
    # another, the rule's choice is not what the other environment measured,
    # which is no conflict about the Hamiltonian, and that is recorded as the
    # outcome `ambiguous`, the pair named, with the group asked for returned;
    # where it holds for no pair, the package raises as for any mismatch.  A
    # refusal over there is recorded with its gate.  And a declaration in a
    # substrate's site irreps, which a free-standing layer's site groups name
    # nothing for, is not compared at all: it says a substrate is meant.  A
    # free-standing layer's words are read on the substrate by restriction, and
    # compared.
    group = got["group"]
    other_environment = ("free-standing" if environment == "substrate"
                         else "substrate")
    other = _realized_selection(
        model, typat=typat, environment=other_environment,
        cross_check=False, context=context,
        decorations=decorations, dz=dz, c=c, nk=nk,
        symmetrize=symmetrize, max_bytes=max_bytes,
        spinor=spinor, spin=spin, descend=descend, check=check,
        n_check=n_check, seed=seed, params=embed_params,
        # the tolerances this side resolved: the same Hamiltonian's noise is
        # judged against the same numbers on the other side
        tolerances=tolerances,
        # `characters=` goes with it: a declaration lowers the group the way a
        # hopping does, so a selection made without it is a different crystal
        # and comparing the two would compare two questions.
        **{k: build_kw[k] for k in ("multiplets", "n_battery", "characters")
           if k in build_kw})
    returned_head = {"number_str": str(group.number_str), "name": str(group.name),
                     "size": int(group.size)}
    if isinstance(other, Refusal):
        refusal = {"gate": str(other.gate), "reason": other.reason,
                   "environment": other_environment}
        cross = {
            "free_standing": None if environment == "substrate" else returned_head,
            "substrate": returned_head if environment == "substrate" else None,
            "rows": (), "missing": (), "extra": (), "flipping": (),
            "unmapped": (),
            "agrees": None, "outcome": "refused", "tie": (),
            "refusal": refusal,
            "note": (
                "the selection in the other environment refused, so the two could "
                "not be compared; a refusal there is not a mismatch here, and this "
                "group is returned. The refusal is in `refusal` and is reproducible "
                f"with realized_group(..., environment={other_environment!r})")}
    elif other.get("unread") is not None:
        # THE DECLARATION SAYS WHICH ENVIRONMENT IS MEANT.  Its words are this
        # environment's site irreps and name nothing in the other's site groups,
        # so the other environment's group, read without them, is the group of
        # another crystal: it is not recorded, and the two are not compared.
        # Nothing lifts the words into the other environment's site irreps.
        mine = ("the substrate's" if environment == "substrate"
                else "the free-standing layer's")
        theirs = ("the free-standing layer" if environment == "substrate"
                  else "the substrate")
        cross = {
            "free_standing": None if environment == "substrate" else returned_head,
            "substrate": returned_head if environment == "substrate" else None,
            "rows": (), "missing": (), "extra": (), "flipping": (),
            "unmapped": (),
            "agrees": None, "outcome": "not compared", "tie": (),
            "refusal": None,
            "note": (
                f"not compared with {theirs}: the declaration is written in {mine} "
                f"site irreps, and {theirs}'s site groups have none of its words. A "
                "declaration in one environment's site irreps says which environment "
                f"is meant: {theirs} is not read with it, and this group is "
                f"returned. A layer meant {other_environment} is declared in its own "
                "site irreps, which "
                f"realized_group(..., environment={other_environment!r}) lists where "
                "it refuses this declaration")}
    else:
        free = group if environment == "free-standing" else other["group"]
        sub = other["group"] if environment == "free-standing" else group
        dropped, ties = _environment_ties(((str(environment), got),
                                           (other_environment, other)))
        cross = dict(_environment_record(free, sub), tie=dropped, refusal=None)
        left = ", ".join(sorted({str(t["name"]) for t in dropped}))
        one = len(ties) == 1
        where = (("the realized group " if one else "the realized groups ")
                 + ", and ".join(ties))
        if cross["agrees"] is False:
            # A TIE IS A SET OF CANDIDATES.  Where an environment's measurement
            # left several subgroups of the largest order, whatever made the tie,
            # and the tie rule kept one, the statement is held to every pair of the
            # two environments' candidates.  Holding for another pair, the
            # failure is the rule's choice and not what the other environment
            # measured, which is no conflict about the Hamiltonian: the group
            # asked for is returned and the record names the candidates, the
            # pair the statement holds for, and the operations the rule left
            # out.  Holding for none, it is the package disagreeing with itself.
            mine, theirs = _environment_candidates(got), _environment_candidates(other)
            free_c, sub_c = ((mine, theirs) if environment == "free-standing"
                             else (theirs, mine))
            pair = _environment_explained(free_c, sub_c)
            if pair is None:
                _raise_environment_contradiction(cross, held=len(free_c) * len(sub_c))
            cross.update(agrees=None, outcome="ambiguous", note=(
                where
                + (", is not unique, and the one returned there is" if one else
                   ", are not unique, and the ones returned there are")
                + f" the first by the tie rule, without {left} -- and the subgroup "
                "statement does not hold for " + ("that one" if one else "those")
                + f"; it holds for {pair[0]} free-standing and {pair[1]} on a substrate, "
                "which the measurement does not tell from the pair returned. The group "
                "asked for is returned, and it is the one the model can answer."))
        elif ties and cross["agrees"]:
            # ... and where the statement holds for the one the rule kept, the
            # two environments agree, with the tie named beside the agreement
            cross["note"] += (
                ". " + where[:1].upper() + where[1:]
                + (", is the one the tie rule keeps" if one else
                   ", are the ones the tie rule keeps")
                + f", without {left}, and the statement holds for "
                + ("it" if one else "them"))
    return got, cross, presentation


def _environment_ties(records) -> tuple:
    r"""Where each environment's group was the tie rule's choice, and what it left out.

    A measurement can leave several subgroups of the largest order with
    nothing measured between them, and the tie rule keeps the first
    (:func:`_tie_ranked`), whatever made the tie: the survivors of a round that
    do not close into a group, whose operations only the candidates not kept
    carry are refuted ``"tie"``; the readings of a declared content, none of
    which contains the others; the subgroups that carry the orbital centres
    within a declared precision; and those on which the states carry an
    ordinary representation -- the operations only the subgroups not kept
    carry broken by the declaration, the centres or the class.  ``records``
    pairs each environment's name with its measurement.

    Returns ``(left, words)``: one entry per unitary operation the rule left
    out, its ``environment``, ``ind`` and ``name``; and one clause per tie,
    naming its environment and what tied, empty where nothing did.
    """
    left, words = [], []
    for where, rec in records:
        for step in (rec.get("descent") or ()):
            broken = tuple(step.get("dropped", ()))
            if any(str(b.get("refuted_by")) == "tie" for b in broken):
                left += [{"environment": where, "ind": int(b["ind"]),
                          "name": str(b["name"])}
                         for b in broken if str(b.get("refuted_by")) == "tie"]
                words.append(f"in the {where} environment, where the measured "
                             "survivors contain more than one subgroup of the "
                             "largest order")
            names = (step.get("declaration") or {}).get("names", ())
            for cause, ranked, said in (
                    ("declaration", (step.get("declaration") or {}).get("tied", ()),
                     "the declared content has more than one reading of the largest "
                     "order, " + " and ".join("{" + ", ".join(r) + "}" for r in names)),
                    ("centres", (step.get("centres") or {}).get("tied", ()),
                     "more than one subgroup of the largest order carries the orbital "
                     "centres within the declared precision"),
                    ("projective class", (step.get("projective") or {}).get("tied", ()),
                     "the states carry an ordinary representation on more than one "
                     "subgroup of the largest order")):
                if len(ranked) < 2:
                    continue
                others = {int(i) for r in ranked[1:] for i in r} - {
                    int(i) for i in ranked[0]}
                left += [{"environment": where, "ind": int(b["ind"]),
                          "name": str(b["name"])}
                         for b in broken if str(b.get("refuted_by")) == cause
                         and int(b["ind"]) in others]
                words.append(f"in the {where} environment, where {said}")
    return tuple(left), tuple(words)


def _environment_candidates(rec) -> tuple:
    r"""The groups one environment's measurement leaves: the one returned, and the tie's others.

    Where the round that settled the group left several subgroups of the
    largest order, the tie rule kept the first, whatever made the tie -- the
    survivors that do not close, the readings of a declared content, the
    subgroups that carry the orbital centres within a declared precision, or
    those on which the states carry an ordinary representation -- and the
    others are what the measurement does not tell from it.  Each is read in
    the group that round lowered: its operations the round measured to hold
    and broke for no reason but the choice -- the tie itself, and a closure
    that follows from the one kept -- the largest subgroups inside them where
    they do not close, and beside each operation its partner with time
    reversal where the group returned holds time reversal.  A tie at an
    earlier round was followed by a measurement of the one kept alone, and the
    others there were never measured as groups, so it offers none.  ``rec`` is
    the environment's selection record.

    Returns ``((label, group), ...)``: the group returned first, labelled by its
    name and number, and each other candidate labelled by its operations
    without time reversal.
    """
    from types import SimpleNamespace

    group = rec["group"]
    out = [(f"{group.name} #{group.number_str}", group)]
    steps, parents = tuple(rec.get("descent") or ()), tuple(rec.get("parents") or ())
    held = tuple(rec.get("held") or ())
    if not steps or len(parents) != len(steps) or len(held) != len(steps):
        return tuple(out)
    step, parent, measured = steps[-1], parents[-1], held[-1]
    kept = {int(i) for i in step["kept"]}
    causes = {}
    for b in step["dropped"]:
        causes.setdefault(int(b["ind"]), set()).add(str(b["refuted_by"]))
    by_ind = {int(s.ind): s for s in parent.symmetries}
    lattice = np.asarray(parent.lattice, dtype=float)
    partner = ({_operation_key(s)[:2]: s for s in parent.symmetries if s.time_reversal}
               if any(bool(s.time_reversal) for s in group.symmetries) else {})
    seen = {frozenset(kept)}
    for cause, ranked in (("tie", (step.get("tie") or {}).get("tied", ())),
                          ("declaration", (step.get("declaration") or {}).get("tied", ())),
                          ("centres", (step.get("centres") or {}).get("tied", ())),
                          ("projective class", (step.get("projective") or {}).get("tied", ()))):
        # what the choice alone left out: the operations the round measured to
        # hold and broke by this tie, or by a closure that follows from the one
        # it kept, and by nothing else
        open_to = kept | {i for i, why in causes.items()
                          if why <= {cause, "closure", "tie"} and i in measured}
        for alternative in tuple(ranked)[1:]:
            inside = open_to & {int(i) for i in alternative}
            if _descent_closure(parent, [by_ind[i] for i in inside]):
                resolved, _dropped, tied = _resolve_closure(parent, sorted(inside))
                subs = tied or (resolved,)
            else:
                subs = (inside,)
            for sub in subs:
                sub = frozenset(int(i) for i in sub)
                if sub in seen:
                    continue
                seen.add(sub)
                unitary = [by_ind[i] for i in sorted(sub)]
                ops = unitary + [partner[_operation_key(s)[:2]] for s in unitary
                                 if _operation_key(s)[:2] in partner]
                label = "{" + ", ".join(_record_name(s, lattice) for s in unitary) + "}"
                out.append((label, SimpleNamespace(
                    symmetries=tuple(ops), lattice=parent.lattice,
                    vacuum_dirs=parent.vacuum_dirs, number_str="", name=label,
                    size=len(ops))))
    return tuple(out)


def _environment_explained(free, sub):
    r"""The first pair of candidates the subgroup statement holds for, the pair returned aside.

    ``free`` and ``sub`` are the two environments' candidates
    (:func:`_environment_candidates`), each the group returned first; the
    pairs that keep one side's returned group are held first.  Returns
    ``(free-standing label, substrate label)``, or ``None`` where the statement
    holds for no pair.
    """
    pairs = sorted(((i, j) for i in range(len(free)) for j in range(len(sub)) if i or j),
                   key=lambda ij: (ij[0] > 0 and ij[1] > 0, ij))
    for i, j in pairs:
        if _environment_record(free[i][1], sub[j][1])["agrees"] is True:
            return free[i][0], sub[j][0]
    return None


def _model_sectors(model, group, assignment, spin, validated, *, tolerances, context=None):
    r"""The sectors of the model's conserved matrices acting within sites, read once.

    Read off the identity's solution space the detector measured
    (:func:`.._sectors.read_sectors`), at the declared precision, and kept on
    ``context`` for every later round and the other environment of a layer,
    which measure the same Hamiltonian on the same sites: the key is the model's
    content, its sites as sets of orbitals, the declared spinor pairs and the
    tolerance.  ``None`` where the space is not held.
    """

    key = None
    if context is not None and context._key is not None:
        key = (context._key[0],
               tuple(sorted(tuple(sorted(int(o) for o in orbs))
                            for orbs in assignment.site_orbitals)),
               None if spin is None else tuple(tuple(int(o) for o in pair)
                                                for pair in spin.pairs),
               float(tolerances.matrix))
        if key in context.sectors:
            return context.sectors[key]
    got = read_sectors(model, group, assignment, spin, validated,
                       tol=tolerances.matrix or None)
    if key is not None:
        context.sectors[key] = got
    return got


def _realized_core(model, cell, cell_typat, *, labels, spinor, tolerances, nk, params,
                   context, spin, descend, symmetrize, max_bytes, build_kw,
                   fully_periodic, returned, declared_wider=None, free_standing=None,
                   centres_read=None):
    r"""The group this Hamiltonian obeys — three steps, rounded until they settle.

    One loop around the three the measurement is, and nothing in it knows what
    dimension the model has.

    1. **The group of the cell**: ``typat``, whatever species the environment put
       there and ``decorations=``, identified together before any Hamiltonian is
       read (:func:`_group_of`), and then every reduction ``characters=`` makes
       to that cell (:func:`_declared_cell_reduction`) — two sites of one orbit
       declared different content are different sites, and a declared *function*
       leaves only the operations that carry its span onto itself.  Those
       operations are gone from the cell, so no verdict below ever meets them;
       the reduced cell is identified once, with provenance ``realized``, and is
       the cell every later descent builds on.  The orbitals are assigned to its
       Wyckoff orbits.  Where a tolerance for the centres is declared
       (``tolerances.centre``) they were made exact before the cell was handed
       in (:func:`_exact_centres`, whose record is ``centres_read``), so the cell
       is identified on exact centres; where the whole group did not carry them
       within it, the position pass runs here instead -- the lossless step, and
       before the detector, so that what the detector reads is the hoppings' own
       defect and not the geometry's -- and the cell is built again on the
       centres it moved (:func:`_position_pass`).  Where nothing declares it, the
       centres are measured here all the same, and a defect that alone would
       break operations at the detector's tolerance is refused before anything
       is built (:func:`_centres_not_exactly_symmetric`).
    2. **One verdict per operation** of that group, the antiunitary coset
       included: the detector's own (:func:`_validate_group`), the tensor
       structure of the operations it keeps -- on a spinful model each one it
       found obeyed held to its own spin lift first, and each one it leaves
       undetermined told apart from its partners -- the **declared
       orbital content** where ``characters=`` states one, the direct
       measurement of a pure translation, and last, on a spinless reading whose
       every operation those leave standing, the **projective class** of the
       states -- operations no single-valued representation on them carries
       (:func:`_operation_verdicts`).  A
       declaration is read in the site tables of the group the loop is on and
       carried from there, re-keyed through every re-identification below: what
       the orbitals *are* is stated once, and the operations a smaller group no
       longer has leave the constraint.  The same reading also **pins** what
       the hoppings fix: the scalar each orbit of multiplets carries relative
       to the anchor, and the coupled parts of the crystal.  What they leave
       open it does not choose.  The reading leaves here on the record and
       :func:`.._build._build_rep` assembles from it rather than reading a
       solution space again, the labelling chosen there once, by the
       lowest-angular-momentum rule among the characters the declarations
       leave.
    3. **The realized group**: the operations measured to hold, resolved to a
       group where the measurement leaves them open, carried by a cell of their
       own and identified (:func:`_descend_once`), with the orbitals re-assigned
       to it.

    Then round again, because the smaller group's k-action classes are finer and
    an operation that shared a verdict can be refuted on its own.  Every step
    strictly shrinks the group, so the chain ends — at ``P1`` at worst.

    Two further measurements stand between rounds, and both re-enter the same
    loop.  Time reversal is measured on the group the hoppings settled on
    (:func:`_time_reversal_measured`) -- the detector's verdict, and its square
    against the sign the tables read here ask for, ``+1`` single-valued and
    ``-1`` double-valued: broken, the same cell is re-identified as the unitary
    group it is, because a grey record with rows deleted would be neither.
    Each round keeps what it measured of the operations combined with time
    reversal it asks last (:func:`_antiunitary_settled`) -- each descent the
    ones whose unitary part it leaves out, the group time reversal is measured
    on all of its own -- and once the loop settles they are carried onto the
    group it settled on (:func:`_carried_onto`), to be read once against it
    where its representation is built (:func:`_beyond_grey`): the descent the
    projective class makes comes after time reversal is measured.  And
    an operation obeyed within the declared covariance gate
    (``tolerances.energy``) but not to the exact gates is **near**-symmetric,
    which is the signal that licenses projecting the Hamiltonian onto its
    group-invariant part (:func:`..symmetrization.symmetrize`) and measuring
    everything again on the projected model; so are multiplets that hold only
    within the declared tolerance, which the multiplet measurement records, and
    coupled parts of the crystal the hoppings join only below it, which the
    projection then holds exactly apart.

    ``fully_periodic`` and ``free_standing`` are what a layer's environment
    decides and this measurement is handed (:func:`_layer_realized`): whether
    the cell has a side at all, which the refusal of a cell group the
    Hamiltonian does not realize words for it, and on a substrate the
    free-standing cell of the same point set, on which the states' projective
    class is read once and from which the substrate carries its part that
    preserves the layer's normal (:func:`.._descent._projective_class_carried`); ``None`` elsewhere.  The tensor
    test is the same in every environment.
    ``returned`` is false for the selection the
    cross-check makes, whose group is a comparison and is never returned, so the
    magnetic-scope warning belongs to the other one, and a declaration its site
    groups name nothing for is read where its words were written and restricted
    to its own site groups, or, where it names nothing there either, recorded
    rather than refused.  ``spin`` is the
    declared :class:`..spin.SpinBasis` itself, resolved where the call started, or
    ``None``, and every step here is handed that record.  ``tolerances`` is the
    :class:`.._tolerance.Tolerances` the call resolved from its declaration,
    and every judgement of noise here reads it: the covariance gate, the
    detector's cut and window, the centres' tolerance, the multiplets' block
    tests (``tolerances.matrix``, :func:`.._tolerance.noise_floor`) and the
    symmetry finder's distance.  ``max_bytes`` is the memory the hopping projection
    may take at its peak, handed to :func:`..symmetrization.symmetrize`, and
    the ceiling the detector may raise its own budget to on a spectrum with
    exactly degenerate levels, handed to :func:`_validate_group`; the
    detector's window valve does not move with it.  ``declared_wider`` is
    step 1's reading of the declaration in a larger group of the same point
    set, a layer's free-standing group handed in for the layer on a substrate
    (:func:`_layer_declared_reading`), whose exclusions step 1 carries onto
    this cell (:func:`_carried_reduction`); on the cross-check's selection it
    is also where words this cell's site groups name nothing for are read and
    restricted from (:func:`_restricted_contents`).  ``None`` where there is
    none.

    Returns the record the measurement ends in — the model as measured, the cell
    and its species, the group, the assignment, the detector's verdicts, the
    step 1 ``"reduction"`` where the declared content made one, the descent
    steps, the group each of them lowered (``"parents"``) and the operations
    its round measured to hold (``"held"``), the position and
    symmetrizing passes, the time-reversal verdict,
    what the rounds measured of the operations combined with time reversal
    (``"antiunitary"``, carried onto the group returned where time reversal is
    broken), the last round's own reading of the solution spaces (``"choices"``, which
    the representation is then assembled from), the sectors the conserved
    matrices acting within sites split the states into, read on the first round
    and carried through every later one (``"sectors"``,
    :func:`.._sectors.read_sectors`) and, on the cross-check's selection, the
    refusal a declaration this environment's site groups name nothing for would
    have met (``"unread"``, ``None`` where the words were read) — or a
    :class:`..refusal.Refusal`.
    """

    group = _group_of(cell, typat=cell_typat, spinor=bool(spinor),
                      symprec=tolerances.finder)
    if isinstance(group, Refusal):
        return group                     # no group yet, so none to name
    # the inner identification is a cell identification, but the flow that owns
    # this record measured it: the outer stamp wins, and it is applied *here* so
    # that everything downstream -- the assignment, the detector, and above all
    # `rep.group`, which is what `characters()` echoes -- carries it.
    group = replace(group, provenance="realized")

    # The cell as described, kept for the whole call: every descent below
    # rebuilds from THIS cell plus one generic orbit, never from a previous
    # descent's cell, so a second descent's orbit is the only one in the record.
    base_cell, base_typat = cell, list(cell_typat)
    steps: list = []
    # the group each step lowered, and the operations its round measured to hold
    # before anything else left one out, in which a tie's other candidates are
    # read when the two environments of a layer are compared
    parents: list = []
    held: list = []

    def _stamp(g):
        """The provenance of a group returned from here: descended, or realized."""
        return replace(g, provenance="descended" if steps else "realized")

    positions = symmetrization = validated = None

    def _refused(refusal, settled=False):
        """A refusal made here, the group it was reached on named in it.

        Before the verdicts have kept every operation of the group, that is the
        group the cell has at this point, an upper bound on what the Hamiltonian
        obeys; once they have (``settled``: the projection, and the declaration
        read at the end), the group identified.  A position defect measured and
        not removed is quoted.
        """
        return _quote_position_defect(
            _named_refusal(refusal, group, validated if settled else None),
            positions, symmetrization)

    assignment = _assign_orbitals(model, group, typat=labels)
    if isinstance(assignment, Refusal):
        return _refused(assignment)
    # centres the whole group carried within the declaration were moved before
    # the cell was identified, and that move is the record's
    if centres_read is not None and centres_read["branch"] == "carried":
        positions = dict(centres_read["positions"])

    layout = choices = None
    time_reversal = None
    # the sectors the conserved matrices acting within sites split the states
    # into, read once where the model enters, on the first round's solution
    # spaces, and carried through every later round
    sectors = None
    sectors_read = False
    # what each round measured of the operations combined with time reversal
    # it asks last (`_antiunitary_settled`), kept for the one reading of them
    # once the group settles
    asked: list = []
    contents: tuple = ()
    # the declared words read in a wider site group where the surviving
    # operations' own cannot name them (`_declared_contents_wider`), carried
    # beside `contents` through every re-identification
    wider: tuple = ()
    pending = None
    # the cross-check's selection only: a declaration this environment's site
    # groups name nothing for, read nowhere, which is not compared
    # (`_layer_realized`)
    unread = None
    # ... and, there, the larger group of the same point set the words were
    # written in, which they are read in and restricted from where this
    # environment's site groups name nothing for them
    written_in = None if returned else declared_wider
    reduction = None
    read_in: dict = {}
    reduce_pending = build_kw.get("characters") is not None
    reversal_measured = reversal_head = weighed = projected = False
    centres_measured = False
    while True:
        # STEP 1, and before anything reads a hopping: the reductions the
        # DECLARED CONTENT makes to the cell itself.  Two sites of one orbit
        # declared different content are different sites, and a declared
        # function whose span its site group does not preserve leaves only the
        # operations that do -- one rule, decided here, and never asked again:
        # the operations it removes are gone from the group, so no verdict
        # below has anything to find there.  Where the declaration cannot be
        # read in THIS cell's site groups it is carried, exactly as the
        # resolution below carries it, and asked again on the next group.
        if reduce_pending:
            got = _declared_cell_reduction(model, group, assignment, spin=spin,
                                           characters=build_kw["characters"])
            if declared_wider is not None:
                # ... and what the same words say in the larger group of this
                # point set, carried onto the cell as it stands before anything
                # is read: once, since both are read on the cell itself
                got = _carried_reduction(group, assignment, got, declared_wider)
                declared_wider = None
            if not isinstance(got, Refusal):
                kept, excluded, where_read = got
                reduce_pending = False
                # WHERE THE DECLARED WORD WAS READ is worth carrying only where
                # the cell moved under it: with nothing excluded the resolution
                # below reads it in the same place and says so once.
                read_in = dict(where_read) if excluded else {}
                if excluded:
                    made = _declared_cell(base_cell, base_typat, group, kept,
                                          spinor=bool(spinor),
                                          include_TR=bool(group.include_TR),
                                          symprec=tolerances.finder)
                    if isinstance(made, Refusal):
                        return _refused(made)
                    lowered, cell, cell_typat, pairs, point, orbit, species = made
                    reduction = {
                        "from": {"number_str": str(group.number_str),
                                 "name": str(group.name), "size": int(group.size)},
                        "to": {"number_str": str(lowered.number_str),
                               "name": str(lowered.name), "size": int(lowered.size)},
                        "excluded": tuple(excluded[i] for i in sorted(excluded)),
                        "n_excluded": len(excluded),
                        "kept": tuple(sorted(int(i) for i in kept)),
                        "generic_point": tuple(float(x) for x in point),
                        "orbit": tuple(tuple(float(x) for x in row) for row in orbit),
                        "species": species, "pairs": pairs}
                    # a cell reduction, not a descent: the group returned is the
                    # declared crystal's own cell, identified once
                    group = replace(lowered, provenance="realized")
                    base_cell, base_typat = cell, list(cell_typat)
                    # anything already resolved follows the cell, through this
                    # identification's own pairing, exactly as it follows a
                    # descent: a content is resolved once and carried, never
                    # resolved again in a site group it may not belong to
                    contents = _rekey_contents(contents, pairs, {})
                    wider = _rekey_contents(wider, pairs, {})
                    assignment = _assign_orbitals(model, group, typat=labels)
                    if isinstance(assignment, Refusal):
                        return _refused(assignment)
                    continue
        # The CENTRES, next -- and only where a tolerance for them was declared
        # (`tolerances.centre`, a precision's `p` or `position_tol`), which says
        # what "noise" means for them, and the whole group the cell allows did
        # not carry them within it, so that they were not moved before the cell
        # was identified.  This is the lossless step (a k-dependent gauge,
        # spectrum untouched) and it precedes the lossy one (the hopping
        # projection), which could not remove a position defect anyway: in
        # convention I centres off by `delta` cost H(k) a covariance residual of
        # order 2*pi*delta*max|H| whatever the hoppings do.  The centres are
        # moved, within the declaration, onto the positions of the largest group
        # of this round's operations that carries them within it, and the group
        # is written in the origin they then satisfy; the operations outside
        # that group are the centres' verdict, read in the verdict stage below
        # like any other.  It runs once, and the cell every later identification
        # is built on is built again on the moved centres, so that each of them
        # reads centres that are exact.
        centres = None
        if (tolerances.centre is not None
                and not (positions or {}).get("adjusted")):
            got = _position_pass(model, group, tolerances, labels, params, centres_read)
            if isinstance(got, Refusal):
                return _refused(got)
            before = group
            model, group, assignment, positions, centres, extras_now = got
            env_cell, env_typat = centres_read["rebuild"](model, extras_now)
            base_cell, base_typat = _cell_carried(
                env_cell, env_typat, base_cell, base_typat,
                len(centres_read["frame"]["points"]),
                (np.asarray(group._shiftUC, dtype=float)
                 - np.asarray(before._shiftUC, dtype=float)) * centres_read["frame"]["scale"])
            cell, cell_typat = base_cell, list(base_typat)
            if free_standing is not None:
                # the free-standing cell of the same point set, on the moved centres
                free_standing["cell"], free_standing["cell_typat"] = _decorated_cell(
                    model, free_standing["labels"], environment="free-standing",
                    decorations=extras_now)
                free_standing.pop("group", None)
        elif tolerances.centre is None and not centres_measured:
            # ... and where nothing declares what noise means for the centres,
            # they are measured all the same, once: a defect that alone breaks
            # operations at the detector's tolerance would make every verdict
            # below the geometry's as much as the hoppings', so it is refused
            # here, before anything is built, with the number to act on.
            centres_measured = True
            refused = _centres_not_exactly_symmetric(model, group, assignment,
                                                     tol=tolerances.energy,
                                                     params=params)
            if refused is not None:
                return _refused(refused)
        got = _validate_group(model, group, nk=nk, tol=tolerances.energy,
                              cut=tolerances.cut, window=tolerances.window,
                              max_bytes=max_bytes, context=context, params=params,
                              _floor=tolerances.floor)
        if isinstance(got, Refusal):
            # the detector's memory limit: fired before the measurement settles,
            # so what is named is the cell's group at this point, an upper bound
            return _refused(got)
        validated = got
        # EACH OPERATION'S OWN FORM, read once and before anything reads a
        # solution space: of every space the detector kept, the members carrying
        # each site's states onto its image's, with the operation's own spin lift
        # on states that carry a spin.  Everything below reads those, and an
        # operation none of whose members has its own form is refuted by it.
        covariance = _grid_covariance(model, group, nk=nk, params=params, context=context)
        validated, own_form = _own_form_read(model, group, assignment, spin, validated,
                                             tolerances=tolerances, covariance=covariance)
        # THE SECTORS, read once, on the first round: the identity's solution
        # space is every matrix that commutes with H, the same on every round,
        # and what it says about the states is said once
        if not sectors_read:
            sectors_read = True
            sectors = _model_sectors(model, group, assignment, spin, validated,
                                     tolerances=tolerances, context=context)
        # TIME REVERSAL'S MEMBER of the sign the tables ask for, where its solution
        # space holds one the detector's own is not: the intertwiner every step
        # below reads, the verdicts and the multiplets' blocks as much as the
        # representation, so that each reads one member.  Whether time reversal
        # is obeyed is still measured once, where the unitary group settles.
        square = _time_reversal_square(group, validated, tolerances=tolerances)
        validated = _time_reversal_member(validated, square)
        # WHAT A MULTIPLET IS, measured on this group's own solution spaces
        # before any verdict reads them: the finest partition of every site's
        # orbitals that still admits a block-form intertwiner, which is what
        # the tensor test, the symmetrizing pass and the representation are all
        # written on.  Re-measured after every descent, because a smaller group
        # glues less; what the conserved matrices join is joined on every round,
        # read off the sectors read once
        layout = _site_multiplets(model, group, assignment, spin, validated,
                                  declared=build_kw.get("multiplets"),
                                  tol=tolerances.matrix, sectors=sectors)
        # WHAT THE ORBITALS ARE, as `characters=` states it.  Resolved once, in the
        # group the loop is on, and carried from there: a declaration states what
        # each site holds and is read on the multiplets measured above, which it
        # never changes, and every re-identification below re-keys the
        # constraint rather than resolving the label again.
        decls = ()
        if build_kw.get("characters") is not None:
            # once resolved, the contents themselves are what is read on every
            # later round: the constraint is carried and restricted, never the
            # label resolved again in a site group it may not belong to
            stated = ({tuple(int(o) for o in c.orbitals): c for c in contents}
                      if contents else build_kw["characters"])
            got = _declared_contents(model, group, assignment, validated, layout,
                                     spin=spin, characters=stated, read_in=read_in)
            if (isinstance(got, Refusal) and getattr(got, "gate", None) == "unknown_label"
                    and not contents and written_in is not None):
                # THE CROSS-CHECK'S SELECTION, of words written in the larger
                # group's site irreps -- `A2''` of a free-standing layer's
                # `-6m2`, which a substrate's `3m` names nothing for: they are
                # read where they were written, on the multiplets measured here,
                # and restricted to this group's site groups by re-keying, as
                # after any descent; from here on they are carried as contents
                restricted = _restricted_contents(model, written_in, group, layout,
                                                  spin=spin,
                                                  characters=build_kw["characters"])
                if restricted:
                    got = _declared_contents(
                        model, group, assignment, validated, layout, spin=spin,
                        characters={tuple(int(o) for o in c.orbitals): c
                                    for c in restricted},
                        read_in=read_in)
            if isinstance(got, Refusal):
                # A declaration is read in the site tables of the group the loop
                # is ON, and a group the Hamiltonian is about to break can have
                # site groups the declaration's own words do not belong to -- a layer
                # presented as a slab carries sigma_h, whose `-6m2` site group
                # has no `A1` at all, and the in-plane `3m` the declaration means is
                # two operations away.  So an unreadable declaration is carried
                # rather than refused, and asked again on the smaller group.
                pending = got
                if not contents and not wider:
                    # ... and read, the first time, where the words do name a
                    # content: the site groups of this group, not only of the
                    # operations it keeps, or their in-plane part.  Carried
                    # beside the words, it is what the smaller group reads if
                    # the loop settles with the words themselves still naming
                    # nothing there.
                    wider = _declared_contents_wider(
                        model, group, assignment, layout, spin=spin,
                        characters=build_kw["characters"], read_in=read_in)
            else:
                pending = None
                decls, layout = got
                contents = _declared_parts_of(decls)
        got = _operation_verdicts(model, group, assignment, validated,
                                  tolerances=tolerances,
                                  nk=nk, params=params, context=context, spin=spin,
                                  layout=layout, decls=decls,
                                  reference_orbital=int(
                                      build_kw.get("reference_orbital", 0) or 0),
                                  max_candidates=int(
                                      build_kw.get("max_candidates", 2 ** 20)
                                      or 2 ** 20),
                                  free_standing=free_standing,
                                  centres=(centres or {}).get("refuted"),
                                  sectors=sectors, own_form=own_form)
        if isinstance(got, Refusal):
            # fired before the group is settled: what is named is the cell's
            # group at this point, an upper bound -- further rounds may lower it
            return _refused(got)
        refuted, lattice, declaration, choices, projective = got
        if lattice and lattice["realized"]:
            return _refused(_supercell_refusal(group, validated, lattice))
        # An antiunitary operation refuted here has nothing for a UNITARY descent
        # to lower, so it does not start one: time reversal is its own
        # measurement, made on the group this settles to.  It is recorded
        # whenever a descent does run.
        antiunitary = {int(s.ind) for s in group.symmetries if s.time_reversal}
        if any(i not in antiunitary for i in set(validated.broken) | set(refuted)):
            # The *unitary* part is what the cell's own claim rests on: a cell
            # that allows operations the Hamiltonian breaks is the wrong cell, so
            # the cell is lowered to the operations that were measured to hold
            # and only that group is ever returned.
            if not descend:
                # Where the DECLARATION is the whole verdict, the honest refusal
                # is the one that names the subgroup the declared content is
                # covariant on -- which is exactly what `_build_rep` says when it
                # is handed a fixed group, and this call was told not to lower
                # one.  A Hamiltonian that also breaks operations by its own
                # hoppings is a different statement, and keeps its own refusal.
                if (declaration is not None and declaration["refuted"]
                        and not (set(validated.broken) | set(refuted))
                        - set(declaration["refuted"])):
                    return _refused(_declaration_breaks(
                        group, assignment, decls, choices,
                        standing=validated.indices,
                        subgroup=declaration["subgroup"],
                        broken=declaration["refuted"],
                        readings=declaration["candidates"],
                        not_lowered=("This call was told `descend=False`, "
                                     "which asks for the group the cell "
                                     "allows, so the descent was not made.")))
                return _refused(_cell_group_not_realized(group, validated,
                                                         fully_periodic, refuted))
            got = _descend_once(base_cell, base_typat, group, validated,
                                spinor=bool(spinor),
                                include_TR=bool(group.include_TR),
                                symprec=tolerances.finder, step=len(steps) + 1,
                                refuted=refuted, lattice_record=lattice,
                                declaration=declaration, projective=projective,
                                centres=centres)
            if isinstance(got, Refusal):
                return _refused(got)
            # THE OPERATIONS COMBINED WITH TIME REVERSAL the descent leaves
            # behind, whose unitary part it does not keep: the group it arrives
            # at never asks them, so this round's verdicts are their last, kept
            # for the one reading of them once the group settles
            left = _antiunitary_settled(group, validated, refuted, square,
                                        got[3]["kept"], tolerances=tolerances)
            if left["n"]:
                asked.append(left)
            parents.append(group)
            held.append(frozenset(int(i) for i in validated.indices))
            group, cell, cell_typat, record = got
            steps.append(record)
            group = _stamp(group)
            # The declared contents follow the cell: the constraint is re-keyed
            # onto the new group's own `.ind` through the step's own pairing, and
            # the operations the smaller group no longer has drop out -- which IS
            # the restriction of the declared irrep to the smaller site group.
            contents = _rekey_contents(contents, record["pairs"], refuted)
            wider = _rekey_contents(wider, record["pairs"], refuted)
            assignment = _assign_orbitals(model, group, typat=labels)
            if isinstance(assignment, Refusal):
                return _refused(assignment)
            continue

        # The Hamiltonian obeys every unitary operation of the group it holds.
        # Time reversal is measured once, here: its detector verdict, and its
        # square against the sign the tables read here ask for.  Where it is
        # broken, what this round measured of every operation combined with time
        # reversal is kept beside what each descent measured of the ones it left
        # behind, for the one reading of them once the group settles.
        if not reversal_measured:
            reversal_measured = True
            time_reversal, reversal, validated = _time_reversal_measured(
                group, validated, refuted, tolerances=tolerances)
            if reversal is not None:
                asked.append(reversal)
            if time_reversal["outcome"] != "kept":
                if returned:
                    # Said once per call, about the group that is returned: the
                    # cross-check's other environment runs this same branch on
                    # the same Hamiltonian, and the group it reaches is a
                    # comparison and is never returned.
                    warnings.warn(
                        "pythtqc: time reversal is broken by this Hamiltonian, so only the "
                        "unitary (type-I) subgroup of its magnetic symmetry group is returned. "
                        "Antiunitary operations of the cell that hold are named on "
                        "beyond_grey, and those beyond the cell's own are not searched for; "
                        "the labels, EBR rows and indicator that follow are the unitary "
                        "group's own.",
                        MagneticScopeWarning, skip_file_prefixes=(_PACKAGE_DIR,))
                # Re-identify the same cell as the unitary group it is.  A fresh
                # identification, never the grey record with rows removed: the
                # tables the labels come from have to be the returned group's own.
                unitary = _group_of(cell, typat=cell_typat, spinor=bool(spinor),
                                    include_TR=False, symprec=tolerances.finder)
                if isinstance(unitary, Refusal):
                    return _refused(unitary)
                # a fresh identification stamps a fresh `.ind` order, so the
                # declared contents are re-keyed onto it before anything reads
                # them again; and it reads its translations off the cell again,
                # so it is carried onto the origin of the operations it
                # identifies again, the one the centres satisfy
                held = [s for s in group.symmetries if not s.time_reversal]
                pairs, lost, extra = _match_operations(
                    held, [s for s in unitary.symmetries if not s.time_reversal],
                    tolerances.finder, group.lattice)
                if not lost and not extra:
                    unitary = _shared_origin(unitary, held, pairs)
                contents = _rekey_contents(contents, pairs, {})
                wider = _rekey_contents(wider, pairs, {})
                group = _stamp(unitary)
                assignment = _assign_orbitals(model, group, typat=labels)
                if isinstance(assignment, Refusal):
                    return _refused(assignment)
                reversal_head = True
                continue
        if reversal_head:
            reversal_head = False
            time_reversal["number_str"] = str(group.number_str)
            time_reversal["name"] = str(group.name)
            time_reversal["size"] = int(group.size)

        # The symmetrizing pass, on the signal that licenses it: a measurement
        # that holds within the DECLARED tolerance but not exactly.  Either an
        # operation the Hamiltonian obeys within it but not to the exact
        # pipeline's own gate, or multiplets that hold only within it -- an
        # orbital frame off the one the operations carry by less than the
        # declared precision, whose multiplets the Hamiltonian's own symmetry
        # matrices mix at the size of the turn, which the multiplet measurement
        # records (`SiteMultiplets._near`) and which is read here, not measured
        # again -- or parts of the crystal the verdict read apart, each with a
        # one-dimensional character of its own, that the hoppings still couple
        # below the declared precision, where a coupling counts as absent -- or
        # identical copies the sectors read at that precision, which the
        # hoppings couple or tell apart below it.
        # Everything after this point then runs on the symmetrized model, at
        # the exact gates -- nothing downstream is loosened.
        if not weighed:
            weighed = True
            # against the exact gate the representation of THIS model and group
            # will be built with, measured once: the detection record carries
            # neither the number of states nor max|H|
            exact_gate = (None if symmetrize is False
                          else _exact_gate(model, group, validated, build_kw, params))
            classes = ({} if symmetrize is False
                       else classify_operations(validated, tolerances.energy,
                                                exact_gate=exact_gate))
            # THE PARTS OF THE CRYSTAL THE VERDICT READ, where it read several
            # the sectors read at the declared precision also keep apart, and
            # the largest coupling the Hamiltonian holds between two of them:
            # the build gives each part a character of its own, which leaves
            # every covariance as it was only where the parts are exactly apart
            apart = None if symmetrize is False else _parts_apart(choices, layout, sectors)
            coupling = 0.0 if apart is None else _parts_coupling(model, apart, params)
            coupled = apart is not None and coupling > exact_gate
            # ... and what the sectors read at the declared precision keep
            # apart -- IDENTICAL COPIES, and SECTORS SPLITTING MULTIPLETS that no
            # operation exchanges -- and how far the Hamiltonian is from keeping
            # them apart exactly: a representation of the copies that mixes
            # them is covariant only on exact copies, and one carried by each
            # such sector on its own only where nothing couples them
            held_apart = (None if symmetrize is False
                          else _held_sets(sectors, layout, model, group, assignment, spin,
                                          validated))
            off = _held_off(model, held_apart, params)
            alike = held_apart is not None and max(off.values()) > exact_gate
            # the DECLARED CONTENTS as they stand on this group, not the words
            # they were written as: the projection builds a representation, and
            # a label read in the cell's group need not name anything in the one
            # the loop has reached.
            sym_kw = {**build_kw, "multiplets": layout}
            if contents:
                sym_kw["characters"] = {
                    tuple(int(o) for o in c.orbitals): c for c in contents}
            # NEAR, READ ON THE MATRICES THE PROJECTION IS BUILT WITH.  The
            # detector reads any member of an operation's solution space, so a
            # basis turned inside a site -- a spin frame off the tables', orbitals
            # off the multiplets the representation is written on -- is exact to
            # it, while the matrix the tables' spin lift and those multiplets
            # anchor misses H by the turn.  Where nothing has said near yet, that
            # representation is built, up to the labelling, which no covariance
            # sees, and its covariance read operation by operation; the
            # projection, where it runs, builds it whole.  An operation exact to
            # the detector and on its own matrix stays exact; a build that cannot
            # be made here reads nothing, and the build on what settles says what
            # it says.
            if (symmetrize == "auto" and not layout._near and not coupled and not alike
                    and not any(c == "near" for c in classes.values())):
                provisional, _record = _projection_operators(
                    model, group, assignment, validated, tol=tolerances.energy,
                    params=params, build_kw={**sym_kw, "spin": spin, "_labelled": False})
                if provisional:
                    classes = classify_operations(
                        validated, tolerances.energy, exact_gate=exact_gate,
                        anchored=provisional.residuals)
            if (symmetrize is True or any(c == "near" for c in classes.values())
                    or (symmetrize == "auto" and (layout._near or coupled or alike))):
                got = _symmetrize(model, group, assignment, validated,
                                  tol=tolerances.energy, params=params, spin=spin,
                                  max_bytes=max_bytes,
                                  _parts=(None if apart is None else
                                          {"orbitals": apart, "coupling": coupling}),
                                  _held=({"sets": held_apart, "off": off} if alike else None),
                                  **sym_kw)
                if isinstance(got, Refusal):
                    if got.gate == "rep_not_covariant":
                        # The provisional build's covariance gate IS the declared
                        # tolerance every operation was kept at, so a miss there is
                        # the consistency guard's case, as the final build's is --
                        # on a free-standing layer as anywhere: where the hoppings
                        # cannot tell an operation from its product with the
                        # horizontal mirror, the mirror's chosen sign gives the
                        # product its partner's matrix, so both members hold.
                        _raise_not_covariant(
                            got, group, validated, steps,
                            where=(", built at the declared tolerance to project the "
                                   "Hamiltonian onto its invariant part,"))
                    # the projection's own verdict, or its provisional build's:
                    # every operation of this group was kept at the declared
                    # tolerance, so it is the group identified that is named
                    return _refused(got, settled=True)
                model, symmetrization = got
                # The symmetrized model is built from **numbers** at these
                # parameter values -- the inverse transform writes a hopping
                # list, not a symbolic one -- so nothing downstream may be handed
                # `params` again (pythtb raises on a parameter the model does not
                # have).
                params = {}
                assignment = _assign_orbitals(model, group, typat=labels)
                if isinstance(assignment, Refusal):
                    return _refused(assignment, settled=True)
                # A projection onto the group-invariant part can still leave an
                # operation refuted -- the projector is built from the operations
                # that were near-symmetric, not from the ones that were broken --
                # so the descent is available on exactly the same terms, and the
                # detector keeps the declared tolerance, because re-measuring at a
                # *tighter* one asks a different question and can answer it
                # wrongly: the eigen route's accuracy floor is
                # `eps*‖H‖ / (smallest level splitting)`, which symmetrization
                # does not change.  The default-tolerance re-detection is measured
                # all the same and is on `symmetrization.verification`.
                projected = True
                continue
        if projected and validated.broken:
            # only antiunitary operations can remain: the loop settles when every
            # unitary one is obeyed, and time reversal was measured before the
            # projection ran.
            return _refused(_cell_group_not_realized(group, validated, fully_periodic))
        if pending is not None:
            if (wider and not contents
                    and _restricts_to(wider, group, assignment)):
                # The group settled and its site tables still name nothing
                # the words say -- a descent that went straight past the
                # site group they were written in.  What they were read as
                # in the larger site group the loop was on, restricted to
                # this one by every re-identification since, is the
                # declaration here, as after any other descent: one more
                # round reads it, and the verdicts and choices are its own.
                contents, wider, pending = wider, (), None
                continue
            if returned:
                # the group settled and the declaration still cannot be read in
                # its site tables: now it is a grammar error about this crystal,
                # and the group that settled is named in it
                return _refused(pending, settled=True)
            # ... but this run is the CROSS-CHECK's selection: the same
            # Hamiltonian read in a layer's other environment, whose site groups
            # are not the ones the declaration was written in -- `-6m2` on a
            # free-standing honeycomb, which keeps sigma_h, against `3m` on a
            # substrate.  Words written in the larger group of the same points
            # -- the free-standing layer's, read on a substrate -- were read
            # where they were written and restricted here, and are compared.
            # Words read in no round at all, neither in this environment's site
            # groups nor by restriction, are the other environment's site
            # irreps -- a substrate's `3m` words, which a free-standing `-6m2`
            # names nothing for and no restriction reaches: they say which
            # environment is meant, and a reading made without them is of
            # another crystal.  So the two are not compared, which is recorded,
            # and nothing lifts the words into this environment's site irreps.
            if not contents and getattr(pending, "gate", None) == "unknown_label":
                unread = pending
            pending = None
        break

    # The declared contents as they stand on the group that settled, each
    # carrying the operations it broke on the way: `_build_rep` is handed THESE
    # rather than with the declaration as written, because a label read in the cell's
    # group need not exist in the returned one and the restriction is what the row
    # carries (`A2` of `3m` returned as `A` of `3`).
    declared = None
    if contents:
        broke = {}
        for st in steps:
            for b in st["dropped"]:
                if b.get("refuted_by") == "declaration":
                    for o in b.get("orbitals", ()):
                        broke.setdefault(int(o), []).append(str(b["name"]))
        contents = tuple(
            replace(c, broke=tuple(dict.fromkeys(
                n for o in c.orbitals for n in broke.get(int(o), []))))
            for c in contents)
        declared = {tuple(int(o) for o in c.orbitals): c for c in contents}
    # THE OPERATIONS COMBINED WITH TIME REVERSAL each round asked last, carried
    # onto the group that settled: where time reversal is broken they are read
    # once, against it, where its representation is built (`_beyond_grey`)
    if (time_reversal or {}).get("outcome") == "measured broken":
        asked = list(_carried_onto(group, asked, tolerances.finder))
    return {"model": model, "cell": cell, "cell_typat": cell_typat,
            "base_typat": base_typat, "group": group, "assignment": assignment,
            "validated": validated, "descent": tuple(steps), "parents": tuple(parents),
            "held": tuple(held),
            "positions": positions,
            "symmetrization": symmetrization, "params": params, "layout": layout,
            "time_reversal": time_reversal, "antiunitary": tuple(asked),
            "declared": declared, "choices": choices, "reduction": reduction,
            "unread": unread, "sectors": sectors}


def _exact_centres(model, cell, cell_typat, *, extras, rebuild, tolerances, params):
    r"""The centres made exact under a declared precision, before the cell's setting is read.

    A declared precision says each centre is known to ``p``
    (``tolerances.centre``), in reduced coordinates.  The cell's operations come
    first, read off the centres as they are by the symmetry finder at its own
    distance -- its non-magnetic search, since a cell of centres carries no
    moment and time reversal moves no centre, so that the grey group's
    operations are these and their products with it -- and which site each
    sends each site to is read there, once (:func:`_site_action`).  The centres
    are then projected onto the nearest set those operations carry exactly
    (:func:`_symmetric_sites`), which needs no table and fits the origin to them.

    Where that moves no centre further than ``p``, they are moved there.  A
    shift of every centre together that is noise -- at most ``p``, after which
    every translation is a rational of the model's own cell
    (:func:`_rational_offset`) -- is removed with it, so that the centres are
    exact in the model's own frame; a larger one, or none that makes the
    translations rational, says where the model's origin is, and the centres
    stay about the origin fitted to them.  The move is a k-dependent gauge
    transformation, and the spectrum is measured unchanged.  The cell is built
    again from the moved centres, so that the tables' setting, and everything
    after it, is read on centres that are exact.

    Where the whole group does not carry the centres within ``p``, they are not
    moved here: the cell is built on the centres projected onto the whole
    group's positions, which names the group the cell allows and nothing else,
    and the measurement moves them onto the largest group of its operations
    that carries them within ``p`` (:func:`_position_pass`), every operation
    outside it broken by the centres.  Where the centres cannot be read as a
    symmetric set at all -- the operations read off them do not act on the
    sites as a group does, as when two sites of one species sit closer together
    than the finder can tell apart -- nothing is projected, the cell is
    identified as it is, and the same pass reads the group it is identified as.

    ``rebuild`` builds the identification cell from a model and the ``extras``
    decorating it -- the environment's species and ``decorations=``, geometry
    declared beside the model, which moves with the centres.

    Returns ``(model, cell, cell_typat, read)``, ``read`` the record the
    measurement continues from -- ``branch`` (``"carried"``, ``"named"`` or
    ``"as found"``), ``frame`` (the cell's sites as handed in,
    :func:`_cell_sites`), ``source`` (the model as handed in), ``extras`` (the
    decorations as they now stand), ``rebuild`` and, where the centres were
    moved, ``positions`` -- or the refusal the spectrum check made.
    """
    import spglib


    p = float(tolerances.centre)
    extras = tuple(extras or ())
    frame = _cell_sites(cell, cell_typat, tolerances.finder, n_extra=len(extras))
    read = {"branch": "as found", "frame": frame, "source": model, "extras": extras,
            "rebuild": rebuild, "positions": None}
    if not len(frame["moving"]):                   # pragma: no cover - a model has orbitals
        return model, cell, cell_typat, read
    found = spglib.get_symmetry_dataset(
        (frame["lattice"], frame["first"], frame["species"]),
        symprec=float(tolerances.finder))
    if found is None:
        return model, cell, cell_typat, read
    rotations = np.asarray(found.rotations, dtype=float)
    translations = np.asarray(found.translations, dtype=float)
    action = _site_action(frame, rotations, translations)
    if action is None:
        return model, cell, cell_typat, read
    perm, shift = action
    rows = list(range(len(rotations)))
    x, trans, closure = _symmetric_sites(frame, perm, shift, rotations, rows)
    if not closure <= _EXACT_CLOSURE:
        return model, cell, cell_typat, read
    own = frame["own"]
    dim_r = int(model.lattice.dim_r)
    moves = _moves_of(frame, x)
    whole = float(np.abs(moves[:own]).max(initial=0.0))
    delta = _rational_offset(trans, rotations, rows, model.periodic_dirs)
    in_frame = delta is not None and within(float(np.abs(delta).max(initial=0.0)), p)
    final = x - delta if in_frame else x
    shifted = _moves_of(frame, final)
    raw = np.asarray(model.orb_vecs, dtype=float)
    orb = raw + shifted[:own, :dim_r]
    moved_extras = _moved_extras(extras, shifted, int(cell.norb) - len(extras), dim_r)
    if not within(whole, p):
        # the whole group does not carry them: the group the cell allows is
        # named on the centres it would put them at, and the centres themselves
        # are moved by the measurement, onto the operations that do
        named, named_typat = rebuild(_geometry(model, orb), moved_extras)
        read.update(branch="named", move=whole)
        return model, named, named_typat, read
    if np.array_equal(orb, raw):
        moved, resid, n_check = model, 0.0, 0
    else:
        (moved, resid), n_check = _with_centres(model, orb, params=params), _bloch.CHECK_K
    new_cell, new_typat = rebuild(moved, moved_extras)
    final_trans = np.array([final[perm[g, 0]] + shift[g, 0] - rotations[g] @ final[0]
                            for g in rows])
    defect = _frame_defect(frame, perm, rotations, final_trans, rows)
    offset = (0.0 - np.asarray(delta) * frame["scale"])[:dim_r] if in_frame else np.zeros(dim_r)
    record = {
        "defect": defect, "position_tol": p, "adjusted": True,
        "defect_before": defect,
        "defect_after": _frame_defect(frame, perm, rotations, final_trans, rows, sites=final),
        "max_move": whole, "offset": tuple(float(c) for c in offset),
        "frame": "model" if in_frame else "fitted",
        "spectrum_residual": float(resid), "n_check": int(n_check),
        "n_moved": int(np.count_nonzero(np.abs(orb - raw).max(axis=1) > 0.0)),
        "note": _moved_note(whole, offset, in_frame, delta, p, resid)}
    read.update(branch="carried", extras=moved_extras, positions=record)
    return moved, new_cell, new_typat, read


def _moved_note(move, offset, in_frame, delta, p, resid,
                where="the operations put them at") -> str:
    """What the move of the centres did, in words."""
    shift = float(np.abs(np.asarray(offset)).max(initial=0.0))
    moved_any = bool(move) or bool(in_frame and shift)
    if not moved_any and in_frame:
        return ("the centres sit exactly where the operations put them, in the model's own "
                "frame: nothing was moved.")
    said = (f"the centres were moved onto the positions {where}, none by more than "
            f"{move:.3g} (reduced), the origin fitted to them" if move else
            "the centres sit where the operations put them, the origin fitted to them")
    if in_frame and shift:
        said += (f", and every centre was shifted together by at most {shift:.3g}, within "
                 "the declared precision, which writes them exactly in the model's own "
                 "frame, every operation's translation a rational of the cell")
    elif in_frame:
        said += ", which is the model's own frame"
    elif delta is None:
        said += ("; no shift of all of them together makes every operation's translation a "
                 "rational of the cell, so the model's origin is not one the operations "
                 "single out, and they stay about the origin fitted to them")
    else:
        said += (f"; the shift of all of them together that would write them in the model's "
                 f"own frame, {float(np.abs(delta).max()):.3g}, is beyond the declared "
                 f"precision {p:.3g}, so it is no noise, and they stay about the origin "
                 "fitted to them")
    return said + ((": a k-dependent gauge transformation, so the spectrum is untouched "
                    f"(measured, {float(resid):.3g}).") if moved_any else
                   ": nothing was moved.")


def _centres_carried(group, frame, perm, shift, rotations, ind_of_row, tolerances) -> dict:
    r"""Which operations of the group carry the orbital centres within the declared precision.

    A declared precision ``p`` -- ``tolerances.centre``, a precision's own ``p``
    or a declared ``position_tol`` -- says that each centre is known to ``p``, in
    reduced coordinates: the centres are carried by a group of operations
    within it where projecting them onto the nearest set that group carries
    exactly, the origin fitted to them (:func:`_symmetric_sites`), moves no
    centre further than ``p``.  Where the whole group carries them, nothing is
    refuted.  Where it does not, the centres carry the **largest subgroup**
    that does -- the first by the tie rule every such choice is made by
    (:func:`_tie_ranked`) where several of that order tie, never their
    intersection -- and every operation outside it is broken by the centres,
    its primed partner with it, since time reversal moves no centre.  This is
    the centres' verdict on each operation, read as the declaration's is
    (:func:`.._descent._declaration_refuted`): each operation is told why, in its own
    numbers -- how far carrying the centres under it alone would move one, and,
    where that is within ``p``, that no group holding it and the operations kept
    carries them within it.

    ``perm`` and ``shift`` are the unitary operations' action on the sites of
    ``frame`` (:func:`_site_action`), row by row as ``ind_of_row`` keys them.

    Returns ``{"precision", "move", "kept", "refuted", "tied", "note"}``: the
    move the whole group asks, the ``.ind`` of the operations kept (all of them
    where the whole group carries the centres), the refuted operations keyed by
    ``.ind`` (``refuted_by = "centres"``, with ``miss``, ``precision`` and
    ``note``), the tied subgroups in the rule's order where several tied, and
    a note, empty where nothing is refuted.
    """
    p = float(tolerances.centre)
    syms = group.symmetries
    every = frozenset(int(s.ind) for s in syms)
    row_of = {int(i): r for r, i in enumerate(ind_of_row)}
    own = frame["own"]

    def move(ops):
        """The largest move the operations ``ops`` ask of a centre, the origin fitted."""
        rows = [row_of[int(i)] for i in ops if int(i) in row_of]
        x, _trans, _closure = _symmetric_sites(frame, perm, shift, rotations, rows)
        return float(np.abs(_moves_of(frame, x)[:own]).max(initial=0.0))

    whole = move(every)
    if within(whole, p):
        return {"precision": p, "move": whole, "kept": tuple(sorted(every)),
                "refuted": {}, "tied": (), "note": ""}
    # the primed partner of each unitary operation, through the group's own
    # product table: T * g, with T the pure time reversal, which moves no centre
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(syms)}
    ind_of = [int(s.ind) for s in syms]
    unitary = [int(s.ind) for s in syms if not s.time_reversal]
    t_pos = next((q for q, s in enumerate(syms) if s.time_reversal
                  and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                     np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    partner = ({} if t_pos is None else
               {u: ind_of[int(prod[t_pos, at[u]])] for u in unitary})

    def primed(sub):
        return set(sub) | {partner[u] for u in sub if u in partner}

    subs = _subgroups_inside(group, unitary)
    moves = {s: move(s) for s in subs}
    carrying = [s for s in subs if within(moves[s], p)] or [s for s in subs if len(s) == 1]
    top = max(len(s) for s in carrying)
    ranked = _tie_ranked(group, [s for s in carrying if len(s) == top])
    kept = frozenset(ranked[0])
    tied_in = set().union(*(set(c) for c in ranked[1:])) if len(ranked) > 1 else set()

    def cyclic(g):
        """The operations ``g`` generates, read off the product table."""
        out, x = {g}, g
        while True:
            x = ind_of[int(prod[at[g], at[x]])]
            if x in out:
                return out
            out.add(x)

    lattice = np.asarray(group.lattice, dtype=float)

    def name(i):
        return _record_name(syms[at[int(i)]], lattice)

    kept_names = "{" + ", ".join(name(i) for i in sorted(kept)) + "}"
    refuted = {}
    for u in unitary:
        if u in kept:
            continue
        own_move = move(cyclic(u))
        if not within(own_move, p):
            note = (f"the centres miss it by {own_move:.3g} (reduced): carrying them onto one "
                    f"another under it would move a centre that far, beyond the declared "
                    f"precision {p:.3g}")
        elif u in tied_in:
            note = (f"it carries the centres within the declared precision {p:.3g} on its "
                    f"own (a move of {own_move:.3g}), and so do the largest groups that hold "
                    f"it, but they tie with the one the tie rule keeps, {kept_names}, which "
                    "does not hold it")
        else:
            note = (f"it carries the centres within the declared precision {p:.3g} on its "
                    f"own (a move of {own_move:.3g}), but no group holding it and the "
                    f"operations kept, {kept_names}, carries them within it")
        rec = {"ind": int(u), "refuted_by": "centres", "miss": own_move, "precision": p,
               "note": note}
        refuted[int(u)] = rec
        if u in partner:
            refuted[int(partner[u])] = dict(rec, ind=int(partner[u]), note=(
                f"the primed copy of #{int(u)} {name(u)}, which the centres are not "
                "carried by: time reversal moves no centre, so the two stand or fall "
                "together"))
    note = (f"the orbital centres are not carried by the operations of "
            f"{group.name} #{group.number_str} within the declared precision {p:.3g}: "
            f"moving them onto its positions would move one by {whole:.3g}. The largest "
            f"group of its operations that carries them within it is {kept_names}, "
            + ("which carries them where they are"
               if within(moves[kept], 0.0) else
               f"which moves none by more than {moves[kept]:.3g}")
            + ", and every operation outside it is broken by the centres"
            + (f"; {len(ranked)} such groups tie, and the one kept is the first by the tie "
               f"rule `realized_group` states -- {_tie_rule_words(group)} -- since "
               f"{_tie_reason(group, ranked)}" if len(ranked) > 1 else "")
            + ".")
    return {"precision": p, "move": whole, "kept": tuple(sorted(primed(kept))),
            "refuted": refuted, "tied": ranked if len(ranked) > 1 else (), "note": note}


def _position_pass(model, group, tolerances, labels, params, read):
    r"""The centres moved onto the largest group of the round's operations that carries them.

    The centres the cell was handed in with are read against the group the
    cell allows, under the declared precision: the largest group of its
    operations whose exact positions are within ``p`` of them carries them
    (:func:`_centres_carried`), the whole group wherever it does.  They are
    projected onto that group's positions (:func:`_symmetric_sites`), a shift of
    all of them together that is noise is removed so that they are written in
    the model's own frame (:func:`_rational_offset`), and the model is moved
    there -- a k-dependent gauge transformation, the spectrum measured
    unchanged.  The group is carried onto the origin the moved centres satisfy
    (:func:`_shared_origin`), so every operation the verdicts read carries them
    exactly, and every operation outside the group that carries them is the
    centres' verdict, broken by them in the verdict stage, the descent that
    follows keeping that origin.

    ``read`` is :func:`_exact_centres`' record: the cell's sites as handed in,
    and the decorations that move with the centres.

    Returns ``(model, group, assignment, record, verdict, extras)`` -- ``verdict``
    the centres' own (:func:`_centres_carried`) where it refutes an operation,
    ``None`` otherwise, and ``extras`` the decorations moved with the centres --
    or the :class:`..refusal.Refusal` the move made.
    """
    from types import SimpleNamespace


    p = float(tolerances.centre)
    frame = read["frame"]
    source = read["source"]
    extras = tuple(read["extras"])
    syms = group.symmetries
    unitary = [k for k, s in enumerate(syms) if not s.time_reversal]
    rotations = np.array([np.asarray(syms[k].rotation, dtype=float) for k in unitary])
    translations = np.array([np.asarray(syms[k].translation, dtype=float).reshape(-1)
                             for k in unitary])
    ind_of_row = [int(syms[k].ind) for k in unitary]
    action = _site_action(frame, rotations, translations)
    if action is None:
        return Refusal(
            "positions_not_nearly_symmetric",
            f"the operations of {group.name} #{group.number_str} do not carry the orbital "
            "centres onto one another at the symmetry finder's distance: no set of positions "
            "near them is one this group carries, so moving them would substitute a crystal "
            "the model does not have. Identify the group the positions carry "
            "(`typat=`, `decorations=`).",
            value=float("nan"), threshold=p, context={"group": group})
    perm, shift = action
    verdict = _centres_carried(group, frame, perm, shift, rotations, ind_of_row, tolerances)
    beyond = bool(verdict["refuted"])
    kept = set(int(i) for i in verdict["kept"])
    rows = [r for r, i in enumerate(ind_of_row) if i in kept]
    x, trans, closure = _symmetric_sites(frame, perm, shift, rotations, rows)
    if not closure <= _EXACT_CLOSURE:          # pragma: no cover - a group acts as a group
        raise InternalError(
            f"the {len(rows)} operations of {group.name} #{group.number_str} that carry the "
            "orbital centres do not carry the positions projected onto them, missing by "
            f"{closure:.3g}: their action on the sites is a group's by construction, so this "
            "is arithmetic in this package failing, not a measurement about the model.")
    own = frame["own"]
    dim_r = int(source.lattice.dim_r)
    moves = _moves_of(frame, x)
    worst = float(np.abs(moves[:own]).max(initial=0.0))
    if not within(worst, p):
        return Refusal(
            "positions_not_nearly_symmetric",
            f"an orbital centre would have to move {worst:.3g} (reduced) onto the positions "
            f"the largest group of {group.name} #{group.number_str}'s operations puts it at, "
            f"against the declared precision {p:.3g}: the orbitals at one site are further "
            "apart than the declaration allows. That is a different set of centres, "
            "beyond a numerical defect. Identify the group the positions carry "
            "(`typat=`, `decorations=`).",
            value=worst, threshold=p, context={"group": group})
    delta = _rational_offset(trans, rotations, rows, source.periodic_dirs)
    in_frame = delta is not None and within(float(np.abs(delta).max(initial=0.0)), p)
    final = x - delta if in_frame else x
    shifted = _moves_of(frame, final)
    raw = np.asarray(model.orb_vecs, dtype=float)
    orb = np.asarray(source.orb_vecs, dtype=float) + shifted[:own, :dim_r]
    if np.array_equal(orb, raw):
        moved, resid, n_check = model, 0.0, 0
    else:
        (moved, resid), n_check = _with_centres(model, orb, params=params), _bloch.CHECK_K
    moved_extras = _moved_extras(extras, shifted, len(frame["points"]) - len(extras), dim_r)
    # the group, carried onto the origin the moved centres are exact about: once
    # in the frame the centres were read in, which the defects below are
    # measured in, and once as every later step reads the moved model against
    # it, in the group's own lattice and centred on the model's own centres
    # (`_pad_cell`) -- which along a layer's normal need not be that frame
    pairs = tuple((ind_of_row[r], ind_of_row[r]) for r in rows)
    final_trans = {r: final[perm[r, 0]] + shift[r, 0] - rotations[r] @ final[0] for r in rows}
    read_in = _shared_origin(group, [SimpleNamespace(ind=ind_of_row[r],
                                                     translation=final_trans[r])
                                     for r in rows], pairs)
    _lat, at_group, _pad = _pad_cell(moved, group.vacuum_length, lattice=group.lattice)
    move_sites = [int(s) for s in frame["moving"]]
    first = int(frame["site_orbs"][move_sites[0]][0])

    def as_read(r):
        """Operation ``r``'s translation on the moved model, read against the group."""
        image = int(frame["site_orbs"][move_sites[int(perm[r, 0])]][0])
        if image >= own:                          # pragma: no cover - sites lead with orbitals
            return final_trans[r]
        t = at_group[image] - rotations[r] @ at_group[first]
        return t + np.rint(final_trans[r] - t)

    on = _shared_origin(group, [SimpleNamespace(ind=ind_of_row[r], translation=as_read(r))
                                for r in rows], pairs)
    asg = _assign_orbitals(moved, on, typat=labels)
    if isinstance(asg, Refusal):
        return asg
    on_trans = np.array([np.asarray(read_in.symmetries[k].translation, dtype=float).reshape(-1)
                         for k in unitary])
    every = list(range(len(unitary)))
    offset = (0.0 - np.asarray(delta) * frame["scale"])[:dim_r] if in_frame else np.zeros(dim_r)
    record = {
        "defect": _frame_defect(frame, perm, rotations, on_trans, every),
        "position_tol": p, "adjusted": True,
        "defect_before": _frame_defect(frame, perm, rotations, on_trans, rows),
        "defect_after": _frame_defect(frame, perm, rotations, on_trans, rows, sites=final),
        "max_move": worst, "offset": tuple(float(c) for c in offset),
        "frame": "model" if in_frame else "fitted",
        "spectrum_residual": float(resid), "n_check": int(n_check),
        "n_moved": int(np.count_nonzero(np.abs(orb - np.asarray(source.orb_vecs, dtype=float))
                                        .max(axis=1) > 0.0))}
    if beyond:
        said = _moved_note(worst, offset, in_frame, delta, p, resid, where="of that group")
        record["move"] = float(verdict["move"])
        record["carried"] = len(verdict["kept"])
        record["note"] = verdict["note"] + " " + said[:1].upper() + said[1:]
    else:
        record["note"] = _moved_note(worst, offset, in_frame, delta, p, resid)
    return moved, on, asg, record, (verdict if beyond else None), moved_extras


def _cell_carried(env_cell, env_typat, old_cell, old_typat, n_env, delta):
    r"""The cell built again on the moved centres, an orbit added to it since carried along.

    ``env_cell`` is the environment's cell on the moved model; ``old_cell`` the
    cell the call had reached, whose first ``n_env`` points are the
    environment's and whose others, where it has any, are the orbit the
    declared content added (:func:`_declared_cell`).  That orbit was generated
    by exact operations about the origin the group was named in, so carried by
    the shift ``delta`` that writes the group in the moved centres' origin it is
    exact there too.
    """
    from pythtb import Lattice, TBModel

    if int(old_cell.norb) <= int(n_env):
        return env_cell, list(env_typat)
    dim_r = int(env_cell.lattice.dim_r)
    added = (np.asarray(old_cell.orb_vecs, dtype=float)[int(n_env):]
             + np.asarray(delta, dtype=float).reshape(-1)[:dim_r])
    pos = np.vstack([np.asarray(env_cell.orb_vecs, dtype=float), added])
    out = TBModel(Lattice(np.asarray(env_cell.lat_vecs, dtype=float), pos.tolist(),
                          periodic_dirs=list(env_cell.periodic_dirs)), spinful=False)
    setattr(out, _OWN_ORBITALS, int(getattr(env_cell, _OWN_ORBITALS, env_cell.norb)))
    return out, list(env_typat) + list(old_typat)[int(n_env):]


def _quote_position_defect(refusal, positions, symmetrization) -> Refusal:
    """Put a measured-and-unremoved position defect into the refusal that followed.

    A position defect is not something a *hopping* projection can reach below —
    in convention I it costs ``H(k)`` a covariance residual of order
    ``2π·δ·max|H|`` on its own — so when the pipeline refuses downstream of one,
    that number is the thing to act on and it belongs in the message rather than
    only in a record the refusal replaces.  Under a declared precision the
    centres are always moved within it, so the one way to have one is no
    declaration at all, with the hopping pass measuring one anyway
    (:attr:`..symmetrization.Symmetrization.operators`\\ ``["position_defect"]``),
    in which case the route is to declare one.

    Anything else — no measurement, or a defect that *was* removed — is returned
    unchanged: quoting a number that is no longer there would misdirect.
    """

    if not isinstance(refusal, Refusal):
        return refusal
    if (positions is None and symmetrization is not None
            and float(symmetrization.operators.get("position_defect", 0.0))
            > EXACT_GATE):
        # `EXACT_GATE` is the exact pipeline's own rounding scale rather than a
        # gate on this number (which is reduced-coordinate, not in units of H):
        # it is what separates "the centres are exact" from "the centres are a
        # measurement of their own", and the message is the only consequence.
        defect = float(symmetrization.operators["position_defect"])
        note = (
            f"No precision was declared for the centres, and the hopping projection "
            f"measured this model's centres to miss the group's own action by "
            f"{defect:.3g} (reduced) -- which in convention I costs H(k) a covariance "
            "residual of order 2*pi*delta*max|H| whatever the hoppings do, so a "
            "projection of the hoppings cannot remove it and this refusal may be "
            "entirely that defect. Declare it: `realized_group(..., precision=)` "
            "measures the centres first and moves them, within the declaration, onto "
            "the positions the group puts them at -- a k-dependent gauge "
            "transformation, so the spectrum is untouched.")
    else:
        return refusal
    return replace(refusal, reason=f"{refusal.reason} {note}",
                   context=dict(refusal.context, position_defect=defect,
                                positions=positions))


def _tensor_structure_entry(record) -> dict:
    """What a refusal says of an operation its tensor structure refuted.

    The ``note``, the condition of its own form that failed (``own_form``),
    where that was read, and the dimensionless ``residual`` the verdict rests
    on: how far the matrix of that form nearest the operation's solution space
    misses it, over that matrix's own size.
    """
    return {"note": record.get("note"),
            **({"own_form": record["own_form"]} if record.get("own_form") else {}),
            "residual": float(record.get("scaled", float("nan")))}


def _cell_group_not_realized(group, validated, three_d, refuted=None) -> Refusal:
    """The refusal for a Hamiltonian that breaks part of the group its cell allows.

    Separated out so the message — the only thing :func:`realized_group` returns
    when the audit fires — can be read whole.

    ``refuted`` is the verdict of :func:`_own_form_read` and
    :func:`.._descent._ansatz_refuted`: operations the **detector** kept (their
    k-action class survives) whose **tensor structure** refutes them — the
    ``σ_h`` family on a slab, an operation whose k-action the hoppings carry
    only through a spin factor other than its own lift, or one they carry only
    with the sites permuted as another operation permutes them.  They are counted with
    the detector's own reasons under ``"tensor structure"``, because they are
    broken in exactly the same sense: the cell allows them and the Hamiltonian
    does not obey them.
    Those refuted by the states' **projective class**
    (:func:`.._descent._projective_class_refuted`) are counted under their own name: no
    hopping breaks them, and no single-valued representation carries them.
    So are those the orbital **centres** are not carried by within a declared
    precision (:func:`_centres_carried`), whose verdict stands for them whatever
    the detector read on them.
    """
    refuted = dict(refuted or {})
    classed = {i for i, r in refuted.items()
               if isinstance(r, dict) and r.get("refuted_by") == "projective class"}
    centred = {i for i, r in refuted.items()
               if isinstance(r, dict) and r.get("refuted_by") == "centres"}
    structural = {i: r for i, r in refuted.items() if i not in classed | centred}
    lifted = sum(1 for r in structural.values()
                 if isinstance(r, dict) and bool(r.get("spin_lift")))
    reasons = {}
    for d in validated.discrepancies:
        if int(d["ind"]) not in centred:
            reasons[d["refuted_by"]] = reasons.get(d["refuted_by"], 0) + 1
    if structural:
        reasons["tensor structure"] = len(structural)
    if classed:
        reasons["projective class"] = len(classed)
    if centred:
        reasons["centres"] = len(centred)
    n_broken = len(set(validated.broken) | set(refuted))
    n_obeyed = len(set(validated.indices) - set(refuted))
    proven = reasons.get("no symmetry matrix", 0)
    counts = [f"{n} {_DROPPED_BY_WORDS['centres']}" if why == "centres" else f"{n} by {why}"
              for why, n in reasons.items()]
    by = (", ".join(counts[:-1]) + " and " + counts[-1]) if len(counts) > 1 else \
        "".join(counts)
    # a cause that broke every operation broken is said without a count, which
    # would only repeat the total
    if len(reasons) == 1 and sum(reasons.values()) == n_broken:
        why = next(iter(reasons))
        by = ("every one " + (_DROPPED_BY_WORDS["centres"] if why == "centres"
                              else f"by {why}"))

    def the(n, said):
        """``said`` of the ``n`` broken one way: all of them, or that many."""
        return f" Those {said}" if n == n_broken else f" The {n} {said}"
    lattice = ""
    if int(getattr(group, "_lattice_index", 1) or 1) > 1:
        taus = tuple(getattr(group, "_fractional_translations", ()) or ())
        tau = np.round(np.asarray(taus[0], dtype=float), 6).tolist() if taus else []
        lattice = (
            f" Among the broken operations is the pure translation {tau} (reduced) and "
            f"its coset partners: this cell's point set repeats under it, so what the "
            f"cell allows is a lattice supergroup of index "
            f"{int(group._lattice_index)}, and the Hamiltonian breaking the translation "
            "is what makes the model's own lattice its translation group. With "
            "descend=True the first step of the descent is that lattice descent.")
    route = ("`typat=` where two sites are chemically inequivalent, or "
             "`decorations=` where a distortion lowered the symmetry: a bond "
             "distortion moves no atom, so no symmetry finder can see it in the "
             "point set, and one non-orbital species where the distortion is "
             "centred is how the cell is told about it"
             if three_d else
             "`typat=` for sublattices that are chemically inequivalent, or "
             "`decorations=` for a crystal field the flat point set does not "
             "carry")
    return Refusal(
        "cell_group_not_realized",
        f"the Hamiltonian obeys {n_obeyed} of the {group.size} operations the cell "
        f"allows ({group.number_str} {group.name}), {_PER_COSET}"
        + (f" and {_DOUBLE_GROUP}" if group.spinor else "")
        + f"; {n_broken} are broken, {by}."
        + (the(proven, "with no symmetry matrix are broken provably: no matrix with "
                       "their k-action carries H(k) onto itself at all.") if proven else "")
        + (the(len(structural), "broken by tensor structure have matrices with their "
                                "k-action that carry H(k) onto itself, and none of those "
                                "matrices has the form O(g) (x) S(g) that a "
                                "representation of the operation needs.")
           + ((" For every one of them" if lifted == len(structural) else
               f" For {lifted} of them")
              + " the hoppings carry the operation's action on k only through a spin "
              "factor other than its own lift S(g)." if lifted else "")
           if structural else "")
        + (the(len(classed), "broken by the projective class of the states are "
                             "obeyed by the hoppings, and their matrices on these states "
                             "carry a sign no single-valued representation has -- two "
                             "commuting operations whose matrices anticommute, the "
                             "spin-1/2 sign of spinors stored in the orbital index -- so "
                             "they are not operations of the spinless crystal; `spin=` "
                             "reads the same states in the double group.")
           if classed else "")
        + (the(len(centred), "broken by the centres do not carry the orbital centres "
                             "onto one another within the declared precision: moving the "
                             "centres onto the positions they put them at would move one "
                             "further than the precision allows.")
           if centred else "")
        + lattice
        + " "
        f"{group.number_str} is what the positions allow, an unmeasured upper "
        "bound, and its corep labels describe a Hamiltonian that obeys all of it; "
        "returning a representation of it here would label these bands with a "
        "group the model does not have. Identify again from a cell that asks the "
        f"question the Hamiltonian answers — {route} — and pass the result back "
        "through this call, or pass descend=True (the default) and have the cell "
        "lowered to the operations that were measured to hold. "
        "`context['validated']` is the verdict and residual on every operation.",
        value=float(n_broken), threshold=0.0,
        context={"group": group, "validated": validated,
                 "number_str": group.number_str, "size": int(group.size),
                 "n_obeyed": int(n_obeyed),
                 "broken": tuple(validated.broken) + tuple(
                     i for i in sorted(refuted) if i not in set(validated.broken)),
                 "tensor_structure_refuted": tuple(sorted(structural)),
                 "tensor_structure": {int(i): _tensor_structure_entry(r)
                                      for i, r in structural.items()},
                 "refuted_by": reasons,
                 **({"projective_class": {i: refuted[i] for i in sorted(classed)}}
                    if classed else {})})
