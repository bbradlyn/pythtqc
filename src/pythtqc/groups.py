r"""The space group a Hamiltonian obeys, and that group's tables in the model's own cell.

:func:`realized_group` measures the group: the space group of the lattice of
occupied sites, lowered to the operations ``H(k)`` obeys, with each orbital's
Wyckoff position and the representation that labels the bands.  The other
calls read the group it returns:

* :func:`operations`, :func:`seitz` and :func:`operation_index`: each
  operation by its index, the key every per-operation record is written on,
  with its Seitz symbol in the tables' setting and its name in the model's
  Cartesian frame; and the index a Seitz symbol names;
* :func:`high_symmetry_points`, :func:`little_group` and :func:`k_grid`: the
  tabulated k-points, the operations that leave a k invariant, and a mesh with
  the high-symmetry points it holds;
* :func:`corep_table` and :func:`identify_from_characters`: the character table
  of the little group at a high-symmetry point, and the coreps measured
  characters carry;
* :func:`wyckoff_sites_tables` and :func:`wyckoff_positions_spglib`: each
  site's Wyckoff position in the tables' setting, which every table is read
  with, and the group's Wyckoff positions in spglib's;
* :func:`ebr_data`, :func:`indicator_group` and :func:`decompose_ebrs`: the
  group's elementary band representations, its indicator group, and a symmetry
  vector written in its elementary band representations;
* :func:`stack_presentation`: a layer repeated along its normal with no hopping
  between the copies, the three-dimensional crystal its rows at ``k_z = ½`` are
  measured on;
* :func:`tables_status`: whether the tables are installed.

Each record returned prints what it holds with ``summary()``, and a call that
cannot complete returns a falsy ``Refusal`` in its place.

The tables are irrep's, the ``irreptables`` package, for the 230 space groups
and their grey groups, time reversal adjoined; a Hamiltonian that breaks time
reversal is returned the unitary group, with a
:class:`..refusal.MagneticScopeWarning`.  Every k-point taken or returned is in
the model's own reduced coordinates, generally not the tables', so the points
are taken from :func:`high_symmetry_points`.  The characters and labels are in
the physical convention for a lattice translation, ``e^{-ik·t}``, where the
tables' own pages write ``e^{+ik·t}``: away from the time-reversal-invariant
momenta, a label read at ``k`` here is the tables' label at ``-k``.  A model
periodic in two directions is a layer, read as a slab, a three-dimensional cell
with vacuum along the layer's normal, whose group is a space group.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import itertools
import math
import os
import re
import string
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np

# `irreptables.irreps` imports `irrep.utility`, so a process that imports it
# first makes irrep's own `from irreptables.irreps import IrrepTable` run
# against a half-built module and bind the `DummyMethodTables` fallback instead
# of the real tables; importing irrep here binds the real one first.
import irrep.spacegroup_irreps  # noqa: F401

from . import _bloch
from ._tolerance import (ASSEMBLED, CHARACTER, DEFAULT_ENERGY, DEFAULT_FINDER, EXACT,
                         HEADROOM, IN_SPACE, IN_SPAN, ROUNDING, SINGULAR, SNAP,
                         WYCKOFF_REFERENCE_TOL, ZERO, _CARRY_TOL, _CHAR_TOL, _CLASS_MARGIN,
                         _EIGEN_WINDOW_CEIL, _EXACT_CLOSURE, _IN_BLOCKS, _LATTICE_TOL,
                         _MOVED_PRESENTATION_RELATIVE, _PHASE_REAL_TOL, _SU2_TOL,
                         _TIME_REVERSAL_SQUARE_MARGIN, Tolerances, eigen_window,
                         finder_distance, noise_floor, within)
from .refusal import InternalError, MagneticScopeWarning, Refusal
from .symmetry import (SpaceGroupRep, _content_by_position, _GridHamiltonians,
                       _find_intertwiners, _index_ranges)

__all__ = [
    "GroupData", "LittleGroup", "WyckoffPosition", "OrbitalAssignment",
    "IrrepContent", "KGrid", "ValidatedGroup", "HighSymmetryPoints",
    "Operation", "OperationTable",
    "tables_status", "wyckoff_positions_spglib",
    "operations", "seitz", "operation_index",
    "little_group",
    "high_symmetry_points", "corep_table", "identify_from_characters", "ebr_data",
    "indicator_group", "decompose_ebrs", "k_grid",
    "RealizedGroup", "realized_group", "stack_presentation",
    "SLAB_C", "STACK_C",
    "DECORATION_DZ",
    "SiteReference", "wyckoff_sites_tables",
    "WYCKOFF_REFERENCE_TOL",
]

# ITA Wyckoff letters: 'a'..'z' then 'A' (space group 47 has 27 positions).
_WYCKOFF_LETTERS = string.ascii_lowercase + string.ascii_uppercase

#: Where this package's own sources live.  A warning about the physics belongs
#: at the line that asked the question, and the measurement that raises it sits
#: several calls down from there, so every frame under this prefix is skipped
#: and the warning is reported against the call that a reader wrote.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# whether the character tables are in place
# --------------------------------------------------------------------------

def _set_irreptables_version(info: dict[str, Any]) -> None:
    """The installed ``irreptables`` version, written onto a status record."""
    try:
        import irreptables
        info["irreptables_version"] = getattr(irreptables, "__version__", "unknown")
    except ImportError:  # pragma: no cover - a spec that will not import
        pass


def tables_status() -> dict[str, Any]:
    """Whether irrep's character tables are in place, and ``ortools`` installed.

    Without the ``irreptables`` package, or with it imported before ``irrep``
    in the same process, ``irrep`` binds a stand-in that raises nothing and
    names no label, and every call that reads the tables refuses
    ``irrep_tables``.

    Returns
    -------
    dict
        ``tables``: whether the real tables are in place, and ``reason``, where
        they are not, which of the two causes it is and what to do;
        ``ortools``: whether the integer solver a decomposition needs is
        installed; ``irrep_version`` and ``irreptables_version``: the versions
        installed, ``None`` where a package is absent.
    """
    info: dict[str, Any] = {"tables": False, "ortools": False,
                            "irrep_version": None, "irreptables_version": None,
                            "reason": None}
    try:
        import irrep
        from irrep import spacegroup_irreps as _sgi
        from irrep.utility import DummyMethodTables
    except ImportError as exc:  # pragma: no cover - environment without irrep
        info["reason"] = f"irrep is not importable: {exc}"
        return info
    info["irrep_version"] = getattr(irrep, "__version__", "unknown")
    bound = _sgi.IrrepTable
    if isinstance(bound, DummyMethodTables):
        import importlib.util
        if importlib.util.find_spec("irreptables") is None:
            info["reason"] = ("irrep substituted its DummyMethodTables fallback for IrrepTable: "
                              "the `irreptables` package is not installed, so every irrep "
                              "identification would silently return 'None' labels. "
                              "Install it: pip install 'irreptables>=3.1'")
            return info
        _set_irreptables_version(info)
        info["reason"] = (
            "irrep substituted its DummyMethodTables fallback for IrrepTable: `irreptables` "
            "is installed, but `irreptables.irreps` was imported before `irrep` in this "
            "process. `irreptables.irreps` imports `irrep.utility`, so irrep's own `from "
            "irreptables.irreps import IrrepTable` ran against a half-built module, failed "
            "on that circular import and bound the fallback for the rest of the process, "
            "and every irrep identification would silently return 'None' labels. Import "
            "`pythtqc` (or `irrep`) before `irreptables`.")
        return info
    _set_irreptables_version(info)
    info["tables"] = True
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        info["ortools"] = True
    except ImportError:
        info["ortools"] = False
    return info


def _refuse_no_tables(status: dict[str, Any]) -> Refusal:
    return Refusal(
        gate="irrep_tables",
        reason=status["reason"] or "irrep did not load real character tables",
        context={k: v for k, v in status.items() if k != "reason"},
    )


# --------------------------------------------------------------------------
# what the records' summaries print
# --------------------------------------------------------------------------

#: The unit a count of operations is in, named wherever the count is what is
#: read: the operations listed are one per coset of the lattice translations.
_PER_COSET = "coset representatives of the lattice translations"

#: What a spinful group's count adds, once in a printout: the operations listed
#: are the single-valued group's, and the double group holds each one's negative.
_DOUBLE_GROUP = ("the single-valued group's, each standing for itself and its negative "
                 "in the double group")

def _coordinate_text(x) -> str:
    """One reduced coordinate as the fraction it is, or as a decimal where it is none."""
    x = float(x)
    value = Fraction(x).limit_denominator(_SEITZ_DENOMINATOR)
    if abs(float(value) - x) > _LATTICE_TOL:
        return f"{x:.4g}"
    return (f"{value.numerator}" if value.denominator == 1
            else f"{value.numerator}/{value.denominator}")


def _point_text(v) -> str:
    """A point in reduced coordinates, ``(1/3, 2/3)``."""
    return "(" + ", ".join(_coordinate_text(x)
                           for x in np.asarray(v, dtype=float).reshape(-1)) + ")"


def _affine_text(jacobian, offset) -> str:
    """A Wyckoff position's map ``A·(x, y, z) + b`` as its three coordinates, ``(x, -x, 1/2)``."""
    out = []
    for row, b in zip(np.asarray(jacobian, dtype=float), np.asarray(offset, dtype=float)):
        terms = []
        for coefficient, symbol in zip(row, "xyz"):
            c = Fraction(float(coefficient)).limit_denominator(_SEITZ_DENOMINATOR)
            if c == 0:
                continue
            size = abs(c)
            body = (symbol if size == 1 else f"{size.numerator}{symbol}"
                    if size.denominator == 1 else
                    f"{'' if size.numerator == 1 else size.numerator}{symbol}/{size.denominator}")
            terms.append(("-" if c < 0 else "+", body))
        constant = Fraction(float(b)).limit_denominator(_SEITZ_DENOMINATOR)
        if constant != 0 or not terms:
            terms.append(("-" if constant < 0 else "+", _coordinate_text(abs(constant))))
        text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sign, body in terms[1:]:
            text += f" {sign} {body}"
        out.append(text)
    return "(" + ", ".join(out) + ")"


def _lattice_vectors(dirs) -> str:
    """Lattice-vector indices in words: ``the first two``, ``the third``, ``the first and third``."""
    words = ("first", "second", "third")
    dirs = sorted(int(d) for d in dirs)
    if dirs == [0, 1]:
        return "the first two"
    if len(dirs) == 3:
        return "all three"
    return "the " + " and ".join(words[d] for d in dirs)


def _names_text(names, shown=6) -> str:
    """Names in a sentence, the first ``shown`` of them and how many more."""
    names = [str(n) for n in names]
    return (", ".join(names[:shown])
            + (f" and {len(names) - shown} more" if len(names) > shown else ""))


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupData:
    """A space group, written in the model's own cell.

    The group ``realized_group(model).group`` holds, and the one every table
    call takes: :func:`operations`, :func:`high_symmetry_points`,
    :func:`corep_table`, :func:`ebr_data`.  ``summary()`` prints it.

    Attributes
    ----------
    name : str
        The Hermann–Mauguin symbol, ``"P6/mmm1'"``, a trailing ``1'`` marking
        time reversal adjoined.
    number_str : str
        The label the tables file the group under, ``"191.234"``.
    sg_number : int
        The space-group number, ``191``, with time reversal adjoined or not.
    spinor, include_TR, magnetic : bool
        Whether it is the double group, of states with spin one-half; whether
        time reversal is adjoined; whether it is read in the magnetic tables,
        as a grey group is.
    symmetries : list
        Its operations, one per coset of the lattice translations, in the
        tables' order: ``irrep``'s record of each, ``.ind`` its index.
        :func:`operations` writes them out.
    u_symmetries, au_symmetries : list
        The unitary operations, and those combined with time reversal.
    size : int
        How many operations :attr:`symmetries` lists.
    lattice : ndarray, shape (3, 3)
        The three-dimensional cell the group was found on, a row per lattice
        vector, a layer's with its vacuum.
    periodic_dirs, vacuum_dirs : tuple of int
        The axes of :attr:`lattice` that are periodic, and a layer's vacuum.
    vacuum_length : float
        The vacuum along a layer's normal, in the model's length units; the
        group does not depend on it.
    dim_k, dim_r : int
        The model's periodic and spatial dimensions.
    site_positions, site_orbitals, typat : ndarray, tuple, ndarray
        The points the group was found on, in reduced coordinates of
        :attr:`lattice`, the orbitals on each and each point's species: the
        orbitals' sites, and any species added to the cell.
    symprec : float
        The symmetry finder's tolerance, a Cartesian length in the model's
        units.
    provenance : {"realized", "descended"}
        Measured on the Hamiltonian, or measured after the cell was lowered,
        which :attr:`RealizedGroup.descent` records.
    """

    sg_number: int
    number_str: str
    name: str
    spinor: bool
    magnetic: bool
    include_TR: bool
    symmetries: list
    lattice: np.ndarray
    site_positions: np.ndarray
    site_orbitals: tuple
    typat: np.ndarray
    periodic_dirs: tuple
    vacuum_dirs: tuple
    vacuum_length: float
    # the transformation from the model's cell to the tables' conventional one,
    # as `irrep` determined it, and whether it was found: every table lookup
    # reads it, and one made without it refuses `tables_not_ready`, saying why
    # (`_setup_note`)
    _refUC: np.ndarray
    _shiftUC: np.ndarray
    _tables_ready: bool
    symprec: float
    dim_k: int
    dim_r: int
    _setup_note: str | None = None
    provenance: str = "cell"
    # how far the operations as spglib found them missed closing into a group,
    # reduced modulo the lattice -- 0.0 for a cell built from exact numbers,
    # rounding for one whose centres a declared precision made exact -- and the
    # largest translation the repair from the tabulated operations moved
    _closure_measured: float = 0.0
    _closure_repaired: float = 0.0
    # how many cosets of pure translations the cell's operations carry, and the
    # non-zero ones: 1 and empty on every group a call returns, since a cell that
    # repeats under a fractional translation is refused `supercell` or descended
    _lattice_index: int = 1
    _fractional_translations: tuple = ()
    # how the stored SU(2) lifts were put in the tables' own convention: the
    # frame of the tables' matrices, the sign each lift was corrected by, and the
    # defects measured; None for a spinless group
    _spin_frame: dict | None = field(default=None, repr=False, compare=False)
    _sg: Any = field(default=None, repr=False, compare=False)
    #: Memo of :meth:`get_product_table`, keyed by ``get_diff``.  ``init=False``
    #: so ``dataclasses.replace`` starts a fresh one rather than inheriting a
    #: table computed for some other ``_sg``.
    _product_tables: dict = field(default_factory=dict, init=False, repr=False,
                                  compare=False)
    #: Memo of :func:`_wyckoff_setting_audit` for this group — the check that
    #: ``pyxtal``'s Wyckoff list is written in the same setting as the tables.
    #: It is a property of the group and not of a site, so it runs once.
    #: ``init=False`` for the same reason as above.
    _wyckoff_audit: dict = field(default_factory=dict, init=False, repr=False,
                                 compare=False)
    #: Memo of :func:`_subgroups_inside` for this group, keyed by the operations
    #: kept and the pairs avoided: the subgroups inside a set are a property of
    #: the group, asked of it by every reading that searches them.  ``init=False``
    #: for the same reason as above.
    _subgroups: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # -- delegated to ``irrep`` --------------------------------------------
    @property
    def u_symmetries(self) -> list:
        """The unitary operations of :attr:`symmetries`."""
        return self._sg.u_symmetries

    @property
    def au_symmetries(self) -> list:
        """The operations of :attr:`symmetries` combined with time reversal, none without it."""
        return self._sg.au_symmetries

    @property
    def size(self) -> int:
        """How many operations :attr:`symmetries` lists, one per coset of the lattice translations."""
        return len(self.symmetries)

    def get_product_table(self, get_diff: bool = True):
        """The group's multiplication table, with the lattice translation and spin sign of each product.

        These are what a representation of the group reproduces.

        Parameters
        ----------
        get_diff : bool, default True
            Whether to return the translations and the signs beside the table.

        Returns
        -------
        tuple of ndarray
            ``(table, translations, signs)``: ``table[i, j]`` the position in
            :attr:`symmetries` of the product of the operations at positions
            ``i`` and ``j``; ``translations[i, j]`` the lattice vector by which
            that product's translation exceeds the listed operation's; and
            ``signs[i, j]`` the ``±1`` by which its spin matrix differs from
            the listed operation's, ``1`` throughout for a spinless group.
        ndarray
            The table alone, where ``get_diff`` is False.
        """
        got = self._product_tables.get(True)
        if got is None:
            got = self._sg.get_product_table(get_diff=True)
            self._product_tables[True] = got
        if get_diff:
            return tuple(np.array(a) for a in got)
        return np.array(got[0])

    def get_inverse_table(self):
        """The position in :attr:`symmetries` of each operation's inverse.

        Returns
        -------
        ndarray of int
            Entry ``i`` is the position of the inverse of the operation at
            position ``i``: positions in the list, one less than the indices.
        """
        return self._sg.get_inverse_table()

    # -- k between the model and the 3D cell -------------------------------
    def k_to_3d(self, k) -> np.ndarray:
        """The model's reduced k, as the same points in the three-dimensional cell.

        A layer's cell has its vacuum along the layer's normal, so the
        periodic components carry over and the one along the vacuum is zero.

        Parameters
        ----------
        k : array_like, shape (dim_k,) or (n, dim_k)
            Points in the model's reduced coordinates.

        Returns
        -------
        ndarray, shape (n, 3)
            The same points in reduced coordinates of :attr:`lattice`.
        """
        k = np.atleast_2d(np.asarray(k, dtype=float))
        out = np.zeros((k.shape[0], 3))
        out[:, list(self.periodic_dirs)] = k
        return out

    def k_from_3d(self, k3) -> np.ndarray:
        """Points of the three-dimensional cell in the model's reduced coordinates, the vacuum's component dropped."""
        k3 = np.atleast_2d(np.asarray(k3, dtype=float))
        return k3[:, list(self.periodic_dirs)]

    def in_plane(self, k3, atol: float = 1e-9) -> bool:
        """Whether points of the three-dimensional cell lie in a layer's plane, within ``atol``."""
        k3 = np.atleast_2d(np.asarray(k3, dtype=float))
        if not self.vacuum_dirs:
            return True
        return bool(np.all(np.abs(k3[:, list(self.vacuum_dirs)]) <= atol))

    def k_action_of(self, op):
        """One operation's action on the model's reduced k, ``k -> R k``.

        The image is left unreduced: ``H(k + G)`` is not ``H(k)`` in pythtb's
        convention, and an operation carries ``H(k)`` to ``H`` at the image
        itself.

        Parameters
        ----------
        op : irrep.symmetry_operation.SymmetryOperation
            One operation of :attr:`symmetries`.

        Returns
        -------
        callable or None
            ``k -> R k``, for one point or an array of them; ``None`` for an
            operation that turns a layer's plane out of itself.
        """
        probe = np.eye(self.dim_k)
        rot = op.transform_k(self.k_to_3d(probe))
        if not self.in_plane(rot):
            return None
        M = self.k_from_3d(rot)  # rows: images of the unit k-vectors

        def action(k, _M=M):
            return np.atleast_2d(np.asarray(k, dtype=float)) @ _M
        return action

    def summary(self) -> str:
        """The group, a line for each part, with the attribute that holds it."""
        lines = [f"{self.name} #{self.number_str}   (.name, .number_str)",
                 "  its operations, one per coset of the lattice translations   (.symmetries)"]
        if self.spinor:
            lines.append("  the double group also holds each one's negative   (.spinor)")
        if self.vacuum_dirs:
            lines.append(f"  a layer: periodic along {_lattice_vectors(self.periodic_dirs)} "
                         f"lattice vectors, vacuum along {_lattice_vectors(self.vacuum_dirs)}"
                         "   (.periodic_dirs, .vacuum_dirs)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        if int(self._lattice_index) > 1:
            tag = (f" [lattice supergroup, index {int(self._lattice_index)}; "
                   "tables on the finer lattice]")
        else:
            tag = "" if self._tables_ready else " [tables not ready]"
        return (f"GroupData({self.name} #{self.number_str}, "
                f"spinor={self.spinor}, TR={self.include_TR}, "
                f"periodic={self.periodic_dirs}, {self.provenance}{tag})")


@dataclass(frozen=True)
class LittleGroup:
    """The operations that leave one k invariant up to a reciprocal lattice vector.

    What :func:`little_group` returns; ``summary()`` prints it, and ``len`` is
    the number of its unitary operations.

    Attributes
    ----------
    k : ndarray, shape (dim_k,)
        The point, in the model's reduced coordinates.
    k3 : ndarray, shape (3,)
        The same point in the three-dimensional cell.
    unitary : list
        Its unitary operations, ``irrep``'s records, in index order: the
        columns a character vector at ``k`` is written in.
    antiunitary : list
        Its operations combined with time reversal, those whose rotation
        carries ``k`` to ``−k``, in index order; a corep table has no
        characters for them.
    indices : tuple of int
        The indices of :attr:`unitary`.
    G : ndarray, shape (n_unitary, 3)
        For each unitary operation, the reciprocal lattice vector ``R⁻ᵀk − k``
        it moves ``k`` by, in the three-dimensional cell.
    """

    k: np.ndarray
    k3: np.ndarray
    unitary: list
    antiunitary: list
    indices: tuple
    G: np.ndarray

    def __len__(self) -> int:
        return len(self.unitary)

    def summary(self) -> str:
        """The little group at one k, a line for each part, with the attribute that holds it."""
        lines = [f"k = {_point_text(self.k)}, in the model's reduced coordinates   (.k)",
                 f"  its unitary operations, by index: {_index_ranges(self.indices)}   "
                 "(.unitary, .indices)",
                 "  the reciprocal lattice vector each of them moves k by   (.G)"]
        if self.antiunitary:
            lines.append("  its operations combined with time reversal, by index: "
                         + _index_ranges(int(s.ind) for s in self.antiunitary)
                         + "   (.antiunitary)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        k = ", ".join(f"{float(x):.4g}" for x in np.asarray(self.k, dtype=float).reshape(-1))
        primed = _index_ranges(int(s.ind) for s in self.antiunitary)
        return (f"LittleGroup(k = ({k}): unitary operations {_index_ranges(self.indices)}"
                + (f"; with time reversal {primed}" if primed else "") + ")")


@dataclass(frozen=True)
class WyckoffPosition:
    """One Wyckoff position of the group, in the model's own cell, lettered in spglib's setting.

    One entry of what :func:`wyckoff_positions_spglib` returns; ``summary()``
    prints it.  Its letter is spglib's, and the tables' letter for the position
    may differ: :func:`wyckoff_sites_tables` gives each site's letter in the
    tables' setting.

    Attributes
    ----------
    letter : str
        Its Wyckoff letter in spglib's setting.
    expression : str
        The position as ``irrep`` writes it, its three coordinates in ``x``,
        ``y`` and ``z``, in the model's reduced coordinates.
    jacobian, offset : ndarray, shapes (3, 3) and (3,)
        ``A`` and ``b`` of the map ``A·(x, y, z) + b`` whose image is the
        position.
    n_free : int
        Its free parameters, the rank of ``A``: 0 for a point, 1 a line, 2 a
        plane, 3 the general position.
    """

    letter: str
    expression: str
    offset: np.ndarray
    jacobian: np.ndarray
    n_free: int

    def distance(self, position, metric: np.ndarray) -> float:
        """The Cartesian distance from a point to this position, the nearest lattice translate of it.

        Parameters
        ----------
        position : array_like, shape (3,)
            The point, in reduced coordinates.
        metric : ndarray, shape (3, 3)
            The lattice, a row per lattice vector, which turns reduced
            differences into Cartesian ones.

        Returns
        -------
        float
            The distance, in the lattice's length units.
        """
        best = np.inf
        A, b = self.jacobian, self.offset
        for shift in _NEIGHBOUR_SHIFTS:
            rhs = np.asarray(position, dtype=float) + shift - b
            if self.n_free == 0:
                resid = rhs
            else:
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                resid = rhs - A @ sol
            best = min(best, float(np.linalg.norm(resid @ metric)))
        return best

    def summary(self) -> str:
        """The position, a line for each part, with the attribute that holds it."""
        freedom = {0: "a point", 1: "a line, one free parameter",
                   2: "a plane, two free parameters"}.get(
            int(self.n_free), "the general position, three free parameters")
        return "\n".join([
            f"{_affine_text(self.jacobian, self.offset)}, in the model's cell   "
            "(.expression, .jacobian, .offset)",
            f"  spglib's letter {self.letter}, in spglib's setting; the tables' letter for "
            "the position may differ   (.letter)",
            f"  {freedom}   (.n_free)"])


_NEIGHBOUR_SHIFTS = np.array([[i, j, k] for i in (-1, 0, 1)
                              for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


@dataclass(frozen=True)
class SiteReference:
    """One site's Wyckoff position in the tables' setting, the one every table is read with.

    One entry of what :func:`wyckoff_sites_tables` returns, which prints its
    position, its free parameters and its closure in a line.

    Attributes
    ----------
    index : int
        The site's place in the list handed in, by default
        :attr:`GroupData.site_positions`.
    position : ndarray, shape (3,)
        The site as handed in, in reduced coordinates of the model's
        three-dimensional cell.
    reference : ndarray, shape (3,)
        The same point in the tables' conventional cell, reduced into
        ``[0, 1)``.
    letter, multiplicity : str and int
        The most special Wyckoff position whose variety holds the point, the
        fewest free parameters and then the smallest multiplicity, and its
        multiplicity in the conventional cell: ``8`` for ``8a``.
    n_free : int
        Its free parameters: 0 a point, 1 a line, 2 a plane, 3 the general
        position.  A polar group's positions are lines at best, ``P4mm``'s
        ``1a`` being ``(0, 0, z)``.
    varieties : tuple of str
        The letter of every position whose variety holds the point, most
        special first.
    closure : tuple of str
        The letters on the closure of the site's variety at which the tables
        induce elementary band representations, where an orbital confined to
        the variety can be moved: ``('a', 'b')`` for the line ``6c``,
        ``(0, 0, z)``, of ``R-3m``.
    residual : float
        How far the point sits from its variety, in reduced coordinates of the
        conventional cell.
    """

    index: int
    position: np.ndarray
    reference: np.ndarray
    letter: str
    multiplicity: int
    n_free: int
    varieties: tuple
    closure: tuple
    residual: float

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SiteReference({self.multiplicity}{self.letter}, "
                f"n_free={self.n_free}, closure {self.closure})")


@dataclass(frozen=True)
class OrbitalAssignment:
    """Where the orbitals sit: each one's site, the sites' Wyckoff orbits, and how the operations move the sites.

    The record ``realized_group(model).assignment`` holds; ``summary()``
    prints it by Wyckoff position.

    Attributes
    ----------
    site_of_orbital, orbit_of_orbital : ndarray, shape (norb,)
        The site each orbital sits on, and that site's orbit.
    site_positions, site_orbitals : ndarray and tuple
        The sites, in reduced coordinates of the group's cell, and the
        orbitals on each.
    orbits : tuple of tuple of int
        The sites, grouped into the orbits the operations carry them through.
    orbit_of_site : ndarray, shape (nsite,)
        The orbit each site is on.
    wyckoff_tables : tuple of str
        Each site's Wyckoff letter in the tables' setting, the one every table
        is read with; ``""`` where none could be read.
    site_symmetry : tuple of str
        Each site's site-symmetry group, the point group of the unitary
        operations that fix it, by its Hermann–Mauguin symbol: ``"3m"``.
    site_group : tuple of tuple of int
        The indices of the operations that fix each site.
    permutations : ndarray, shape (nsym, nsite)
        ``permutations[g, i]``, the site the operation at position ``g`` of
        the group's list carries site ``i`` onto.
    lattice_shift : ndarray, shape (nsym, nsite, 3)
        The lattice vector that image lands in, ``R·τ_i + t − τ_{g(i)}``,
        which sets the phases of a representation.
    wyckoff_spglib, wyckoff_spglib_dataset : tuple of str
        The letters in spglib's setting, one per orbit, and one per site read
        on the positions alone: information about the geometry, never a table
        key.
    max_distance : float
        The largest Cartesian mismatch accepted in matching an image to a
        site.
    """

    orbits: tuple
    orbit_of_site: np.ndarray
    orbit_of_orbital: np.ndarray
    site_of_orbital: np.ndarray
    site_group: tuple
    permutations: np.ndarray
    lattice_shift: np.ndarray
    wyckoff_spglib: tuple
    max_distance: float
    site_positions: np.ndarray = None
    site_orbitals: tuple = ()
    wyckoff_spglib_dataset: tuple = ()
    site_symmetry: tuple = ()
    wyckoff_tables: tuple = ()

    def summary(self) -> str:
        """Where the orbitals sit, a line for each part, with the attribute that holds it."""
        norb, nsite = len(self.site_of_orbital), len(self.site_group)
        site_of = np.asarray(self.site_of_orbital, dtype=int)
        lines = [f"{norb} orbital{'' if norb == 1 else 's'} on {nsite} "
                 f"site{'' if nsite == 1 else 's'}   (.site_of_orbital, .site_positions)",
                 "  the orbits, by Wyckoff letter in the tables' setting and site symmetry   "
                 "(.orbits, .wyckoff_tables, .site_symmetry)"]
        for orbit in self.orbits:
            sites = sorted(int(s) for s in orbit)
            first = sites[0]
            letter = (self.wyckoff_tables[first] if first < len(self.wyckoff_tables)
                      else "") or "?"
            symmetry = (self.site_symmetry[first] if first < len(self.site_symmetry)
                        else "") or "?"
            orbitals = [o for o in range(norb) if int(site_of[o]) in sites]
            lines.append(f"    {letter} ({symmetry}): site{'s' if len(sites) > 1 else ''} "
                         f"{_index_ranges(sites)}, orbital{'s' if len(orbitals) > 1 else ''} "
                         f"{_index_ranges(orbitals)}")
        lines += ["  the operations that fix each site, by index   (.site_group)",
                  "  the site each operation carries each site onto, and the lattice vector "
                  "it lands in   (.permutations, .lattice_shift)"]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        norb, nsite = len(self.site_of_orbital), len(self.site_group)
        parts = []
        for orbit in self.orbits:
            s = int(orbit[0])
            bits = [f"sites {tuple(int(i) for i in orbit)}"]
            if s < len(self.wyckoff_tables) and self.wyckoff_tables[s]:
                bits.append(f"letter {self.wyckoff_tables[s]}")
            if s < len(self.site_symmetry) and self.site_symmetry[s]:
                bits.append(f"site symmetry {self.site_symmetry[s]}")
            parts.append(", ".join(bits))
        return (f"OrbitalAssignment({norb} orbital{'' if norb == 1 else 's'} on "
                f"{nsite} site{'' if nsite == 1 else 's'}: " + "; ".join(parts) + ")")


@dataclass(frozen=True)
class IrrepContent:
    """The coreps each measured multiplet carries at one high-symmetry point.

    What :func:`identify_from_characters` returns, and what
    ``CorepContent.content`` holds; ``summary()`` prints it.

    Attributes
    ----------
    kpname, k : str and ndarray
        The tabulated point, by name and in the model's reduced coordinates.
    labels : tuple of dict
        Each multiplet's coreps, ``{corep: multiplicity}``, one per multiplet.
    dims : ndarray, shape (nrow,)
        Each multiplet's dimension as its coreps make it up: each multiplicity
        times its corep's dimension, summed.
    partition : tuple of int
        Which multiplets carry the same characters: two share a number exactly
        when their characters agree, whatever labelling or phase convention
        the labels rest on.
    indices : tuple of int
        The indices of the little group's unitary operations, the characters'
        columns.
    multiplicities, label_order : ndarray and tuple
        The multiplicities before rounding, a row per multiplet and a column
        per corep, the coreps in the order of ``label_order``.
    integrality : float
        How far the furthest multiplicity sat from an integer.
    """

    kpname: str
    k: np.ndarray
    indices: tuple
    labels: tuple
    multiplicities: np.ndarray
    label_order: tuple
    dims: np.ndarray
    partition: tuple
    integrality: float

    def summary(self) -> str:
        """The content at one k, a line for each part, with the attribute that holds it."""
        lines = [f"{self.kpname} {_point_text(self.k)}, in the model's reduced coordinates   "
                 "(.kpname, .k)",
                 "  each multiplet's coreps, and its dimension   (.labels, .dims)"]
        for row, (labels, dim) in enumerate(zip(self.labels, np.asarray(self.dims).reshape(-1))):
            content = " + ".join((f"{int(m)}x{label}" if int(m) > 1 else str(label))
                                 for label, m in labels.items())
            lines.append(f"    multiplet {row}, dimension {int(round(float(dim)))}: "
                         f"{content or 'no corep'}")
        if len(self.partition) > 1:
            lines.append("  which multiplets carry the same characters, by number: "
                         + ", ".join(str(int(p)) for p in self.partition) + "   (.partition)")
        lines += [f"  the little group's unitary operations, the characters' columns, by "
                  f"index: {_index_ranges(self.indices)}   (.indices)",
                  f"  the multiplicities before rounding, at most {float(self.integrality):.2g} "
                  "from integers   (.multiplicities, .label_order, .integrality)"]
        return "\n".join(lines)


@dataclass(frozen=True)
class KGrid:
    """A Γ-centred mesh that respects the group, with the high-symmetry points it holds.

    What :func:`k_grid` returns; ``summary()`` prints it.  Its :attr:`mesh` is
    pythtb's own, handed to a ``pythtb.WFArray`` as it is.

    Attributes
    ----------
    divisions : tuple of int
        The divisions along the three axes of the three-dimensional cell,
        ``1`` along a layer's vacuum.
    symmetric : bool
        Whether the group carries the mesh onto itself.
    mesh : pythtb.Mesh
        The mesh, ``k`` in ``[0, 1)`` along each periodic direction, one axis
        per periodic direction.
    irreducible : ndarray, shape (n, dim_k), or None
        The points the group carries onto all the others; ``None`` for a mesh
        that is not symmetric.
    contains : dict
        ``{name: bool}``, whether the mesh holds each tabulated high-symmetry
        point.
    missing : tuple of str
        The tabulated points it does not hold, as an odd mesh misses the zone
        boundary; a scan reaches one with its coordinates, from :attr:`hsp`,
        added.
    hsp : dict
        ``{name: k}``, every tabulated point of the zone in the model's
        reduced coordinates.
    """

    divisions: tuple
    symmetric: bool
    mesh: Any
    irreducible: np.ndarray | None
    contains: dict
    missing: tuple
    hsp: dict

    @property
    def points(self) -> np.ndarray:
        """The mesh's points, ``(n, dim_k)``, in the model's reduced coordinates."""
        return np.asarray(self.mesh.flat)

    def summary(self) -> str:
        """The mesh, a line for each part, with the attribute that holds it."""
        sampled = [str(name) for name, hit in self.contains.items() if hit]
        lines = [f"{'x'.join(str(int(d)) for d in self.divisions)} mesh, "
                 + ("symmetric" if self.symmetric else "not symmetric") + " under the group   "
                 "(.divisions, .symmetric)",
                 "  pythtb's own mesh, ready for a WFArray   (.mesh, .points)"]
        if self.irreducible is not None:
            lines.append(f"  its irreducible points, {len(self.irreducible)} of "
                         f"{len(self.points)}: the group carries them onto all the others   "
                         "(.irreducible)")
        lines.append("  the high-symmetry points on it: " + (", ".join(sampled) or "none")
                     + "   (.contains, .hsp)")
        if self.missing:
            lines.append("  the high-symmetry points it misses: "
                         + ", ".join(str(m) for m in self.missing) + "   (.missing)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        sampled = [str(name) for name, hit in self.contains.items() if hit]
        return (f"KGrid({'x'.join(str(int(d)) for d in self.divisions)} mesh, "
                f"{'symmetric' if self.symmetric else 'not symmetric'}; samples "
                + (", ".join(sampled) if sampled else "no high-symmetry point")
                + (f"; misses {', '.join(str(m) for m in self.missing)}"
                   if self.missing else "") + ")")


@dataclass(frozen=True)
class ValidatedGroup:
    """The verdict of the Hamiltonian on each operation of a group, with how well each one holds.

    The record ``realized_group(model).validated`` holds, and the one a
    ``cell_group_not_realized`` refusal carries in ``context["validated"]``;
    ``summary()`` prints it.

    Attributes
    ----------
    verdicts : dict
        ``{ind: verdict}``: ``"obeyed"``; ``"broken"``; or ``"undetermined"``,
        a verdict shared with another operation that acts on ``k`` alike, as an
        operation and its product with the horizontal mirror do on a
        free-standing layer, which the representation then settles.
    indices : tuple of int
        The operations kept, those obeyed and those undetermined.
    obeyed, undetermined, broken : tuple of int
        The indices with each verdict.
    operations : OperationTable
        The operations kept, by index, with their Seitz symbols and names.
    discrepancies : tuple of dict
        One entry per broken operation: its ``ind``; ``refuted_by``, ``"no
        symmetry matrix"`` where no matrix with its action on ``k`` carries
        ``H(k)`` onto itself, a proof, ``"covariance"`` where the nearest one
        misses by more than ``tol``, or ``"out of plane"`` for an operation of
        a layer's slab that the model's ``k`` cannot represent; and its
        ``residual``.
    residuals : dict
        ``{ind: residual}``: each operation's covariance residual,
        ``max_k ‖U H(k) U† − H(Rk)‖`` in the units of ``H``, infinite where no
        matrix exists.
    tol, n_kpts : float and int
        The tolerance the residuals are read against, in the units of ``H``,
        and the number of ``k``-points they are measured on.
    k_action_classes : tuple of tuple of int
        The operations grouped by their action on ``k`` and whether they carry
        time reversal: one measurement settles a class.
    closed : bool
        Whether the operations kept form a group.
    group_label : str
        The group measured, ``"P6/mmm1' #191.234"``.
    """

    indices: tuple
    discrepancies: tuple
    residuals: dict
    # the dimension of each operation's space of solutions, the identity's
    # counting the matrices that commute with H(k) at every k
    _nullspace_dims: dict
    # one member of each kept operation's space of solutions, the unitary
    # nearest it: unnormalized in phase, and generally not of the operation's
    # own form where the space holds more than one matrix
    _intertwiners: dict
    closed: bool
    tol: float
    n_kpts: int
    k_action_classes: tuple = ()
    verdicts: dict = field(default_factory=dict)
    group_label: str = ""
    # what each verdict rests on, per operation solved: the space's dimension,
    # the residual, the solver's own report, the tolerance, cut and window it
    # was read at, and `solved_on`, the operation whose solve this one is
    _records: dict = field(default_factory=dict)
    # each kept operation's whole space of solutions, (dim, nstate, nstate),
    # shared by reference across a k-action class; in the record
    # `realized_group` returns, the members of the operation's own form
    _bases: dict = field(default_factory=dict)
    _group: Any = field(default=None, repr=False, compare=False)
    # ``None`` until each operation's space is read in its own form, once
    # (`_own_form_read`); then, for each operation none of whose members has
    # it, the condition that failed, the covariance of the matrix of that form
    # nearest its space where one was measured, and the detector's whole
    # space -- no matrix of the operation's own, read only for whether any
    # matrix with its action on k has the square the tables ask for
    # (`_antiunitary_square`)
    _own_form: Any = field(default=None, repr=False, compare=False)

    @property
    def operations(self) -> OperationTable:
        """The operations kept, by index, as :func:`operations` writes them.

        Returns
        -------
        OperationTable or Refusal
            A row for each entry of :attr:`indices`.

        Refusals
        --------
        ``tables_not_ready``
            The group has no conventional setting to write a Seitz symbol in.
        """
        table = _operation_rows(self._group)
        if isinstance(table, Refusal):
            return table
        return OperationTable((table[int(i)] for i in sorted(self.indices)),
                              name=table.name, number_str=table.number_str)

    @property
    def broken(self) -> tuple:
        """The indices of the operations the Hamiltonian breaks."""
        return tuple(d["ind"] for d in self.discrepancies)

    @property
    def obeyed(self) -> tuple:
        """The indices of the operations obeyed, each the only one of its class to act on ``k`` as it does."""
        return tuple(i for i in self.indices if self.verdicts.get(i) == "obeyed")

    @property
    def undetermined(self) -> tuple:
        """The indices of the operations kept whose verdict another operation, acting on ``k`` alike, shares.

        The measurement says some operation of the class holds; the
        representation settles which.
        """
        return tuple(i for i in self.indices
                     if self.verdicts.get(i) == "undetermined")

    def summary(self) -> str:
        """The verdicts, a line for each part, with the attribute that holds it."""
        n_kept = len(self.indices)
        lines = [(f"{self.group_label}: " if self.group_label else "")
                 + f"{n_kept} of {n_kept + len(self.discrepancies)} {_PER_COSET} obeyed   "
                 "(.indices, .verdicts)"]
        if bool(getattr(self._group, "spinor", False)):
            lines.append(f"  {_DOUBLE_GROUP}")
        if self.discrepancies:
            table = _operation_rows(self._group) if self._group is not None else {}
            names = [(table[int(i)].name if not isinstance(table, Refusal)
                      and int(i) in table else f"#{int(i)}") for i in self.broken]
            lines.append(f"  broken: {_names_text(names)}   (.broken, .discrepancies)")
        shared = len(self.undetermined)
        if shared:
            lines.append(("  every one acts on k as another does"
                          if shared == n_kept else
                          f"  {shared} act on k as another does")
                         + ", so H alone does not say which of each pair holds, and the "
                         "representation decides   (.undetermined, .k_action_classes)")
        if not self.closed:
            lines.append("  those obeyed do not form a group   (.closed)")
        held = [float(self.residuals[i]) for i in self.indices
                if i in self.residuals and np.isfinite(self.residuals[i])]
        if held:
            lines.append(f"  each one's covariance residual, at most {max(held):.2g} against "
                         f"the tolerance {float(self.tol):.2g}, on {int(self.n_kpts)} k-points   "
                         "(.residuals, .tol, .n_kpts)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        shared = len(self.undetermined)
        return (f"ValidatedGroup({self.group_label + ', ' if self.group_label else ''}"
                f"{len(self.indices)} of {len(self.indices) + len(self.discrepancies)} "
                f"{_PER_COSET} obeyed, which "
                + ("form a group" if self.closed else "do not form a group")
                + (f"; they are {_DOUBLE_GROUP}"
                   if bool(getattr(self._group, "spinor", False)) else "")
                + (f"; {shared} act on every k as another does, so the Hamiltonian alone "
                   "does not say which of each pair holds, and the representation "
                   "decides" if shared else "")
                + ")")


# --------------------------------------------------------------------------
# cell construction
# --------------------------------------------------------------------------

#: The name under which a cell :func:`_decorated_cell` builds carries how many of
#: its points are the model's own orbitals, the leading ones.  The vacuum a slab
#: is given is measured from those alone (:func:`_pad_cell`), so it is set
#: once, by the model, and a species added to the cell -- a substrate's shadow,
#: a decoration, the generic orbit of a descent -- never moves it.
_OWN_ORBITALS = "_pythtqc_own_orbitals"


def _pad_cell(model, vacuum_length: float, *, lattice=None):
    """Present a (possibly two-dimensional) pythtb cell as a 3D slab cell.

    A non-periodic lattice vector is **scaled** here and never turned: by the
    time a layer's cell is given its vacuum, its vector lies along the layer's
    normal, where the slab presentation puts a tilted one together with the
    move of
    the orbitals that goes with it (:func:`_slab_presentation`), so the
    reciprocal vectors dual to the periodic directions — and therefore
    pythtb's reduced k — are untouched; missing dimensions get orthogonal
    vectors of length ``vacuum_length``.  Positions along a vacuum direction are
    re-centred on the origin so that a horizontal mirror appears without a
    translation.

    The vacuum is measured from the **model's own orbitals**: a cell built on
    a model (:func:`_decorated_cell`) is given the vacuum the model is, however
    far out of the model's plane the species added to it sit, so every cell
    built on one model shares that model's lattice, vacuum included.  Where the
    model's own vector is shorter than its orbitals' spread plus
    ``vacuum_length``, the vector's length follows that spread, and two models of
    one layer whose centres differ along the normal by noise are given vacua
    apart by as much.  So a model read **against a group** is set in the group's
    own lattice
    (``lattice``, :attr:`GroupData.lattice`): each of its non-periodic vectors
    scaled to that lattice's length along it, the model then read in the frame
    the group was identified in.

    Returns
    -------
    lattice : ndarray (3, 3)
    positions : ndarray (norb, 3), reduced, wrapped into [0, 1)
    vacuum_dirs : tuple of int
    """
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    lat = np.asarray(model.lat_vecs, dtype=float)
    orb = np.asarray(model.orb_vecs, dtype=float)

    lat3 = np.zeros((3, 3))
    lat3[:dim_r, :dim_r] = lat
    pos3 = np.zeros((orb.shape[0], 3))
    pos3[:, :dim_r] = orb
    own = int(getattr(model, _OWN_ORBITALS, orb.shape[0]))
    own = pos3[:own] if 0 < own <= orb.shape[0] else pos3

    padded = [d for d in range(dim_r) if d not in per]

    # scale the model's own non-periodic vectors until the vacuum reaches
    # vacuum_length -- or, read against a group, to the length its lattice has
    for d in padded:
        length = float(np.linalg.norm(lat3[d]))
        if lattice is not None:
            scale = float(np.linalg.norm(np.asarray(lattice, dtype=float)[d])) / length
        else:
            span = float(own[:, d].max() - own[:, d].min()) if own.shape[0] else 0.0
            scale = max(1.0, span + vacuum_length / max(length, 1e-30))
        lat3[d] = lat3[d] * scale
        pos3[:, d] = pos3[:, d] / scale

    # fill the missing dimensions with an orthogonal complement of the real ones
    if dim_r < 3:
        used = lat3[:dim_r]
        _, _, vh = np.linalg.svd(used)
        comp = vh[dim_r:]
        for j, d in enumerate(range(dim_r, 3)):
            lat3[d] = comp[j] * vacuum_length
            padded.append(d)

    # centre the slab on the origin along every vacuum direction
    for d in padded:
        if pos3.shape[0]:
            mid = 0.5 * (pos3[:, d].max() + pos3[:, d].min())
            pos3[:, d] -= mid

    return lat3, pos3 % 1.0, tuple(sorted(padded))


def _sites_and_typat(positions, typat, symprec, lattice):
    """Group orbitals sitting at the same position into sites, and label species.

    The default species label of a site is its orbitals' own labels, sorted,
    so sites carrying different orbital *content* are never declared
    equivalent while sites carrying the same content are — which is the
    lattice-honest default.  It cannot know that two same-content sites are
    chemically different: that is what ``typat=`` is for, and what
    :func:`_validate_group` measures.
    """
    n = positions.shape[0]
    labels = [0] * n if typat is None else list(typat)
    if len(labels) != n:
        raise ValueError(f"typat must have one entry per orbital ({n}), got {len(labels)}")

    site_pos: list[np.ndarray] = []
    site_orbs: list[list[int]] = []
    for i in range(n):
        for s, p in enumerate(site_pos):
            d = positions[i] - p
            d -= np.round(d)
            if np.linalg.norm(d @ lattice) < symprec:
                site_orbs[s].append(i)
                break
        else:
            site_pos.append(positions[i].copy())
            site_orbs.append([i])

    keys = [tuple(sorted(str(labels[o]) for o in orbs)) for orbs in site_orbs]
    uniq = {k: i + 1 for i, k in enumerate(dict.fromkeys(keys))}
    return (np.array(site_pos), tuple(tuple(o) for o in site_orbs),
            np.array([uniq[k] for k in keys], dtype=int))


# --------------------------------------------------------------------------
# the spinor branch: irrep's fitted spin frame, replaced by the table's own,
# and its per-operation lifts replaced by the tables' own lift SET
# --------------------------------------------------------------------------

_PAULI = np.array([[[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]],
                   [[1, 0], [0, -1]]], dtype=complex)

#: The Cartesian frame the shipped ``SymopTable.S`` matrices are written in,
#: **relative to the conventional cell's standard orientation** (``a`` along
#: ``+x``, ``b`` in the ``+y`` half of the ``xy`` plane, ``c`` with positive
#: ``z`` — the lower-triangular Cholesky factor of the cell's Gram matrix).
#: Tried in the order given; the first that reproduces every shipped ``S`` as a
#: conjugate of the operation's own SU(2) lift, up to a sign, is the group's.
#: These two frames between them cover all 460 unitary and all 460 grey shipped
#: tables — see :func:`_canonical_spinor_signs`.
_TABLE_SPIN_FRAMES = (
    ("standard", np.eye(3)),
    ("hexagonal", np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])),
)


def _su2_lift(rot) -> np.ndarray:
    """An SU(2) lift of a proper rotation ``rot``.

    Either lift will do: this matrix is only ever used to **conjugate**, and the
    two lifts differ by an overall sign that cancels there.  Closed form — the
    axis and angle of a 3×3 rotation, then ``cos(θ/2) − i sin(θ/2) n·σ``.  No
    eigen-decomposition and no optimizer, so the lift is a deterministic
    function of the bits in ``rot``.
    """
    rot = np.asarray(rot, dtype=float)
    cos = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(cos))
    if ang < 1e-9:
        return np.eye(2, dtype=complex)
    if abs(ang - np.pi) < 1e-6:
        # sin θ = 0: the axis comes from (R + 1)/2 = n nᵀ, not from the
        # antisymmetric part.  Pick the best-conditioned column and orient it by
        # its first non-negligible entry, so the branch is a function of `rot`.
        sym = (rot + np.eye(3)) / 2.0
        col = int(np.argmax(np.diag(sym)))
        axis = sym[:, col] / np.sqrt(max(float(sym[col, col]), 1e-30))
        lead = int(np.argmax(np.abs(axis) > 1e-9))
        if axis[lead] < 0:
            axis = -axis
    else:
        axis = np.array([rot[2, 1] - rot[1, 2],
                         rot[0, 2] - rot[2, 0],
                         rot[1, 0] - rot[0, 1]]) / (2.0 * np.sin(ang))
    axis = axis / np.linalg.norm(axis)
    return (np.cos(ang / 2) * np.eye(2, dtype=complex)
            - 1j * np.sin(ang / 2) * np.einsum("i,ijk->jk", axis, _PAULI))


def _lift_set_cocycle(rots, lifts, corrections):
    r"""The ℤ₂ factor system of a lift set, and where a sign vector twists it.

    A set of SU(2) matrices :math:`\{S(g)\}` indexed by the rotations
    :math:`\{R_g\}` of a group is a *lift set*: since :math:`R_g R_h = R_{gh}`,
    the two matrices :math:`S(g)S(h)` and :math:`S(gh)` lift the same rotation
    and therefore differ by a sign,

    .. math::  S(g)\,S(h) \;=\; \omega(g,h)\;S(gh), \qquad \omega \in \{\pm 1\},

    which is the ℤ₂ half of the group's factor system (the translation half is
    the ``e^{-2\pi i k\cdot t}`` phase, and it is a separate story — see
    :func:`_transported_table`).  Twisting a lift set by a sign vector
    :math:`b(g)` multiplies :math:`\omega` by the **coboundary**
    :math:`(\delta b)(g,h) = b(g)b(h)b(gh)`, so two lift sets of the same
    rotations related by :math:`b` carry the same :math:`\omega` **iff** ``b`` is
    a homomorphism onto :math:`\{\pm1\}`.  That is the whole content of the
    mismatch this machinery exists to repair: ``irrep`` orients each operation's
    rotation axis from an eigenvector, one operation at a time, so its lift set
    is a *per-operation* arbitrary sign away from the tables' — and a
    per-operation sign vector is generically **not** a homomorphism.

    Parameters
    ----------
    rots : sequence of integer arrays
        One rotation per operation, in a basis in which composition is the
        matrix product (any basis will do — the same one for all of them).
    lifts : sequence of (2, 2) arrays
        The lift set whose factor system is wanted.
    corrections : sequence of int
        A ``±1`` per operation.  The returned ``mixed`` counts the pairs at which
        ``δ(corrections)`` is ``−1``, i.e. the pairs where the *uncorrected* set
        disagrees with the corrected one as a factor system.

    Returns
    -------
    tuple
        ``(defect, mixed, pairs, worst_at)`` — the worst distance from ``±1``,
        the number of composable pairs the correction twists, the number of
        composable pairs examined, and the ``(g, h)`` 1-based indices of the
        worst pair.

    Notes
    -----
    Operations that share a rotation (a centred listing, or a screw and its
    coset partner) share a lift, so keying by the rotation alone is
    well-defined; the first index carrying each rotation is used.

    The statement is about the SU(2) set indexed by **rotations**, so a grey
    group's antiunitary half enters through the rotations of its unitary
    partners (whose lifts it shares) and contributes no new content — it doubles
    the pair count and nothing else.  That is as it should be: the bar is a
    property of the *valuedness*, not of time reversal, and the antiunitary
    branch ``iσ_y conj(S)`` is a function of the same ``S``.
    """
    rot = np.round(np.asarray(rots, dtype=float)).astype(int)
    lift = np.asarray(lifts, dtype=complex)
    sign = np.asarray(corrections, dtype=int)
    n = len(rot)
    where: dict[bytes, int] = {}
    for i in range(n):
        where.setdefault(rot[i].tobytes(), i)
    prod = np.einsum("iab,jbc->ijac", rot, rot)
    idx = np.array([[where.get(prod[i, j].tobytes(), -1) for j in range(n)]
                    for i in range(n)], dtype=int)
    ok = idx >= 0                             # the listing is closed at these pairs
    ii, jj = np.nonzero(ok)
    kk = idx[ii, jj]
    lam = np.einsum("nab,nbc,nac->n", lift[ii], lift[jj], lift[kk].conj()) / 2
    dev = np.maximum(np.abs(np.abs(lam) - 1.0), np.abs(lam.imag))
    pairs = int(ii.size)
    if not pairs:                             # pragma: no cover - defensive
        return 0.0, 0, 0, None
    worst = int(np.argmax(dev))
    mixed = int(np.count_nonzero(sign[ii] * sign[jj] * sign[kk] < 0))
    return (float(dev[worst]), mixed, pairs,
            (int(ii[worst]) + 1, int(jj[worst]) + 1))


def _canonical_spinor_signs(sg, *, tol=1e-3):
    r"""The ``±1`` per operation that carries ``irrep``'s lifts onto the tables'.

    *What the sign is.*  ``irreptables`` ships, with every tabulated operation,
    the SU(2) matrix ``S`` its characters were computed in; ``irrep`` builds its
    own lift ``sym.spinor_rotation`` from the operation's Cartesian axis and
    angle in **this model's** frame.  The two are the same lift of the same
    rotation seen in two Cartesian frames, so they are related by one spin-basis
    conjugation and a sign per operation,

    .. math::  S_\text{stored}(g) \;=\; b(g)\;
               R\, S_\text{table}(g)\, R^{\dagger},

    which ``irrep`` stores as ``sym.sign`` and this package instead **applies to
    the lift** (:func:`_pin_spinor_branch`).  ``irrep`` obtains ``R`` by
    ``scipy.optimize.minimize`` from an **unseeded** ``np.random.random`` start
    (``match_spinor_rotations``), and the residual it minimises is blind to the
    sign, so *both* branches of the fit are exact minima: the fitted branch — and
    therefore which tabulated name lands on which measured level — changes from
    run to run.

    *Why the sign cannot be left on the character.*  ``b`` is fixed one operation
    at a time (``irrep`` orients each rotation's axis from an eigenvector of the
    Cartesian rotation matrix, ``_get_operation_type``; ``np.linalg.eig`` fixes
    that eigenvector only up to a sign, and flipping the axis flips the lift for
    **every** angle, since ``exp(iπ n·σ) = −1``).  A per-operation sign vector is
    generically not a homomorphism, and a lift set twisted by a non-homomorphic
    ``b`` carries a *different* ℤ₂ factor system: ``ω_stored = (δb)·ω_table`` with
    ``(δb)(g,h) = b(g)b(h)b(gh)`` (see :func:`_lift_set_cocycle`).  Multiplying
    the tabulated characters by ``b`` then prints a page in ``ω_stored`` rather
    than in BCS's ``ω_table``, and the two differ element by element wherever
    the bars mix — the cross-check is ``K`` of ``#183.186``, where BCS's ``−K4``
    carries ``−i`` on all three mirrors and leaving the sign on the character
    would return ``(+i, −i, −i)``.  What has to be solved instead is the
    coboundary equation ``δc = ω_stored/ω_table``, whose solution is ``c = b`` up
    to a ℤ₂ character and which the frame ``R`` pins outright; applying it to
    the lifts leaves ``sym.sign`` ``+1`` on every operation and the returned page
    BCS's own.

    *Why there are exactly two.*  If ``R`` solves the system so does ``R U`` for
    any ``U`` with ``U S_\text{table}(g) U^\dagger = \chi(g) S_\text{table}(g)``,
    and ``χ`` is then a real linear character of the point group; the sign
    vector picks up that twist.  ``χ`` is forced to ``+1`` on every operation
    whose spinor lift has a non-zero trace (conjugation preserves traces), which
    leaves only the operations whose *proper part* is a two-fold rotation —
    mirrors, and the in-plane two-folds of an axial group.  For a group whose
    improper operations are all mirrors (crystal classes ``m``, ``mm2``, ``3m``,
    ``4mm``, ``6mm``) that twist is exactly the **determinant character**:
    ``+1`` on the proper operations, ``−1`` on the improper ones.  SG 183's
    ``P6mm1'`` is such a group — the two branches are
    ``(1,1,1,−1,1,1,+1,+1,−1,−1,+1,+1)`` and
    ``(1,1,1,−1,1,1,−1,−1,+1,+1,−1,−1)``, whose ratio is ``det R`` operation by
    operation — and at ``K`` it exchanges the conjugate pair ``−K4 ↔ −K5``.

    *How this pins it.*  ``R`` is not a free parameter: it is the SU(2) lift of
    the **frame map** carrying the Cartesian frame the tables' ``S`` are written
    in onto this cell's own, and that map is data.  The conventional cell in
    this cell's Cartesian frame is ``A_c = refUCᵀ A``; the same cell in its standard
    orientation is the lower-triangular Cholesky factor ``L`` of ``A_c A_cᵀ``
    (``a`` along ``+x``, ``b`` in the ``+y`` half plane, ``c`` with positive
    ``z``); so ``F = A_cᵀ L^{-T}`` is orthogonal and takes the standard
    orientation to this cell's.  The tables' own frame sits at a fixed offset ``T``
    from that standard orientation — ``T = 1`` for six of the seven lattice
    systems and a single fixed rotation for the hexagonal one (see
    :data:`_TABLE_SPIN_FRAMES`) — and ``R = _su2_lift(F T)``.  An improper
    ``F T`` is lifted through its proper part, which is the same physical
    convention that makes a spatial inversion act trivially on spinors.

    *And it is checked, not assumed.*  Each candidate frame is **verified** on
    the group in hand: it is accepted only if every shipped ``S``, so
    conjugated, reproduces that operation's own ``sym.spinor_rotation`` up to a
    real ``±1``.  A wrong frame fails by :math:`\mathcal{O}(1)`, so the offsets
    in :data:`_TABLE_SPIN_FRAMES` are re-measured at every call rather than
    trusted.  Where the frames the tables admit differ only by the two-fold
    rotation about the principal axis — the residue the shipped data itself
    cannot resolve, and the whole content of the branch — the ordered list
    settles it, identically for every group and every model.

    *That per-operation match is the solvability statement.*  ``c = b`` solves
    the coboundary equation because every ``S_stored(g)`` is ``±`` the carried
    table matrix; a frame under which some operation is **not** ``±`` the table's
    is a frame under which the equation has no solution at all, and that is the
    ``InternalError`` below.  What remains is a property of the *shipped data*
    and is measured here rather than assumed: the tables' own ``S`` must close as
    a lift set, ``S_t(g)S_t(h) = ω_table(g,h) S_t(gh)`` with ``ω_table = ±1``
    (it does, at all 460 shipped double-valued instances, with a worst ``|½tr|``
    deviation of ~1e-5 — the tables' own printed precision).

    Parameters
    ----------
    sg : irrep.spacegroup_irreps.SpaceGroupIrreps
        A **spinor** group whose ``set_irreptables`` matching succeeded, i.e.
        whose ``symmetries`` are sorted into table order.
    tol : float, optional
        Absolute tolerance on "the conjugated table matrix is ``±1`` times the
        stored lift".  Default ``1e-3``: ``irreptables`` writes ``S`` to about five
        decimals (worst 1.9e-5 over the shipped tables), and the competing
        branch is off by order 1, so nothing in between is possible.

    Returns
    -------
    dict
        ``{"frame": name, "correction": T, "spin_transf": R, "signs": tuple,
        "defect": float, "cocycle": float, "mixed": int, "pairs": int}`` —
        ``signs[i]`` is the ``±1`` carrying ``sg.symmetries[i]``'s stored lift
        onto the tables'; ``cocycle`` is how far the tables' own lift set sits
        from closing (``½ tr`` against ``±1``); ``mixed`` counts the composable
        pairs at which ``δ(signs) = −1``, i.e. the pairs at which the stored
        lifts realize a *different* factor system from the tables' — ``0`` means
        the stored set was already ω_table-consistent and the correction is a
        harmless character twist.

    Raises
    ------
    ..refusal.InternalError
        When no shipped frame reproduces the tables' ``S`` (the coboundary
        equation has no solution), or when the tables' own ``S`` do not close as
        a lift set (there is no ``ω_table`` to solve against).  Both are
        statements about *this package's* use of a convention the tables fix
        outright — not measurements about a model — so they are raised and never
        returned as a :class:`..refusal.Refusal`.
    """
    from irreptables.irreps import IrrepTable

    from .refusal import InternalError

    table = IrrepTable(str(sg.number_str), True, magnetic=bool(sg.magnetic))
    tsyms = table.u_symmetries + table.au_symmetries
    where = (f"group #{sg.number_str}: the tables fix the spin frame completely "
             f"(the shipped SU(2) matrices are written in the conventional cell's "
             f"own Cartesian frame). Failing to pin it is a bug in this package's "
             f"application of that convention, not a property of the model")
    if len(tsyms) != len(sg.symmetries):
        raise InternalError(
            f"{where}: the group carries {len(sg.symmetries)} operations against the "
            f"table's {len(tsyms)}, so there is no operation-by-operation matching "
            f"to write a spin frame against.")

    lattice = np.asarray(sg.real_lattice, dtype=float)
    conv = np.asarray(sg.refUC, dtype=float).T @ lattice     # conventional cell, this frame
    gram = conv @ conv.T
    try:
        standard = np.linalg.cholesky(gram)                  # ... in standard orientation
    except np.linalg.LinAlgError as exc:                     # pragma: no cover - defensive
        raise InternalError(
            f"{where}: the conventional cell refUCᵀ·lattice is singular "
            f"(Gram determinant {np.linalg.det(gram):.3g}), so it has no standard "
            f"orientation to write the tables' frame against: {exc}") from None
    frame = conv.T @ np.linalg.inv(standard.T)
    ortho = float(np.abs(frame @ frame.T - np.eye(3)).max())
    if ortho > 1e-6:                                         # pragma: no cover - defensive
        raise InternalError(
            f"{where}: the map from the standard orientation of the conventional cell "
            f"to this one is not orthogonal (defect {ortho:.3g}), so it is not a change "
            f"of Cartesian frame and no spin rotation corresponds to it.")

    ours = [np.asarray(s.spinor_rotation, dtype=complex) for s in sg.symmetries]
    theirs = [np.asarray(t.S, dtype=complex) for t in tsyms]
    tried = []
    for name, correction in _TABLE_SPIN_FRAMES:
        full = frame @ correction
        rot = full if np.linalg.det(full) > 0 else -full     # improper frames act
        spin = _su2_lift(rot)                                # ... through their proper part
        signs, defect, bad = [], 0.0, None
        for i, (S1, S2) in enumerate(zip(ours, theirs)):
            carried = spin @ S2 @ spin.conj().T
            lam = complex(np.trace(S1 @ carried.conj().T) / 2)
            sign = int(np.sign(lam.real)) or 1
            d = float(max(np.abs(S1 - sign * carried).max(), abs(abs(lam) - 1.0)))
            defect = max(defect, d)
            if d > tol and bad is None:
                bad = (i, lam, d)
            signs.append(sign)
        tried.append((name, defect, bad))
        if bad is None:
            # The tables' own lift set is what the corrected lifts will BE, so
            # its closure is the last thing between here and "the returned factor
            # system is the tables'".  Checked on the rotations the tables list
            # them under.
            cocycle, mixed, pairs, at = _lift_set_cocycle(
                [t.R for t in tsyms], theirs, signs)
            if cocycle > tol:
                raise InternalError(
                    f"{where}: the shipped SU(2) matrices do not close as a lift set — "
                    f"S(g)S(h) is not ±S(gh) at operations {at} (½·tr sits {cocycle:.3g} "
                    f"from ±1, against a tolerance of {tol:.3g}). There is then no "
                    f"tabulated ℤ₂ factor system to write this group's lifts against, and "
                    f"the coboundary equation δc = ω_ours/ω_table has no right-hand side.")
            return {"frame": name, "correction": correction, "spin_transf": spin,
                    "signs": tuple(signs), "defect": defect, "cocycle": cocycle,
                    "mixed": int(mixed), "pairs": int(pairs)}
    report = "; ".join(
        f"{name!r} off by {defect:.3g} (first at operation #{bad[0] + 1}, "
        f"ratio {bad[1]:+.4f})" for name, defect, bad in tried)
    raise InternalError(
        f"{where}: none of the {len(tried)} shipped table frames reproduces this group's "
        f"SU(2) matrices — {report} — against a tolerance of {tol:.3g}. The shipped S "
        f"matrices and irrep's own lift are two descriptions of the same rotations, so one "
        f"conjugation must carry either onto the other; that none does means the frame "
        f"offsets recorded in _TABLE_SPIN_FRAMES do not describe the installed "
        f"irreptables, and every transported character would be silently mis-signed.")


def _pin_spinor_branch(sg) -> dict | None:
    """Put this group's SU(2) lifts in the tables' own convention.

    :func:`_canonical_spinor_signs` returns the ``±1`` per operation that carries
    ``irrep``'s stored lift onto the tables'.  ``irrep`` keeps that number as
    ``sym.sign`` and multiplies it into the *transported characters*; this
    package instead multiplies it into the **lift** and leaves ``sym.sign`` at
    ``+1``.  The two are the same arithmetic at one k-point and are not the same
    convention at all:

    * a per-operation sign vector is generically not a homomorphism, so a lift
      set twisted by it carries a *different* ℤ₂ factor system
      (:func:`_lift_set_cocycle`).  Correcting the characters leaves the lifts —
      and therefore the whole projective structure everything downstream is built
      on, from ``symmetry._build_rep``'s internal factor to the product table — in
      that other convention, and the returned page then disagrees with BCS's
      printed page element by element wherever the bars mix;
    * correcting the **lift** makes ``S(g)S(h) = ω_table(g,h)·S(gh)`` hold on the
      nose, so every consumer inherits the tables' convention at once and the
      returned page is BCS's.

    Multiplying ``irrep``'s own analytic lift by ``±1`` — rather than storing the shipped
    ``S`` conjugated into this frame — is deliberate: the two agree to the
    tables' five printed decimals, and keeping ``irrep``'s matrix preserves the
    ``1e-16`` unitarity and covariance residuals every gate downstream reports.

    The interception point is this package's own boundary — ``irrep`` is left
    exactly as installed.  A spinless group is left alone (``irrep`` never fits
    anything there: ``match_symmetries(signs=False)`` returns all ``+1`` and the
    tables ship ``S = 1``), and ``None`` comes back to say so.
    """
    if not bool(sg.spinor):
        return None
    record = _canonical_spinor_signs(sg)
    for sym, sign in zip(sg.symmetries, record["signs"]):
        if int(sign) < 0:
            sym.spinor_rotation = -np.asarray(sym.spinor_rotation, dtype=complex)
        sym.sign = 1                          # absorbed into the lift, not carried
    sg.spin_transf = record["spin_transf"]   # irrep uses it only for `SymmetryOperation.show`
    return {"frame": record["frame"], "signs": record["signs"],
            "bars": int(sum(1 for s in record["signs"] if s < 0)),
            "defect": record["defect"], "cocycle": record["cocycle"],
            "mixed": record["mixed"], "pairs": record["pairs"],
            "lifts": "table", "source": "table_frame"}


# --------------------------------------------------------------------------
# magnetic scope: the validated universe is types I and II
# --------------------------------------------------------------------------

def _magnetic_type(group) -> str:
    r"""Which of the four magnetic-group types this operation list realizes.

    ``"I"`` (unitary), ``"II"`` (grey), ``"III"`` or ``"IV"`` — **measured from
    the operations**, never parsed out of a BNS number.  Takes anything that
    carries an ``au_symmetries`` list: a :class:`GroupData` or ``irrep``'s own
    ``SpaceGroup``, so the test can be made before the record is built.

    The four cases are the standard Bravais–Heesch classification of a magnetic
    space group :math:`M` over its unitary halving subgroup :math:`G`, read off
    the antiunitary coset:

    ===== =============================================================
    type  what the antiunitary operations look like
    ===== =============================================================
    I     there are none — :math:`M = G`, an ordinary space group
    II    the coset contains time reversal itself, :math:`\{1|0\}\cdot T`
          (grey, :math:`M = G + G\,T`; ``include_TR=True`` builds these)
    III   antiunitary operations exist, but none of them is a translation
          times :math:`T` — every one carries a rotation or reflection
    IV    the coset contains :math:`\{1|t\}\cdot T` with :math:`t` a
          **fractional** translation (an anti-translation; the ``P_C``,
          ``P_a``, … BNS symbols)
    ===== =============================================================

    ``t`` is read **modulo the lattice**, and the lattice meant is the integer
    lattice of the cell handed in — which is the lattice of :math:`G` by
    construction, since :func:`_group_of` returns the operations spglib found on
    *that* cell.  A supercell's own fractional translations are **unitary**
    operations of :math:`G` there (``GroupData._lattice_index``) and do not
    enter this classification: measured on the ``2x1x1`` simple-cubic supercell,
    the grey supergroup meets :math:`\{1|0\}\cdot T` and reads ``"II"``, not
    ``"IV"``, exactly as a primitive grey cell does.

    All four types come out of ordinary cells — one honeycomb cell and one
    two-site square cell reach every one of them: no ``magmom`` →
    ``191.234 P6/mmm1'`` (II); moments ``±z`` on the two sublattices →
    ``191.236 P6'/mm'm`` (III, 12 antiunitary
    operations, none with an identity rotation); a generic non-collinear pair →
    ``1.1 P1`` (I); ``±z`` on ``(0,0)`` and ``(½,½)`` of a square cell →
    ``127.397 P_C4/mbm`` (IV, the antiunitary ``{1|(½,½,0)}·T``).
    """
    au = list(getattr(group, "au_symmetries", ()) or ())
    if not au:
        return "I"
    anti_translation = False
    for op in au:
        rot = np.asarray(getattr(op, "rotation", np.zeros((3, 3))), dtype=float)
        tau = np.asarray(getattr(op, "translation", np.ones(3)), dtype=float)
        if rot.shape == (3, 3) and np.allclose(rot, np.eye(3), atol=1e-8):
            if np.abs(((tau + 0.5) % 1.0) - 0.5).max() <= 1e-6:
                return "II"                     # {1|0}·T, up to a lattice vector
            anti_translation = True             # {1|t}·T with t fractional
    return "IV" if anti_translation else "III"


class _CosetView:
    """The minimum :func:`_magnetic_type` reads: a list of antiunitary operations.

    Its docstring says it takes "anything that carries an ``au_symmetries``
    list", and a *measured survivor set* is exactly such a thing.  Wrapping it
    is how the type of a realization is read with the same detector that types
    a group, rather than with a second one written for the occasion.
    """

    def __init__(self, operations):
        self.au_symmetries = list(operations)


def _antiunitary_entry(group, validated, ind) -> dict:
    """One operation combined with time reversal, as :attr:`RealizedGroup.beyond_grey` names it.

    Its index ``ind`` in the group it was measured in, that group's label
    (``of``), its ``name`` in the generator frame ``characters=`` is written
    in, its ``rotation`` and ``translation`` in the cell's reduced
    coordinates, and the ``residual`` and ``verdict`` the detector measured.
    Where the translation is not a lattice vector of the cell the name carries
    it, in Seitz form in the cell's reduced coordinates -- ``{E|1/2,0,0}'``,
    the half translation of an antiferromagnet's magnetic cell combined with
    time reversal, and never time reversal's own ``E'`` (:func:`_record_name`).
    """
    sym = next(s for s in group.symmetries if int(s.ind) == int(ind))
    return {"ind": int(ind),
            "name": _record_name(sym, group.lattice),
            "of": str(group.number_str),
            "rotation": tuple(tuple(int(round(x)) for x in row)
                              for row in np.asarray(sym.rotation, dtype=float)),
            "translation": tuple(float(x) for x in
                                 np.asarray(sym.translation, dtype=float).reshape(-1)),
            "residual": float(validated.residuals.get(int(ind), float("nan"))),
            "verdict": str(validated.verdicts.get(int(ind), "obeyed"))}


def _antiunitary_settled(group, validated, refuted, square, kept=None, *,
                         tolerances=EXACT) -> dict:
    r"""What one round measured of the operations combined with time reversal it asks last.

    A descent keeps the unitary operations measured to hold, ``kept``, and the
    group it arrives at is asked with time reversal adjoined again, so a
    ``g·T`` whose ``g`` is kept is measured again there.  One whose ``g`` is
    left out -- a mirror the Hamiltonian breaks, combined with time reversal
    -- is asked nowhere else, and this round's verdict on it is its last.  So
    is every operation combined with time reversal of the group time reversal
    is measured on (``kept`` left out), the last group that asks any of them:
    a descent after it -- the projective class of spinor states read
    spinless -- can still leave their unitary part out.

    What is kept is the measurement and no conclusion: whether the detector
    kept the operation, whether its tensor structure or its spin lift refuted
    it (``refuted``), its matrix, and, where its unitary part squares to the
    identity, its own square ``X X̄``, a sign no rescaling of ``X`` changes,
    against the sign the group's multiplication table gives it
    (:func:`_antiunitary_square`; time reversal's own is ``square``,
    :func:`_time_reversal_square`), the member of that sign being the matrix
    kept.  The operations are read once, after the group settles, against the
    group returned (:func:`_antiunitary_held`).

    Returns ``{"group", "ops", "detector", "refuted", "n", "closed",
    "reversal"}``: this round's group; per operation asked last here, keyed by
    its ``.ind``, the operation itself (``sym``), its entry as
    :attr:`RealizedGroup.beyond_grey` names it (``entry``,
    :func:`_antiunitary_entry`), its matrix (``member``, ``None`` where it
    is refuted) and the solution space it is a member of (``space``), which
    holds more than it where the Hamiltonian conserves something, a spin
    rotation; the operations the detector refuted, time reversal among them
    where its square is not the sign asked; those refuted here at all, by the
    detector, by their tensor structure or spin lift, or by their own square;
    how many were asked last here; whether the operations the detector kept
    close in this group; and whether it is the group time reversal was
    measured on.  ``tolerances`` is the record of what the call declared, which
    every reading of a solution space here judges noise against.
    """
    antiunitary = {int(s.ind): s for s in group.symmetries if s.time_reversal}
    if kept is None:
        asked = set(antiunitary)
    else:
        kept = {int(i) for i in kept}
        staying = {_operation_key(s)[:2] for s in group.symmetries
                   if not s.time_reversal and int(s.ind) in kept}
        asked = {i for i, s in antiunitary.items() if _operation_key(s)[:2] not in staying}
    detector = {i for i in antiunitary if i in set(validated.broken)}
    reversal = None if square is None else int(square["ind"])
    if square is not None and not square["obeyed"]:
        detector.add(reversal)
    failed = detector | {i for i in antiunitary if i in set(refuted or ())}
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    ops = {}
    for i in sorted(asked):
        member = None if i in failed else validated._intertwiners.get(i)
        if member is not None and i != reversal:
            # its own square, where its unitary part squares to the identity:
            # the member of the sign the table gives it is the matrix kept
            got = _antiunitary_square(group, validated, at[i], tolerances=tolerances)
            if got is not None:
                member = got["member"] if got["obeyed"] else None
        if member is None:
            failed.add(i)
        space = None if member is None else (getattr(validated, "_bases", None) or {}).get(i)
        ops[i] = {"sym": antiunitary[i], "entry": _antiunitary_entry(group, validated, i),
                  "member": None if member is None else np.asarray(member, dtype=complex),
                  "space": None if space is None else np.asarray(space, dtype=complex)}
    return {"group": group, "ops": ops, "detector": frozenset(detector),
            "refuted": frozenset(failed), "n": len(asked),
            "closed": bool(validated.closed), "reversal": kept is None}


def _time_reversal_measured(group, validated, refuted=None, *, tolerances=EXACT) -> tuple:
    """The verdict on the antiunitary coset, and what it costs.

    A cell is asked as a **grey** group — that is the honest allowance, since a
    point set says nothing about a Hamiltonian's time reversal — and the
    detector then measures the coset.  Two things can come back:

    * **every antiunitary operation obeyed.**  The realization is grey; nothing
      changes, and nothing beyond grey is read.
    * **any one refuted.**  Time reversal is measured broken and the
      realization is a *unitary* group.  The same cell is re-identified
      with ``include_TR=False``, so the labels, EBR rows and indicator that
      follow are the unitary group's own — a grey record with rows deleted
      would be neither.

    **Time reversal is obeyed only as the tables read it.**  Its square
    ``X X̄`` is a sign that belongs to the Hamiltonian (:func:`_time_reversal_square`),
    and the tables ask for one: ``+1`` single-valued, ``-1`` double-valued.  A
    time reversal whose solution space holds no member of the sign asked is
    broken, and the realization descends as it does for any Hamiltonian that
    breaks time reversal -- which is what spinor states in the orbital index of
    a spinless model are, read in single-valued tables.  Where a member of the
    sign asked had to be chosen, it is the intertwiner every later step reads.

    **What holds of the rest is read once, against the group returned.**  On
    the group time reversal is measured on every unitary operation holds, and
    a ``g·T`` kept beside its own ``g`` makes time reversal itself,
    ``T = g⁻¹·(g·T)``; but a descent after it can still leave ``g`` out --
    the projective class of spinor states read spinless lowers the unitary
    group once time reversal is measured, and the flattened ``(p_x, p_y)``
    doublet of ``P4/mmm1'`` returns ``P4/m``, beside which the eight
    operations of ``4/mmm`` outside ``4/m`` combined with time reversal make
    ``P4/mm'm'``.  So this round's verdict on every operation combined with
    time reversal is kept (:func:`_antiunitary_settled`) beside each earlier
    descent's on the ones it left behind -- Haldane's six mirrors combined
    with time reversal, which the group time reversal is measured on does not
    hold -- and all of them are read once the group has settled
    (:func:`_beyond_grey`): the record's ``n_obeyed``, ``n_broken`` and
    ``note`` are written there, by the rule :attr:`RealizedGroup.beyond_grey`
    names its operations by, and an operation whose unitary part is in the
    group returned never counts, making time reversal with it.  Where time
    reversal is obeyed, nothing of them is said.

    ``refuted`` is the verdict stage's own refutations on this group: an
    operation combined with time reversal its own form or its tensor structure
    refutes is not one of the Hamiltonian's, and time reversal itself refuted
    there is broken as one of the wrong square is -- by its spin lift, no
    member of its solution space acting as ``iσ_y`` on every spin pair, as on
    spin pairs declared across two Wannier functions of one site, or by its
    site permutation, no member carrying each site's states onto its own, as
    in an antiferromagnet whose moments alternate between sites the cell's
    point set does not tell apart.

    Returns ``(record, asked, validated)``: the narration that always rides
    on the returned group, whose count and note :func:`_beyond_grey` writes
    where time reversal is broken; what this round measured of the operations
    combined with time reversal (:func:`_antiunitary_settled`), ``None``
    where time reversal is obeyed; and the detector's record with time
    reversal's member of the sign asked in place.  ``tolerances`` is the record
    of what the call declared, which every reading of a solution space here
    judges noise against.
    """
    antiunitary = {int(s.ind): s for s in group.symmetries if s.time_reversal}
    broken = {i for i in validated.broken if i in antiunitary}
    square = (_time_reversal_square(group, validated, tolerances=tolerances)
              if antiunitary else None)
    if square is not None and not square["obeyed"]:
        broken.add(int(square["ind"]))
    # ... and time reversal whose own tensor structure the verdict stage refuted:
    # no member of its solution space carries its lift, i sigma_y on every spin
    # pair, as on spin pairs declared across two Wannier functions of one site
    lift = (None if square is None
            else (refuted or {}).get(int(square["ind"])))
    lifted = (isinstance(lift, dict)
              and str(lift.get("refuted_by", "tensor structure")) == "tensor structure")
    if lifted:
        broken.add(int(square["ind"]))
    validated = _time_reversal_member(validated, square)
    obeyed = sorted(set(antiunitary) - broken)
    undetermined = tuple(i for i in obeyed
                         if validated.verdicts.get(i) == "undetermined")

    record = {
        "n_antiunitary": len(antiunitary),
        "n_obeyed": len(obeyed), "n_broken": len(broken),
        "undetermined": undetermined, "grey_number_str": str(group.number_str),
        "grey_name": str(group.name),
        "square": (None if square is None else
                   {"asked": int(square["asked"]), "measured": float(square["measured"])}),
    }
    if not antiunitary or not broken:
        record["outcome"] = "kept"
        record["note"] = ("the cell was asked as a grey group and every antiunitary "
                          "operation of it is obeyed, so the realization is grey"
                          if antiunitary else
                          "the cell was asked without time reversal, so there was no "
                          "antiunitary coset to measure")
        return record, None, validated

    record["outcome"] = "measured broken"
    itself = ""
    if lifted and lift.get("own_form") == "site permutation":
        # its matrices carry the sites as another operation does: no square of
        # time reversal's own is read off them
        itself = (" Time reversal itself is refuted by its site permutation: no member of "
                  "its solution space carries each site's states onto its own as time "
                  "reversal does -- the matrices with time reversal's action on k carry "
                  "the sites as time reversal composed with another operation does, a "
                  "translation in an antiferromagnet whose moments, or currents, "
                  "alternate between sites the cell's point set does not tell apart.")
    elif square is not None and not square["obeyed"]:
        tables = "single-valued" if square["asked"] > 0 else "double-valued"
        measured = float(square["measured"])
        if np.isfinite(measured):
            kind = ("a spinor's time reversal" if measured < 0
                    else "a spin-free time reversal")
            said = (f"with X its intertwiner the square X conj(X) measures "
                    f"{measured:+.6g}, {kind}, where the {tables} tables read here ask "
                    f"for {square['asked']:+d} -- a rescaling of X changes the square only "
                    f"by a positive factor, so the sign is the Hamiltonian's own, and no "
                    f"member of its solution space carries the other.")
        else:
            said = (f"no member X of its solution space squares, X conj(X), to the "
                    f"{square['asked']:+d} the {tables} tables read here ask for.")
        itself = (
            f" Time reversal itself is broken by its square: {said}"
            + (" A spinless model carrying spinors in its orbital index, as a wannier90 "
               "import does, reads this way; `spin=` declares which orbitals are spin "
               "partners, and the double-valued tables are then read."
               if square["asked"] > 0 else ""))
    elif lifted:
        itself = (" Time reversal itself is refuted by its spin lift: no member of its "
                  "solution space acts as i sigma_y on every spin pair, as time reversal "
                  "acts on the two states of a spinor. A declared spinor basis whose "
                  "pairs join the states of different Wannier functions reads this way.")
    # WHAT THIS ROUND ASKED of every operation combined with time reversal: the
    # last group to ask any of them, a descent after it -- the projective class
    # of spinor states read spinless -- still able to leave a unitary part
    # out.  Its count and its note are written where they are read, once,
    # against the group returned (`_beyond_grey`).
    asked = dict(_antiunitary_settled(group, validated, refuted, square,
                                      tolerances=tolerances),
                 itself=itself)
    record["n_obeyed"] = record["n_broken"] = record["note"] = None
    return record, asked, validated


#: The rule an operation combined with time reversal is held to, as the
#: records' notes say it.
_ANTIUNITARY_RULE = (
    "it holds only where a matrix measured for it and the matrices of the group "
    "returned, up to a one-dimensional character of that group, make a "
    "corepresentation of the magnetic group it generates with that group")


def _carried_onto(group, asked, symprec) -> tuple:
    r"""What each round measured of the operations combined with time reversal, carried onto the group that settled.

    Every round's group holds the group the loop settles on: each step keeps
    a subset of the operations the round before it held, written in the
    origin they were measured in (:func:`_shared_origin`).  So each operation
    of the group returned is paired with the one it is in each round's group
    (:func:`_match_operations`), and each record of ``asked``
    (:func:`_antiunitary_settled`) is handed back with the pairing as
    ``returned``, ``{index there: index in the group returned}``: what the
    one reading of those operations reads them against.

    Raises
    ------
    ..refusal.InternalError
        Where an operation of the group returned is not in a round's group.
        Every step keeps a subset of what it measured, so this is the
        package's own bookkeeping failing, not a measurement.
    """
    unitary = [s for s in group.symmetries if not s.time_reversal]
    out = []
    for rec in asked:
        theirs = [s for s in rec["group"].symmetries if not s.time_reversal]
        pairs, lost, _extra = _match_operations(unitary, theirs, symprec, group.lattice)
        if lost:
            raise InternalError(
                f"the group returned, {group.name} #{group.number_str}, holds "
                f"{len(lost)} operation{'' if len(lost) == 1 else 's'} {lost} that "
                f"{rec['group'].name} #{rec['group'].number_str}, the group of an earlier "
                "round, does not. Every round keeps a subset of the operations the round "
                "before it measured, so this is this package's own bookkeeping failing, "
                "not a measurement about the model.")
        out.append(dict(rec, returned={int(t): int(g) for g, t in pairs}))
    return tuple(out)


def _antiunitary_held(group, matrices, asked, *, tolerances=EXACT) -> dict:
    r"""Which operations of the cell combined with time reversal hold, read once against the group returned.

    ``asked`` is what the rounds measured of the operations combined with time
    reversal, each in the round that asked it last
    (:func:`_antiunitary_settled`), carried onto the group returned
    (:func:`_carried_onto`); ``matrices`` is the representation of the group
    returned that :func:`..symmetry._build_rep` assembled, by ``.ind``, or
    ``None`` where none was built.  It is read where time reversal is broken.

    The rule is one question: whether the matrices measured make a
    representation of the magnetic group ``M`` an operation ``a`` combined
    with time reversal generates with the group returned ``G``.  It has three
    parts:

    * **the group it makes.**  Every operation of ``M`` combined with time
      reversal is kept by the detector, by its own tensor structure and spin
      lift, and by its own square where its unitary part squares to the
      identity, and ``M`` holds no unitary operation outside ``G``.  One whose
      unitary part is in ``G`` makes time reversal itself with it,
      ``T = g⁻¹·(g·T)``, and time reversal is refuted, so it never counts: on
      spinors read in single-valued tables a mirror that holds and that mirror
      with time reversal, squaring to ``+1``, together make time reversal,
      whose square is ``-1``.
    * **its square.**  ``X X̄ = D(a²)`` for its matrix ``X`` and ``D`` the
      matrices of ``G``, with the sign the group's multiplication table gives
      ``a²``: for an ``a`` whose unitary part squares to the identity, the
      sign of ``X X̄``, which no rescaling of ``X`` changes.
    * **how it carries the group returned.**  ``X D(h)* X⁻¹ = D(a h a⁻¹)``
      for each generator ``h`` of ``G``, with the signs the table gives.

    The last two are asked up to a one-dimensional character ``χ`` of ``G``,
    the one freedom the matrices of ``G`` are assembled with: one where the
    phase ``λ(h)`` measured on each generator is ``χ(h) χ(a h a⁻¹)`` and the
    phase of the square is ``χ(a²)`` makes them hold exactly, and these are
    Wigner's conditions for ``X`` and ``D`` to be a representation of ``M``.
    Each phase is read at the margin every phase read off the states is read
    at (:data:`_CLASS_MARGIN`).  Nothing is measured here.

    ``X`` is any member of the operation's measured solution space.  Where
    the space holds one, it is the matrix measured; where it holds more -- a
    Hamiltonian that conserves a spin rotation, whose every member times a
    conserved rotation is a member too -- the question is whether some member
    makes the representation, as time reversal's own square asks whether
    some member has its sign (:func:`_time_reversal_square`).  Once the
    character is fixed the third condition is linear in ``X``, so the members
    that carry ``G`` are read off the space, and the square is asked of each,
    and of the unitary nearest the part of it its transposition keeps with
    the sign the table gives ``a²`` where ``a²`` is the identity, wherever
    that is a member too.

    Each operation is asked on its own, and those counted are the operations
    of one magnetic group combined with time reversal: where they are not
    exactly the ones of one ``M``, no single magnetic group holds them with
    ``G``, and none counts.

    Returns ``{"verdicts", "held"}``: per operation, keyed ``(round, ind)``,
    what decided it -- ``"returned"`` where its unitary part is in the group
    returned, ``"refuted"`` in its own round, ``"closure"`` where the group it
    makes holds a refuted operation, ``"unread"`` where no matrices were built
    to read it with, ``"matrices"`` where they make no representation,
    ``"apart"`` where those that hold are not the operations of one magnetic
    group, or ``"held"`` -- and the ``(round, ind)`` that hold, in order.
    """
    from .symmetry import (_characters_of_subgroup, _identity_position,
                           _minimal_generators, _proportionality, _unitarized)

    # the members of a solution space read off a model symmetric only within
    # its precision: how small a direction of it is none, and how far a unitary
    # read off it may sit outside it
    zero, in_span = tolerances.noise(ZERO), tolerances.noise(IN_SPAN)
    own = group.symmetries
    by_ind = {int(s.ind): s for s in own}
    found = _minimal_generators(range(len(own)), group.get_product_table(get_diff=False),
                                _identity_position(group))
    # the relation asked of every operation, where no small generating set is
    # found, is the one asked of the generators and more
    gens = [int(own[p].ind) for p in (found if found is not None else range(len(own)))]
    characters = None
    verdicts, held, cosets = {}, [], {}
    for r, rec in enumerate(asked):
        cell = rec["group"]
        syms = cell.symmetries
        at = {int(s.ind): p for p, s in enumerate(syms)}
        prod, _tdiff, spf = cell.get_product_table(get_diff=True)
        ident = _identity_position(cell)
        onto = rec["returned"]
        inside = {at[i] for i in onto}
        here = {int(g): at[int(i)] for i, g in onto.items()}
        unitary = {_operation_key(s)[:2]: p for p, s in enumerate(syms)
                   if not s.time_reversal}
        # the spin lift each operation of the group returned carries here: the
        # same rotation, its lift the same SU(2) element or its negative
        lift = {}
        for i, g in onto.items():
            q, sign = at[int(i)], 1
            if bool(cell.spinor) and bool(group.spinor):
                mine = np.asarray(syms[q].spinor_rotation, dtype=complex)
                theirs = np.asarray(by_ind[int(g)].spinor_rotation, dtype=complex)
                sign = (1 if np.allclose(mine, theirs, atol=1e-6) else
                        -1 if np.allclose(mine, -theirs, atol=1e-6) else 0)
                if not sign:
                    raise InternalError(
                        f"the operation {g} of {group.name} #{group.number_str} and its "
                        f"pair {i} in {cell.name} #{cell.number_str} carry spin lifts that "
                        "are not one SU(2) element up to a sign, though they are one "
                        "rotation: this is this package's bookkeeping failing, not a "
                        "measurement about the model.")
            lift[q] = sign

        def _matrix(q):
            """The matrix of the group returned at position ``q`` here, in this group's lift."""
            return lift[q] * np.asarray(matrices[int(onto[int(syms[q].ind)])], dtype=complex)

        def _carried(p, back, X):
            """Whether ``X`` and the group returned's matrices are a representation of the group they make."""
            norm = float(np.linalg.norm(X))
            if not norm > 0.0:
                return False
            X = X * (np.sqrt(X.shape[0]) / norm)
            try:
                inverse = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                return False
            # the square: X conj(X) = D(a^2), with the sign the table gives
            q2 = int(prod[p, p])
            mu, res = _proportionality(X @ X.conj(), int(spf[p, p]) * _matrix(q2))
            if res > _CLASS_MARGIN or abs(mu) <= _CLASS_MARGIN:
                return False
            # how it carries the group returned: X D(h)* X^-1 = D(a h a^-1)
            phases = []
            for g in gens:
                q = here[g]
                pq = int(prod[p, q])
                c = int(prod[pq, back])
                sign = int(spf[p, q]) * int(spf[pq, back]) * int(spf[p, back])
                rho, res = _proportionality(X @ _matrix(q).conj() @ inverse,
                                            sign * _matrix(c))
                if res > _CLASS_MARGIN or abs(rho) <= _CLASS_MARGIN:
                    return False
                phases.append((g, int(onto[int(syms[c].ind)]), rho / abs(rho)))
            square = mu / abs(mu)
            a2 = int(onto[int(syms[q2].ind)])
            # ... up to a one-dimensional character of the group returned
            return any(abs(square - chi[a2]) <= _CLASS_MARGIN
                       and all(abs(lam - chi[h] * chi[c]) <= _CLASS_MARGIN
                               for h, c, lam in phases)
                       for chi in characters)

        def _carrying(p, back, space):
            """The members of a space of more than one that carry the group returned, as a character asks.

            For each character the condition ``X D(h)* = chi(h) chi(c) D(c) X``,
            ``c = a h a^-1`` with the table's sign, on every generator is linear
            in ``X``: its solutions in the space are read off one SVD, offered
            with a fixed combination of them, and each, where ``a²`` is the
            identity, also as the unitary nearest the part of it its
            transposition keeps with the sign the table gives ``a²``, wherever
            that is still a member of the space.
            """
            if space is None or len(space) < 2:
                return []
            space = np.asarray(space, dtype=complex)
            orth, tri = np.linalg.qr(space.reshape(len(space), -1).T)
            span = orth[:, np.abs(np.diag(tri)) > zero * max(1.0, float(np.abs(tri).max()))]
            q2 = int(prod[p, p])
            out = []
            for chi in characters:
                rows = []
                for g in gens:
                    qg = here[g]
                    pq = int(prod[p, qg])
                    c = int(prod[pq, back])
                    sign = int(spf[p, qg]) * int(spf[pq, back]) * int(spf[p, back])
                    lam = sign * chi[g] * chi[int(onto[int(syms[c].ind)])]
                    D, Dc = _matrix(qg).conj(), _matrix(c)
                    rows.append(np.stack([(B @ D - lam * (Dc @ B)).reshape(-1)
                                          for B in space], axis=1))
                _u, sv, vh = np.linalg.svd(np.vstack(rows))
                rank = int(np.sum(sv > zero * max(float(sv[0]), 1.0))) if sv.size else 0
                found = [np.einsum("j,jab->ab", v, space) for v in vh[rank:].conj()]
                if len(found) > 1:
                    weights = np.random.default_rng(17).normal(size=(len(found), 2))
                    found.append(np.einsum("k,kab->ab", weights[:, 0] + 1j * weights[:, 1],
                                           np.asarray(found)))
                for Y in found:
                    out.append(Y)
                    if q2 == ident:
                        U = _unitarized(0.5 * (Y + int(spf[p, p]) * Y.T), tol=zero)
                        if U is not None and float(np.abs(
                                U.reshape(-1) - span @ (span.conj().T @ U.reshape(-1))).max()
                                ) <= in_span:
                            out.append(U)
            return out

        for ind in sorted(rec["ops"]):
            key = (r, int(ind))
            p = at[int(ind)]
            if unitary.get(_operation_key(syms[p])[:2]) in inside:
                verdicts[key] = "returned"
                continue
            if ind in rec["refuted"]:
                verdicts[key] = "refuted"
                continue
            # THE GROUP IT MAKES with the group returned G: G and the coset aG,
            # where a squares into G and carries G onto itself -- otherwise the
            # group they generate holds a unitary operation outside G -- and every
            # operation of aG kept in this round, the last that asked it
            back = next(q for q in range(len(syms)) if int(prod[p, q]) == ident)
            coset = {int(prod[p, q]) for q in inside}
            if (int(prod[p, p]) not in inside
                    or any(int(prod[int(prod[p, here[g]]), back]) not in inside
                           for g in gens)
                    or any(int(syms[q].ind) in rec["refuted"]
                           or int(syms[q].ind) not in rec["ops"] for q in coset)):
                verdicts[key] = "closure"
                continue
            if matrices is None or any(int(g) not in matrices for g in onto.values()):
                verdicts[key] = "unread"
                continue
            if characters is None:
                characters = _characters_of_subgroup(group, sorted(by_ind), roots=True)
            op = rec["ops"][ind]
            if not (_carried(p, back, op["member"])
                    or any(_carried(p, back, X)
                           for X in _carrying(p, back, op.get("space")))):
                verdicts[key] = "matrices"
                continue
            verdicts[key] = "held"
            held.append(key)
            cosets[key] = frozenset(int(syms[q].ind) for q in coset)
    # ... and those that hold are the operations of one magnetic group
    if held and (len({r for r, _i in held}) > 1
                 or {i for _r, i in held} != cosets[held[0]]):
        for key in held:
            verdicts[key] = "apart"
        held = []
    return {"verdicts": verdicts, "held": tuple(held)}


def _beyond_grey(group, matrices, asked, record, *, tolerances=EXACT) -> tuple:
    r"""The operations combined with time reversal that hold while it is broken, read once.

    Where time reversal is measured broken, every operation of the cell
    combined with time reversal that a round measured -- each descent the ones
    whose unitary part it left out, the group time reversal is measured on
    all of its own (:func:`_antiunitary_settled`) -- is read here, once the
    group has settled, against the group returned and the matrices
    :func:`..symmetry._build_rep` assembled for it (``matrices``, ``None``
    where none were built), by the one rule :func:`_antiunitary_held` states.
    Where time reversal is kept nothing is read and nothing is named.

    The time-reversal ``record`` is completed from it: ``n_obeyed`` and
    ``n_broken`` count the operations of the group time reversal was measured
    on by that rule, and the ``note`` says what decided them.  The ones that
    hold are what :attr:`RealizedGroup.beyond_grey` names, each by its index
    in the group it was measured in, with the magnetic type they make.

    ``tolerances`` is the record of what the call declared, which the reading of
    the solution spaces judges noise against.

    Returns ``(record, beyond)``: the record, completed, and the beyond-grey
    record, ``None`` where nothing holds.
    """
    if record is None or record.get("outcome") != "measured broken":
        return record, None
    read = _antiunitary_held(group, matrices, asked, tolerances=tolerances)
    verdicts, held = read["verdicts"], read["held"]
    r_t = next(r for r, rec in enumerate(asked) if rec["reversal"])
    reversal = asked[r_t]
    # the count is of the group time reversal was measured on, and the note
    # says what decided the operations of it the detector kept
    mine = [v for (r, i), v in verdicts.items()
            if r == r_t and i not in reversal["detector"]]
    n_all = int(record["n_antiunitary"])
    n_held = mine.count("held")
    unread = mine.count("unread")
    returned = mine.count("returned")
    failing = len(mine) - n_held - unread
    refuted = n_all - n_held
    # where every one is refuted nothing is counted, and the first count the
    # note does make names the unit, once
    unit = [] if refuted == n_all else [True]

    def named(n):
        if unit:
            return ""
        unit.append(True)
        if n == 1:
            return (", a coset representative of the lattice translations"
                    + (", the single-valued group's, standing for itself and its negative "
                       "in the double group" if group.spinor else "") + ",")
        return (f", {_PER_COSET}" + (f", {_DOUBLE_GROUP}" if group.spinor else "") + ",")

    said = ""
    if failing == refuted:
        said += ((" It has" if failing == 1 else " Every one of them has")
                 + " a symmetry matrix and fails the rule an operation combined "
                 f"with time reversal is held to: {_ANTIUNITARY_RULE}.")
    elif failing:
        said += (f" Of those, {failing}{named(failing)} {'has' if failing == 1 else 'have'} "
                 f"a symmetry matrix and {'fails' if failing == 1 else 'fail'} the rule an "
                 f"operation combined with time reversal is held to: {_ANTIUNITARY_RULE}.")
    if failing and returned == failing:
        said += ((" It combines" if failing == 1 else " Each of them combines")
                 + " time reversal with an operation of the group returned, so it "
                 "generates time reversal with its own unitary part, and time "
                 "reversal is refuted.")
    elif returned:
        said += (f" {returned} of them{named(returned)} "
                 f"combine{'s' if returned == 1 else ''} time "
                 "reversal with an operation of the group returned, so each generates "
                 "time reversal with its own unitary part, and time reversal is "
                 "refuted.")
    if n_held:
        said += (f" The other {n_held} {'holds' if n_held == 1 else 'hold'} by "
                 + ("that rule" if failing else
                    "the rule an operation combined with time reversal is held to: "
                    + _ANTIUNITARY_RULE)
                 + ("; it combines" if n_held == 1 else "; each combines")
                 + " time reversal with an operation outside the group returned, and "
                 f"`beyond_grey` names {'it' if n_held == 1 else 'them'}.")
    if unread:
        said += (f" {unread} more{named(unread)} combine{'s' if unread == 1 else ''} time "
                 "reversal with an operation outside the group returned and "
                 f"{'is' if unread == 1 else 'are'} not read against it: the matrices of "
                 "the group returned to read them with were not built.")
    record = dict(record, n_obeyed=n_held, n_broken=refuted, note=(
        "time reversal measured broken: "
        + (f"every operation of {record['grey_name']} #{record['grey_number_str']} "
           "combined with time reversal is refuted by the Hamiltonian"
           if refuted == n_all else
           f"{refuted} of the {n_all} operation{'' if n_all == 1 else 's'} of "
           f"{record['grey_name']} #{record['grey_number_str']} combined with time "
           f"reversal, {_PER_COSET}"
           + (f" and {_DOUBLE_GROUP}" if group.spinor else "")
           + ", are refuted by the Hamiltonian")
        + ", so the realized group is unitary and the cell was re-identified without "
        "time reversal."
        + said
        + " Labels, EBR rows and the indicator that follow are the unitary group's "
        f"own.{reversal['itself']}"))
    if not held:
        return record, None

    rec = asked[held[0][0]]
    inds = [i for _r, i in held]
    entries = tuple(rec["ops"][i]["entry"] for i in inds)
    kind = _magnetic_type(_CosetView([rec["ops"][i]["sym"] for i in inds]))
    n = len(entries)
    beyond = {
        "operations": entries,
        "n_operations": n,
        "n_measured": sum(int(r["n"]) for r in asked),
        "magnetic_type": kind if kind in ("III", "IV") else "magnetic, beyond grey",
        "closed": bool(rec["closed"]),
        "note": (
            f"{n} operation{'' if n == 1 else 's'} of {rec['group'].name} "
            f"#{rec['group'].number_str} combined with time reversal, "
            + ("a coset representative of the lattice translations" if n == 1
               else _PER_COSET)
            + ((", the single-valued group's, standing for itself and its negative in "
                "the double group" if n == 1 else f", {_DOUBLE_GROUP}")
               if group.spinor else "")
            + f", {'was' if n == 1 else 'were'} measured to hold "
            "while time reversal itself was refuted, read once, against the group "
            "returned, by the rule an operation combined with time reversal is held "
            f"to: {_ANTIUNITARY_RULE}. "
            + ("It combines time reversal with a unitary operation outside the group "
               "returned, which a descent left out, and is named by its index in the group "
               "it was measured in." if n == 1 else
               "All of them combine time reversal with a unitary operation outside the "
               "group returned, which a descent left out, and are named by their index in "
               "the group they were measured in.")
            + " So the "
            "Hamiltonian's magnetic group is larger than the group "
            "returned here. The returned group is the largest one inside this "
            "package's validated universe -- the 230 space groups and their grey "
            "extensions -- and the operations listed above are not represented, "
            "because magnetic groups beyond grey are not validated here. Every "
            "returned label, EBR row and indicator is the unitary group's own and is "
            "therefore in scope; what is out of scope is what those operations would "
            "additionally constrain."),
    }
    return record, beyond


def _magnetic_scope_refusal(group) -> Refusal | None:
    """The package's magnetic scope statement, or ``None`` when in scope.

    One gate, ``"magnetic_type_out_of_scope"``, shared by :func:`_group_of` —
    where it stops a type-III/IV group being built at all — and by
    :func:`..ebr.symmetry_indicator`, which keeps its own check so that a group
    reaching it by any other route is still refused.
    """
    mtype = _magnetic_type(group)
    if mtype in ("I", "II"):
        return None
    number_str = str(getattr(group, "number_str", "?"))
    name = str(getattr(group, "name", "?"))
    au = list(getattr(group, "au_symmetries", ()) or ())
    kind = ("it carries antiunitary operations but no anti-translation and not "
            "time reversal itself" if mtype == "III" else
            "its antiunitary operations include an anti-translation {1|t}.T with t "
            "a fractional translation, and not time reversal itself")
    return Refusal(
        "magnetic_type_out_of_scope",
        f"#{number_str} ({name}) is a magnetic group of type {mtype}: {kind}. This "
        "package is validated over magnetic types I and II only -- the 230 space "
        "groups as unitary groups, and the 230 grey groups with time reversal "
        "adjoined (`include_TR=True`). Beyond them data are missing as well as "
        "conventions: for types III and IV irreptables ships the elementary band "
        "representations and the characters of the unitary part's irreps at the "
        "maximal k-points, but not the corep type at a k-point -- how the "
        "antiunitary operations glue those irreps -- nor the site-symmetry coreps, "
        "and nothing this package reads carries the coordinates of the magnetic "
        "Wyckoff positions; and "
        "every convention this package applies -- the EBR and character transport, "
        "the factor-system conjugation, the time-reversal re-gluing, the corep "
        "naming and the SU(2) lift canonicalization -- is written for types I and "
        "II, and none of it covers g.T pairings or anti-translations. Types I and "
        "II are reachable from the same call: drop "
        "`magmom=` (or hand it zero moments) for the grey group, and a magnetic "
        "arrangement whose symmetry admits no antiunitary operation at all is "
        "type I and is returned.",
        context={"number_str": number_str, "name": name, "magnetic_type": mtype,
                 "n_antiunitary": len(au), "in_scope": ("I", "II"),
                 "record": "validated scope: magnetic types I and II"})


# --------------------------------------------------------------------------
# the lattice: the Hamiltonian sets the primitive cell
# --------------------------------------------------------------------------

def _pure_translations(symmetries, symprec, lattice) -> tuple:
    r"""The non-zero pure translations among the unitary operations, reduced.

    A ``{1|τ}`` with ``τ`` fractional says the *point set* repeats inside the
    cell handed in — the cell is a supercell of the crystal's own primitive
    cell, and the operations that come with it are the whole lattice
    supergroup.  Read modulo the lattice, which is the only sense in which a
    space-group translation is defined, and sorted so that they form a
    reproducible key.  A translation is zero when it is no longer than the
    finder's distance ``symprec``, read as the length it is on ``lattice``
    (rows the cell's vectors), in the units the finder's distance is in.
    """
    ident = np.eye(3, dtype=int)
    lat = np.asarray(lattice, dtype=float)
    out = []
    for s in symmetries:
        if bool(s.time_reversal):
            continue
        if not np.array_equal(np.rint(np.asarray(s.rotation, dtype=float)).astype(int),
                              ident):
            continue
        t = np.asarray(s.translation, dtype=float).reshape(-1)
        t = t - np.rint(t)
        if float(np.linalg.norm(t @ lat)) > float(symprec):
            out.append(tuple(float(x) for x in t))
    return tuple(sorted(out))


def _hermite_basis(rows) -> np.ndarray:
    """A basis of the integer lattice spanned by ``rows`` (row Hermite form).

    Plain integer linear algebra — the row-style Hermite normal form of an
    ``m x 3`` integer matrix of full column rank — used only to *report* the
    finer lattice a supercell's fractional translations generate.  No group
    data enters it.
    """
    M = [[int(x) for x in r] for r in np.asarray(rows, dtype=np.int64)]
    basis: list[list[int]] = []
    for col in range(3):
        while sum(1 for r in M if r[col] != 0) > 1:
            M.sort(key=lambda r: (r[col] == 0, abs(r[col])))
            head = M[0]
            for i in range(1, len(M)):
                if M[i][col] != 0:
                    q = M[i][col] // head[col]
                    M[i] = [a - q * b for a, b in zip(M[i], head)]
            M = [r for r in M if any(r)]
        pivot = [r for r in M if r[col] != 0]
        if not pivot:                               # pragma: no cover - defensive
            raise InternalError(
                "the generators of this cell's finer lattice do not span a rank-3 "
                "lattice, which cannot happen for a set containing the cell's own "
                "lattice vectors. This is arithmetic in this package failing.")
        head = pivot[0] if pivot[0][col] > 0 else [-v for v in pivot[0]]
        basis.append(head)
        M = [r for r in M if r[col] == 0]
    return np.array(basis, dtype=np.int64)


def _primitive_lattice(lattice, translations, *, max_den=64) -> dict:
    r"""The finer lattice the cell's pure translations generate.

    ``Z³ + span_Z(τ)`` in the model's reduced coordinates, and the Cartesian
    vectors that go with it.  The translations are exact rationals for a typed
    cell (``½``, ``⅓``, …); they are snapped to the nearest rational with
    denominator at most ``max_den`` and the whole generating set is scaled by
    the common denominator, so the reduction is integer arithmetic throughout.
    Returns ``{"reduced", "cartesian", "index", "denominator"}``.
    """
    from fractions import Fraction

    den = 1
    for tau in translations:
        for x in tau:
            d = Fraction(float(x)).limit_denominator(int(max_den)).denominator
            den = den * d // math.gcd(den, d)
    rows = [[den if i == j else 0 for j in range(3)] for i in range(3)]
    for tau in translations:
        rows.append([int(round(den * float(x))) for x in tau])
    basis = _hermite_basis(rows).astype(float) / float(den)
    lattice = np.asarray(lattice, dtype=float)
    return {"reduced": tuple(tuple(float(x) for x in row) for row in basis),
            "cartesian": tuple(tuple(float(x) for x in row)
                               for row in basis @ lattice),
            "index": int(round(1.0 / abs(float(np.linalg.det(basis))))),
            "denominator": int(den)}


def _lattice_not_descended(group) -> Refusal | None:
    r"""The gate every table consumer applies to a lattice supergroup.

    A record with ``GroupData._lattice_index`` above ``1`` is the group of a
    **supercell**: its operations are honest — spglib found every one of them on
    the model's own lattice — but its *tables* are the crystal's, written on the
    finer lattice, and reading them here would label a folded Brillouin zone.
    ``None`` when the record is primitive, which is every record a table
    consumer normally sees.

    A :class:`~.refusal.Refusal` handed in where a group belongs is returned
    unchanged: the call that was meant to build the group refused, and that
    verdict is the one to read, so every entry point applying this gate first
    propagates it instead of reaching for a field a refusal does not carry.
    """
    if isinstance(group, Refusal):
        return group
    index = int(getattr(group, "_lattice_index", 1) or 1)
    if index <= 1:
        return None
    taus = tuple(getattr(group, "_fractional_translations", ()) or ())
    tau = np.round(np.asarray(taus[0], dtype=float), 6) if taus else np.zeros(3)
    return Refusal(
        "lattice_not_descended",
        f"this cell's point set allows the pure translation {tau.tolist()} (reduced), "
        f"so what {group.name} #{group.number_str} describes is a lattice supergroup: "
        f"the model's lattice is an index-{index} sublattice of the crystal's own, and "
        "the group's tables -- k-points, coreps, EBR rows, indicators -- live on the "
        "finer lattice and would label a folded zone here. Every label read off them "
        "would be a label of the wrong Brillouin zone. `realized_group(model)` descends "
        "to the model's own lattice by measuring the Hamiltonian: a Hamiltonian that "
        "obeys the translation is a supercell and is refused (gate='supercell'), and "
        "one that breaks it is returned the group it realizes. Alternatively reduce the "
        "cell to its primitive one, or pass `typat=` to say that sites the default "
        "treats as one species are chemically distinct.",
        value=float(index), threshold=1.0,
        context={"number_str": str(group.number_str), "name": str(group.name),
                 "lattice_index": index, "translations": taus, "group": group})


# --------------------------------------------------------------------------
# the operations themselves: do they close, and what to do when they do not
# --------------------------------------------------------------------------

def _translation_closure(symmetries) -> float:
    r"""``max_{g,h} |t_g + A_g t_h − t_{gh}|`` modulo the lattice.

    **The question this answers is whether the identified operations are a
    group at all**, and it is not rhetorical for a numerical cell.  spglib
    derives each operation's translation from the atom pairs it maps onto each
    other, so on positions carrying a Wannierization's noise every operation
    picks that noise up *separately* — measured: one Kane-Mele centre displaced
    by (+1e-8, −1e-8) gives translations of 0, 1e-8 and 2e-8 where the exact
    cell gives 0, and this residual reads **2.0e-8** against exactly ``0.0``
    for the exact cell.

    Nothing downstream survives that quietly.  With operations that do not
    close there is **no exactly symmetric configuration of positions** they
    describe, and :func:`..symmetry.bloch_operator`'s integer-shift identity,
    the product table's translation differences and the sewing phases at
    nonsymmorphic points are all asserted against operations that are not
    consistent with one another.  The centres themselves are moved onto an
    exactly symmetric set without reading these translations at all
    (:func:`_symmetric_sites`).

    The product ``gh`` is found by its **rotation** (an exact integer matrix)
    and its antiunitarity, never by comparing translations; the residual is
    taken modulo the lattice, which is the only sense in which a space-group
    translation is defined.

    **A lattice supergroup carries several operations per rotation**, and the
    residual is then the distance to the *nearest* of them: on a cell that is a
    supercell of the crystal's own primitive cell the operations come in
    ``_lattice_index`` cosets of the pure translations
    (``GroupData._lattice_index``), every member of a coset has the same
    rotation, and "which one is ``gh``" is settled by the translation.  Picking
    one representative per rotation instead — which is what a rotation-keyed
    lookup does — reads a perfectly exact group as not closing at all: measured
    **0.5** for the ``2x1x1`` simple-cubic supercell's 64 operations, and
    exactly ``0.0`` here.
    """
    n = len(symmetries)
    R = np.array([np.asarray(s.rotation, dtype=np.int64) for s in symmetries])
    t = np.array([np.asarray(s.translation, dtype=float) for s in symmetries])
    tr = np.array([bool(s.time_reversal) for s in symmetries])

    # the distinct rotations (48 even for a 27-fold supercell of a cubic cell),
    # so the product's rotation is found in a table of that size rather than in
    # an n x n one -- what keeps a supergroup's closure affordable at all
    rot_of, rots = {}, []
    rot_id = np.empty(n, dtype=int)
    for i in range(n):
        key = R[i].tobytes()
        if key not in rot_of:
            rot_of[key] = len(rots)
            rots.append(R[i])
        rot_id[i] = rot_of[key]
    nrot = len(rots)
    prod_rot = np.empty((nrot, nrot), dtype=int)
    for a in range(nrot):
        for b in range(nrot):
            key = (rots[a] @ rots[b]).tobytes()
            if key not in rot_of:                   # pragma: no cover - defensive
                return float("inf")                 # not closed even as rotations
            prod_rot[a, b] = rot_of[key]

    # every operation with a given (rotation, antiunitarity), not only one: the
    # candidates the product may be, one per coset of the pure translations
    members: dict = {}
    for i in range(n):
        members.setdefault((int(rot_id[i]), bool(tr[i])), []).append(i)
    width = max(len(v) for v in members.values())
    have = np.zeros((nrot, 2), dtype=bool)
    cand = np.zeros((nrot, 2, width), dtype=int)
    for (a, b), idx in members.items():
        have[a, int(b)] = True
        cand[a, int(b), :] = idx + [idx[-1]] * (width - len(idx))

    worst = 0.0
    for g in range(n):
        c = prod_rot[rot_id[g], rot_id]                    # (n,)
        b = (tr[g] != tr).astype(int)                      # (n,)
        if not bool(have[c, b].all()):              # pragma: no cover - defensive
            return float("inf")                     # a product with no partner at all
        j = cand[c, b]                                     # (n, width)
        target = t[g] + t @ R[g].astype(float).T           # (n, 3)
        d = target[:, None, :] - t[j]                      # (n, width, 3)
        d -= np.rint(d)
        worst = max(worst, float(np.abs(d).max(axis=2).min(axis=1).max()))
    return float(worst)


def _exact_operations(sg, symprec, tables_ready, lattice) -> tuple | Refusal:
    r"""Make the identified operations close **exactly** — and only if they do not.

    Returns ``(closure_measured, closure_repaired)``, both reduced, on the cell
    whose vectors are the rows of ``lattice``.  When the closure measures
    exactly ``0.0`` — every cell built from exact numbers, which is every model
    whose positions were typed rather than computed — this returns
    ``(0.0, 0.0)`` and **changes nothing at all, bit for bit**.  The same is true
    at machine precision where there is nothing to repair from: a cell ``irrep``
    could not match whose closure is at or below :data:`_EXACT_CLOSURE` is
    arithmetic, not noise, and is left alone with the measured number recorded.

    Otherwise each translation is rebuilt from the **tabulated** reference-cell
    operation through the one origin shift.  ``irrep``'s own map is
    ``t_ref = refUC⁻¹·(t + R·shiftUC − shiftUC)``
    (``SymmetryOperation.translation_refUC``), so its inverse is

    .. math:: t = \mathrm{refUC}\cdot t_\text{ref} + s - R\,s,
              \qquad s = \mathrm{shiftUC},

    with ``t_ref`` read off ``irreptables`` and snapped to the
    small-denominator rational it is, since the tables print it to five
    decimals (``0.33333``), and the operation matched by its **reference-cell
    rotation**
    ``refUC⁻¹ R refUC`` (an integer matrix) together with its antiunitarity,
    never by a float translation.  spglib's own integer part is kept:
    ``t ← t + round(t_spglib − t)``.

    **Why this is exact whatever spglib's origin refinement did.**  The
    tabulated operations close exactly, and conjugating a group by a *single*
    vector ``s`` — which is all ``shiftUC`` is — cannot break that: the closure
    of the shifted set carries the extra term ``(R_{gh} − R_g R_h)·s``, and
    ``R_{gh} = R_g R_h`` exactly, integer matrix by integer matrix.  So a
    ``shiftUC`` sitting at a general point, as a wannier90 cell's does
    ((−0.3333333233, −0.3333333433, 0) on the displaced Kane-Mele cell), costs
    nothing and nothing has to be rationalized.  On that cell the closure goes
    from 2.0e-8 to **0.0**, with every translation moved by at most 2.0e-8.

    The rotations are never touched, and a rebuilt translation further than
    ``symprec`` from spglib's, read as the length it is in the model's units as
    the finder's distance is, is **refused** rather than made — at that distance
    the tabulated operation is not the one this cell has, and what would come
    back is a different cell rather than a repaired one.

    A tabulated set that cannot be matched to this cell operation by operation,
    or a rebuilt set that still does not close, is a defect in this package's own
    use of the tables rather than anything measured about the model, and raises
    :class:`..refusal.InternalError`.
    """
    before = _translation_closure(sg.symmetries)
    if before == 0.0:
        return 0.0, 0.0
    if not tables_ready:
        if before <= _EXACT_CLOSURE:
            # binary floating point, not a defect in the cell: a supercell whose
            # translations are exact thirds cannot close to a bit-exact zero, and
            # there is nothing here to repair.  See `_EXACT_CLOSURE`.
            return float(before), 0.0
        return Refusal(
            "operations_not_consistent",
            f"the operations spglib found for this cell do not close: "
            f"max |t_g + A_g t_h - t_gh| = {before:.3g} (reduced, modulo the lattice) "
            "against 0.0 for a cell built from exact numbers. spglib derives each "
            "operation's translation from the atom pairs it maps, so numerically "
            "determined positions give every operation its own share of the noise; "
            "with operations that do not close there is no exactly symmetric "
            "configuration at all, and every operation's verdict and every label after "
            "it would be measured against a set that is not a group. The repair is to "
            "rebuild the translations from the tabulated reference-cell operations -- and "
            "`irrep` could not match this cell to the BCS standard setting, so there "
            "is no tabulated operation to rebuild from. Hand a cell whose positions "
            "carry less noise, or one irrep can match.",
            value=float(before), threshold=0.0,
            context={"number_str": str(sg.number_str), "closure": float(before)})

    from irreptables.irreps import IrrepTable

    table = IrrepTable(str(sg.number_str), bool(sg.spinor), magnetic=bool(sg.magnetic))
    tsyms = list(table.symmetries)
    refUC = np.asarray(sg.refUC, dtype=float)
    shiftUC = np.asarray(sg.shiftUC, dtype=float)
    inv = np.linalg.inv(refUC)
    # the tables print a translation to five decimals (0.33333 for a third), so
    # each is snapped to the small-denominator rational it is, as
    # `_reference_operations` snaps the same numbers
    tabulated = {(np.rint(np.asarray(tb.R, dtype=float)).astype(int).tobytes(),
                  bool(tb.time_reversal)): _snap_rational(np.asarray(tb.t, dtype=float))
                 for tb in tsyms}
    if len(tabulated) != len(tsyms) or len(tsyms) != len(sg.symmetries):
        raise InternalError(                         # pragma: no cover - guard
            f"group #{sg.number_str}: the cell carries {len(sg.symmetries)} operations "
            f"against the table's {len(tsyms)} ({len(tabulated)} distinct rotations), so "
            "there is no operation-by-operation matching to rebuild the translations "
            "from. This is a statement about this package's use of irreptables, not "
            "about the model.")

    lat = np.asarray(lattice, dtype=float)
    rebuilt, defect, moved = [], 0.0, 0.0
    for s in sg.symmetries:
        R = np.asarray(s.rotation, dtype=float)
        Rref = inv @ R @ refUC
        if float(np.abs(Rref - np.rint(Rref)).max()) > 1e-6:
            raise InternalError(                     # pragma: no cover - guard
                f"group #{sg.number_str}: the rotation refUC^-1 R refUC is not integer "
                f"(defect {float(np.abs(Rref - np.rint(Rref)).max()):.3g}), so the "
                "operation cannot be matched to a tabulated one. irrep's own "
                "`rotation_refUC` asserts the same thing.")
        key = (np.rint(Rref).astype(int).tobytes(), bool(s.time_reversal))
        if key not in tabulated:                       # pragma: no cover - guard
            raise InternalError(
                f"group #{sg.number_str}: operation #{s.ind}'s reference-cell rotation "
                f"is not among the tabulated ones, so the shipped symmetry list and "
                "the identified one are not the same group.")
        t_map = refUC @ tabulated[key] + shiftUC - R @ shiftUC
        t_old = np.asarray(s.translation, dtype=float)
        t_new = t_map + np.rint(t_old - t_map)
        defect = max(defect, float(np.abs(t_new - t_old).max()))
        # the same move as a length, in the units the finder's distance is in
        moved = max(moved, float(np.linalg.norm((t_new - t_old) @ lat)))
        rebuilt.append(t_new)

    if moved > float(symprec):
        return Refusal(
            "operations_not_nearly_consistent",
            f"the operations spglib found for this cell do not close "
            f"(max |t_g + A_g t_h - t_gh| = {before:.3g}, reduced), and rebuilding "
            f"their translations from the tabulated ones would move one by "
            f"{moved:.3g} in the model's length units -- further than the declared "
            f"symprec = {float(symprec):.3g}. "
            "A translation that far from spglib's is not the same operation: the "
            "tabulated group is not the group these positions carry at this "
            "tolerance, so repairing them would substitute a crystal the model does "
            "not have. Identify again at a symprec the positions can support, or with "
            "a cell (`typat=`, `decorations=`) that expresses what they do.",
            value=float(moved), threshold=float(symprec),
            context={"number_str": str(sg.number_str), "closure": float(before)})

    for s, t_new in zip(sg.symmetries, rebuilt):
        s.translation = t_new                          # the rotations are untouched
    after = _translation_closure(sg.symmetries)
    if after > 1e-12:
        raise InternalError(                         # pragma: no cover - guard
            f"group #{sg.number_str}: the rebuilt operations still do not close "
            f"(max |t_g + A_g t_h - t_gh| = {after:.3g} against 1e-12), though every "
            "translation came from a tabulated one through a single origin shift, "
            "which cannot break a closure. This is arithmetic in this package "
            "failing, not a measurement about the model.")
    return float(before), float(defect)


#: The identification memo: a bounded, process-level LRU of :func:`_group_of`
#: records, keyed by the cell it was asked about.  An identification is spglib
#: plus ``irrep``'s setting search plus the spinor-branch fit plus the table
#: load — 5.4 s of the 9.2 s an unmemoized Kane-Mele ``realized_group``
#: spends — and it is paid once for the cell, again for every descent step's
#: re-identification and again for the other environment, on cells
#: that are frequently the same one.  A hit never changes what is returned: what
#: it hands back is the object the first call built.
_IDENT_MEMO: "OrderedDict[tuple, GroupData]" = OrderedDict()

#: How many identifications the memo keeps.  Modest on purpose: it is a memo for
#: a pipeline and a suite, not a database, and a ``GroupData`` holds ``irrep``'s
#: whole ``SpaceGroupIrreps`` object.
_IDENT_MEMO_MAX = 64

_IDENT_MEMO_STATS = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}

#: The vacuum a layer's cell is given before the symmetry finder reads it,
#: in the model's length units (:func:`_pad_cell`): geometry, read by no
#: judgement.
_PAD_LENGTH = 20.0

#: The denominator every translation of a crystal's operations is a multiple of,
#: in the model's own reduced coordinates, about an origin the tables can use:
#: halves, thirds, quarters, sixths and eighths of a lattice vector, which 24
#: holds.  A cell that repeats inside itself adds its pure translations' own.
#: Exact arithmetic, read by no judgement of noise.
_TRANSLATION_DENOMINATOR = 24


def _identification_memo_stats() -> dict:
    """The memo's counters and its size — a copy, so nothing here is reachable."""
    return dict(_IDENT_MEMO_STATS, size=len(_IDENT_MEMO), maxsize=int(_IDENT_MEMO_MAX))


def _identification_memo_clear() -> None:
    """Empty the memo and zero its counters."""
    _IDENT_MEMO.clear()
    for key in _IDENT_MEMO_STATS:
        _IDENT_MEMO_STATS[key] = 0


#: The seeds ``numpy``'s global random state is set to for ``irrep``'s search of
#: the tables' setting, first to last.  The search fits the spin basis the
#: tables' SU(2) matrices are written in by a minimization from a start drawn
#: from that state (``irrep.spacegroup_irreps.match_spinor_rotations``), and the
#: fit can stop short of the tables from an unlucky start.  A fixed first seed
#: makes the identification the same in every process; the others are asked only
#: where the search before them failed after drawing from that state.
_SETTING_SEEDS = (0, 1, 2, 3)


def _setting_search(kwargs) -> tuple:
    r"""``irrep``'s identification with the tables' setting, seeded and asked again.

    ``SpaceGroupIrreps.from_cell(search_cell=True)`` is asked with ``numpy``'s
    global random state at each of :data:`_SETTING_SEEDS` in turn, until one
    search matches the cell to the tables' standard setting.  A search that
    failed without drawing from that state -- no setting of the cell's
    operations matches the tables' at all, which is arithmetic and no start
    point -- fails the same way from every seed, and is not asked again.  The
    state the call arrived with is restored afterwards, whatever happened, so
    that nothing outside this identification sees a draw it made.  ``kwargs``
    are ``from_cell``'s own.

    Returns ``(sg, None, n)`` from the first search that succeeds, or
    ``(None, error, n)`` with the ``RuntimeError`` the last one raised; ``n``
    is how many searches ran.
    """
    from irrep.spacegroup_irreps import SpaceGroupIrreps

    state = np.random.get_state()
    error, asked = None, 0
    try:
        for seed in _SETTING_SEEDS:
            np.random.seed(int(seed))
            fresh = np.random.get_state()
            asked += 1
            try:
                return SpaceGroupIrreps.from_cell(search_cell=True, **kwargs), None, asked
            except RuntimeError as exc:
                error = exc
            now = np.random.get_state()
            if now[2] == fresh[2] and np.array_equal(now[1], fresh[1]):
                break                    # nothing was drawn: no seed changes this
    finally:
        np.random.set_state(state)
    return None, error, asked


def _origin_nearest(sg, lattice, vacuum_dirs, symprec):
    r"""The tables' origin nearest the model's own, where ``irrep`` read another.

    The tables' origin sits at ``shiftUC`` in the cell's reduced coordinates,
    and the same tables are read about every point ``shiftUC + v`` with
    ``(R − 1)·v`` a lattice vector for each rotation ``R`` of the group: every
    operation keeps its translation modulo the lattice there, while the
    Wyckoff letter a site is named with and the irrep a k-point's states are
    named with can change.  The point kept is the one nearest the model's own
    origin, by Cartesian distance modulo the lattice and across the periodic
    directions alone -- a layer's vacuum direction is the slab presentation's,
    not the model's.  Along a direction every rotation fixes, the origin is
    free and no name depends on it, so the distance is not measured along it.
    Points nearer by less than the finder's distance ``symprec`` are equally
    near, and among them ``irrep``'s own choice is kept; where it is not among
    the nearest, the first of them on the grid of :data:`_TRANSLATION_DENOMINATOR`
    is.

    Returns the ``shiftUC`` to ask ``irrep`` for, or ``None`` where its own is
    the one the rule keeps.
    """
    per = [d for d in range(3) if d not in tuple(vacuum_dirs)]
    if not per:                                     # pragma: no cover - a cell has a lattice
        return None
    lattice = np.asarray(lattice, dtype=float)
    shift_uc = np.asarray(sg.shiftUC, dtype=float).reshape(-1)
    ident = np.eye(3)
    rots = {}
    for s in sg.symmetries:
        R = np.rint(np.asarray(s.rotation, dtype=float))
        rots.setdefault(R.tobytes(), R)
    den = _TRANSLATION_DENOMINATOR
    axes = np.meshgrid(*[np.arange(den) / den for _ in per], indexing="ij")
    shifts = np.zeros((den ** len(per), 3))
    for k, d in enumerate(per):
        shifts[:, d] = axes[k].ravel()
    allowed = np.ones(len(shifts), dtype=bool)
    for R in rots.values():
        image = shifts @ (R - ident).T
        allowed &= np.abs(image - np.rint(image)).max(axis=1) <= _EXACT_CLOSURE
    shifts = shifts[allowed]
    # the directions every rotation fixes, in Cartesian components, orthonormal
    stack = np.vstack([(R - ident)[:, per] for R in rots.values()])
    _u, sing, vt = np.linalg.svd(stack)
    rank = int(np.count_nonzero(sing > _EXACT_CLOSURE * max(float(sing.max(initial=0.0)), 1.0)))
    fixed = vt[rank:] @ lattice[per]                 # rows: fixed directions, Cartesian
    basis = np.linalg.qr(fixed.T)[0] if len(fixed) else np.zeros((3, 0))
    near = np.array(np.meshgrid(*[(-1, 0, 1)] * len(per), indexing="ij")).reshape(
        len(per), -1).T

    def distance(v):
        o = (shift_uc + v)[per]
        o = o - np.rint(o)
        cart = (o[None, :] + near) @ lattice[per]
        cart = cart - (cart @ basis) @ basis.T
        return float(np.linalg.norm(cart, axis=1).min())

    dist = np.array([distance(v) for v in shifts])
    nearest = float(dist.min(initial=np.inf))
    if distance(np.zeros(3)) <= nearest + float(symprec):
        return None
    pick = int(np.flatnonzero(dist <= nearest + float(symprec))[0])
    return shift_uc + shifts[pick]


def _identification_key(lattice, site_pos, site_orbs, species, vacuum_dirs,
                        vacuum_length, symprec, spinor, include_TR, magmom,
                        periodic_dirs, dim_k, dim_r) -> tuple:
    """Everything :func:`_group_of` reads before it asks ``irrep``, as one key.

    Exact bytes, not rounded: two cells built by the same code produce
    bit-identical arrays, and a near-miss is a *different* cell that the finder
    is entitled to answer differently at the same ``symprec``.
    """
    return (np.ascontiguousarray(np.asarray(lattice, dtype=float)).tobytes(),
            np.ascontiguousarray(np.asarray(site_pos, dtype=float)).tobytes(),
            tuple(tuple(int(o) for o in orbs) for orbs in site_orbs),
            np.ascontiguousarray(np.asarray(species)).tobytes(),
            tuple(int(d) for d in vacuum_dirs), float(vacuum_length), float(symprec),
            bool(spinor), bool(include_TR),
            None if magmom is None else np.ascontiguousarray(
                np.asarray(magmom, dtype=float)).tobytes(),
            tuple(int(d) for d in periodic_dirs), int(dim_k), int(dim_r))


def _group_of(model, *, spinor=None, include_TR=True, symprec=DEFAULT_FINDER, typat=None,
              magmom=None, vacuum_length=_PAD_LENGTH) -> GroupData | Refusal:
    """Internal to :func:`realized_group`: the space group the cell allows.

    Read from the lattice and the orbital positions alone, so what comes back
    is an upper bound on the group a Hamiltonian on this cell can obey, stamped
    :attr:`GroupData.provenance` ``= "cell"``.  ``typat`` labels orbital
    classes, ``include_TR=True`` asks the cell as a grey group — a point set
    says nothing about a Hamiltonian's time reversal — ``spinor`` follows
    ``model.nspin``, and ``symprec`` and ``vacuum_length`` are the finder's
    tolerance, a length in the model's units at which every translation here
    is compared, and the slab presentation's vacuum, which no identification
    may depend on.  The cell is periodic in two or three directions:
    :func:`realized_group` refuses a model periodic in fewer before any cell is
    identified, and a layer's cell is built on its slab presentation, whose
    non-periodic vector lies along the layer's normal
    (:func:`_slab_presentation`).

    Refusals
    --------
    ``irrep_tables``
        ``irreptables`` is not installed, so every label downstream would come
        back ``"None"``.
    ``single_valued_spinful``
        ``spinor=False`` on a model whose states carry the electron's spin-1/2.
    ``magnetic_type_out_of_scope``
        The operations found are of magnetic type III or IV, outside the 230
        space groups and their grey extensions.  ``magmom`` is what reaches it.
    ``operations_not_nearly_consistent`` / ``operations_not_consistent``
        The operations do not close and could not be rebuilt from the tabulated
        ones — a **numerical** cell only (``GroupData._closure_measured``),
        which :func:`realized_group` hands here where no precision is declared
        for its centres, or where they cannot be read as a symmetric set at all.

    Notes
    -----
    **Under a declared precision the cell is exact.**  :func:`realized_group`
    reads the operations off the centres as they are, moves the centres onto
    the nearest set those operations carry exactly and removes a shift of every
    centre together that is noise, all before it asks here
    (:func:`_exact_centres`): the setting, the tables' origin and the names are
    read on exact centres, the operations close, and their repair from the
    tabulated ones (:func:`_exact_operations`) is not reached.  The repair and
    its two refusals are reached by a numerical cell with no precision
    declared, and by one whose centres cannot be read as a symmetric set at
    all -- two sites of one species closer together than the symmetry finder
    can tell apart, so that an operation does not carry the sites onto one
    another one for one -- which is identified as it is.

    **The tables' setting is searched deterministically** (:func:`_setting_search`):
    from a fixed random seed, so that a cell is identified the same way in every
    process, and again from further fixed seeds where a search that drew from
    it fails.  Where every one fails the group is returned without tables,
    ``GroupData._setup_note`` carrying ``irrep``'s own message, and every row
    of an atomic-limit table built on it says so.

    **The tables' origin is the one nearest the model's own.**  The tables are
    read the same way about every point the operations' translations allow as
    an origin, and the Wyckoff letters and k-point irreps they give can differ
    between them; the one read is the nearest to the model's own origin
    (:func:`_origin_nearest`), so that an orbital the model writes at the origin
    is named at the tables' origin wherever that point can be one.  Equally
    near points are a tie, and ``irrep``'s own choice on the cell is kept.  On
    every cell built from exact numbers measured, ``irrep``'s own choice is
    that nearest point already.

    **The Hamiltonian sets the primitive cell.**  A cell whose orbitals repeat
    under a fractional translation ``τ`` gets the whole lattice supergroup —
    ``{1|τ}`` and its coset partners included, ``_tables_ready = False``, every
    table consumer refusing (``gate="lattice_not_descended"``) because the
    tables are the crystal's and would label a folded zone.  Only
    :func:`realized_group` can say whether the model *is* a supercell: it
    measures ``τ``, refuses ``supercell`` where the Hamiltonian obeys it, and
    descends where it does not.
    """
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)

    from irrep.spacegroup_irreps import SpaceGroupIrreps

    if spinor is None:
        spinor = int(model.nspin) == 2

    # `nspin == 2` invokes the double group at every level, and the inference
    # above is the only route that ever sets `spinor` when it is not asked for
    # -- so this fires exactly on an explicit `spinor=False` for a model
    # whose states carry the electron's own spin-1/2.  There is no single-valued
    # question to put to such a model, and returning it the single-valued tables
    # anyway would let a build reach `symmetry._table_anchored` carrying a
    # question the tables cannot answer.  Enforced here as well as in
    # `_build_rep`, so the invalid pair cannot be constructed in the first place.
    if int(model.nspin) == 2 and not bool(spinor):
        return Refusal(
            "single_valued_spinful",
            "spinor=False was asked for a spinful model (nspin=2). A spinful model's "
            "states carry the electron's own spin-1/2, its little-group content is "
            "double-valued, and there is no single-valued question to put to it. Use "
            "spinor=True, or leave the default `spinor=None`, which infers it from "
            "`nspin`. (The converse, `spinor=True` for a spinless model, is answered: "
            "group data is a property of the crystal and of the valuedness asked for, "
            "and its representation is refused `rep_double_valued_spinless`.)",
            context={"nspin": int(model.nspin), "spinor": bool(spinor),
                     "spinful": bool(getattr(model, "spinful", int(model.nspin) == 2))})

    lattice, positions, vacuum_dirs = _pad_cell(model, float(vacuum_length))
    site_pos, site_orbs, species = _sites_and_typat(positions, typat, symprec, lattice)

    # THE IDENTIFICATION MEMO.  Everything below this point -- spglib, irrep's
    # setting search, the spinor-branch fit, the closure repair and the table
    # load -- is a pure function of the cell derived above and of the four flags,
    # and the pipeline asks about the same cell repeatedly: once for it, again
    # after a descent re-identifies a subgroup that turns out to be the cell
    # again, and again for the other environment.  A hit returns the
    # object the first call built, memos and all, which is also what makes the
    # product table (`GroupData.get_product_table`) a once-per-cell cost.
    memo_key = _identification_key(
        lattice, site_pos, site_orbs, species, vacuum_dirs, vacuum_length, symprec,
        spinor, include_TR, magmom, model.periodic_dirs,
        model.lattice.dim_k, model.lattice.dim_r)
    memoized = _IDENT_MEMO.get(memo_key)
    if memoized is not None:
        _IDENT_MEMO.move_to_end(memo_key)
        _IDENT_MEMO_STATS["hits"] += 1
        return memoized
    _IDENT_MEMO_STATS["misses"] += 1

    kwargs = dict(real_lattice=lattice, positions=site_pos, typat=species,
                  spinor=bool(spinor), include_TR=bool(include_TR),
                  symprec=float(symprec), magmom=magmom)
    note = None
    # THE TABLES' SETTING, searched from fixed seeds and asked again where a
    # search fails: irrep's spin-basis fit starts from a random point, and a
    # start it cannot converge from is no statement about the cell.  Only where
    # every seed fails is the group returned without tables, and then its note
    # carries irrep's own message and every row built on it says so.
    sg, failure, asked = _setting_search(kwargs)
    tables_ready = sg is not None
    if sg is None:
        sg = SpaceGroupIrreps.from_cell(search_cell=False, **kwargs)
        note = ("irrep could not match this cell to the BCS standard setting"
                + (f" in {asked} searches from fixed seeds" if asked > 1 else "")
                + f", so irrep tables are unavailable for it: {failure}")

    # THE LATTICE.  A pure fractional translation says the point set repeats
    # inside the cell handed in: the model's lattice is a sublattice of the
    # crystal's own, and what this cell allows is the whole LATTICE SUPERGROUP.
    # It is returned -- every operation of it, coset partners and translations
    # included -- but never with tables: those are the crystal's, written on the
    # finer lattice, and reading them here would label a folded zone.  The
    # `search_cell` branch is forced off for the same reason, so that no record
    # carrying a fractional translation ever claims a refUC/shiftUC.
    translations = _pure_translations(sg.symmetries, symprec, lattice)
    if translations:
        if tables_ready:
            sg = SpaceGroupIrreps.from_cell(search_cell=False, **kwargs)
            translations = _pure_translations(sg.symmetries, symprec, lattice)
            tables_ready = False
        note = (
            f"this cell is a supercell: its point set allows the pure translation "
            f"{np.round(np.asarray(translations[0], dtype=float), 6).tolist()} "
            f"(reduced), so the operations here are the lattice supergroup of index "
            f"{len(translations) + 1} and the tables live on the finer lattice.")

    # THE TABLES' ORIGIN.  The same tables are read about any point the
    # operations' translations allow, and the names of sites and of k-points can
    # differ between them; the one nearest the model's own origin is the one
    # read, so that an orbital the model writes at the origin is named at the
    # tables' origin wherever that point can be one.  irrep's own choice stands
    # where it is that point, or ties with it.
    if tables_ready:
        nearer = _origin_nearest(sg, lattice, vacuum_dirs, symprec)
        if nearer is not None:
            again, _failure, _asked = _setting_search(
                dict(kwargs, refUC=np.asarray(sg.refUC, dtype=float).T, shiftUC=nearer))
            if again is not None:
                sg = again

    # irrep fits the spin-basis transformation from an unseeded random start and
    # both branches of that fit are exact minima; replace it with the frame the
    # tables' own SU(2) matrices are written in (see `_canonical_spinor_signs`).
    # Without tables there is nothing to match against and no sign was ever fitted.
    spin_frame = _pin_spinor_branch(sg) if tables_ready else None

    out_of_scope = _magnetic_scope_refusal(sg)
    if out_of_scope is not None:
        return out_of_scope

    # Are the operations a group?  On a cell built from exact numbers they are,
    # exactly, and nothing below runs.  On a NUMERICAL cell spglib reads each
    # translation off noisy atom pairs and they close only to that noise -- and
    # then no exactly symmetric set of positions exists for anything downstream
    # to be measured against.  See `_exact_operations`.
    consistent = _exact_operations(sg, symprec, tables_ready, lattice)
    if isinstance(consistent, Refusal):
        return consistent
    closure_measured, closure_repaired = consistent

    number_str = str(sg.number_str)
    try:
        sg_number = int(number_str.split(".")[0])
    except ValueError:  # pragma: no cover - defensive
        sg_number = -1

    out = GroupData(
        sg_number=sg_number, number_str=number_str,
        name=str(sg.name), spinor=bool(sg.spinor), magnetic=bool(sg.magnetic),
        include_TR=bool(include_TR), symmetries=list(sg.symmetries),
        lattice=lattice, site_positions=site_pos, site_orbitals=site_orbs,
        typat=species, periodic_dirs=tuple(int(d) for d in model.periodic_dirs),
        vacuum_dirs=vacuum_dirs, vacuum_length=float(vacuum_length),
        _refUC=np.array(sg.refUC, dtype=float), _shiftUC=np.array(sg.shiftUC, dtype=float),
        _tables_ready=tables_ready, symprec=float(symprec),
        dim_k=int(model.lattice.dim_k), dim_r=int(model.lattice.dim_r),
        _setup_note=note, provenance="cell", _spin_frame=spin_frame, _sg=sg,
        _closure_measured=float(closure_measured),
        _closure_repaired=float(closure_repaired),
        _lattice_index=len(translations) + 1,
        _fractional_translations=translations)
    _IDENT_MEMO[memo_key] = out
    _IDENT_MEMO_STATS["stores"] += 1
    while len(_IDENT_MEMO) > int(_IDENT_MEMO_MAX):
        _IDENT_MEMO.popitem(last=False)
        _IDENT_MEMO_STATS["evictions"] += 1
    return out


# --------------------------------------------------------------------------
# Wyckoff data and orbital assignment
# --------------------------------------------------------------------------

def wyckoff_positions_spglib(group: GroupData) -> tuple | Refusal:
    """Every Wyckoff position of the group, in the model's own cell, lettered in spglib's setting.

    Each position is a map ``A·(x, y, z) + b`` that a site can be tested
    against.  Its letter is spglib's, and the tables' letter for the position
    may differ: :func:`wyckoff_sites_tables` gives each site's letter in the
    tables' setting, and the orbits of
    ``realized_group(model).assignment`` are read off the group's own
    operations.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``; its cell and its sites are
        read, at spglib's own tolerance.

    Returns
    -------
    tuple of WyckoffPosition or Refusal
        The general position first and ``a`` last; ``summary()`` prints each.
        A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``wyckoff_enumeration``
        ``pyxtal`` and spglib disagree about the group, and no list of its
        positions is made.
    """
    import sympy
    from irrep.spacegroup import SpaceGroup as _SG

    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    cell = (group.lattice, group.site_positions, np.asarray(group.typat, dtype=int))
    try:
        exprs = _SG.wyckoff_positions(cell)
    except Exception as exc:  # pragma: no cover - pyxtal/spglib disagreement
        return Refusal("wyckoff_enumeration",
                       f"irrep could not enumerate Wyckoff positions for #{group.number_str}: {exc}",
                       context={"number_str": group.number_str})

    x, y, z = sympy.symbols("x y z")
    out = []
    n = len(exprs)
    for i, s in enumerate(exprs):
        parts = [sympy.sympify(t) for t in s.split(",")]
        b = np.array([float(p.subs({x: 0, y: 0, z: 0})) for p in parts])
        A = np.zeros((3, 3))
        for j, sym in enumerate((x, y, z)):
            sub = {x: 0.0, y: 0.0, z: 0.0}
            sub[sym] = 1.0
            A[:, j] = np.array([float(p.subs(sub)) for p in parts]) - b
        A[np.abs(A) < 1e-12] = 0.0
        b[np.abs(b) < 1e-12] = 0.0
        letter = _WYCKOFF_LETTERS[n - 1 - i] if n - 1 - i < len(_WYCKOFF_LETTERS) else "?"
        out.append(WyckoffPosition(letter=letter, expression=s, offset=b, jacobian=A,
                                   n_free=int(np.linalg.matrix_rank(A, tol=1e-9))))
    return tuple(out)


# --------------------------------------------------------------------------
# Wyckoff letters in the TABLES' setting
# --------------------------------------------------------------------------



@lru_cache(maxsize=64)
def _pyxtal_wyckoff(sg: int) -> tuple:
    r"""``pyxtal``'s Wyckoff list of a space-group number, **in the tables' setting**.

    Returns ``(positions, centering)``.  Each position is
    ``{"letter", "multiplicity", "n_free", "arms"}`` with ``arms`` the affine
    maps :math:`x = A p + b` of the position's symmetry-equivalent copies — one
    per operation of ``pyxtal``'s ``wp.ops`` — each carrying the projector
    :math:`P = 1 - A A^+` onto the complement of its variety, which is what a
    containment test needs.  ``centering`` is measured, not typed: the
    **general** position's ops with an identity rotation are exactly the
    conventional cell's centering translations
    (``(0,0,0), (0,½,½), (½,0,½), (½,½,0)`` for an ``F`` lattice).

    ``pyxtal`` is already this package's Wyckoff source —
    :func:`wyckoff_positions_spglib` reaches it through
    ``irrep.spacegroup.SpaceGroup.wyckoff_positions`` — but *there* it
    is composed with spglib's transformation into the calculation cell.  Here it
    is read raw, in its own conventional setting, which is the setting the
    tables use: for ``Fd-3m`` its ``8a`` is at ``(1/8, 1/8, 1/8)``, ITA origin
    choice 2.  That the two settings agree is not assumed — it is
    :func:`_wyckoff_setting_audit`, which measures the stabilizer of every
    position the EBR list names against the group's *own* operations.

    The class is taken off ``irrep.spacegroup``, which is where
    :func:`wyckoff_positions_spglib` already reaches it: ``irrep`` pins
    ``pyxtal==1.1.1`` exactly, so this adds no dependency — it uses ``irrep``'s
    own binding of the one it already has.
    """
    from irrep.spacegroup import Group

    group = Group(int(sg))
    positions, centering = [], []
    for wp in group.Wyckoff_positions:
        arms = []
        for op in wp.ops:
            affine = np.asarray(op.affine_matrix, dtype=float)
            A, b = affine[:3, :3], affine[:3, 3] % 1.0
            A = np.where(np.abs(A) < 1e-12, 0.0, A)
            projector = np.eye(3) - A @ np.linalg.pinv(A)
            arms.append((A, b, projector))
        positions.append({"letter": str(wp.letter),
                          "multiplicity": int(wp.multiplicity),
                          "n_free": int(wp.get_dof()),
                          "arms": tuple(arms)})
    general = max(positions, key=lambda p: p["multiplicity"])
    for A, b, _P in general["arms"]:
        if np.abs(A - np.eye(3)).max() < 1e-9:
            centering.append(b % 1.0)
    return tuple(positions), np.array(centering) if centering else np.zeros((1, 3))


def _variety_residual(arms, points) -> np.ndarray:
    """``min_arms min_n |P (x + n - b)|`` per point — 0 exactly on the variety."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    best = np.full(points.shape[0], np.inf)
    for _A, b, P in arms:
        # (n_point, 27, 3): every point, shifted by every neighbouring lattice
        # vector, projected onto the complement of this arm's variety
        delta = (points[:, None, :] - b[None, None, :] + _NEIGHBOUR_SHIFTS[None, :, :])
        resid = np.abs(delta @ P.T).max(axis=2).min(axis=1)
        best = np.minimum(best, resid)
    return best


def _variety_contains_variety(inner, outer, tol) -> bool:
    """Is the whole of one arm's affine subspace inside another's, mod the lattice?"""
    A_in, b_in, _ = inner
    _A_out, b_out, P_out = outer
    if float(np.abs(P_out @ A_in).max()) > tol:
        return False                      # the directions do not fit
    delta = b_in - b_out + _NEIGHBOUR_SHIFTS
    return bool(np.abs(delta @ P_out.T).max(axis=1).min() <= tol)


def _reference_operations(group: GroupData) -> tuple:
    r"""The group's own unitary operations, written in the reference cell.

    ``(R_ref, t_ref)`` per operation, with
    :math:`R_\text{ref} = \mathrm{refUC}^{-1} R\,\mathrm{refUC}` (an integer
    matrix, asserted) and
    :math:`t_\text{ref} = \mathrm{refUC}^{-1}(t + R s - s)`, ``s = shiftUC`` —
    ``irrep``'s own ``SymmetryOperation.translation_refUC``, the map
    :func:`_exact_operations` inverts.  Translations are snapped to
    small-denominator rationals the way :func:`_table_kpoints` snaps k-points,
    and for the same reason: a reference-cell translation is a rational with a
    small denominator, and ``irreptables`` prints its own to five decimals
    (``0.33333``), which is 3.3e-6 away from ``1/3`` — enough to lose a
    stabilizer at a 1e-6 gate.
    """
    refUC = np.asarray(group._refUC, dtype=float)
    shift = np.asarray(group._shiftUC, dtype=float)
    inv = np.linalg.inv(refUC)
    out = []
    for sym in group.symmetries:
        if bool(sym.time_reversal):
            continue
        R = np.asarray(sym.rotation, dtype=float)
        R_ref = inv @ R @ refUC
        if float(np.abs(R_ref - np.rint(R_ref)).max()) > 1e-6:
            raise InternalError(
                f"#{group.number_str}: the reference-cell rotation refUC^-1 R refUC of "
                f"operation #{sym.ind} is not an integer matrix (defect "
                f"{float(np.abs(R_ref - np.rint(R_ref)).max()):.3g}), so this group's "
                "operations cannot be written in the setting its own tables are written "
                "in. irrep's `rotation_refUC` asserts the same thing.")
        t_ref = inv @ (np.asarray(sym.translation, dtype=float)
                       + R @ shift - shift)
        out.append((np.rint(R_ref).astype(int), _snap_rational(t_ref)))
    return tuple(out)


def _stabilizer_order(point, ops_ref, centering, tol=1e-6) -> int:
    """How many reference-cell operations (centering included) fix ``point``."""
    count = 0
    for R, t in ops_ref:
        image = R @ point + t
        delta = image[None, :] + centering - point[None, :]
        count += int((np.abs(delta - np.rint(delta)).max(axis=1) <= tol).sum())
    return count


def _wyckoff_setting_audit(sg: int, ops_ref, ebr_positions) -> dict:
    r"""**Is ``pyxtal``'s Wyckoff list written in the tables' setting for this group?**

    The one thing the reference-cell letter rests on, measured rather than
    assumed.  For every position the group's EBR list names —
    ``"8a(43m1',43m)"`` gives the multiplicity ``8``, the letter ``a`` and the
    site symbol — two statements must hold:

    * ``pyxtal`` carries that letter with **that conventional multiplicity**;
    * the **stabilizer** of ``pyxtal``'s own representative point under the
      group's operations *written in the reference cell*, together with the
      conventional cell's centering translations, has order
      :math:`|P|\,n_\text{centering} / M`.  That is the orbit–stabilizer theorem
      in the conventional cell, and it is the sharp test: the two sides come
      from independent sources (``pyxtal`` supplies the point, ``irrep`` /
      ``irreptables`` supply the operations), so an origin choice or an axis
      convention that differed between them would move the point off its own
      site-symmetry group and the order would collapse.

    The representative is taken at a **generic** value of the free parameters,
    which is where a position's site-symmetry group is the one its symbol names;
    a special value would land on a *more* special position and measure that
    one's group instead.

    Returns ``{"ok": bool, "offenders": (...), "point_group": n, "centering": n,
    "checked": n}``.  Over all 230 space groups, in both the unitary and the grey
    (type-II) tables, there are **zero offenders**.

    Notes
    -----
    Both valuednesses share the letters — a double-valued EBR list induces from
    the same positions — so the sweep runs once per group.
    """
    positions, centering = _pyxtal_wyckoff(int(sg))
    by_letter = {p["letter"]: p for p in positions}
    order = len(ops_ref) * len(centering)
    offenders, checked, seen = [], 0, {}
    for multiplicity, letter, symbol in ebr_positions:
        if letter in seen:
            continue
        seen[letter] = True
        wp = by_letter.get(letter)
        if wp is None:
            offenders.append({"letter": letter, "reason": "absent from pyxtal's list",
                              "measured": None, "expected": int(multiplicity),
                              "site_symbol": symbol})
            continue
        if wp["multiplicity"] != int(multiplicity):
            offenders.append({"letter": letter, "reason": "multiplicity",
                              "measured": wp["multiplicity"],
                              "expected": int(multiplicity), "site_symbol": symbol})
            continue
        A, b, _P = wp["arms"][0]
        point = A @ _GENERIC_PARAMETER + b
        measured = _stabilizer_order(point, ops_ref, centering)
        expected = order // int(multiplicity) if order % int(multiplicity) == 0 else None
        if measured != expected:
            offenders.append({"letter": letter, "reason": "stabilizer order",
                              "measured": measured, "expected": expected,
                              "site_symbol": symbol,
                              "point": np.round(point, 6).tolist()})
        checked += 1
    return {"ok": not offenders, "offenders": tuple(offenders),
            "point_group": len(ops_ref), "centering": len(centering),
            "checked": checked}


#: A generic value of a Wyckoff position's free parameters.  Any point of the
#: variety that is not on a *more* special one will do; these three are not a
#: rational with a small denominator in any combination the settings use, so no
#: position of any of the 230 groups passes through this point accidentally.
_GENERIC_PARAMETER = np.array([0.13, 0.29, 0.41])


def wyckoff_sites_tables(
        group: GroupData, positions=None, *,
        position_tol: float = WYCKOFF_REFERENCE_TOL) -> tuple | Refusal:
    """The Wyckoff position of each site, lettered in the tables' setting, the one every table is read with.

    A letter is a table key, and settings disagree about it: the Fu–Kane–Mele
    diamond's orbit is ``8a`` here and ``8b`` in spglib's setting, and a site
    irrep at two positions can carry the same content at every k-point and
    different real-space invariants.  The orbitals' own letters are in
    ``realized_group(model).assignment.wyckoff_tables``.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.  A grey group's positions
        are those of its unitary subgroup.
    positions : array, shape (n, 3), optional
        Points in reduced coordinates of the group's three-dimensional cell.
        Default: the points the group was found on,
        :attr:`GroupData.site_positions`.
    position_tol : float, optional
        How far a point may sit from a position and lie on it, in reduced
        coordinates of the tables' conventional cell.  Default
        :data:`WYCKOFF_REFERENCE_TOL`, ``1e-4``.

    Returns
    -------
    tuple of SiteReference or Refusal
        One per point: its letter and multiplicity, its free parameters, and
        the positions an orbital confined there can be moved to.  A
        ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell.
    ``wyckoff_setting``
        ``pyxtal``'s list of the group's positions is not written in the
        tables' setting, so no letter would be a table key.
    ``wyckoff_unresolved``
        A point lies on no position within ``position_tol``; the refusal names
        the nearest and its distance.
    ``ebr_tables``, ``irrep_tables``, ``ebr_convention``
        From :func:`ebr_data`, whose positions the letters are checked
        against.

    Examples
    --------
    >>> from pythtb.models import fu_kane_mele                         # doctest: +SKIP
    >>> r = realized_group(fu_kane_mele(1.0, 0.125, [0, 0, 0, 0]))     # doctest: +SKIP
    >>> [(s.multiplicity, s.letter) for s in wyckoff_sites_tables(r.group)]  # doctest: +SKIP
    [(8, 'a'), (8, 'a')]
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or "irrep tables are not set up for this cell, "
                       "so there is no refUC/shiftUC to write positions in the "
                       "reference cell with",
                       context={"number_str": group.number_str})
    data = ebr_data(group)
    if isinstance(data, Refusal):
        return data
    ebr_positions = []
    for _name, position in data["ebrs"]:
        match = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", str(position))
        if match is None:                              # pragma: no cover - guard
            raise InternalError(
                f"#{group.number_str}: EBR position {position!r} is not "
                "'<multiplicity><letter>(<site symbols>)', so its Wyckoff letter cannot "
                "be read")
        ebr_positions.append((int(match.group(1)), match.group(2), match.group(3)))
    listed = frozenset(letter for _m, letter, _s in ebr_positions)

    audit = group._wyckoff_audit.get("audit")
    if audit is None:
        audit = _wyckoff_setting_audit(int(group.sg_number),
                                       _reference_operations(group),
                                       ebr_positions)
        group._wyckoff_audit["audit"] = audit
    if not audit["ok"]:
        bad = audit["offenders"][0]
        return Refusal(
            "wyckoff_setting",
            f"pyxtal's Wyckoff list for space group {group.sg_number} is not written in "
            f"the setting #{group.number_str}'s own tables are: position "
            f"{bad['expected'] if bad['reason'] != 'multiplicity' else ''}{bad['letter']} "
            f"fails the {bad['reason']} check (measured {bad['measured']}, expected "
            f"{bad['expected']}). Every Wyckoff letter this function would return is a "
            "key into the wrong setting, so none is returned",
            value=None if bad["measured"] is None else float(bad["measured"]),
            threshold=None if bad["expected"] is None else float(bad["expected"]),
            context={"number_str": str(group.number_str), "sg": int(group.sg_number),
                     "offenders": audit["offenders"], "audit": audit})

    pyxtal_positions, _centering = _pyxtal_wyckoff(int(group.sg_number))
    raw = (np.asarray(group.site_positions, dtype=float) if positions is None
           else np.atleast_2d(np.asarray(positions, dtype=float)))
    inv = np.linalg.inv(np.asarray(group._refUC, dtype=float))
    shift = np.asarray(group._shiftUC, dtype=float)
    reference = np.array([(inv @ (x - shift)) % 1.0 for x in raw])
    reference = np.where(np.abs(reference - 1.0) < 1e-9, 0.0, reference)

    residuals = {p["letter"]: _variety_residual(p["arms"], reference)
                 for p in pyxtal_positions}
    out = []
    for i, (x, x_ref) in enumerate(zip(raw, reference)):
        hits = [p for p in pyxtal_positions
                if residuals[p["letter"]][i] <= position_tol]
        if not hits:                     # pragma: no cover - the general position's
            # variety is the whole cell, so this needs a pyxtal list without one
            nearest = min(pyxtal_positions, key=lambda p: residuals[p["letter"]][i])
            return Refusal(
                "wyckoff_unresolved",
                f"the position {np.round(x, 6).tolist()} is "
                f"{np.round(x_ref, 6).tolist()} in #{group.number_str}'s reference cell "
                f"and lies on no Wyckoff variety of space group {group.sg_number} there: "
                f"the nearest is {nearest['multiplicity']}{nearest['letter']}, "
                f"{float(residuals[nearest['letter']][i]):.3g} away in reduced "
                f"coordinates. A centre that far off every orbit is a modelling defect",
                value=float(residuals[nearest["letter"]][i]),
                threshold=float(position_tol),
                context={"site": i, "position": x, "reference": x_ref,
                         "nearest": nearest["letter"],
                         "number_str": str(group.number_str)})
        hits.sort(key=lambda p: (p["n_free"], p["multiplicity"], p["letter"]))
        own = hits[0]
        out.append(SiteReference(
            index=i, position=x, reference=x_ref, letter=own["letter"],
            multiplicity=own["multiplicity"], n_free=own["n_free"],
            varieties=tuple(p["letter"] for p in hits),
            closure=_letters_on_closure(int(group.sg_number), own["letter"], listed,
                                        float(position_tol)),
            residual=float(residuals[own["letter"]][i])))
    return tuple(out)


@lru_cache(maxsize=256)
def _letters_on_closure(sg: int, letter: str, listed: frozenset,
                        tol: float = WYCKOFF_REFERENCE_TOL) -> tuple:
    """The letters of ``listed`` whose varieties lie on the closure of ``letter``'s, in order.

    A position whose variety lies inside another's is where an orbital confined
    to the larger variety can be moved without leaving it; ``letter``'s own
    variety lies inside itself, so it is among them wherever ``listed`` holds it.
    """
    positions, _centering = _pyxtal_wyckoff(int(sg))
    own = next((p for p in positions if p["letter"] == str(letter)), None)
    if own is None:
        return ()
    return tuple(sorted(
        p["letter"] for p in positions
        if p["letter"] in listed
        and any(_variety_contains_variety(arm, own_arm, tol)
                for arm in p["arms"] for own_arm in own["arms"])))


def _assign_orbitals(model, group: GroupData, *, typat=None,
                     symprec=None) -> OrbitalAssignment | Refusal:
    """Which site orbit each orbital sits on, and how the operations move them.

    Returned as :attr:`RealizedGroup.assignment`.  Each operation's image
    ``R·τ + t`` is looked up among the
    sites of the same species, and the permutation that comes out — with the
    integer lattice shift beside it, which is what fixes a permuting
    operation's embedding phase — is the data a representation is built from.
    Orbits, site-symmetry groups and each orbit's Wyckoff position are read off
    it.

    Parameters
    ----------
    model : pythtb.TBModel
        The orbital positions are read from **this** model, not from the group,
        so a model whose orbitals have drifted off the ideal positions can be
        checked against a group determined elsewhere (or at a looser
        ``symprec``).  Its lattice must match the group's 3D cell: the
        periodic vectors the same, and a layer's vector along its normal read at
        the length the group's cell gives it.
    group : GroupData
        The group whose operations do the permuting; its ``symprec`` is the
        default matching tolerance.
        A group comes from :func:`realized_group`:
        ``realized_group(model, ...).group``.
    typat : sequence, length norb, optional
        Orbital-class labels, with the same meaning and default as
        :func:`realized_group`'s: the value given there.
    symprec : float, optional
        Matching tolerance — Cartesian, in the model's length units.  Defaults
        to the group's own ``symprec``.

    Returns
    -------
    OrbitalAssignment or Refusal

    Refusals
    --------
    ``wyckoff_orbit``
        An orbital's image under some operation lands on no orbital — the
        refusal names the orbital, the operation and the **distance** to the
        nearest candidate.  A misplaced orbital is a modelling bug and is never
        silently symmetrized away.

    Raises
    ------
    ValueError
        When ``typat`` does not carry one entry per orbital, or the model's
        cell, its vacuum included, is not the group's: there is no assignment to make
        between a model and a group built on another cell.

    Notes
    -----
    The orbit structure reported here is *measured* from the group's
    operations.  The letters are read beside it, in both settings: from
    :func:`wyckoff_positions_spglib` per orbit and from
    :func:`wyckoff_sites_tables` per site (left empty where the enumeration or
    the reference cell was unavailable), with spglib's own per-site letter for
    the sites alone, in the group of the positions, alongside.
    """
    tol = float(group.symprec if symprec is None else symprec)
    lat, positions, _ = _pad_cell(model, group.vacuum_length, lattice=group.lattice)
    if not np.allclose(lat, group.lattice, atol=1e-9):
        apart = float(np.abs(np.asarray(lat, dtype=float)
                             - np.asarray(group.lattice, dtype=float)).max())
        raise ValueError(f"the model's cell, its vacuum included, is not the group's, "
                         f"{apart:.3g} apart: the group was built for another cell.")
    pos, site_orbs, species = _sites_and_typat(positions, typat, tol, lat)
    nsite = pos.shape[0]
    nsym = len(group.symmetries)

    perm = np.full((nsym, nsite), -1, dtype=int)
    shift = np.zeros((nsym, nsite, 3), dtype=int)
    worst = 0.0
    for g, sym in enumerate(group.symmetries):
        images = pos @ np.asarray(sym.rotation, dtype=float).T + sym.translation
        for i in range(nsite):
            d = pos - images[i]
            d = d - np.round(d)
            dist = np.linalg.norm(d @ lat, axis=1)
            same = species == species[i]
            dist_masked = np.where(same, dist, np.inf)
            j = int(np.argmin(dist_masked))
            if not np.isfinite(dist_masked[j]) or dist_masked[j] > tol:
                orbs = site_orbs[i]
                return Refusal(
                    "wyckoff_orbit",
                    f"orbital(s) {list(orbs)} at reduced position {np.round(pos[i], 6)} are not "
                    f"carried onto any orbital by symmetry operation #{sym.ind} "
                    f"(rotation {sym.rotation.tolist()}, translation "
                    f"{np.round(sym.translation, 6).tolist()}): the nearest orbital of the same "
                    f"class is {float(np.min(dist_masked)):.6g} away. A misplaced orbital is a "
                    f"modelling bug, not something to symmetrize away.",
                    value=float(np.min(dist_masked)), threshold=tol,
                    context={"site": i, "orbitals": list(orbs), "operation_ind": int(sym.ind),
                             "position": pos[i], "image": images[i] % 1.0,
                             "distances": dist})
            worst = max(worst, float(dist_masked[j]))
            perm[g, i] = j
            shift[g, i] = np.round(images[i] - pos[j]).astype(int)

    # orbits = connected components of the permutation action
    orbit_of = np.full(nsite, -1, dtype=int)
    orbits: list[tuple] = []
    for i in range(nsite):
        if orbit_of[i] >= 0:
            continue
        members = sorted(set(int(perm[g, i]) for g in range(nsym)))
        idx = len(orbits)
        for j in members:
            orbit_of[j] = idx
        orbits.append(tuple(members))

    site_sym = tuple(tuple(int(group.symmetries[g].ind) for g in range(nsym) if perm[g, i] == i)
                     for i in range(nsite))

    # Wyckoff letters: irrep returns ONE representative per position, so a site may
    # sit on a symmetry-related copy of it -- test the whole orbit and keep the most
    # special (fewest free parameters) match.
    # A lattice supergroup's letters are the finer lattice's, so none is read:
    # the geometry above is the whole of what this record can honestly carry.
    supergroup = int(getattr(group, "_lattice_index", 1) or 1) > 1
    letters: list[str] = []
    wy = (Refusal("lattice_not_descended", "") if supergroup
          else wyckoff_positions_spglib(group))
    for orb in orbits:
        if isinstance(wy, Refusal):
            letters.append("")
            continue
        best, best_free = "", 99
        for w in wy:
            if w.n_free >= best_free:
                continue
            if any(w.distance(pos[i], lat) <= max(tol, 1e-6) for i in orb):
                best, best_free = w.letter, w.n_free
        letters.append(best)

    orbit_of_orbital = np.zeros(model.norb, dtype=int)
    site_of_orbital = np.zeros(model.norb, dtype=int)
    for s, orbs in enumerate(site_orbs):
        for o in orbs:
            orbit_of_orbital[o] = orbit_of[s]
            site_of_orbital[o] = s

    # the letter in the TABLES' setting -- the one a table lookup needs.
    # A refusal here is information -- the letter is left empty -- and not a
    # defect: the orbit structure above is measured from the operations and does
    # not depend on it.
    ref_letters: tuple = ()
    if not supergroup:
        sites_ref = wyckoff_sites_tables(group, pos)
        if not isinstance(sites_ref, Refusal):
            ref_letters = tuple(s.letter for s in sites_ref)

    # spglib's own per-site letter for the sites alone: a letter of the group of
    # the positions, which can be larger than the group handed in
    spg_letters: tuple = ()
    try:
        import spglib
        ds = None if supergroup else spglib.get_symmetry_dataset(
            (lat, pos, np.asarray(species, dtype=int)), symprec=tol)
        if ds is not None:
            spg_letters = tuple(ds.wyckoffs)
    except Exception:  # pragma: no cover - spglib API drift must not break the record
        pass

    # the site-symmetry group's NAME, spelled as the atomic-limit table spells
    # it: the point group of the site's own unitary operations, read off those
    # operations.  The sites of one orbit have conjugate site groups and so one
    # name, which is read once per orbit.
    from . import _sitechars

    unitary = {int(s.ind) for s in group.symmetries if not s.time_reversal}
    orbit_names = [_sitechars.point_group_symbol(
        group, [i for i in site_sym[int(orb[0])] if i in unitary]) for orb in orbits]
    site_names = tuple(orbit_names[int(orbit_of[i])] for i in range(nsite))

    return OrbitalAssignment(
        orbits=tuple(orbits), orbit_of_site=orbit_of, orbit_of_orbital=orbit_of_orbital,
        site_of_orbital=site_of_orbital, site_group=site_sym,
        permutations=perm, lattice_shift=shift,
        wyckoff_spglib=tuple(letters), max_distance=worst,
        site_positions=pos, site_orbitals=site_orbs,
        wyckoff_spglib_dataset=spg_letters, site_symmetry=site_names,
        wyckoff_tables=ref_letters)


# --------------------------------------------------------------------------
# the operations by index, and their Seitz symbols
# --------------------------------------------------------------------------

#: The order of the rotation part of an integer rotation matrix, read off the
#: trace of its proper part ``det(R)·R``, whose eigenvalues are
#: ``1, e^{±2πi/n}``.  The determinant then separates ``n`` from ``n̄``, which
#: one trace leaves open and ITA's symbols keep apart.
_ROTATION_ORDER = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}

#: Largest denominator a conventional-setting translation is written with.  The
#: tables carry halves, thirds, quarters, sixths, eighths and twelfths, and a
#: translation that snaps to nothing at this denominator within
#: :data:`_LATTICE_TOL` is arithmetic failing rather than a fraction.
_SEITZ_DENOMINATOR = 24


def _primitive_direction(u) -> tuple:
    """A direction as its smallest integer triple, first nonzero component positive."""
    v = np.rint(np.asarray(u, dtype=float)).astype(int)
    divisor = 0
    for x in v:
        divisor = math.gcd(divisor, abs(int(x)))
    if divisor > 1:
        v = v // divisor
    for x in v:
        if x:
            if x < 0:
                v = -v
            break
    return tuple(int(x) for x in v)


def _rotation_axis(proper) -> tuple:
    """The axis of a proper rotation — a reflection's normal — as an integer triple.

    The axis spans the kernel of ``proper − 1``, so it is orthogonal to that
    matrix's rows and the cross product of two independent ones gives it
    exactly, in integers, with no eigenvalue problem to solve.
    """
    rows = np.rint(np.asarray(proper, dtype=float)).astype(int) - np.eye(3, dtype=int)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        u = np.cross(rows[i], rows[j])
        if u.any():
            return _primitive_direction(u)
    raise InternalError(                             # pragma: no cover - guard
        "an operation's rotation fixes a plane rather than an axis, which no "
        "proper rotation other than the identity does. This is arithmetic in this "
        "package failing, not a measurement about the model.")


def _direction_text(axis) -> str:
    """An integer direction as ITA writes it, one signed digit per component."""
    if max(abs(int(x)) for x in axis) > 9:
        raise InternalError(                         # pragma: no cover - guard
            f"the symmetry direction {tuple(int(x) for x in axis)} has a component "
            "above nine, which no crystallographic axis has, so writing it digit by "
            "digit would not read back. This is arithmetic in this package failing.")
    return "".join(str(int(x)) for x in axis)


def _parse_direction(text) -> tuple | None:
    """An integer direction read back from the digits it is written as."""
    out, sign, digits = [], 1, str(text)
    while digits:
        head, digits = digits[0], digits[1:]
        if head == "-":
            sign = -1
            continue
        if not head.isdigit():
            return None
        out.append(sign * int(head))
        sign = 1
    return tuple(out) if len(out) == 3 else None


def _rotation_sense(sym, group, axis) -> int:
    """``+1`` or ``−1``: which way the rotation part turns about ``axis``.

    The turn is a fact about the crystal and not about a setting, so it is
    measured where the geometry is — in the model's own Cartesian frame, by
    :func:`.._sitechars.classify` — and then referred to the direction the
    symbol names, which is written in the tables' conventional basis.
    """
    from . import _sitechars

    info = _sitechars.classify(sym)
    direction = ((np.asarray(group._refUC, dtype=float) @ np.asarray(axis, dtype=float))
                 @ np.asarray(group.lattice, dtype=float))
    return int(info["sense"]) * (1 if float(direction @ info["axis"]) >= 0 else -1)


def _fraction_mod_one(x) -> Fraction:
    """One translation component as an exact fraction of the conventional cell."""
    value = Fraction(float(x)).limit_denominator(_SEITZ_DENOMINATOR)
    if abs(float(value) - float(x)) > _LATTICE_TOL:
        raise InternalError(                         # pragma: no cover - guard
            f"the conventional-setting translation component {float(x):.9g} is no "
            f"fraction of denominator {_SEITZ_DENOMINATOR} or less, though every "
            "tabulated translation is one. This is arithmetic in this package "
            "failing, not a measurement about the model.")
    return value % 1


def _reduced_translation(t, centering) -> tuple:
    r"""``t`` reduced modulo the crystal's own lattice, as three exact fractions.

    A Seitz symbol's translation is fixed only up to a lattice translation, and
    on a centred lattice the conventional cell holds more than one
    representative of the same operation — the ``F`` lattice holds four.  This
    is the shortest of them: smallest sum of components, ties broken towards
    the smaller first component.  It is what a written symbol is *matched*
    against, so that any representative names the operation it belongs to; what
    is written is the tables' own (:func:`_written_translation`).  The two agree
    on every operation of all 230 space groups in both valuednesses but 108,
    which are the ``c``-glide F-centred cubic groups #219, #226 and #228, where
    the tables keep ``1/2,1/2,1/2``.
    """
    best = None
    for shift in ((0.0, 0.0, 0.0),) + tuple(centering):
        parts = tuple(_fraction_mod_one(float(a) + float(b))
                      for a, b in zip(np.asarray(t, dtype=float), shift))
        key = (sum(parts), parts)
        if best is None or key < best[0]:
            best = (key, parts)
    return best[1]


def _centering_of(group) -> tuple:
    """The centering translations of this group's Bravais lattice."""
    letter = str(group.name)[:1]
    if letter not in _CENTERINGS:
        raise InternalError(                         # pragma: no cover - guard
            f"#{group.number_str}'s name {group.name!r} does not start with a Bravais "
            "lattice letter, so the lattice its translations are reduced modulo "
            "cannot be determined.")
    return _CENTERINGS[letter]


def _tabulated_translations(group) -> dict:
    """The group's own tabulated conventional-setting translations, by rotation.

    Keyed ``(reference-cell rotation, antiunitary)``.  Matching this cell to the
    standard setting is what puts the tabulated operations on the record in the
    first place, so these are read off what is already there rather than loaded:
    ``irrep`` keeps them as ``u_symmetries_tables`` and ``au_symmetries_tables``
    and pairs them with the cell's own operations through the same reference-cell
    rotation.  Empty for a record whose setting was never determined, which no
    public call reaches.
    """
    sg = getattr(group, "_sg", None)
    out = {}
    for entry in (list(getattr(sg, "u_symmetries_tables", ()) or ())
                  + list(getattr(sg, "au_symmetries_tables", ()) or ())):
        rotation = np.rint(np.asarray(entry.R, dtype=float)).astype(int)
        out[(rotation.tobytes(), bool(entry.time_reversal))] = np.asarray(
            entry.t, dtype=float)
    return out


def _written_translation(sym, group, tabulated, shortest) -> tuple:
    """The translation the symbol is written with: the tables' own where they carry it.

    A physicist reads this symbol against a printed general position, so the
    representative written is the one the tables hold for that operation.  Where
    the tables carry no entry for it, what is written is the shortest
    representative instead (:func:`_reduced_translation`).
    """
    rotation = np.asarray(sym.rotation_refUC(np.asarray(group._refUC, dtype=float)),
                          dtype=int)
    entry = tabulated.get((rotation.tobytes(), bool(sym.time_reversal)))
    if entry is None:
        return shortest
    return tuple(_fraction_mod_one(x) for x in entry)


def _translation_text(parts) -> str:
    """Three translation components as ITA writes them, comma separated.

    Each is an exact fraction, but in a model's own frame
    (:func:`_own_frame_translation`), where a component that is none is the
    float it is, written as a decimal.
    """
    return ",".join(f"{p:.4g}" if isinstance(p, float)
                    else f"{p.numerator}" if p.denominator == 1
                    else f"{p.numerator}/{p.denominator}" for p in parts)


def _own_frame_translation(t) -> tuple:
    """``t``, in a model's own reduced coordinates, modulo its lattice.

    Each component is the exact fraction it is within :data:`_LATTICE_TOL`, of
    denominator :data:`_SEITZ_DENOMINATOR` or less, as every tabulated
    translation is (:func:`_fraction_mod_one`).  A model whose origin sits off
    every point its operations fix carries other translations in its own
    frame, and such a component is kept as the number it is, modulo one.  A
    lattice vector of the cell reads as three zeros.
    """
    parts = []
    for x in np.asarray(t, dtype=float).reshape(-1):
        x = float(x) % 1.0
        value = Fraction(x).limit_denominator(_SEITZ_DENOMINATOR)
        parts.append(value % 1 if abs(float(value) - x) <= _LATTICE_TOL else x)
    return tuple(parts)


def _record_name(sym, lattice) -> str:
    """One operation of a group as a record lists it.

    Its name in the frame ``characters=`` is written in
    (:func:`._sitechars.operation_name`), with its translation in Seitz form,
    in the cell's reduced coordinates (:func:`_own_frame_translation`), where
    that is not a lattice vector of the cell -- ``{E|1/2,0,0}``, a half
    translation, which ``E`` alone would print as the identity -- and a ``'``
    where it carries time reversal.  The frame a declaration is written in
    names a site's operations without their translations: its own operations
    are told apart by their rotations alone.
    """
    from . import _sitechars

    name = _sitechars.operation_name(sym, np.asarray(lattice, dtype=float))
    parts = _own_frame_translation(sym.translation)
    if any(parts):
        name = "{" + name + "|" + _translation_text(parts) + "}"
    return name + ("'" if bool(sym.time_reversal) else "")


def _seitz_key(sym, group) -> tuple:
    """``(det, order, axis, sense, translation)`` of one operation, conventional setting.

    Everything the symbol is written from, and everything a written symbol is
    matched against — the same quantities either way, so that a spelling that
    names this operation resolves to it whichever representative it uses.
    """
    refUC = np.asarray(group._refUC, dtype=float)
    rotation = np.asarray(sym.rotation_refUC(refUC), dtype=int)
    det = int(round(float(np.linalg.det(rotation))))
    proper = det * rotation
    trace = int(round(float(np.trace(proper))))
    if trace not in _ROTATION_ORDER:                   # pragma: no cover - guard
        raise InternalError(
            f"operation #{int(sym.ind)}'s rotation has trace {trace} and determinant "
            f"{det} in the conventional setting, which is no crystallographic "
            "rotation. This is arithmetic in this package failing, not a measurement "
            "about the model.")
    order = _ROTATION_ORDER[trace]
    axis = () if order == 1 else _rotation_axis(proper)
    sense = _rotation_sense(sym, group, axis) if order > 2 else 0
    t = sym.translation_refUC(refUC, np.asarray(group._shiftUC, dtype=float))
    return det, order, axis, sense, _reduced_translation(t, _centering_of(group))


def _seitz_text(key, time_reversal, written=None) -> str:
    """One operation's Seitz symbol, from the quantities it is written from."""
    det, order, axis, sense, parts = key
    parts = parts if written is None else written
    if order == 1:
        rotation = "1" if det > 0 else "-1"
    elif order == 2:
        rotation = ("2" if det > 0 else "m") + "_" + _direction_text(axis)
    else:
        rotation = (("" if det > 0 else "-") + str(order) + ("+" if sense > 0 else "-")
                    + "_" + _direction_text(axis))
    return ("{" + rotation + "|" + _translation_text(parts) + "}"
            + ("'" if time_reversal else ""))


def _parse_seitz(text, centering) -> tuple | None:
    """A written Seitz symbol read back, or ``None`` when it is not one.

    Gives ``((det, order, axis, sense, translation), time_reversal)`` in the
    same quantities :func:`_seitz_key` measures, with ``sense`` ``0`` where the
    spelling left it out.  The direction is turned the way
    :func:`_primitive_direction` turns it and the sense turns with it, since
    ``n⁺`` about ``[uvw]`` and ``n⁻`` about ``[-u-v-w]`` are one operation; the
    translation is reduced modulo the crystal's lattice, so any representative
    of it names the same operation.
    """
    body = str(text).strip()
    time_reversal = body.endswith("'")
    if time_reversal:
        body = body[:-1].rstrip()
    if not (body.startswith("{") and body.endswith("}") and "|" in body):
        return None
    rotation, _, translation = body[1:-1].partition("|")
    rotation = "".join(rotation.split())
    det = 1
    if rotation.startswith("-"):
        det, rotation = -1, rotation[1:]
    if not rotation:
        return None
    head, rest = rotation[0], rotation[1:]
    if head == "m":
        det, order = -1, 2
    elif head in "12346":
        order = int(head)
    else:
        return None
    sense = 0
    if order > 2 and rest[:1] in ("+", "-"):
        sense = 1 if rest[0] == "+" else -1
        rest = rest[1:]
    rest = rest.lstrip("_")
    axis = _parse_direction(rest) if rest else ()
    if axis is None or bool(axis) != (order > 1):
        return None
    if axis:
        turned = _primitive_direction(axis)
        if turned != tuple(axis):
            sense = -sense
        axis = turned
    translation = translation.strip()
    if "," not in translation:
        translation = ",".join(translation.split())
    parts = translation.split(",")
    if len(parts) != 3:
        return None
    try:
        values = [Fraction(p.strip()) for p in parts]
    except (ValueError, ZeroDivisionError):
        return None
    return ((det, order, axis, sense,
             _reduced_translation([float(v) for v in values], centering)),
            time_reversal)


def _matches(key, time_reversal, wanted, wanted_tr) -> bool:
    """Whether a written symbol names the operation one row carries."""
    if bool(time_reversal) != bool(wanted_tr):
        return False
    det, order, axis, sense, parts = key
    w_det, w_order, w_axis, w_sense, w_parts = wanted
    if (det, order, tuple(axis), parts) != (w_det, w_order, tuple(w_axis), w_parts):
        return False
    return w_sense == 0 or w_sense == sense


@dataclass(frozen=True)
class Operation:
    """One operation of a group: its index, its Seitz symbol in the tables' setting, and its name in the model's frame.

    One row of an :class:`OperationTable`, ``r.operations[ind]``;
    ``summary()`` prints it.

    Attributes
    ----------
    ind : int
        Its index in the tables' order, the key every per-operation record is
        written on: ``rep.matrices``, ``validated.verdicts``, the entries of a
        descent.
    seitz : str
        Its Seitz symbol ``{R|t}`` in the tables' conventional setting,
        ``{-4+_001|0,0,1/2}``, with a trailing ``'`` where it is combined with
        time reversal.
    name : str
        The same operation in the model's own Cartesian frame, the frame
        ``characters=`` is written in: ``C4(z)``, ``m(x)``,
        ``S4^-1([1-10])``, an axis along none of ``x``, ``y`` and ``z`` named
        by its direction in the lattice vectors; primed as the Seitz symbol is.
    rotation, translation : ndarray, shapes (3, 3) and (3,)
        Its rotation, integer, and its translation, in the model's reduced
        coordinates.
    rotation_cartesian : ndarray, shape (3, 3)
        Its rotation in the Cartesian axes the lattice vectors are written in,
        a layer's normal as ``z``.
    spin_lift : ndarray, shape (2, 2), or None
        Its matrix on the spin, ``e^{−iθ n̂·σ/2}`` with ``n̂`` in those axes
        and the spin along their ``z``, an improper operation's that of its
        proper part; ``None`` in a group of states without spin.
    time_reversal : bool
        Whether it is combined with time reversal, as half of a grey group's
        operations are.
    """

    ind: int
    seitz: str
    name: str
    rotation: np.ndarray
    translation: np.ndarray
    time_reversal: bool
    rotation_cartesian: np.ndarray | None = None
    spin_lift: np.ndarray | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operation):
            return NotImplemented
        return (self.ind == other.ind and self.seitz == other.seitz
                and self.name == other.name and self.time_reversal == other.time_reversal
                and np.array_equal(self.rotation, other.rotation)
                and np.array_equal(self.translation, other.translation))

    def summary(self) -> str:
        """The operation, a line for each part, with the attribute that holds it."""
        lines = [f"operation {int(self.ind)}: {self.seitz} in the tables' setting, "
                 f"{self.name} in the model's frame   (.ind, .seitz, .name)",
                 "  its rotation and translation in the model's reduced coordinates   "
                 "(.rotation, .translation)"]
        if self.rotation_cartesian is not None:
            lines.append("  its rotation in the model's Cartesian axes   (.rotation_cartesian)")
        if self.spin_lift is not None:
            lines.append("  its matrix on the spin, spin along the z of those axes   "
                         "(.spin_lift)")
        if self.time_reversal:
            lines.append("  combined with time reversal, which the ' marks   (.time_reversal)")
        return "\n".join(lines)


class OperationTable(dict):
    """The operations of one group, a mapping from each index to its :class:`Operation`.

    What :func:`operations` returns and ``r.operations`` holds: ``table[4]`` is
    operation 4, and ``.values()`` gives the operations in index order.  It
    prints as a table of each operation's index, Seitz symbol and name, which
    ``summary()`` returns.

    Attributes
    ----------
    name, number_str : str
        The group's Hermann–Mauguin symbol, and the label the tables file it
        under.
    """

    def __init__(self, rows=(), *, name="", number_str=""):
        super().__init__((row.ind, row) for row in rows)
        #: The Hermann–Mauguin symbol of the group these operations belong to.
        self.name = str(name)
        #: The label the tables file that group under, ``"191.234"``.
        self.number_str = str(number_str)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"OperationTable({self.name} #{self.number_str}, operations "
                f"{_index_ranges(self)})")

    def summary(self) -> str:
        """The table itself: each operation's index, its Seitz symbol and its name."""
        return str(self)

    def __str__(self) -> str:
        rows = list(self.values())
        width = max([len(row.seitz) for row in rows] + [5])
        head = (f"{self.name} #{self.number_str}: one operation per coset of the lattice "
                "translations"
                + ("; the double group also holds each one's negative"
                   if any(row.spin_lift is not None for row in rows) else ""))
        lines = [head, f"{'ind':>4}  {'Seitz':<{width}}  name"]
        lines += [f"{row.ind:>4}  {row.seitz:<{width}}  {row.name}" for row in rows]
        return "\n".join(lines)


def _operation_argument(group_or_rep) -> tuple:
    """The group one of the accepted records is about, and the indices it names."""
    if isinstance(group_or_rep, Refusal):
        return group_or_rep, ()
    if isinstance(group_or_rep, SpaceGroupRep):
        return group_or_rep.group, tuple(sorted(int(i) for i in group_or_rep.matrices))
    if isinstance(group_or_rep, RealizedGroup):
        group_or_rep = group_or_rep.group
    if isinstance(group_or_rep, GroupData):
        return group_or_rep, tuple(int(s.ind) for s in group_or_rep.symmetries)
    raise TypeError(
        "operations are read off a measured record, a group or a representation "
        f"of one, and {type(group_or_rep).__name__} is none of those. From "
        "`r = realized_group(model)` the group is `r.group` and the representation "
        "`r.rep`, and `r` itself is read as its group.")


def _operation_rows(group_or_rep) -> OperationTable | Refusal:
    """Every operation the argument names, as rows."""
    group, indices = _operation_argument(group_or_rep)
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready", group._setup_note or "irrep tables are not set "
                       "up for this cell", context={"number_str": group.number_str})
    from . import _sitechars

    lattice = np.asarray(group.lattice, dtype=float)
    tabulated = _tabulated_translations(group)
    wanted = set(indices)
    rows = []
    for sym in group.symmetries:
        if int(sym.ind) not in wanted:
            continue
        prime = "'" if sym.time_reversal else ""
        key = _seitz_key(sym, group)
        rows.append(Operation(
            ind=int(sym.ind),
            seitz=_seitz_text(key, bool(sym.time_reversal),
                              _written_translation(sym, group, tabulated, key[4])),
            name=_sitechars.operation_name(sym, lattice) + prime,
            rotation=np.rint(np.asarray(sym.rotation, dtype=float)).astype(int),
            translation=np.asarray(sym.translation, dtype=float).copy(),
            time_reversal=bool(sym.time_reversal),
            rotation_cartesian=np.asarray(sym.rotation_cart, dtype=float).copy(),
            spin_lift=(np.asarray(sym.spinor_rotation, dtype=complex).copy()
                       if group.spinor else None)))
    rows.sort(key=lambda row: row.ind)
    return OperationTable(rows, name=group.name, number_str=group.number_str)


def operations(group_or_rep) -> OperationTable | Refusal:
    """Every operation of a group by its index, with its Seitz symbol and its name in the model's frame.

    The index is the key every per-operation record is written on:
    ``rep.matrices``, ``validated.verdicts``, the entries of a descent.  The
    Seitz symbol ``{R|t}`` is the one the tables and the Bilbao
    Crystallographic Server print, in their conventional setting, with an
    underscore before the axis and a minus sign for an overbar: ``-4+_001`` is
    ``4̄⁺₀₀₁``.  The name is the same operation in the model's Cartesian frame,
    the frame ``characters=`` is written in.

    Parameters
    ----------
    group_or_rep : RealizedGroup, GroupData or SpaceGroupRep
        A group, every operation of which is a row, or a representation, whose
        rows are the operations it represents.  ``r``, ``r.group`` and
        ``r.rep`` may each be handed in, and ``r.operations`` is the same
        table.

    Returns
    -------
    OperationTable or Refusal
        ``table[ind]`` is one :class:`Operation`, with its ``seitz``, its
        ``name``, its matrices and whether it carries time reversal;
        ``print(table)`` lists them all.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        The argument is no group, representation or record of one.

    Refusals
    --------
    A refusal handed in as ``group_or_rep`` is returned as it is.

    ``tables_not_ready``
        The group has no conventional setting to write a Seitz symbol in.
    """
    return _operation_rows(group_or_rep)


def seitz(group_or_rep, ind) -> str | Refusal:
    """The Seitz symbol of one operation, by its index: the one :func:`operations` writes in its row.

    Parameters
    ----------
    group_or_rep : RealizedGroup, GroupData or SpaceGroupRep
        The group, or a representation of it.
    ind : int
        The operation's index.

    Returns
    -------
    str or Refusal
        The symbol, ``{3+_001|0,0,0}``.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    KeyError
        No operation of the group, or of the representation handed in, has
        that index.
    TypeError
        The argument is no group, representation or record of one.

    Refusals
    --------
    A refusal handed in as ``group_or_rep`` is returned as it is.

    ``tables_not_ready``
        The group has no conventional setting to write a Seitz symbol in.
    """
    rows = _operation_rows(group_or_rep)
    if isinstance(rows, Refusal):
        return rows
    if int(ind) in rows:
        return rows[int(ind)].seitz
    carried = list(rows)
    shown = ", ".join(str(i) for i in carried[:12]) + (", ..." if len(carried) > 12 else "")
    raise KeyError(f"no operation #{int(ind)} in {rows.name} #{rows.number_str}; "
                   f"the {len(carried)} indices carried here are {shown}")


def operation_index(group_or_rep, seitz) -> int | Refusal:
    """The index of the operation a Seitz symbol names.

    A rotation written about either direction of its axis, ``4+`` about
    ``[001]`` or ``4-`` about ``[00-1]``, and any representative of its
    translation modulo the lattice name the same operation, so a symbol copied
    off a printed table is read whichever choices it makes; spaces and the
    underscore before the axis are free.

    Parameters
    ----------
    group_or_rep : RealizedGroup, GroupData or SpaceGroupRep
        The group, or a representation of it, whose own operations are then
        the ones searched.
    seitz : str
        A Seitz symbol ``{R|t}``, with a trailing ``'`` for an operation
        combined with time reversal.

    Returns
    -------
    int or Refusal
        The operation's index.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    TypeError
        The argument is no group, representation or record of one.

    Refusals
    --------
    A refusal handed in as ``group_or_rep`` is returned as it is.

    ``tables_not_ready``
        The group has no conventional setting to read a Seitz symbol in.
    ``unknown_operation``
        No operation of the group is written so; the refusal names the nearest
        spellings it carries.
    """
    rows = _operation_rows(group_or_rep)
    if isinstance(rows, Refusal):
        return rows
    group, _ = _operation_argument(group_or_rep)
    parsed = _parse_seitz(seitz, _centering_of(group))
    if parsed is not None:
        wanted, wanted_tr = parsed
        kept = {row.ind for row in rows.values()}
        for sym in group.symmetries:
            if int(sym.ind) not in kept:
                continue
            if _matches(_seitz_key(sym, group), bool(sym.time_reversal),
                        wanted, wanted_tr):
                return int(sym.ind)
    carried = [row.seitz for row in rows.values()]
    near = difflib.get_close_matches(str(seitz), carried, n=3, cutoff=0.4) or carried[:3]
    return Refusal(
        "unknown_operation",
        f"no operation of {rows.name} #{rows.number_str} is written {str(seitz)!r}. "
        f"The nearest spellings it carries are {', '.join(near)}. A symbol is the "
        "conventional-setting `{R|t}`: the rotation part `1`, `-1`, `m`, `2`, `3`, "
        "`4`, `6`, `-3`, `-4` or `-6`, a `+` or `-` for the sense of a rotation of "
        "order above two, the direction of the axis after an underscore, then the "
        "translation as three fractions -- `{-4+_001|0,0,1/2}` -- with a trailing "
        "`'` where the operation is antiunitary. `operations(group_or_rep)` prints "
        "every symbol this group carries against the index it is keyed on.",
        context={"number_str": str(rows.number_str), "seitz": str(seitz),
                 "nearest": tuple(near), "carried": tuple(carried)})


# --------------------------------------------------------------------------
# little groups and irrep identification
# --------------------------------------------------------------------------

def little_group(group: GroupData, k) -> LittleGroup | Refusal:
    """The operations that leave ``k`` invariant up to a reciprocal lattice vector.

    The group a band's corep at ``k`` is a corep of: only these operations act
    within the states there.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    k : array, shape (dim_k,)
        The point, in the model's reduced coordinates.  A layer takes a point
        of its three-dimensional cell too, where it lies in the plane.

    Returns
    -------
    LittleGroup or Refusal
        Its unitary operations in index order, the columns a character vector
        at ``k`` is written in, and those combined with time reversal, which
        carry no characters; ``summary()`` prints it.  A ``Refusal`` in its
        place is falsy.

    Raises
    ------
    ValueError
        ``k`` has neither ``dim_k`` components nor three.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``out_of_plane``
        A point with a component along a layer's vacuum, a direction its zone
        does not have.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    k = np.asarray(k, dtype=float).ravel()
    if k.size == 3 and group.dim_k != 3:
        if not group.in_plane(k[None, :]):
            return Refusal("out_of_plane",
                           f"k = {np.round(k, 6)} has a component out of the model's periodic "
                           f"plane (periodic directions {group.periodic_dirs}); a slab model "
                           f"has no dispersion there.",
                           value=float(np.max(np.abs(k[list(group.vacuum_dirs)]))),
                           threshold=0.0, context={"k": k, "periodic_dirs": group.periodic_dirs})
        k3 = k
        k = group.k_from_3d(k3[None, :])[0]
    elif k.size == group.dim_k:
        k3 = group.k_to_3d(k[None, :])[0]
    else:
        raise ValueError(f"k must have {group.dim_k} (or 3) components, got {k.size}")

    uni, anti = [], []
    for sym in group.symmetries:
        G = sym.transform_k(k3) - k3
        if not np.allclose(G, np.round(G), atol=1e-8):
            continue
        (anti if sym.time_reversal else uni).append(sym)
    uni.sort(key=lambda s: s.ind)
    anti.sort(key=lambda s: s.ind)
    Gs = [np.round(s.transform_k(k3) - k3) for s in uni]
    return LittleGroup(k=k, k3=k3, unitary=uni, antiunitary=anti,
                       indices=tuple(int(s.ind) for s in uni),
                       G=np.array(Gs) if Gs else np.zeros((0, 3)))


def _snap_rational(x, max_den=24, atol=1e-4):
    """Snap a coordinate to the nearest small-denominator rational, if one is close.

    ``irreptables`` stores high-symmetry k-points to six decimals, so a
    tabulated ``K`` arrives as ``0.333333`` — 6.7e-7 away from ``1/3``.  That
    is harmless for a table lookup and *not* harmless for evaluating a model
    there: a Dirac point evaluated 7e-7 off K opens a spurious gap of the same
    order, and a mesh-membership test at 1e-8 declares a mesh that does contain
    K to miss it.  High-symmetry coordinates are rationals with small
    denominators, so snapping is exact recovery, not rounding.
    """
    x = np.asarray(x, dtype=float)
    out = x.copy()
    for idx, v in np.ndenumerate(x):
        best, err = v, atol
        for q in range(1, max_den + 1):
            p = round(v * q)
            cand = p / q
            if abs(cand - v) < err:
                best, err = cand, abs(cand - v)
        out[idx] = best
    return out


def _table_kpoints(group: GroupData) -> dict:
    """``{kpname: k in the model's reduced 3D coordinates}`` from ``irreptables``.

    The irrep table's points, and, for a group without time reversal, the
    partners ``−k`` its elementary band representations list on their own
    (:func:`_partner_points`).
    """
    from irreptables.irreps import IrrepTable
    table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic)
    inv = np.linalg.inv(group._refUC.T)
    partner = ({} if group.magnetic else
               {base: arm for arm, base, _naming
                in _partner_points(group.number_str, group.spinor)})
    listed: dict[str, np.ndarray] = {}
    for irr in table.irreps:
        if irr.kpname in listed:
            continue
        listed[irr.kpname] = np.asarray(irr.k, dtype=float)
        if irr.kpname in partner:                 # its partner -k, next to it
            listed[partner[irr.kpname]] = -listed[irr.kpname]
    return {name: _snap_rational((inv @ _snap_rational(k)) % 1.0)
            for name, k in listed.items()}


@lru_cache(maxsize=None)
def _partner_points(number_str: str, spinor: bool) -> tuple:
    r"""The points ``−k`` a unitary group's EBR rows list and its irrep table does not.

    Without time reversal, ``k`` and ``−k`` are different points of the zone
    wherever no operation of the group carries one onto the other, and the
    tables' elementary band representations carry rows at both: ``KA``, ``HA``,
    ``PA`` and ``WA`` beside ``K``, ``H``, ``P`` and ``W``.  The irrep table
    lists only the first.  The little group at ``−k`` is the one at ``k`` and
    its characters are the conjugates, the Bloch phases conjugating with them,
    so the partner's table is the listed point's, conjugated
    (:func:`_transported_table`); what is read here is which conjugate each
    partner label names.

    An elementary band representation induced from a site irrep ``ρ`` carries
    at ``−k`` the conjugate of what the one induced from ``ρ*`` carries at
    ``k``: the induced character's phases :math:`e^{-2\pi i\,k\cdot m}`
    conjugate and so does the site character.  So each partner label is the
    conjugate of the listed label whose row, read on the time-reversed columns,
    is its row: :math:`m_j(XA_{\pi(n)}) = m_{\bar j}(X_n)` for every column
    ``j``, ``j̄`` the column of the conjugate site irrep at the same position
    and ``j`` itself for a real one.  The tables name a partner's irreps as the
    conjugates of the listed point's, ``XA_n`` of ``X_n``, and that naming is
    taken wherever the rows hold it, which includes labels whose rows the
    columns do not tell apart; a different correspondence is taken only where
    it is the one the rows hold.

    Returns
    -------
    tuple
        ``((partner, listed point, ((listed label, partner label), ...)), ...)``
        in the physical convention of :func:`ebr_data`; empty for a table whose
        EBR rows list no such point or that ships no EBR rows.

    Raises
    ------
    ..refusal.InternalError
        No correspondence reproduces the rows, or several do and the tables'
        naming is not among them: the shipped rows and their conjugation
        disagree, which is a defect, never a label to guess.
    """
    from irreptables.ebrs import load_ebr_data
    from irreptables.irreps import IrrepTable
    try:
        raw = load_ebr_data(number_str, spinor)
    except Exception:                             # no EBR rows: nothing lists a partner
        return ()
    labels = [str(label) for label in raw["basis"]["irrep_labels"]]
    listed = {irr.kpname for irr in IrrepTable(number_str, spinor, magnetic=False).irreps}
    arms = sorted({_label_kpoint(label) for label in labels} - listed)
    if not arms:
        return ()
    physical = _to_physical_convention(raw, number_str, spinor, False)
    labels = [str(label) for label in physical["basis"]["irrep_labels"]]
    matrix = np.array([e["vector"] for e in physical["ebrs"]], dtype=int).T
    partner = list(_ebr_transport(number_str, spinor, False)[2])
    index = {label: i for i, label in enumerate(labels)}
    out = []
    for arm in arms:
        base = arm[:-1] if len(arm) > 1 and arm.endswith("A") else ""
        theirs = [label for label in labels if _label_kpoint(label) == arm]
        ours = [label for label in labels if _label_kpoint(label) == base]
        if base not in listed or len(theirs) != len(ours):
            raise InternalError(
                f"#{number_str}: the EBR rows list {arm}, which the irrep table does "
                "not, and it is not the partner -k of a point the table lists with as "
                "many irreps, so its labels cannot be read off any listed point.")
        fits = {n: [a for a in theirs
                    if np.array_equal(matrix[index[a]], matrix[index[n], partner])]
                for n in ours}
        named = {n: n.replace(base, arm, 1) for n in ours}
        if all(named[n] in fits[n] for n in ours):
            pi = named
        else:
            every = [dict(zip(ours, pick)) for pick in itertools.product(*fits.values())
                     if len(set(pick)) == len(ours)]
            if len(every) != 1:
                raise InternalError(
                    f"#{number_str} at {arm}: {len(every)} correspondences between its "
                    f"labels and the conjugates of {base}'s reproduce the EBR rows, and "
                    f"the tables' naming ({arm}n the conjugate of {base}n) is not among "
                    "them. The rows at a point -k and their conjugation disagree; that "
                    "is a defect in the shipped rows or in this reading of them, not a "
                    "label to choose.")
            pi = every[0]
        out.append((arm, base, tuple(sorted(pi.items()))))
    return tuple(out)


class HighSymmetryPoints(dict):
    """The tabulated high-symmetry points of the model's zone, by name, in the model's reduced coordinates.

    What :func:`high_symmetry_points` returns, a mapping ``{name: k}``;
    ``summary()`` prints it.

    Attributes
    ----------
    missing : tuple of str
        The tabulated points out of a layer's plane, ``A``, ``H`` and ``L`` of
        a hexagonal group, which its zone does not have at any mesh: a symmetry
        vector read on this mapping alone is partial there.
    """

    __slots__ = ("missing",)

    def __init__(self, points=(), *, missing=()):
        super().__init__(points)
        #: The tabulated names out of the periodic plane, in the tables' order.
        self.missing = tuple(missing)

    def summary(self) -> str:
        """The points, a line for each part, with the attribute that holds it."""
        lines = [f"{len(self)} point{'' if len(self) == 1 else 's'}, by name, in the model's "
                 "reduced coordinates   ([name])"]
        lines += [f"    {name} {_point_text(k)}" for name, k in self.items()]
        if self.missing:
            lines.append(f"  out of the layer's plane: {', '.join(self.missing)}   (.missing)")
        return "\n".join(lines)


def high_symmetry_points(group: GroupData) -> HighSymmetryPoints | Refusal:
    """The tabulated high-symmetry points, in the model's own reduced coordinates.

    The points a symmetry vector is read at.  The tables write them in their
    conventional cell, generally as other numbers: graphene's ``K`` is
    ``(1/3, 1/3, 0)`` there and ``(1/3, 2/3)`` in pythtb's honeycomb cell.
    Without time reversal a point and its partner ``−k`` are two points
    wherever no operation carries one onto the other, ``KA`` beside ``K``, and
    both are returned, the partner's irreps the complex conjugates of the
    point's.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.

    Returns
    -------
    HighSymmetryPoints or Refusal
        ``{name: k}``, each ``k`` with ``dim_k`` components; its ``missing``
        names the tabulated points out of a layer's plane.  ``summary()``
        prints it.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell, to
        carry their points into the model's coordinates.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready", group._setup_note or "irrep tables are not set up "
                       "for this cell", context={"number_str": group.number_str})
    out, missing = {}, []
    for name, k3 in _table_kpoints(group).items():
        if group.in_plane(k3[None, :], atol=1e-9):
            out[name] = group.k_from_3d(k3[None, :])[0]
        else:
            missing.append(name)
    return HighSymmetryPoints(out, missing=missing)


def _resolve_kpname(group: GroupData, lg: LittleGroup, kpname, ktol):
    """The tabulated label of the model's k, or a Refusal listing the options."""
    kmap = _table_kpoints(group)
    if kpname is not None:
        return kpname
    for name, k3 in kmap.items():
        d = lg.k3 % 1.0 - k3
        d -= np.round(d)
        if np.abs(d).max() <= ktol:
            return name
    avail = {n: np.round(group.k_from_3d(v[None, :])[0], 6)
             for n, v in kmap.items() if group.in_plane(v[None, :])}
    return Refusal(
        "kpoint_not_tabulated",
        f"k = {np.round(lg.k, 6)} is not one of the tabulated high-symmetry points of "
        f"#{group.number_str} in this cell. Tabulated (in this cell's reduced "
        f"coordinates): {avail}. A point in the star of a tabulated one but not the "
        f"tabulated arm carries a conjugate little group and is not transported here.",
        context={"k": lg.k, "available": avail, "all_points": kmap})


#: Centering translations of each Bravais lattice, in the BCS **conventional**
#: basis the tables use, keyed by the leading letter of the table's
#: Hermann–Mauguin name (``IrrepTable.name``).  This is the one piece of
#: crystallography that cannot be read off a character table, and it is needed
#: because "``g`` fixes ``k``" and "``t`` is a lattice vector" are statements
#: about the *primitive* lattice while the tables are written in the
#: conventional one.  Checked, not trusted: the little groups it predicts are
#: asserted equal to ``irreptables``' own tabulated little-group indices at all
#: 2700 (space group, valuedness, k-point) entries by the sweep in
#: ``tests/test_groups.py``.
_CENTERINGS = {
    "P": (),
    "A": ((0.0, 0.5, 0.5),),
    "B": ((0.5, 0.0, 0.5),),
    "C": ((0.5, 0.5, 0.0),),
    "I": ((0.5, 0.5, 0.5),),
    "F": ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
    "R": ((2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)),
}


def _centering(table) -> tuple:
    """The centering translations of ``table``'s Bravais lattice."""
    letter = str(table.name)[:1]
    if letter not in _CENTERINGS:
        raise _EBRTransportError(
            f"#{table.number_str}'s tabulated name {table.name!r} does not start "
            "with a Bravais lattice letter, so its centering translations — and "
            "hence its little groups — cannot be determined.")
    return _CENTERINGS[letter]


def _is_translation(v, centering) -> bool:
    """``v`` (conventional reduced) is a direct-lattice translation."""
    v = np.asarray(v, dtype=float)
    for c in ((0.0, 0.0, 0.0),) + tuple(centering):
        d = v - np.asarray(c, dtype=float)
        if np.abs(d - np.round(d)).max() <= _LATTICE_TOL:
            return True
    return False


def _is_reciprocal(v, centering) -> bool:
    """``v`` (conventional reduced) is a reciprocal-lattice vector.

    Integer *and* orthogonal to every centering translation — for ``I`` that is
    the ``h+k+l`` even rule, which is exactly why ``P = (½,½,½)`` of a
    body-centred group is **not** a TRIM.
    """
    v = np.asarray(v, dtype=float)
    if np.abs(v - np.round(v)).max() > _LATTICE_TOL:
        return False
    return all(abs(float(v @ np.asarray(c)) - round(float(v @ np.asarray(c))))
               <= _LATTICE_TOL for c in centering)


def _operations(table) -> tuple:
    """``(R, t, S, antiunitary)`` per tabulated symmetry, ``irreptables``' order."""
    return tuple((np.asarray(s.R, dtype=float), np.asarray(s.t, dtype=float),
                  np.asarray(s.S, dtype=complex), bool(s.time_reversal))
                 for s in table.symmetries)


def _as_representative(R, t, S, ops, centering, spinor) -> tuple:
    r"""Write a unitary ``{R|t}`` (spinor part ``S``) as ``σ·{E|m}·g_ℓ``.

    Returns ``(ℓ, m, σ)`` with ``ℓ`` a **one-based** ``irreptables`` symmetry
    index, ``m`` the lattice translation and ``σ = ±1`` the double-group sign
    (always ``+1`` for a single-valued table).  Raises when the element is not
    in the tabulated group at all, which nothing upstream can provoke.
    """
    for j, (Rj, tj, Sj, anti) in enumerate(ops):
        if anti or np.abs(R - Rj).max() > 1e-6:
            continue
        m = t - tj
        if not _is_translation(m, centering):
            continue
        if not spinor:
            return j + 1, m, 1.0
        if np.abs(S - Sj).max() < _SU2_TOL:
            return j + 1, m, 1.0
        if np.abs(S + Sj).max() < _SU2_TOL:
            return j + 1, m, -1.0
        break
    raise _EBRTransportError(                          # pragma: no cover - guard
        "a product of tabulated operations is not a tabulated operation times a "
        "lattice translation; the shipped symmetry list is not a group.")


def _herring_data(table, k, centering) -> tuple:
    r"""The convention-free half of the Wigner–Herring test at ``k``.

    Returns ``(H, coset, squares, twist)``:

    * ``H`` — one-based indices of the unitary little group of ``k``,
      ``R^{-T}k ≡ k`` modulo the **reciprocal** lattice;
    * ``coset`` — indices of the antiunitary elements with ``-R^{-T}k ≡ k``,
      i.e. of the extended little group's antiunitary coset.  Empty when no
      antiunitary fixes ``k`` at all (the ``K``/``KA`` "other arm" situation),
      and otherwise of size ``|H|``;
    * ``squares`` — ``(ℓ, m, σ)`` per ``a ∈ coset`` with
      ``a² = σ·{E|m}·g_ℓ``.  With ``a = uΘ`` and ``Θ S Θ⁻¹ = S`` (SU(2) is
      pseudo-real) this is ``a² = u²·Θ²``, so ``σ`` carries both the SU(2)
      accumulation of ``u²`` **and** ``Θ² = −1`` for a double-valued table;
    * ``twist`` — ``(ℓ, m, σ)`` per ``h ∈ H`` writing ``a₀⁻¹ h a₀`` (``a₀`` the
      first coset element) the same way.  ``Θ`` drops out of this one entirely,
      so it is the plain conjugation by ``a₀``'s unitary part.

    Nothing here depends on the ``e^{±ik·t}`` convention: the convention enters
    only when ``m`` is turned into the phase ``e^{s·2πi k·m}``.
    """
    ops = _operations(table)
    spinor = bool(table.spinor)
    k = np.asarray(k, dtype=float)
    H, coset = [], []
    for i, (R, _t, _S, anti) in enumerate(ops):
        d = np.linalg.inv(R).T @ k
        if _is_reciprocal((-d if anti else d) - k, centering):
            (coset if anti else H).append(i + 1)
    squares, twist = {}, {}
    if coset:
        eps = -1.0 if spinor else 1.0                  # Θ² = −1 for spin ½
        for a in coset:
            R, t, S, _ = ops[a - 1]
            ell, m, sig = _as_representative(R @ R, t + R @ t, S @ S,
                                             ops, centering, spinor)
            squares[a] = (ell, m, eps * sig)
        R0, t0, S0, _ = ops[coset[0] - 1]
        inv = np.linalg.inv(R0)
        for h in H:
            Rh, th, Sh, _ = ops[h - 1]
            ell, m, sig = _as_representative(
                inv @ Rh @ R0, inv @ (Rh @ t0 + th - t0),
                np.linalg.inv(S0) @ Sh @ S0, ops, centering, spinor)
            twist[h] = (ell, m, sig)
    return tuple(H), tuple(coset), squares, twist


def _corep_gluing(chars, k, H, squares, twist, s):
    r"""The corep list at one k-point, in the convention ``s`` names.

    ``s = +1`` is BCS's convention and ``s = −1`` the physical one; ``chars``
    maps each unitary little-group irrep label to its characters
    ``{ind: χ}`` **in that convention**.  Returns a sorted list of
    ``(constituent labels, type)`` with the Wigner type ``'a'``/``'b'``/``'c'``,
    or ``None`` when the data does not support the classification.

    The two ingredients, both evaluated from the table's own numbers:

    .. math::
        \bar D(h) = \overline{D(a_0^{-1} h a_0)},\qquad
        I_s(D) = \frac{1}{|H|}\sum_{a}\chi_D(a^2),

    with every ``{E|m}`` in either expression contributing ``e^{s·2πi k·m}``.
    ``I_s = +1`` (``\bar D ≅ D``) is type **a**, the corep is ``D`` alone;
    ``I_s = −1`` is type **b**, ``D ⊕ D``; ``I_s = 0`` (``\bar D ≇ D``) is type
    **c**, ``D ⊕ \bar D``.  Because *every* ``m`` here carries the convention,
    the corep list can — and at a minority of the tabulated k-points does — differ
    between ``s = +1`` and ``s = −1``: that is the fourth face of the convention
    artifact, and it changes which irrep doubles, hence the corep
    **dimensions**.
    """
    phase = lambda m: np.exp(s * 2j * np.pi * float(np.asarray(k) @ m))  # noqa: E731
    ind = {}
    for name, row in chars.items():
        ind[name] = sum(sig * phase(m) * row[ell]
                        for ell, m, sig in squares.values()) / len(H)
    partner = {}
    for name, row in chars.items():
        bar = {h: np.conj(sig * phase(m) * row[ell])
               for h, (ell, m, sig) in twist.items()}
        hits = [n for n, r in chars.items()
                if max(abs(bar[h] - r[h]) for h in H) < _CHAR_TOL]
        if len(hits) != 1:
            return None
        partner[name] = hits[0]
    out, seen = [], set()
    for name in sorted(chars):
        if name in seen:
            continue
        w = ind[name]
        if partner[name] == name:
            if abs(abs(w) - 1.0) > 0.2 or abs(w.imag) > 0.2:
                return None
            out.append(((name,), "a") if w.real > 0 else ((name, name), "b"))
            seen.add(name)
        else:
            if abs(w) > 0.2 or partner[partner[name]] != name:
                return None
            out.append((tuple(sorted((name, partner[name]))), "c"))
            seen.update((name, partner[name]))
    return sorted(out)


def _corep_name(parts, double: bool) -> str:
    """``('-A7', '-A8') → '-A7A8'`` — ``irreptables``' own composite-name pattern.

    The constituents are concatenated in **lexicographic** order of their bare
    names (which is why the tables print ``-A11A9``, not ``-A9A11``) with the
    double-valued ``-`` marker carried once at the front.  The rule reproduces
    every one of the 2700 shipped grey corep names from its own constituents,
    which the sweep in ``tests/test_groups.py`` checks.
    """
    bare = sorted(p[1:] if str(p).startswith("-") else str(p) for p in parts)
    return ("-" if double else "") + "".join(bare)


def _decompose_into(chars, H, vector):
    """Non-negative integer multiplicities of ``chars`` in ``vector``, or ``None``."""
    names = sorted(chars)
    a = np.array([[complex(chars[n][h]) for h in H] for n in names]).T
    b = np.array([complex(vector[h]) for h in H])
    n, *_ = np.linalg.lstsq(a, b, rcond=None)
    if (np.abs(a @ n - b).max() > _CHAR_TOL
            or np.abs(n - np.round(n.real)).max() > _CHAR_TOL
            or np.round(n.real).min() < -0.5):
        return None
    return tuple(x for name, m in zip(names, np.round(n.real).astype(int))
                 for x in [name] * int(m))


@lru_cache(maxsize=None)
def _parent_table(number_str: str, spinor: bool):
    """The unitary (``magnetic=False``) table of a grey table's parent group."""
    from irreptables.irreps import IrrepTable
    return IrrepTable(str(number_str).split(".")[0], spinor, magnetic=False)


@lru_cache(maxsize=None)
def _unitary_conjugation(number_str: str, spinor: bool) -> tuple:
    """:func:`_table_conjugation` of the parent unitary table, cached and read-only."""
    return _table_conjugation(_parent_table(number_str, spinor))


def _lifted_conjugation(table) -> dict:
    r"""``{k-point: {corep label: physical corep label}}`` for a **grey** table.

    The whole fourth mechanism in one function.  A grey corep is a *composite*
    object — its BCS name spells out the unitary little-group irreps it is glued
    from — so its physical name is not a free choice: it is fixed by the
    physical names of its constituents, which the unitary table's own
    conjugation map ``c`` already decides.  Concretely, per tabulated k-point:

    1. decompose each shipped corep's characters into the parent unitary
       table's irrep characters (BCS convention, both tables' shipped numbers) —
       the gluing, read off rather than assumed;
    2. check it against the **BCS Herring indicators** computed from the table's
       own operations (:func:`_corep_gluing` at ``s = +1``).  A shipped table
       that violates its own convention's indicators is a data defect and is
       reported by returning nothing for that k-point;
    3. map every constituent through the unitary ``c`` and re-glue, naming the
       glued corep by :func:`_corep_name`;
    4. check *that* against the **physical Herring indicators** (``s = −1``),
       which is the independent confirmation that the re-gluing is right.

    Step 3 is exact rather than heuristic because
    ``χ_phys[c(D)] = conj(χ_BCS[D])`` gives, term by term in the Herring sum,
    ``I_{-1}(c(D)) = conj(I_{+1}(D)) = I_{+1}(D)``: the physical corep list is
    always the ``c``-image of the BCS one.  Where the image label is not a
    shipped label, the point has been **re-glued**: the same unitary irreps are
    partitioned differently, so the corep a given irrep sits in has a different
    dimension, and the EBR basis at that point changes with it.

    Because the unitary ``c`` is itself the character-matched naming, this lift
    usually *agrees* with matching the grey characters directly — BCS's own
    corep names spell BCS's own constituents, so the image of a
    conjugation-closed shipped set is a shipped set, which is why the re-glued
    residue is only 7 k-point entries.  It is still consulted **first**, because
    where the two differ the spelling has to win: at SG 203's spinful ``W`` the
    grey rows admit an exchange that the parent's do not, and taking it would
    give a corep a name whose constituents the unitary table does not name.

    Returns ``{}`` for a unitary table, and omits any k-point where the lift is
    not available (an "other arm" point the unitary table does not carry) or
    where a check above fails; :func:`_table_conjugation` then falls back to
    matching character vectors, or reports the point as broken.
    """
    if not getattr(table, "au_symmetries", ()):
        return {}
    parent = _parent_table(table.number_str, table.spinor)
    pops, gops = _operations(parent), _operations(table)
    if len(gops) != 2 * len(pops):
        return {}
    for (R1, t1, S1, _), (R2, t2, S2, _) in zip(gops[:len(pops)], pops):
        if (np.abs(R1 - R2).max() > 1e-6 or np.abs(t1 - t2).max() > _LATTICE_TOL
                or np.abs(S1 - S2).max() > _SU2_TOL):
            return {}                        # not the parent's grey extension
    centering = _centering(table)
    double = bool(table.spinor)
    by_kp_u: dict[str, list] = {}
    for irr in parent.irreps:
        by_kp_u.setdefault(irr.kpname, []).append(irr)
    by_kp_g: dict[str, list] = {}
    for irr in table.irreps:
        by_kp_g.setdefault(irr.kpname, []).append(irr)

    out: dict[str, dict] = {}
    c_u, _conj_u, broken_u = _unitary_conjugation(table.number_str, table.spinor)
    for kp, girreps in by_kp_g.items():
        if kp not in by_kp_u or kp in broken_u:
            continue
        uirreps = by_kp_u[kp]
        k = np.asarray(uirreps[0].k, dtype=float)
        H, coset, squares, twist = _herring_data(table, k, centering)
        if sorted(H) != sorted(int(i) for i in uirreps[0].characters):
            continue
        bcs = {irr.name: {int(i): complex(v) for i, v in irr.characters.items()}
               for irr in uirreps}
        parts = {}
        for irr in girreps:
            got = _decompose_into(bcs, H, {int(i): complex(v)
                                           for i, v in irr.characters.items()})
            if got is None or _corep_name(got, double) != irr.name:
                parts = None
                break
            parts[irr.name] = got
        if parts is None:
            continue
        if not coset:                        # nothing glued: coreps are the irreps
            if sorted(parts.values()) != sorted((n,) for n in bcs):
                continue
            out[kp] = {name: _corep_name([c_u[p] for p in got], double)
                       for name, got in parts.items()}
            continue
        if len(coset) != len(H):
            continue
        want = _corep_gluing(bcs, k, H, squares, twist, +1)   # step 2
        if want is None or (sorted(v for v, _ in want)
                            != sorted(parts.values())):
            continue                         # shipped gluing ≠ BCS's own indicators
        lift = {name: _corep_name([c_u[p] for p in got], double)
                for name, got in parts.items()}               # step 3
        physchars = {name: {h: np.conj(bcs[c_u[name]][h]) for h in H}
                     for name in bcs}
        phys = _corep_gluing(physchars, k, H, squares, twist, -1)   # step 4
        if phys is None or (sorted(_corep_name(v, double) for v, _ in phys)
                            != sorted(lift.values())):
            continue
        out[kp] = lift
    return out


def _matching(chi: dict, image) -> dict | None:
    """``{name: other}`` with ``chi[other] == image(chi[name])``, or ``None``.

    The one primitive both naming rules of :func:`_table_conjugation` are built
    from: ``image = conj`` asks which *shipped* row prints what the physical
    convention prints (the name follows the character), ``image = conj/φ`` asks
    which shipped row the tabulated content ``Γ`` carries onto (the tabulated
    labelling).  ``None`` as soon as any row has no unique partner.
    """
    out: dict[str, str] = {}
    for name, vec in chi.items():
        want = image(vec)
        hit = [other for other, w in chi.items()
               if np.abs(w - want).max() < _CHAR_TOL]
        if len(hit) != 1:
            return None
        out[name] = hit[0]
    return out


def _table_conjugation(table) -> tuple[dict, frozenset, tuple]:
    r"""Per tabulated k-point: which BCS name each **physical** row is returned under.

    Returns ``(c, conjugated, broken)``.

    * ``c`` maps each tabulated label to the label whose **physical** character
      is the conjugate of its tabulated one, ``χ_phys[c(D)] = conj(χ_BCS[D])``.
      For a grey table the image may be a corep name the tables do not ship —
      see :func:`_lifted_conjugation` and the re-gluing paragraph below.
    * ``conjugated`` is the set of k-point names where :func:`_transported_table`
      returns the whole-character conjugate rather than the phase-flipped
      tabulated row — which it must do wherever ``c`` is not the pairing that
      keeping the tabulated ``Γ`` would realize.
    * ``broken`` lists k-points where neither branch applies, i.e. where the
      tabulated data is inconsistent with itself.

    **The name follows the character.**  Write
    ``w_D = conj(χ_D)`` for the whole-character conjugate; conjugating a
    representation conjugates its factor system *and* carries BCS's
    time-reversal gluing ``{Γ, conj(Γ)φ}`` onto the physical one
    ``{conj(Γ), Γ conj(φ)}`` (``φ(g) = e^{-4πi k·t_g}``), so **the w-set is
    always the set of physical little-group characters**.  The question this
    function answers is only what to *call* each ``w_D``, and the rule is: call
    it by the BCS row it prints.  So ``c(D)`` is the shipped label ``E`` with
    ``χ_E = w_D``, whenever every ``D`` has one.  The returned printed table is
    then literally BCS's printed table — same names against the same numbers,
    so the time-reversal doubling rules read as they do in BCS's book — and the
    convention divergence shows up only in *which* row a physical band lands
    on, which is what the EBR relabelling ``n^phys_ρ[c(D)] = n^BCS_{ρ*}[D]``
    carries (:func:`_ebr_transport`).

    Where the shipped character set is **not closed under conjugation** no such
    naming exists: no permutation of BCS's rows is the physical table, and the
    label has to be anchored to something else — to what the name *spells*,
    i.e. to the constituents the parent unitary table already named
    (:func:`_lifted_conjugation`), falling back to the pairing ``v_{c(D)} =
    w_D`` that keeping the tabulated ``Γ`` realizes (``v_D = χ_D·φ``), and
    finally to the identity.  59 of the 5462 tabulated (space group,
    valuedness, k-point) entries are in that class, every one of them with at
    least one complex character; the other 5403 are returned under BCS's own
    names.

    What the 59 mean is *not* that two conventions disagree about a doubling.
    The entrywise conjugate of the returned table is the shipped table exactly
    (that is ``χ_phys[c(D)] = conj(χ_BCS[D])`` read backwards), and time
    reversal maps the ``k`` sector antiunitarily onto the ``−k`` one, so the
    shipped page there tabulates the content of the **other star member**: at
    SG 80's ``P`` the returned ``{-P3 (1,−i), -P4P4 (2,+2i)}`` conjugates to the
    shipped ``{-P4 (1,+i), -P3P3 (2,−2i)}``.  The doubling itself is
    convention-independent — it is the trace of the zero-translation
    little-group generator on the degenerate level — which is exactly why no
    relabelling is admissible there.

    Three mechanisms put a k-point in the "not the tabulated pairing" branch,
    and all are checked so that a *tabulation* defect is never mistaken for a
    convention artifact:

    1. the tabulated factor system ``ω₀(R,S) = e^{2πi (Rᵀk − k)·t_S}`` is not
       real — a 3₁/4₁/6₁-type screw at a zone-boundary point (SG 91/92/95/96
       ``A``/``Z``/``R``, 98/122 ``N``/``P``, 151-154 and 178-181 ``A``/``H``/
       ``L``, 210 ``L``/``W``, 212/213 ``M``/``R``/``X``), where keeping ``Γ``
       is not a representation of ``ω₀*`` at all;
    2. ``φ`` is not real, so BCS's time-reversal gluing of a **grey** corep is
       not the physical one (this arises only in grey tables: SG 76/78
       ``A``/``Z``, 109/110 ``P``, 144/145 ``A``, 169-172 ``A``/``H``);
    3. the **gluing itself** is convention-sensitive: the Wigner–Herring
       indicators that decide which unitary irrep doubles carry the same
       ``e^{s·2πi k·m}`` phases, so at a minority of the tabulated k-points the
       physical corep list is a *different partition* of the same unitary
       irreps — different constituents, sometimes different dimensions.  A grey
       corep's *name* is therefore never decided by character matching where the
       parent unitary table has already spoken: the
       lift (:func:`_lifted_conjugation`) is consulted first, so that the name
       always spells the constituents the unitary table names.  Where the two
       agree — the common case, because BCS's own names
       spell BCS's own constituents — the lift *is* the character
       matching; where they disagree the lift wins, which is the only way a
       renamed corep can keep meaning what it says (SG 203's spinful ``W`` is
       such a point: the grey rows admit an exchange that the parent's do not,
       so the exchange is refused there).

    A k-point where **no** naming rule applies while ``ω₀`` *and* ``φ`` are both
    real is none of the three: there the two conventions produce identical
    numbers, so a set that is neither conjugation-closed nor ``φ``-paired would
    be a defect of the shipped table and is returned in ``broken``, and
    ``broken`` is empty for all 460 unitary and all 460 grey instances.  SG 80's
    grey spinful ``P`` is the closest call: it is a mechanism-3 point, named by
    the lift, and its shipped ``-P3P3 ⊕ -P4`` is exactly what BCS's own
    indicators require in BCS's convention.
    """
    t_of = {i + 1: np.asarray(s.t, dtype=float) for i, s in enumerate(table.symmetries)}
    rot_of = {i + 1: np.asarray(s.R, dtype=float) for i, s in enumerate(table.symmetries)}
    by_kp: dict[str, list] = {}
    for irr in table.irreps:
        by_kp.setdefault(irr.kpname, []).append(irr)
    lift = _lifted_conjugation(table)

    conj: dict[str, str] = {}
    conjugated: list[str] = []
    broken: list[str] = []
    for kp, irreps in by_kp.items():
        k = np.asarray(irreps[0].k, dtype=float)
        inds = sorted(irreps[0].characters)
        if any(sorted(irr.characters) != inds for irr in irreps):
            broken.append(kp)                 # ragged little groups within one k
            continue
        phase = np.array([np.exp(-4j * np.pi * k.dot(t_of[i])) for i in inds])
        chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
               for irr in irreps}
        pairs = _matching(chi, lambda vec: np.conj(vec) / phase)
        plain = _matching(chi, np.conj)
        if kp in lift:                        # the corep naming the unitary table forces
            here = lift[kp]
        elif plain is not None:               # the name follows the printed character
            here = plain
        elif pairs is not None:               # the tabulated labelling is physical
            here = pairs
        else:
            omega = max(abs(np.sin(2 * np.pi * (rot_of[i].T @ k - k).dot(t_of[j])))
                        for i in inds for j in inds)
            if (omega <= _PHASE_REAL_TOL
                    and np.abs(phase.imag).max() <= _PHASE_REAL_TOL):
                broken.append(kp)             # no convention difference to blame
                continue
            here = {name: name for name in chi}
        conj.update(here)
        if here != pairs:                     # not the tabulated pairing
            conjugated.append(kp)
    return conj, frozenset(conjugated), tuple(sorted(set(broken)))


@lru_cache(maxsize=None)
def _transport_naming(number_str: str, spinor: bool, magnetic: bool) -> tuple:
    """``(conjugated k-points, ((label, image), ...))`` of one table.

    Cached per ``(number_str, spinor, magnetic)``: it is a property of
    ``irreptables``' tables, not of any model or cell.  The pair is returned
    together because both halves come out of the same
    :func:`_table_conjugation` pass, which loads the parent unitary table.
    """
    from irreptables.irreps import IrrepTable
    conj, conjugated, _broken = _table_conjugation(
        IrrepTable(number_str, spinor, magnetic=magnetic))
    return conjugated, tuple(sorted(conj.items()))


def _conjugated_kpoints(number_str: str, spinor: bool, magnetic: bool) -> frozenset:
    """k-point names of one table returned as whole-character conjugates."""
    return _transport_naming(number_str, spinor, magnetic)[0]


@lru_cache(maxsize=None)
def _divergent_kpoints(number_str: str, spinor: bool, magnetic: bool) -> frozenset:
    """k-point names whose returned table is **not** a permutation of BCS's rows.

    The public census of where a reader cross-checking against the Bilbao
    Crystallographic Server will not
    find the returned numbers: everywhere else the name follows the character
    (:func:`_table_conjugation`) and the returned table *is* BCS's printed table,
    so only the label a measured band lands on can differ from BCS's own
    analysis.  Here the shipped character set is not closed under conjugation,
    so no permutation of BCS's rows is the physical table at all — its
    *entrywise conjugate* is, which is the statement that the shipped page
    tabulates the content of the other star member ``−k``.

    Of the 5462 tabulated (space group, valuedness, k-point) entries, 59 are in
    this class — 25 unitary and 34 grey — every one with at least one complex
    character, and all at ``P``/``W`` of a body- or face-centred group
    (SG 24/73/80/88/98/109/110/122/141/142/199/203/206/210/214/220/227/228/230
    and their grey partners).
    """
    from irreptables.irreps import IrrepTable
    table = IrrepTable(number_str, spinor, magnetic=magnetic)
    by_kp: dict[str, list] = {}
    for irr in table.irreps:
        by_kp.setdefault(irr.kpname, []).append(irr)
    out = []
    for kp, irreps in by_kp.items():
        inds = sorted(irreps[0].characters)
        if any(sorted(irr.characters) != inds for irr in irreps):
            continue                          # ragged: already `broken`
        chi = {irr.name: np.array([complex(irr.characters[i]) for i in inds])
               for irr in irreps}
        if _matching(chi, np.conj) is None:
            out.append(kp)
    return frozenset(out)


def _transported_table(group: GroupData, kpname: str, k3) -> dict | Refusal:
    r"""``{label: {ind: character}}`` on **this cell's** coset representatives.

    ``SpaceGroupIrreps.get_irreps_from_table`` re-implemented, with the
    **factor system conjugated** at the boundary, reading the tabulated
    characters out of ``irreptables``.

    *The convention statement.*  BCS — and therefore ``irrep``/``irreptables`` —
    tabulates little-group representations in the historically **unphysical**
    convention in which a lattice translation is represented by
    :math:`e^{+2\pi i\,k\cdot t}`.  The physical convention, the one
    :mod:`..sewing` measures in (``(g f)(r) = f(g^{-1}r)`` acting on Bloch
    functions :math:`\psi_k \sim e^{+ik\cdot r}`, so
    :math:`\psi_k(r-m) = e^{-2\pi i\,k\cdot m}\psi_k(r)`), represents it by
    :math:`e^{-2\pi i\,k\cdot t}`.  The mismatch is a historical artifact of the
    tables and lives **entirely in the projective structure**: the tabulated
    character factorises as

    .. math::
        \chi_\text{table}(\{R|t_\text{table}\})
            = e^{+2\pi i\,k_\text{ref}\cdot t_\text{table}}\;\Gamma(R),

    where :math:`\Gamma(R)` — the loaded, point-group half — is gauge-free.  The
    boundary rule is therefore *conjugate the factor system and nothing else*:
    keep :math:`\Gamma(R)` exactly as tabulated and re-attach the **physical**
    phase on this cell's own coset representative,

    .. math::
        \chi_\text{phys}(\{R|t_\text{phys}\})
            = e^{-2\pi i\,k_\text{ref}\cdot t_\text{phys}}\;\Gamma(R)
            = \chi_\text{table}\;\texttt{sym.sign}\;
              e^{-2\pi i\,(t_\text{table} + t_\text{phys})\cdot k_\text{ref}},

    with :math:`t_\text{phys}` from ``sym.translation_refUC(refUC, shiftUC)``
    and everything on the
    right in the BCS conventional basis.  ``k`` itself, the star/k-point
    matching, the time-reversal corep pairing and the tabulated point-group
    content are all left untouched: flipping ``k`` instead would repair the same
    multiplicity defect while exchanging distinct stars (SG 199's ``P`` and
    ``PA``, distinct exactly because the group has no inversion) and disturbing
    the corep pairing.

    The setting transformation is ``irrep``'s own.  ``sym.sign`` is ``+1`` on
    every operation of a group this package built: ``irrep`` fits a ``±1`` per
    operation from an unseeded random start and carries it on the *characters*,
    while :func:`_pin_spinor_branch` computes it from the tables' own spin frame
    (:func:`_canonical_spinor_signs`) and applies it to the **lifts** instead.
    That is what makes the returned *names* reproducible and the returned *rows*
    BCS's own: the little group's stored lifts then realize the tabulated ℤ₂
    factor system exactly, so the tabulated character needs no bar at all.
    Against
    ``irrep``'s formula (``χ_table · sign · exp(+2πi·δt·k_ref)`` with
    ``δt = t_table − t_phys``) this differs by ``exp(-4πi·t_table·k_ref)``,
    which is 1 whenever ``k·t_table`` is a half-integer — i.e. everywhere except
    a k-point that is not time-reversal invariant *and* whose little group
    carries fractional translations.

    *Where keeping* ``Γ`` *is not available: whole-character conjugation.*  The
    rule above keeps the tabulated ``Γ`` and re-attaches the physical phase,
    which is a
    representation of the physical factor system :math:`\omega_0^*` only where
    :math:`\omega_0` is **real**, and a physical *corep* only where the phase
    :math:`\varphi(g) = e^{-4\pi i k\cdot t_g}` that carries BCS's
    time-reversal gluing into the physical one is real as well.  Both conditions have the
    same visible consequence, and it is the one tested here (via
    :func:`_table_conjugation`): the tabulated set of character vectors is
    closed under conjugation-composed-with-``φ`` exactly when the tabulated
    labelling already names the physical representations.  Where it is not,
    what is returned is the **whole-character conjugate**

    .. math::
        \chi_\text{phys}(\{R|t_\text{phys}\})
            = \overline{\Gamma(R)}\;e^{-2\pi i\,k_\text{ref}\cdot t_\text{phys}}
            = \overline{\chi_\text{table}(\{R|t_\text{phys}\})},

    under the name :math:`c(D)` that :func:`_table_conjugation` supplies.
    :math:`D\mapsto D^*` is the canonical bijection from the
    :math:`\omega_0`-projective representations to the
    :math:`\omega_0^*`-projective ones (conjugating a representation conjugates
    its factor system, and it carries BCS's gluing
    :math:`\{\Gamma,\overline{\Gamma}\varphi\}` onto the physical one
    :math:`\{\overline\Gamma, \Gamma\overline\varphi\}`).

    *And the name follows the character.*  ``c(D)`` is the **BCS label of the
    row returned here** wherever the shipped character set is closed
    under conjugation, so at those points the returned table is BCS's printed
    table verbatim, whichever branch was taken; only where it is not closed is
    the label anchored to what the name spells instead — for a **grey** table
    to whatever the parent unitary table's relabelling forces on the corep's
    constituents, which at the re-glued points is a corep the tables do not
    ship at all (SG 80's ``-P3P3 ⊕ -P4`` becomes ``-P3 ⊕ -P4P4``: same four
    units of dimension, different partition).

    Over the 5462 tabulated (space group, valuedness, k-point) entries — 460
    unitary and 460 grey instances, with ``broken`` empty —
    **685** are returned by conjugating the whole character (398 unitary, 287
    grey), and at **59** of those the printed rows are not a permutation of
    BCS's shipped rows at all (25 unitary, 34 grey) — the only entries where a
    reader cross-checking against the Bilbao Crystallographic Server will not
    find the returned numbers under
    the returned name, and the only ones where a returned corep name can be one the
    tables do not ship (7 entries; SG 80's spinful ``P`` is the anchor).  At the
    other 5403 the printed table is BCS's — **element by element**, because the
    spinor matching sign of :func:`_canonical_spinor_signs` lives in the lift
    and not on the character, so the row printed under a returned name is a row
    ``irreptables`` ships under that name, entry for entry, with only the
    physical translation phase re-attached.  Carried on the character instead it would
    print a bar wherever the stored lifts realize a different ℤ₂ factor system
    from the tables' (``K`` of ``#183.186``: ``-K4`` as ``(+i, -i, -i)`` over
    the three mirrors against BCS's ``(-i, -i, -i)``).

    *A partner point* ``−k`` *the character table does not list* (``KA`` and
    the like, :func:`_partner_points`) is read as its listed point's table
    conjugated: the little group is the same, and the conjugate of a physical
    irrep at ``k`` is one at ``−k``.

    Refusals
    --------
    ``kpoint_not_tabulated``
        The label and the coordinates disagree, or the label is not tabulated:
        both are verdicts on what the shipped tables carry.
    """
    from irreptables.irreps import IrrepTable
    # A POINT -k THE IRREP TABLE DOES NOT LIST: its little group is the listed
    # point's, and its characters are the listed point's conjugated, each under
    # the partner label the EBR rows give the conjugate (`_partner_points`)
    if not group.magnetic:
        for arm, base, naming in _partner_points(group.number_str, group.spinor):
            if kpname != arm:
                continue
            rows = _transported_table(group, base, (-np.asarray(k3, dtype=float)) % 1.0)
            if isinstance(rows, Refusal):
                return rows
            naming = dict(naming)
            return {naming[label]: {ind: complex(np.conj(c)) for ind, c in row.items()}
                    for label, row in rows.items()}
    table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic)
    sg = group._sg
    conjugated = kpname in _conjugated_kpoints(group.number_str, group.spinor,
                                               group.magnetic)
    naming = dict(_transport_naming(group.number_str, group.spinor,
                                    group.magnetic)[1])
    k2 = np.round(np.asarray(k3, dtype=float), 5) % 1
    out: dict[str, dict[int, complex]] = {}
    for irr in table.irreps:
        if irr.kpname != kpname:
            continue
        k_ref = np.asarray(irr.k, dtype=float)
        k1 = np.round(np.linalg.inv(group._refUC.T).dot(k_ref), 5) % 1
        if not np.allclose(k1, k2):
            return Refusal(
                "kpoint_not_tabulated",
                f"the kpoint {k2} does not correspond to the point {kpname} "
                f"({np.round(k_ref, 3)} in refUC / {k1} in primUC) in the table",
                context={"kpname": kpname, "k": k2,
                         "number_str": str(group.number_str)})
        row: dict[int, complex] = {}
        for i, (sym, sym_table) in enumerate(zip(sg.symmetries, table.symmetries)):
            ind = i + 1
            if ind not in irr.characters:
                continue                      # not in the little group (or antiunitary)
            t_table = np.asarray(sym_table.t, dtype=float)
            t_ours = sym.translation_refUC(group._refUC, group._shiftUC)
            #  Γ(R) = χ_table · e^{-2πi k·t_table};  χ_phys = Γ · e^{-2πi k·t_ours}
            if conjugated:                    # ... with Γ → conj(Γ), name c(D)
                row[ind] = (np.conj(complex(irr.characters[ind])
                                    * np.exp(-2j * np.pi * t_table.dot(k_ref)))
                            * sym.sign * np.exp(-2j * np.pi * t_ours.dot(k_ref)))
            else:
                row[ind] = (complex(irr.characters[ind]) * sym.sign
                            * np.exp(-2j * np.pi * (t_table + t_ours).dot(k_ref)))
        out[naming.get(irr.name, irr.name) if conjugated else irr.name] = row
    if not out:
        return Refusal(
            "kpoint_not_tabulated",
            f"the k-point with name {kpname} is not found in the spacegroup "
            f"{table.number_str}",
            context={"kpname": kpname, "k": k2,
                     "number_str": str(group.number_str)})
    return out


def corep_table(group: GroupData, k, *, kpname=None, ktol=1e-5) -> dict | Refusal:
    """The character table of the little group at a high-symmetry point: each corep's character on each unitary operation.

    The table measured characters are read against, in the package's
    convention for a lattice translation, so characters measured on the model
    compare with it directly; the points where that convention moves a label,
    or changes how time reversal glues the irreps, are named in
    ``ebr_data(group)["convention"]``.  Operations combined with time reversal
    carry no characters.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    k : array, shape (dim_k,)
        A tabulated point, in the model's reduced coordinates, as
        :func:`high_symmetry_points` gives it.
    kpname : str, optional
        The point's tabulated name, taken as given instead of found from
        ``k``.
    ktol : float, optional
        How far ``k`` may sit from a tabulated point and be it, in reduced
        reciprocal coordinates.  Default ``1e-5``.

    Returns
    -------
    dict or Refusal
        ``{corep: {ind: character}}``, one entry per corep of the little group,
        its character on each unitary operation by index.  A ``Refusal`` in
        its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum.
    ``kpoint_not_tabulated``
        ``k`` is no tabulated point in the model's coordinates, or not the
        tabulated member of its star; the refusal lists the tabulated points.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or "irrep tables are not set up for this cell",
                       context={"number_str": group.number_str})
    lg = little_group(group, k)
    if isinstance(lg, Refusal):
        return lg
    name = _resolve_kpname(group, lg, kpname, ktol)
    if isinstance(name, Refusal):
        return name
    return _transported_table(group, name, lg.k3 % 1.0)


def identify_from_characters(group: GroupData, k, characters, *, multiplet_dims=None,
                             kpname=None, label_tol=1e-3, ktol=1e-5) -> IrrepContent | Refusal:
    r"""The coreps each multiplet carries at a high-symmetry point, from its measured characters.

    Each row of characters is expanded over the coreps of the little group,
    ``m = Σ_g χ_corep(g) χ(g)* / Σ_g |χ_corep(g)|²``, and a multiplicity is a
    nonnegative integer or the characters are wrong.  Levels merged into one
    multiplet read as a sum of coreps.  :func:`..sewing.corep_content`
    measures the characters on a set of bands and reads them here.

    Parameters
    ----------
    group : GroupData
        The group the characters were measured in,
        ``realized_group(model).group``.
    k : array, shape (dim_k,)
        A tabulated point, in the model's reduced coordinates, as
        :func:`high_symmetry_points` gives it.
    characters : array, shape (n_multiplets, n_unitary)
        A row per degenerate multiplet and a column per unitary operation of
        the little group in index order, ``little_group(group, k).indices``:
        the trace over the multiplet's states of ``rep.sewing_operator(ind,
        k)``, as :func:`..sewing.characters` measures it.
    multiplet_dims : array, shape (n_multiplets,), optional
        Each multiplet's degeneracy, which the coreps found must add up to.
    kpname : str, optional
        The point's tabulated name, taken as given instead of found from
        ``k``.
    label_tol : float, optional
        How far a multiplicity may sit from an integer, and two rows'
        characters from each other to share a number in the partition,
        dimensionless.  Default ``1e-3``.
    ktol : float, optional
        How far ``k`` may sit from a tabulated point and be it, in reduced
        reciprocal coordinates.  Default ``1e-5``.

    Returns
    -------
    IrrepContent or Refusal
        Each multiplet's coreps and dimension, and the partition of the
        multiplets by their characters, which stands under any labelling and
        is the statement to quote where a label rests on one; ``summary()``
        prints it.  A ``Refusal`` in its place is falsy.

    Raises
    ------
    ValueError
        ``characters`` has not one column per unitary operation of the little
        group, or ``multiplet_dims`` not one entry per row.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        The group has no transformation to the tables' conventional cell.
    ``out_of_plane``
        ``k`` has a component along a layer's vacuum.
    ``kpoint_not_tabulated``
        ``k`` is no tabulated point in the model's coordinates, or not the
        tabulated member of its star; the refusal lists the tabulated points.
    ``little_group_mismatch``
        The little group measured in the model's cell is not the table's.
    ``integer_multiplicity``
        A multiplicity further than ``label_tol`` from a nonnegative integer:
        characters in another convention, or one level split across
        multiplets, which reads one half for a two-dimensional corep.
    ``multiplet_dimension``
        The coreps found do not add up to ``multiplet_dims``: the rows were
        grouped into multiplets wrongly.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready",
                       group._setup_note or "irrep tables are not set up for this cell",
                       context={"number_str": group.number_str})

    lg = little_group(group, k)
    if isinstance(lg, Refusal):
        return lg

    chars = np.atleast_2d(np.asarray(characters, dtype=complex))
    if chars.shape[1] != len(lg.unitary):
        raise ValueError(
            f"characters must have one column per unitary little-group operation "
            f"({len(lg.unitary)} at k={np.round(lg.k, 6)}, indices {lg.indices}), "
            f"got {chars.shape[1]}")

    kpname = _resolve_kpname(group, lg, kpname, ktol)
    if isinstance(kpname, Refusal):
        return kpname
    table = corep_table(group, lg.k, kpname=kpname, ktol=ktol)
    if isinstance(table, Refusal):
        return table

    table_inds = set(next(iter(table.values())).keys())
    if not set(lg.indices) <= table_inds:
        return Refusal(
            "little_group_mismatch",
            f"the measured little group at {kpname} carries the operation indices "
            f"{sorted(lg.indices)} but the table lists {sorted(table_inds)}; matching the "
            f"operations to the standard setting did not close, so no label would be meaningful.",
            context={"ours": sorted(lg.indices), "table": sorted(table_inds), "kpname": kpname})

    label_order = tuple(table)
    mult = np.zeros((chars.shape[0], len(label_order)), dtype=complex)
    dims = np.zeros(len(label_order))
    # the identity is in every little group; its character is the irrep dimension
    identity = next((int(s.ind) for s in lg.unitary
                     if np.array_equal(s.rotation, np.eye(3, dtype=int))
                     and np.abs(s.translation - np.round(s.translation)).max() < 1e-8),
                    min(lg.indices))
    for c, lab in enumerate(label_order):
        ir = np.array([table[lab][i] for i in lg.indices], dtype=complex)
        norm = float((np.abs(ir) ** 2).sum())
        dims[c] = abs(table[lab][identity])
        mult[:, c] = ir @ chars.conj().T / norm

    defect = float(np.abs(mult - np.round(mult.real)).max()) if mult.size else 0.0
    rounded = np.round(mult.real).astype(int)
    bad = (np.abs(mult - rounded) > label_tol) | (rounded < 0)
    if bad.any():
        rows = sorted(set(int(r) for r in np.nonzero(bad)[0]))
        return Refusal(
            "integer_multiplicity",
            f"irrep multiplicities at {kpname} are not non-negative integers (worst defect "
            f"{defect:.3g} > {label_tol:g}) for multiplet row(s) {rows}. Either the characters were "
            f"not computed in the convention pythtqc.groups documents (its module docstring), or "
            f"a degeneracy was split into several multiplets by numerical noise.",
            value=defect, threshold=float(label_tol),
            context={"kpname": kpname, "k": lg.k, "indices": lg.indices,
                     "labels": label_order, "multiplicities": mult,
                     "characters": chars, "rows": rows})

    total_dims = rounded @ dims
    if multiplet_dims is not None:
        want = np.asarray(multiplet_dims, dtype=float).ravel()
        if want.shape[0] != chars.shape[0]:
            raise ValueError("multiplet_dims must have one entry per character row")
        if not np.allclose(total_dims, want, atol=1e-6):
            return Refusal(
                "multiplet_dimension",
                f"identified irrep dimensions {total_dims.tolist()} do not match the supplied "
                f"multiplet degeneracies {want.tolist()} at {kpname}; the degeneracy grouping "
                f"and the characters disagree.",
                value=float(np.abs(total_dims - want).max()), threshold=1e-6,
                context={"kpname": kpname, "got": total_dims, "expected": want})

    labels = tuple({lab: int(rounded[r, c]) for c, lab in enumerate(label_order)
                    if rounded[r, c] != 0} for r in range(chars.shape[0]))

    part = np.full(chars.shape[0], -1, dtype=int)
    nxt = 0
    for r in range(chars.shape[0]):
        if part[r] >= 0:
            continue
        part[r] = nxt
        for r2 in range(r + 1, chars.shape[0]):
            if part[r2] < 0 and np.abs(chars[r] - chars[r2]).max() <= label_tol:
                part[r2] = nxt
        nxt += 1

    return IrrepContent(kpname=kpname, k=lg.k, indices=lg.indices, labels=labels,
                        multiplicities=mult, label_order=label_order, dims=total_dims,
                        partition=tuple(int(p) for p in part), integrality=defect)


# --------------------------------------------------------------------------
# EBRs and indicators (a thin layer over irrep/irreptables)
# --------------------------------------------------------------------------

class _EBRTransportError(RuntimeError):
    """The EBR rows cannot be carried into the physical convention (see below)."""

    def __init__(self, message, points=()):
        super().__init__(message)
        self.points = tuple(points)


def _label_kpoint(label: str) -> str:
    """``'-GM4GM5' → 'GM'`` — the leading run of non-digits after the ``-`` marker.

    Duplicated (three lines) from :func:`..ebr.kpoint_of_label` on purpose:
    :mod:`.ebr` imports :mod:`.groups`, never the other way round.
    """
    s = str(label)[1:] if str(label).startswith("-") else str(label)
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return s[:i]


def _conjugate_site_name(ebr_name: str) -> str | None:
    """``'1E↑G(1)' → '2E↑G(1)'`` — the complex-conjugate *site* irrep, or ``None``.

    BCS names a complex-conjugate pair of site representations ``1X``/``2X``
    (``1E``/``2E``, ``1E1``/``2E1``, ``-1Eg``/``-2Eg``, …), with the ``-``
    marking a double-valued one; every other name is self-conjugate, and in a
    grey group a conjugate pair that time reversal has already glued carries
    both halves in one name (``1E2E``, ``-1E2E1``) and is self-conjugate too.
    Returns ``None`` when the name is self-conjugate by that rule.

    Measured, not assumed: over **all 230 space groups plus their grey
    extensions, single- and double-valued — 10 074 EBR columns — this rule
    agrees with the Γ-block requirement below in every case and disagrees in
    none**.  The Γ check is applied at run time anyway.
    """
    site = str(ebr_name).split("\u2191")[0]
    tail = str(ebr_name)[len(site):]
    sign, rest = ("-", site[1:]) if site.startswith("-") else ("", site)
    if rest[:1] == "1":
        return f"{sign}2{rest[1:]}{tail}"
    if rest[:1] == "2":
        return f"{sign}1{rest[1:]}{tail}"
    return None


def _conjugation_involution(table, labels) -> dict:
    r"""``{label: label}`` — the map ``c`` with ``χ_phys[c(D)] = conj(χ_BCS[D])``.

    At most tabulated k the physical-convention table is the tabulated one with
    its factor system conjugated (:func:`_transported_table`), i.e. on the
    table's own coset representatives

    .. math::
        \chi^\text{phys}_D(\{R|t\}) = \chi^\text{BCS}_D(\{R|t\})\,
                                      e^{-4\pi i\,k\cdot t},

    and ``c`` is fixed by matching character vectors — no naming convention is
    consulted.  ``c`` is always an **involution**: writing
    :math:`\varphi(g) = e^{-4\pi i k\cdot t_g}` (unimodular),
    :math:`\chi_{c(D)}\varphi = \overline{\chi_D}` applied twice gives
    :math:`\chi_{c(c(D))}\varphi = \overline{\chi_{c(D)}}
    = \chi_D\overline{\varphi^{-1}} = \chi_D\varphi`, hence ``c(c(D)) = D``.
    That is checked here rather than trusted.

    Labels of an EBR-basis k-point the *irrep* table does not carry — the
    ``KA``/``HA``/``PA``/``WA`` "other arm" points, which the grey (corep)
    tables do carry but the unitary ones mostly do not — are mapped by
    transporting ``c`` from the partner point through the name (``KA2`` ↔
    ``K2``).  That is BCS's own arm convention, and it is verified against the
    grey tables, where both arms are tabulated with
    :math:`\chi_{XA_n} = \overline{\chi_{X_n}}` at :math:`k_{XA} = -k_X`.

    **Where** :func:`_transported_table` **returns the whole-character conjugate**
    — the k-points :func:`_table_conjugation` marks, where the tabulated
    labelling names no physical representation and the physical table is
    :math:`\overline{\chi_\text{BCS}}` — the definition above is unchanged, and
    it is also what fixes the returned *name*: ``c(D)`` is the
    BCS label whose shipped characters are :math:`\overline{\chi_D}` wherever
    one exists, so the EBR relabelling and the returned table move together by
    construction.  Only where no such label exists does ``c`` come from
    somewhere else — for a **grey** table from the corep relabelling lifted from
    the parent unitary table, whose image at the re-glued points is a corep name
    the EBR basis does not contain.  The transport formula is still the same
    relabelling — the derivation of
    :math:`n^\text{phys}_\rho[c(C)] = n^\text{BCS}_{\rho^*}[C]` only uses
    :math:`\chi_\text{phys}[c(C)] = \overline{\chi_\text{BCS}[C]}`, which the
    lift satisfies constituent by constituent — but the *basis itself* moves
    there, so ``c`` is a bijection onto a label set that is not the one it
    started from.  It is therefore checked to be an involution only where the
    image is a label of the same basis, and to preserve dimension always.

    Raises :class:`_EBRTransportError` only for a **tabulation** defect: a
    k-point whose irreps disagree about which operations are in the little
    group, or one where the tabulated character set is not conjugation-closed
    although both ``ω₀`` and ``φ`` are real and no gluing statement can explain
    it either.  No point shipped in ``irreptables`` is of either kind.
    """
    conj, _conjugated, broken = _table_conjugation(table)
    if broken:
        raise _EBRTransportError(
            f"the tabulated little-group characters of #{table.number_str} at "
            f"{list(broken)} are not a consistent set: they are not closed under "
            "complex conjugation, and neither the tabulated factor system nor the "
            "time-reversal gluing phase is complex there, so the e^{+ik.t}/e^{-ik.t} "
            "convention difference cannot account for it. This is a defect of the "
            "shipped table, not of the transport, so the EBR rows are refused "
            "rather than carried across it.", points=list(broken))

    for label in labels:                       # the untabulated 'A' arms
        if label in conj:
            continue
        kp = _label_kpoint(label)
        # ``len(kp) > 1`` keeps the hexagonal ``A`` point out of the arm rule.
        base = label.replace(kp, kp[:-1]) if len(kp) > 1 and kp.endswith("A") else None
        if base is None or base not in conj:
            raise _EBRTransportError(
                f"the EBR-basis label {label!r} (k-point {kp!r}) is not in the "
                "irrep table and is not the 'other arm' of one that is, so its "
                "convention transport is undefined.", points=[kp])
        conj[label] = conj[base].replace(kp[:-1], kp)

    images = {conj[label] for label in labels}
    if len(images) != len(labels):
        raise _EBRTransportError(
            "the conjugation map is not injective on the EBR basis "
            f"({len(labels)} labels, {len(images)} images); the tabulated "
            "characters are not what this transport assumes.")
    for label in labels:
        image = conj[label]
        if image in conj and conj[image] != label:
            raise _EBRTransportError(
                f"the conjugation map is not an involution at {label!r} "
                f"({label} → {image} → {conj.get(image)}); "
                "the tabulated characters are not what this transport assumes.",
                points=[_label_kpoint(label)])
        if _label_kpoint(image) != _label_kpoint(label):
            raise _EBRTransportError(          # pragma: no cover - structural
                f"the conjugation map moves {label!r} to another k-point "
                f"({image!r}); the transport is per k-point by construction.",
                points=[_label_kpoint(label)])
    return conj


@lru_cache(maxsize=None)
def _ebr_transport(number_str: str, spinor: bool, magnetic: bool) -> tuple:
    r"""Permutations carrying ``irreptables``' EBR rows into the physical convention.

    Returns ``(labels, conjugation, partner, changed_labels, new_labels, perm)``:
    the tabulated EBR-basis label order, the per-k conjugation ``c`` of
    :func:`_conjugation_involution`, the index ``ρ ↦ ρ*`` of the time-reversed
    EBR column, which labels move, the **physical** basis labels in the
    same positions, and the row permutation that goes with them.  Cached per
    ``(number_str, spinor, magnetic)`` — it is a property of the tables, not of
    any model.

    The physics in one line.  A band representation induced from the site
    representation ``ρ`` at Wyckoff position ``q`` has, at the k-point ``k``,
    the character

    .. math::
        \chi^{(s)}(\{R|t\}) = \sum_{j:\,g q_j = q_j}
            e^{s\,2\pi i\,k\cdot m_j}\,\chi_\rho(g_j),
        \qquad m_j = R\tau_j + t - \tau_j\in\mathbb{Z}^3 ,

    with ``s = +1`` in BCS's convention (what ``irreptables`` ships) and
    ``s = −1`` in the physical one (what :mod:`..sewing` measures).  Because
    :math:`\chi_\rho(g_j)` and :math:`m_j` are convention-free,

    .. math::
        \chi^{(-)}_\rho = \overline{\chi^{(+)}_{\rho^*}},

    where :math:`\rho^*` is the complex-conjugate site representation — note
    that conjugating only the phase would be wrong for a complex ``1E``/``2E``
    site pair.  Pairing that with the definition of ``c`` gives the whole
    transport as a **relabelling**,

    .. math::
        n^\text{phys}_\rho[\,c(D)\,] = n^\text{BCS}_{\rho^*}[\,D\,],

    which is what comes back.  Two consequences are worth naming:
    the Wyckoff **position never moves** (``ρ*`` sits at ``q`` too), and at
    ``k = 0`` the transport is the identity — asserted below on the Γ block of
    every column, which is the run-time check that ``ρ*`` was identified right.

    *Re-glued k-points.*  Where ``c`` carries a grey corep label to one the
    tables do not ship, the formula above is unchanged but the **basis** at that
    k-point is not: the physical corep list is a
    different partition of the same unitary irreps, so the labels there are
    renamed in place rather than permuted (``-P3P3, -P4 → -P4P4, -P3`` for
    ``80.30``, multiplicities ``1, 2 → 1, 2`` — read positionally, which is what
    ``n^phys[c(C)] = n^BCS[C]`` says).  ``c`` preserves the dimension of every
    constituent, so the ``degeneracies`` array is untouched and the per-k-point
    dimension ``Σ_C n_C · dim C`` is conserved; both are asserted here.

    Raises :class:`_EBRTransportError` (never returns partly transported rows).
    """
    from irreptables.ebrs import load_ebr_data
    from irreptables.irreps import IrrepTable

    raw = load_ebr_data(number_str, spinor)
    labels = tuple(str(l) for l in raw["basis"]["irrep_labels"])
    try:
        names = [str(e["ebr_name"]) for e in raw["ebrs"]]
        wyck = [str(e["wyckoff_position"]) for e in raw["ebrs"]]
    except KeyError as exc:                       # e.g. 80.31 ships no position
        raise _EBRTransportError(
            f"the EBR data of #{number_str} is missing {exc}; it cannot be "
            "carried into the physical convention (nor read by irrep's own "
            "`get_ebr_names_and_positions`).") from exc
    matrix = np.array([e["vector"] for e in raw["ebrs"]], dtype=int).T

    table = IrrepTable(number_str, spinor, magnetic=magnetic)
    conj = _conjugation_involution(table, labels)

    degen = {l: int(d) for l, d in zip(labels, raw["basis"]["degeneracies"])}
    bad = [l for l in labels
           if conj[l] in degen and degen[l] != degen[conj[l]]]
    if bad:
        raise _EBRTransportError(
            f"conjugation changes the dimension of {bad}; the tabulated "
            "degeneracies and characters disagree.",
            points=[_label_kpoint(l) for l in bad])

    partner = []
    for i, (name, pos) in enumerate(zip(names, wyck)):
        want = _conjugate_site_name(name)
        hits = [j for j, (n2, p2) in enumerate(zip(names, wyck))
                if n2 == want and p2 == pos]
        if want is None or not hits:
            partner.append(i)                     # self-conjugate site representation
        elif len(hits) == 1:
            partner.append(hits[0])
        else:
            raise _EBRTransportError(
                f"the EBR data of #{number_str} lists {want!r}@{pos} more than "
                "once, so the time-reversed partner of "
                f"{name!r}@{pos} is ambiguous.")
    if any(partner[partner[i]] != i for i in range(len(partner))):
        raise _EBRTransportError(
            f"the time-reversal pairing of #{number_str}'s EBR columns is not an "
            "involution.")

    index = {l: i for i, l in enumerate(labels)}
    #  Per k-point block: where ``c`` maps the block onto itself the labels stay
    #  put and the multiplicities permute (the ordinary case: the returned rows are
    #  then the shipped rows permuted); where it does not — a re-glued point —
    #  the labels are renamed in place.
    block: dict[str, list] = {}
    for i, l in enumerate(labels):
        block.setdefault(_label_kpoint(l), []).append(i)
    new_labels, perm = list(labels), list(range(len(labels)))
    for kp, pos in block.items():
        here = [labels[i] for i in pos]
        if {conj[l] for l in here} == set(here):
            for i in pos:
                perm[i] = index[conj[labels[i]]]
        else:
            for i in pos:
                new_labels[i] = conj[labels[i]]
    new_labels, perm = tuple(new_labels), np.array(perm)

    gamma = [l for l in labels if _label_kpoint(l) == "GM"]
    for i in range(matrix.shape[1]):
        for l in gamma:
            if matrix[index[l], i] != matrix[index[conj[l]], partner[i]]:
                raise _EBRTransportError(
                    f"the Γ block of {names[i]}@{wyck[i]} is not invariant under "
                    "the transport (k = 0 carries no site-displacement phase, so "
                    "it must be): the conjugate-site pairing or the tabulated "
                    "Γ characters are not what this transport assumes.",
                    points=["GM"])

    #  Dimension conservation, k-point by k-point and column by column: the band
    #  representation has the same number of states at every k in either
    #  convention, whether or not the corep basis there was re-glued.
    deg = np.array([degen[l] for l in labels], dtype=int)
    for i in range(matrix.shape[1]):
        col = matrix[perm, partner[i]]
        for kp, pos in block.items():
            was = int(sum(matrix[j, i] * deg[j] for j in pos))
            now = int(sum(col[j] * deg[j] for j in pos))
            if was != now:
                raise _EBRTransportError(
                    f"the transport changes the dimension of {names[i]}@{wyck[i]} "
                    f"at {kp} ({was} → {now}); the re-glued corep basis is not a "
                    "partition of the same unitary irreps.", points=[kp])

    moved = tuple(l for l, l2, j in zip(labels, new_labels, perm)
                  if l != l2 or any(matrix[index[l], i] != matrix[j, partner[i]]
                                    for i in range(matrix.shape[1])))
    return labels, conj, tuple(partner), moved, new_labels, tuple(int(p) for p in perm)


def _to_physical_convention(raw: dict, number_str: str, spinor: bool,
                            magnetic: bool) -> dict:
    """``raw`` with every EBR row rewritten in the physical convention.

    A new ``dict`` (``irreptables`` re-reads its JSON on every call, so nothing
    shared is mutated): the ``vector``/``irrep_list`` of each EBR, the basis
    ``irrep_labels`` where a k-point was re-glued, and the Smith form, which is
    transported exactly rather than recomputed — with ``M' = P M Q`` for the
    label permutation ``P`` and the column permutation ``Q`` (both involutions),
    ``M = U⁻¹RV⁻¹`` gives ``M' = (UP)⁻¹R(QV)⁻¹`` with the elementary divisors
    ``R`` untouched.  A re-glued block contributes the identity to ``P`` (its
    labels are renamed in place, not permuted), so the Smith form is carried
    across it unchanged, which is right: the multiplicity *vector* does not move
    there, only what its entries are called.
    """
    labels, conj, partner, moved, new_labels, perm = _ebr_transport(
        number_str, spinor, magnetic)
    perm = np.array(perm)
    matrix = np.array([e["vector"] for e in raw["ebrs"]], dtype=int).T

    out = dict(raw)
    out["basis"] = dict(raw["basis"])
    if new_labels != labels:
        out["basis"]["irrep_labels"] = list(new_labels)
    out["ebrs"] = []
    for i, entry in enumerate(raw["ebrs"]):
        col = matrix[perm, partner[i]]
        new = dict(entry)
        new["vector"] = [float(v) for v in col]
        new["irrep_list"] = [l for l, m in zip(new_labels, col)
                             for _ in range(int(m))]
        out["ebrs"].append(new)

    # ``r`` — the elementary divisors — is invariant under both permutations, and
    # each of ``u``/``v`` is rewritten only when its permutation is not the
    # identity: ``irreptables`` stores a degenerate Smith factor as a scalar when
    # the group has a single EBR column (measured on SG 4, 7, 9, 29, …), and
    # there the column permutation is the identity anyway.
    smith = raw.get("smith_form")
    if smith is not None:
        moves = {"u": (list(perm) != list(range(len(labels))), perm, 1),
                 "v": (list(partner) != list(range(len(raw["ebrs"]))),
                       list(partner), 0)}
        if any(m[0] for m in moves.values()):
            out["smith_form"] = dict(smith)
            for key, (needed, order, axis) in moves.items():
                if not needed:
                    continue
                arr = np.asarray(smith[key], dtype=int)
                if arr.ndim != 2:
                    raise _EBRTransportError(
                        f"#{number_str}'s tabulated Smith factor {key!r} is not a "
                        f"matrix (ndim={arr.ndim}), so the transport cannot be "
                        "carried onto it.")
                out["smith_form"][key] = (arr[:, order] if axis else
                                          arr[order, :]).tolist()
    return out


def _transport_record(number_str: str, spinor: bool, magnetic: bool) -> dict:
    """The ``"convention"`` entry :func:`ebr_data` publishes — what moved, and why."""
    labels, conj, partner, moved, new_labels, _perm = _ebr_transport(
        number_str, spinor, magnetic)
    reglued = tuple(sorted({_label_kpoint(l) for l, l2 in zip(labels, new_labels)
                            if l != l2}))
    return {
        "name": "physical (translations as e^{-ik.t})",
        "applied": True,
        "changed_labels": moved,
        "changed_points": tuple(sorted({_label_kpoint(l) for l in moved})),
        "conjugated_points": tuple(sorted(
            _conjugated_kpoints(number_str, spinor, magnetic))),
        "divergent_points": tuple(sorted(
            _divergent_kpoints(number_str, spinor, magnetic))),
        "reglued_points": reglued,
        "conjugation": {l: conj[l] for l in labels},
        "labels_bcs": labels,
        "labels_physics_convention": new_labels,
        "row_permutation": tuple(int(p) for p in _perm),
        "tr_partner": tuple(partner),
        "note": ("EBR rows transported out of BCS's e^{+ik.t} convention; label "
                 "pairings at non-TRIM points therefore differ from BCS's shipped "
                 "BANDREP output. At `conjugated_points` the little-group "
                 "characters are returned as whole-character conjugates, but under "
                 "the BCS name of the row that comes out, so the printed table is "
                 "BCS's own and only which label a band lands on differs. At "
                 "`divergent_points` not even that is possible -- the shipped page "
                 "there tabulates the other star member -k -- so the printed rows "
                 "are not BCS's; and at `reglued_points` (a subset, always grey) "
                 "the time-reversal gluing itself differs, so the corep basis is a "
                 "different partition of the same unitary irreps and a given irrep "
                 "can sit in a corep of a different dimension. See "
                 "docs/background/conventions.md, 'Reading across to BCS'."),
    }


@lru_cache(maxsize=None)
def _improper_letters(sg: int) -> frozenset:
    r"""The Wyckoff letters of space group ``sg`` whose site group is improper.

    A letter's site group is every operation of the group that fixes its
    representative point at a generic value of its free parameters, the point
    and the operations both ``pyxtal``'s in the tables' setting
    (:func:`_pyxtal_wyckoff`: the general position's operations, the
    conventional cell's centring among them).  It is named by
    :func:`.._sitechars.point_group_of`, and improper is that name starting
    with a bar: ``-1``, ``-3``, ``-4``, ``-6``, ``-3m``, ``-42m``, ``-6m2`` or
    ``-43m``.
    """
    from . import _sitechars

    positions, _centering = _pyxtal_wyckoff(int(sg))
    general = max(positions, key=lambda p: p["multiplicity"])
    out = set()
    for position in positions:
        A, b, _P = position["arms"][0]
        point = A @ _GENERIC_PARAMETER + b
        fixing = []
        for R, t, _Q in general["arms"]:
            delta = R @ point + t - point
            if float(np.abs(delta - np.rint(delta)).max()) <= 1e-6:
                fixing.append(R)
        if _sitechars.point_group_of(fixing).startswith("-"):
            out.add(position["letter"])
    return frozenset(out)


def _barred(position: str, improper) -> str:
    """``"3a(3m1',3m)" → "3a(-3m1',-3m)"``: an EBR position, its site group with its bar.

    The tables' own EBR files write every improper site group without its bar:
    ``3m`` for the ``-3m`` at ``3a`` of ``R-3m``, ``6m2`` for the ``-6m2`` at
    ``2c`` of ``P6/mmm``, ``4`` for ``-4``.  At a letter of ``improper``
    (:func:`_improper_letters`) the bar goes back on the tables' own symbol,
    which keeps the orientation that symbol writes: which axis carries which
    operation, and so which irrep is ``B1`` and which ``B2``.
    """
    match = re.match(r"^(\d+)([a-zA-Z])\((.*)\)$", position)
    if match is None or match.group(2) not in improper:
        return position
    symbols = ",".join(part if part.startswith("-") else "-" + part
                       for part in match.group(3).split(","))
    return f"{match.group(1)}{match.group(2)}({symbols})"


def ebr_data(group: GroupData) -> dict | Refusal:
    """Every elementary band representation of the group: the bands each site irrep at each Wyckoff position induces.

    The atomic limits the group has, and the columns a symmetry vector is
    decomposed over; :func:`..ebr.decompose` and
    :func:`..ebr.symmetry_indicator` read them, a layer's restricted to the
    points of its plane.  The rows are in the package's convention for a
    lattice translation, ``e^{-ik·t}``, where the tables' own pages write
    ``e^{+ik·t}``: away from the time-reversal-invariant momenta a row pairs
    labels with Wyckoff positions otherwise than the Bilbao Crystallographic
    Server's page does, and ``convention`` says where.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``; only its label, its
        valuedness and whether it is grey are read.

    Returns
    -------
    dict or Refusal
        ``matrix``: the EBR matrix, a row per corep at each high-symmetry
        point of the group, those out of a layer's plane included, and a
        column per elementary band representation;
        ``labels_physics_convention``: the rows' coreps, in the labels
        :func:`identify_from_characters` returns;
        ``ebrs``: each column as ``(name, position)``, the position with its
        site group, an improper one with its bar, ``"3a(-3m1',-3m)"``;
        ``smith``: the Smith normal form ``(U, D, V)``, ``U·M·V = D``;
        ``convention``: what the convention moved, by point and by label, with
        a ``note``: ``changed_points`` and ``changed_labels``,
        ``conjugated_points``, ``divergent_points``, ``reglued_points``;
        ``conjugation``, each label's partner, and ``tr_partner``, each
        column's, the column of the complex-conjugate site irrep;
        ``labels_bcs``, the server's names of the rows;
        and ``row_permutation``, ``P`` with ``y_BCS[j] = y[P[j]]``, which
        re-keys a form written in the server's labels;
        ``raw``: the tables' own record, carried into this convention.
        A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``irrep_tables``
        ``irreptables`` is not installed.
    ``ebr_tables``
        ``irreptables`` ships no elementary band representations for this
        group and valuedness.
    ``ebr_convention``
        An entry of the shipped file with no Wyckoff position, so its row
        cannot be carried into this convention; every known case is a magnetic
        group of type III, outside the groups returned.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)
    from irrep.ebrs import get_ebr_matrix, get_ebr_names_and_positions, get_smith_form
    from irreptables.ebrs import load_ebr_data

    try:
        raw = load_ebr_data(group.number_str, group.spinor)
    except (FileNotFoundError, OSError) as exc:
        return Refusal("ebr_tables",
                       f"irreptables has no EBR data for group #{group.number_str} "
                       f"({'double' if group.spinor else 'single'} valued): {exc}",
                       context={"number_str": group.number_str, "spinor": group.spinor})
    try:
        raw = _to_physical_convention(raw, group.number_str, group.spinor,
                                      group.magnetic)
        record = _transport_record(group.number_str, group.spinor, group.magnetic)
    except _EBRTransportError as exc:
        return Refusal(
            "ebr_convention",
            f"the EBR rows of #{group.number_str} cannot be carried out of BCS's "
            f"e^(+ik.t) convention into this package's physical one: {exc}",
            context={"number_str": group.number_str, "spinor": group.spinor,
                     "magnetic": group.magnetic, "points": exc.points})
    except (FileNotFoundError, OSError) as exc:
        return Refusal("irrep_tables",
                       f"irreptables has no irrep table for group #{group.number_str}, "
                       f"so its EBR rows cannot be transported: {exc}",
                       context={"number_str": group.number_str, "spinor": group.spinor})
    improper = _improper_letters(int(str(group.number_str).split(".")[0]))
    return {"raw": raw, "matrix": get_ebr_matrix(raw),
            "labels_physics_convention": list(raw["basis"]["irrep_labels"]),
            "ebrs": [(str(name), _barred(str(position), improper))
                     for name, position in get_ebr_names_and_positions(raw)],
            "smith": get_smith_form(raw), "convention": record}


#: Why :func:`indicator_group` withholds ``irreptables``' named forms from a
#: group that is not grey, one string per case.
_SI_NOTES = {
    "bare_key": (
        "no named form is returned: for a unitary group irreptables' non-magnetic "
        "indicator file is the grey table keyed by the bare space-group number "
        "(single_indicators.json and double_indicators.json are byte for byte the "
        "type-II entries of the magnetic files), so its factors are grey corep "
        "labels -- against a unitary group's own EBR basis only 25 of 70 single and "
        "48 of 320 double forms even have the right label set, and only 39 / 128 "
        "vanish on that group's own EBRs. Summing one over unitary labels is a "
        "silent partial sum, not an indicator. `elementary_divisors` is unaffected: "
        "it is the Smith form of this group's own transported EBR matrix. For "
        "values, use ebr.symmetry_indicator, which computes the class from the "
        "Smith form and returns a named form only after checking it vanishes on "
        "every EBR of the group it is being read on"),
    "unaudited_bns": (
        "no named form is returned: irreptables does ship indicator forms under this "
        "magnetic (BNS) key, but only the grey entries of those files have been "
        "audited here, and the corner that has not is measurably unclean -- five "
        "shipped z2 forms (216.74, 226.122, 226.127, 227.128, 227.130) are not "
        "indicators of their own group at all, being nonzero on real EBRs. "
        "`elementary_divisors` is unaffected: it is the Smith form of this group's "
        "own transported EBR matrix. For values, use ebr.symmetry_indicator"),
    "grey": (
        "irreptables' named forms for this grey group, as shipped: their "
        "coefficients are keyed by BCS labels and must be re-keyed by "
        "ebr_data(group)['convention']['row_permutation'] before they are evaluated "
        "on a returned symmetry vector (in scope that bites once, the z2 of 220.90). "
        "ebr.symmetry_indicator does that re-keying, and checks the form on the "
        "group's own EBRs, before it reports a value"),
    "grey_none": (
        "irreptables ships no named indicator formula for this grey group, a "
        "verdict in itself -- read `elementary_divisors` for the indicator group "
        "itself, and ebr.symmetry_indicator for a class"),
}


def indicator_group(group: GroupData) -> dict | Refusal:
    """The group's symmetry-indicator group, and the tables' named indicator formulas for it.

    For a band set's class, :func:`..ebr.symmetry_indicator` is the call: it
    reads the class off the Smith normal form, and names it with a formula only
    after checking that the formula vanishes on every elementary band
    representation of the group.  This one returns the group's own data, a
    layer's out-of-plane points included.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``; only its label, its
        valuedness and whether it is grey are read.

    Returns
    -------
    dict or Refusal
        ``elementary_divisors``: the divisors above 1 of the Smith normal form
        of the group's EBR matrix, the indicator group ``⊕ ℤ_d``; ``None``
        where :func:`ebr_data` refuses, and that refusal is the one to read.
        ``indicators``: the tables' named formulas, ``{name: {"factors",
        "mod"}}``, returned for a grey group only; ``orders``,
        ``{name: mod}``, each the order of one formula, so that equal or
        dependent formulas make them something other than the divisors.
        ``withheld``: whether formulas exist and are left out, as for every
        group that is not grey, ``reason`` saying why; ``indicators`` empty
        with ``withheld`` False is a verdict, no formula named for this group.
        ``note``: the caveat to read either way, that a formula's coefficients
        are keyed by the tables' labels and are re-keyed by
        ``ebr_data(group)["convention"]["row_permutation"]`` before they are
        read on a returned symmetry vector.  A ``Refusal`` in its place is
        falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``irrep_tables``
        ``irreptables`` is not installed.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)
    from irreptables.symmetry_indicators import load_si_table

    data = ebr_data(group)
    divisors = None
    if not isinstance(data, Refusal):
        r = np.asarray(data["smith"][1])
        # single EBR column: R is stored flat, and the vector is then already the
        # diagonal extended by the empty rows (the shape ..ebr._smith_system normalizes)
        d = r if r.ndim == 1 else r.diagonal()
        divisors = d[d > 1]

    if _magnetic_type(group) != "II":
        note = _SI_NOTES["unaudited_bns" if group.magnetic else "bare_key"]
        return {"indicators": {}, "orders": {}, "elementary_divisors": divisors,
                "withheld": True, "reason": note, "note": note}

    si = load_si_table(spinor=group.spinor, magnetic=group.magnetic)
    entry = si.get(group.number_str, {}).get("indicators", {})
    return {"indicators": entry,
            "orders": {k: v["mod"] for k, v in entry.items()},
            "elementary_divisors": divisors,
            "withheld": False, "reason": None,
            "note": _SI_NOTES["grey" if entry else "grey_none"]}


def _ebr_solve_shim(ebr_matrix, y, original_exc):
    """``irrep.ebrs.compute_ebr_decomposition`` with the ortools-9.15 status fix.

    Identical model and callback to ``irrep``'s (``varArraySolutionObtainer`` is
    reused verbatim); only ``solver.status_name()`` — which ortools 9.15 broke —
    is replaced by comparing the enum ``Solve`` returns.
    """
    from ortools.sat.python import cp_model
    from irrep.or_solutions_obtainer import varArraySolutionObtainer

    n_ir, n_ebr = ebr_matrix.shape
    ok = (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def solve(bounds):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        x = [model.NewIntVar(bounds[0], bounds[1], f"x{i}") for i in range(n_ebr)]
        obtainer = varArraySolutionObtainer(x)
        for i in range(n_ir):
            model.Add(sum(int(ebr_matrix[i, j]) * x[j] for j in range(n_ebr)) == int(y[i]))
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(model, obtainer)
        return obtainer.n_smallest_solutions(5), status

    solutions, status = solve((0, 50))
    if status in ok and solutions:
        return solutions, True
    solutions, status = solve((-50, 50))
    if status in ok and solutions:
        return solutions, False
    return None, False


def decompose_ebrs(group: GroupData, irrep_counts: dict) -> dict | Refusal:
    """A symmetry vector written in the group's elementary band representations.

    For a model's band set, :func:`..ebr.decompose` is the call: it reads a
    layer's vector in its plane and lists the decompositions.  This one takes a
    vector over every high-symmetry point of the group and reads it as ``irrep``
    does.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    irrep_counts : dict
        ``{corep: multiplicity}``, the band set's coreps at every high-symmetry
        point of the group, as ``HSPTable.counts`` holds them.

    Returns
    -------
    dict or Refusal
        ``classification``: ``"stable topological"`` where no integer
        combination of elementary band representations reproduces the vector,
        ``"fragile topological"`` where only one with a negative coefficient
        does, and ``"atomic limit"`` where a nonnegative one does, which an
        atomic limit needs and which does not make one; ``stable_nontrivial``,
        whether it is the first; ``solutions``, the combinations found,
        written out in ``decomposition_strings``, and ``is_positive``, whether
        one is nonnegative; ``y`` and ``y_prime``, the vector and its Smith
        coordinates; ``ebrs``, the columns; ``unused``,
        the coreps handed in that no column carries; ``indicators``, the
        tables' named formulas summed over the vector, ``None`` where they are
        withheld, with ``indicators_note`` saying why, or the formulas'
        caveat.  A ``Refusal`` in its place is falsy.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``incomplete_content``
        A high-symmetry point of the group carries no corep in the vector,
        which would read as no bands there; ``context["missing_points"]`` names
        each.  A layer's vector misses the points out of its plane, and
        :func:`..ebr.decompose` reads it.
    ``ortools``
        Telling an atomic limit from fragile bands needs the integer solver
        ``ortools``, which is not installed.
    ``ebr_tables``, ``irrep_tables``, ``ebr_convention``
        From :func:`ebr_data`.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    data = ebr_data(group)
    if isinstance(data, Refusal):
        return data
    from irrep.ebrs import (ORTOOLS_AVAILABLE, compose_ebr_string,
                            compute_ebr_decomposition, compute_topological_classification_vector)

    basis = data["labels_physics_convention"]
    unused = sorted(set(irrep_counts) - set(basis))
    # COMPLETE, or nothing.  `create_symmetry_vector` reads an unmentioned point
    # as zero, and a zero row is a constraint -- the wrong
    # one -- so a vector with a point missing decomposes into the EBRs of a
    # different band set.  A slab is where this bites: the EBR basis of the 3D
    # space group carries k_z = 1/2 points its zone does not have.
    points = sorted({_label_kpoint(lab) for lab in basis})
    covered = {_label_kpoint(lab) for lab in irrep_counts}
    missing_points = [p for p in points if p not in covered]
    if missing_points:
        return Refusal(
            "incomplete_content",
            f"the symmetry vector carries no label at {missing_points}, and the EBR "
            f"basis of #{group.number_str} is written over {points}. A decomposition "
            "of a partial symmetry vector is a confident wrong answer: an unmentioned "
            "point is read as zero content there, which is a constraint on a "
            "different band set. Supply counts covering "
            "every point, or -- for a 2D model, whose zone does not contain the "
            "basis's out-of-plane points at all -- use ebr.decompose, which "
            "restricts the EBR matrix to the measurable rows and checks that the "
            "restriction loses nothing.",
            value=float(len(missing_points)), threshold=0.0,
            context={"missing_points": tuple(missing_points), "points": tuple(points)})
    y, y_prime, nontrivial = compute_topological_classification_vector(irrep_counts, data["raw"])

    si = indicator_group(group)
    indicators, indicators_note = None, None
    if not isinstance(si, Refusal):
        # whether the named forms are returned is `withheld`, not the emptiness of
        # the mapping: a grey group can ship no form at all, which is a verdict --
        # and it is returned as the empty sum, `{}`, not as `None`.
        indicators_note = si["reason"] if si["withheld"] else si["note"]
        if not si["withheld"]:
            indicators = {}
            for name, spec in si["indicators"].items():
                total = sum(v * irrep_counts.get(lab, 0)
                            for lab, v in spec["factors"].items())
                indicators[name] = total % spec["mod"]

    out = {"y": y, "y_prime": y_prime, "stable_nontrivial": bool(nontrivial),
           "indicators": indicators, "indicators_note": indicators_note,
           "ebrs": data["ebrs"], "unused": unused,
           "solutions": None, "is_positive": None}
    if nontrivial:
        out["classification"] = "stable topological"
        return out
    if not ORTOOLS_AVAILABLE:
        return Refusal("ortools",
                       "the band set has integer EBR solutions, so it is trivial or fragile, "
                       "but telling them apart needs the ortools integer solve, which is not "
                       "installed. Install it: pip install 'ortools>=9.10'",
                       context={"y": y, "y_prime": y_prime})
    try:
        solutions, is_positive = compute_ebr_decomposition(data["raw"], y)
    except (TypeError, AttributeError) as exc:
        # ortools 9.15 broke CpSolver.status_name(): its internal
        # `_checked_response.status` became an attribute while cp_model.py still
        # calls it, so irrep's ebrs.get_solutions raises. Everything else in the
        # ortools path works, so run irrep's own model with the status enum
        # compared directly. The shim is deliberately minimal and defers to
        # irrep whenever irrep works.
        solutions, is_positive = _ebr_solve_shim(data["matrix"], y, exc)
    out["solutions"] = solutions
    out["is_positive"] = bool(is_positive)
    out["classification"] = "atomic limit" if is_positive else "fragile topological"
    if solutions:
        out["decomposition_strings"] = [compose_ebr_string(s, data["ebrs"]) for s in solutions]
    return out


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def k_grid(group: GroupData, n, *, allow_asymmetric=False, ktol=1e-8) -> KGrid | Refusal:
    """A Γ-centred mesh that respects the group, with the high-symmetry points it holds.

    A mesh holds only the high-symmetry points it happens to contain: ``Γ``
    always, and the zone boundary only with an even number of divisions.  A
    label read off a mesh at a point it misses is no label, so ``contains``
    names the points it holds and ``hsp`` gives every tabulated point's
    coordinates.  The mesh is pythtb's own, ``k`` in ``[0, 1)``, periodic in
    the gauge a ``WFArray`` reads.

    Parameters
    ----------
    group : GroupData
        The group, ``realized_group(model).group``.
    n : int or sequence of int
        The divisions along each periodic direction, or one number for all.
    allow_asymmetric : bool, optional
        Return a mesh the group does not carry onto itself, with no
        irreducible points, instead of refusing it.  Default ``False``.
    ktol : float, optional
        How far a mesh point may sit from a tabulated point and be it, in
        reduced reciprocal coordinates.  Default ``1e-8``.

    Returns
    -------
    KGrid or Refusal
        The mesh, ready for a ``pythtb.WFArray``, with the tabulated points it
        holds and those it misses; ``summary()`` prints it.  A ``Refusal`` in
        its place is falsy.

    Raises
    ------
    ValueError
        ``n`` is a sequence of other than ``dim_k`` numbers.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``asymmetric_grid``
        The group does not carry the mesh onto itself, so a sum over it would
        weigh points unlike their partners; ``allow_asymmetric=True`` returns
        it all the same.
    """
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    from pythtb import Mesh
    if np.isscalar(n):
        divs = [int(n)] * group.dim_k
    else:
        divs = [int(v) for v in n]
        if len(divs) != group.dim_k:
            raise ValueError(f"n must be a scalar or have {group.dim_k} entries")
    grid3 = [1, 1, 1]
    for j, d in enumerate(group.periodic_dirs):
        grid3[d] = divs[j]

    symmetric = bool(group._sg.is_grid_symmetrical(grid3))
    if not symmetric and not allow_asymmetric:
        return Refusal("asymmetric_grid",
                       f"the {tuple(grid3)} mesh is not symmetric under #{group.number_str}; "
                       f"reading symmetry-related quantities off it double-counts. Pass "
                       f"allow_asymmetric=True to override deliberately.",
                       context={"grid": tuple(grid3), "number_str": group.number_str})

    mesh = Mesh(["k"] * group.dim_k)
    mesh.build_grid(tuple(divs), gamma_centered=False, k_endpoints=False)
    points = np.asarray(mesh.flat)

    irr = None
    if symmetric:
        irr3 = group._sg.get_irreducible_kpoints_grid(grid3)
        irr = group.k_from_3d(irr3)

    hsp = high_symmetry_points(group)
    contains, missing, hsp_out = {}, [], {}
    if not isinstance(hsp, Refusal):
        for name, kk in hsp.items():
            hsp_out[name] = kk
            hit = bool(np.any(np.abs(((points - kk) + 0.5) % 1.0 - 0.5).max(axis=1) <= ktol))
            contains[name] = hit
            if not hit:
                missing.append(name)

    return KGrid(divisions=tuple(grid3), symmetric=symmetric, mesh=mesh, irreducible=irr,
                 contains=contains, missing=tuple(missing), hsp=hsp_out)


# --------------------------------------------------------------------------
# the "irrep says / Hamiltonian does" split
# --------------------------------------------------------------------------

def _detection_grid(dim_k: int, nk: int) -> np.ndarray:
    """The detection mesh, from pythtb's own Monkhorst-Pack hypercube.

    Not a hand-rolled meshgrid: same points, one owner.  Shared by
    :func:`_validate_group` and by the lattice measurement of
    :func:`_translation_residual`, so that a translation is measured on exactly
    the k-set the rest of the group was.
    """
    from pythtb import Mesh

    mesh = Mesh(["k"] * int(dim_k))
    mesh.build_grid((int(nk),) * int(dim_k), gamma_centered=False, k_endpoints=False)
    return np.asarray(mesh.flat)


# --------------------------------------------------------------------------
# the per-call measurement context
# --------------------------------------------------------------------------

def _param_fingerprint(value) -> str:
    """A content-complete identity for one ``params`` value.

    ``repr`` is not one: numpy truncates the repr of an array past 1000 elements
    with ``...``, so two arrays that differ only beyond that point repr alike
    and would share one cached ``H(k)``.  Arrays and scalars are hashed by
    dtype, shape and bytes; anything numpy cannot hold (an object array) falls
    back to ``repr``.
    """
    arr = np.asarray(value)
    if arr.dtype != object:
        h = hashlib.blake2b(digest_size=16)
        h.update(repr((arr.dtype.str, arr.shape)).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
    return repr(value)


def _model_key(model) -> str:
    """What identifies a Hamiltonian to :class:`_MeasurementContext`: its content.

    The lattice, the orbital centres, the on-site block and the whole hopping
    list — the Hamiltonian and nothing else — hashed together from the arrays
    pythtb stores them in, at every size: a wannier90 import's 723,408
    hoppings hash in under a tenth of a second.  Two models with the same key
    have the same ``H(k)`` at every ``k``, which is the whole claim the context
    rests on: it is what lets the cross-check's realized group in the other
    environment, whose slab presentation is a *different object* with
    bit-identical content (measured, and refused if it were not), share the
    grid and the solution spaces the first one paid for.  Where no term
    carries a parameter, the model's lattice-vector blocks are kept under the
    same key (:func:`.._bloch.model_key`).
    """
    return _bloch.model_key(model)


class _MeasurementContext:
    r"""``H(k)`` on the detection grid, and one solution space per k-action.

    For the bundled pipeline that goes straight from model to realized group
    there is no reason to recompute ``H(k)`` and the solution spaces at every
    step of the descent.  One of these is created per :func:`realized_group`
    call and handed to every :func:`_validate_group` that call makes — the first
    measurement, one per descent step, the re-measurement after the hopping
    projection, and the cross-check's other environment — so that

    * ``H(k)`` on the detection mesh is evaluated **once**, by
      :class:`..symmetry._GridHamiltonians` from the model's lattice-vector
      blocks — in one call where the grid fits its memory budget, in bounded
      k-chunks where it does not — and
    * each **k-action class** is solved once, keyed by the in-plane action the
      detector poses its problem in — the rotation restricted to the periodic
      directions together with the antiunitary flag — and never by an
      operation's index, which is renumbered by every descent and differs
      between the two environments.  The translation is deliberately **not** in
      the key: :func:`..symmetry._find_intertwiners` never sees it (a
      translation's phase is absorbed into the intertwiner ``X``), which is why
      :func:`_validate_group` already shares one solve among the members of a
      class within a single call.  This lifts that same sharing to the whole
      pipeline.

    That the two environments share almost everything is not a coincidence: the
    substrate group is the ``R[2,2] = +1`` subgroup of the free-standing one and
    ``σ_h`` acts trivially on in-plane ``k``, so the two pose the *same* set of
    in-plane k-actions.

    **Invalidation is structural, not remembered.**  The key carries the
    model's content (:func:`_model_key`, whatever the model's size), the k-set,
    the model parameters and the six numbers the solution space depends on —
    ``tol``, ``cut``, ``window``, ``solver``, the detector's ceiling
    ``max_bytes``, and the floor below which its nullspace keeps a candidate,
    which a declared precision sets at the gate — and any difference rebuilds
    grid and spaces together.  The two steps of the chain that change the
    Hamiltonian, the position pass and the hopping projection, both hand back a
    new model with new content and therefore invalidate it by construction.

    The counters are measurements: ``hits`` and ``misses`` count classes reused
    and classes solved, ``rebuilds`` counts invalidations (the first bind
    included).

    ``sectors`` keeps the sectors of each model the call measured
    (:func:`.._sectors.read_sectors`), keyed by the model's content with the
    sites and spinor pairs they are read on and the tolerance they are read at:
    read where a model enters, on the first round, and kept through every later
    one and the other environment, which measure the same Hamiltonian.  A model
    the hopping projection made is a new one, and the reading made before it is
    the one its rounds carry.
    """

    __slots__ = ("_key", "grid", "spaces", "hits", "misses", "rebuilds", "sectors")

    def __init__(self):
        self._key = None
        self.grid = None
        self.spaces = {}
        self.hits = 0
        self.misses = 0
        self.rebuilds = 0
        self.sectors = {}

    def bind(self, model, k_pts, params, *, tol, cut, window, solver, max_bytes=None,
             floor=None):
        """The shared grid for this request, rebuilt when anything it rests on moved."""
        key = (_model_key(model),
               np.ascontiguousarray(np.asarray(k_pts, dtype=float)).tobytes(),
               tuple(sorted((str(k), _param_fingerprint(v))
                            for k, v in (params or {}).items())),
               float(tol), None if cut is None else float(cut),
               None if window is None else float(window), str(solver),
               None if max_bytes is None else float(max_bytes),
               None if floor is None else float(floor))
        if key != self._key:
            self._key = key
            self.grid = _GridHamiltonians(model, k_pts, params)
            self.spaces = {}
            self.rebuilds += 1
        return self.grid

    def solved(self, key):
        """The solution space already found for this k-action class, or ``None``."""
        got = self.spaces.get(key)
        if got is None:
            self.misses += 1
        else:
            self.hits += 1
        return got

    def remember(self, key, space) -> None:
        """Keep a solution space for the rest of this call."""
        self.spaces[key] = space

    def grid_for(self, model, k_pts):
        """The bound grid when it is this model on exactly this k-set, else ``None``.

        The lattice measurement (:func:`_translation_residual`) runs on the
        detection mesh by construction — see :func:`_detection_grid` — so it can
        read ``H`` off the grid the detector already evaluated instead of
        evaluating it again.  Both halves of that are **checked** rather than
        assumed: the model has to be the one the grid was built on (identity, so
        a context rebound to some other Hamiltonian can never be read here), and
        the k-set has to be the same points byte for byte.
        """
        if self.grid is None or self.grid.model is not model:
            return None
        k = np.atleast_2d(np.asarray(k_pts, dtype=float))
        if (k.shape == self.grid.k_pts.shape
                and np.ascontiguousarray(k).tobytes()
                == np.ascontiguousarray(self.grid.k_pts).tobytes()):
            return self.grid
        return None

    def record(self) -> dict:
        """The counters, so what the sharing did can be reported."""
        return {"hits": int(self.hits), "misses": int(self.misses),
                "rebuilds": int(self.rebuilds), "spaces": len(self.spaces)}


def _validate_group(model, group: GroupData, *, nk=8, tol=DEFAULT_ENERGY, cut=None,
                    k_pts=None, include_antiunitary=True, keep_intertwiners=True,
                    window=None, solver="eigen", max_bytes=None, context=None,
                    params=None, _floor=None) -> ValidatedGroup | Refusal:
    """Which operations of a group ``H(k)`` obeys, one residual each.

    Returned as :attr:`RealizedGroup.validated`; run it directly to put the same
    question to a group at a tighter ``tol``, on a different mesh, or to a
    subgroup assembled by hand.  Each candidate operation goes to
    :func:`..symmetry._find_intertwiners`, which solves ``X H(k) = H(Rk) X`` (or
    ``X H(k)* = H(Rk) X`` for an antiunitary one) over a k-grid with no tensor
    structure assumed for what the operation does to orbitals, so an empty
    solution space is a *proof* that the operation is broken rather than a
    threshold verdict
    (``docs/background/measurement.md#the-cell-and-a-verdict-on-each-operation``).

    Parameters
    ----------
    model : pythtb.TBModel
        The Hamiltonian under test; its ``hamiltonian`` is evaluated on the
        detection mesh and nothing about its geometry is re-derived here.
    group : GroupData
        The candidate operations.  Which group goes with which model is not
        checked — see the validity conditions below.  A group comes from
        :func:`realized_group`: ``realized_group(model, ...).group``.
    nk : int, optional
        Divisions of the Γ-centred detection mesh per periodic direction.
        Default 8.
    tol : float, optional
        **The declared symmetry tolerance**, absolute and in the units of H:
        the covariance gate on ``max_k ‖U H U† − H(Rk)‖_max``, and the number
        the detector's other two follow — ``cut`` defaults to it, and the
        matching ``window`` is derived from it as
        ``clip(10·sqrt(tol), 1e-3, 0.3)``.  So one declaration still sets gate,
        cut and window, and they cannot silently disagree.  A **wannier90**
        Hamiltonian is symmetric only to its own numerical defect, several
        decades above machine precision, so such a model must declare it:
        ``_validate_group(model, group, tol=1e-6)``.  Every per-operation entry
        then carries that number, which is what the verdict rests on
        (``docs/guide/numerical.md``).
    cut : float, optional
        The detector's **nullspace cut**, relative and dimensionless: a
        singular value counts as zero when ``s <= cut·‖L‖``, or when it is at
        most ``sqrt(n)·cut`` in the units of ``H``, the most a unitary candidate
        within the cut can have
        (:func:`..symmetry._find_intertwiners`).  ``None`` (the
        default) is ``tol``.  Declaring it apart says that this model's
        *arithmetic* is noisier than the physics the gate judges — the gate
        stays where the residuals are read, the cut moves to where the
        solution space is found.
    k_pts : array, optional
        Use this k-set instead of the default mesh.
    include_antiunitary : bool, optional
        Also test the group's antiunitary operations.
    keep_intertwiners : bool, optional
        Keep, for each surviving operation, the polar-extracted unitary
        (``ValidatedGroup._intertwiners``) **and the detector's whole
        solution space** (``ValidatedGroup._bases``).  That space is what
        :func:`..symmetry._build_rep`'s ``strategy="readoff"`` reads the
        representation's discrete choices off.
    window, solver : optional
        Forwarded to :func:`..symmetry._find_intertwiners`: the eigen route's
        matching window (``None`` derives it from ``tol``) and which solver
        runs (``"eigen"`` by default, ``"stacked"`` for the audit oracle).
    max_bytes : float, optional
        The memory the detector's trial basis may take at a k-point where a
        spectrum's exactly degenerate levels need more than its own
        ``512e6`` bytes: the ceiling
        :func:`..symmetry._find_intertwiners` raises its budget to there
        (``degenerate_max_bytes``).  Never below the detector's own budget,
        which is also how wide an energy window it reads on a nearly
        degenerate spectrum and which this does not move.  ``None`` (default)
        keeps that budget as the ceiling.
    context : optional
        A shared measurement context, which :func:`realized_group` creates once
        per call and hands to every measurement it makes, so that a k-action
        class posed twice is solved once.  ``None`` (the default) is this
        function's behaviour bit for bit, and what comes back is identical
        either way: the solve is a deterministic function of the model, the
        k-set and ``tol``/``cut``/``window``/``solver``/``max_bytes``.
    params : dict, optional
        Model parameter values, as ``model.hamiltonian`` takes them.

    Returns
    -------
    ValidatedGroup or Refusal
        Carries the surviving indices, the discrepancy list (with reasons and
        residuals), and :attr:`ValidatedGroup.closed` — whether the survivors
        form a subgroup under the group's own product table.

    Refusals
    --------
    ``"detector_too_large"``
        one operation's solve needs more memory than ``max_bytes`` allows, its
        spectrum's exactly degenerate pairs alone being past it: handed back
        as :func:`..symmetry._find_intertwiners` returns it, with what the
        solve needs and the ``max_bytes=`` that runs it.  A limit on memory,
        not a statement about the model, so no operation is recorded broken.

    Notes
    -----
    **Validity conditions.**

    * ``reason="no symmetry matrix"`` is a *proof* the operation is broken: no
      matrix with that k-action commutes with ``H(k)`` at all.  At a declared
      ``tol`` above machine precision it is a proof at that tolerance, not a
      claim that the operation fails to machine precision — for the defect
      itself, measure it with ``solver="stacked"``.
      ``reason="covariance"`` is weaker: a solution exists but does
      not close the gate, which usually means an almost-symmetry or a badly
      conditioned grid.
    * The extracted ``intertwiners`` carry an **arbitrary overall phase**.
      They are candidates for :func:`..symmetry.normalize_rep`, not finished
      representations: a stored operator equal to ``(−i)×`` the true one passes
      every covariance test and relabels every corep at once.
    * ``closed`` is a self-check: a surviving set that is not closed under
      composition means the gate, the grid or the model is wrong, and it is
      reported rather than patched.
    * **Verdicts are per k-action class, not per operation.**  Operations
      sharing a k-action and an antiunitarity — for a layer, every pair
      ``{g, σ_h·g}`` — pose one intertwiner problem and share one verdict, so a
      survivor of a multi-member class reads ``"undetermined"`` rather than
      ``"obeyed"``.  **"Everything survives" on a layer is therefore not a
      statement that the horizontal mirror holds.**
    * **The model/group pairing belongs to whoever makes it** and is
      deliberately unchecked: the operations are measured against whatever
      Hamiltonian is handed in.  That is what lets a cell's candidates be
      measured against a model whose own point set would give a larger group —
      but a wrong pairing is then silent.

    Examples
    --------
    A square-lattice model whose cell is ``P4/mmm`` (16 operations) but whose
    hoppings are anisotropic keeps only the 8 operations of ``Pmmm``; the
    discrepancy list names the eight that fail and their residuals.
    """
    if k_pts is None:
        k_pts = _detection_grid(int(group.dim_k), int(nk))
    k_pts = np.atleast_2d(np.asarray(k_pts, dtype=float))

    # One evaluation of H on the detection grid for the whole group: H(k) does
    # not depend on which operation is being tested, and it is summed from the
    # model's lattice-vector blocks inside a memory budget whatever the model's
    # size.  With a shared context that evaluation is for the whole *pipeline*
    # rather than for this call -- the descent's steps, the re-measurement after
    # a projection and the other environment's realized group measure the same H
    # on the same mesh.
    if context is None:
        grid_H = _GridHamiltonians(model, k_pts, params)
    else:
        grid_H = context.bind(model, k_pts, params, tol=tol, cut=cut,
                              window=window, solver=solver, max_bytes=max_bytes,
                              floor=_floor)

    keep, disc, residuals, dims, intertwiners = [], [], {}, {}, {}
    actions, solved_class, records, bases = {}, {}, {}, {}
    probe = np.eye(group.dim_k)
    # the identity, whose member is the unit matrix: it carries every H(k) onto
    # itself exactly, so its residual is that of the unit matrix, 0.0, and never
    # the rounding of a member found numerically on a nearly degenerate spectrum
    from .symmetry import _identity_position
    identity = int(group.symmetries[_identity_position(group)].ind)
    for sym in group.symmetries:
        ind = int(sym.ind)
        if sym.time_reversal and not include_antiunitary:
            continue
        action = group.k_action_of(sym)
        if action is None:
            disc.append({"ind": ind, "refuted_by": "out of plane", "residual": np.inf})
            continue
        actions[ind] = (action, bool(sym.time_reversal))
        # operations sharing (k-action, antiunitarity) pose the *identical*
        # problem, so it is solved once and `solved_on` names which operation
        key = (np.round(np.asarray(action(probe), dtype=float), 10).tobytes(),
               bool(sym.time_reversal))
        if key not in solved_class:
            # The class may already have been solved earlier in this pipeline --
            # by the measurement before a descent step, or by the other
            # environment, which poses the identical in-plane actions.
            # `solved_on` stays THIS group's own first member either way, so the
            # record says which operation of the group that was asked about.
            space = None if context is None else context.solved(key)
            if space is None:
                space = _find_intertwiners(
                    model, action, k_pts, antiunitary=bool(sym.time_reversal), tol=tol,
                    cut=cut, window=window, solver=solver, degenerate_max_bytes=max_bytes,
                    hamiltonians=grid_H, params=params, _floor=_floor)
                if isinstance(space, Refusal):
                    return space            # the detector's memory limit, as it came
                if context is not None:
                    context.remember(key, space)
            solved_class[key] = (ind, space)
        solved_on, space = solved_class[key]
        dims[ind] = space.dim
        best = float(np.min(space.residuals)) if space.dim else np.inf
        if ind == identity and space.dim:
            best = 0.0
        residuals[ind] = best
        solved = {"smallest_singular_value": float(space.singular_values[0] / space.scale),
                  "method": space.method, "n_kpts_solve": int(space.n_kpts_solve)}
        records[ind] = {"dim": int(space.dim), "residual": best, **solved,
                        "window": float(space.window), "solver": str(solver),
                        "tol": float(tol), "cut": float(space.cut),
                        "basis_size": tuple(space.basis_size),
                        "escalations": tuple(space.escalations),
                        "k_solve": int(space.k_solve), "solved_on": int(solved_on)}
        if space.dim == 0:
            disc.append({"ind": ind, "refuted_by": "no symmetry matrix",
                         "residual": np.inf})
        elif best > tol:
            disc.append({"ind": ind, "refuted_by": "covariance", "residual": best})
        else:
            keep.append(ind)
            if keep_intertwiners:
                intertwiners[ind] = (np.eye(int(model.nstate), dtype=complex)
                                     if ind == identity else
                                     space.unitaries[int(np.argmin(space.residuals))])
                # The whole space, not only the member with the best residual:
                # where `dim >= 2` the solution of tensor-structure form is
                # generally a *different* member (see ValidatedGroup._bases), and
                # `symmetry._build_rep`'s read-off needs the span to find it.
                # Stored by reference, so a k-action class costs one array.
                bases[ind] = space.basis

    closed = True
    if keep:
        prod = group.get_product_table(get_diff=False)
        pos = {int(s.ind): i for i, s in enumerate(group.symmetries)}
        kept = set(keep)
        for a in keep:
            for b in keep:
                c = int(prod[pos[a], pos[b]])
                if int(group.symmetries[c].ind) not in kept:
                    closed = False
                    break
            if not closed:
                break

    # Group the tested operations by (k-action, antiunitarity): members of one
    # class posed the identical intertwiner problem, so a class survival is
    # shared while a class failure is individually conclusive (see the
    # ValidatedGroup docstring).  Unitary and antiunitary operations with the
    # same k-action solve *different* problems and are never classmates.
    by_key = {}
    for ind, (action, tr) in actions.items():
        # the k-action is a linear callable on reduced k; probing the basis
        # rows recovers its matrix, which keys the class
        key = (np.round(np.asarray(action(probe), dtype=float), 10).tobytes(), tr)
        by_key.setdefault(key, []).append(ind)
    classes = tuple(tuple(sorted(v)) for v in
                    sorted(by_key.values(), key=lambda v: min(v)))
    kept_set = set(keep)
    verdicts = {}
    for members in classes:
        for ind in members:
            if ind not in kept_set:
                verdicts[ind] = "broken"            # class failure breaks every member
            else:
                verdicts[ind] = "obeyed" if len(members) == 1 else "undetermined"

    return ValidatedGroup(indices=tuple(keep), discrepancies=tuple(disc),
                          residuals=residuals, _nullspace_dims=dims, _intertwiners=intertwiners,
                          closed=closed, tol=float(tol), n_kpts=int(k_pts.shape[0]),
                          k_action_classes=classes, verdicts=verdicts,
                          group_label=f"{group.name} #{group.number_str}",
                          _records=records, _bases=bases, _group=group)


# --------------------------------------------------------------------------
# the realized group of a 2D model (the slab presentation, made API)
# --------------------------------------------------------------------------

#: Vacuum between a layer and its periodic image in the slab presentation, in
#: the model's own length units: only the cell sees it, never ``H(k)``.
SLAB_C = 30.0

#: How far under each site the substrate sits, in reduced coordinates of the
#: non-periodic lattice vector: any depth off the plane removes the same
#: operations, the ones that flip the layer's side.
DECORATION_DZ = 0.05

#: The vacuum between the copies of the decoupled stack, in the model's own
#: length units: the period is the layer's extent along its normal plus this.
#: No hopping crosses it, so no symmetry content can depend on it.
STACK_C = 8.0


def _is_presented_as_a_slab(model, c=SLAB_C) -> bool:
    """Whether ``model`` already *is* its own slab presentation.

    True for a ``dim_r = 3`` model whose non-periodic lattice vector lies along
    the layer's normal and is at least ``c`` long — deep enough for the model
    and a decorated cell to share one lattice, its vacuum included, which
    :func:`_assign_orbitals` requires — and for a ``dim_r = 3`` model with no
    non-periodic direction to measure.  A vector tilted off the normal is put
    on it by :func:`_slab_presentation`, however long it is.
    :func:`realized_group` asks this before calling
    :func:`_slab_presentation`, which builds a new model in every case: the
    decision is made outside, so that a model handed back untouched is the
    original object and not a copy that happens to compare equal.
    """
    if int(model.lattice.dim_r) != 3:
        return False
    per = tuple(int(d) for d in model.periodic_dirs)
    if len(per) != 2:
        return True
    off = next(d for d in range(3) if d not in per)
    lat = np.asarray(model.lat_vecs, dtype=float)
    if np.any(lat[list(per)] @ lat[off] != 0.0):
        return False
    length = float(np.linalg.norm(lat[off]))
    return length <= 0.0 or length >= float(c)


def _slab_presentation(model, *, c=SLAB_C, check=True, n_check=_bloch.CHECK_K,
                       seed=_bloch.CHECK_SEED, params=None, record=None):
    r"""Internal to :func:`realized_group`: a layer on a 3D slab lattice.

    A ``dim_r = 2`` model: the third lattice vector becomes ``c·ẑ`` and stays
    non-periodic, every orbital keeps ``z = 0``, and every hopping is copied
    with a zero third component.  A ``dim_k = 2`` Bloch phase never sees that
    direction, so ``H(k)`` comes out bit-identical — measured here rather
    than assumed.

    A ``dim_r = 3`` slab is copied with its non-periodic lattice vector along
    the layer's normal, stretched to ``c`` where it is shorter, every orbital
    kept where it is.  Stretching moves no atom: the reduced coordinate along
    the vector shrinks by exactly the factor the vector grows.  A vector tilted
    off the normal -- a slab cut from a crystal along one of the crystal's own
    lattice vectors has one -- is put on the normal at the same length,
    because a layer's group cannot depend on how that vector was chosen: there
    is no translation along it, while a symmetry finder reads the cell as a
    three-dimensional lattice on it, a lattice of copies of the layer stacked
    aslant.  Every orbital keeps its Cartesian position, so its reduced
    coordinates along the periodic directions move by ``Δ = h·t``, its reduced
    height ``h`` along the old vector times the vector's tilt ``t``, the
    in-plane part of the vector in those coordinates.  In pythtb's convention
    the Bloch phase carries ``e^{2πi k·τ}`` per orbital, so ``H(k)`` becomes
    ``Λ(k)† H(k) Λ(k)`` with ``Λ(k) = diag(e^{2πi k·Δ})``: a k-dependent gauge
    transformation, which moves no energy.  A layer whose orbitals all lie in
    the plane through the origin moves none and comes out bit-identical.

    A new model comes back every time.  ``record``, where one is handed in, is
    filled with what was measured: ``residual``, the largest difference between
    the presentation's ``H(k)`` and the model's, carried back through the move,
    over ``n_check`` random ``k`` (``0.0`` where ``check`` is off), and
    ``tilted``, ``None`` unless the vector was tilted off the normal, and then
    its ``tilt``, the largest move of an orbital's reduced coordinates
    (``max_move``), how many orbitals moved (``n_moved``), the largest
    difference of the two spectra over the same ``k`` (``spectrum_residual``)
    and a ``note``.

    Raises
    ------
    InternalError
        The model is not periodic in exactly two directions of a two- or
        three-dimensional cell: :func:`realized_group` refuses a model periodic
        in fewer before it presents anything, and reads one periodic in three
        as its own cell.  Or the presentation changed the Hamiltonian: not
        bit-identical where no orbital moved, or beyond rounding once carried
        back through the move where some did.  A presentation changes nothing
        in exact arithmetic, so that is the package disagreeing with itself.
    """
    from pythtb import Lattice, TBModel

    params = dict(params or {})
    record = {} if record is None else record
    record["residual"], record["tilted"] = 0.0, None
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    if dim_r not in (2, 3) or len(per) != 2:
        raise InternalError(
            f"a slab presentation was asked for a model periodic in {len(per)} "
            f"direction{'' if len(per) == 1 else 's'} of a {dim_r}-dimensional cell. "
            "realized_group refuses a model periodic in fewer than two directions "
            "before it presents anything, and a fully periodic one is its own cell, so "
            "only a layer is presented as a slab.")
    if dim_r == 3:
        # Already a slab -- so this is a copy, with its vacuum stretched to `c`
        # where it is thinner.  That moves no atom (the reduced coordinate
        # shrinks by exactly the factor the vector grows) and cannot touch
        # `H(k)`, since a dim_k = 2 Bloch phase never sees the non-periodic
        # direction; a cell decorated on it is given the model's own
        # vacuum whatever the stretch (`_pad_cell`).  Measured below like every
        # other presentation built here, `grow = 1` -- the plain copy --
        # included.
        lat = np.asarray(model.lat_vecs, dtype=float)
        orb = np.asarray(model.orb_vecs, dtype=float)
        orb3 = orb.copy()
        lat3 = lat.copy()
        off = next(d for d in range(3) if d not in per)
        length = float(np.linalg.norm(lat[off]))
        grow = float(c) / length if 0.0 < length < float(c) else 1.0
        plane = lat[list(per)]
        overlap = plane @ lat[off]
        move = None
        if np.any(overlap != 0.0):
            # THE VECTOR TILTED OFF THE LAYER'S NORMAL, put on it.  The layer is
            # the same layer whichever non-periodic vector it was written with,
            # and the cell a symmetry finder reads must not stack its copies
            # aslant.  Every orbital stays where it is: the vector's in-plane
            # part, `tilt` on the periodic directions, is carried into each
            # orbital's in-plane coordinates in proportion to its height, and
            # the height is re-expressed on the normal, which the vector now is.
            normal = np.cross(plane[0], plane[1])
            normal = normal / np.linalg.norm(normal)
            rise = float(np.dot(lat[off], normal))
            if rise < 0.0:
                normal, rise = -normal, -rise       # keep the model's own sense of up
            tilt = np.linalg.solve(plane @ plane.T, overlap)
            depth = length * grow
            lat3[off] = normal * depth
            move = np.outer(orb[:, off], tilt)
            for j, d in enumerate(per):
                orb3[:, d] = orb[:, d] + move[:, j]
            orb3[:, off] = orb[:, off] * (rise / depth)
        elif grow != 1.0:
            lat3[off] = lat[off] * grow
            orb3[:, off] = orb3[:, off] / grow
        out = TBModel(Lattice(lat3, orb3.tolist(), periodic_dirs=list(per)),
                      spinful=(int(model.nspin) == 2))
        out.set_onsite(model.onsite, mode="set")
        # the model's own hopping table, row for row and in one pass, so the
        # copy sums exactly the terms the model sums
        out._append_hops(*_bloch.hopping_arrays(model))
        moved = move is not None and bool(np.any(move))
        worst = spectrum = 0.0
        gate = 0.0
        if check:
            k = np.random.default_rng(seed).uniform(-0.5, 0.5,
                                                    size=(int(n_check), len(per)))
            a = np.asarray(_bloch.hamiltonian(model, k, params))
            b = np.asarray(_bloch.hamiltonian(out, k, params))
            if moved:
                # the model's H(k) carried through the move, Λ(k)† H(k) Λ(k):
                # the presentation's, to the rounding of the moved phases
                phase = np.exp(2j * np.pi * (k @ move.T))
                if int(model.nspin) == 2:
                    phase = np.repeat(phase, 2, axis=1)
                carried = np.conj(phase)[:, :, None] * a * phase[:, None, :]
                worst = float(np.abs(carried - b).max())
                gate = _MOVED_PRESENTATION_RELATIVE * max(float(np.abs(a).max()), 1e-300)
                spectrum = float(np.abs(np.linalg.eigvalsh(a) - np.linalg.eigvalsh(b)).max())
            else:
                worst = float(np.abs(a - b).max())
            if worst > gate:                        # pragma: no cover - a defect, raised
                raise InternalError(
                    ("this slab presentation, carried back through the move of its "
                     "orbitals, is not the model's Hamiltonian to rounding: "
                     f"max|Lambda H_slab(k) Lambda^dagger - H(k)| = {worst:.3g} over "
                     f"{n_check} random k. The hoppings were copied in order, the "
                     "non-periodic lattice vector was put on the layer's normal and "
                     "every orbital kept where it was, so in pythtb's convention H(k) "
                     "changed by the k-dependent gauge transformation "
                     "Lambda(k) = diag(e^{2 pi i k.Delta}) alone, Delta each orbital's "
                     "move." if moved else
                     "this slab presentation is not bit-identical to the model: "
                     f"max|H_slab(k) - H(k)| = {worst:.3g} over {n_check} random k. The "
                     "hoppings and the orbitals were copied in order, and at most the "
                     "non-periodic lattice vector and the reduced coordinate along it "
                     "were scaled, by reciprocal factors, or the vector put on the "
                     "layer's normal with every orbital in the plane through the "
                     "origin, so no atom moved and a dim_k = 2 Hamiltonian cannot "
                     "see it.")
                    + f" It is held to {gate:.3g}, and a presentation changes nothing, "
                    "so this is the package disagreeing with itself, not a property "
                    "of the model.")
        record["residual"] = worst
        if move is not None:
            n_moved = int(np.count_nonzero(np.abs(move).max(axis=1) > 0.0))
            biggest = float(np.abs(move).max(initial=0.0))
            record["tilted"] = {
                "tilt": tuple(float(x) for x in tilt),
                "max_move": biggest, "n_moved": n_moved,
                "spectrum_residual": spectrum,
                "note": (
                    "the slab's non-periodic lattice vector was tilted off the layer's "
                    "normal, its in-plane part "
                    f"({', '.join(f'{float(x):.6g}' for x in tilt)}) on the periodic "
                    "directions, and it is put on the normal, every orbital kept where "
                    "it is: a layer's group cannot depend on how that vector was "
                    "chosen, since there is no translation along it. "
                    + (f"{n_moved} of {int(model.norb)} orbitals lie off the plane "
                       "through the origin, and each one's reduced coordinates in the "
                       "plane moved by its reduced height times that tilt, at most "
                       f"{biggest:.3g}: a k-dependent gauge transformation of H(k), "
                       "which moves no energy"
                       + (f" (measured, {spectrum:.3g})." if check else ".")
                       if n_moved else
                       "Every orbital lies in the plane through the origin, so none "
                       "moved and H(k) is unchanged bit for bit."))}
        return out

    lat2 = np.asarray(model.lat_vecs, dtype=float)
    lat3 = np.zeros((3, 3))
    lat3[:2, :2] = lat2
    lat3[2] = [0.0, 0.0, float(c)]
    orb3 = np.zeros((int(model.norb), 3))
    orb3[:, :2] = np.asarray(model.orb_vecs, dtype=float)

    out = TBModel(Lattice(lat3, orb3.tolist(), periodic_dirs=[0, 1]),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    # the model's own hopping table, row for row and in one pass, each lattice
    # vector given a zero third component
    amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    R3 = np.zeros((R.shape[0], 3), dtype=int)
    R3[:, :dim_r] = R
    out._append_hops(amps, i_idx, j_idx, R3)

    if check:
        k = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(int(n_check), 2))
        a = np.asarray(_bloch.hamiltonian(model, k, params))
        b = np.asarray(_bloch.hamiltonian(out, k, params))
        worst = float(np.abs(a - b).max())
        if worst != 0.0:
            raise InternalError(
                "the 3D presentation is not bit-identical to the 2D model: "
                f"max|H3(k) - H2(k)| = {worst:.3g} over {n_check} random k. The "
                "presentation is a relabelling of the same Hamiltonian and changes "
                "nothing, so this is the package disagreeing with itself, not a "
                "property of the model.")
        record["residual"] = worst
    return out


def stack_presentation(model, *, c=STACK_C, params=None):
    """The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every ``k_z``.

    The three-dimensional crystal a layer's rows at ``k_z = ½`` are measured
    on: the layer's own lattice and a third vector along its normal, every
    orbital at its own height, every hopping within one copy.  Its
    ``H(k, k_z)`` is the layer's ``H(k)`` up to the phases of the orbitals'
    heights, ``Λ(k_z)† H(k) Λ(k_z)`` with ``Λ`` diagonal, which each call
    measures, so its spectrum and symmetry are the layer's at every ``k_z``
    and ``c`` changes neither.

    Parameters
    ----------
    model : pythtb.TBModel
        A layer: periodic in two directions, in a cell of two or three.
    c : float, optional
        The vacuum between the copies, in the model's length units; the period
        is the layer's extent along its normal plus ``c``.  Default
        :data:`STACK_C`, ``8.0``.
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    pythtb.TBModel or Refusal
        The stack, periodic in three directions.  A ``Refusal`` in its place is
        falsy.

    Raises
    ------
    ..refusal.InternalError
        The stack's ``H`` is not the layer's up to the heights' phases: the
        package disagreeing with itself.

    Refusals
    --------
    ``not_a_layer``
        The model is not periodic in exactly two directions of a two- or
        three-dimensional cell.
    ``hopping_across_the_stack``
        A hopping runs along the non-periodic direction, so the model is not
        one decoupled layer.

    Examples
    --------
    >>> from pythtb.models import kane_mele                       # doctest: +SKIP
    >>> st = stack_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))   # doctest: +SKIP
    >>> st.lattice.dim_k, tuple(st.periodic_dirs)                  # doctest: +SKIP
    (3, (0, 1, 2))
    """
    from pythtb import Lattice, TBModel

    params = dict(params or {})
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    if int(model.lattice.dim_k) != 2 or len(per) != 2 or dim_r not in (2, 3):
        return Refusal(
            "not_a_layer",
            f"the decoupled stack is built from a layer -- two periodic directions in a "
            f"2D or 3D cell; this model has dim_r = {dim_r}, dim_k = "
            f"{int(model.lattice.dim_k)} and periodic directions {per}.",
            value=float(int(model.lattice.dim_k)), threshold=2.0,
            context={"dim_r": dim_r, "dim_k": int(model.lattice.dim_k),
                     "periodic_dirs": per})

    lat = np.asarray(model.lat_vecs, dtype=float)
    plane = np.zeros((2, 3))
    plane[:, :dim_r] = lat[list(per)]
    # the normal that makes the stack's cell right-handed, as pythtb requires of
    # every cell, whichever of a slab's axes is its vacuum: a slab whose vacuum is
    # its middle axis has that vector opposite to this normal, and a cell built
    # along the vacuum vector would be left-handed
    normal = np.cross(plane[0], plane[1])
    normal = normal / np.linalg.norm(normal)
    stack_dir = None
    if dim_r == 3:
        stack_dir = next(d for d in range(3) if d not in per)

    # every hopping must live inside the plane, structurally
    amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    n_hop = int(R.shape[0])
    worst_hop = int(np.abs(R[:, stack_dir]).max()) if dim_r == 3 and n_hop else 0
    if worst_hop:
        return Refusal(
            "hopping_across_the_stack",
            f"{worst_hop} lattice units of hopping run along the non-periodic direction, "
            "so this is not one decoupled layer. Stacking it would either drop those "
            "hoppings or give H(k) a k_z it does not have; neither is the same "
            "Hamiltonian in another presentation.",
            value=float(worst_hop), threshold=0.0,
            context={"n_hoppings": n_hop, "stacking_direction": stack_dir})

    # every orbital at its own height along the normal: the layer's extent
    # there, plus the vacuum c, is the period
    cart = np.asarray(model.orb_vecs, dtype=float) @ lat
    if cart.shape[1] == 2:
        cart = np.column_stack([cart, np.zeros(cart.shape[0])])
    height = cart @ normal
    top = float(height.max()) if height.size else 0.0
    bottom = float(height.min()) if height.size else 0.0
    extent = top - bottom
    period = extent + float(c)

    lat3 = np.zeros((3, 3))
    lat3[:2] = plane
    lat3[2] = normal * period
    # The new a1, a2 ARE the model's own, so the in-plane reduced coordinates
    # are copied rather than recomputed -- a projection would round them at
    # 1e-17 and the bit-identity gate below would (rightly) see it.  The only
    # correction is the in-plane part of a non-periodic third vector, which is
    # exactly zero for every presentation this package builds and is measured
    # rather than assumed.
    orb = np.asarray(model.orb_vecs, dtype=float)
    orb3 = np.zeros((int(model.norb), 3))
    orb3[:, :2] = orb[:, list(per)]
    if stack_dir is not None:
        overlap = plane @ np.asarray(lat[stack_dir], dtype=float)
        if np.any(overlap != 0.0):
            orb3[:, :2] += np.outer(orb[:, stack_dir],
                                    np.linalg.solve(plane @ plane.T, overlap))
    if extent > 0.0:
        # each orbital's height from the layer's middle plane, the frame the
        # layer's group is identified in; a flat layer's are all exactly zero
        orb3[:, 2] = (height - 0.5 * (top + bottom)) / period

    out = TBModel(Lattice(lat3, orb3.tolist(), periodic_dirs=[0, 1, 2]),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    # the model's own hopping table, row for row and in one pass, each lattice
    # vector written on the two in-plane directions; the component dropped is
    # zero on every row (gated above), so no two rows become one term
    R3 = np.zeros((n_hop, 3), dtype=int)
    R3[:, 0] = R[:, per[0]]
    R3[:, 1] = R[:, per[1]]
    out._append_hops(amps, i_idx, j_idx, R3)

    # the identity, measured on every call, on the k-set every presentation
    # here is measured on
    rng = np.random.default_rng(_bloch.CHECK_SEED)
    k2 = rng.uniform(-0.5, 0.5, size=(_bloch.CHECK_K, 2))
    a = np.asarray(_bloch.hamiltonian(model, k2, params))
    stack = _bloch.evaluator(out, params)
    phase = 2j * np.pi * orb3[:, 2]
    if int(model.nspin) == 2:
        phase = np.repeat(phase, 2)
    # the phases round where the heights are not zero, and nowhere else
    gate = 0.0 if extent == 0.0 else _MOVED_PRESENTATION_RELATIVE * float(np.abs(a).max())
    for kz in (0.0, 0.25, 0.5, *rng.uniform(-0.5, 0.5, size=3)):
        k3 = np.concatenate([k2, np.full((_bloch.CHECK_K, 1), float(kz))], axis=1)
        b = np.asarray(stack(k3))
        lam = np.exp(phase * float(kz))
        d = float(np.abs(np.conj(lam)[None, :, None] * a * lam[None, None, :] - b).max())
        held = 0.0 if float(kz) == 0.0 else gate
        if d > held:
            raise InternalError(
                "the decoupled stack is not the layer: max|H_stack(k, k_z) - "
                f"Lambda^dagger H_layer(k) Lambda| = {d:.3g} at k_z = {float(kz)}, "
                f"against {held:.3g}, over {_bloch.CHECK_K} random k, Lambda the phases "
                "of the orbitals' heights. The presentation is the same Hamiltonian and "
                "changes nothing but that gauge, so this is the package disagreeing with "
                "itself, not a property of the model.")
    return out


def _layer_on_stack(group, assignment, stack) -> tuple:
    r"""A layer's group and orbital sites, carried onto the layer's decoupled stack.

    :func:`stack_presentation` repeats the layer along its normal with no
    hopping between the copies, and an operation of the layer carries each copy
    onto itself as it carries the layer, so the stack's group is the layer's:
    the same operations, each the same Cartesian rotation and translation,
    written in the stack's cell, and the same sites, each operation moving them
    as it moves the layer's.  Nothing is identified again.  The stack's cell
    keeps the layer's in-plane lattice vectors and puts its normal at another
    length, reversed where the slab's vacuum is its middle axis (the stack's
    cell is right-handed), so every operation keeps its reduced form, up to the
    order and the sense in which the two cells list their axes, and the tables'
    setting is carried in that same order: a k-point at ``k_z = ½`` of the
    layer's group is one of the stack's own zone.  Both cells are written about
    the layer's middle plane, and a site off it is carried at its height from
    that plane, so the lattice vector each operation moves a site by is read
    again in the stack's cell.

    ``group`` and ``assignment`` are the group a layer's representation was
    built in, on its slab cell, and the orbital sites in it; ``stack`` is that
    layer's decoupled stack.  Returns ``(group, assignment)`` on the stack's
    cell: three periodic directions and none along a vacuum.

    Raises
    ------
    ..refusal.InternalError
        Where the two cells are not one layer's, or an operation, a site or a
        lattice shift does not read as one of the stack's cell.  The stack is
        built from the layer's own lattice and orbitals, so either is a defect
        here.
    """
    old = np.asarray(group.lattice, dtype=float)
    new = np.asarray(stack.lat_vecs, dtype=float)
    # reduced coordinates of the layer's cell, to Cartesian, to the stack's
    carry = np.linalg.solve(new.T, old.T)
    # the order of the axes alone, the normal's length divided out
    order = carry / np.linalg.norm(carry, axis=0)[None, :]
    axes = np.rint(order)
    if (float(np.abs(order - axes).max()) > _CARRY_TOL
            or not np.array_equal(np.abs(axes).sum(axis=0), np.ones(3))
            or not np.array_equal(np.abs(axes).sum(axis=1), np.ones(3))
            or any(abs(float(carry[:, int(d)] @ axes[:, int(d)]) - 1.0) > _CARRY_TOL
                   for d in group.periodic_dirs)):
        raise InternalError(
            f"#{group.number_str}: the decoupled stack's cell is not the layer's cell "
            "with its normal at another length or reversed -- the cell the group was "
            f"measured on has lattice {np.round(old, 6).tolist()} and the stack's "
            f"{np.round(new, 6).tolist()} -- so its operations cannot be carried "
            "onto the stack. The stack is built from the layer's own lattice, so "
            "this is a defect here, not a property of the model.")
    inverse = np.linalg.inv(carry)
    padded = [int(d) for d in group.vacuum_dirs]

    def _miss(x):
        x = np.asarray(x, dtype=float)
        return float(np.abs(x - np.rint(x)).max()) if x.size else 0.0

    def _centred(x):
        """Reduced coordinates, the vacuum one taken about the middle plane, in [-½, ½)."""
        x = np.array(x, dtype=float, copy=True)
        if padded and x.size:
            # a whole cell subtracted, so a coordinate already there is untouched
            x[..., padded] -= np.floor(x[..., padded] + 0.5)
        return x

    def _placed(x):
        """Positions carried onto the stack's cell at their heights, wrapped into the cell."""
        y = _centred(x) @ carry.T
        if y.size:
            y[..., 2] = y[..., 2] % 1.0
        return y

    moved, worst = [], 0.0
    for sym in group.symmetries:
        rotation = carry @ np.asarray(sym.rotation, dtype=float) @ inverse
        own = _centred(np.asarray(sym.translation, dtype=float))
        translation = carry @ own
        # an operation of a layer moves nothing along its normal, so its
        # translation reads alike whatever length the normal has
        worst = max(worst, _miss(rotation), _miss(translation - axes @ own))
        new_sym = copy.copy(sym)
        for cached in ("rotation_cart", "translation_cart", "lattice_inv",
                       "reciprocal_lattice", "det_cart", "det", "rotation_inv",
                       "is_identity"):
            new_sym.__dict__.pop(cached, None)
        new_sym.rotation = np.rint(rotation).astype(int)
        new_sym.translation = translation
        new_sym.real_lattice = new
        moved.append(new_sym)
    sites = _placed(group.site_positions)
    placed_sites = (None if assignment.site_positions is None
                    else _placed(assignment.site_positions))
    if placed_sites is None:
        shifts = np.asarray(assignment.lattice_shift, dtype=float) @ carry.T
    else:
        # the lattice vector each operation moves a site by, R tau_i + t - tau_g(i),
        # read again in the stack's cell, where the sites sit at their heights
        perm = np.asarray(assignment.permutations, dtype=int)
        shifts = np.stack([placed_sites @ np.asarray(s.rotation, dtype=float).T
                           + np.asarray(s.translation, dtype=float) - placed_sites[perm[p]]
                           for p, s in enumerate(moved)])
    worst = max(worst, _miss(shifts))
    if worst > _CARRY_TOL:
        raise InternalError(
            f"#{group.number_str}: an operation of the layer, or a lattice vector it "
            f"moves a site by, reads in the decoupled stack's cell {worst:.3g} away from "
            "one of that cell. The stack shares the layer's in-plane lattice and every "
            "operation of a layer keeps its plane, so this is a defect here, not a "
            "property of the model.")
    ref = axes @ np.asarray(group._refUC, dtype=float)
    shift = axes @ np.asarray(group._shiftUC, dtype=float)
    sg = copy.copy(group._sg)
    for cached in ("lattice_inv", "reciprocal_lattice", "translations_cart"):
        sg.__dict__.pop(cached, None)
    sg.real_lattice, sg.symmetries, sg.positions = new, moved, sites
    sg.refUC, sg.shiftUC = ref, shift
    on_stack = replace(group, symmetries=list(moved), lattice=new, site_positions=sites,
                       periodic_dirs=(0, 1, 2), vacuum_dirs=(), dim_k=3, dim_r=3,
                       _refUC=ref, _shiftUC=shift, _sg=sg)
    placed = replace(assignment, lattice_shift=np.rint(shifts).astype(int),
                     site_positions=placed_sites)
    return on_stack, placed


def _off_plane_direction(model3d) -> int:
    """The one non-periodic lattice direction of a slab cell (``2`` if none is)."""
    per = tuple(int(d) for d in model3d.periodic_dirs)
    return next((d for d in range(3) if d not in per), 2)


def _substrate_decorations(model3d, typat=None, *, dz=DECORATION_DZ) -> tuple:
    r"""Internal to :func:`realized_group`: the substrate under every site.

    A layer lying on something has one side different from the other, and the
    way to say that to a symmetry finder is to put the substrate in the cell:
    one non-orbital atom of its own species at the site's own height minus
    ``dz``, under every one of the model's sites.  Every operation with
    ``R[2,2] = −1`` would then carry a shadow onto the layer, so every such
    operation is gone — ``σ_h``, the in-plane twofolds, ``S_n`` and
    three-dimensional inversion with them — while the in-plane group and the
    layer's own geometry, buckling included, are untouched.  What is left is
    the subgroup of the free-standing layer group that preserves the layer's
    normal.  Under
    *every* site rather than once at the origin, because a single shadow would
    also remove every operation that does not fix the origin, which is a
    statement about the coordinates the model was written in.

    Returns
    -------
    tuple
        ``((position, label), ...)`` — the decoration sequence
        :func:`_decorated_cell` takes, one entry per distinct site.
    """
    norb = int(model3d.norb)
    labels = [0] * norb if typat is None else list(typat)
    if all(isinstance(t, (int, np.integer)) for t in labels):
        shadow = max(int(t) for t in labels) + 1
    else:                                # labels are not numeric: use a reserved name
        shadow = "__substrate__"
    off = _off_plane_direction(model3d)
    step = np.zeros(3)
    step[off] = float(dz)
    seen, out = [], []
    for row in np.asarray(model3d.orb_vecs, dtype=float):
        p = np.asarray(row, dtype=float).reshape(-1).copy()
        if any(float(np.abs((p - q + 0.5) % 1.0 - 0.5).max()) <= 1e-9 for q in seen):
            continue
        seen.append(p)
        out.append(([float(x) for x in (p - step)], shadow))
    return tuple(out)


def _decorated_cell(model3d, typat=None, *, dz=DECORATION_DZ, environment=None,
                    decorations=None):
    r"""Internal to :func:`realized_group`: the cell carrying the crystal field.

    The point set the group is identified on: the model's orbitals, plus any
    species that state a distortion the positions do not show.  Geometry only —
    it never carries a Hamiltonian, and the model is not modified.

    ``environment`` chooses the base: ``"substrate"`` (the default) puts one
    shadow species under every site (:func:`_substrate_decorations`), leaving
    the subgroup that preserves the layer's normal, and ``"free-standing"``
    puts nothing under
    the layer.  ``decorations`` adds on top of that base, and in 3D — where a
    model is its own cell and there is no base — it is the whole mechanism: one
    atom at the strengthened bond's midpoint lowers diamond to ``R-3m``, which
    is the group the distorted Fu-Kane-Mele Hamiltonian obeys.

    Raises
    ------
    ValueError
        When ``typat`` does not carry one entry per orbital.

    Returns
    -------
    (pythtb.TBModel, list)
        The decorated cell and the species list to identify it with.  Pass the
        *model's own* ``typat`` to :func:`_assign_orbitals`, not this one.
    """
    from pythtb import Lattice, TBModel

    norb = int(model3d.norb)
    labels = [0] * norb if typat is None else list(typat)
    if len(labels) != norb:
        raise ValueError(f"typat must have one entry per orbital ({norb}), "
                         f"got {len(labels)}")
    extra_atoms = () if decorations is None else tuple(decorations)
    # A fully periodic cell has no side to make different: it is its own cell in
    # every environment, and only `decorations=` adds to it.
    periodic = tuple(int(d) for d in model3d.periodic_dirs)
    is_slab = any(d not in periodic for d in range(3))
    base_environment = "substrate" if environment is None else environment
    base = (_substrate_decorations(model3d, labels, dz=dz)
            if is_slab and base_environment == "substrate" else ())

    pos = np.asarray(model3d.orb_vecs, dtype=float).tolist()
    extra = []
    for p, lab in base + extra_atoms:
        pos.append([float(x) for x in np.asarray(p, dtype=float).reshape(-1)])
        extra.append(lab)
    cell = TBModel(Lattice(np.asarray(model3d.lat_vecs, dtype=float), pos,
                           periodic_dirs=list(model3d.periodic_dirs)), spinful=False)
    # the model's own orbitals lead the cell, and a cell built on a cell keeps
    # the count of the model it was built on: the vacuum is theirs
    setattr(cell, _OWN_ORBITALS, int(getattr(model3d, _OWN_ORBITALS, norb)))
    return cell, labels + extra


# --------------------------------------------------------------------------
# on a substrate: the subgroup of the layer group that preserves the layer's normal
# --------------------------------------------------------------------------

def _operation_key(sym, tol=1e-6) -> tuple:
    r"""One operation as a comparable key: ``(rotation, translation mod 1, coset)``.

    A space-group operation is its integer rotation together with its
    translation **modulo the lattice**, and the antiunitary coset is a third
    coordinate rather than a detail — a primed operation is not the unprimed one.
    Used to match the two environments' selections against each other, which is
    the only place two independently identified groups of the *same* cell have
    to be compared operation for operation.

    The reduction snaps ``1 − ε`` and ``ε`` alike onto ``0``, so a translation
    that spglib returns just below one is the same key as one just above zero.
    """
    R = np.rint(np.asarray(sym.rotation, dtype=float)).astype(int)
    t = np.asarray(sym.translation, dtype=float).reshape(-1)

    def _reduced(v):
        w = float(v) % 1.0
        return 0.0 if min(w, 1.0 - w) < tol else round(w, 6)

    return (tuple(int(x) for x in R.ravel()), tuple(_reduced(x) for x in t),
            bool(sym.time_reversal))


def _side_preserving(sym, padded) -> bool | None:
    r"""Does this operation keep the layer's side?  ``None`` if it is not a layer op.

    A layer operation is block-diagonal in the non-periodic direction —
    ``diag(g₂, ±1)`` — and the sign is what a substrate decides: ``+1`` keeps
    each side where it is, ``−1`` exchanges them and a substrate removes it.
    ``None`` where the rotation mixes the plane with the vacuum direction, which
    is not a layer operation at all.
    """
    R = np.rint(np.asarray(sym.rotation, dtype=float)).astype(int)
    d = int(padded)
    per = [i for i in range(3) if i != d]
    if (R[per, d] != 0).any() or (R[d, per] != 0).any():
        return None
    if abs(int(R[d, d])) != 1:                  # pragma: no cover - integer cell
        return None
    return int(R[d, d]) == 1


def _environment_record(free_group, sub_group) -> dict:
    r"""The two environments of one layer, compared — the consistency gate.

    **A substrate means a subgroup.**  It removes every operation that flips the
    layer's side and nothing else, so the group of a layer on a substrate is the
    ``R[2,2] = +1`` **subgroup** of the free-standing layer group of the *same*
    crystal — the same point set, the same coordinates, one species added below
    it.  That is a statement two independent selections can be held to, and this
    holds them to it, operation for operation:

    (a) every operation of the substrate group is an operation of the
        free-standing group that preserves the layer's normal, matched by rotation and translation modulo the
        lattice (:func:`_operation_key`) — anything else is ``missing``;
    (b) every operation of the free-standing group that preserves the layer's
        normal is in the
        substrate group — anything else is ``extra``;
    (c) both cosets, unitary and antiunitary, are compared the same way, since
        the key carries ``time_reversal``;
    (d) no operation of the substrate group flips the side (``flipping``), and
        no operation of either group mixes the plane with the vacuum direction
        (``unmapped``), which would not be a layer operation at all.

    Nothing is projected and every operation carries its own lift, which is what
    makes the subgroup statement exact rather than approximate: a buckled
    honeycomb on a substrate keeps precisely the six operations of its
    ``p-3m1`` that preserve the layer's normal, and reads ``p3m1``, not the ``p6mm`` that the in-plane image
    of a point set flattened onto one plane would report.

    Returns the record; ``agrees`` is ``False`` on any failure and ``None``
    where the two cells cannot be compared at all (no single vacuum direction).
    """
    lp = tuple(int(d) for d in free_group.vacuum_dirs)
    wp = tuple(int(d) for d in sub_group.vacuum_dirs)
    head = {"free_standing": {"number_str": str(free_group.number_str),
                              "name": str(free_group.name),
                              "size": int(free_group.size)},
            "substrate": {"number_str": str(sub_group.number_str),
                          "name": str(sub_group.name), "size": int(sub_group.size)}}
    if len(lp) != 1 or lp != wp:
        return dict(head, agrees=None, outcome=None, rows=(), missing=(), extra=(),
                    flipping=(), unmapped=(),
                    note=("the two cells do not share a single non-periodic direction "
                          f"(free-standing {lp}, substrate {wp}), so which operations "
                          "preserve the layer's side is not defined and the two "
                          "environments were not compared"))
    padded = int(lp[0])
    free_lat = np.asarray(free_group.lattice, dtype=float)
    sub_lat = np.asarray(sub_group.lattice, dtype=float)

    unmapped, expected = [], {}
    for s in free_group.symmetries:
        side = _side_preserving(s, padded)
        if side is None:
            unmapped.append({"group": "free-standing", "ind": int(s.ind),
                             "name": _record_name(s, free_lat)})
        elif side:
            expected[_operation_key(s)] = s

    got, flipping = {}, []
    for s in sub_group.symmetries:
        side = _side_preserving(s, padded)
        if side is None:
            unmapped.append({"group": "substrate", "ind": int(s.ind),
                             "name": _record_name(s, sub_lat)})
            continue
        if not side:
            flipping.append({"ind": int(s.ind), "name": _record_name(s, sub_lat)})
            continue
        got[_operation_key(s)] = s

    # each substrate operation paired with the free-standing one it is: the same
    # rotation and coset, and the same translation modulo the lattice to the
    # key's own tolerance -- compared as numbers, so two translations a rounding
    # apart that the keys' six decimals would put on two sides of a boundary
    # (a group written about an origin fitted to noisy centres has such
    # translations) are still one operation
    def _same(a, b):
        if a[0] != b[0] or a[2] != b[2]:
            return False
        d = (np.asarray(got[a].translation, dtype=float).reshape(-1)
             - np.asarray(expected[b].translation, dtype=float).reshape(-1))
        return float(np.abs(d - np.rint(d)).max()) < 1e-6

    paired, free_keys = {}, sorted(expected)
    for key in sorted(got):
        match = next((k for k in free_keys if k not in paired.values() and _same(key, k)),
                     None)
        if match is not None:
            paired[key] = match

    rows = []
    for key in sorted(got):
        s = got[key]
        f = expected.get(paired[key]) if key in paired else None
        rows.append({"substrate": int(s.ind),
                     "free_standing": None if f is None else int(f.ind),
                     "name": _record_name(s, sub_lat),
                     "rotation": key[0], "translation": key[1],
                     "time_reversal": bool(s.time_reversal)})
    missing = tuple({"ind": int(got[k].ind), "name": _record_name(got[k], sub_lat)}
                    for k in sorted(got) if k not in paired)
    extra = tuple({"ind": int(expected[k].ind), "name": _record_name(expected[k], free_lat)}
                  for k in sorted(set(expected) - set(paired.values())))
    agrees = not (missing or extra or flipping or unmapped)
    return dict(head, agrees=bool(agrees), outcome=("agree" if agrees else None),
                rows=tuple(rows), missing=missing, extra=extra,
                flipping=tuple(flipping), unmapped=tuple(unmapped),
                note=("the substrate group is exactly the subgroup of the free-standing "
                      "layer group that preserves the layer's normal -- every operation with "
                      "R[2,2] = +1 and no other, in both cosets")
                if agrees else "")


def _raise_environment_contradiction(record, held=1):
    """The two environments contradicting the subgroup statement — raised, never returned.

    A substrate removes every operation that flips the layer's side and nothing
    else, so the substrate group of a layer is the subgroup of its
    free-standing group that preserves the layer's normal, and two selections of one Hamiltonian that
    contradict it are this package disagreeing with itself.  Where one of them
    returned the candidate the tie rule keeps among several its measurement
    could not choose between, a failure that another pair of the candidates
    does not have is that choice and not a contradiction, and it is an outcome
    of the cross-check rather than this.  ``held`` is how many pairs of the two
    environments' candidates were held to the statement, the pair returned
    among them.

    Raises
    ------
    ..refusal.InternalError
        Always, naming the operations each side carries that the other does not.
    """
    free, sub = record["free_standing"], record["substrate"]

    def _list(rows):
        return ", ".join(f"{r['name']} (#{r['ind']})" for r in rows[:4]) + (
            ", ..." if len(rows) > 4 else "")

    bits = []
    if record["missing"]:
        bits.append(f"{len(record['missing'])} substrate operations are not "
                    "operations of the free-standing group that preserve the layer's "
                    "normal: "
                    + _list(record["missing"]))
    if record["extra"]:
        bits.append(f"{len(record['extra'])} operations of the free-standing group "
                    "that preserve the layer's normal are missing from the substrate "
                    "group: "
                    + _list(record["extra"]))
    if record["flipping"]:
        bits.append(f"{len(record['flipping'])} operations of the substrate group flip "
                    "the layer's side (R[2,2] = -1), which a substrate removes by "
                    "construction: " + _list(record["flipping"]))
    if record.get("unmapped"):
        bits.append(f"{len(record['unmapped'])} operations mix the plane with the "
                    "vacuum direction and are not layer operations at all: "
                    + _list(record["unmapped"]))
    raise InternalError(
        f"the free-standing realized group ({free['name']} #{free['number_str']}) and the "
        f"one on a substrate ({sub['name']} #{sub['number_str']}) do not describe the "
        f"same Hamiltonian, the operations counted as {_PER_COSET}: "
        + "; ".join(bits)
        + ". A substrate removes every operation that flips the layer's side and "
        "nothing else, so the substrate group is the R[2,2] = +1 subgroup of the "
        "free-standing layer group of the same crystal -- the two selections agreeing "
        "is not a coincidence to be checked politely, it is what makes either of them "
        "the group of this model."
        + (f" Where the tie rule chose, the subgroups it did not keep were held to it "
           f"too, {int(held) - 1} more pair{'' if int(held) == 2 else 's'} of the two "
           "environments' candidates, and it holds for none." if int(held) > 1 else "")
        + " Neither is returned: two selections of one "
        "Hamiltonian contradicting it is a defect here, not a measurement about the "
        "model, and `realized_group(..., cross_check=False)` shows each environment's "
        "realized group on its own.")


def _precision_holding(steps, scale_of):
    r"""The precision a model's numbers would have to be declared at for every operation a descent broke to hold.

    An operation broken by its own form beyond the covariance gate -- the
    member of its solution space nearest that form, its part in the form,
    carries ``H(k)`` onto ``H(gk)`` only to the entry's ``covariance`` -- holds
    under a declared precision ``p`` whose gate, ``10 · p · max|H|``, reaches
    that far, as spin blocks written in a frame turned against the lattice by
    an angle within the model's own precision do.  ``scale_of`` measures
    ``max|H|``, and is asked only where every operation broken was broken so.

    A precision is named only where the detector can read the model at it: its
    matching window, ``clip(10·sqrt(10·p), 1e-3, 0.3)``
    (:func:`.._tolerance.eigen_window`), below its ceiling.  A defect that
    needs more is no model's numerical precision -- spin blocks written along
    another axis, or turned by a sizeable angle, carry ``H`` that far -- and
    none is named for it.

    Returns ``(worst, p)``: the largest such covariance, in the units of ``H``,
    and the least precision whose gate holds it, rounded up to two figures;
    ``None`` where an operation was broken by anything else, where nothing
    was, or where that precision is past the window's ceiling.
    """
    broken = [b for st in steps for b in st["dropped"]
              if b["refuted_by"] not in ("closure", "tie")]
    if not broken or any(b["refuted_by"] != "tensor structure" or b.get("covariance") is None
                         for b in broken):
        return None
    scale = float(scale_of())
    if not scale > 0.0:
        return None
    worst = max(float(b["covariance"]) for b in broken)
    least = worst / (HEADROOM * scale)
    figure = 10.0 ** (math.floor(math.log10(least)) - 1)
    least = math.ceil(least / figure) * figure
    if eigen_window(HEADROOM * least) >= _EIGEN_WINDOW_CEIL:
        return None
    return worst, least


# --------------------------------------------------------------------------
# the unitary descent: the cell lowered to the group the Hamiltonian obeys
# --------------------------------------------------------------------------

#: The generic point whose orbit is added to a cell to lower its group.  Its
#: reduced coordinates are deliberately unlike any Wyckoff coordinate — off
#: every axis, plane and diagonal — so that its **stabilizer is trivial**: an
#: operation preserving the orbit of such a point belongs to the set the orbit
#: was generated from, which is what makes the construction exact rather than
#: approximate.  The third entry is used only along a genuinely periodic
#: direction; a vacuum one is placed by the two numbers below.
_DESCENT_POINT = (0.1234, 0.2718, 0.0577)

#: Along a **vacuum** direction the generic point sits at this fraction of the
#: cell's own half-extent, measured from the cell's own mid-plane.  Staying
#: *inside* the extent is what keeps :func:`_pad_cell`'s re-centring
#: bit-identical to the parent cell's -- its vacuum is the model's own whatever
#: the cell holds -- so the descended group is read in the same frame the parent
#: was.
_DESCENT_PAD_FRACTION = 0.1234

#: ... and where the cell has **no** extent along a vacuum direction (every
#: point in one plane, which is what a free-standing flat layer leaves), this
#: reduced offset is used instead: a point in the plane would leave ``σ_h``
#: unbroken by construction and the descent could then never remove it.
_DESCENT_PAD_OFFSET = 0.0577

#: Species label of the added orbit — one ``cell_typat`` value, distinct from
#: every orbital and decoration species (a suffix is appended if it collides),
#: so the orbit is one species and its own orbit under the kept set.
_DESCENT_SPECIES = "descent"

#: The generic point **step 1** carries its orbit from, and its species.  A cell
#: the declared content has already reduced is the base every later descent
#: builds on, so the two orbits share a cell and must not share a point: these
#: periodic coordinates are as unlike :data:`_DESCENT_POINT`'s as they are
#: unlike any Wyckoff coordinate, and no point of one orbit is a point of the
#: other.  The coordinate along the vacuum is placed by the same two numbers
#: above, and the vacuum is the model's own (:func:`_pad_cell`), so adding
#: either orbit
#: leaves the lattice, its vacuum included, where it was.
_DECLARED_POINT = (0.3141, 0.1592, 0.0919)
_DECLARED_SPECIES = "declared"

#: The generic points a descent, and a step 1 reduction, build their orbit
#: from, in order.  The first of each is the point above.  A generic point's
#: stabilizer is trivial, so the cell its orbit decorates carries exactly the
#: operations the orbit was generated from -- up to an accident of the
#: identification, a cell still read as non-primitive or with operations other
#: than the kept set, which the next point of the sequence does not share.  The
#: coordinates of all six are distinct from one another and from every Wyckoff
#: coordinate, so no orbit of one sequence meets an orbit of the other.
_DESCENT_POINTS = (_DESCENT_POINT, (0.3719, 0.0857, 0.2461), (0.2087, 0.4421, 0.1693))
_DECLARED_POINTS = (_DECLARED_POINT, (0.0671, 0.3853, 0.2297), (0.4462, 0.2239, 0.0383))


def _descent_frame(cell, group) -> tuple:
    r"""How the cell's own reduced coordinates map onto those of the group's 3D cell.

    :func:`_group_of` does not read a cell's positions directly: :func:`_pad_cell`
    first *scales* every non-periodic lattice vector until the vacuum reaches
    ``vacuum_length`` and then re-centres the positions along it on the origin, and
    the operations that come back are written in **that** frame.  A point added
    to the cell therefore has to be placed in the cell's frame while its orbit is
    computed in the group's, and this measures the one map between them:
    ``y = x / scale − offset`` (mod 1), with ``scale`` read off the two lattices
    and ``offset`` off the first orbital.  Both are exactly ``1`` and ``0`` on a
    periodic direction, and on a fully periodic cell the map is the identity.

    Returns ``(scale, offset, positions, vacuum_dirs)`` — the last two as
    :func:`_pad_cell` returns them, so the cell's own extent along a vacuum
    direction can be read off the same numbers.
    """
    lat_pad, pos_pad, padded = _pad_cell(cell, group.vacuum_length)
    lat_cell = np.asarray(cell.lat_vecs, dtype=float)
    scale = np.array([float(np.linalg.norm(lat_pad[d])
                            / max(np.linalg.norm(lat_cell[d]), 1e-30))
                      for d in range(3)])
    offset = np.asarray(cell.orb_vecs, dtype=float)[0] / scale - pos_pad[0]
    return scale, offset - np.rint(offset), pos_pad, tuple(padded)


def _generic_orbit(cell, group, operations, seed=_DESCENT_POINT) -> tuple:
    r"""The orbit of one generic point under ``operations``, in both frames.

    The point is ``seed`` along every periodic direction — :data:`_DESCENT_POINT`
    for a descent and :data:`_DECLARED_POINT` for step 1's cell reduction, two
    points that never coincide, since a reduced cell is the base a later descent
    adds its own orbit to — and sits off the cell's own mid-plane along a vacuum
    one (see :data:`_DESCENT_PAD_FRACTION`).  Its stabilizer in the cell's group
    is trivial, so the point set ``sites ∪ orbit`` is invariant under exactly the
    operations the orbit was generated from — plus, in principle, an accident,
    which is why the identification is **verified** rather than trusted, and
    an accident rebuilt from the next point of a fixed sequence
    (:func:`_orbit_cell`).

    Returns ``(point, orbit_padded, orbit_cell)``: the generic point and the
    orbit in the group's 3D cell, and the same orbit in the cell's own reduced
    coordinates, ready for :func:`_decorated_cell`.  Components along the
    vacuum are carried on the ``(−½, ½]`` branch, which is the branch the
    cell's own
    positions are written in and the one :func:`_pad_cell` measures its extent
    from.
    """
    scale, offset, pos_pad, padded = _descent_frame(cell, group)
    point = np.array([float(seed[d]) for d in range(3)])
    for d in padded:
        signed = pos_pad[:, d] - np.rint(pos_pad[:, d])
        lo, hi = float(signed.min()), float(signed.max())
        point[d] = 0.5 * (lo + hi) + (_DESCENT_PAD_FRACTION * 0.5 * (hi - lo)
                                      if hi - lo > 1e-9 else _DESCENT_PAD_OFFSET)

    orbit: list = []
    for sym in operations:
        y = (np.asarray(sym.rotation, dtype=float) @ point
             + np.asarray(sym.translation, dtype=float).reshape(-1)) % 1.0
        for d in padded:                       # the cell's own branch, not [0, 1)
            y[d] -= np.rint(y[d])
        if not any(np.abs((y - o + 0.5) % 1.0 - 0.5).max() <= 1e-9 for o in orbit):
            orbit.append(y)
    orbit_padded = np.array(orbit) if orbit else np.zeros((0, 3))
    return point, orbit_padded, (orbit_padded + offset) * scale


def _descent_closure(group, kept) -> tuple:
    r"""Products of the kept set that leave it — the descent's own closure gate.

    A set of operations can only *be* a group's operation set if it closes, and
    the kept set is a **measurement**: the detector's verdicts are per operation
    (per k-action class, strictly), and nothing forces the survivors to compose.
    :attr:`ValidatedGroup.closed` reports the same thing for the whole surviving
    set; this asks it of the **unitary** part alone, which is the part a descent
    is built on.  Returns the offending ``(a, b, ab)`` triples of ``.ind``, empty
    when the set closes.  The product is taken modulo lattice translations,
    through the group's own product table.
    """
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    keep = {int(s.ind) for s in kept}
    leaks = []
    for a in sorted(keep):
        for b in sorted(keep):
            c = int(group.symmetries[int(prod[at[a], at[b]])].ind)
            if c not in keep:
                leaks.append((a, b, c))
    return tuple(leaks)


def _subgroups_inside(group, kept, avoid=None) -> set:
    r"""Every subgroup of ``group`` whose operations all lie in ``kept``.

    Enumerated the only way that is both complete and cheap: a subgroup is the
    **join of its cyclic subgroups**, so the cyclic closures of the kept
    operations are computed first and then joined pairwise until nothing new
    appears, with any closure that leaves ``kept`` abandoned as soon as it does.
    The cell groups this runs on have at most 192 operations and the kept sets
    at most a few dozen, so the whole lattice is found in milliseconds.

    ``kept`` is the **unitary** kept set (the descent's own), and the products
    are taken modulo lattice translations through the group's own product table.
    ``avoid`` is a set of pairs of ``.ind``, each a pair no subgroup returned may
    hold both of -- two operations whose matrices on these states carry a sign
    no single-valued representation has -- and a closure is abandoned as soon as
    it holds one; every subgroup of a subgroup free of such pairs is free of
    them too, so the enumeration stays complete.  Returns a set of frozensets of
    ``.ind``; ``{E}`` is always among them.  The subgroups inside one set are
    found once per group and kept on it, since every reading that searches them
    asks the same group again.
    """
    keep = frozenset(int(i) for i in kept)
    memo = getattr(group, "_subgroups", None)
    key = (keep, frozenset(frozenset(int(i) for i in pair) for pair in (avoid or ())))
    if memo is not None and key in memo:
        return set(memo[key])
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    ind_of = [int(s.ind) for s in group.symmetries]
    apart: dict = {}
    for pair in (avoid or ()):
        a, b = (int(i) for i in pair)
        apart.setdefault(a, set()).add(b)
        apart.setdefault(b, set()).add(a)

    def close(seed):
        cur, frontier = set(seed), list(seed)
        if apart and any(apart.get(x, set()) & cur for x in cur):
            return None
        while frontier:
            a = frontier.pop()
            for b in list(cur):
                for x, y in ((a, b), (b, a)):
                    c = ind_of[int(prod[at[x], at[y]])]
                    if c not in cur:
                        if c not in keep or (apart and apart.get(c, set()) & cur):
                            return None
                        cur.add(c)
                        frontier.append(c)
        return frozenset(cur)

    subs = {c for c in (close({a}) for a in sorted(keep)) if c is not None}
    frontier = set(subs)
    while frontier:
        fresh = set()
        for s in frontier:
            for t in subs:
                j = close(s | t)
                if j is not None and j not in subs and j not in fresh:
                    fresh.add(j)
        subs |= fresh
        frontier = fresh
    if memo is not None:
        memo[key] = frozenset(subs)
    return subs


def _resolve_closure(group, kept):
    r"""The group inside a kept set that does not close — what it cannot carry, removed.

    The measurement's survivors need not compose: the detector's verdicts are
    per operation, and nothing forces the set they leave to be a group.  An
    operation sitting on the wrong side of the declared tolerance breaks
    closure while its partners keep it.  A **layer** cell's ``σ_h`` fibers,
    both members of each in the cell, do not: the tensor test holds each member
    to its own lift, on the multiplets where each is one orbital and, in a
    spinful model, on the whole solution space whatever the multiplets, and
    keeps one member of each fiber the Hamiltonian breaks -- on the SRSI
    paper's ``P41'`` cell read free-standing ``C4`` and ``C4⁻¹``, ``S4`` and
    ``S4⁻¹`` refuted by tensor structure -- and no layer fixture reaches this
    step.

    So: enumerate the subgroups inside the kept set (:func:`_subgroups_inside`)
    and return the largest.  Where the largest order is reached by **more than
    one** subgroup, the measurement does not choose between them, and the one
    returned is the first by the tie rule every such choice is made by
    (:func:`_tie_ranked`): on a layer the most operations that keep its side,
    then the most rotations, then the highest-order rotation, then its axis in
    the conventional cell.  Their intersection would keep less than any of them
    carries.

    Returns ``(resolved, dropped, tied)`` — the returned set of ``.ind``; per
    dropped operation ``(ind, why)`` with ``why`` ``"closure"`` for one no
    candidate subgroup carries — the kept set does not close with it — and
    ``"tie"`` for one some candidates carry and the one kept does not; and the
    candidates in the rule's order, the one kept first, or ``()`` where the
    largest is unique.
    """
    subs = _subgroups_inside(group, kept)
    top = max(len(s) for s in subs)
    ranked = _tie_ranked(group, [s for s in subs if len(s) == top])
    keep = sorted(int(i) for i in kept)
    in_some = set().union(*(set(c) for c in ranked))
    resolved = frozenset(ranked[0])
    dropped = tuple((i, "tie" if i in in_some else "closure")
                    for i in keep if i not in resolved)
    return resolved, dropped, (ranked if len(ranked) > 1 else ())


def _setting_axes(group) -> dict:
    r"""Each unitary operation's determinant, rotation order and axis, in the conventional cell.

    ``{ind: (det, order, axis)}``, read the way a Seitz symbol is written
    (:func:`_seitz_key`): the rotation carried into the tables' own conventional
    setting, the order that of its proper part ``det·R``, and the axis that
    part turns about -- a mirror's normal -- as the integer direction
    :func:`_primitive_direction` writes, empty for the identity and inversion.
    A group with no conventional setting is read in its own cell.
    """
    ref = getattr(group, "_refUC", None)
    out = {}
    for sym in group.symmetries:
        if bool(sym.time_reversal):
            continue
        rotation = (sym.rotation_refUC(np.asarray(ref, dtype=float)) if ref is not None
                    else sym.rotation)
        rotation = np.rint(np.asarray(rotation, dtype=float)).astype(int)
        det = int(round(float(np.linalg.det(rotation))))
        proper = det * rotation
        order = _ROTATION_ORDER[int(round(float(np.trace(proper))))]
        out[int(sym.ind)] = (det, order, () if order == 1 else _rotation_axis(proper))
    return out


def _direction_rank(axis) -> tuple:
    r"""Where a direction of the conventional cell stands in the tie rule.

    ``axis`` is ``[uvw]``, its first nonzero index positive.  Fewest nonzero
    indices first, then the smallest sum of their sizes, then the larger
    ``w``, ``u`` and ``v`` in that order: ``[001]``, ``[100]``, ``[010]`` --
    the ``c``, ``a`` and ``b`` axes -- then ``[101]``, ``[011]``, ``[110]`` and
    ``[1-10]``, and ``[111]`` first among the body diagonals.  No direction at
    all -- inversion's -- stands before every one.
    """
    if not axis:
        return ()
    u, v, w = (int(x) for x in axis)
    return (sum(1 for x in (u, v, w) if x), abs(u) + abs(v) + abs(w), (-w, -u, -v))


def _principal_axis(rows) -> tuple:
    r"""The axis the tie rule reads third, from a subgroup's ``(det, order, axis)`` rows.

    That of its highest-order rotation, or where it keeps no rotation but the
    identity, of its highest-order improper operation; the first in the order
    :func:`_direction_rank` fixes where several share that order, and ``()``
    where no operation has an axis.
    """
    proper = [r for r in rows if r[0] > 0]
    top = max(r[1] for r in proper)
    pool = [r for r in proper if r[1] == top and r[2]]
    if not pool:
        improper = [r for r in rows if r[0] < 0 and r[2]]
        if improper:
            highest = max(r[1] for r in improper)
            pool = [r for r in improper if r[1] == highest]
    return min((r[2] for r in pool), key=_direction_rank, default=())


def _layer_sides(group) -> dict:
    r"""Which unitary operations of a layer group keep the layer's side, ``{ind: bool}``.

    ``True`` for ``R[2,2] = +1`` along the one non-periodic direction
    (:func:`_side_preserving`), ``False`` for an operation that exchanges the
    two sides.  Empty where the group is no layer's -- a fully periodic crystal
    has no non-periodic direction -- and where none of its operations flips the
    side, a substrate's group among them: the tie rule's first key then has
    nothing to count.
    """
    padded = tuple(int(d) for d in getattr(group, "vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return {}
    sides = {int(s.ind): _side_preserving(s, padded[0]) is True
             for s in group.symmetries if not s.time_reversal}
    return {} if all(sides.values()) else sides


def _tie_key(subgroup, axes, sides=None) -> tuple:
    r"""The key the tie rule takes the smallest of.

    Among subgroups of one order, each read through ``axes``
    (:func:`_setting_axes`), and on a layer through ``sides``
    (:func:`_layer_sides`): first, on a layer, the most operations that keep
    the layer's side, since those hold in every environment and are what the
    layer keeps on a substrate; then the most rotations -- proper operations,
    the identity among them -- so that a subgroup keeping rotations stands
    before one keeping mirrors or rotoinversions; then the highest order among
    its rotations; then its principal axis (:func:`_principal_axis`) in the
    order :func:`_direction_rank` fixes; then every operation's order and axis
    the same way; and last the operations' indices, so that the rule is total.
    Where ``sides`` is empty the first entry is ``0`` for every subgroup, and the
    rule is the rest of it.
    """
    rows = [axes[int(i)] for i in subgroup]
    proper = [r for r in rows if r[0] > 0]
    every = sorted((-r[1], r[0] < 0, _direction_rank(r[2])) for r in rows)
    kept_side = sum(1 for i in subgroup if (sides or {}).get(int(i)))
    return (-kept_side, -len(proper), -max(r[1] for r in proper),
            _direction_rank(_principal_axis(rows)), every,
            tuple(sorted(int(i) for i in subgroup)))


def _tie_ranked(group, subgroups) -> tuple:
    r"""Subgroups of one order the measurement does not choose between, as the tie rule ranks them.

    Wherever the group a Hamiltonian obeys is the largest subgroup of some set
    and several subgroups of that order qualify -- the survivors that do not
    close (:func:`_resolve_closure`), the readings a declaration leaves
    (:func:`..symmetry._covariant_subgroup`), the subgroups on which spinless
    states carry an ordinary representation (:func:`_projective_class_refuted`)
    -- the one kept is the first here, never their intersection, which would
    keep less than any of them carries.  The rule, read in the conventional
    cell of ``group`` (:func:`_tie_key`): on a layer whose group has operations
    that flip its side, first the most operations that keep the side
    (``R[2,2] = +1``), since those hold in every environment and are what the
    layer keeps on a substrate; then the most rotations, so that a subgroup
    keeping rotations stands before one keeping mirrors or rotoinversions; then
    the highest-order rotation; then its axis, ``c`` before ``a`` before ``b``
    before the diagonals; then every other operation's axis the same way.

    Returns the subgroups, each as its ``.ind`` in order, the one kept first.
    """
    axes = _setting_axes(group)
    sides = _layer_sides(group)
    return tuple(sorted((tuple(sorted(int(i) for i in s)) for s in subgroups),
                        key=lambda s: _tie_key(s, axes, sides)))


def _tie_rule_words(group) -> str:
    r"""The tie rule as the record states it: its first keys, in words.

    The layer's own key is said only where it can decide something -- on a
    layer whose group has operations that flip its side (:func:`_layer_sides`).
    """
    return (("on a layer the most operations that keep its side, then " if
             _layer_sides(group) else "")
            + "the most rotations, then the highest-order rotation, then its axis in "
            "the conventional cell")


def _tie_reason(group, ranked) -> str:
    r"""Why the tie rule keeps the first of ``ranked`` rather than the second, in words.

    The first key of :func:`_tie_key` on which the two differ, said as what the
    one kept has: more operations that keep a layer's side, more rotations, a
    rotation of higher order, or a principal axis that comes first in the
    conventional cell.
    """
    axes = _setting_axes(group)
    sides = _layer_sides(group)
    kept, other = ([axes[int(i)] for i in s] for s in ranked[:2])
    mine, theirs = (_tie_key(s, axes, sides) for s in ranked[:2])
    if mine[0] != theirs[0]:
        return (f"it holds {-mine[0]} operations that keep the layer's side "
                f"(R[2,2] = +1) where the next holds {-theirs[0]}, each counted once per "
                "coset of the lattice translations")
    if mine[1] != theirs[1]:
        return (f"it holds {-mine[1]} rotations (proper operations) where the next "
                f"holds {-theirs[1]}, each counted once per coset of the lattice "
                "translations")
    if mine[2] != theirs[2]:
        return (f"its highest rotation is {-mine[2]}-fold where the next one's is "
                f"{-theirs[2]}-fold")
    if mine[3] != theirs[3]:
        here, there = _principal_axis(kept), _principal_axis(other)
        if not here:
            # only inversion is left to read, and no direction at all stands
            # before every one
            return (f"it keeps inversion where the next keeps an axis, "
                    f"[{_direction_text(there)}], and no direction stands before any")
        return (f"its principal axis, [{_direction_text(here)}], comes before "
                f"[{_direction_text(there)}] in the conventional cell")
    if mine[4] != theirs[4]:
        return ("its other operations' orders and axes come first in the conventional "
                "cell")
    return "its operations come first in the group's own order"


def _mirror_partners(group, validated, assignment) -> dict | None:
    r"""On a flat layer, each operation composed with the horizontal mirror, and its partner.

    Where every orbital of a layer lies in one plane, the horizontal mirror
    ``σ_h`` moves no orbital and no in-plane ``k``, so an operation ``g`` and
    its product ``σ_h·g`` act alike on everything a Hamiltonian without spin
    has: they share an action on ``k`` (:attr:`ValidatedGroup.k_action_classes`)
    and a site permutation, one solution space holds the matrices of both, and
    the hoppings cannot tell them apart.  What does is the sign each orbital
    carries under ``σ_h`` (:func:`_mirror_sign`).

    Returns ``{ind of σ_h·g: ind of g}`` -- ``g`` keeping the layer's side and
    ``σ_h·g`` turning it over, either of them combined with time reversal where
    ``g`` is -- for every class of operations the detector left undetermined,
    the identity and the mirror itself among them; or ``None`` where one of
    those classes has another shape, two members that move the orbitals
    differently among them, which the orbitals' positions tell apart.  Empty
    where nothing is undetermined.
    """
    undetermined = {int(i) for i in validated.undetermined}
    if not undetermined:
        return {}
    padded = tuple(int(d) for d in getattr(group, "vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return None
    sides = {int(s.ind): _side_preserving(s, padded[0]) for s in group.symmetries}
    perm = np.asarray(assignment.permutations, dtype=int)
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    partners = {}
    for members in validated.k_action_classes:
        pair = [int(i) for i in members if int(i) in undetermined]
        if not pair:
            continue
        if (len(pair) != 2 or len(members) != 2
                or any(sides.get(i) is None for i in pair) or sides[pair[0]] == sides[pair[1]]):
            return None
        keep, flip = pair if sides[pair[0]] else pair[::-1]
        if not np.array_equal(perm[at[keep]], perm[at[flip]]):
            return None
        partners[flip] = keep
    return partners


def _mirror_sign(group, partners, decls=()) -> dict:
    r"""The sign the horizontal mirror carries on a flat layer read without spin, chosen.

    The hoppings fix how a flat layer's orbitals transform under ``σ_h``
    relative to one another and not the sign itself: ``σ_h`` moves nothing, so
    its matrix is one commuting with ``H(k)`` at every ``k``, and on each
    part of the crystal the hoppings couple that is a multiple of the identity
    -- one sign per coupled part, ``D(σ_h)`` and ``−D(σ_h)`` alike.  The sign
    is therefore chosen: the parity the declared content states where it
    states one (``s`` even, ``p_z`` odd), and otherwise the even one the
    lowest-angular-momentum rule assigns, the lowest function of any in-plane
    content lying in the plane.  Fixing it fixes every operation composed with
    the mirror from its partner, ``D(σ_h·g) = D(σ_h) D(g)``, the inversion
    ``i = σ_h·C2(z)`` among them, and it is never refused: a coupled part
    declared two parities is a crystal without the mirror, which the
    declaration's own verdict lowers before this is read.  The states'
    projective class is read on one coupled part, so one sign is returned.

    Returns ``{"sign", "determined_by"}``: ``+1`` or ``-1``, and
    ``"declaration"`` or ``"lowest-l"``.
    """
    from .symmetry import _identity_position

    identity = int(group.symmetries[_identity_position(group)].ind)
    mirror = next((f for f, g in partners.items() if g == identity), None)
    if mirror is not None:
        for content in _declared_parts_of(decls or ()):
            chi = (getattr(content, "constraints", None) or {}).get(mirror)
            if chi is None:
                continue
            parity = complex(chi).real / max(1, int(content.dim))
            if abs(abs(parity) - 1.0) <= 1e-6:
                return {"sign": 1 if parity > 0 else -1, "determined_by": "declaration"}
    return {"sign": 1, "determined_by": "lowest-l"}


def _class_members(group, validated, partners=None, sign=1):
    r"""One matrix per operation of a spinless reading, where the class can be read off them.

    The class is read off the detector's own members, and only where a member
    is fixed up to a phase: a unitary, single-valued group whose every
    operation stands, with nothing but the identity commuting with ``H``, each
    operation alone in its action on ``k`` -- or, on a flat layer, sharing it
    with its product with the horizontal mirror (``partners``,
    :func:`_mirror_partners`), which is given the mirror's chosen ``sign``
    times its partner's matrix, ``D(σ_h·g) = D(σ_h) D(g)``.

    Returns ``(D, positions)`` over every operation of the group, or ``None``
    where the class is not read here.
    """
    from .symmetry import _identity_position

    syms = group.symmetries
    if bool(group.spinor) or any(bool(s.time_reversal) for s in syms):
        return None
    inds = [int(s.ind) for s in syms]
    if set(inds) - {int(i) for i in validated.indices}:
        return None
    partners = dict(partners or {})
    if {int(i) for i in validated.undetermined} - set(partners) - set(partners.values()):
        return None
    ident = _identity_position(group)
    if int((validated._nullspace_dims or {}).get(int(syms[ident].ind), 0) or 0) != 1:
        return None
    members = validated._intertwiners or {}
    D = {}
    for p, ind in enumerate(inds):
        member = members.get(partners.get(ind, ind))
        if member is None:
            return None
        D[p] = (int(sign) if ind in partners else 1) * np.asarray(member, dtype=complex)
    return D, list(range(len(syms)))


def _class_kept(group, D, positions):
    r"""The largest subgroup on which matrices ``D`` are an ordinary representation, by the tie rule.

    Two witnesses, neither of which a choice of phases moves: the ratio of two
    commuting operations' matrices taken in the two orders
    (:func:`..symmetry._commuting_pair_phases`), which names the pairs, and the
    relations a rephasing has to close on a subgroup
    (:func:`..symmetry._rephasable`), which decides it.  The subgroups holding
    no such pair are enumerated (:func:`_subgroups_inside`), the largest order
    at which one of them is rephasable is the order kept, and where several of
    that order are, the one kept is the first by the tie rule
    (:func:`_tie_ranked`), never their intersection.

    Returns ``(pairs, kept, ranked)`` -- the pairs ``(a, b, phase)`` by
    position, the ``.ind`` kept and the subgroups of the kept order in the
    rule's order -- or ``None`` where no pair of commuting operations carries
    a phase the table lacks.
    """
    from .symmetry import _commuting_pair_phases, _identity_position, _rephasable

    syms = group.symmetries
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
    pairs = _commuting_pair_phases(D, positions, syms, prod, spf, tol=_CLASS_MARGIN)
    if not pairs:
        return None
    at = {int(s.ind): p for p, s in enumerate(syms)}
    subs = _subgroups_inside(group, [int(syms[p].ind) for p in positions], avoid={
        frozenset((int(syms[a].ind), int(syms[b].ind))) for a, b, _c in pairs})
    kept, ranked = frozenset({int(syms[_identity_position(group)].ind)}), ()
    for size in sorted({len(s) for s in subs}, reverse=True):
        good = [s for s in subs if len(s) == size
                and _rephasable(D, [at[i] for i in s], group, prod, spf, tol=_CLASS_MARGIN)]
        if good:
            ranked = _tie_ranked(group, good)
            kept = frozenset(ranked[0])
            break
    return pairs, kept, ranked


def _class_record(group, pairs, kept, ranked, *, rule_group=None, rule_ranked=None):
    r"""The operations a class reading leaves out, each with the pair that names it, and the step's record.

    ``pairs`` are :func:`_class_kept`'s, by position in ``group``; ``kept`` the
    ``.ind`` kept and ``ranked`` the tied subgroups, the one kept first.
    ``rule_group`` is the group the tie was settled in, where that is not
    ``group``, and ``rule_ranked`` the tied subgroups there, so that the rule's
    words, and the reason its first candidate was kept, are the ones it was
    read with.
    """
    from .symmetry import _fmt_char, _proportionality

    syms = group.symmetries
    _prod, _tdiff, spf = group.get_product_table(get_diff=True)
    at = {int(s.ind): p for p, s in enumerate(syms)}
    lattice = np.asarray(group.lattice, dtype=float)

    def _name(i):
        return _record_name(syms[at[int(i)]], lattice)

    witnesses = []
    for a, b, c in pairs:
        Sa = np.asarray(syms[a].spinor_rotation, dtype=complex)
        Sb = np.asarray(syms[b].spinor_rotation, dtype=complex)
        half, _res = _proportionality(Sa @ Sb, int(spf[a, b]) * int(spf[b, a]) * (Sb @ Sa))
        witnesses.append({"pair": (int(syms[a].ind), int(syms[b].ind)),
                          "names": (_name(syms[a].ind), _name(syms[b].ind)),
                          "commutation_phase": complex(c),
                          "spin_half": bool(abs(complex(half) - complex(c)) <= _CLASS_MARGIN)})

    refuted = {}
    for i in sorted(set(at) - set(kept)):
        mine = [w for w in witnesses if i in w["pair"]]
        best = next((w for w in mine if sum(1 for j in w["pair"] if j in kept) == 1),
                    mine[0] if mine else None)
        if best is None:
            refuted[i] = {"refuted_by": "projective class", "pair": None, "names": None,
                          "commutation_phase": None, "spin_half": None, "note": (
                              f"no subgroup holding {_name(i)} together with the operations "
                              "kept carries a single-valued representation on these states")}
            continue
        a, b = best["names"]
        refuted[i] = dict(best, refuted_by="projective class", note=(
            f"{a} and {b} commute, and on these states their matrices do not: "
            f"D({a}) D({b}) = {_fmt_char(best['commutation_phase'])} "
            f"D({b}) D({a}), a phase no rescaling of either matrix removes"
            + (" -- the sign the spin-1/2 lifts of the two operations carry"
               if best["spin_half"] else "")
            + f". No single-valued representation of a group holding both exists on "
            f"these states, so {_name(i)} is not an operation of the spinless crystal: "
            "the largest subgroup on which the states carry one is kept, and it does not "
            f"hold {_name(i)}."))
    record = {"pairs": tuple(witnesses), "kept": tuple(sorted(kept)),
              "tied": tuple(ranked) if len(ranked) > 1 else ()}
    if len(ranked) > 1:
        # the rule's keys as the group they were read in reads them, and why it
        # kept the first, for the descent's note
        settled = rule_group if rule_group is not None else group
        record["rule"] = _tie_rule_words(settled)
        record["reason"] = _tie_reason(settled,
                                       rule_ranked if rule_ranked is not None else ranked)
    return refuted, record


def _projective_class_refuted(group, validated, *, assignment=None, decls=()):
    r"""The operations of a spinless reading that no single-valued representation carries.

    A spinless reading reads the states in single-valued tables, and the matrices
    the Hamiltonian's operations are on them form a projective representation.
    Where its class is not trivial -- spin-½ states in the orbital index of a
    spinless model are the case, where two perpendicular twofolds commute and
    their matrices anticommute -- no single-valued representation of the group
    exists on the states, and one of every such pair is not an operation of the
    spinless crystal.  So the spinless group is the **largest subgroup on which
    the states carry an ordinary representation**, and what is refuted here is
    the rest.

    The class is read off the detector's own members, and only where a member is
    fixed up to a phase (:func:`_class_members`): a unitary, single-valued group
    whose every operation stands, with nothing but the identity commuting with
    ``H``, each operation alone in its action on ``k`` -- or, on a flat layer,
    sharing it with its product with the horizontal mirror, whose matrix is
    then the mirror's chosen sign times its partner's (:func:`_mirror_partners`,
    :func:`_mirror_sign`): ``D(σ_h·g) = D(σ_h) D(g)``, so that every operation
    of the free-standing layer has its matrix and the class is read there as
    anywhere else.  The sign commutes with every matrix, and the class does not
    depend on it.  Two witnesses, neither of which a choice of phases moves,
    decide it (:func:`_class_kept`).

    Where several subgroups of the kept order carry one, one is chosen by the
    tie rule every such choice is made by (:func:`_tie_ranked`), never their
    intersection: on a layer the most operations that keep its side first;
    then the most rotations, so that a subgroup keeping rotations stands
    before one keeping mirrors; then the highest-order rotation; then its axis
    in the conventional cell -- ``c``, then ``a``, then ``b``, then the
    diagonals -- and last every other operation's axis the same way.

    ``assignment`` is the orbitals' orbits in ``group``, which say whether an
    operation and its product with the mirror move the orbitals alike, and
    ``decls`` the declared contents the round read, which say the mirror's
    parity where they state one.

    Returns ``(refuted, record)``: per operation left out, keyed by ``.ind``,
    ``refuted_by = "projective class"`` with the ``pair`` of operations that
    names it -- a commuting partner whose matrix does not commute with its own,
    one that is kept where there is one -- their ``names``, the
    ``commutation_phase``, whether that phase is the sign the two operations'
    spin-½ lifts carry (``spin_half``), and a ``note``; and the step's own
    record, every such pair and the subgroups of the kept order in the rule's
    order, the first of them kept, with the rule's keys as this group reads
    them (``rule``) and why it kept the first (``reason``) where several tie,
    and on a flat layer the mirror's sign and how it was determined
    (``horizontal_mirror``).  ``({}, None)`` where the class is the table's, or
    is not read here.
    """
    if bool(group.spinor) or any(bool(s.time_reversal) for s in group.symmetries):
        return {}, None
    partners = {}
    if validated.undetermined:
        partners = (None if assignment is None
                    else _mirror_partners(group, validated, assignment))
        if partners is None:
            return {}, None
    mirror = _mirror_sign(group, partners, decls) if partners else None
    members = _class_members(group, validated, partners,
                             sign=1 if mirror is None else mirror["sign"])
    if members is None:
        return {}, None
    got = _class_kept(group, *members)
    if got is None:
        return {}, None
    refuted, record = _class_record(group, *got)
    if mirror is not None:
        record["horizontal_mirror"] = dict(mirror)
    return refuted, record


def _free_standing_lift(model, group, free_standing):
    r"""A substrate group with the horizontal mirror adjoined: the free-standing layer's group.

    ``free_standing`` is the free-standing cell of the same point set
    (:func:`_layer_realized`), whose group is identified the first time it is
    asked for and kept on the record.  On a flat layer that group holds the
    mirror ``σ_h``, which moves no orbital, and every operation of ``group``
    -- each keeps the layer's side -- matched by rotation and by translation
    modulo the lattice (:func:`_match_operations`); the free-standing group of
    ``group`` is ``group``
    with ``σ_h`` adjoined, the whole cell group where ``group`` is the whole
    substrate cell's and otherwise a cell carrying exactly those operations,
    identified as a descent's is (:func:`_orbit_cell`), so that the tie rule
    reads it in its own conventional cell, as the free-standing layer's own
    measurement does.

    Returns ``(free, of)`` -- the free-standing group, and per unitary
    operation of it ``(ind of g in group, whether it is σ_h·g)`` -- or
    ``None`` where the layer is not flat: the mirror is not an operation of
    that cell, or it moves an orbital.

    Raises
    ------
    ..refusal.InternalError
        Where an operation of ``group`` is not one of the free-standing cell's:
        a substrate only removes operations, so this is the package
        disagreeing with itself.
    """
    parent = free_standing.get("group")
    if parent is None:
        parent = _group_of(free_standing["cell"], typat=free_standing["cell_typat"],
                           spinor=False, include_TR=False, symprec=free_standing["symprec"])
        free_standing["group"] = parent
    if isinstance(parent, Refusal):
        return None
    padded = tuple(int(d) for d in getattr(parent, "vacuum_dirs", ()) or ())
    if len(padded) != 1:
        return None
    assignment = _assign_orbitals(model, parent, typat=free_standing["labels"])
    if isinstance(assignment, Refusal):
        return None
    reflect = np.eye(3, dtype=int)
    reflect[padded[0], padded[0]] = -1
    psyms = parent.symmetries
    perm = np.asarray(assignment.permutations, dtype=int)
    still = np.arange(perm.shape[1])
    mirror = next((p for p, s in enumerate(psyms) if not s.time_reversal
                   and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int), reflect)
                   and np.array_equal(perm[p], still)), None)
    if mirror is None:
        return None
    pairs, missing, _extra = _match_operations(
        [s for s in group.symmetries if not s.time_reversal],
        [s for s in psyms if not s.time_reversal], free_standing["symprec"],
        group.lattice)
    if missing:
        by_ind = {int(s.ind): s for s in group.symmetries}
        lattice = np.asarray(group.lattice, dtype=float)
        raise InternalError(
            f"{len(missing)} operations of {group.name} #{group.number_str}, read on a "
            "substrate, are not operations of the free-standing cell of the same point "
            f"set ({parent.name} #{parent.number_str}): "
            + ", ".join(_record_name(by_ind[int(i)], lattice) for i in missing)
            + ". A substrate only removes operations, so this is the package disagreeing "
            "with itself, not a measurement about the model.")
    at = {int(s.ind): p for p, s in enumerate(psyms)}
    prod = parent.get_product_table(get_diff=False)
    of_parent = {}
    for here, there in pairs:
        of_parent[at[int(there)]] = (int(here), False)
        of_parent[int(prod[mirror, at[int(there)]])] = (int(here), True)
    unitary = {p for p, s in enumerate(psyms) if not s.time_reversal}
    if set(of_parent) == unitary:
        return parent, {int(psyms[p].ind): v for p, v in of_parent.items()}
    kept = [psyms[p] for p in sorted(of_parent)]
    species = _DESCENT_SPECIES
    while any(str(t) == species for t in free_standing["cell_typat"]):
        species += "_"

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What the identification of the free-standing cell was read as, and why that is a defect."""
        return (
            f"the free-standing cell built to carry {group.name} #{group.number_str} with the "
            f"horizontal mirror adjoined ({len(kept)} operations of {parent.name} "
            f"#{parent.number_str}) was identified as {got.name} #{got.number_str}, whose "
            f"{len(unitary)} unitary operations are not that set ({len(unmatched_kept)} "
            f"missing, {len(unmatched_found)} extra), so this is the package's "
            "identification disagreeing with its own construction.")

    made = _orbit_cell(free_standing["cell"], free_standing["cell_typat"], parent, kept,
                       points=_DESCENT_POINTS, species=species, spinor=False,
                       include_TR=False, symprec=free_standing["symprec"],
                       diagnose=_diagnose)
    if isinstance(made, Refusal):
        return None
    free, _cell, _typat, pairs, _point, _orbit = made
    of_ind = {int(psyms[p].ind): v for p, v in of_parent.items()}
    return free, {int(f): of_ind[int(k)] for k, f in pairs}


def _projective_class_carried(model, group, validated, *, free_standing):
    r"""A flat layer's projective class on a substrate: the free-standing layer's reading, carried.

    On a flat layer read without spin the free-standing group is this one with
    the horizontal mirror ``σ_h`` adjoined, and its matrices are these times the
    mirror's sign, every operation composed with the mirror given
    ``D(σ_h) D(g)`` (:func:`_mirror_sign`): nothing a substrate measures is
    missing from it, and nothing is measured again.  So the class is read there
    once (:func:`_free_standing_lift`, :func:`_class_kept`) -- the largest
    subgroup on which the states carry an ordinary representation, by the one
    tie rule in the free-standing layer's own conventional cell -- and the
    substrate carries its part that preserves the layer's normal, as it carries a declaration read
    in the free-standing layer's site groups (:func:`_carried_reduction`):
    every operation the class leaves out of the free-standing layer is left out
    here too.  The two environments then read one class, and the group kept on
    a substrate is the half of the one kept free-standing that preserves the
    layer's normal.  This
    group's own matrices decide only whether there is a class to read -- the
    free-standing pairs are its pairs and their products with the mirror -- and
    name the pairs on the record.  The sign is the lowest-angular-momentum
    rule's: a substrate reads no declared content under a mirror it does not
    have, and the class does not depend on the sign.

    Where the layer is not flat -- the mirror moves an orbital, or is not an
    operation of the free-standing cell -- the orbitals' positions tell an
    operation from its product with the mirror, the free-standing group is not
    this one with the mirror adjoined, and the class is read here
    (:func:`_projective_class_refuted`).

    Returns ``(refuted, record)`` as :func:`_projective_class_refuted` does, the
    pairs those of this group and the record naming the free-standing group the
    class was read on (``read_on``) and how many of its operations were kept
    there.
    """
    from .symmetry import _commuting_pair_phases

    members = _class_members(group, validated)
    if members is None:
        return {}, None
    D, positions = members
    prod, _tdiff, spf = group.get_product_table(get_diff=True)
    pairs = _commuting_pair_phases(D, positions, group.symmetries, prod, spf,
                                   tol=_CLASS_MARGIN)
    if not pairs:
        # the free-standing pairs are these and their products with the mirror,
        # which commutes with every matrix: no pair here, none there
        return {}, None
    lift = _free_standing_lift(model, group, free_standing)
    if lift is None:
        return _projective_class_refuted(group, validated)
    free, of = lift
    # each operation composed with the mirror is given the mirror's sign times
    # its partner's matrix, the sign the even one of the lowest-angular-momentum
    # rule: D(sigma_h g) = D(sigma_h) D(g)
    sign = 1
    at = {int(s.ind): p for p, s in enumerate(group.symmetries)}
    fD = {p: (sign if of[int(s.ind)][1] else 1) * D[at[of[int(s.ind)][0]]]
          for p, s in enumerate(free.symmetries) if not s.time_reversal}
    got = _class_kept(free, fD, sorted(fD))
    if got is None:                     # pragma: no cover - the pairs above are its pairs too
        raise InternalError(
            f"the projective class of the states measured in {group.name} "
            f"#{group.number_str} has {len(pairs)} pairs of commuting operations whose "
            "matrices do not commute, and the free-standing group with the horizontal mirror "
            f"adjoined, {free.name} #{free.number_str}, has none: the mirror's matrix "
            "commutes with every other, so this is the package disagreeing with itself.")
    _fpairs, fkept, franked = got

    def _here(subgroup):
        """A subgroup of the free-standing group, by its operations here that preserve the layer's normal."""
        return tuple(sorted(of[int(i)][0] for i in subgroup if not of[int(i)][1]))

    ranked = tuple(_here(c) for c in franked)
    refuted, record = _class_record(group, pairs, frozenset(_here(fkept)), ranked,
                                    rule_group=free, rule_ranked=franked)
    record["read_on"] = {"number_str": str(free.number_str), "name": str(free.name),
                         "size": int(free.size), "kept": len(fkept)}
    return refuted, record


def _match_operations(kept, found, symprec, lattice) -> tuple:
    r"""Pair each kept operation with the identified one that *is* it.

    Two operations are the same when their rotations are the same integer matrix
    and their translations agree **modulo the lattice** -- the sense in which a
    space-group translation is defined -- to the finder's distance ``symprec``,
    their difference read as the length it is on ``lattice`` (rows the cell's
    vectors), in the units the finder's distance is in.  The
    pairing is one-to-one by construction — a matched operation is removed from
    the pool — so an identification that is still too large shows up as an
    unmatched *found* operation and one that has moved as an unmatched *kept*
    one.  Returns ``(pairs, kept_unmatched, found_unmatched)``.
    """
    lat = np.asarray(lattice, dtype=float)
    pool = list(found)
    pairs, missing = [], []
    for sym in kept:
        rot = np.rint(np.asarray(sym.rotation, dtype=float)).astype(int)
        tra = np.asarray(sym.translation, dtype=float).reshape(-1)
        for j, other in enumerate(pool):
            if not np.array_equal(
                    rot, np.rint(np.asarray(other.rotation, dtype=float)).astype(int)):
                continue
            d = tra - np.asarray(other.translation, dtype=float).reshape(-1)
            if float(np.linalg.norm((d - np.rint(d)) @ lat)) <= float(symprec):
                pairs.append((int(sym.ind), int(other.ind)))
                pool.pop(j)
                break
        else:
            missing.append(int(sym.ind))
    return (tuple(pairs), tuple(missing), tuple(int(o.ind) for o in pool))


def _shared_origin(found, kept, pairs) -> GroupData:
    r"""The identified group, written in the origin of the operations it was built to carry.

    An identification names a group, puts it in the tables' setting and orders
    its operations, and it reads each translation off the cell it is handed.
    The operations kept are the ones every earlier step was measured against,
    written in the origin the centres satisfy.  Where the cell an
    identification reads was built on exact centres -- every cell under a
    declared precision is, the centres made exact before any is identified
    (:func:`_exact_centres`, :func:`_position_pass`) -- the two agree to
    rounding and nothing moves.  Where they differ, it is by one shift of the
    origin, ``t' = t − (1 − A)·δ``: the group the cell allows, named on the
    centres projected onto its own positions, carried onto the ones the
    centres were moved to where it does not carry them, or a cell read with
    centres no declaration made exact.  The identified group is carried onto
    the kept operations by that one shift (:func:`_moved_origin`): every
    translation moved by ``(1 − A)·δ`` and the origin of the tables' setting
    (``shiftUC``) by ``δ``, which leaves every operation's tabulated
    translation where it was.

    ``δ`` is read off the pairs.  Two copies of one group, each closing
    exactly, differ by translations ``d_g = t_kept(g) − t_found(g)`` that
    compose as the group does, ``d_{gh} = d_g + A_g·d_h``, and their average
    over the group is the ``δ`` with ``d_g = (1 − A_g)·δ`` for every ``g``.

    ``pairs`` are :func:`_match_operations`' ``(kept, identified)`` pairs of
    ``.ind``, one for every unitary operation of ``found``.  Where every pair's
    translations agree to rounding -- every cell built from exact numbers --
    ``found`` is returned as it is, bit for bit; otherwise a new record is
    returned and the identification's own is left as it was made.

    Raises
    ------
    ..refusal.InternalError
        Where no single shift carries the identified operations onto the kept
        ones.  Two exact copies of one group are always related by one, so
        this is arithmetic in this package failing rather than a measurement.
    """
    by_kept = {int(s.ind): s for s in kept}
    by_found = {int(s.ind): s for s in found.symmetries}
    diffs = []
    for k, f in pairs:
        d = (np.asarray(by_kept[int(k)].translation, dtype=float).reshape(-1)
             - np.asarray(by_found[int(f)].translation, dtype=float).reshape(-1))
        diffs.append((np.asarray(by_found[int(f)].rotation, dtype=float), d - np.rint(d)))
    if not diffs or max(float(np.abs(d).max()) for _r, d in diffs) <= _EXACT_CLOSURE:
        return found
    delta = np.mean([d for _r, d in diffs], axis=0)
    miss = max(float(np.abs((np.eye(3) - r) @ delta - d).max()) for r, d in diffs)
    if miss > _EXACT_CLOSURE:
        raise InternalError(
            f"group #{found.number_str}: the operations identified on the cell and the "
            f"{len(diffs)} it was built to carry are not one group about two origins -- "
            f"the best single shift misses by {miss:.3g}. Both close exactly, so this "
            "is arithmetic in this package failing, not a measurement about the model.")
    return _moved_origin(found, delta)


def _moved_origin(group, delta) -> GroupData:
    r"""``group`` written about an origin moved by ``delta``, reduced, in its 3D cell.

    Every translation moves by ``(1 − A)·δ`` and the origin of the tables'
    setting (``shiftUC``) by ``δ``, which leaves every operation's tabulated
    translation where it was: the same group, about another point.  A new
    record is returned and ``group`` is left as it was.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    moved = []
    for s in group.symmetries:
        new = copy.copy(s)
        for cached in ("translation_cart", "is_identity"):
            new.__dict__.pop(cached, None)
        new.translation = (np.asarray(s.translation, dtype=float)
                           + (np.eye(3) - np.asarray(s.rotation, dtype=float)) @ delta)
        moved.append(new)
    shift = np.asarray(group._shiftUC, dtype=float) + delta
    sg = copy.copy(group._sg)
    sg.__dict__.pop("translations_cart", None)
    sg.symmetries, sg.shiftUC = moved, shift
    return replace(group, symmetries=list(moved), _shiftUC=shift, _sg=sg)


def _orbit_cell(base_cell, base_typat, group, kept, *, points, species, spinor,
                include_TR, symprec, diagnose):
    r"""The cell that carries exactly ``kept``, identified — from the first point that does.

    The parent cell's own points plus the orbit of one generic point under
    ``kept``, as one extra species (:func:`_generic_orbit`), handed to
    :func:`_group_of`, the one identification that can name the subgroup and
    stamp the ``.ind`` order every table lookup keys on, and then verified
    against ``kept`` operation by operation (:func:`_match_operations`).  A
    generic point's orbit under a group has that group for its stabilizer, so a
    cell read as still non-primitive, or with operations other than ``kept``,
    is an accident of that point and not a statement about the operations: the
    orbit is rebuilt from the next point of ``points``.

    Returns ``(group, cell, cell_typat, pairs, point, orbit)`` — ``orbit`` in the
    identified group's 3D cell — or the :class:`..refusal.Refusal` the
    identification made.  The group carries the operations ``kept``
    themselves, in the origin they were measured in (:func:`_shared_origin`),
    and not the translations the identification reads off the parent cell's
    own points: those are the positions as handed in, and where a declared
    precision has moved the centres it is ``kept`` that they satisfy -- the
    origin they were written in is the one fitted to the moved centres, also
    where the descent is the centres' own, the operations they are not carried
    by within the precision left out.

    Raises
    ------
    ..refusal.InternalError
        When every point of ``points`` is read that way.  ``diagnose`` writes
        what the last one was read as, from ``(group, unitary operations,
        kept unmatched, found unmatched, orbit)``.
    """
    last = None
    for seed in points:
        point, orbit_padded, orbit_cell = _generic_orbit(base_cell, group, kept, seed=seed)
        cell, cell_typat = _decorated_cell(
            base_cell, base_typat, environment="free-standing",
            decorations=[([float(x) for x in row], species) for row in orbit_cell])
        got = _group_of(cell, typat=cell_typat, spinor=bool(spinor),
                        include_TR=bool(include_TR), symprec=symprec)
        if isinstance(got, Refusal):
            return got
        unitary = [s for s in got.symmetries if not s.time_reversal]
        pairs, unmatched_kept, unmatched_found = _match_operations(kept, unitary, symprec,
                                                                   group.lattice)
        if int(got._lattice_index) <= 1 and not unmatched_kept and not unmatched_found:
            return (_shared_origin(got, kept, pairs), cell, cell_typat, pairs, point,
                    orbit_padded)
        last = (got, unitary, unmatched_kept, unmatched_found, orbit_padded)
    raise InternalError(
        diagnose(*last) + f" That is the last of {len(points)} generic points tried, "
        "and the orbit of none of them was identified as exactly the kept operations.")


def _centres_not_exactly_symmetric(model, group, assignment, *, tol,
                                   params) -> Refusal | None:
    r"""Centres whose defect alone breaks operations — refused before anything is built.

    In convention I the Bloch phase carries ``e^{2πi k·(R + τ_j − τ_i)}``, so
    centres that miss the group's own action by ``δ`` cost ``H(k)`` a covariance
    residual of order ``2π·δ·max|H|`` whatever the hoppings do.  Where nothing
    declares what noise means for the centres (no ``position_tol``, no
    ``precision``) and that estimate is above the detector's tolerance, the
    defect alone would break operations: which ones the Hamiltonian obeys would
    be the geometry's verdict as much as the hoppings', and everything built on
    it would be built again once the precision is declared.  So it is measured
    as soon as the cell's group is identified, and refused there, loudly, with
    the number to act on.

    Returns the ``positions_not_exactly_symmetric`` refusal, or ``None`` when the
    estimate is within ``tol`` and the detector's verdicts are the hoppings' own.
    """
    from .symmetrization import _position_defect

    delta = float(_position_defect(model, group, assignment,
                                   {int(s.ind): s for s in group.symmetries}))
    if not delta > 0.0:
        return None
    max_h = _max_hamiltonian(model, params=params)
    estimate = 2.0 * np.pi * delta * max_h
    if estimate <= float(tol):
        return None
    return Refusal(
        "positions_not_exactly_symmetric",
        f"the orbital centres miss the cell's own symmetry by {delta:.3g} (reduced): "
        "declare this model's precision -- `realized_group(..., precision=)`, or "
        "`position_tol=` for the centres alone -- and centres within it are moved onto "
        "the positions the group puts them at, a k-dependent gauge transformation "
        "that leaves the spectrum untouched. Undeclared, that defect alone costs H(k) a "
        f"covariance residual of order 2*pi*delta*max|H| = {estimate:.3g}, above the "
        f"tolerance every operation's residual is held to, tol = {float(tol):.3g} "
        f"(max|H| = {max_h:.4g} about the mean on-site energy, measured here), so which "
        "operations this Hamiltonian obeys would be read off the geometry as much as "
        "off the hoppings. Nothing was built. The cell's group is "
        f"{group.name} #{group.number_str}, an upper bound on what this Hamiltonian "
        "obeys.",
        value=delta, threshold=float(tol) / (2.0 * np.pi * max_h),
        context={"group": group, "number_str": str(group.number_str),
                 "position_defect": delta, "max_h": max_h, "tol": float(tol)})


def _time_reversal_square(group, validated, *, tolerances=EXACT):
    r"""The pure time reversal's square, against the one the tables read here ask for.

    Time reversal is an antiunitary operation ``A = X K``, and the detector
    measures its intertwiner ``X`` like any other.  Its square ``A² = X X̄`` is a
    unitary symmetry of ``H(k)``, and rescaling ``X → λX`` multiplies it by
    ``|λ|²``, so where it is a multiple of the identity its **sign** belongs to
    the Hamiltonian: ``+1`` on spin-free states, ``-1`` on spinors.  The tables
    ask for one sign, the one the group's own multiplication table gives
    ``T·T`` -- ``+1`` in single-valued tables, ``-1`` in double-valued ones --
    and a time reversal whose square is the other one is not the time reversal
    those tables describe: a spinless model carrying spinors in its orbital
    index, as a wannier90 import does, read single-valued, or a model read
    double-valued whose only antiunitary symmetry with the k-action of time
    reversal squares to ``+1``.

    Where the solution space is more than one-dimensional -- a commutant, whose
    members can square to either sign -- the question is whether *some* member
    has the sign asked, and it is linear: time reversal's space is closed under
    transposition, and a unitary ``X`` squares to ``s`` exactly where
    ``Xᵀ = s X``.  So the member kept is the detector's own where its square
    already is the sign; otherwise the unitary nearest its part of that
    symmetry (or of one of three fixed combinations of the space), kept only
    where it squares to the sign and stays inside the space; and where neither
    is found, the detector's own again if its square reads the sign within the
    margin a numerical model's noise leaves it.  That member is the
    intertwiner every later step reads.

    Returns ``None`` where the group has no pure time reversal or the detector
    refuted it; otherwise ``ind``, ``asked`` (the sign the tables ask for),
    ``measured`` (the sign of the square of the member the verdict rests on --
    the one kept, or where none has the sign asked the detector's own -- where
    that square is a multiple of the identity, else ``nan``), ``obeyed`` and
    ``member`` (the member of the sign asked, or ``None`` where the space holds
    none).
    """
    position = next((p for p, s in enumerate(group.symmetries) if s.time_reversal
                     and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                        np.eye(3, dtype=int))
                     and np.abs(((np.asarray(s.translation, dtype=float) + 0.5) % 1.0)
                                - 0.5).max() <= 1e-6), None)
    if position is None:
        return None
    # Pure time reversal squares to the identity, and the sign the tables ask for
    # is their valuedness: -1 in double-valued tables, +1 in single-valued ones.
    # Nothing else would be read off the multiplication table, so it is not built.
    return _antiunitary_square(group, validated, position,
                               asked=-1 if group.spinor else 1, tolerances=tolerances)


def _antiunitary_square(group, validated, position, *, asked=None, tolerances=EXACT):
    r"""One antiunitary operation's square, against the sign the group's table gives it.

    :func:`_time_reversal_square`'s reading, for any antiunitary operation
    ``g·T`` whose unitary part squares to the identity modulo the lattice: its
    square ``X X̄`` is then an intertwiner of the identity, rescaling ``X``
    moves it only by ``|λ|²``, and where it is a multiple of the identity its
    sign is the Hamiltonian's own, to be read against the sign the group's
    multiplication table gives ``(g·T)²``.  The solution space of such an
    operation is closed under transposition as time reversal's is, so a member
    of the sign asked is looked for the same way: in the members of its own
    form, or where none has it (:func:`_own_form_read`) in the detector's
    whole space, which then says whether any matrix with its action on ``k``
    squares to the sign asked.

    Returns ``None`` where the detector refuted the operation, it carries no
    member, or its unitary part does not square to the identity -- a threefold
    or a fourfold with time reversal, whose square is another operation with a
    phase of its own, and so no sign to read; otherwise the record
    :func:`_time_reversal_square` returns.  ``asked`` is the sign the table
    would give, where that is known without it, and the table is then not built.
    ``tolerances`` is the record of what the call declared: a member read off a
    model symmetric only within its precision squares to the sign that far
    off, and its solution space holds a member to that precision.
    """
    from .symmetry import _identity_position, _unitarized

    ind = int(group.symmetries[position].ind)
    member = validated._intertwiners.get(ind)
    if ind in set(validated.broken) or member is None:
        return None
    if asked is None:
        prod, _tdiff, spf = group.get_product_table(get_diff=True)
        if int(prod[position, position]) != _identity_position(group):
            return None                          # no square of its own to read
        asked = int(spf[position, position])
    asked = int(asked)
    zero, snap, in_span = (tolerances.noise(ZERO), tolerances.noise(SNAP),
                           tolerances.noise(IN_SPAN))
    X = np.asarray(member, dtype=complex)
    eye = np.eye(X.shape[0])
    square = X @ X.conj()
    scalar = complex(np.trace(square)) / X.shape[0]
    # the sign, which is what no rescaling of X changes: a member the detector
    # left unnormalized squares to a positive multiple of it
    measured = (float(np.sign(scalar.real))
                if (float(np.abs(square - scalar * eye).max()) <= _TIME_REVERSAL_SQUARE_MARGIN
                    and scalar.real != 0.0)
                else float("nan"))
    out = {"ind": ind, "asked": asked, "measured": measured, "obeyed": True, "member": X}
    if float(np.abs(square - asked * eye).max()) <= snap:
        return out                               # the detector's own, of the sign asked
    basis = (getattr(validated, "_bases", None) or {}).get(ind)
    if basis is None or not len(np.asarray(basis)):
        # an operation its own form refuted has no matrix of its own: whether
        # any matrix with its action on k squares to the sign asked is read
        # off the detector's whole space
        basis = ((getattr(validated, "_own_form", None) or {}).get(ind) or {}).get("space")
    basis = (None if basis is None or len(np.asarray(basis)) < 2
             else np.asarray(basis, dtype=complex))
    if basis is not None:
        flat = basis.reshape(len(basis), -1)
        q, r = np.linalg.qr(flat.T)                  # the space, orthonormal
        span = q[:, np.abs(np.diag(r)) > zero * max(1.0, float(np.abs(r).max()))]
        rng = np.random.default_rng(17)
        seeds = [X] + [np.einsum("j,jab->ab",
                                 rng.normal(size=len(basis))
                                 + 1j * rng.normal(size=len(basis)), basis)
                       for _ in range(3)]
        for Y in seeds:
            U = _unitarized(0.5 * (Y + asked * Y.T), tol=zero)
            if U is None:
                continue
            inside = U.reshape(-1) - span @ (span.conj().T @ U.reshape(-1))
            if (float(np.abs(inside).max()) <= in_span
                    and float(np.abs(U @ U.conj() - asked * eye).max())
                    <= _TIME_REVERSAL_SQUARE_MARGIN):
                out["member"] = U
                out["measured"] = float(np.sign((complex(np.trace(U @ U.conj()))
                                                 / U.shape[0]).real))
                return out
    if float(np.abs(square - asked * eye).max()) <= _TIME_REVERSAL_SQUARE_MARGIN:
        return out                               # the sign asked, read off a noisy member
    out.update(obeyed=False, member=None)
    return out


def _time_reversal_member(validated, square):
    """The detector's record with time reversal's member of the sign asked in place.

    ``square`` is :func:`_time_reversal_square`'s reading; where it kept the
    detector's own member, or found none of the sign asked, the record is
    returned as it stands.
    """
    if (square is None or not square["obeyed"]
            or square["member"] is validated._intertwiners.get(square["ind"])):
        return validated
    return replace(validated, _intertwiners={**validated._intertwiners,
                                            int(square["ind"]): square["member"]})


def _ansatz_refuted(model, group, assignment, validated, *, layout, spin,
                    blocks=None, max_candidates=2 ** 20, skip=()):
    r"""The operations the detector left undetermined, resolved by tensor structure.

    The detector's verdicts are per **k-action class**, and a surviving class of
    more than one member proves only that *some* operator with that k-action is
    a symmetry: its members are marked ``"undetermined"``
    (:attr:`ValidatedGroup.verdicts`) and are individually undetermined there.
    On a slab that is the whole horizontal-mirror family — every class is the
    pair ``{g, σ_h·g}``, the in-plane partner satisfies the intertwiner problem,
    and no presentation can change that, because a layer's ``H(k)`` does not
    depend on ``k_z`` and ``σ_h·g`` and ``g`` therefore impose the identical
    constraint.

    What *does* separate them is the **tensor structure** of the representation:
    ``D(g) = O(g) ⊗ S(g)``, with ``O`` fixed by the site permutation (the
    identity on a flat layer's sites) and ``S(g)`` the operation's own spin
    lift, and ``S(σ_h·g)`` is a different SU(2) element from ``S(g)``.  Each
    operation's own form -- each site's states carried onto its image's, with
    its own spin lift -- was read where the detector spoke
    (:func:`_own_form_read`): the solution space every reading here reads is
    the members of that form, which tells a pair apart by its lift on a
    multiplet of any dimension, and the operations none of whose members has
    it are refuted there and passed in as ``skip``.  What is left is the form
    on the **multiplets** (:func:`..symmetry._ansatz_defects`): each carried
    where the operation carries its site, with the tables' lift on a
    one-dimensional multiplet and the model's own block on a wider one, which
    on a spinless model is what tells a pair apart.  An operation it refutes
    -- its scaled defect above the gate -- is **broken by tensor structure**, a
    per-operation verdict the descent below can act on where ``_build_rep``'s
    own covariance gate (a *global* "no candidate closes") cannot.

    Applied **exactly to the operations the detector declares undetermined** —
    nowhere else.  Where every k-action class is a singleton (every fully
    periodic model on its primitive cell, and every 2D model on a substrate)
    this runs on nothing and costs nothing: a singleton class carries the
    detector's own ``"obeyed"`` verdict, and its own form is the whole question.

    **One lift per operation, and it is the tables' own.**  An operation is never
    represented through its ``σ_h`` partner's lift: representing ``g``
    by ``S(σ_h)·S(g)`` is representing a *different* operation, and a model
    written for the opposite spin quantization axis reads the group it
    genuinely has in pythtb's convention.

    **So the test asks the operation's own branch, in every environment**, as
    its own form is read.  On the
    ``g⁻¹`` branch a member is read as ``D(g)†``, whose spin factor is
    ``S(g)†``: on a **fourfold** ``S(C4)† = −S(σ_h)·S(C4)``, the ``σ_h``
    partner's lift exactly, and on an antiunitary twofold it can be the
    partner's too, so accepting either branch keeps an operation on a lift the
    tables do not give it.  On a substrate cell the partner is not in the cell
    at all; on a free-standing one it is, and the operation kept on its
    partner's lift reaches the group besides it.  Measured on lin3g
    free-standing, where every multiplet is one spinor orbital: on the ``g``
    branch exactly one member of each of the sixteen ``σ_h`` pairs holds, and
    the two ``-4`` members and the time-reversal partners of the two diagonal
    twofolds, which pass on the ``g⁻¹`` branch alone, are refuted with the
    rest; and on the SRSI paper's ``P41'`` model in the opposite-axis
    transcription the ``g`` branch keeps ``{E, C2(z), m(x), m(y)}`` at ~7e-16
    on each, where both branches together also keep ``C4`` and ``C4⁻¹`` (``g``
    branch 1.0, ``g⁻¹`` branch 6.7e-16) and the survivors are not a group.
    Only the acceptance of an operation reads one branch: which branch the
    representation is built on is :func:`..symmetry._build_rep`'s convention.

    Returns ``(tested, refuted)`` — the per-operation record of the test, and
    the subset that failed — or the :class:`..refusal.Refusal` a declared
    spinor pair the operations do not carry onto one pair makes, or that a
    multiplet's own block enumeration made past ``max_candidates``
    (``rep_search_too_large``, where ``blocks`` was not already built for it).
    """
    from .symmetry import _ansatz_defects

    shared = set(validated.undetermined) - set(skip or ())
    if not shared:
        return {}, {}
    positions = [i for i, s in enumerate(group.symmetries) if int(s.ind) in shared]
    got = _ansatz_defects(model, group, assignment, validated, positions,
                          layout=layout, spin=spin, blocks=blocks,
                          max_candidates=max_candidates, orients=(1,))
    if isinstance(got, Refusal):
        return got
    return got, {i: r for i, r in got.items() if not r["realizable"]}


def _grid_covariance(model, group, *, nk, params=None, context=None):
    r"""A candidate matrix's covariance residual, measured as the detector measures its own.

    Returns the measure :func:`..symmetry._own_form_spaces` hands a
    candidate to: called with an operation's position in
    ``group.symmetries`` and a matrix, it returns
    ``max_k ‖U H(k)^(*) U† − H(Rk)‖`` for ``U`` the unitary nearest that matrix,
    over the detection grid the detector solved on -- the same grid, read off
    the call's measurement context where it holds it, and the same number the
    detector certifies each of its own candidates by.
    """
    from .symmetry import (_GridHamiltonians, _as_k_action, _candidate_measures,
                           _ConstraintPair)

    k_pts = _detection_grid(int(group.dim_k), int(nk))
    grid = None if context is None else context.grid_for(model, k_pts)
    if grid is None:
        grid = _GridHamiltonians(model, k_pts, params)
    syms = group.symmetries

    def residual(position, candidate):
        sym = syms[int(position)]
        act = _as_k_action(group.k_action_of(sym), int(group.dim_k))
        pair = _ConstraintPair(grid, grid.rotated(np.asarray(act(grid.k_pts), dtype=float)),
                               bool(sym.time_reversal))
        _unitaries, got, _unitarity = _candidate_measures(
            [np.asarray(candidate, dtype=complex)], int(grid.nstate), pair=pair)
        return float(got[0])

    return residual


def _own_form_read(model, group, assignment, spin, validated, *, tolerances, covariance=None):
    r"""Every kept operation's solution space, read once in its own form.

    The detector's spaces are per action on ``k``, and hold every matrix with
    that action: another operation's, or one composed with a symmetry the
    Hamiltonian has beyond the crystal's.  Each operation is held to its own
    form -- each site's states carried onto its image's, with its own spin lift
    on states that carry a spin -- once, here, and the members of that form are
    what every later reading of the operation reads: the multiplets, the
    transport round an orbit, the tensor structure, the sectors and the
    representation (:func:`..symmetry._own_form_spaces`).  ``tolerances`` is the
    record of what the call declared, every threshold judging noise against
    it, and ``covariance`` is :func:`_grid_covariance`'s measure, which a form
    missed by more than rounding is held to as well: the member nearest it,
    its part in the form, carries ``H(k)`` onto ``H(gk)`` within the covariance
    gate the operation was kept at, or the operation is not one of this
    Hamiltonian's -- as spin blocks written in a frame turned against the
    lattice by more than the gate allows show.

    Returns ``(validated, refuted)``: the detector's record with each kept
    operation's space in its own form and its intertwiner a member of it -- the
    detector's own, made unitary on the form where it lies outside -- and the
    records of the operations none of whose members has their form, refuted by
    it (``"tensor structure"``, with ``own_form`` naming the condition that
    failed and ``covariance`` how far the part in the form carries ``H``,
    where that was measured).  A refuted operation's space is empty, and its
    intertwiner and the detector's whole space are left as the detector
    measured them, for the one question asked of them: whether any matrix
    with its action on ``k`` has the square the tables ask for, which time
    reversal's verdict says (:func:`_antiunitary_square`).  The record
    returned says it was read, so that it is read once:
    :func:`..symmetry._build_rep`, handed it, reads it as it stands.
    """
    from .symmetry import _own_form_spaces, _unitarized

    kept = set(validated.indices)
    positions = [p for p, s in enumerate(group.symmetries) if int(s.ind) in kept]
    spaces, records = _own_form_spaces(model, group, assignment, validated, positions,
                                       spin=spin, tol=tolerances.matrix or None,
                                       covariance=covariance)
    bases = dict(validated._bases)
    intertwiners = dict(validated._intertwiners)
    own_form = {}
    inside = tolerances.noise(IN_SPACE)
    for ind, space in spaces.items():
        if bases.get(ind) is space:
            continue                       # every member already of the form
        if not len(space):
            own_form[int(ind)] = {"own_form": records[ind]["own_form"],
                                  "covariance": records[ind].get("covariance"),
                                  "space": bases.get(ind)}
        bases[ind] = space
        own = intertwiners.get(ind)
        if not len(space) or own is None:
            continue
        own = np.asarray(own, dtype=complex)
        flat = np.asarray(space).reshape(len(space), -1)
        y = np.linalg.lstsq(flat.T, own.reshape(-1), rcond=None)[0]
        near = np.einsum("j,jab->ab", y, space)
        if float(np.linalg.norm(own - near)) <= inside * max(float(np.linalg.norm(own)), 1.0):
            continue                       # the detector's own member is of the form
        member = _unitarized(near)
        intertwiners[ind] = member if member is not None else _unitarized(space[0])
    refuted = {int(i): dict(r) for i, r in records.items() if not r["realizable"]}
    return (replace(validated, _bases=bases, _intertwiners=intertwiners, _own_form=own_form),
            refuted)


def _own_form_once(model, group, assignment, spin, validated, *, tol=None, params=None):
    r"""The record each operation's own form was read on: as it stands where it was read, else read here, once.

    :func:`realized_group` reads every record in the verdict stage
    (:func:`_own_form_read`); a record handed in as the detector measured it,
    :func:`_validate_group`'s own, is read here instead, by the same function.
    ``tol`` is the declared tolerance, relative to ``max|H|``, the reading
    judges noise against -- the one the multiplets handed in were measured
    at, ``None`` for the exact defaults -- and an operation whose members miss
    their form by more than rounding holds where the member nearest it, its
    part in the form, carries ``H(k)`` within the covariance gate it was kept
    at, measured over a mesh as many points as the record's own.
    """
    if getattr(validated, "_own_form", None) is not None:
        return validated
    dim = max(int(group.dim_k), 1)
    side = max(2, int(round(float(max(int(validated.n_kpts), 1)) ** (1.0 / dim))))
    covariance = _grid_covariance(model, group, nk=side, params=params)
    tolerances = replace(EXACT, matrix=float(tol or 0.0))
    got, _refuted = _own_form_read(model, group, assignment, spin, validated,
                                   tolerances=tolerances, covariance=covariance)
    return got


def _translation_permutation(model, assignment, position):
    r"""The matrix carrying every orbital onto its translate's under one pure translation.

    ``P[σ(i), i] = 1`` for the site permutation ``σ`` the operation at
    ``position`` makes, each site's orbitals onto its image's in the order the
    sites list them, spin by spin.  ``None`` where a site and its image hold
    different numbers of orbitals.
    """
    perm = np.asarray(assignment.permutations, dtype=int)
    site_orbs = tuple(assignment.site_orbitals)
    nspin, nstate = int(model.nspin), int(model.nstate)
    P = np.zeros((nstate, nstate), dtype=complex)
    for i, orbs in enumerate(site_orbs):
        dst = site_orbs[int(perm[int(position), i])]
        if len(dst) != len(orbs):                   # pragma: no cover - defensive
            return None
        for slot, a in enumerate(orbs):
            for s in range(nspin):
                P[int(dst[slot]) * nspin + s, int(a) * nspin + s] = 1.0
    return P


def _translation_held(model, group, assignment, validated, position, *, spin=None,
                      tol=None, commuting=()):
    r"""Whether ``H(k)`` obeys one pure translation once each site's orbitals are carried onto its translate's within the site.

    A translation carries every site onto a copy of it, and a model may write
    the copy's orbitals in another basis -- a frame turned from one site to the
    next, a sign or a phase on an orbital -- and still be one crystal written on
    a bigger cell.  So the translation is read as every operation is, on its own
    solution space: the matrices commuting with ``H(k)`` at every ``k``, since a
    translation acts on ``k`` as the identity does, restricted to the blocks
    carrying each site's states onto its translate's
    (:func:`..symmetry._block_members`).  It holds where that restriction has a
    member nonsingular on every block (:func:`..symmetry._filled_check`): any
    unitary on each site's orbitals is allowed, the permutation of the sites
    itself among them, and on states that carry a spin, native or declared,
    the spin untouched -- every operation is read with its own spin rotation,
    and a translation's is the identity.  ``tol`` is the declared tolerance,
    relative to ``max|H|``, every one of those readings takes; ``None`` where
    nothing was declared.

    Translations commute, so the matrix a translation holds with is read among
    the members commuting with ``commuting``, the matrices already read for the
    others: the member nearest the permutation itself
    (:func:`_translation_permutation`), or the first of the fixed generic ones
    the verdict reads (:func:`..symmetry._generic_members`) where that one is
    singular, made unitary block by block.

    Returns ``(held, matrix, turn)``: whether the translation holds so; the
    unitary it holds with, ``None`` where no member commuting with
    ``commuting`` is nonsingular; and how far that unitary turns each site's
    orbitals onto its translate's -- half the smallest arc of the unit circle
    holding every eigenvalue of its blocks, so that one phase common to all of
    them is no turn, and ``0.0`` within the declared tolerance.  A space the
    detector kept none of, or one past the read-off's own ceiling, is read as
    not holding: the permutation alone is measured there
    (:func:`_translation_residual`).
    """
    from .symmetry import (_READOFF_MAX_BYTES, _ansatz_blocks, _atoms_of_site,
                           _block_members, _channel_permutation, _filled_check,
                           _generic_members, _multiplets_of)

    nothing = (False, None, float("nan"))
    basis = (getattr(validated, "_bases", None) or {}).get(
        int(group.symmetries[int(position)].ind))
    nstate = int(model.nstate)
    if basis is None:
        return nothing
    basis = np.asarray(basis)
    if (basis.ndim != 3 or not basis.shape[0]
            or basis.shape[0] * nstate * nstate * 16 > _READOFF_MAX_BYTES):
        return nothing
    # every site one block, carried onto its translate's whole
    site_orbs = tuple(tuple(int(o) for o in orbs) for orbs in assignment.site_orbitals)
    sites = _multiplets_of(model, assignment, spin, tuple((orbs,) for orbs in site_orbs),
                           tol=tol)
    image = _channel_permutation(assignment, int(position), model, sites)
    blocks = _ansatz_blocks(sites._chan_states, image, 1)
    if any(len(rows) != len(cols) for rows, cols in blocks):   # pragma: no cover - defensive
        return nothing
    pieces, lifts = blocks, None
    if int(model.nspin) == 2 or spin is not None:
        # a translation turns no spin: every operation is read with its own
        # spin rotation, the identity here, so the change of basis acts on the
        # orbitals alone, the same on both states of every spinor
        spinors = [[(2 * o, 2 * o + 1) for o in orbs] for orbs in site_orbs]
        if spin is not None:
            spinors = [[tuple(int(x) for x in a) for a in site]
                       for site in _atoms_of_site(site_orbs, spin)]
        pieces = [(np.asarray(to, dtype=int), np.asarray(of, dtype=int))
                  for s in range(len(site_orbs)) for to in spinors[int(image[s])]
                  for of in spinors[s]]
        lifts = [np.eye(2, dtype=complex)] * len(pieces)
    V, _reach = _block_members(basis, pieces, tol=noise_floor(_IN_BLOCKS, tol), lifts=lifts)
    held, _unfilled = _filled_check(V, blocks, tol=tol)
    if not held:
        return nothing
    if commuting:
        # the members commuting with every matrix already read for another
        # translation, to the precision a relation among measured matrices has
        A = np.stack([np.concatenate([(X @ D - D @ X).reshape(-1) for D in commuting])
                      for X in V], axis=1)
        _u, s, vh = np.linalg.svd(A, full_matrices=False)
        gate = noise_floor(ASSEMBLED, tol) * max(float(s[0]), 1.0)
        keep = [i for i in range(len(s)) if float(s[i]) <= gate]
        V = (np.stack([np.einsum("j,jab->ab", vh[i].conj(), V) for i in keep])
             if keep else V[:0])
    if not len(V):
        return True, None, float("nan")
    P = _translation_permutation(model, assignment, position)
    nearest = np.linalg.lstsq(V.reshape(len(V), -1).T, P.reshape(-1), rcond=None)[0]
    singular = noise_floor(SINGULAR, tol)
    member = None
    for X in (np.einsum("j,jab->ab", nearest, V), *_generic_members(V)):
        scale = max(float(np.abs(X).max()), 1.0)
        if all(float(np.linalg.svd(X[np.ix_(rows, cols)], compute_uv=False)[-1])
               > singular * scale for rows, cols in blocks):
            member = X
            break
    if member is None:
        return True, None, float("nan")
    D = np.zeros((nstate, nstate), dtype=complex)
    phases = []
    for rows, cols in blocks:
        u, _s, wh = np.linalg.svd(member[np.ix_(rows, cols)])
        D[np.ix_(rows, cols)] = u @ wh
        phases.append(np.angle(np.linalg.eigvals(u @ wh)))
    around = np.sort(np.mod(np.concatenate(phases), 2.0 * np.pi))
    gap = float(np.max(np.diff(np.concatenate([around, around[:1] + 2.0 * np.pi]))))
    turn = (2.0 * np.pi - gap) / 2.0
    return True, D, (0.0 if turn <= noise_floor(SNAP, tol) else turn)


def _translation_residual(model, assignment, position, k_pts, params, grid=None):
    r"""How far ``H(k)`` is from obeying one **pure translation** as written.

    For a model that writes every copy of a site in one basis, in pythtb's
    convention I

    .. math:: T_\tau\,|\varphi_i(k)\rangle = e^{-2\pi i\,k\cdot\tau}\,
              |\varphi_{\sigma(i)}(k)\rangle,

    so ``D(τ)`` is the orbital permutation ``σ`` times a *global* phase — no
    internal factor (a translation rotates nothing), no SU(2) lift (it acts on
    spin as the identity) — and the covariance statement collapses to
    ``P H(k) P† = H(k)``, ``P`` the permutation matrix
    (:func:`_translation_permutation`); ``max_k`` of its defect is returned.  A
    model that writes the copies in bases turned from site to site obeys the
    translation with that change of basis beside the permutation, which
    :func:`_translation_held` reads.  Returns ``None`` when the permutation
    cannot be built (translationally paired sites carrying different numbers of
    orbitals).
    """
    P = _translation_permutation(model, assignment, position)
    if P is None:                                   # pragma: no cover - defensive
        return None
    # `grid` is the detector's own evaluation of H on this same mesh, handed in
    # by `realized_group`'s measurement context when the two k-sets are byte for
    # byte the same one (checked there, not assumed here).  It is the same array
    # either way -- `_GridHamiltonians` and the call below evaluate the model's one
    # Bloch sum on the same points -- so this measurement does not change, only
    # its cost.
    H = (np.asarray(grid.full()) if grid is not None
         else np.asarray(_bloch.hamiltonian(model, np.atleast_2d(k_pts), params)))
    return float(np.abs(np.einsum("ab,kbc,dc->kad", P, H, P.conj()) - H).max())


def _lattice_verdict(model, group, assignment, validated, refuted, tol, k_pts,
                     params, context=None, *, spin=None, matrix_tol=None) -> dict:
    r"""What the Hamiltonian did with the cell's own pure translations.

    A lattice supergroup carries ``_lattice_index - 1`` fractional translations
    ``{1|τ}`` and, with each of them, a whole coset of partners ``τ·g``.  Every
    one of them sits in the **identity's k-action class** — a pure translation
    acts trivially on ``k``, and ``τ·g`` acts exactly as ``g`` does — so the
    detector marks them ``"undetermined"``, and the tensor structure tells the
    partners apart (:func:`_ansatz_refuted`).

    **A translation is read at the sites.**  It carries every site onto a copy
    of it, and a model may write the copies in bases turned from site to site,
    so it holds where its solution space has a member carrying each site's
    orbitals onto its translate's by any change of basis within the site
    (:func:`_translation_held`); where the copies are written alike that member
    is the permutation of the sites itself, measured directly against ``tol``
    (:func:`_translation_residual`), and nothing is read beyond it.
    Translations commute, so each is read among the members commuting with the
    matrices the others hold with.  A Hamiltonian obeying several translations
    only with changes of basis that do not commute -- magnetic translations,
    a flux threading the cell -- obeys no crystal's translations, and of those
    only the translations obeyed as written are counted.  A translation the
    declaration or the centres refuted stays refuted whatever the hoppings
    obey: the crystal they state has no such translation.

    Returns the record the descent step carries:
    ``index``, ``fractional_translations``, ``operations`` (one entry per
    translation, with the ``.ind``, the reduced and Cartesian ``τ``, the direct
    covariance ``residual`` of the permutation itself, the scaled
    tensor-structure defect, and ``refuted_by`` in the descent's own words: the
    detector's reason, ``"no symmetry matrix"`` or ``"covariance"``, when the
    detector's own measurement refuted the translation, the refuting
    measurement's own word where the verdicts that read the detector's spaces
    refuted it -- ``"tensor structure"``, ``"declaration"``, ``"centres"`` --
    ``"translation"`` when none of them did and the Hamiltonian does not obey
    it, and ``None`` when it does), ``realized`` (the translations the
    Hamiltonian **obeys** — a supercell, if any), ``dressing`` (for each of
    them, the ``matrix`` it is obeyed with, which carries each site's orbitals
    onto its translate's, and the ``turn`` that matrix makes,
    :func:`_translation_held`), ``refuted`` (the translations refuted here and
    by nothing before) and ``coset_partners_refuted`` (how many refuted
    unitary operations are not themselves pure translations).
    """
    index = int(getattr(group, "_lattice_index", 1) or 1)
    if index <= 1:
        return {}
    broken = set(validated.broken)
    reason_of = {int(d["ind"]): str(d["refuted_by"]) for d in validated.discrepancies}
    refuted = dict(refuted or {})
    lattice = np.asarray(group.lattice, dtype=float)
    ident = np.eye(3, dtype=int)
    antiunitary = {int(s.ind) for s in group.symmetries if s.time_reversal}
    gone = (broken | set(refuted)) - antiunitary

    found, pure_inds = [], set()
    for position, s in enumerate(group.symmetries):
        if bool(s.time_reversal):
            continue
        if not np.array_equal(np.rint(np.asarray(s.rotation, dtype=float)).astype(int),
                              ident):
            continue
        t = np.asarray(s.translation, dtype=float).reshape(-1)
        t = t - np.rint(t)
        # zero as the identification read it: no longer than the finder's distance
        if float(np.linalg.norm(t @ lattice)) <= float(group.symprec):
            continue
        ind = int(s.ind)
        pure_inds.add(ind)
        # which judge refuted it: the detector's own measurement (its reason),
        # or a verdict read on the detector's spaces, in its own word
        why = (reason_of[ind] if ind in broken
               else str(refuted[ind].get("refuted_by", "tensor structure"))
               if ind in refuted else None)
        scaled = (float(refuted[ind]["scaled"]) if ind in refuted and "scaled" in refuted[ind]
                  else float("nan"))
        residual = _translation_residual(
            model, assignment, position, k_pts, params,
            grid=None if context is None else context.grid_for(model, k_pts))
        found.append((position, ind, tuple(float(x) for x in t),
                      tuple(float(x) for x in t @ lattice), why, scaled, residual))

    # THE HOPPINGS' TO DECIDE: every translation the detector, the declaration
    # and the centres left standing.  Those obeyed as written carry the
    # permutation itself; every other one is read at the sites, among the
    # members commuting with the matrices read before it
    standing = [f for f in found if f[1] not in broken and f[4] in (None, "tensor structure")]
    bare = [f for f in standing if f[6] is not None and f[6] <= float(tol)]
    commuting = [_translation_permutation(model, assignment, f[0]) for f in bare]
    dressing = [{"translation": f[2], "turn": 0.0, "matrix": M}
                for f, M in zip(bare, commuting)]
    commute = True
    for position, _ind, tau, _cart, _why, _scaled, residual in standing:
        if residual is not None and residual <= float(tol):
            continue
        held, matrix, turn = _translation_held(model, group, assignment, validated,
                                               position, spin=spin, tol=matrix_tol,
                                               commuting=tuple(commuting))
        if held and matrix is None:
            commute = False
        elif held:
            commuting.append(matrix)
            dressing.append({"translation": tau, "turn": turn, "matrix": matrix})
    if not commute:
        # translations obeyed only with changes of basis that do not commute:
        # a magnetic translation group, which is no crystal's, so only the
        # translations obeyed as written are counted
        dressing = dressing[:len(bare)]
    realized = tuple(d["translation"] for d in dressing)

    entries = tuple(
        {"ind": ind, "translation": tau, "cartesian": cart,
         "refuted_by": (None if tau in realized
                        else "translation" if why is None else why),
         "scaled": scaled, "residual": residual, "tol": float(tol)}
        for _p, ind, tau, cart, why, scaled, residual in found)
    # what the direct measurement refuted and nothing before it: a pure
    # translation is never left to inference, so it joins the refuted set with
    # its own reason and the descent removes it like any other broken operation
    add = {int(e["ind"]): {
        "scaled": float(e["residual"]), "defect": float(e["residual"]),
        "scale": float(e["tol"]), "orientation": 0, "dim": 0,
        "refuted_by": "translation", "residual": float(e["residual"])}
        for e in entries
        if e["refuted_by"] == "translation" and e["residual"] is not None}
    removed = gone | set(add)
    return {"index": index,
            "fractional_translations": tuple(group._fractional_translations),
            "operations": entries,
            "realized": realized,
            "dressing": tuple(dressing),
            "refuted": add,
            "coset_partners_refuted": len(removed - pure_inds)}


def _supercell_refusal(group, validated, lattice_record) -> Refusal:
    r"""The verdict on a Hamiltonian that obeys the cell's own pure translation.

    Not a tolerance and not a defect in the model: the model **is** a supercell,
    and every k-point label, EBR row and SRSI computed on its folded zone would
    be a statement about the wrong Brillouin zone.  Where the copies are
    written in bases turned from site to site, the message says how far, and
    ``context["dressing"]`` carries the matrices each translation is obeyed
    with.  A falsy :class:`..refusal.Refusal`, never a returned group; the loop
    that measured it names the cell's group in it and carries it in
    ``context["group"]`` (:func:`_named_refusal`).
    """
    # the finer lattice is the one the Hamiltonian was measured to have, i.e. the
    # one the REALIZED translations generate -- not every translation the point
    # set allows, which is what the cell allows and is a different statement
    taus = tuple(lattice_record["realized"])
    lattice = np.asarray(group.lattice, dtype=float)
    prim = _primitive_lattice(lattice, taus)
    tau = np.round(np.asarray(taus[0], dtype=float), 6)
    cart = np.round(np.asarray(taus[0], dtype=float) @ lattice, 6)
    turn = max(float(d["turn"]) for d in lattice_record["dressing"])
    turned = ("" if turn == 0.0 else
              ", each site's orbitals carried onto its translate's by a change of basis "
              f"within the site that turns them by up to {turn:.3g} rad")
    rebuild = ("Rebuild the model on its primitive cell"
               if turn == 0.0 else
               "Rebuild the model on its primitive cell, every copy's orbitals written in "
               "one basis -- context[\"dressing\"] holds the matrices carrying each site's "
               "orbitals onto its translate's --")
    return Refusal(
        "supercell",
        f"the Hamiltonian obeys the pure translation {tau.tolist()} (reduced) = "
        f"{cart.tolist()} (Cartesian) that this cell's point set allows{turned}, so the "
        f"model is a supercell: its lattice is an index-{int(prim['index'])} "
        "sublattice of the crystal's own, whose primitive vectors are "
        f"{np.round(np.asarray(prim['cartesian'], dtype=float), 6).tolist()} "
        "(Cartesian). Every "
        "k-point label, EBR row and symmetry indicator computed here would be "
        "computed on a folded Brillouin zone -- the bands at one k of this cell are "
        "the crystal's bands at several k of its own zone, and no table names that "
        f"combination -- so no group is returned. {rebuild} and ask again; or, if "
        "the translation is meant to be broken, break it in the Hamiltonian -- a "
        "supercell whose hoppings distinguish the copies is a primitive model of a "
        "bigger cell and is descended to automatically.",
        value=float(lattice_record["index"]), threshold=1.0,
        context={"translations": taus, "index": int(lattice_record["index"]),
                 "primitive_lattice": prim,
                 "dressing": tuple(lattice_record["dressing"]),
                 "validated": validated, "lattice": lattice_record})


def _declared_cell_reduction(model, group, assignment, *, spin, characters):
    r"""Step 1: the cell the **declared content** leaves, before any Hamiltonian.

    The first of the three steps a measurement is: ``typat=`` says
    the atoms are different and ``characters=`` says what they *are*, and both
    are statements about the crystal that are made before a hopping is read.
    An operation the declared content is not carried by is therefore not an
    operation of this cell at all — it is not a symmetry the Hamiltonian
    breaks, and it never reaches a verdict.

    Two reductions, one rule (:func:`..symmetry._content_compatible`): the
    **species split**, where two sites of one orbit were declared different
    content -- the whole content each holds, however the words partition its
    orbitals, and as corep content where a word states that alone -- and the
    operations exchanging them go, and the **stabilizer of a declared
    function**, where a shorthand names a Cartesian polynomial whose span at
    its site is not invariant and the operations carrying it out of itself go.
    A single ``px`` at a ``3m`` site is a crystal with one mirror.  What is
    kept is the largest group inside the operations left, where a declaration
    that leaves sites of an orbit undeclared leaves operations that do not
    compose.

    Returns ``(kept, excluded, read_in)`` — the ``.ind`` kept, one record
    per excluded operation with the reason it was excluded, and where each
    declaration's word was read — or a :class:`..refusal.Refusal` where the
    declaration cannot be read in *this* cell's site groups at all, which is a
    statement about the group the loop is on and not about the crystal: the
    loop carries it and asks again on the next one.
    """
    from . import _sitechars
    from .symmetry import _content_compatible, _declared_site_contents

    try:
        contents, read_in = _declared_site_contents(
            model, group, assignment, characters, spin=spin)
    except _sitechars.DeclarationError as exc:
        return Refusal(exc.gate, exc.message, context=dict(exc.context))
    except Exception as exc:                     # noqa: BLE001 - a site group with no table
        return Refusal(getattr(exc, "gate", "site_group"), str(exc), context={})
    kept, excluded = _content_compatible(group, assignment, contents)
    return kept, excluded, read_in


def _layer_declared_reading(model, labels, *, extras, spinor, spin, symprec, characters):
    r"""Step 1 read in a layer's own site groups: the free-standing layer group's.

    What a declaration says two sites are is read where its words were
    written, and a layer's site groups hold ``σ_h``: ``E'`` and ``E''`` are
    different irreps of ``-6m2``, as ``s`` and ``p_z`` are different functions
    there, and a site declared the one and a site declared the other are
    different sites in every environment, as two sites of different ``typat``
    are.  On a substrate ``σ_h`` is gone and both restrict to one irrep of
    ``3m``, so the substrate's own site groups cannot say it: this reads the
    declaration in the site groups of the same point set with nothing under
    it, the free-standing cell, and the substrate's step 1 carries what it
    excludes (:func:`_carried_reduction`).  Where the substrate's site groups
    name nothing for the words at all -- a free-standing layer's declaration,
    read on a substrate by the cross-check -- they are read in this cell's and
    restricted (:func:`_restricted_contents`).

    Returns ``{"group", "assignment", "kept", "excluded", "read_in"}`` -- that
    cell's group, the orbitals' orbits in it and :func:`_declared_cell_reduction`'s
    reading there -- or ``None`` where the words name nothing in those site
    groups, written in the in-plane names a substrate's own site groups read,
    or the cell is not identified.
    """
    cell, cell_typat = _decorated_cell(model, labels, environment="free-standing",
                                       decorations=extras)
    group = _group_of(cell, typat=cell_typat, spinor=bool(spinor), symprec=symprec)
    if isinstance(group, Refusal):
        return None
    assignment = _assign_orbitals(model, group, typat=labels)
    if isinstance(assignment, Refusal):
        return None
    got = _declared_cell_reduction(model, group, assignment, spin=spin,
                                   characters=characters)
    if isinstance(got, Refusal):
        return None
    kept, excluded, read_in = got
    return {"group": group, "assignment": assignment, "kept": kept,
            "excluded": excluded, "read_in": read_in}


def _restricted_contents(model, written, group, layout, *, spin, characters) -> tuple:
    r"""The declared words read where they were written, restricted to this group's site groups.

    ``written`` is the declaration as a larger group of the same point set
    reads it -- a layer's free-standing group, for the layer on a substrate
    (:func:`_layer_declared_reading`) -- whose site irreps the words are:
    ``E'`` and ``A2''`` of a ``-6m2`` site, which a substrate's ``3m`` names
    nothing for.  Read there, on the multiplets this group's Hamiltonian
    measures, each content is re-keyed onto this group's own operations,
    matched by rotation and translation modulo the lattice
    (:func:`_operation_key`), and the operations this group does not have drop
    out: that is the restriction of the declared content to the smaller site
    groups, ``A2''`` read as the ``A1`` of ``3m`` it restricts to, as after any
    descent.

    Returns one :class:`.._sitechars.DeclaredContent` per declared content, or
    ``()`` where the words name nothing in the larger group's site groups
    either.
    """
    from . import _sitechars
    from .symmetry import _normalize_declaration, _site_tables

    larger, there = written["group"], written["assignment"]
    kept_u = {int(s.ind) for s in larger.symmetries if not s.time_reversal}
    kept_au = {int(s.ind) for s in larger.symmetries if s.time_reversal}
    try:
        tables = _site_tables(larger, there, kept_u, kept_au, range(len(there.site_orbitals)))
        decls = _normalize_declaration(characters, model, there, larger, tables, layout,
                                       spin=spin, read_in=written.get("read_in"))
    except _sitechars.DeclarationError:
        return ()
    theirs = {_operation_key(s): int(s.ind) for s in larger.symmetries}
    pairs = [(theirs[_operation_key(s)], int(s.ind)) for s in group.symmetries
             if _operation_key(s) in theirs]
    return _rekey_contents(tuple(d["content"] for d in decls), pairs, {})


def _carried_reduction(group, assignment, own, wider):
    r"""Step 1 on this cell, with the declared content also read in a larger group of its points.

    ``wider`` is the declaration read in the site groups of a group of the same
    point set that holds more operations -- a layer's free-standing group, for
    the layer on a substrate (:func:`_layer_declared_reading`) -- and every
    operation it excludes is excluded here too wherever this cell has it,
    matched by rotation and translation modulo the lattice
    (:func:`_operation_key`): two sites declared different content are
    different sites whatever group is asked about them.  ``own`` is this cell's
    own reading (:func:`_declared_cell_reduction`), whose exclusions stand
    beside those, or the refusal it made where the words name nothing in this
    cell's site groups; where it read them, it also says where they were read.

    What is carried from either reading is what the declared content is not
    carried by.  The operations kept are then the largest group inside the
    operations neither excludes, asked once, here
    (:func:`..symmetry._largest_compatible_group`), so an operation either
    reading left out only because its own largest group did not hold it is
    asked again rather than carried.

    Returns ``(kept, excluded, read_in)`` on this group, as
    :func:`_declared_cell_reduction` does.
    """
    from .symmetry import _largest_compatible_group, _outside_the_group

    if isinstance(own, Refusal):
        excluded, read_in = {}, dict(wider["read_in"])
    else:
        _kept, excluded, read_in = own
        excluded, read_in = ({i: rec for i, rec in excluded.items()
                              if not _outside_the_group(rec)}, dict(read_in))
    lattice = np.asarray(group.lattice, dtype=float)
    theirs = {_operation_key(s): int(s.ind) for s in wider["group"].symmetries}
    # a site of the larger group's orbits by the one it is here, through an
    # orbital on it: the two groups number their sites each their own way
    there = np.asarray(wider["assignment"].site_of_orbital, dtype=int)
    here = np.asarray(assignment.site_of_orbital, dtype=int)
    site_here = {int(there[o]): int(here[o]) for o in range(len(there))}
    for sym in group.symmetries:
        ind = int(sym.ind)
        rec = wider["excluded"].get(theirs.get(_operation_key(sym)))
        if rec is None or ind in excluded or _outside_the_group(rec):
            continue
        name = _record_name(sym, lattice)
        a, b = (site_here[int(s)] for s in rec["sites"])
        where = next((w for orbs, w in wider["read_in"].items()
                      if set(orbs) & set(rec["orbitals"])), "")
        excluded[ind] = dict(
            rec, ind=ind, name=name, sites=(a, b),
            note=(f"#{ind} {name}: {rec['reason']} ("
                  + (f"site {a}" if a == b else f"site {a} -> site {b}")
                  + f", orbitals {rec['orbitals']})"
                  + (f", as the {where} reads it" if where else "")))
    declared = {int(o) for reading in (read_in, wider["read_in"]) for orbs in reading
                for o in orbs}
    kept, excluded = _largest_compatible_group(group, assignment, declared, excluded)
    return kept, excluded, read_in


def _declared_cell(base_cell, base_typat, group, kept, *, spinor, include_TR, symprec):
    r"""The reduced cell of a step 1 reduction, identified and verified.

    Built exactly as a descent cell is (:func:`_descend_once`) — the parent
    cell's own points plus the orbit of one generic point under the operations
    kept, as one extra species — because only ``_group_of`` can *name* a
    subgroup and stamp the ``.ind`` order every table lookup keys on, and it
    answers about a cell.  What differs is what the cell **is**: not a group
    the Hamiltonian selected out of a larger one, but the declared crystal's
    own cell, so it carries no descent step and its provenance is ``realized``.

    Returns ``(group, cell, cell_typat, pairs, point, orbit, species)``, or the
    :class:`..refusal.Refusal` the identification itself made.

    Raises
    ------
    ..refusal.InternalError
        Where the operations kept do not close, or no generic point of
        :data:`_DECLARED_POINTS` gives a cell identified as them
        (:func:`_orbit_cell`).  The operations kept are a group by
        construction -- the largest subgroup inside the ones the declared
        content is carried by (:func:`..symmetry._largest_compatible_group`)
        -- and the orbit of a generic point under a subgroup has exactly it for
        a stabilizer, so anything else is arithmetic in this package failing
        rather than a measurement about the model: a cell reduction, like a
        descent, can only land on a smaller group, never fail.
    """
    kept_syms = [s for s in group.symmetries
                 if int(s.ind) in set(kept) and not s.time_reversal]
    leaks = _descent_closure(group, kept_syms)
    if leaks:                                    # pragma: no cover - defensive
        raise InternalError(
            f"the {len(kept_syms)} operations kept from those the declared orbital content "
            f"is carried by are not a group: {len(leaks)} of their products leave the set, "
            f"the first being #{leaks[0][0]} * #{leaks[0][1]} = #{leaks[0][2]}. They are the "
            "largest subgroup inside the compatible operations by construction, so this is "
            "a defect here and not a statement about the model.")
    species = _DECLARED_SPECIES
    while any(str(t) == species for t in base_typat):   # pragma: no cover - defensive
        species += "_"
    names = {int(s.ind): _record_name(s, group.lattice) for s in group.symmetries}

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What the identification of one reduced cell was read as, and why that is a defect."""
        return (
            f"the cell built to carry the {len(kept_syms)} operations the declared orbital "
            f"content leaves was identified as {got.name} #{got.number_str}, whose "
            f"{len(unitary)} unitary operations are not that set: "
            + (f"{len(unmatched_kept)} of the operations kept "
               + ", ".join(f"#{i} {names.get(int(i), '?')}" for i in unmatched_kept)
               + " were not found in it" if unmatched_kept else "")
            + (" and " if unmatched_kept and unmatched_found else "")
            + (f"{len(unmatched_found)} of its operations {unmatched_found} are not among "
               "them" if unmatched_found else "")
            + ". The operations kept are a subgroup (the largest inside the compatible "
            "ones) and the orbit of a generic point under a subgroup has exactly that "
            "subgroup for a stabilizer, so this is arithmetic in this package failing "
            f"rather than a measurement about the model. The orbit is "
            f"{np.asarray(orbit, dtype=float).tolist()} at symprec {float(symprec):.3g}.")

    made = _orbit_cell(base_cell, base_typat, group, kept_syms, points=_DECLARED_POINTS,
                       species=species, spinor=spinor, include_TR=include_TR,
                       symprec=symprec, diagnose=_diagnose)
    if isinstance(made, Refusal):
        return made
    got, cell, cell_typat, pairs, point, orbit_padded = made
    return got, cell, cell_typat, pairs, point, orbit_padded, species


def _declared_contents(model, group, assignment, validated, layout, *, spin, characters,
                       read_in=None):
    r"""``(decls, layout)`` — the declared content, and the multiplets it is read on.

    ``characters=`` is resolved in the site tables of the group the loop is
    currently on, into one :class:`.._sitechars.DeclaredContent` per declared
    content, and read on the multiplets the Hamiltonian measures: a
    declaration states what each site holds, and how the orbitals are
    partitioned inside a site is their basis
    (:func:`..symmetry._declared_blocks`).  The multiplets are the measured
    ones, whatever the declaration.

    ``characters`` may be the declaration as written or a mapping of contents
    already resolved on a larger group, which is what a descent hands back
    round: the constraint is then carried and restricted rather than resolved
    again, since a fresh resolution in a smaller site group refuses exactly
    where the restriction is merely reducible.

    ``read_in`` is step 1's own reading of the same words
    (:func:`_declared_cell_reduction`), so that a content resolved in a cell
    the declaration itself reduced still says where the word was read.

    A grammar error here is the same :class:`..refusal.Refusal`
    :func:`..symmetry._build_rep` would return for it.
    """
    from . import _sitechars
    from .symmetry import _declared_blocks, _normalize_declaration, _site_tables

    kept_u = {int(s.ind) for s in group.symmetries
              if int(s.ind) in set(validated.indices) and not s.time_reversal}
    kept_au = {int(s.ind) for s in group.symmetries
               if int(s.ind) in set(validated.indices) and s.time_reversal}
    try:
        tables = _site_tables(group, assignment, kept_u, kept_au,
                              range(len(assignment.site_orbitals)))
        decls = _normalize_declaration(characters, model, assignment, group, tables,
                                       layout, spin=spin, read_in=read_in)
    except _sitechars.DeclarationError as exc:
        return Refusal(exc.gate, exc.message, context=dict(exc.context, multiplets=layout))
    except Exception as exc:                     # noqa: BLE001 - a site group that has no table
        return Refusal(getattr(exc, "gate", "site_group"), str(exc),
                       context={"multiplets": layout})
    decls = _declared_blocks(decls, layout, assignment, tables, spin=spin)
    chan_of = layout._chan_of
    for d in decls:
        d["channels"] = tuple(sorted({int(chan_of[int(o)])
                                      for o in d.get("block", d["orbitals"])}))
    return decls, layout


def _declared_parts_of(decls) -> tuple:
    """The declared contents a round was read from, a block's in place of its one sum.

    What is carried to the next round and handed to the build: the
    declaration as it was written and resolved, so that every round reads it
    against its own measured multiplets (:func:`..symmetry._declared_blocks`).
    """
    return tuple(g["content"] for d in decls for g in d.get("groups", (d,)))


def _declared_contents_wider(model, group, assignment, layout, *, spin, characters,
                             read_in=None) -> tuple:
    r"""The declared words read in the site groups of the group the loop is on.

    The verdict stage reads a declaration in the site tables of the operations
    the detector kept (:func:`_declared_contents`), and a Hamiltonian that
    breaks most of the cell leaves those too small to name the word: ``A1`` of
    a ``3m`` site names nothing in the trivial site group the surviving
    operations leave.  What the word means is fixed where it was written — the
    site groups of the group the loop is on, or, on a layer, their in-plane
    part, since a free-standing layer's site groups carry ``σ_h`` and so have
    primed names only — and that reading, carried through the descent and
    restricted to the smaller site group, is the declaration there, as it is
    after any other descent.

    Returns one :class:`.._sitechars.DeclaredContent` per declared multiplet,
    or ``()`` where the words name nothing in either.
    """
    from . import _sitechars
    from .symmetry import _normalize_declaration, _site_tables

    readings = [list(group.symmetries)]
    padded = tuple(int(d) for d in getattr(group, "vacuum_dirs", ()) or ())
    if len(padded) == 1:
        readings.append([s for s in group.symmetries
                         if _side_preserving(s, padded[0])])
    for syms in readings:
        kept_u = {int(s.ind) for s in syms if not s.time_reversal}
        kept_au = {int(s.ind) for s in syms if s.time_reversal}
        try:
            tables = _site_tables(group, assignment, kept_u, kept_au,
                                  range(len(assignment.site_orbitals)))
            decls = _normalize_declaration(characters, model, assignment, group, tables,
                                           layout, spin=spin, read_in=read_in)
        except InternalError:
            raise
        except _sitechars.DeclarationError:
            continue
        except Exception:                        # noqa: BLE001 - a site group that has no table
            continue
        return tuple(d["content"] for d in decls)
    return ()


def _restricts_to(contents, group, assignment) -> bool:
    r"""Whether carried contents are **restrictions** to this group's site groups.

    A content read in a larger site group and carried down says what every
    operation of the smaller one does, which is what makes it the restriction
    there.  One read in a *smaller* group than the site group it lands in — the
    in-plane part of a free-standing layer's site group, read before a descent
    that never came — leaves the other operations unsaid, and is not a reading
    of the word in that site group at all.
    """
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    unitary = {int(s.ind) for s in group.symmetries if not s.time_reversal}
    for content in contents:
        site = int(site_of[int(content.orbitals[0])])
        here = {int(i) for i in assignment.site_group[site]} & unitary
        if not here <= {int(k) for k in content.constraints}:
            return False
    return True


def _rekey_contents(contents, pairs, refuted):
    """The declared contents on a re-identified group, through ``(old, new)`` pairs.

    Operations the smaller group no longer has drop out of every constraint,
    and that dropping **is** the restriction of the declared irrep to the
    smaller site group: nothing is re-resolved, so a restriction that has become
    reducible is carried where a fresh resolution would refuse it.
    """
    drop = {int(i) for i in (refuted or ())}
    keep = tuple((int(a), int(b)) for a, b in pairs if int(a) not in drop)
    return tuple(c.rekeyed(keep) for c in contents)


def _declaration_refuted(group, validated, *, refuted, verdict, tolerances=EXACT):
    r"""Which operations the **declared orbital content** breaks, and what is left.

    The fourth per-operation verdict, written from the one reading of the
    solution spaces (:func:`..symmetry._verdicts_and_choices`, made once per
    round for every measurement that needs it).  ``characters=`` is a statement
    about what the orbitals *are*, and the hoppings cannot contradict it — they
    can only be the hoppings of a crystal with less symmetry.

    What is returned is the largest subgroup on which one representation per
    orbit of the sectors of the Hamiltonian's conserved matrices under **that
    subgroup**, of those the hoppings leave open, gives every declared site
    what was declared -- the same content where the whole site was declared, a
    content holding it where part was (:func:`..symmetry._covariant_subgroup`,
    :class:`.._sectors.SectorSearch`) -- the first by the tie rule where
    several of that order exist (:func:`_tie_ranked`); and the identity alone
    always does, which is what makes a descent by declaration land somewhere
    rather than refuse.  A coupled part of the crystal is one orbit of one
    sector, carried by every one-dimensional character, and uncoupled orbitals
    a symmetry exchanges are an orbit of several, which a smaller group splits.
    Where the sectors read nothing, the coupled parts' requirements decide: an
    operation whose declared multiplets of one part need different scalars,
    or whose solution space holds no nonsingular member in the block form of
    the multiplets a ``multiplets=`` states, is *individually* disallowed, and
    the subgroup is the largest on which every part has a one-dimensional
    character of it meeting all of its requirements.  Every declaration is such a
    constraint, wherever it sits, and an undeclared orbital, the reference
    orbital included, imposes none: declaring one sublattice of an orbit is
    the same statement as declaring the other, where the hoppings carry the
    two alike.  Every operation outside the subgroup is refuted the same way,
    and so is the primed partner of each: a site character is a character of
    the *unitary* site group, and the antiunitary copy of a broken operation is
    broken with it.

    Returns ``{"refuted", "subgroup", "candidates", "rows"}``, or the
    ``rep_search_too_large`` :class:`..refusal.Refusal` of a search past its
    ceiling.
    """
    from .symmetry import _covariant_subgroup, _joint_reason, _returned_rows

    syms = group.symmetries
    standing = set(validated.indices) - set(refuted)
    kept_u = [int(s.ind) for s in syms
              if int(s.ind) in standing and not s.time_reversal]
    got = verdict
    unitary_only = not any(s.time_reversal and int(s.ind) in standing for s in syms)
    sub = _covariant_subgroup(group, kept_u, got["requirements"],
                              unitary_only=unitary_only,
                              tol=tolerances.noise(CHARACTER),
                              joint=got.get("joint", ()),
                              search=got.get("search"))
    if isinstance(sub, Refusal):
        return sub
    lattice = np.asarray(group.lattice, dtype=float)
    # the primed partner of a broken unitary operation, through the group's own
    # product table: T * g, with T the pure time reversal
    prod = group.get_product_table(get_diff=False)
    at = {int(s.ind): i for i, s in enumerate(syms)}
    partner = {}
    t_pos = next((p for p, s in enumerate(syms) if s.time_reversal
                  and np.array_equal(np.rint(np.asarray(s.rotation)).astype(int),
                                     np.eye(3, dtype=int))
                  and np.allclose(np.asarray(s.translation) % 1.0, 0.0, atol=1e-8)), None)
    if t_pos is not None:
        for u in kept_u:
            q = int(syms[int(prod[t_pos, at[int(u)]])].ind)
            if q in standing:
                partner[int(u)] = q
    broken = {}
    # the contents read together, and those the sectors alone weigh, each named
    # where it is read
    read_together = tuple(got.get("joint", ())) + tuple(got.get("by_sectors", ()))
    for ind in sub["broken"]:
        rows = tuple(r for r in got["rows"] if int(r["ind"]) == int(ind))
        culprit = next((r for r in rows if r["required"] is not None
                        and int(ind) in got["reasons"]), rows[-1] if rows else None)
        note = got["reasons"].get(int(ind)) or _joint_reason(read_together, ind)
        if note is None:
            note = ("the declared content is carried by this operation on its own, but "
                    "no group of operations containing it and those kept carries every "
                    "declared multiplet in its declared irrep at once")
        together = next((j for j in read_together
                         if any(int(ind) in row for row in j["theta"].values())), None)
        if culprit is None and together is not None:
            # a content read over several multiplets: the declared value there,
            # and what its multiplets carry, each against its own part's scale
            culprit = {"declared": together["decl"]["constraints"].get(int(ind), 0.0),
                       "measured": sum(v for row in together["theta"].values()
                                       for g, v in row.items() if int(g) == int(ind))}
        orbitals = ({int(o) for r in rows for o in r["orbitals"]} if rows else
                    {int(o) for o in (together["decl"]["orbitals"] if together else ())})
        rec = {"ind": int(ind), "refuted_by": "declaration", "note": str(note),
               "orbitals": tuple(sorted(orbitals)),
               "declared": (None if culprit is None else complex(culprit["declared"])),
               "measured": (None if culprit is None else complex(culprit["measured"])),
               "rows": _returned_rows(rows)}
        broken[int(ind)] = rec
        if int(ind) in partner:
            mate = dict(rec, ind=int(partner[int(ind)]))
            mate["note"] = (
                f"the primed copy of #{int(ind)} "
                f"{_record_name(syms[at[int(ind)]], lattice)}, which the "
                "declared orbital content breaks: a site character is a character of the "
                "unitary site group, so the two stand or fall together")
            broken[int(partner[int(ind)])] = mate
    # the returned subgroup is the GREY one the survivors generate: a site
    # character decides about the unitary half, and the primed copy of an
    # operation it keeps is kept with it
    grey = set(sub["subgroup"]) | {partner[i] for i in sub["subgroup"]
                                   if i in partner}
    return {"refuted": broken, "subgroup": tuple(sorted(grey)),
            "candidates": sub["candidates"],
            "rows": _returned_rows(got["rows"])}


def _operation_verdicts(model, group, assignment, validated, *, tolerances, nk, params,
                        context, spin, layout, decls=None,
                        reference_orbital=0, max_candidates=2 ** 20, free_standing=None,
                        centres=None, sectors=None, own_form=None):
    r"""One verdict per operation of the group, each with the reason behind it.

    Each verdict judges noise against the call's own tolerances
    (:class:`.._tolerance.Tolerances`).  The detector has already spoken for most of them:
    :attr:`ValidatedGroup.discrepancies` carries ``"no symmetry matrix"``,
    ``"covariance"`` and ``"out of plane"`` per operation, antiunitary coset
    included.  So have the orbital **centres**, where a precision is declared:
    ``centres`` is their verdict (:func:`_centres_carried`), the operations they
    are not carried by within the declared precision, ``"centres"``, which
    stands for each of them whatever else reads it and which every verdict below
    reads as already given.  Four measurements finish the verdict where those
    cannot decide.

    The **own form** of every operation the detector kept, read once where the
    detector spoke (:func:`_own_form_read`) and handed in as ``own_form``, the
    records of the operations it refuted: a matrix with the operation's action
    on ``k`` represents it only where it carries each site's states onto its
    image's, with the operation's own spin lift on states that carry a spin
    (``"tensor structure"``, the record's ``own_form`` naming the condition
    that failed).  Where it refutes a unitary operation the detector found
    obeyed, the group is lowered at once: the multiplets, the declared content
    and everything read on them belong to the group that holds, and are read
    there on the next round.

    The **tensor structure** of the operations the detector leaves undetermined:
    a surviving k-action class of more than one member proves only that *some*
    operator with that k-action is a symmetry, and each member's own form, and
    on the multiplets :func:`_ansatz_refuted`, say which member it is
    (``"tensor structure"``) -- its own spin lift telling a layer's ``σ_h``
    pair apart where the multiplets are wider than one orbital and cannot.

    The **declared orbital content**, where there is one, read as what each
    site holds on the multiplets the Hamiltonian measures: every operation
    outside the largest subgroup the content is covariant on is broken by the
    declaration (:func:`_declaration_refuted`), each subgroup asked on its own
    orbits of the sectors of the Hamiltonian's conserved matrices -- ``sectors``,
    read once where the model entered (:func:`.._sectors.read_sectors`).

    The **pure translations** of a cell whose point set repeats under one, each
    read at the sites, as written or with each site's orbitals carried onto its
    translate's by a change of basis within the site (:func:`_lattice_verdict`).
    Their refutations join the rest under ``"translation"``; a translation the
    Hamiltonian *obeys* means the model is a supercell, which the record carries
    under ``realized`` for the step that owns that verdict.

    And what the same reading **pins**, by the same function: the scalar each
    orbit of multiplets carries relative to the anchor where the hoppings fix
    it, the ratios inside a part they leave free of it, and the coupled parts
    themselves (:func:`..symmetry._verdicts_and_choices`).  What the hoppings
    leave open is no measurement, and nothing here chooses it: the labelling
    is chosen once, by the lowest-angular-momentum rule inside
    :func:`..symmetry._build_rep`, among the characters the declarations leave.
    The reading is carried out of here so that the build assembles what was
    measured rather than measuring it a second time.  ``reference_orbital``
    is the one that build will be given: the orbital the lowest-angular-momentum
    rule reads the crystal from first, so that the descent and the build read
    one crystal.

    And last, once every other verdict leaves each unitary operation standing,
    the **projective class** of the states on a spinless reading: where the
    matrices the operations are on these states carry a class the single-valued
    tables do not -- two commuting operations whose matrices anticommute, the
    spin-½ sign of spinors stored in the orbital index -- the operations outside
    the largest subgroup that carries an ordinary representation are refuted
    (:func:`_projective_class_refuted`), and a tie between subgroups of that
    order is settled by the tie rule (:func:`_tie_ranked`), never by their
    intersection.  On a flat layer the class is read once, on the free-standing
    layer, every operation composed with the horizontal mirror given the
    mirror's chosen sign times its partner's matrix; a layer on a substrate
    carries the part of that reading that preserves the layer's normal
    (``free_standing``, the
    free-standing cell of its point set, :func:`_projective_class_carried`)
    rather than reading the class itself.

    Returns ``(refuted, lattice, declaration, choices, projective)`` -- the
    operations refuted past the detector, keyed by ``.ind``, the lattice record
    (empty where the cell carries no fractional translation), the declaration's
    own record (``None`` where nothing was declared), the one reading of the
    solution spaces and the projective class's own record (``None`` where
    nothing was refuted by it) -- or the :class:`..refusal.Refusal` the tensor
    test's own multiplet permutation, the declared reading or the block
    enumeration made.
    """
    from .symmetry import _multiplet_blocks, _verdicts_and_choices
    syms = group.symmetries
    centred = dict(centres or {})
    own_form = dict(own_form or {})
    # THE OWN FORM of every operation the detector found obeyed, first: a
    # unitary operation none of whose matrices carries each site onto its
    # image with its own spin lift is not an operation of this Hamiltonian,
    # and the multiplets the rest of this round reads were measured with it.
    # So the group is lowered on that verdict at once, with the pure
    # translations measured as for any descent, and the next round reads the
    # rest on the group that holds.
    obeyed = set(validated.obeyed)
    lifted = {i: r for i, r in own_form.items() if i in obeyed}
    if any(int(s.ind) in lifted and not s.time_reversal for s in syms):
        lattice = _lattice_verdict(model, group, assignment, validated, lifted,
                                   tolerances.energy,
                                   _detection_grid(int(group.dim_k), int(nk)), params,
                                   context=context, spin=spin,
                                   matrix_tol=tolerances.matrix or None)
        refuted = dict(lifted)
        if lattice and lattice["refuted"]:
            refuted = {**refuted, **lattice["refuted"]}
        return {**refuted, **centred}, lattice, None, None, None
    # THE MULTIPLET BLOCKS, once per round.  The tensor test, the declared
    # reading and the representation built on the group that settles all read
    # them, and they are one object; building them per reader is the one cost a
    # wannier90 Hamiltonian cannot carry.  They are read on the operations that
    # hold their own form: one refuted by it has no matrix to read a block off
    blocks = _multiplet_blocks(
        model, group, assignment, spin, validated, layout,
        [p for p, s in enumerate(group.symmetries)
         if int(s.ind) in set(validated.indices) and int(s.ind) not in own_form],
        ref_chan=int(np.asarray(layout._chan_of, dtype=int)[0]),
        max_candidates=max_candidates, sectors=sectors)
    if isinstance(blocks, Refusal):
        return blocks
    # THE UNDETERMINED OPERATIONS, each told apart from its partners: by its own
    # form, read once above, and on the multiplets by its tensor structure
    shared = {i: r for i, r in own_form.items() if i not in obeyed}
    got = _ansatz_refuted(model, group, assignment, validated,
                          layout=layout, spin=spin, blocks=blocks,
                          max_candidates=max_candidates, skip=set(shared))
    if isinstance(got, Refusal):
        return got
    # the operations their own form or their tensor test refuted, and any
    # antiunitary one found obeyed that its own form refuted: those start no
    # descent, as no antiunitary refutation does; and the centres' verdict,
    # which every reading below takes as given
    refuted = {**lifted, **got[1], **shared, **centred}
    # THE ONE READING of the solution spaces: the verdict on the declared
    # content and the choice the representation is built from, together, over
    # the operations that are still standing.
    standing = set(validated.indices) - set(refuted)
    positions = [p for p, sym in enumerate(syms)
                 if int(sym.ind) in standing and not sym.time_reversal]
    # the time reversal that anchors the phases a crystal of multiplets is read
    # in, chosen as the build chooses it: the pure one where it stands
    reversing = [p for p, sym in enumerate(syms)
                 if int(sym.ind) in standing and sym.time_reversal]
    pure = [p for p in reversing
            if np.array_equal(np.rint(np.asarray(syms[p].rotation)).astype(int),
                              np.eye(3, dtype=int))
            and np.allclose(np.asarray(syms[p].translation) % 1.0, 0.0, atol=1e-8)]
    choices = _verdicts_and_choices(model, group, assignment, spin, validated,
                                    layout, decls or (), positions, blocks=blocks,
                                    ref_orb=int(reference_orbital),
                                    space_tol=tolerances.noise(IN_SPACE),
                                    tol=tolerances.noise(CHARACTER),
                                    max_candidates=max_candidates,
                                    reversal=(pure or reversing or [None])[0],
                                    sectors=sectors)
    if isinstance(choices, Refusal):
        return choices
    declaration = None
    if decls:
        declaration = _declaration_refuted(group, validated,
                                           refuted=refuted, verdict=choices,
                                           tolerances=tolerances)
        if isinstance(declaration, Refusal):
            return declaration
        # integer operation indices as keys: a mapping merge, not keyword
        # arguments to a call
        refuted = {**refuted, **declaration["refuted"]}
    # THE LATTICE, measured on the same group and before anything is concluded
    # from the rest.  A cell whose point set repeats under a pure translation was
    # returned as the lattice supergroup, and only the Hamiltonian can say what
    # that means: obeyed, the model IS a supercell; broken, the model's own
    # lattice is its translation group and the first descent is the one to it.
    lattice = _lattice_verdict(model, group, assignment, validated, refuted,
                               tolerances.energy,
                               _detection_grid(int(group.dim_k), int(nk)), params,
                               context=context, spin=spin,
                               matrix_tol=tolerances.matrix or None)
    if lattice and lattice["refuted"]:
        refuted = {**refuted, **lattice["refuted"]}
    # THE STATES' PROJECTIVE CLASS, last, and read on a group: once every other
    # verdict leaves each unitary operation of it standing.  A spinless reading
    # whose states carry a class the single-valued tables do not -- spinors in a
    # spinless model's orbital index -- has operations no single-valued
    # representation carries, and they leave the group here exactly as a broken
    # one does, for the descent to lower.  A layer reads it once, free-standing:
    # there every operation composed with the horizontal mirror has the
    # mirror's chosen sign times its partner's matrix, and a substrate carries
    # the part of that reading that preserves the layer's normal rather than
    # making its own.
    projective = None
    unitary = {int(s.ind) for s in syms if not s.time_reversal}
    if not unitary & (set(validated.broken) | set(refuted)):
        if free_standing is not None:
            more, projective = _projective_class_carried(model, group, validated,
                                                         free_standing=free_standing)
        else:
            more, projective = _projective_class_refuted(group, validated,
                                                         assignment=assignment,
                                                         decls=decls or ())
        refuted = {**refuted, **more}
    return {**refuted, **centred}, lattice, declaration, choices, projective


def _descend_once(base_cell, base_typat, group, validated, *, spinor, include_TR,
                  symprec, step, refuted=None, lattice_record=None,
                  declaration=None, projective=None, centres=None) -> tuple | Refusal:
    r"""Lower one cell to the unitary operations the Hamiltonian was measured to obey.

    The kept set ``K`` is known before this runs — it is
    :attr:`ValidatedGroup.indices` minus the antiunitary operations — so nothing
    here searches for a group.  What the cell is *for* is the identification
    step: only ``_group_of`` (spglib + ``irrep`` on a point set) can **name** the
    subgroup, supply the setting transformation ``refUC``/``shiftUC`` and stamp
    the ``.ind`` order every table lookup downstream keys on, and it answers
    about a cell.  So a cell carrying exactly ``K`` is built — the parent cell's
    own points plus the orbit of one generic point under ``K``, as one extra
    species — and handed back.  **The model is never touched.**

    ``refuted`` is the tensor-structure verdict of :func:`_own_form_read` — the
    operations none of whose matrices carries each site's states onto its
    image's with the operation's own spin lift — and of :func:`_ansatz_refuted`
    — the operations the detector left undetermined (``"undetermined"``) whose
    tensor structure on the multiplets refutes them — together with the
    operations the **declared orbital content** breaks
    (:func:`_declaration_refuted`, ``refuted_by = "declaration"``).  They leave ``K`` exactly as a detector discrepancy does
    and are recorded beside them, ``refuted_by`` saying which measurement made
    each verdict.

    ``declaration`` is that verdict's own record, carried onto the step so the
    returned group can say which readings of the declared content were possible
    where several tie (``tied``, in the order the tie rule ranks them, the one
    returned first).

    ``centres`` is the orbital centres' verdict under a declared precision
    (:func:`_centres_carried`), whose refuted operations are among ``refuted``
    under ``refuted_by = "centres"`` and stand for them over the detector's
    reading: carried onto the step as ``centres``, the move the parent group
    asked of the centres, the operations that carry them within the
    precision and a note.

    ``projective`` is the record of the states' **projective class**
    (:func:`_projective_class_refuted`), whose refuted operations are among
    ``refuted`` under ``refuted_by = "projective class"``: carried onto the step
    as ``projective``, the pairs of commuting operations whose matrices do not
    commute and, where subgroups of the kept order tied, all of them in the
    order the rule ranked them; on a layer, the sign the horizontal mirror was
    given free-standing, or on a substrate the free-standing group the class
    was read on (:func:`_projective_class_carried`).

    ``lattice_record`` is :func:`_lattice_verdict`'s record when the parent is a
    **lattice supergroup** — a cell that is a supercell of the crystal's own
    primitive cell whose Hamiltonian was measured to break the translation.  The
    step then *is* the lattice descent, and the identification is asserted to
    come back **primitive** (``_lattice_index == 1``): the generic orbit under a
    group whose translations are the model's own lattice has no finer
    translation, so anything else is a defect here rather than a measurement.

    A kept set that does **not close** is resolved rather than refused
    (:func:`_resolve_closure`): the largest subgroup inside it, or where several
    tie the first by the tie rule (:func:`_tie_ranked`), down to the identity
    alone — every Hamiltonian obeys some space group and the question which one
    is never refused.  The operations that leaves out are recorded beside the
    refuted ones, ``refuted_by = "closure"`` for one no candidate subgroup
    carries — an operation sitting the wrong side of the declared tolerance —
    and ``"tie"``
    for one only the candidates the rule did not keep carry; the step's ``tie``
    record lists the candidates in the rule's order and says why the first was
    kept.

    The cell is identified and verified by :func:`_orbit_cell`: an
    identification that does not reproduce ``K``, or is still not primitive, is
    an accident of the generic point, and the orbit is rebuilt from the next one
    of :data:`_DESCENT_POINTS`.

    Returns ``(group, cell, cell_typat, record)``, or the
    :class:`..refusal.Refusal` the identification itself made.

    Raises
    ------
    ..refusal.InternalError
        When no generic point of the sequence gives a cell identified as ``K``:
        the package's identification disagreeing with its own construction.
    """
    refuted = dict(refuted or {})
    antiunitary = {int(s.ind) for s in group.symmetries if s.time_reversal}
    keep = set(validated.indices) - antiunitary - set(refuted)
    kept = [s for s in group.symmetries
            if int(s.ind) in keep and not s.time_reversal]
    lattice = np.asarray(group.lattice, dtype=float)
    by_ind = {int(s.ind): s for s in group.symmetries}

    def _name(ind):
        """The operation as the descent's record lists it (:func:`_record_name`)."""
        return _record_name(by_ind[int(ind)], lattice)

    # THE SURVIVORS RESOLVED TO A GROUP.  An operation the declared tolerance
    # puts on the wrong side of the gate leaves the kept set here rather than
    # being returned or refused; where several candidate subgroups tie, the one
    # kept is the first by the tie rule (`_resolve_closure`), and the step says
    # which and why.
    left_out: tuple = ()
    tie = None
    if _descent_closure(group, kept):
        resolved, dropped, tied = _resolve_closure(group, [int(s.ind) for s in kept])

        def _set(inds):
            return "{" + ", ".join(_name(i) for i in inds) + "}"

        if tied:
            left = [_name(i) for i, why in dropped if why == "tie"]
            tie = {"tied": tied, "kept": tied[0], "note": (
                f"the operations measured to hold in #{group.number_str} do not close "
                "into a group, and the largest subgroups inside them tie: "
                + " and ".join(_set(c) for c in tied)
                + ". The one kept is the first by the tie rule `realized_group` states -- "
                f"{_tie_rule_words(group)} -- since {_tie_reason(group, tied)}; "
                + " and ".join(left)
                + (" is" if len(left) == 1 else " are")
                + " left out, under `refuted_by = \"tie\"`.")}
        left_out = tuple(dict({"ind": int(i), "name": _name(i),
                               "refuted_by": str(why), "residual": float("nan"),
                               "step": int(step), "of": str(group.number_str)},
                              **({"note": (
                                  "the largest subgroups the operations measured to hold "
                                  f"contain tie, and the one the tie rule keeps, "
                                  f"{_set(tied[0])}, does not hold it")}
                                 if why == "tie" else {}))
                         for i, why in dropped)
        keep = {i for i in keep if i in resolved}
        kept = [s for s in kept if int(s.ind) in resolved]

    def _entry(i):
        """One broken operation, in the fields the measurement that broke it has.

        A verdict about the declared content carries no residual: nothing was
        near a gate, the hoppings are not those of the crystal that was
        declared, and what the record has to say is which orbitals, what they
        were declared to be and what they measure.
        """
        rec = refuted[i]
        why = str(rec.get("refuted_by", "tensor structure"))
        base = {"ind": int(i), "name": _name(i), "refuted_by": why,
                "step": int(step), "of": str(group.number_str)}
        if why == "centres":
            # a verdict about where the centres are carries no residual of H:
            # what it rests on is how far the operation alone would move one
            return dict(base, residual=float("nan"), miss=float(rec["miss"]),
                        precision=float(rec["precision"]), note=str(rec["note"]))
        if why == "declaration":
            return dict(base, residual=float("nan"),
                        orbitals=tuple(int(o) for o in rec.get("orbitals", ())),
                        declared=rec.get("declared"), measured=rec.get("measured"),
                        note=str(rec.get("note", "")))
        if why == "projective class":
            # a verdict about the states' class carries no residual either: the
            # sign it rests on sits a whole unit from the table's
            return dict(base, residual=float("nan"), pair=rec.get("pair"),
                        names=rec.get("names"),
                        commutation_phase=rec.get("commutation_phase"),
                        spin_half=rec.get("spin_half"), note=str(rec.get("note", "")))
        return dict(base, residual=float(rec["scaled"]),
                    # how far its own form carries H, where that was measured
                    **({} if rec.get("covariance") is None else
                       {"covariance": float(rec["covariance"])}))

    # the centres' verdict stands for the operations it refutes, whatever the
    # detector read on them: each broken operation is listed once, with its reason
    centred = {int(i) for i, r in refuted.items()
               if isinstance(r, dict) and r.get("refuted_by") == "centres"}
    broken = tuple(
        {"ind": int(d["ind"]), "name": _name(d["ind"]),
         "refuted_by": str(d["refuted_by"]),
         "residual": float(d["residual"]),
         "step": int(step), "of": str(group.number_str)}
        for d in validated.discrepancies if int(d["ind"]) not in centred) + tuple(
        _entry(i) for i in sorted(refuted)) + left_out

    species = _DESCENT_SPECIES
    while any(str(t) == species for t in base_typat):   # pragma: no cover - defensive
        species += "_"

    def _diagnose(got, unitary, unmatched_kept, unmatched_found, orbit):
        """What the identification of one descent cell was read as, and why that is a defect."""
        where = np.round(np.asarray(orbit, dtype=float), 6).tolist()
        if int(got._lattice_index) > 1:
            return (
                f"the cell built to carry the {len(kept)} operations the Hamiltonian obeys "
                f"(from {group.name} #{group.number_str}) was identified as {got.name} "
                f"#{got.number_str} and is still not primitive: it carries the pure "
                "translation "
                f"{np.round(np.asarray(got._fractional_translations[0], dtype=float), 6).tolist()} "
                f"(index {int(got._lattice_index)}). The generic orbit added to it is a free "
                "orbit under a group whose translations are the model's own lattice, and "
                "such an orbit cannot carry a finer translation, so this is arithmetic in "
                "this package failing rather than a measurement about the model. The orbit "
                f"is {where} at symprec {float(symprec):.3g}.")
        return (
            f"the cell built to carry the {len(kept)} operations the Hamiltonian obeys (from "
            f"{group.name} #{group.number_str}) was identified as {got.name} "
            f"#{got.number_str}, whose {len(unitary)} unitary operations are not that set: "
            + (f"{len(unmatched_kept)} kept operations {unmatched_kept} were not found in it"
               if unmatched_kept else "")
            + (" and " if unmatched_kept and unmatched_found else "")
            + (f"{len(unmatched_found)} of its operations {unmatched_found} are not among "
               "the kept ones" if unmatched_found else "")
            + ". Returning this group would label the bands with a group that is either "
            "larger than the one measured or in a different position, so the descent "
            f"stops here. The generic orbit added to the cell is {where}; a symprec "
            f"({float(symprec):.3g}) too tight for it, or a cell whose own points are "
            "closer together than symprec, is what makes the identification miss.")

    # The cell handed in already carries the environment that was asked for --
    # its shadow species, where there is one, are part of the point set the
    # descent is lowering -- so nothing is put under it here: the orbit, and
    # exactly the orbit, is what this cell adds.
    made = _orbit_cell(base_cell, base_typat, group, kept, points=_DESCENT_POINTS,
                       species=species, spinor=spinor, include_TR=include_TR,
                       symprec=symprec, diagnose=_diagnose)
    if isinstance(made, Refusal):
        return made
    got, cell, cell_typat, pairs, point, orbit_padded = made

    record = {
        "step": int(step),
        "from": {"number_str": str(group.number_str), "name": str(group.name),
                 "size": int(group.size)},
        "to": {"number_str": str(got.number_str), "name": str(got.name),
               "size": int(got.size)},
        "dropped": broken, "n_dropped": len(broken),
        # how each verdict was reached: the detector's own reasons, "tensor
        # structure" for the operations it kept and the tensor structure
        # refuted -- an undetermined member of a class, or an operation whose
        # own spin lift its space does not carry -- and the rest in the words
        # above
        "refuted_by": {r: sum(1 for b in broken if b["refuted_by"] == r)
                       for r in sorted({b["refuted_by"] for b in broken})},
        "kept": tuple(sorted(int(s.ind) for s in kept)),
        "pairs": pairs,
        "cell": {"generic_point": tuple(float(x) for x in point),
                 "orbit": tuple(tuple(float(x) for x in row) for row in orbit_padded),
                 "species": species},
    }
    if tie is not None:
        record["tie"] = tie
    if declaration is not None:
        readings = tuple(declaration.get("candidates", ()))
        record["declaration"] = {
            "subgroup": tuple(declaration.get("subgroup", ())),
            "tied": readings,
            "rows": tuple(declaration.get("rows", ())),
        }
        if readings:
            # the readings, one by one in the rule's order, by their operations'
            # names, which is how the environments' comparison names them
            record["declaration"]["names"] = tuple(
                tuple(_name(i) for i in r) for r in readings)
            record["declaration"]["note"] = (
                "several readings of the declared content are maximal and none contains "
                "the others: "
                + " and ".join("{" + ", ".join(_name(i) for i in r) + "}" for r in readings)
                + ". The one returned is the first by the tie rule `realized_group` states "
                f"-- {_tie_rule_words(group)} -- since {_tie_reason(group, readings)}; "
                "`descent['tied']` lists them in that order.")
    if projective is not None:
        record["projective"] = dict(projective)
    if centres is not None:
        record["centres"] = {"precision": float(centres["precision"]),
                             "move": float(centres["move"]),
                             "kept": tuple(int(i) for i in centres["kept"]),
                             "tied": tuple(tuple(int(i) for i in c)
                                           for c in centres.get("tied", ())),
                             "note": str(centres["note"])}
    if lattice_record:
        record["lattice"] = dict(lattice_record)
        record["lattice"]["note"] = _lattice_note(lattice_record, got)
    return got, cell, cell_typat, record


def _joined(parts) -> str:
    """Clauses joined as a sentence lists them: ``a``, ``a and b``, ``a, b and c``."""
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


#: Each cause :func:`_descent_counts` counts, as the record's one-line summary says it.
_DROPPED_BY_WORDS = {"centres": "by the centres", "hoppings": "by the hoppings",
                     "declaration": "by the declared content",
                     "closure": "left out to close a group", "tie": "by the tie rule",
                     "time reversal": "with time reversal",
                     "projective class": "by the states' projective class"}


def _descent_counts(steps, time_reversal) -> dict:
    r"""What a descent dropped, every step together, by cause.

    ``{cause: count}``, in the order the measurement meets the causes: the
    orbital centres beyond a declared precision; the hoppings; the declared
    orbital content; ``"closure"``, operations the Hamiltonian obeys that are
    left out so that the operations kept form a group; ``"tie"``, operations
    only the subgroups the tie rule did not keep hold; ``"time reversal"``, the
    operations combined with it where it was measured broken between two steps;
    and the states' projective class.  Each step drops operations of the cell
    no earlier step dropped, so the counts add up to the cell's group less the
    group the descent arrives at, a reduction of the cell by the declared
    content between two steps aside.  The descent's ``dropped_by`` holds the
    ones that are not zero, which its note and the record's one-line summary
    read.
    """
    dropped = [str(b["refuted_by"]) for st in steps for b in st["dropped"]]
    named = {r: dropped.count(r)
             for r in ("centres", "declaration", "closure", "tie", "projective class")}
    tr = time_reversal or {}
    between = sum(int(steps[k]["to"]["size"]) - int(steps[k + 1]["from"]["size"])
                  for k in range(len(steps) - 1)
                  if tr.get("outcome") == "measured broken"
                  and steps[k]["to"]["number_str"] == tr.get("grey_number_str"))
    return {"centres": named["centres"], "hoppings": len(dropped) - sum(named.values()),
            "declaration": named["declaration"], "closure": named["closure"],
            "tie": named["tie"], "time reversal": int(between),
            "projective class": named["projective class"]}


def _descent_clauses(counts) -> tuple:
    r"""``(split, lowered)``: a descent's counts by cause, and what its operations kept carry, in words.

    ``counts`` is :func:`_descent_counts`'.  ``split`` says what dropped the
    cell's operations: in the words the note has always had where only the
    hoppings and the declared content did, and otherwise each cause once, in
    the order the measurement meets them.  An operation the states' projective
    class leaves out is one the Hamiltonian obeys, and so is one left out so
    that the operations kept form a group, or by the tie rule.  ``lowered``
    names the group the cell was lowered to where the centres, the closure, the
    tie rule or the states' class dropped some, the largest group of its
    operations that carries what each of them asks, and is ``None`` where the
    hoppings and the declared content alone did.
    """
    d_cen, d_hop, d_decl = counts["centres"], counts["hoppings"], counts["declaration"]
    d_closure, d_tie = counts["closure"], counts["tie"]
    d_tr, d_proj = counts["time reversal"], counts["projective class"]
    # WHO BROKE WHAT.  The hoppings and the declared orbital content are two
    # different statements about the crystal, and a record that added them up
    # would hide which of the two is the one to act on.  So is the states'
    # projective class, which no hopping breaks: the spinless reading of spinor
    # states has no single-valued representation there.  And so are the
    # centres, beyond the declared precision: the geometry, not the hoppings.
    split = (
        f"the Hamiltonian broke {d_hop} of them by its hoppings and the declared "
        f"orbital content broke {d_decl} more" if d_hop and d_decl else
        f"the declared orbital content broke {d_decl} of them" if d_decl else
        f"the Hamiltonian broke {d_hop} of them")
    if d_cen or d_closure or d_tie or d_tr or d_proj:
        said = []
        if d_cen:
            said.append(f"the orbital centres are not carried by {d_cen} of them within "
                        "the declared precision")
        if d_hop:
            said.append(f"the Hamiltonian broke {d_hop} {'more' if said else 'of them'} "
                        "by its hoppings")
        if d_decl:
            said.append(f"the declared orbital content broke {d_decl} "
                        f"{'more' if said else 'of them'}")
        if d_closure:
            said.append(f"{d_closure} {'more' if said else 'of them'} "
                        f"{'is' if d_closure == 1 else 'are'} left out so that the operations "
                        "kept form a group")
        if d_tie:
            said.append(f"{d_tie} {'more' if said else 'of them'} "
                        f"belong{'s' if d_tie == 1 else ''} only to the subgroups the tie "
                        "rule did not keep")
        if d_tr:
            said.append("time reversal is measured broken, which leaves out the "
                        f"{d_tr} operations combined with it")
        if d_proj:
            said.append(f"{d_proj} {'more' if said else 'of them'}, which the Hamiltonian "
                        "obeys, carry no single-valued representation on these states")
        split = ", and ".join(said)
    lowered = None
    if d_cen or d_closure or d_tie or d_proj:
        kept_as = []
        if d_cen:
            kept_as.append("that carry the orbital centres within the declared precision")
        if d_hop or d_closure or d_tie:
            kept_as.append("that the Hamiltonian obeys")
        if d_decl:
            kept_as.append("that carry the declared orbital content")
        if d_proj:
            kept_as.append("on which these states carry a single-valued representation")
        lowered = (f"the largest group of its {'unitary ' if d_tr else ''}operations "
                   + _joined(kept_as) + (", the first by the tie rule" if d_tie else ""))
    return split, lowered


def _lattice_note(lattice_record, got=None) -> str:
    """The lattice descent in words, and the group it arrived at where ``got`` names it.

    ``got`` is left out where that group is not what holds: a grey group on
    which time reversal is then measured broken.
    """
    tau = np.round(np.asarray(lattice_record["fractional_translations"][0],
                              dtype=float), 6)
    return (
        f"the cell allowed a lattice supergroup of index "
        f"{int(lattice_record['index'])} -- its point set repeats under the pure "
        f"translation {tau.tolist()} (reduced) -- and the Hamiltonian broke that "
        f"translation, together with "
        f"{int(lattice_record['coset_partners_refuted'])} coset partners of it. So "
        "this cell is "
        "not a supercell after all: the model's own lattice is its translation group"
        + (f", and the group returned is {got.name} #{got.number_str} on it."
           if got is not None else "."))


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


#: How ``max|H|`` is measured: the largest entry of ``H(k)`` over this many
#: seeded random k, floored by the model's own largest hopping or on-site
#: magnitude (which it bounds, ``|t_ij(R)| ≤ max_k |H_ij(k)|``, so the floor only
#: ever catches a k-sample that undershot) -- all of it with the mean on-site
#: energy taken off the diagonal.
_MAX_H_NK, _MAX_H_SEED = 64, 3


def _max_hamiltonian(model, *, params=None) -> float:
    r"""``max|H|`` — the Hamiltonian's own magnitude, measured on the model.

    The declared symmetry tolerance is **absolute**, in the units of ``H``, so a
    *relative* precision only becomes one after being read against this number
    (``docs/guide/numerical.md#how-to-read-it``).  It is measured two ways and
    the larger is taken:
    the largest entry of ``H(k)`` on a fixed set of random k — the thing the
    covariance residual is a residual *of* — and the largest hopping or
    on-site magnitude in the model's own numbers.
    The second is a floor rather than a competitor: each hopping is a Fourier
    coefficient of ``H_ij(k)``, so ``|t_ij(R)| ≤ max_k |H_ij(k)|`` and the
    read-off can only win where the finite k-sample undershot.

    **Both with the energy zero taken off.**  A constant shift ``H → H + c·1``
    moves no covariance residual, ``U(H + c)U† − (H(gk) + c) = U H U† − H(gk)``,
    and no centre defect's either, since a centre carries no phase onto its own
    on-site term; it moves every entry on the diagonal.  So the magnitude is
    that of ``H − μ·1``, ``μ`` the mean on-site energy per state
    (:func:`.._bloch.onsite_mean`): subtracting a Fermi energy from a wannier90
    import moves no tolerance, and a model whose on-site energies average to
    zero reads the largest entry of ``H`` itself.
    """
    mu = _bloch.onsite_mean(model, params)
    site = _bloch.onsite(model, params)
    got = (float(np.abs(site - mu).max()) if site.ndim == 1
           else _bloch.magnitude(site, mu)) if site.size else 0.0
    amps = _bloch.hopping_arrays(model)[0]
    if amps.size:
        got = max(got, float(np.abs(amps).max()))
    dim_k = int(model.lattice.dim_k)
    if dim_k:
        k = np.random.default_rng(_MAX_H_SEED).uniform(-0.5, 0.5,
                                                       size=(_MAX_H_NK, dim_k))
        got = max(got, _bloch.magnitude(_bloch.hamiltonian(model, k, params), mu))
    return float(got)


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
        shell (``"p"``), a site irrep (``"A1"``) or a sum of them
        (``"A1 + E"``), one per orbital, or a mapping from an orbital, or from
        several orbitals of one site, to such a word.  The group loses the
        operations that do not carry the declared content.  An orbital left
        undeclared takes the lowest angular momentum the hoppings allow.
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
    :func:`..symmetry._build_rep` to take by hand.

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
    :func:`..symmetry._build_rep`, and to the measurement before it where a
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
    from .symmetrization import _memory_budget, _search_ceiling
    from .symmetry import (_BUILD_KEYWORDS, _declaration_grammar, _names_joint_keys,
                           _resolve_spin, _spin_pair_split, _build_rep)

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
        from ._sectors import sector_action
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
    (:func:`_ansatz_refuted`).

    **What the declared content says the sites are**, read in the layer's own
    site groups on a substrate too (:func:`_layer_declared_reading`): a site
    declared ``E'`` and one declared ``E''`` of ``-6m2`` are different sites in
    both environments, as two sites of different ``typat`` are, although
    without ``σ_h`` both restrict to one ``E`` of ``3m``.

    **The states' projective class**, read once, on the free-standing layer:
    there each operation composed with ``σ_h`` has the mirror's chosen sign
    times its partner's matrix (:func:`_projective_class_refuted`), and on a
    substrate the free-standing cell of the same point set is handed down so
    that the class is read on the substrate's group with ``σ_h`` adjoined and
    its part that preserves the layer's normal carried
    (:func:`_projective_class_carried`).

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
    # (`symmetry._covariant_subgroup`), the subgroups carrying the centres
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
    from ._sectors import read_sectors

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
       :func:`..symmetry._build_rep` assembles from it rather than reading a
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
    preserves the layer's normal (:func:`_projective_class_carried`); ``None`` elsewhere.  The tensor
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
    from .symmetrization import (_held_off, _parts_coupling, _projection_operators,
                                 classify_operations)
    from .symmetrization import _symmetrized as _symmetrize
    from .symmetry import (_declaration_breaks, _exact_gate, _held_sets, _parts_apart,
                           _site_multiplets)

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


# --------------------------------------------------------------------------
# the centres, made exact before the cell's setting is read
# --------------------------------------------------------------------------

def _cell_sites(cell, cell_typat, symprec, *, n_extra=0) -> dict:
    r"""The cell's sites as the symmetry finder reads them, and which of them may move.

    The cell is set in 3D and its points grouped into sites at the finder's
    distance exactly as :func:`_group_of` groups them, so the operations found
    on these sites are the ones it finds.  For the centres' projection a site
    sits at its points' position -- their mean, each read on the branch of the
    site's first point, where they differ -- and weighs as many points as it
    holds, since moving a site moves each of them.  The shadow species a
    substrate puts under the sites (:func:`_decorated_cell`) are no centres:
    they are read with the rest, so that they remove the operations they
    remove, and they are built again under the moved sites.

    Returns ``lattice`` and ``vacuum_dirs`` (:func:`_pad_cell`), ``points`` (every
    point of the cell in the 3D cell's frame), ``first``, ``site_orbs`` and
    ``species`` (:func:`_sites_and_typat`), ``sites`` and ``weights`` (each
    site's position and count), ``scale`` (how a coordinate of the 3D cell
    maps onto the cell's own, per direction), ``own`` (how many points are the model's
    orbitals, which lead the cell) and ``moving`` (the sites that are not
    shadows, the ones the projection moves).
    """
    lattice, points, padded = _pad_cell(cell, _PAD_LENGTH)
    first, site_orbs, species = _sites_and_typat(points, cell_typat, symprec, lattice)
    norb = int(cell.norb)
    own = int(getattr(cell, _OWN_ORBITALS, norb))
    shadows = set(range(own, norb - int(n_extra)))
    moving = [s for s, orbs in enumerate(site_orbs)
              if any(int(o) not in shadows for o in orbs)]
    return _sites_frame(cell, lattice, points, padded, site_orbs, species=species,
                        own=own, moving=moving)


def _sites_frame(model, lattice, points, padded, site_orbs, *, species=None, own=None,
                 moving=None) -> dict:
    """The record :func:`_cell_sites` returns, for points already set in 3D and grouped."""
    points = np.asarray(points, dtype=float)
    first = np.array([points[int(orbs[0])] for orbs in site_orbs])
    sites = []
    for s, orbs in enumerate(site_orbs):
        on = np.array([points[int(o)] - np.rint(points[int(o)] - first[s]) for o in orbs])
        sites.append(first[s] if np.array_equal(on, np.repeat(on[:1], len(on), axis=0))
                     else on.mean(axis=0))
    lat_model = np.asarray(model.lat_vecs, dtype=float)
    scale = np.ones(3)
    for d in range(lat_model.shape[0]):
        scale[d] = float(np.linalg.norm(lattice[d]) / np.linalg.norm(lat_model[d]))
    return {"lattice": np.asarray(lattice, dtype=float), "vacuum_dirs": tuple(padded),
            "points": points, "first": first,
            "site_orbs": tuple(tuple(int(o) for o in orbs) for orbs in site_orbs),
            "species": None if species is None else np.asarray(species),
            "sites": np.array(sites, dtype=float),
            "weights": np.array([float(len(orbs)) for orbs in site_orbs]),
            "scale": scale, "own": int(points.shape[0] if own is None else own),
            "moving": np.array(range(len(site_orbs)) if moving is None else moving,
                               dtype=int)}


def _assigned_sites(model, group, assignment) -> dict:
    """A model's sites as its assignment to ``group`` groups them, read against the group.

    The record :func:`_cell_sites` returns, built from the model alone -- set
    in the group's lattice (:func:`_pad_cell`), every orbital its own and every
    site one the projection moves -- so that the assignment's permutations and
    lattice shifts are the action :func:`_symmetric_sites` reads.
    """
    lattice, points, padded = _pad_cell(model, group.vacuum_length, lattice=group.lattice)
    return _sites_frame(model, lattice, points, padded, assignment.site_orbitals)


def _site_action(frame, rotations, translations):
    r"""Which moving site each operation sends each one to, and by which lattice vector.

    Each image ``R·x + t`` is read against the sites of the same species, the
    nearest by Cartesian distance modulo the lattice, as :func:`_assign_orbitals`
    reads it -- once, on the positions as they are.  Returns ``(perm, shift)``,
    ``perm[g, s]`` the image of moving site ``s`` among the moving sites and
    ``shift[g, s]`` the lattice vector with ``R·x_s + t ≈ x_perm + shift``, or
    ``None`` where an operation does not permute them.
    """
    lattice = frame["lattice"]
    move = frame["moving"]
    sites = frame["sites"][move]
    species = frame["species"][move]
    same = species[:, None] == species[None, :]
    n = len(sites)
    perm = np.empty((len(rotations), n), dtype=int)
    shift = np.empty((len(rotations), n, 3))
    for g, (R, t) in enumerate(zip(rotations, translations)):
        images = sites @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float).reshape(-1)
        d = images[:, None, :] - sites[None, :, :]
        whole = np.rint(d)
        dist = np.where(same, np.linalg.norm((d - whole) @ lattice, axis=2), np.inf)
        j = np.argmin(dist, axis=1)
        if len(np.unique(j)) != n:
            return None
        perm[g] = j
        shift[g] = whole[np.arange(n), j]
    return perm, shift


def _generating(rotations, rows) -> list:
    r"""Operations among ``rows`` that carry every constraint all of them carry.

    Every pure translation among them, and as many others as it takes for their
    rotations to generate every rotation there: two operations of one group with
    one rotation differ by a pure translation, so a set of points these carry is
    carried by all of ``rows``.
    """
    ident = np.eye(3, dtype=int)
    rot = {int(g): np.rint(np.asarray(rotations[g], dtype=float)).astype(int) for g in rows}
    out = [int(g) for g in rows if np.array_equal(rot[int(g)], ident)]
    reached = {ident.tobytes()}
    for g in rows:
        if rot[int(g)].tobytes() in reached:
            continue
        out.append(int(g))
        gens = [rot[h] for h in out if not np.array_equal(rot[h], ident)]
        mats, reached = [ident], {ident.tobytes()}
        k = 0
        while k < len(mats):
            for h in gens:
                m = mats[k] @ h
                if m.tobytes() not in reached:
                    reached.add(m.tobytes())
                    mats.append(m)
            k += 1
    return out


def _symmetric_sites(frame, perm, shift, rotations, rows):
    r"""The moving sites carried, by the least move, onto a set the operations ``rows`` carry exactly.

    Each operation ``g`` sends site ``s`` to ``π_g(s)`` by the lattice vector
    ``n_g(s)`` (:func:`_site_action`), so the sites it carries exactly solve
    ``R_g (x_s − x_0) − (x_π(s) − x_π(0)) = n_g(s) − n_g(0)``: the translation
    eliminated through the first site, so that none is read off the positions
    and no table is needed.  The solutions are an affine space, and the sites
    are moved onto its nearest point, the move measured in Cartesian length and
    weighed by each site's points: one linear projection, on the constraints
    of operations that carry all of the others' (:func:`_generating`).  A
    shift of every site together carries the constraints into themselves, so
    the move has no common part: the origin is fitted to the centres, a shift
    of all of them being no displacement.

    Returns ``(sites, translations, closure)``: the moved sites in the 3D
    cell's frame; each operation's translation on them, ``t_g = x_π(0) + n_g(0) − R_g
    x_0``; and how far the moved sites miss being carried by every operation of
    ``rows`` with those translations -- rounding wherever the operations act on
    the sites as a group does.
    """
    lattice = frame["lattice"]
    move = frame["moving"]
    y = frame["sites"][move]
    w = frame["weights"][move]
    n = len(y)
    rows = [int(g) for g in rows]
    inv = np.linalg.inv(lattice)
    x = y.copy()
    gens = _generating(rotations, rows)
    if n > 1 and gens:
        lt = inv.T                                   # a Cartesian move, read as a reduced one
        i, s = np.arange(n - 1), np.arange(1, n)
        blocks, rhs = [], []
        for g in gens:
            R = np.asarray(rotations[g], dtype=float)
            rl = R @ lt
            block = np.zeros((n - 1, 3, n, 3))
            block[i, :, s, :] += rl
            block[i, :, 0, :] -= rl
            block[i, :, perm[g, 1:], :] -= lt
            block[i, :, perm[g, 0], :] += lt
            blocks.append(block.reshape(3 * (n - 1), 3 * n))
            rhs.append(((shift[g, 1:] - shift[g, 0])
                        - ((y[1:] - y[0]) @ R.T - (y[perm[g, 1:]] - y[perm[g, 0]])))
                       .reshape(-1))
        A = np.vstack(blocks)
        b = np.concatenate(rhs)
        if float(np.abs(b).max(initial=0.0)) > 0.0:
            root = np.repeat(np.sqrt(w), 3)
            u = np.linalg.lstsq(A / root, b, rcond=None)[0]
            x = y + (u / root).reshape(n, 3) @ inv
    trans = np.zeros((len(rotations), 3))
    closure = 0.0
    for g in rows:
        R = np.asarray(rotations[g], dtype=float)
        trans[g] = x[perm[g, 0]] + shift[g, 0] - R @ x[0]
        miss = x @ R.T + trans[g] - x[perm[g]] - shift[g]
        closure = max(closure, float(np.abs(miss).max(initial=0.0)))
    return x, trans, closure


def _rational_offset(trans, rotations, rows, periodic):
    r"""The shift of every site after which each translation is a rational of the cell.

    Exact sites fix each operation's translation up to the origin it is written
    about: moved together by ``−δ``, the translation ``t_g`` moves by ``−(1 −
    R_g)·δ``.  About the origin the model's own numbers are written in, every
    translation is a multiple of ``1/24`` (:data:`_TRANSLATION_DENOMINATOR`,
    with a repeating cell's own pure translations' denominators beside it), so
    ``(1 − R_g)·δ = t_g − round(t_g)`` is solved for every operation of ``rows``
    together, across the periodic directions; a direction every rotation fixes
    is left where it is.  Returns ``δ`` where that solution is exact to
    rounding, and ``None`` where the translations are on no such grid about any
    point: a crystal written about an origin the operations do not single out.
    """
    from fractions import Fraction

    per = [int(d) for d in periodic]
    rows = [int(g) for g in rows]
    if not per or not rows:
        return np.zeros(3)
    den = _TRANSLATION_DENOMINATOR
    for g in rows:
        if np.array_equal(np.rint(np.asarray(rotations[g], dtype=float)), np.eye(3)):
            for t in trans[g][per]:
                d = Fraction(float(t) % 1.0).limit_denominator(64).denominator
                den = den * d // math.gcd(den, d)
    mat = np.vstack([(np.eye(3) - np.asarray(rotations[g], dtype=float))[np.ix_(per, per)]
                     for g in rows])
    t = np.array([np.asarray(trans[g], dtype=float)[per] for g in rows])
    rhs = (t - np.rint(t * den) / den).reshape(-1)
    sol = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    if float(np.abs(mat @ sol - rhs).max(initial=0.0)) > _EXACT_CLOSURE:
        return None
    delta = np.zeros(3)
    delta[per] = sol
    # rounding is no shift: centres written in the model's own frame stay
    # exactly where they are
    delta[np.abs(delta) <= ROUNDING] = 0.0
    return delta


def _moves_of(frame, x) -> np.ndarray:
    r"""Each point's move onto its site at ``x``, in the cell's own reduced coordinates.

    A point moves from where it is to the position of the site it is on,
    modulo the lattice: two orbitals of one site written on the two sides of a
    cell edge are one site, and a whole lattice vector is no move.  A move at
    rounding is none, and the shadows the projection does not move stay at
    zero.
    """
    pts = frame["points"]
    out = np.zeros_like(pts)
    for k, s in enumerate(frame["moving"]):
        for o in frame["site_orbs"][int(s)]:
            d = x[k] - pts[int(o)]
            out[int(o)] = (d - np.rint(d)) * frame["scale"]
    out[np.abs(out) <= ROUNDING] = 0.0
    return out


def _frame_defect(frame, perm, rotations, trans, rows, sites=None) -> float:
    r"""How far the centres miss the operations ``rows``, with the translations ``trans``.

    ``max |R·τ_a + t − τ_b|`` modulo the lattice, over the operations and the
    model's own orbitals, ``τ_b`` the first orbital of the site the operation
    sends ``τ_a``'s to, in the 3D cell's frame: the position defect
    (:func:`..symmetrization._position_defect`), read on the centres as they
    are, or with every orbital on its site at ``sites`` where that is given.
    """
    pts = frame["points"]
    move = [int(s) for s in frame["moving"]]
    if sites is not None:
        pts = pts.copy()
        for k, s in enumerate(move):
            for o in frame["site_orbs"][s]:
                pts[int(o)] = sites[k]
    own = frame["own"]
    worst = 0.0
    for g in rows:
        R = np.asarray(rotations[g], dtype=float)
        for k, s in enumerate(move):
            target = int(frame["site_orbs"][move[int(perm[g, k])]][0])
            for o in frame["site_orbs"][s]:
                if int(o) >= own:
                    continue
                r = R @ pts[int(o)] + trans[g] - pts[target]
                worst = max(worst, float(np.abs(r - np.rint(r)).max()))
    return worst


def _geometry(model, orb):
    """A model's lattice with the centres ``orb``: geometry, carrying no Hamiltonian."""
    from pythtb import Lattice, TBModel

    return TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                           np.asarray(orb, dtype=float).tolist(),
                           periodic_dirs=list(model.periodic_dirs)), spinful=False)


def _moved_extras(extras, moves, start, dim_r) -> tuple:
    """The decorations moved with the centres: point ``start + k`` of the cell is extra ``k``."""
    return tuple((tuple(float(c) for c in (np.asarray(pos, dtype=float).reshape(-1)
                                           + moves[int(start) + k, :dim_r])), lab)
                 for k, (pos, lab) in enumerate(extras))


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

    from .symmetrization import _with_centres

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
    (:func:`_declaration_refuted`): each operation is told why, in its own
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

    from .symmetrization import _with_centres

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
    from .symmetrization import EXACT_GATE

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
    :func:`_ansatz_refuted`: operations the **detector** kept (their
    k-action class survives) whose **tensor structure** refutes them — the
    ``σ_h`` family on a slab, an operation whose k-action the hoppings carry
    only through a spin factor other than its own lift, or one they carry only
    with the sites permuted as another operation permutes them.  They are counted with
    the detector's own reasons under ``"tensor structure"``, because they are
    broken in exactly the same sense: the cell allows them and the Hamiltonian
    does not obey them.
    Those refuted by the states' **projective class**
    (:func:`_projective_class_refuted`) are counted under their own name: no
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
