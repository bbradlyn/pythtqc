"""The space group of a cell, and the records later measurements are written on.

The records are the group, the little group of a k-point, a site's Wyckoff
position in spglib's setting and in the tables' setting, the orbitals' sites
and their orbits, the little-group representations a measured multiplet
carries, a k-mesh and a term of a sum of site irreps.  The group is identified
by ``irrep`` on the cell's point set.  Its SU(2) lifts are put in the tables'
convention, its operations closed exactly, its pure translations read for the
primitive lattice and its magnetic type read off its antiunitary operations.
The files ``irreptables`` ships are read here, and the group's multiplication
table is computed here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
import math
import os
import string
from typing import Any
import warnings
import weakref

import irrep.spacegroup_irreps  # noqa: F401
import numpy as np

from .refusal import InternalError, Refusal
from ._tolerance import (
    DEFAULT_FINDER, WYCKOFF_REFERENCE_TOL, _EXACT_CLOSURE, _LATTICE_TOL,
    _SAME_TRANSLATION)
from ._text import _index_ranges, _lattice_vectors, _point_text

__all__ = ["tables_status", "GroupData", "LittleGroup", "SiteReference",
           "CorepIdentification", "KGrid", "Term"]


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
    """Write the installed ``irreptables`` version into ``info``.

    ``info`` is the mapping :func:`tables_status` builds.  Its
    ``"irreptables_version"`` is set to the version, ``"unknown"`` where the
    package names none, and left as it is where the package does not import.
    """
    try:
        import irreptables
        info["irreptables_version"] = getattr(irreptables, "__version__", "unknown")
    except ImportError:  # pragma: no cover - a spec that will not import
        pass


def tables_status() -> dict[str, Any]:
    """The state of ``irrep``'s character tables and of ``ortools``.

    The tables are loaded where the package ``irreptables`` is installed and
    its module ``irreptables.irreps`` is not imported before ``irrep`` in the
    same process.  Otherwise the tables are empty.  Every call that reads them
    then refuses ``irrep_tables``.

    Returns
    -------
    dict
        Five keys.

        - ``"tables"``: whether the tables are loaded.
        - ``"reason"``: where the tables are not loaded, which cause it is
          and what to do.  It is ``None`` where they are loaded.
        - ``"ortools"``: whether ``ortools``, the integer solver
          ``decompose`` needs, is installed.
        - ``"irrep_version"`` and ``"irreptables_version"``: the versions
          installed, ``None`` where a package is absent.
    """
    info: dict[str, Any] = {"tables": False, "ortools": False,
                            "irrep_version": None, "irreptables_version": None,
                            "reason": None}
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        info["ortools"] = True
    except ImportError:
        info["ortools"] = False
    try:
        import irrep
        from irrep import spacegroup_irreps as _sgi
        from irrep.utility import DummyMethodTables
    except ImportError as exc:  # pragma: no cover - environment without irrep
        info["reason"] = f"the package irrep cannot be imported: {exc}. Install it."
        return info
    info["irrep_version"] = getattr(irrep, "__version__", "unknown")
    bound = _sgi.IrrepTable
    if isinstance(bound, DummyMethodTables):
        import importlib.util
        if importlib.util.find_spec("irreptables") is None:
            info["reason"] = ("the package irreptables, which holds the character tables, "
                              "is not installed. Install it: "
                              "pip install 'irreptables>=3.1'")
            return info
        _set_irreptables_version(info)
        info["reason"] = (
            "the package irreptables is installed, and its module irreptables.irreps "
            "was imported before irrep in this process, so irrep holds no character "
            "tables for the rest of this process. In a new process, import pythtqc "
            "before irreptables.")
        return info
    _set_irreptables_version(info)
    info["tables"] = True
    return info


def _refuse_no_tables(status: dict[str, Any]) -> Refusal:
    return Refusal(
        gate="irrep_tables",
        reason=status["reason"],
        context={k: v for k, v in status.items() if k != "reason"},
    )


# --------------------------------------------------------------------------
# the tables as irreptables ships them
# --------------------------------------------------------------------------

def _irrep_table(number_str, spinor, magnetic):
    """One group's table of little-group representations, from ``irreptables``.

    ``number_str`` is the label the tables file the group under, ``"191"`` or
    ``"191.234"``.  ``spinor`` asks for the double-valued table, of states
    with spin one-half, and ``magnetic`` for the table of a magnetic group, a
    grey group among them, whose entries are coreps.  Returns an
    ``irreptables.irreps.IrrepTable``.  A table is shared by the calls that
    ask for it, and is never written into.
    """
    from irreptables.irreps import IrrepTable
    return _read_irrep_table(IrrepTable, str(number_str), bool(spinor), bool(magnetic))


@lru_cache(maxsize=32)
def _read_irrep_table(reader, number_str, spinor, magnetic):
    """The table ``reader(number_str, spinor, magnetic=magnetic)`` returns."""
    return reader(number_str, spinor, magnetic=magnetic)


def _ebr_file(number_str, spinor):
    """One group's elementary band representations, from ``irreptables``.

    ``number_str`` is the label the tables file the group under, ``"191"`` or
    ``"191.234"``, and ``spinor`` asks for the double-valued representations,
    of states with spin one-half.  Returns the mapping
    ``irreptables.ebrs.load_ebr_data`` reads, with the keys ``"basis"``,
    ``"ebrs"`` and ``"smith_form"``.  A file read is shared by the calls that
    ask for it, and is never written into.
    """
    from irreptables.ebrs import load_ebr_data
    return _read_ebr_file(load_ebr_data, str(number_str), bool(spinor))


@lru_cache(maxsize=32)
def _read_ebr_file(reader, number_str, spinor):
    """The file ``reader(number_str, spinor)`` returns."""
    return reader(number_str, spinor)


def _indicator_table(spinor, magnetic):
    """The symmetry-indicator formulas ``irreptables`` ships, keyed by group.

    ``spinor`` asks for the double-valued table, of states with spin
    one-half, and ``magnetic`` for the table keyed by a magnetic group's
    number, ``"166.98"``, where the other is keyed by the space-group number,
    ``"166"``.  Each table is read once, and the same object is returned to
    every call, so it is read and never written.
    """
    from irreptables.symmetry_indicators import load_si_table
    return _read_indicator_table(load_si_table, bool(spinor), bool(magnetic))


@lru_cache(maxsize=None)
def _read_indicator_table(reader, spinor, magnetic):
    """The table ``reader(spinor=spinor, magnetic=magnetic)`` returns."""
    return reader(spinor=spinor, magnetic=magnetic)


# --------------------------------------------------------------------------
# the multiplication table
# --------------------------------------------------------------------------

def _product_table(sg, *, block_bytes: float = 6.4e7):
    r"""The multiplication table of the operations ``irrep`` lists in ``sg``.

    ``sg`` is an ``irrep`` space group, its operations listed one per coset of
    the lattice translations.  The product of two operations is matched to the
    listed operation with the product's rotation, primed where exactly one of
    the two is, whose translation differs from the product's by a lattice
    vector, to within :data:`.._tolerance._SAME_TRANSLATION`.  In the double
    group the product of the two spin matrices is compared with the listed
    operation's.  It is negated where both operations are primed, since time
    reversal squares to ``-1`` on spin one-half.  ``block_bytes`` is the
    memory, in bytes, that sets how many rows of the table are built at once.
    Returns ``(table, translations, signs)``, integer arrays as
    :meth:`GroupData._get_product_table` describes them.

    Raises
    ------
    InternalError
        The list does not close under multiplication, or a product's spin
        matrix is neither sign of the listed operation's.
    """
    syms = sg.symmetries
    N = len(syms)
    R = np.array([np.asarray(s.rotation) for s in syms], dtype=int)
    t = np.array([np.asarray(s.translation, dtype=float) for s in syms])
    tr = np.array([bool(s.time_reversal) for s in syms])
    spinor = bool(sg.spinor)
    S = (np.array([np.asarray(s.spinor_rotation, dtype=complex) for s in syms])
         if spinor else None)
    # an operation's rotation and whether it is primed, as one row of ten integers
    own = np.concatenate([R.reshape(N, 9), tr.astype(int)[:, None]], axis=1)
    prod = np.zeros((N, N), dtype=int)
    diff = np.zeros((N, N, 3), dtype=int)
    spf = np.ones((N, N), dtype=int)
    # a block of rows of the table at a time
    rows = max(1, int(block_bytes // (8 * N * (10 + 3 + (16 if spinor else 0)))))
    for lo in range(0, N, rows):
        hi = min(lo + rows, N)
        B = hi - lo
        Rij = np.einsum("iab,jbc->ijac", R[lo:hi], R)
        tij = t[lo:hi, None, :] + np.einsum("iab,jb->ija", R[lo:hi], t)
        trij = tr[lo:hi, None] != tr[None, :]
        theirs = np.concatenate([Rij.reshape(B * N, 9),
                                 trij.astype(int).reshape(B * N, 1)], axis=1)
        # the listed operations with each product's rotation, primed as it is
        _uniq, inverse = np.unique(np.concatenate([own, theirs]), axis=0,
                                   return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        op_cls, cls = inverse[:N], inverse[N:].reshape(B, N)
        counts = np.bincount(op_cls, minlength=int(inverse.max()) + 1)
        order = np.argsort(op_cls, kind="stable")
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        cand = -np.ones((len(counts), max(1, int(counts.max()))), dtype=int)
        cand[op_cls[order], np.arange(N) - first[op_cls[order]]] = order
        block = -np.ones((B, N), dtype=int)
        dblock = np.zeros((B, N, 3), dtype=int)
        for m in range(cand.shape[1]):
            k = cand[cls, m]
            still = (block < 0) & (k >= 0)
            if not still.any():
                break
            dt = tij - t[np.where(k >= 0, k, 0)]
            near = np.round(dt)
            match = still & np.all(np.abs(dt - near) < _SAME_TRANSLATION, axis=-1)
            block[match] = k[match]
            dblock[match] = near[match].astype(int)
        if (block < 0).any():
            i, j = (int(x[0]) for x in np.nonzero(block < 0))
            raise InternalError(
                f"the product of operations {syms[lo + i].ind} and {syms[j].ind} is not "
                "among the operations of the group. Report it, with the model and the call.")
        prod[lo:hi], diff[lo:hi] = block, dblock
        if spinor:
            Sij = np.einsum("iab,jbc->ijac", S[lo:hi], S)
            Sij = np.where((tr[lo:hi, None] & tr[None, :])[:, :, None, None], -Sij, Sij)
            Sk = S[block]
            plus = np.isclose(Sij, Sk).all(axis=(2, 3))
            minus = np.isclose(Sij, -Sk).all(axis=(2, 3))
            if not np.all(plus | minus):
                i, j = (int(x[0]) for x in np.nonzero(~(plus | minus)))
                raise InternalError(
                    f"the product of the spin matrices of operations {syms[lo + i].ind} and "
                    f"{syms[j].ind} is not plus or minus the spin matrix of their product. "
                    "Report it, with the model and the call.")
            spf[lo:hi] = np.where(plus, 1, -1)
    return prod, diff, spf


#: Each group's multiplication table, computed once for the ``irrep`` object
#: that lists its operations.
_PRODUCT_TABLES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupData:
    """A space group, written in the cell of the model's pythtb Lattice.

    The record held by ``realized_group(model).group``.  The calls that read
    the tables take it, among them :func:`.._seitz.operations` and
    :func:`.._tables.corep_table`.  Its operations, one per coset of the
    lattice translations, are listed by ``realized_group(model).operations``,
    each with an index counted from 1.  The group's three-dimensional cell is
    the cell of the pythtb Lattice.  For a layer, the lattice vector out of
    its plane lies along its normal and spans vacuum.  The group
    :func:`..srsi.srsi_tables` reads from a space-group number has no cell.
    ``summary()`` prints it.

    Attributes
    ----------
    name : str
        The Hermann-Mauguin symbol, ``"P6/mmm1'"``.  A trailing ``1'`` marks
        time reversal as one of its operations.
    number_str : str
        The group's number in the tables: the BNS number of the magnetic
        space group, ``"191.234"``, for a grey group, and the space-group
        number, ``"168"``, for a group without time reversal.
    spinor : bool
        Whether it is the double group, of states with spin one-half.
    magnetic_type : str
        The type of the magnetic space group: ``"II"`` for a grey group, which
        contains time reversal as an operation of its own, and ``"I"`` for a
        group with no operation combined with time reversal.
    periodic_dirs : tuple of int or None
        If not ``None``, the lattice vectors along which the crystal is
        periodic, by their index in the group's three-dimensional cell:
        ``(0, 1, 2)`` for a crystal periodic in three directions, and
        ``(0, 1)`` for a layer whose vacuum is along the third.  ``None`` for
        the group of a space-group number, which has no cell.
    """

    name: str
    number_str: str
    spinor: bool
    magnetic_type: str
    periodic_dirs: tuple
    # the space-group number; whether irrep's magnetic tables are read, as for
    # a grey group; whether time reversal was asked of the symmetry finder
    _sg_number: int
    _magnetic: bool
    _include_TR: bool
    # irrep's record of each operation, one per coset of the lattice
    # translations, in the tables' order
    _symmetries: list
    # the three-dimensional cell the group was found on, a row per lattice
    # vector, a layer's with its vacuum; the points it was found on, in its
    # reduced coordinates, the orbitals on each and each point's species
    _lattice: np.ndarray
    _site_positions: np.ndarray
    _site_orbitals: tuple
    _typat: np.ndarray
    _vacuum_dirs: tuple
    _vacuum_length: float
    # the transformation from the model's cell to the tables' conventional one,
    # as `irrep` determined it, and whether it was found: every table lookup
    # reads it, and one made without it refuses `tables_not_ready`, saying why
    # (`_setup_note`)
    _refUC: np.ndarray
    _shiftUC: np.ndarray
    _tables_ready: bool
    # the symmetry finder's distance, in the model's length units, and the
    # model's periodic and spatial dimensions
    _symprec: float
    _dim_k: int
    _dim_r: int
    _setup_note: str | None = None
    # "cell", "realized", "descended" or "number": read off the cell's points,
    # measured on the Hamiltonian, measured after the cell was lowered, or the
    # grey group of a space-group number, with no cell, that the tables of the
    # invariants read from a number hold
    _provenance: str = "cell"
    # how far the operations as spglib found them missed closing into a group,
    # reduced modulo the lattice: 0.0 for a cell built from exact numbers, and
    # rounding for one whose centers a declared precision made exact; then the
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
    #: Memo of :func:`_wyckoff_setting_audit` for this group, the check that
    #: ``pyxtal``'s Wyckoff list is written in the same setting as the tables.
    #: It is a property of the group and not of a site, so it runs once.
    #: ``init=False``, so that a copy made by ``dataclasses.replace``, which may
    #: carry other operations, starts its own.
    _wyckoff_audit: dict = field(default_factory=dict, init=False, repr=False,
                                 compare=False)

    # -- delegated to ``irrep`` --------------------------------------------
    @property
    def _u_symmetries(self) -> list:
        """The unitary operations of :attr:`_symmetries`."""
        return self._sg.u_symmetries

    @property
    def _au_symmetries(self) -> list:
        """The operations of :attr:`_symmetries` combined with time reversal.

        The list is empty for a group of magnetic type I.
        """
        return self._sg.au_symmetries

    @property
    def _size(self) -> int:
        """The number of operations :attr:`_symmetries` lists."""
        return len(self._symmetries)

    # -- computed here, once for each ``irrep`` object ----------------------
    def _get_product_table(self, get_diff: bool = True):
        """The multiplication table of :attr:`_symmetries`.

        A representation of the group reproduces the table, its lattice
        vectors and its spin signs.  ``get_diff`` is whether the lattice
        vectors and spin signs are returned with the table.  Returns
        ``(table, translations, signs)``: ``table[i, j]`` is the position in
        :attr:`_symmetries` of the product of the operations at positions
        ``i`` and ``j``; ``translations[i, j]`` the lattice vector
        by which that product's translation exceeds the listed operation's, in
        reduced coordinates of the group's three-dimensional cell; and
        ``signs[i, j]`` the ``±1`` by which its spin matrix differs from the
        listed operation's, ``1`` throughout for a spinless group.  With
        ``get_diff=False``, the table alone.  Each is a copy of the table
        kept for the group's ``irrep`` object.
        """
        got = _PRODUCT_TABLES.get(self._sg)
        if got is None:
            got = _product_table(self._sg)
            _PRODUCT_TABLES[self._sg] = got
        if get_diff:
            return tuple(np.array(a) for a in got)
        return np.array(got[0])

    # -- delegated to ``irrep`` --------------------------------------------
    def _get_inverse_table(self):
        """The position in :attr:`_symmetries` of each operation's inverse.

        ``irrep`` computes it.  A position is one less than the operation's
        index.
        """
        return self._sg.get_inverse_table()

    # -- k between the model and the 3D cell -------------------------------
    def _k_to_3d(self, k) -> np.ndarray:
        """The model's reduced ``k`` as points of the three-dimensional cell.

        ``k`` has shape ``(dim_k,)`` or ``(n, dim_k)``.  Returns shape
        ``(n, 3)``: the periodic components carry over, and the component
        along a layer's vacuum is zero.
        """
        k = np.atleast_2d(np.asarray(k, dtype=float))
        out = np.zeros((k.shape[0], 3))
        out[:, list(self.periodic_dirs)] = k
        return out

    def _k_from_3d(self, k3) -> np.ndarray:
        """Points ``k3`` of the three-dimensional cell as the model's reduced k.

        ``k3`` has shape ``(3,)`` or ``(n, 3)``.  Returns shape ``(n, dim_k)``,
        the components along a layer's vacuum dropped.
        """
        k3 = np.atleast_2d(np.asarray(k3, dtype=float))
        return k3[:, list(self.periodic_dirs)]

    def _in_plane(self, k3, atol: float = 1e-9) -> bool:
        """Whether the points ``k3`` lie in a layer's plane.

        ``k3`` holds points of the three-dimensional cell, shape ``(3,)`` or
        ``(n, 3)``.  A point lies in the plane where each of its components
        along the vacuum is at most ``atol`` in absolute value.  Always
        ``True`` for a crystal periodic in three directions.
        """
        k3 = np.atleast_2d(np.asarray(k3, dtype=float))
        if not self._vacuum_dirs:
            return True
        return bool(np.all(np.abs(k3[:, list(self._vacuum_dirs)]) <= atol))

    def _k_action_of(self, op):
        """One operation's action on the model's reduced k, as a callable.

        ``op`` is one of :attr:`_symmetries`.  Returns ``k -> R k``, or
        ``k -> -R k`` for an operation combined with time reversal, taking
        points of shape ``(dim_k,)`` or ``(n, dim_k)`` to shape
        ``(n, dim_k)``; ``None`` for an operation that turns a layer's plane
        out of itself.  The image is left unreduced: in pythtb's convention
        ``H(k + G)`` is not ``H(k)``, and an operation carries ``H(k)`` to
        ``H`` at the image itself.
        """
        probe = np.eye(self._dim_k)
        rot = op.transform_k(self._k_to_3d(probe))
        if not self._in_plane(rot):
            return None
        M = self._k_from_3d(rot)  # rows: images of the unit k-vectors

        def action(k, _M=M):
            return np.atleast_2d(np.asarray(k, dtype=float)) @ _M
        return action

    def summary(self) -> str:
        """The group as text, a line for each quantity.

        The attributes that hold each quantity are named beside it.
        """
        lines = [f"{self.name} #{self.number_str}   (.name, .number_str)"]
        if self.magnetic_type == "II":
            lines.append("  a grey group, magnetic type II: time reversal is one of its "
                         "operations   (.magnetic_type)")
        else:
            lines.append("  magnetic type I: no operation is combined with time reversal   "
                         "(.magnetic_type)")
        if self.spinor:
            lines.append("  the double group: each operation's negative is in it too   "
                         "(.spinor)")
        if self._vacuum_dirs:
            lines.append(f"  a layer: periodic along {_lattice_vectors(self.periodic_dirs)} "
                         f"lattice vectors, vacuum along {_lattice_vectors(self._vacuum_dirs)}"
                         "   (.periodic_dirs)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        if int(self._lattice_index) > 1:
            tag = (f" [lattice supergroup, index {int(self._lattice_index)}; "
                   "tables on the finer lattice]")
        else:
            tag = "" if self._tables_ready else " [tables not ready]"
        if _is_number_group(self):
            tag = ""
        periodic = None if self.periodic_dirs is None else tuple(self.periodic_dirs)
        return (f"GroupData({self.name} #{self.number_str}, "
                f"spinor={self.spinor}, magnetic_type={self.magnetic_type!r}, "
                f"periodic_dirs={periodic}{tag})")


@dataclass(frozen=True)
class LittleGroup:
    """The operations that leave one k invariant up to a reciprocal lattice vector.

    The record :func:`.._tables.little_group` returns.  ``summary()`` prints
    it, and ``len()`` is the number of its unitary operations.

    Attributes
    ----------
    k : ndarray, shape (dim_k,)
        The point, in the model's reduced coordinates.
    kpoint : str or None
        If not ``None``, the tables' name of the point, ``"X"``.  ``None``
        where ``k`` is none of the group's tabulated points, up to a
        reciprocal lattice vector of the model's cell, or where the group has
        no transformation to the tables' conventional cell.
    unitary : tuple of int
        The indices of its unitary operations, ascending.  The characters of
        a little-group representation at ``k`` are given on these
        operations, in this order.
    antiunitary : tuple of int
        If nonempty, the indices of its operations combined with time
        reversal, ascending: those whose rotation carries ``k`` to ``-k``, up
        to a reciprocal lattice vector.  The tables give no character on
        them.
    G : dict
        ``{index: G}`` for each unitary operation ``g``, with ``G = gk - k``
        the reciprocal lattice vector by which ``g`` moves ``k``.  Its three
        components are in reduced coordinates of the group's
        three-dimensional cell, the one along a layer's normal included.
    """

    k: np.ndarray
    kpoint: Any
    unitary: tuple
    antiunitary: tuple
    G: dict
    # the point in the three-dimensional cell; irrep's records of the unitary
    # and antiunitary operations, in index order; the group, for `operations`
    _k3: np.ndarray = field(default=None, repr=False, compare=False)
    _unitary_operations: list = field(default_factory=list, repr=False, compare=False)
    _antiunitary_operations: list = field(default_factory=list, repr=False,
                                          compare=False)
    _group: Any = field(default=None, repr=False, compare=False)

    @property
    def operations(self):
        """Its operations, as ``realized_group(model).operations`` writes them.

        Returns
        -------
        OperationTable or Refusal
            A row for each operation of :attr:`unitary` and
            :attr:`antiunitary`, all in ascending index.  A ``Refusal``
            returned instead of the table is falsy.

        Refusals
        --------
        ``tables_not_ready``
            The group has no transformation to the tables' conventional cell,
            in which each Seitz symbol is written.
        """
        from ._seitz import _operation_rows
        return _operation_rows(self._group, tuple(self.unitary) + tuple(self.antiunitary))

    def __len__(self) -> int:
        return len(self.unitary)

    def summary(self) -> str:
        """The little group as text, a line for each quantity.

        The attributes that hold each quantity are named beside it.
        """
        lines = [(f"{self.kpoint} {_point_text(self.k)}, in the model's reduced coordinates   "
                  "(.kpoint, .k)" if self.kpoint else
                  f"k = {_point_text(self.k)}, in the model's reduced coordinates   (.k)"),
                 "  its operations, by index and Seitz symbol   (.operations)",
                 f"  its unitary operations, by index: {_index_ranges(self.unitary)}   "
                 "(.unitary)",
                 "  the reciprocal lattice vector each of them moves k by   (.G)"]
        if self.antiunitary:
            lines.append("  its operations combined with time reversal, by index: "
                         + _index_ranges(self.antiunitary) + "   (.antiunitary)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        primed = _index_ranges(self.antiunitary)
        point = (f"{self.kpoint} {_point_text(self.k)}" if self.kpoint
                 else f"k = {_point_text(self.k)}")
        return (f"LittleGroup({point}: unitary operations "
                f"{_index_ranges(self.unitary)}"
                + (f"; with time reversal {primed}" if primed else "") + ")")


@dataclass(frozen=True)
class _WyckoffPosition:
    """One Wyckoff position of a group, lettered in spglib's setting.

    One entry of what :func:`_wyckoff_positions_spglib` returns.  The
    position is the image of the map ``A·(x, y, z) + b``, in reduced
    coordinates of the group's three-dimensional cell.

    Attributes
    ----------
    letter : str
        Its letter in spglib's setting, which can differ from the tables'
        letter for the position.
    expression : str
        The position as ``irrep`` writes it, in ``x``, ``y`` and ``z``.
    offset, jacobian : ndarray, shapes (3,) and (3, 3)
        ``b`` and ``A``.
    n_free : int
        The rank of ``A``, the number of free parameters of the position.
    """

    letter: str
    expression: str
    offset: np.ndarray
    jacobian: np.ndarray
    n_free: int

    def distance(self, position, metric: np.ndarray) -> float:
        """The distance from the point ``position`` to this position.

        ``position`` is in reduced coordinates of the cell whose lattice
        vectors are the rows of ``metric``.  For each translate of the point by
        a lattice vector with components in ``{-1, 0, 1}``, its residual from
        the position is found by least squares in reduced coordinates and its
        Cartesian length taken.  Returns the smallest, in the length units of
        ``metric``.  At a position with no free parameter it is the Cartesian
        distance; on a line or a plane it can exceed it, unless the cell's
        vectors are orthogonal and of one length.
        """
        best = np.inf
        A, b = self.jacobian, self.offset
        for shift in _NEIGHBOR_SHIFTS:
            rhs = np.asarray(position, dtype=float) + shift - b
            if self.n_free == 0:
                resid = rhs
            else:
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                resid = rhs - A @ sol
            best = min(best, float(np.linalg.norm(resid @ metric)))
        return best


_NEIGHBOR_SHIFTS = np.array([[i, j, k] for i in (-1, 0, 1)
                              for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


@dataclass(frozen=True)
class SiteReference:
    """One site's Wyckoff position in the tables' setting.

    The tables' setting is the conventional cell used by the irreptables
    package.  The record is one entry of what
    :func:`.._tables.wyckoff_sites_tables` returns.  It prints as one line:
    its position, its free parameters and the positions it slides onto.

    Attributes
    ----------
    wyckoff : str
        The Wyckoff position the site lies on, written with its multiplicity
        in the conventional cell, ``"8a"``.  Where the site lies on several
        positions, within the ``position_tol`` of
        :func:`.._tables.wyckoff_sites_tables`, it is the one with the fewest
        free parameters, and then the smallest multiplicity.
    slides_onto : tuple of str
        The Wyckoff positions at which the tables list elementary band
        representations and whose points all lie on the site's own position,
        that position included where the tables list it.  An orbital at the
        site can be moved onto each of them without leaving its position.
        For the line ``6c``, ``(0, 0, z)``, of ``R-3m`` they are
        ``("3a", "3b")``, and for a site at ``3a`` they are ``("3a",)``.
    n_free : int
        The number of its free parameters: 0 for a point, 1 a line, 2 a
        plane, 3 the general position.  In a polar group no position is a
        point: ``P4mm``'s ``1a`` is ``(0, 0, z)``.
    position : ndarray, shape (3,)
        The site, in reduced coordinates of the group's three-dimensional
        cell, not of the tables' conventional cell.
    """

    wyckoff: str
    slides_onto: tuple
    n_free: int
    position: np.ndarray
    # the letter of every position whose variety holds the point, most special
    # first; the site's place in the list handed in; the point in the tables'
    # conventional cell, reduced into [0, 1); how far it sits from its variety
    _varieties: tuple = field(default=(), repr=False, compare=False)
    _index: int = field(default=0, repr=False, compare=False)
    _reference: np.ndarray = field(default=None, repr=False, compare=False)
    _residual: float = field(default=0.0, repr=False, compare=False)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"SiteReference({self.wyckoff}, n_free={self.n_free}, "
                f"slides_onto {self.slides_onto})")


@dataclass(frozen=True)
class _SiteReading:
    """One point's Wyckoff position in the tables' setting.

    One entry of what :func:`.._tables._sites_in_tables_setting` returns.

    Attributes
    ----------
    index : int
        The point's place in the list handed in.
    position, reference : ndarray, shape (3,)
        The point, in reduced coordinates of the group's three-dimensional
        cell, and of the tables' conventional cell reduced into ``[0, 1)``.
    letter, multiplicity, n_free : str, int and int
        The most special position whose variety holds the point: its letter,
        its multiplicity in the conventional cell and the number of its free
        parameters.
    varieties : tuple of str
        The letters of every position whose variety holds the point, most
        special first.
    closure : tuple of str
        The letters on the closure of its variety at which the tables induce
        elementary band representations.
    residual : float
        How far the point sits from its variety, the largest component in
        reduced coordinates of the conventional cell.
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


@dataclass(frozen=True)
class OrbitalAssignment:
    """Each orbital's site, the sites' orbits, and how the operations move them.

    Each operation ``{R|t}`` of the group carries each site ``τ`` to
    ``R τ + t``, which is matched to the nearest site of the same species.
    Two sites are of one species where their orbitals carry the same labels
    of ``typat=``, as many of each.  With no ``typat=``, they are of one
    species where they hold as many orbitals.  ``summary()`` prints it by
    Wyckoff orbit.

    Attributes
    ----------
    site_of_orbital, orbit_of_orbital : ndarray, shape (norb,)
        The site each orbital sits on, and that site's orbit.
    site_positions, site_orbitals : ndarray and tuple
        The sites, in reduced coordinates of the group's three-dimensional
        cell, and the orbitals on each.
    orbits : tuple of tuple of int
        The sites, grouped into the orbits the operations carry them through.
    orbit_of_site : ndarray, shape (nsite,)
        The orbit each site is on.
    wyckoff_tables : tuple of str
        If nonempty, each site's Wyckoff letter in the tables' setting, the
        conventional cell used by the irreptables package.  Empty where the
        letters could not be read.
    site_symmetry : tuple of str
        Each site's site-symmetry group, the point group of the unitary
        operations that fix it, by its Hermann-Mauguin symbol: ``"3m"``.
    site_group : tuple of tuple of int
        The indices of the operations that fix each site, those combined with
        time reversal included.
    permutations : ndarray, shape (nsym, nsite)
        ``permutations[g, i]`` is the site onto which the operation of row
        ``g`` carries site ``i``.  Row ``g`` is the operation of index
        ``g + 1`` in ``realized_group(model).operations``.
    lattice_shift : ndarray, shape (nsym, nsite, 3)
        The lattice vector by which each image differs from the site it is
        matched to, in reduced coordinates: ``R τ_i + t - τ_j`` for the
        operation ``{R|t}`` of row ``g``, with ``τ_i`` the position of site
        ``i`` and ``j = permutations[g, i]``.
    wyckoff_spglib : tuple of str
        The Wyckoff letter of each orbit in spglib's setting, ``""`` where
        spglib finds none.  A letter can differ from the tables' letter for
        the same position, in :attr:`wyckoff_tables`.
    wyckoff_spglib_dataset : tuple of str
        If nonempty, the Wyckoff letter of each site in spglib's setting,
        read by spglib from the sites alone.  Their group can be larger than
        the Hamiltonian's.  Empty where spglib gives no letters.
    max_distance : float
        The largest Cartesian distance between an operation's image of a site
        and the site it was matched to, in the model's length units.
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
        """The orbitals' sites and orbits as text, a line for each quantity.

        The attributes that hold each quantity are named beside it.
        """
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
class CorepIdentification:
    """The little-group representations each measured multiplet carries at one k.

    The record :func:`.._tables.identify_from_characters` returns, and
    ``coreps_at(...).numerical.identification`` holds.  The multiplicities
    are read off each multiplet's characters at a high-symmetry point and
    rounded to integers.  ``summary()`` prints it.

    Attributes
    ----------
    kpoint, k : str and ndarray
        The tabulated point, by name and in the model's reduced coordinates.
    labels : tuple of dict
        Each multiplet's little-group representations,
        ``{name: multiplicity}``.
    dims : ndarray
        The dimension of each multiplet: each multiplicity times the
        dimension of its representation, summed.
    indices : tuple of int
        The indices of the little group's unitary operations, on which the
        characters are given.
    multiplicities : ndarray
        The multiplicities before rounding, a complex array with a row per
        multiplet and a column per little-group representation.
    label_order : tuple of str
        The little-group representations at the point, by name, in the order
        of the columns of :attr:`multiplicities`.
    integrality : float
        The largest distance of a multiplicity before rounding from the
        nearest integer.
    """

    kpoint: str
    k: np.ndarray
    indices: tuple
    labels: tuple
    multiplicities: np.ndarray
    label_order: tuple
    dims: np.ndarray
    integrality: float
    # which multiplets carry the same characters, numbered from the lowest by
    # first appearance: two share a number when their characters agree within
    # the label tolerance
    _partition: tuple = field(default=(), repr=False)
    # what the summary calls the labels: "irreps" in a group with no time
    # reversal, "coreps" otherwise
    _kind: str = field(default="coreps", repr=False, compare=False)

    def summary(self) -> str:
        """The multiplets as text, a line for each quantity.

        The attributes that hold each quantity are named beside it.
        """
        lines = [f"{self.kpoint} {_point_text(self.k)}, in the model's reduced coordinates   "
                 "(.kpoint, .k)",
                 f"  each multiplet's {self._kind}, and its dimension   (.labels, .dims)"]
        for row, (labels, dim) in enumerate(zip(self.labels, np.asarray(self.dims).reshape(-1))):
            content = " + ".join((f"{int(m)}x{label}" if int(m) > 1 else str(label))
                                 for label, m in labels.items())
            lines.append(f"    multiplet {row}, dimension {int(round(float(dim)))}: "
                         f"{content or 'no ' + self._kind[:-1]}")
        lines += [f"  the little group's unitary operations, the characters' columns, by "
                  f"index: {_index_ranges(self.indices)}   (.indices)",
                  f"  the multiplicities before rounding, at most {float(self.integrality):.2g} "
                  "from integers   (.multiplicities, .label_order, .integrality)"]
        return "\n".join(lines)


@dataclass(frozen=True)
class KGrid:
    """A mesh of k-points that holds GM, with the high-symmetry points on it.

    The record :func:`.._tables.k_grid` returns.  ``summary()`` prints it.
    Its :attr:`mesh` is a ``pythtb.Mesh``, which ``pythtb.WFArray`` takes
    directly.

    Attributes
    ----------
    divisions : tuple of int
        The number of k-points along each reciprocal lattice vector of the
        group's three-dimensional cell, ``1`` along a layer's normal.
    symmetric : bool
        Whether the group carries the mesh onto itself.
    mesh : pythtb.Mesh
        The mesh, with ``k`` in ``[0, 1)`` along each periodic direction and
        one axis per periodic direction.
    irreducible : ndarray, shape (n, dim_k), or None
        If not ``None``, one point of the mesh from each orbit of the group,
        in the model's reduced coordinates.  ``None`` for a mesh that is not
        symmetric.
    contains : dict
        If nonempty, ``{name: bool}``: whether the mesh holds each tabulated
        high-symmetry point.  A point is on the mesh where each of its
        reduced coordinates in the model's cell is a multiple of ``1/n``
        along an axis of ``n`` divisions: ``1/2`` needs an even number of
        divisions, and ``1/3`` a multiple of three.
        :func:`.._tables.high_symmetry_points` gives the coordinates of each
        point.  Empty where the high-symmetry points cannot be read.
    """

    divisions: tuple
    symmetric: bool
    mesh: Any
    irreducible: np.ndarray | None
    contains: dict

    @property
    def kpoints(self) -> np.ndarray:
        """The mesh's points, ``(n, dim_k)``, in the model's reduced coordinates."""
        return np.asarray(self.mesh.flat)

    def summary(self) -> str:
        """The mesh as text, a line for each quantity.

        The attributes that hold each quantity are named beside it.
        """
        sampled = [str(name) for name, hit in self.contains.items() if hit]
        missed = [str(name) for name, hit in self.contains.items() if not hit]
        lines = [f"{'x'.join(str(int(d)) for d in self.divisions)} mesh, "
                 + ("symmetric" if self.symmetric else "not symmetric") + " under the group   "
                 "(.divisions, .symmetric)",
                 "  pythtb's own mesh, ready for a WFArray   (.mesh, .kpoints)"]
        if self.irreducible is not None:
            lines.append(f"  its irreducible points, {len(self.irreducible)} of "
                         f"{len(self.kpoints)}: the group carries them onto all the others   "
                         "(.irreducible)")
        lines.append("  the high-symmetry points on it: " + (", ".join(sampled) or "none")
                     + "   (.contains)")
        if missed:
            lines.append("  the high-symmetry points it misses: " + ", ".join(missed)
                         + "   (.contains)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        sampled = [str(name) for name, hit in self.contains.items() if hit]
        missed = [str(name) for name, hit in self.contains.items() if not hit]
        return (f"KGrid({'x'.join(str(int(d)) for d in self.divisions)} mesh, "
                f"{'symmetric' if self.symmetric else 'not symmetric'}; samples "
                + (", ".join(sampled) if sampled else "no high-symmetry point")
                + (f"; misses {', '.join(missed)}" if missed else "") + ")")


@dataclass(frozen=True)
class Term:
    """An integer multiple of one site irrep at a Wyckoff position.

    One term of a band representation written as a sum of band
    representations induced from site irreps.  The columns of
    :func:`.._tables.ebr_data` are terms of multiplicity one, each an
    elementary band representation.

    Attributes
    ----------
    multiplicity : int
        The integer that multiplies the site irrep, never zero.  It is
        negative only in the signed combinations ``decompose`` returns.
    irrep : str
        The site irrep, ``"-E1"``, by the name the list of elementary band
        representations gives it.  A site irrep the list does not name, one
        whose induced band representation is composite, takes the name the
        tables of the stable real-space invariants give it.
    wyckoff : str
        The Wyckoff position, with its multiplicity in the conventional cell,
        ``"1a"``.
    site_symmetry : str
        The site-symmetry group of the position in the space group whose
        band representation the term belongs to, ``"4mm1'"`` in a grey group
        and ``"3"`` in a group without time reversal.
    unitary_site_symmetry : str
        Its unitary part, ``"4mm"``.
    """

    multiplicity: int
    irrep: str
    wyckoff: str
    site_symmetry: str
    unitary_site_symmetry: str


def _written_site_symmetry(name: str) -> str:
    """A site-symmetry group's name as the space groups' names write it.

    ``name`` is a name read from the list of elementary band representations,
    after :func:`.._tables._barred` has put back the bar of an improper site
    group.  The list writes the cubic ``m3`` and ``m3m`` where the space
    groups' names write ``m-3`` and ``m-3m``.  Returns a name that starts
    with ``m3`` with ``m-3`` in its place, so ``m31'`` becomes ``m-31'``, and
    every other name as it is.
    """
    name = str(name)
    if name.startswith("m3"):
        return "m-3" + name[2:]
    return name


# --------------------------------------------------------------------------
# cell construction
# --------------------------------------------------------------------------

#: The name under which a cell :func:`.._layers._decorated_cell` builds carries how many of
#: its points are the model's own orbitals, the leading ones.  The vacuum a slab
#: is given is measured from those alone (:func:`_pad_cell`), so it is set
#: once, by the model.  No species added to the cell moves it: a substrate's
#: shadow, a decoration, the generic orbit of a descent.
_OWN_ORBITALS = "_pythtqc_own_orbitals"


def _pad_cell(model, vacuum_length: float, *, lattice=None):
    """A model's cell as a three-dimensional cell, in vacuum where not periodic.

    ``model`` is a pythtb model, or a cell built on one by
    :func:`.._layers._decorated_cell`, whose leading points are the model's
    own orbitals.  Each non-periodic lattice vector of the model is scaled,
    never turned, so the model's reduced k is unchanged.  A tilted one is put
    along the layer's normal beforehand, by
    :func:`.._layers._slab_presentation`.  Each is lengthened, where it is
    shorter, to the extent of the model's own orbitals along it plus
    ``vacuum_length``, in the model's units of length; the cell's other
    points do not enter.  With ``lattice``, the group's three-dimensional
    lattice (:attr:`GroupData._lattice`), each is scaled to that lattice's
    length along it instead.  A missing dimension is filled with an
    orthogonal vector of length ``vacuum_length``.  Along each vacuum
    direction the points are centered on the origin, so that a horizontal
    mirror carries no translation.  Returns
    ``(lattice, positions, vacuum_dirs)``: the lattice vectors, a row each;
    every point in reduced coordinates, wrapped into ``[0, 1)``; and the
    vacuum directions, sorted.
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
    # vacuum_length, or, read against a group, to the length its lattice has
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

    # center the slab on the origin along every vacuum direction
    for d in padded:
        if pos3.shape[0]:
            mid = 0.5 * (pos3[:, d].max() + pos3[:, d].min())
            pos3[:, d] -= mid

    return lat3, pos3 % 1.0, tuple(sorted(padded))


def _sites_and_typat(positions, typat, symprec, lattice):
    """The points grouped into sites, and each site's species.

    ``positions`` are the points, shape ``(n, 3)``, in reduced coordinates of
    the cell whose lattice vectors are the rows of ``lattice``.  A point
    joins the first site whose first point is within ``symprec`` of it, a
    Cartesian distance modulo the lattice in the units of ``lattice``.
    ``typat`` gives each point's label, and every point is ``0`` where it is
    ``None``.  Sites whose points carry the same labels, sorted, are given one
    species, numbered from 1 in order of first appearance, so that without
    ``typat`` sites differ by their number of points alone.  Returns
    ``(sites, site_points, species)``: each site's first point, the points of
    each site, and each site's species.

    Raises
    ------
    InternalError
        ``typat`` does not have one entry per point.  Every call checks the
        ``typat=`` its user passed before any point is grouped.
    """
    n = positions.shape[0]
    labels = [0] * n if typat is None else list(typat)
    if len(labels) != n:
        raise InternalError(
            f"the sites were read with {len(labels)} species "
            f"label{'' if len(labels) == 1 else 's'} for {n} points. Report it, with the "
            "model and the call.")

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
# the spinor branch: the tables' own spin frame, and their lift set in place
# of irrep's lift of each operation
# --------------------------------------------------------------------------

_PAULI = np.array([[[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]],
                   [[1, 0], [0, -1]]], dtype=complex)

#: The Cartesian frames the shipped ``SymopTable.S`` matrices are written in,
#: relative to the conventional cell's standard orientation: ``a`` along
#: ``+x``, ``b`` in the ``+y`` half of the ``xy`` plane, ``c`` with positive
#: ``z``.  That orientation is the lower-triangular Cholesky factor of the
#: cell's Gram matrix.  Tried in the order given; the first that reproduces
#: every shipped ``S`` as a conjugate of the operation's own SU(2) lift, up to
#: a sign, is the group's (:func:`_canonical_spinor_signs`).  The two cover
#: every shipped double-valued table.
_TABLE_SPIN_FRAMES = (
    ("standard", np.eye(3)),
    ("hexagonal", np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])),
)


def _su2_lift(rot) -> np.ndarray:
    """An SU(2) lift of the proper rotation ``rot``, a Cartesian 3x3 matrix.

    Returns ``cos(θ/2) - i sin(θ/2) n·σ`` for the axis ``n`` and angle ``θ``
    of ``rot``, read in closed form with no optimizer, so the same ``rot``
    always gives the same matrix.  At ``θ = π`` the axis is oriented so that
    its first nonzero component is positive.  Which of the two lifts is
    returned is fixed by no convention: the matrix is used to conjugate,
    where their sign cancels.
    """
    rot = np.asarray(rot, dtype=float)
    cos = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(cos))
    if ang < 1e-9:
        return np.eye(2, dtype=complex)
    if abs(ang - np.pi) < 1e-6:
        # sin θ = 0: the axis comes from (R + 1)/2 = n n^T, not from the
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
    r"""How far a set of SU(2) lifts is from closing, and where signs twist it.

    ``rots`` holds an integer rotation matrix per operation, in any one basis
    in which composition is the matrix product; ``lifts`` an SU(2) matrix
    :math:`S(g)` lifting each; ``corrections`` a sign :math:`b(g) = \pm 1`
    for each.  The product ``gh`` is read as the first operation listed with
    its rotation.  Operations that share a rotation are assumed to share a
    lift.  At each pair :math:`g, h` whose product rotation is in
    ``rots``, a set that closes has
    :math:`S(g) S(h) = \omega(g, h) S(gh)` with :math:`\omega = \pm 1`, and
    multiplying the lifts by :math:`b` multiplies :math:`\omega` by
    :math:`b(g) b(h) b(gh)`.  With
    :math:`\lambda = \mathrm{tr}(S(g) S(h) S(gh)^\dagger)/2`, which is
    :math:`\omega` for a set that closes, returns
    ``(defect, mixed, pairs, worst_at)``: the largest of
    :math:`||\lambda| - 1|` and :math:`|\mathrm{Im}\,\lambda|` over the pairs;
    the number of pairs at which :math:`b(g) b(h) b(gh) = -1`; the number of
    pairs read; and the positions, counted from 1, of the pair where the
    largest is reached.  Returns ``(0.0, 0, 0, None)`` where no pair is read.
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
    r"""The sign carrying each of ``irrep``'s SU(2) lifts onto the tables'.

    ``irreptables`` ships with each operation the SU(2) matrix ``S`` its
    characters were computed with, written in a Cartesian frame fixed relative
    to the conventional cell; ``irrep``'s lift ``sym.spinor_rotation`` is
    written in the model's frame.  The frame map ``F``, orthogonal, carries the
    conventional cell's standard orientation, the lower-triangular Cholesky
    factor of its Gram matrix, onto the cell as the model writes it.  Each frame
    ``T`` of :data:`_TABLE_SPIN_FRAMES` is tried in turn, with ``R`` the SU(2)
    lift of ``F T``, an improper one through its proper part.  The first frame
    under which every ``R S R†`` is ``±1`` times the operation's own lift,
    within ``tol``, is the group's, and those signs are returned.

    Parameters
    ----------
    sg : irrep.spacegroup_irreps.SpaceGroupIrreps
        A double group whose operations ``irrep`` has sorted into the tables'
        order.
    tol : float, default 1e-3
        The largest miss of ``sym.spinor_rotation = ±R S R†`` accepted, entry
        by entry, and the largest ``defect`` of :func:`_lift_set_cocycle` the
        tables' own lifts may close to.  ``irreptables`` writes ``S`` to about
        five decimals, and a wrong frame misses by order one.

    Returns
    -------
    dict
        ``"frame"``, the name of the frame, ``"correction"``, its ``T``, and
        ``"spin_transf"``, its ``R``; ``"signs"``, a tuple whose entry ``i``
        carries the lift of ``sg.symmetries[i]`` onto the tables'; ``"defect"``,
        the largest miss of that match; and ``"cocycle"``, ``"mixed"`` and
        ``"pairs"``, what :func:`_lift_set_cocycle` returns on the tables' own
        lifts with these signs.  ``"mixed"`` is ``0`` where the stored lifts
        already closed with the tables' factor system.

    Raises
    ------
    InternalError
        The group and its table list different numbers of operations, the
        conventional cell has no standard orientation or its frame map is not
        orthogonal, no frame reproduces every ``S``, or the tables' own ``S`` do
        not close.
    """
    table = _irrep_table(sg.number_str, True, sg.magnetic)
    tsyms = table.u_symmetries + table.au_symmetries
    where = f"the spin frame of the tables for group #{sg.number_str}"
    if len(tsyms) != len(sg.symmetries):
        raise InternalError(
            f"{where}: the group lists {len(sg.symmetries)} operations, and its table "
            f"{len(tsyms)}. Report it, with the model and the call.")

    lattice = np.asarray(sg.real_lattice, dtype=float)
    conv = np.asarray(sg.refUC, dtype=float).T @ lattice     # conventional cell, this frame
    gram = conv @ conv.T
    try:
        standard = np.linalg.cholesky(gram)                  # ... in standard orientation
    except np.linalg.LinAlgError as exc:                     # pragma: no cover - defensive
        raise InternalError(
            f"{where}: the conventional cell of the tables, written in this cell, is "
            f"singular (Gram determinant {np.linalg.det(gram):.3g}): {exc}. Report it, "
            f"with the model and the call.") from None
    frame = conv.T @ np.linalg.inv(standard.T)
    ortho = float(np.abs(frame @ frame.T - np.eye(3)).max())
    if ortho > 1e-6:                                         # pragma: no cover - defensive
        raise InternalError(
            f"{where}: the map from the standard orientation of the conventional cell "
            f"to this cell's is not orthogonal (defect {ortho:.3g}). Report it, with "
            f"the model and the call.")

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
            # The corrected lifts are the tables' own lift set, so the factor
            # system returned is the tables' only where that set closes.  It is
            # checked on the rotations the tables list the lifts under.
            cocycle, mixed, pairs, at = _lift_set_cocycle(
                [t.R for t in tsyms], theirs, signs)
            if cocycle > tol:
                raise InternalError(
                    f"{where}: the SU(2) matrices of the tables do not close. S(g)S(h) "
                    f"is not plus or minus S(gh) at the operations {at}: half the trace "
                    f"of S(g)S(h)S(gh)^-1 has a modulus that misses 1, or an imaginary "
                    f"part that misses 0, by {cocycle:.3g}, against a tolerance of "
                    f"{tol:.3g}. Report it, with the model and the call.")
            return {"frame": name, "correction": correction, "spin_transf": spin,
                    "signs": tuple(signs), "defect": defect, "cocycle": cocycle,
                    "mixed": int(mixed), "pairs": int(pairs)}
    report = "; ".join(
        f"{name!r} off by {defect:.3g} (first at operation #{bad[0] + 1}, "
        f"ratio {bad[1]:+.4f})" for name, defect, bad in tried)
    raise InternalError(
        f"{where}: none of the {len(tried)} frames tried reproduces this group's SU(2) "
        f"matrices within {tol:.3g}: {report}. Report it, with the model and the call.")


def _pin_spinor_branch(sg) -> dict | None:
    """The record of the SU(2) lifts of ``sg`` put in the tables' convention.

    ``sg`` is an ``irrep`` group, its operations in the tables' order.  A
    double group is changed in place: each operation's ``spinor_rotation`` is
    multiplied by the sign :func:`_canonical_spinor_signs` returns for it, its
    ``sign`` is set to ``+1``, and ``sg.spin_transf`` is set to the frame's
    SU(2) matrix.  The lifts then
    satisfy ``S(g) S(h) = ω(g, h) S(gh)`` with the tables' own ``ω``, so every
    character built on them is the tables' own.  Returns the record of
    :func:`_canonical_spinor_signs` without its two matrices, with ``"bars"``,
    the number of lifts negated, ``"lifts": "table"`` and ``"source":
    "table_frame"``.  Returns ``None`` for a spinless group, which is left as it
    is.
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
# the magnetic type, and the refusal of types III and IV
# --------------------------------------------------------------------------

def _magnetic_type(group) -> str:
    r"""The magnetic type, ``"I"`` to ``"IV"``, of a group's operations.

    ``group`` is anything that holds its antiunitary operations as
    ``_au_symmetries`` or ``au_symmetries``: a :class:`GroupData`, ``irrep``'s
    own space group before the record is built, or a :class:`_CosetView`.  The
    type is read off those operations, never off a group's number.  It is
    ``"I"`` where there are none.  It is ``"II"`` where one of them is
    ``{1|0}`` with time reversal.  A translation is read modulo the lattice of
    the cell handed in.  It is ``"IV"`` where one is ``{1|t}`` with time
    reversal, ``t`` not a lattice vector, and none is ``{1|0}``, and ``"III"``
    otherwise.
    """
    au = getattr(group, "_au_symmetries", None)
    if au is None:
        au = getattr(group, "au_symmetries", ())
    au = list(au or ())
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
    """A list of antiunitary operations in the form :func:`_magnetic_type` reads.

    ``operations`` are ``irrep`` operations combined with time reversal, such
    as those measured to hold, and are kept as ``au_symmetries``.
    """

    def __init__(self, operations):
        self.au_symmetries = list(operations)


def _magnetic_scope_refusal(group) -> Refusal | None:
    """The package's magnetic scope statement, or ``None`` when in scope.

    One gate, ``"magnetic_type_out_of_scope"``, shared by :func:`_group_of`,
    where it stops a type-III or type-IV group from being built at all, and by
    :func:`..ebr.symmetry_indicator` and :func:`..ebr.regular_content`, which
    keep their own check so that a group reaching them by any other route is
    still refused.
    """
    mtype = _magnetic_type(group)
    if mtype in ("I", "II"):
        return None
    number_str = str(getattr(group, "number_str", "?"))
    name = str(getattr(group, "name", "?"))
    au = getattr(group, "_au_symmetries", None)
    if au is None:
        au = getattr(group, "au_symmetries", ())
    au = list(au or ())
    kind = ("it has antiunitary operations, and none of them is time reversal alone "
            "or combined with a pure translation" if mtype == "III" else
            "it has an anti-translation, time reversal combined with a pure "
            "translation that is not a lattice vector, and not time reversal alone")
    return Refusal(
        "magnetic_type_out_of_scope",
        f"{name} #{number_str} is a magnetic group of type {mtype}: {kind}. pythTQC "
        "covers the magnetic groups of types I and II only.",
        context={"number_str": number_str, "name": name, "magnetic_type": mtype,
                 "n_antiunitary": len(au), "in_scope": ("I", "II"),
                 "record": "validated scope: magnetic types I and II"})


# --------------------------------------------------------------------------
# the lattice: the Hamiltonian sets the primitive cell
# --------------------------------------------------------------------------

def _pure_translations(symmetries, symprec, lattice) -> tuple:
    r"""The nonzero pure translations among the unitary operations, sorted.

    ``symmetries`` are ``irrep`` operations, their translations in reduced
    coordinates of the cell whose lattice vectors are the rows of
    ``lattice``.  The translation ``τ`` of each ``{1|τ}`` is reduced modulo
    the lattice to components in ``[-1/2, 1/2]``, and is zero where its
    Cartesian length is at most ``symprec``, in the units of ``lattice``.  A
    nonzero one says the point set repeats inside the cell, a supercell of
    its primitive cell.  Returns the nonzero ``τ``, each as three floats,
    sorted, so that equal sets compare equal.
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
    """A basis of the integer lattice the integer vectors ``rows`` span.

    ``rows`` is an ``m x 3`` integer array of rank 3.  Returns the basis as
    the rows of a ``3 x 3`` integer array, upper triangular with a positive
    diagonal, found by integer row reduction.  :func:`_primitive_lattice`
    reads with it the finer lattice a supercell's pure translations generate.

    Raises
    ------
    InternalError
        ``rows`` does not span three dimensions.
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
                "the lattice vectors of this cell and its pure translations do not "
                "span three dimensions. Report it, with the model and the call.")
        head = pivot[0] if pivot[0][col] > 0 else [-v for v in pivot[0]]
        basis.append(head)
        M = [r for r in M if r[col] == 0]
    return np.array(basis, dtype=np.int64)


def _primitive_lattice(lattice, translations, *, max_den=64) -> dict:
    r"""The finer lattice the cell's pure translations generate.

    ``lattice`` holds the cell's lattice vectors as rows, Cartesian, and
    ``translations`` the pure translations (:func:`_pure_translations`), in
    reduced coordinates of that cell.  The finer lattice is the set of sums
    of an integer vector and an integer combination of the translations.  Each
    translation is snapped to the nearest rational with denominator at most
    ``max_den``, so that its basis is found in integer arithmetic
    (:func:`_hermite_basis`).  Returns a mapping: ``"reduced"``, the basis
    vectors as rows in reduced coordinates of the cell; ``"cartesian"``, the
    same vectors in Cartesian coordinates; ``"index"``, how many points of
    the finer lattice the cell holds; and ``"denominator"``, the least common
    denominator of the translations' components.
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


def _require_group(call: str, obj) -> None:
    """Raise ``TypeError`` unless ``obj`` is a group with a cell, or a refusal.

    ``call`` is the public call's name, which the message opens with, and
    ``obj`` what that call was handed in place of a group.  A refusal passes,
    and the call returns it as it is.  The group of a space-group number
    (:func:`_group_of_number`), what :func:`.._realized.realized_group`
    returns and a representation each get a message of their own, which
    names what to pass.
    """
    if _is_number_group(obj):
        raise TypeError(f"{call}(): the group passed is read from a space-group number "
                        "and has no cell; pass r.group of what realized_group returns.")
    if isinstance(obj, (GroupData, Refusal)):
        return
    from ._realized import RealizedGroup
    from .symmetry import SpaceGroupRep
    if isinstance(obj, RealizedGroup):
        raise TypeError(f"{call}(): got what realized_group returns; pass its group, "
                        "r.group.")
    if isinstance(obj, SpaceGroupRep):
        raise TypeError(f"{call}(): got a representation; pass the group, r.group.")
    raise TypeError(f"{call}(): got an object of type {type(obj).__name__}; pass "
                    "r.group of what realized_group returns.")


def _group_of_number(sg: int, number_str: str, spinor: bool, name: str) -> GroupData:
    """The grey group of a space-group number, with no cell.

    ``sg`` is the space-group number, ``number_str`` the number of its grey
    group and ``name`` that group's symbol; ``spinor`` says whether it is the
    double group.  The record holds these and the magnetic type ``"II"``,
    with ``periodic_dirs`` ``None`` and no operations or cell.  It is the
    group the tables of the stable real-space invariants hold when they are
    read from a number.  A call that reads a cell raises ``TypeError`` when
    handed it (:func:`_require_group`).
    """
    return GroupData(
        name=str(name), number_str=str(number_str), spinor=bool(spinor),
        magnetic_type="II", periodic_dirs=None, _sg_number=int(sg), _magnetic=True,
        _include_TR=True, _symmetries=[], _lattice=None, _site_positions=None,
        _site_orbitals=(), _typat=None, _vacuum_dirs=(), _vacuum_length=0.0,
        _refUC=None, _shiftUC=None, _tables_ready=False, _symprec=0.0, _dim_k=3,
        _dim_r=3, _provenance="number")


def _is_number_group(obj) -> bool:
    """Whether ``obj`` is the group of a space-group number, which has no cell.

    That group is the one :func:`_group_of_number` returns.
    """
    return isinstance(obj, GroupData) and obj._provenance == "number"


def _lattice_not_descended(group) -> Refusal | None:
    r"""The refusal of a table call handed the group of a supercell.

    ``group`` is a group, or a refusal handed in where a group belongs.  A
    group whose ``_lattice_index`` is above 1 is the group of a cell that
    repeats under a pure translation.  Its operations are the cell's.  Its
    tables are written on the finer lattice, and read on this cell their
    labels would name the k-points of a folded zone.  Returns
    ``lattice_not_descended`` for such a group, naming its first fractional
    translation, and ``None`` for any other.  A refusal handed in is returned
    as it is, so that each table call that applies this first returns the
    refusal it was handed.
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
        f"{group.name} #{group.number_str} was identified on a cell whose sites repeat "
        f"under the pure translation {tau.tolist()}, in reduced coordinates, so "
        f"the model's lattice is a sublattice of index {index} of the group's. Use what "
        "realized_group returns for the model.",
        value=float(index), threshold=1.0,
        context={"number_str": str(group.number_str), "name": str(group.name),
                 "lattice_index": index, "translations": taus, "group": group})


# --------------------------------------------------------------------------
# the operations themselves: do they close, and what to do when they do not
# --------------------------------------------------------------------------

def _translation_closure(symmetries) -> float:
    r"""How far the operations miss closing into a group, modulo the lattice.

    ``symmetries`` are ``irrep`` operations, their translations in reduced
    coordinates.  For each pair ``g``, ``h`` the product is found among the
    operations by its rotation and whether it is antiunitary, and the miss is
    ``t_g + R_g t_h − t_gh`` modulo the lattice, its largest component.
    Where several operations share that rotation and antiunitarity, as on
    the cell of a lattice supergroup, the miss is to the nearest of them.
    Returns the largest miss over every pair: ``0.0`` for operations that
    close exactly, and ``inf`` where the rotations do not close or a product
    has no operation to match.
    """
    n = len(symmetries)
    R = np.array([np.asarray(s.rotation, dtype=np.int64) for s in symmetries])
    t = np.array([np.asarray(s.translation, dtype=float) for s in symmetries])
    tr = np.array([bool(s.time_reversal) for s in symmetries])

    # the distinct rotations, at most 48 however large the supercell, so that
    # the product's rotation is found in a table of that size
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


def _closure_gate(symmetries) -> float:
    r"""The closure at or below which a cell's operations close to rounding.

    ``symmetries`` are ``irrep`` operations, their translations in reduced
    coordinates.  Returns :data:`_EXACT_CLOSURE` times the largest row sum
    of any operation's ``|R|``, and at least :data:`_EXACT_CLOSURE`, in
    reduced coordinates.  The closure ``t_g + R_g t_h − t_gh`` multiplies the
    rounding of ``t_h``, and any miss of the positions, by the entries of
    ``R_g``, which reach hundreds on a strongly skewed lattice basis.  The
    row sum is 1 on a cell whose rotations are signed permutations.
    """
    size = max((float(np.abs(np.rint(np.asarray(s.rotation, dtype=float))).sum(axis=1).max())
                for s in symmetries), default=1.0)
    return float(_EXACT_CLOSURE) * max(size, 1.0)


def _site_miss(symmetries, site_pos, species) -> float:
    r"""The most any site misses the image of a site of its species.

    ``symmetries`` are ``irrep`` operations, ``site_pos`` the sites in
    reduced coordinates of the same cell, and ``species`` each site's
    species.  The miss of site ``a`` under ``g`` is ``R_g x_a + t_g − x_b``
    modulo the lattice, its largest component, ``x_b`` the site of ``a``'s
    species nearest the image.  Returns the largest miss over every operation
    and site, in reduced coordinates.  Where the operations carry the sites
    onto one another one for one, the closure :func:`_translation_closure`
    measures is at most ``2 + r`` times it, ``r`` the largest row sum of the
    rotations' ``|R|`` (:func:`_closure_gate`).
    """
    pos = np.asarray(site_pos, dtype=float).reshape(-1, 3)
    kind = np.asarray([str(s) for s in species])
    same = kind[:, None] == kind[None, :]
    worst = 0.0
    for s in symmetries:
        image = pos @ np.asarray(s.rotation, dtype=float).T + np.asarray(s.translation, dtype=float)
        d = image[:, None, :] - pos[None, :, :]
        d -= np.rint(d)
        d = np.where(same, np.abs(d).max(axis=-1), np.inf)
        worst = max(worst, float(d.min(axis=1).max()))
    return worst


def _rebuilt_translation(refUC, t_ref, shift, R, t_old) -> np.ndarray:
    r"""An operation's translation, rebuilt from the tabulated one.

    ``refUC`` and ``shift``, the origin shift ``s`` (``shiftUC``), carry the
    tables' conventional cell into the cell, ``x = refUC·x_ref + s``.  ``R``
    is the operation's rotation, ``t_ref`` the tabulated translation as an
    exact fraction, and ``t_old`` the translation found on the cell.  Returns
    ``refUC·t_ref + s − R s``, formed exactly in rational arithmetic on the
    stored numbers and rounded once, plus the lattice vector that puts it
    nearest ``t_old``.  Every vector is in reduced coordinates of the cell.
    """
    exact = [Fraction(float(x)) for x in np.ravel(shift)]
    out = []
    for i in range(len(exact)):
        v = exact[i]
        for j in range(len(exact)):
            v += Fraction(float(refUC[i, j])) * Fraction(float(t_ref[j]))
            v -= Fraction(float(R[i, j])) * exact[j]
        out.append(v)
    return np.array([float(v + round(float(t_old[i]) - float(v))) for i, v in enumerate(out)])


def _exact_operations(sg, symprec, tables_ready, lattice) -> tuple | Refusal:
    r"""The cell's operations made to close exactly, where they do not.

    ``sg`` is ``irrep``'s group of the cell, whose operations' translations
    are rebuilt in place.  ``tables_ready`` says whether ``irrep`` matched the
    cell to the tables' setting.  ``lattice`` holds the cell's vectors as
    rows, in the units of ``symprec``, the symmetry finder's distance.  Where
    the closure (:func:`_translation_closure`) is ``0.0``, as on every cell
    built from exact numbers, nothing changes.  Otherwise, on a cell matched
    to the tables' setting, each translation is rebuilt
    (:func:`_rebuilt_translation`) from the tabulated operation with the same
    rotation in the tables' conventional cell and the same antiunitarity, its
    translation snapped to the rational it is.  A cell not matched is left as
    it is where its closure is at most :func:`_closure_gate`, and refused
    otherwise.  The rotations are never changed.

    Returns ``(closure_measured, closure_repaired)``: the closure measured,
    and the largest change of a component of a translation, ``0.0`` where
    nothing was rebuilt, both in reduced coordinates.

    Refusals
    --------
    ``operations_not_consistent``
        The cell is not matched to the tables' setting and its closure
        exceeds :func:`_closure_gate`.
    ``operations_not_nearly_consistent``
        A rebuilt translation lies farther than ``symprec`` from the one
        found, as a Cartesian length.

    Raises
    ------
    InternalError
        The tabulated operations cannot be matched to the cell's one for one,
        or the rebuilt ones still do not close.
    """
    before = _translation_closure(sg.symmetries)
    if before == 0.0:
        return 0.0, 0.0
    gate = _closure_gate(sg.symmetries)
    if not tables_ready:
        if before <= gate:
            # binary floating point, not a defect in the cell: a supercell whose
            # translations are exact thirds cannot close to a bit-exact zero, and
            # there is nothing here to repair.  See `_closure_gate`.
            return float(before), 0.0
        return Refusal(
            "operations_not_consistent",
            f"the operations the symmetry finder reads off the orbital centers miss "
            f"closing into a group by {before:.3g}, in reduced coordinates, where a cell "
            f"built from exact numbers misses by at most {gate:.3g}. Pass a symprec= "
            "smaller than the distance between the sites of one species that are "
            "closest together.",
            value=float(before), threshold=float(gate),
            context={"number_str": str(sg.number_str), "closure": float(before)})

    table = _irrep_table(sg.number_str, sg.spinor, sg.magnetic)
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
            f"{sg.name} #{sg.number_str}: the cell has {len(sg.symmetries)} operations, "
            f"and the table {len(tsyms)} with {len(tabulated)} distinct rotations. "
            "Report it, with the model and the call.")

    lat = np.asarray(lattice, dtype=float)
    rebuilt, defect, moved = [], 0.0, 0.0
    for s in sg.symmetries:
        R = np.asarray(s.rotation, dtype=float)
        Rref = inv @ R @ refUC
        if float(np.abs(Rref - np.rint(Rref)).max()) > 1e-6:
            raise InternalError(                     # pragma: no cover - guard
                f"{sg.name} #{sg.number_str}: the rotation of operation #{s.ind}, "
                "written in the conventional cell of the tables, is not an integer "
                f"matrix (defect {float(np.abs(Rref - np.rint(Rref)).max()):.3g}). "
                "Report it, with the model and the call.")
        key = (np.rint(Rref).astype(int).tobytes(), bool(s.time_reversal))
        if key not in tabulated:                       # pragma: no cover - guard
            raise InternalError(
                f"{sg.name} #{sg.number_str}: the rotation of operation #{s.ind}, "
                "written in the conventional cell of the tables, matches no operation of "
                "the table in rotation and antiunitarity. Report it, with the model and "
                "the call.")
        t_old = np.asarray(s.translation, dtype=float)
        t_new = _rebuilt_translation(refUC, tabulated[key], shiftUC, R, t_old)
        defect = max(defect, float(np.abs(t_new - t_old).max()))
        # the same move as a length, in the units the finder's distance is in
        moved = max(moved, float(np.linalg.norm((t_new - t_old) @ lat)))
        rebuilt.append(t_new)

    if moved > float(symprec):
        return Refusal(
            "operations_not_nearly_consistent",
            f"the operations the symmetry finder reads off the orbital centers miss "
            f"closing into a group by {before:.3g}, in reduced coordinates, and "
            f"rebuilding their translations from the tables' operations would move one "
            f"by {moved:.3g} in the model's units of length, more than symprec = "
            f"{float(symprec):.3g}. Declare "
            "the model's precision with precision=, pass a symprec= the centers "
            "support, or say what the crystal is with typat= or decorations=.",
            value=float(moved), threshold=float(symprec),
            context={"number_str": str(sg.number_str), "closure": float(before)})

    for s, t_new in zip(sg.symmetries, rebuilt):
        s.translation = t_new                          # the rotations are untouched
    after = _translation_closure(sg.symmetries)
    if after > gate:
        raise InternalError(                         # pragma: no cover - guard
            f"{sg.name} #{sg.number_str}: the operations rebuilt from the tables miss "
            f"closing into a group by {after:.3g}, in reduced coordinates, more than "
            f"{gate:.3g}. Report it, with the model and the call.")
    return float(before), float(defect)


#: The identification memo: a bounded, process-level LRU of :func:`_group_of`
#: records, keyed by the cell it was asked about.  One call identifies the same
#: cell several times: for the cell, again where a descent identifies the cell
#: again, and again for a layer's other environment.  A hit never changes what
#: is returned: it hands back the object the first call built.
_IDENT_MEMO: "OrderedDict[tuple, GroupData]" = OrderedDict()

#: How many identifications the memo keeps.  Each holds ``irrep``'s whole
#: ``SpaceGroupIrreps`` object.
_IDENT_MEMO_MAX = 64

_IDENT_MEMO_STATS = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}

#: The vacuum a layer's cell is given before the symmetry finder reads it,
#: in the model's length units (:func:`_pad_cell`): geometry, read by no
#: judgment.
_PAD_LENGTH = 20.0

#: The denominator every translation of a crystal's operations is a multiple of,
#: in the model's own reduced coordinates, about an origin the tables can use:
#: halves, thirds, quarters, sixths and eighths of a lattice vector, which 24
#: holds.  A cell that repeats inside itself adds its pure translations' own.
#: Exact arithmetic, read by no judgment of noise.
_TRANSLATION_DENOMINATOR = 24


def _identification_memo_stats() -> dict:
    """The memo's counters and its size, in a new mapping.

    Returns ``"hits"``, ``"misses"``, ``"stores"`` and ``"evictions"``, with
    ``"size"``, the identifications held, and ``"maxsize"``,
    :data:`_IDENT_MEMO_MAX`.
    """
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


@lru_cache(maxsize=None)
def _grey_symbols() -> dict:
    """``irrep``'s number and symbol for the grey group of each space group.

    Returns a mapping from each space-group number ``n`` to
    ``(number_str, name)``: the BNS number of its grey group, ``"99.164"``,
    and the group's symbol, ``"P4mm1'"``.  Each is read from the line of
    ``irrep``'s ``data/msg_numbers.data`` indexed by the UNI number spglib
    gives that magnetic space group of type II, the line ``from_cell`` reads.

    Raises
    ------
    InternalError
        The two do not give one grey group to each of the 230 space groups.
    """
    import spglib
    from irrep import spacegroup as irrep_spacegroup

    path = os.path.join(os.path.dirname(irrep_spacegroup.__file__), "data", "msg_numbers.data")
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    symbols = {}
    with warnings.catch_warnings():
        # spglib warns once per call about its own error handling
        warnings.simplefilter("ignore", DeprecationWarning)
        types = [spglib.get_magnetic_spacegroup_type(u) for u in range(1, 1652)]
    for uni_number, entry in enumerate(types, start=1):
        if isinstance(entry, dict):                   # spglib before 2.5
            kind, number = entry["type"], entry["number"]
        else:
            kind, number = entry.type, entry.number
        if int(kind) == 2:
            number_str, name = lines[uni_number].strip().split(" ")
            symbols[int(number)] = (number_str, name)
    if sorted(symbols) != list(range(1, 231)) or any(
            s.split(".")[0] != str(n) for n, (s, _name) in symbols.items()):
        raise InternalError(
            "the installed spglib and irrep do not give one grey group to each of the "
            "230 space groups. Report it, with the versions of spglib and irrep.")
    return symbols


#: What ``SpaceGroup.from_cell`` raises, as a ``ValueError``, when spglib's
#: magnetic routine returns no dataset for the cell.
_NO_MAGNETIC_DATASET = "No magnetic space group could be detected!"


def _irrep_group(**kwargs):
    r"""``irrep``'s group of a cell, as ``from_cell`` reads it.

    ``kwargs`` are those of ``SpaceGroupIrreps.from_cell``, and a cell
    ``from_cell`` reads is returned as it reads it.  On some cells, most
    written on a lattice basis that is not reduced, spglib's magnetic routine
    finds no group and ``from_cell`` raises ``ValueError`` with
    :data:`_NO_MAGNETIC_DATASET`.  For a cell asked with time reversal and no
    nonzero ``magmom``, the group is then the grey group of the space group
    spglib's ordinary routine finds: each of its operations alone and with
    time reversal, its transformation to the standard setting and origin
    shift with every entry at most :data:`_EXACT_CLOSURE` set to zero, the
    number and symbol from :func:`_grey_symbols`, and the tables set as
    ``from_cell`` sets them.  Every other error of ``from_cell`` propagates,
    and that one too for a cell with a nonzero moment or asked without time
    reversal.  Returns the ``SpaceGroupIrreps``.

    Raises
    ------
    InternalError
        spglib's ordinary routine finds no group either.
    """
    from irrep.spacegroup_irreps import SpaceGroupIrreps

    try:
        return SpaceGroupIrreps.from_cell(**kwargs)
    except ValueError as exc:
        magmom = kwargs.get("magmom")
        if (str(exc) != _NO_MAGNETIC_DATASET or not kwargs.get("include_TR", True)
                or (magmom is not None and np.any(np.asarray(magmom, dtype=float)))):
            raise

    # spglib's magnetic routine returned nothing for a cell with no moments:
    # its group is the grey group of the space group the ordinary routine reads.
    import spglib

    tables = dict(kwargs)
    lattice = np.array(tables.pop("real_lattice"), dtype=float)
    positions = np.array(tables.pop("positions"), dtype=float)
    typat = np.array(tables.pop("typat"), dtype=int)
    spinor = tables.pop("spinor")
    symprec = tables.pop("symprec")
    for name in ("include_TR", "magmom"):
        tables.pop(name, None)

    dataset = spglib.get_symmetry_dataset((lattice, positions, typat),
                                          symprec=symprec, angle_tolerance=-1)
    if dataset is None:
        raise InternalError(
            f"the symmetry finder, spglib, found no space group for a cell of "
            f"{len(positions)} sites at symprec = {symprec:g}. Report it, with the model "
            f"and the call.")
    if isinstance(dataset, dict):                     # spglib before 2.5
        fields = (dataset["rotations"], dataset["translations"], dataset["number"],
                  dataset["transformation_matrix"], dataset["origin_shift"])
    else:
        fields = (dataset.rotations, dataset.translations, dataset.number,
                  dataset.transformation_matrix, dataset.origin_shift)
    rotations, translations, number, ref_uc, shift_uc = fields
    # An entry of the transformation or the origin shift within rounding of
    # zero is set to zero: its sign would otherwise carry a tabulated point of
    # a layer's plane, k_z = 0, to k_z = 1 in the model's cell.  spglib returns
    # such an entry as 1e-16 or so, and as 1e-14 on a strongly skewed basis; the
    # entries that are not zero are simple fractions.
    ref_uc, shift_uc = (np.where(np.abs(x) <= _EXACT_CLOSURE, 0.0, x)
                        for x in (np.asarray(ref_uc, dtype=float),
                                  np.asarray(shift_uc, dtype=float)))
    number_str, name = _grey_symbols()[int(number)]
    count = len(rotations)
    sg = SpaceGroupIrreps(
        Lattice=lattice, spinor=spinor,
        rotations=np.repeat(np.asarray(rotations), 2, axis=0),
        translations=np.repeat(np.asarray(translations, dtype=float), 2, axis=0),
        time_reversals=[False, True] * count, translation_mod1=True,
        number_str=number_str, name=name, spinor_rotations=None,
        refUC=ref_uc, shiftUC=shift_uc, magnetic=True, positions=positions,
        typat=typat, alat=None, magmom=np.zeros((len(positions), 3), dtype=float),
        verbosity=0)
    sg.set_irreptables(verbosity=0, **tables)
    return sg


def _setting_search(kwargs) -> tuple:
    r"""``irrep``'s group of a cell in the tables' setting, from fixed seeds.

    ``kwargs`` are ``SpaceGroupIrreps.from_cell``'s own, asked with
    ``search_cell=True`` through :func:`_irrep_group`.  The search is asked
    with ``numpy``'s global random state set to each of
    :data:`_SETTING_SEEDS` in turn, until one matches the cell to the tables'
    standard setting.  A search that fails without drawing from that state
    fails the same way from every seed, and is not asked again.  The state
    the call arrived with is restored afterwards.  Returns ``(sg, None, n)``
    from the first search that succeeds, or ``(None, error, n)`` with the
    ``RuntimeError`` the last one raised, ``n`` the number of searches asked.
    An ``InternalError`` propagates, as does any error that is not a
    ``RuntimeError``.
    """
    state = np.random.get_state()
    error, asked = None, 0
    try:
        for seed in _SETTING_SEEDS:
            np.random.seed(int(seed))
            fresh = np.random.get_state()
            asked += 1
            try:
                return _irrep_group(search_cell=True, **kwargs), None, asked
            except InternalError:
                raise
            except RuntimeError as exc:
                error = exc
            now = np.random.get_state()
            if now[2] == fresh[2] and np.array_equal(now[1], fresh[1]):
                break                    # nothing was drawn: no seed changes this
    finally:
        np.random.set_state(state)
    return None, error, asked


def _origin_nearest(sg, lattice, vacuum_dirs, symprec):
    r"""The tables' origin nearest the model's own, if ``irrep`` read another.

    ``sg`` is ``irrep``'s group of the cell in the tables' setting, its
    origin at ``shiftUC`` in reduced coordinates.  ``lattice`` holds the
    cell's vectors as rows, and ``vacuum_dirs`` the directions along which it
    is not periodic.  The tables can be read about every ``shiftUC + v`` with
    ``(R − 1)·v`` a lattice vector for each rotation ``R``, since each
    operation keeps its translation modulo the lattice there.  ``v`` is taken
    on the grid of :data:`_TRANSLATION_DENOMINATOR` along the periodic
    directions.  The Wyckoff letters and the little-group representations
    named at the k-points can differ between these origins.  The distance to
    the model's origin is Cartesian, modulo the lattice, along the periodic
    directions, with the directions every rotation fixes left out.  Points
    within ``symprec``, the symmetry finder's distance, of the nearest are
    equally near.  Returns ``None``
    where ``irrep``'s own origin is among them, and otherwise the
    ``shiftUC`` of the first of them on the grid.
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

    ``lattice``, ``vacuum_dirs`` and ``vacuum_length`` are the cell and its
    vacuum (:func:`_pad_cell`); ``site_pos``, ``site_orbs`` and ``species``
    are its sites as :func:`_sites_and_typat` groups them; ``symprec``,
    ``spinor``, ``include_TR`` and ``magmom`` are what the finder is asked;
    ``periodic_dirs``, ``dim_k`` and ``dim_r`` are the model's.  An array
    enters as its exact bytes, so that a cell that differs in the last bit
    is another key.  Returns the key.
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
    """The space group the cell allows, read from its points alone.

    It is an upper bound on the group a Hamiltonian on the cell obeys, which
    :func:`.._realized.realized_group` measures.  The cell is periodic in two
    or three directions; a layer's is its slab presentation
    (:func:`.._layers._slab_presentation`), set in vacuum by :func:`_pad_cell`.
    The tables' setting is searched from fixed seeds (:func:`_setting_search`)
    and read about the origin nearest the model's own
    (:func:`_origin_nearest`).  Each identification is kept in a memo of
    :data:`_IDENT_MEMO_MAX` entries, keyed by :func:`_identification_key`,
    and a cell asked about again returns the record first built.

    Parameters
    ----------
    model : pythtb.TBModel
        The model, or a cell built on one by
        :func:`.._layers._decorated_cell`.
    spinor : bool or None, default None
        Whether the double group is asked for.  ``None`` reads it from
        ``model.nspin``.
    include_TR : bool, default True
        Whether time reversal is adjoined.  Without a nonzero ``magmom``, the
        group is then grey.
    symprec : float, default 1e-5
        The symmetry finder's distance, in the model's units of length, at
        which points are grouped into sites and translations compared.
    typat : sequence of int, default None
        Each point's label (:func:`_sites_and_typat`).
    magmom : array_like, default None
        Each site's magnetic moment, as ``from_cell`` takes it.
    vacuum_length : float, default 20.0
        The vacuum given a direction that is not periodic, in the model's
        units of length (:func:`_pad_cell`).

    Returns
    -------
    GroupData or Refusal
        The group, its ``_provenance`` ``"cell"``.  Where no search matches
        the cell to the tables' setting, the group has no tables and its
        ``_setup_note`` carries ``irrep``'s message.  For a cell that repeats
        under a pure translation, the group returned is its whole lattice
        supergroup, without tables, its ``_lattice_index`` above 1.

    Refusals
    --------
    ``irrep_tables``
        ``irrep``'s character tables are not loaded (:func:`tables_status`).
    ``single_valued_spinful``
        ``spinor=False`` on a model with ``nspin == 2``.
    ``magnetic_type_out_of_scope``
        The operations found are of magnetic type III or IV, which a nonzero
        ``magmom`` can give.
    ``operations_not_nearly_consistent`` / ``operations_not_consistent``
        The operations do not close and cannot be rebuilt from the tabulated
        ones (:func:`_exact_operations`).
    """
    status = tables_status()
    if not status["tables"]:
        return _refuse_no_tables(status)

    if spinor is None:
        spinor = int(model.nspin) == 2

    # Since `spinor` left out is read from `nspin`, this fires only on an
    # explicit `spinor=False` for a model whose states carry spin one-half,
    # which has no single-valued representation.  `_build._build_rep` refuses
    # the same pair.
    if int(model.nspin) == 2 and not bool(spinor):
        return Refusal(
            "single_valued_spinful",
            "spinor=False is passed for a model with spin, nspin=2. Leave spinor= out, "
            "and it is read from the model.",
            context={"nspin": int(model.nspin), "spinor": bool(spinor),
                     "spinful": bool(getattr(model, "spinful", int(model.nspin) == 2))})

    lattice, positions, vacuum_dirs = _pad_cell(model, float(vacuum_length))
    site_pos, site_orbs, species = _sites_and_typat(positions, typat, symprec, lattice)

    # The identification memo.  Everything below, from spglib to the closure
    # repair, depends on the cell above and the flags alone.  A hit returns
    # the object the first call built, its own memos with it, so that the
    # product table (`GroupData._get_product_table`) is built once a cell.
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
    # The tables' setting, searched from fixed seeds and asked again where a
    # search fails: irrep's spin-basis fit starts from a random point, and a
    # start it cannot converge from is no statement about the cell.  Only where
    # every seed fails is the group returned without tables, and then its note
    # carries irrep's own message, the reason every `tables_not_ready` gives.
    sg, failure, asked = _setting_search(kwargs)
    tables_ready = sg is not None
    if sg is None:
        sg = _irrep_group(search_cell=False, **kwargs)
        note = ("irrep could not match this cell to the BCS standard setting"
                + (f" in {asked} searches from fixed seeds" if asked > 1 else "")
                + f", so irrep tables are unavailable for it: {failure}")

    # The lattice.  A pure fractional translation says the point set repeats
    # inside the cell, and what the cell allows is the whole lattice
    # supergroup, every operation of it, coset partners and translations
    # included.  It is returned without tables, which are written on the finer
    # lattice and would label a folded zone here.  A cell read with the
    # setting search is read again without it, so that the record holds no
    # setting the search matched to the tables.
    translations = _pure_translations(sg.symmetries, symprec, lattice)
    if translations:
        if tables_ready:
            sg = _irrep_group(search_cell=False, **kwargs)
            translations = _pure_translations(sg.symmetries, symprec, lattice)
            tables_ready = False
        note = (
            f"this cell is a supercell: its point set allows the pure translation "
            f"{np.round(np.asarray(translations[0], dtype=float), 6).tolist()} "
            f"(reduced), so the operations here are the lattice supergroup of index "
            f"{len(translations) + 1} and the tables live on the finer lattice.")

    # The tables' origin.  The same tables are read about any point the
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

    # irrep fits the spin-basis transformation from a random start, and both
    # branches of that fit are exact minima; replace it with the frame the
    # tables' own SU(2) matrices are written in (see `_canonical_spinor_signs`).
    # Without tables there is nothing to match against and no sign was ever fitted.
    spin_frame = _pin_spinor_branch(sg) if tables_ready else None

    out_of_scope = _magnetic_scope_refusal(sg)
    if out_of_scope is not None:
        return out_of_scope

    # Whether the operations close into a group.  On a cell built from exact
    # numbers they close exactly and nothing is changed.  On a numerical cell
    # spglib reads each translation off noisy atom pairs and they close only
    # to that noise, so they are rebuilt from the tabulated ones.  See
    # `_exact_operations`.
    consistent = _exact_operations(sg, symprec, tables_ready, lattice)
    if isinstance(consistent, Refusal):
        if consistent.gate == "operations_not_consistent":
            # how far the sites miss these operations, which bounds the closure:
            # read where it is known whether a precision was declared
            consistent = replace(consistent, context=dict(
                consistent.context, name=str(sg.name),
                site_miss=_site_miss(sg.symmetries, site_pos, species),
                rotation_size=_closure_gate(sg.symmetries) / float(_EXACT_CLOSURE)))
        return consistent
    closure_measured, closure_repaired = consistent

    number_str = str(sg.number_str)
    try:
        sg_number = int(number_str.split(".")[0])
    except ValueError:  # pragma: no cover - defensive
        sg_number = -1

    out = GroupData(
        _sg_number=sg_number, number_str=number_str,
        name=str(sg.name), spinor=bool(sg.spinor), magnetic_type=_magnetic_type(sg),
        _magnetic=bool(sg.magnetic),
        _include_TR=bool(include_TR), _symmetries=list(sg.symmetries),
        _lattice=lattice, _site_positions=site_pos, _site_orbitals=site_orbs,
        _typat=species, periodic_dirs=tuple(int(d) for d in model.periodic_dirs),
        _vacuum_dirs=vacuum_dirs, _vacuum_length=float(vacuum_length),
        _refUC=np.array(sg.refUC, dtype=float), _shiftUC=np.array(sg.shiftUC, dtype=float),
        _tables_ready=tables_ready, _symprec=float(symprec),
        _dim_k=int(model.lattice.dim_k), _dim_r=int(model.lattice.dim_r),
        _setup_note=note, _provenance="cell", _spin_frame=spin_frame, _sg=sg,
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
# Wyckoff positions in spglib's setting
# --------------------------------------------------------------------------

def _wyckoff_positions_spglib(group: GroupData) -> tuple | Refusal:
    """Every Wyckoff position of the group, lettered in spglib's setting.

    ``group`` is a group, or a refusal handed in.  The positions are those of
    the space group spglib finds on the group's cell and sites at its default
    tolerance, as ``irrep`` writes them from ``pyxtal``'s list.  Returns each
    as a :class:`_WyckoffPosition`, in reduced coordinates of the group's
    three-dimensional cell, the general position first and ``a`` last.  Each
    letter is spglib's, and the tables' letter for the position can differ
    (:func:`.._tables.wyckoff_sites_tables` gives the tables').

    Refusals
    --------
    ``lattice_not_descended``
        The group is that of a supercell; a refusal handed in is returned as
        it is (:func:`_lattice_not_descended`).
    ``wyckoff_enumeration``
        ``irrep`` raises while it writes the positions.
    """
    import sympy
    from irrep.spacegroup import SpaceGroup as _SG

    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    cell = (group._lattice, group._site_positions, np.asarray(group._typat, dtype=int))
    try:
        exprs = _SG.wyckoff_positions(cell)
    except Exception as exc:  # pragma: no cover - pyxtal/spglib disagreement
        return Refusal("wyckoff_enumeration",
                       f"irrep could not list the Wyckoff positions of {group.name} "
                       f"#{group.number_str}: {exc}",
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
        out.append(_WyckoffPosition(letter=letter, expression=s, offset=b, jacobian=A,
                                   n_free=int(np.linalg.matrix_rank(A, tol=1e-9))))
    return tuple(out)


# --------------------------------------------------------------------------
# Wyckoff letters in the tables' setting
# --------------------------------------------------------------------------



@lru_cache(maxsize=64)
def _pyxtal_wyckoff(sg: int) -> tuple:
    r"""``pyxtal``'s Wyckoff positions of a group, in the tables' setting.

    ``sg`` is the space-group number.  The list is ``pyxtal``'s in its own
    conventional setting, which :func:`_wyckoff_setting_audit` measures to be
    the tables'.  Returns ``(positions, centering)``.  Each position is a
    mapping with ``"letter"``, ``"multiplicity"``, ``"n_free"`` and
    ``"arms"``.  Its arms are one for each of ``pyxtal``'s operations of the
    position: the affine map :math:`x = A p + b` and the projector
    :math:`P = 1 - A A^+` onto the complement of its directions, :math:`A^+`
    the pseudoinverse.  ``centering`` holds the conventional cell's
    centering translations, the zero vector among them: the translations of
    the general position's operations whose rotation is the identity.
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
    """How far each point lies off the variety of a Wyckoff position.

    ``arms`` are the position's arms (:func:`_pyxtal_wyckoff`), and
    ``points`` one point or an array of shape ``(n, 3)``, in reduced
    coordinates of the tables' conventional cell.  Returns, for each point,
    the least over the arms and over the lattice vectors ``n`` with
    components in ``{-1, 0, 1}`` of the largest component of
    ``P (x + n − b)``: zero on the variety.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    best = np.full(points.shape[0], np.inf)
    for _A, b, P in arms:
        # (n_point, 27, 3): every point, shifted by every neighboring lattice
        # vector, projected onto the complement of this arm's variety
        delta = (points[:, None, :] - b[None, None, :] + _NEIGHBOR_SHIFTS[None, :, :])
        resid = np.abs(delta @ P.T).max(axis=2).min(axis=1)
        best = np.minimum(best, resid)
    return best


def _variety_contains_variety(inner, outer, tol) -> bool:
    """Whether one arm's subspace lies inside another's, modulo the lattice.

    ``inner`` and ``outer`` are arms ``(A, b, P)`` (:func:`_pyxtal_wyckoff`).
    Returns ``True`` where every entry of ``P_out A_in`` is at most ``tol``,
    and every component of ``P_out (b_in − b_out + n)`` is too for some
    lattice vector ``n`` with components in ``{-1, 0, 1}``.
    """
    A_in, b_in, _ = inner
    _A_out, b_out, P_out = outer
    if float(np.abs(P_out @ A_in).max()) > tol:
        return False                      # the directions do not fit
    delta = b_in - b_out + _NEIGHBOR_SHIFTS
    return bool(np.abs(delta @ P_out.T).max(axis=1).min() <= tol)


def _in_tables_cell(group: GroupData, points) -> np.ndarray:
    r"""Points of the model's cell, written in the tables' conventional cell.

    ``points`` is one point or an array of shape ``(n, 3)``, in reduced
    coordinates of ``group``'s three-dimensional cell.  Each is carried by
    :math:`x_\text{ref} = \mathrm{refUC}^{-1}(x - \mathrm{shiftUC}) \bmod 1`
    through the group's stored ``refUC`` and ``shiftUC``, which every Wyckoff
    letter and every name read at a site goes through, and a coordinate
    within rounding of 1 is written 0.  Returns an array of shape ``(n, 3)``.
    """
    raw = np.atleast_2d(np.asarray(points, dtype=float))
    inv = np.linalg.inv(np.asarray(group._refUC, dtype=float))
    shift = np.asarray(group._shiftUC, dtype=float)
    reference = np.array([(inv @ (x - shift)) % 1.0 for x in raw])
    return np.where(np.abs(reference - 1.0) < 1e-9, 0.0, reference)


def _reference_operations(group: GroupData) -> tuple:
    r"""The group's unitary operations, in the tables' conventional cell.

    ``group`` is a group matched to the tables' setting.  Returns
    ``(R_ref, t_ref)`` for each operation, with
    :math:`R_\text{ref} = \mathrm{refUC}^{-1} R\,\mathrm{refUC}`, an integer
    matrix, and :math:`t_\text{ref} = \mathrm{refUC}^{-1}(t + R s - s)`,
    ``s = shiftUC``, as ``irrep``'s
    ``SymmetryOperation.translation_refUC`` writes it.  Each translation is
    snapped to the nearest rational with a small denominator
    (:func:`_snap_rational`).

    Raises
    ------
    InternalError
        Some ``R_ref`` is not an integer matrix.
    """
    refUC = np.asarray(group._refUC, dtype=float)
    shift = np.asarray(group._shiftUC, dtype=float)
    inv = np.linalg.inv(refUC)
    out = []
    for sym in group._symmetries:
        if bool(sym.time_reversal):
            continue
        R = np.asarray(sym.rotation, dtype=float)
        R_ref = inv @ R @ refUC
        if float(np.abs(R_ref - np.rint(R_ref)).max()) > 1e-6:
            raise InternalError(
                f"{group.name} #{group.number_str}: the rotation of operation "
                f"#{sym.ind}, written in the conventional cell of the tables, is not an "
                f"integer matrix (defect {float(np.abs(R_ref - np.rint(R_ref)).max()):.3g}). "
                "Report it, with the model and the call.")
        t_ref = inv @ (np.asarray(sym.translation, dtype=float)
                       + R @ shift - shift)
        out.append((np.rint(R_ref).astype(int), _snap_rational(t_ref)))
    return tuple(out)


def _stabilizer_order(point, ops_ref, centering, tol=1e-6) -> int:
    """How many operations of the conventional cell fix ``point``.

    ``point`` is in reduced coordinates of the tables' conventional cell,
    ``ops_ref`` holds the group's operations there
    (:func:`_reference_operations`), and ``centering`` the cell's centering
    translations, the zero vector among them.  Returns the number of pairs
    of an operation and a centering translation that carry ``point`` onto
    itself modulo the lattice, each component within ``tol``.
    """
    count = 0
    for R, t in ops_ref:
        image = R @ point + t
        delta = image[None, :] + centering - point[None, :]
        count += int((np.abs(delta - np.rint(delta)).max(axis=1) <= tol).sum())
    return count


def _wyckoff_setting_audit(sg: int, ops_ref, ebr_positions) -> dict:
    r"""Whether ``pyxtal``'s Wyckoff list is written in the tables' setting.

    ``sg`` is the space-group number, ``ops_ref`` the group's operations in
    the tables' conventional cell (:func:`_reference_operations`), and
    ``ebr_positions`` the positions the group's EBR list names, each
    ``(M, letter, site symbol)`` with ``M`` the multiplicity.  Each letter is
    checked once.  It passes where ``pyxtal`` lists it with multiplicity
    ``M``, and where its point at a generic value of the free parameters
    (:data:`_GENERIC_PARAMETER`) has a stabilizer of order
    :math:`|P|\,n_\text{centering} / M` under ``ops_ref`` and the centering
    translations (:func:`_stabilizer_order`), the orbit-stabilizer theorem in
    the conventional cell, with :math:`|P|` the number of operations in
    ``ops_ref``.  Returns a mapping: ``"ok"``, whether every letter passes;
    ``"offenders"``, each letter that fails with what failed;
    ``"point_group"``, :math:`|P|`; ``"centering"``, the number of centering
    translations; and ``"checked"``, how many stabilizers were measured.
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


@lru_cache(maxsize=256)
def _letters_on_closure(sg: int, letter: str, listed: frozenset,
                        tol: float = WYCKOFF_REFERENCE_TOL) -> tuple:
    """The letters of ``listed`` whose varieties lie inside that of ``letter``.

    ``sg`` is the space-group number, ``letter`` one of its Wyckoff letters
    and ``listed`` the letters considered; ``tol`` is the containment
    tolerance of :func:`_variety_contains_variety`, in reduced coordinates of
    the tables' conventional cell.  An orbital confined to ``letter``'s
    variety can be moved onto each of these positions without leaving it.
    Returns the letters sorted, ``letter`` among them where ``listed`` holds
    it, and none where ``pyxtal`` lists no ``letter``.
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


def _snap_rational(x, max_den=24, atol=1e-4):
    """Coordinates snapped to the nearest rational with a small denominator.

    ``x`` is an array of coordinates.  Each entry is replaced by the nearest
    ``p/q`` with ``q`` at most ``max_den``, where that lies closer than
    ``atol``, and is kept otherwise.  ``irreptables`` prints high-symmetry
    k-points to six decimals and translations to five, and the coordinates
    they stand for are such rationals, so the snap recovers them.  Returns a
    new array of the shape of ``x``.
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


#: Centering translations of each Bravais lattice, in the conventional basis
#: the tables use, that of the Bilbao Crystallographic Server, keyed by the
#: leading letter of the table's Hermann-Mauguin name (``IrrepTable.name``).
#: This is the one piece of crystallography that cannot be read off a character
#: table, and it is needed because "``g`` fixes ``k``" and "``t`` is a lattice
#: vector" are statements about the primitive lattice while the tables are
#: written in the conventional one.  Checked, not trusted: the little groups
#: it predicts are asserted equal to ``irreptables``' own tabulated
#: little-group indices at all 2700 (space group, valuedness, k-point) entries
#: by the sweep in ``tests/test_groups.py``.
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
    """The centering translations of ``table``'s Bravais lattice.

    ``table`` is an ``irreptables`` table, its name opening on the lattice's
    letter.  Returns the nonzero centering translations in the tables'
    conventional cell (:data:`_CENTERINGS`), none for ``P``.

    Raises
    ------
    _EBRTransportError
        The name does not open on a Bravais lattice letter.
    """
    letter = str(table.name)[:1]
    if letter not in _CENTERINGS:
        raise _EBRTransportError(
            f"the name the tables give #{table.number_str}, {table.name!r}, does not "
            "start with the letter of a Bravais lattice.")
    return _CENTERINGS[letter]


def _is_translation(v, centering) -> bool:
    """Whether ``v`` is a translation of the direct lattice.

    ``v`` is in reduced coordinates of the tables' conventional cell, and
    ``centering`` holds its centering translations (:func:`_centering`).
    Returns ``True`` where ``v`` is an integer vector, or a centering
    translation plus an integer vector, each component within
    :data:`_LATTICE_TOL`.
    """
    v = np.asarray(v, dtype=float)
    for c in ((0.0, 0.0, 0.0),) + tuple(centering):
        d = v - np.asarray(c, dtype=float)
        if np.abs(d - np.round(d)).max() <= _LATTICE_TOL:
            return True
    return False


def _is_reciprocal(v, centering) -> bool:
    """Whether ``v`` is a vector of the reciprocal lattice.

    ``v`` is in reduced coordinates of the reciprocal of the tables'
    conventional cell, and ``centering`` holds that cell's centering
    translations (:func:`_centering`).  Returns ``True`` where ``v`` is an
    integer vector whose product with every centering translation is an
    integer, each within :data:`_LATTICE_TOL`.  For an ``I`` lattice that is
    the rule that ``h + k + l`` is even, so the point ``P = (1/2, 1/2, 1/2)``
    of a body-centered group is not a time-reversal invariant momentum.
    """
    v = np.asarray(v, dtype=float)
    if np.abs(v - np.round(v)).max() > _LATTICE_TOL:
        return False
    return all(abs(float(v @ np.asarray(c)) - round(float(v @ np.asarray(c))))
               <= _LATTICE_TOL for c in centering)


# --------------------------------------------------------------------------
# the elementary band representations' convention, where it cannot be carried
# --------------------------------------------------------------------------

class _EBRTransportError(RuntimeError):
    """The failure to carry a group's EBR rows out of the tables' convention.

    Raised with a message and ``points``, the labels of the k-points where
    the carrying fails, empty where no point is at fault.
    :func:`.._tables.ebr_data` returns it as the refusal ``ebr_convention``,
    its ``points`` in the context.
    """

    def __init__(self, message, points=()):
        super().__init__(message)
        self.points = tuple(points)
