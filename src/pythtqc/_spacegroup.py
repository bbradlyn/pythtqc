"""The space group of a cell, found on its point set and written in the model's
own cell.

The records every later measurement is written on are here: the group, the
little group of a k-point, a Wyckoff position, a site's position in the tables'
setting, the orbitals' assignment to their sites, the coreps a measured
multiplet carries, a k-mesh and the verdict on each operation of a group.  The
group is identified by ``irrep`` from the cell's point set, its SU(2) lifts put
in the tables' convention, its operations closed exactly, its pure translations
read for the primitive lattice, its magnetic type read off its antiunitary
operations, and each orbital assigned to the Wyckoff orbit its site lies on.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import math
import os
import string
from typing import Any

import irrep.spacegroup_irreps  # noqa: F401
import numpy as np

from .refusal import InternalError, Refusal
from ._tolerance import (
    DEFAULT_FINDER, WYCKOFF_REFERENCE_TOL, _EXACT_CLOSURE, _LATTICE_TOL)
from ._text import _affine_text, _index_ranges, _lattice_vectors, _point_text

__all__ = ["tables_status", "GroupData", "LittleGroup", "WyckoffPosition",
           "SiteReference", "OrbitalAssignment", "IrrepContent", "KGrid",
           "wyckoff_positions_spglib"]


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
# records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupData:
    """A space group, written in the model's own cell.

    The group ``realized_group(model).group`` holds, and the one every table
    call takes: :func:`.._seitz.operations`, :func:`.._tables.high_symmetry_points`,
    :func:`.._tables.corep_table`, :func:`.._tables.ebr_data`.  ``summary()`` prints it.

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
        :func:`.._seitz.operations` writes them out.
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
    #: Memo of :func:`.._subgroups._subgroups_inside` for this group, keyed by the operations
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

    What :func:`.._tables.little_group` returns; ``summary()`` prints it, and ``len`` is
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
    may differ: :func:`.._tables.wyckoff_sites_tables` gives each site's letter in the
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

    One entry of what :func:`.._tables.wyckoff_sites_tables` returns, which prints its
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

    What :func:`.._tables.identify_from_characters` returns, and what
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

    What :func:`.._tables.k_grid` returns; ``summary()`` prints it.  Its :attr:`mesh` is
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


# --------------------------------------------------------------------------
# cell construction
# --------------------------------------------------------------------------

#: The name under which a cell :func:`.._layers._decorated_cell` builds carries how many of
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
    the orbitals that goes with it (:func:`.._layers._slab_presentation`), so the
    reciprocal vectors dual to the periodic directions — and therefore
    pythtb's reduced k — are untouched; missing dimensions get orthogonal
    vectors of length ``vacuum_length``.  Positions along a vacuum direction are
    re-centred on the origin so that a horizontal mirror appears without a
    translation.

    The vacuum is measured from the **model's own orbitals**: a cell built on
    a model (:func:`.._layers._decorated_cell`) is given the vacuum the model is, however
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
    :func:`.._measure._validate_group` measures.
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
    :func:`.._tables._transported_table`).  Twisting a lift set by a sign vector
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
      on, from ``_build._build_rep``'s internal factor to the product table — in
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
    (:func:`.._centres._symmetric_sites`).

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
    """Internal to :func:`.._realized.realized_group`: the space group the cell allows.

    Read from the lattice and the orbital positions alone, so what comes back
    is an upper bound on the group a Hamiltonian on this cell can obey, stamped
    :attr:`GroupData.provenance` ``= "cell"``.  ``typat`` labels orbital
    classes, ``include_TR=True`` asks the cell as a grey group — a point set
    says nothing about a Hamiltonian's time reversal — ``spinor`` follows
    ``model.nspin``, and ``symprec`` and ``vacuum_length`` are the finder's
    tolerance, a length in the model's units at which every translation here
    is compared, and the slab presentation's vacuum, which no identification
    may depend on.  The cell is periodic in two or three directions:
    :func:`.._realized.realized_group` refuses a model periodic in fewer before any cell is
    identified, and a layer's cell is built on its slab presentation, whose
    non-periodic vector lies along the layer's normal
    (:func:`.._layers._slab_presentation`).

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
        which :func:`.._realized.realized_group` hands here where no precision is declared
        for its centres, or where they cannot be read as a symmetric set at all.

    Notes
    -----
    **Under a declared precision the cell is exact.**  :func:`.._realized.realized_group`
    reads the operations off the centres as they are, moves the centres onto
    the nearest set those operations carry exactly and removes a shift of every
    centre together that is noise, all before it asks here
    (:func:`.._realized._exact_centres`): the setting, the tables' origin and the names are
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
    :func:`.._realized.realized_group` can say whether the model *is* a supercell: it
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
    # anyway would let a build reach `_matrices._table_anchored` carrying a
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
    may differ: :func:`.._tables.wyckoff_sites_tables` gives each site's letter in the
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
    small-denominator rationals the way :func:`.._tables._table_kpoints` snaps k-points,
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


# --------------------------------------------------------------------------
# the elementary band representations' convention, where it cannot be carried
# --------------------------------------------------------------------------

class _EBRTransportError(RuntimeError):
    """The EBR rows cannot be carried into the physical convention (see below)."""

    def __init__(self, message, points=()):
        super().__init__(message)
        self.points = tuple(points)
