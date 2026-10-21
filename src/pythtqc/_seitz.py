"""Each operation of a group by its index, with its Seitz symbol in the tables'
setting and its name in the model's frame: the key every per-operation record
is written on.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from fractions import Fraction
import math

import numpy as np

from . import _sitechars
from .refusal import InternalError, Refusal
from ._tolerance import _LATTICE_TOL
from ._text import _SEITZ_DENOMINATOR, _index_ranges
from ._spacegroup import GroupData, _CENTERINGS, _lattice_not_descended

__all__ = ["Operation", "OperationTable", "operations", "seitz", "operation_index"]


# --------------------------------------------------------------------------
# the operations by index, and their Seitz symbols
# --------------------------------------------------------------------------

#: The order of the rotation part of an integer rotation matrix, read off the
#: trace of its proper part ``det(R)·R``, whose eigenvalues are
#: ``1, e^{±2πi/n}``.  The determinant then separates ``n`` from ``n̄``, which
#: one trace leaves open and ITA's symbols keep apart.
_ROTATION_ORDER = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}


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
    if isinstance(group_or_rep, GroupData):
        return group_or_rep, tuple(int(s.ind) for s in group_or_rep.symmetries)
    group = getattr(group_or_rep, "group", None)
    if isinstance(group, GroupData):
        matrices = getattr(group_or_rep, "matrices", None)
        if isinstance(matrices, dict):
            # a representation: the operations it holds matrices for
            return group, tuple(sorted(int(i) for i in matrices))
        # a realized group: every operation of its group
        return group, tuple(int(s.ind) for s in group.symmetries)
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
