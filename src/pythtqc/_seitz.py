"""The operations of a group by index, each with its Seitz symbol and its name.

The Seitz symbol is written in the conventional cell used by the irreptables
package, and the name in the Cartesian axes in which the pythtb Lattice is
specified.  Records with an entry for each operation of the group name the
operation by its index.
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
from ._spacegroup import GroupData, _CENTERINGS, _lattice_not_descended, _require_group

__all__ = ["Operation", "OperationTable", "operations", "operation_index"]


# --------------------------------------------------------------------------
# the operations by index, and their Seitz symbols
# --------------------------------------------------------------------------

#: The order ``n`` of the proper part ``det(R)·R`` of an integer rotation matrix
#: ``R``, the ``n`` of the Seitz symbol ``n`` or ``-n`` (``m`` for ``-2``), read
#: off its trace, since its eigenvalues are ``1, e^{±2πi/n}``.  The determinant
#: then tells the rotation ``n`` from the rotoinversion ``-n``, which the trace
#: alone leaves open.
_ROTATION_ORDER = {3: 1, 2: 6, 1: 4, 0: 3, -1: 2}


def _primitive_direction(u) -> tuple:
    """A direction as its smallest integer triple, its first nonzero component positive.

    ``u`` is a direction, three numbers that are rounded to integers.  Returns
    the integer components.
    """
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
    """The axis of the proper rotation ``proper``, as an integer triple.

    ``proper`` is an integer rotation matrix other than the identity.  For a
    mirror or a rotoinversion ``R`` it is ``det·R``, whose axis is the mirror's
    normal or the rotoinversion's axis.  The axis is the cross product of two
    independent rows of ``proper − 1``, which are orthogonal to it.  Returns it
    as :func:`_primitive_direction` writes it.
    """
    rows = np.rint(np.asarray(proper, dtype=float)).astype(int) - np.eye(3, dtype=int)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        u = np.cross(rows[i], rows[j])
        if u.any():
            return _primitive_direction(u)
    raise InternalError(                             # pragma: no cover - guard
        "the proper rotation det(R)·R of an operation fixes a plane, and is not the "
        "identity. Report it, with the model and the call.")


def _direction_text(axis) -> str:
    """The integer direction ``axis`` as a Seitz symbol writes it, ``1-10``.

    Each component is one digit, with a minus sign before a negative one.
    Returns the string.
    """
    if max(abs(int(x)) for x in axis) > 9:
        raise InternalError(                         # pragma: no cover - guard
            f"the axis {tuple(int(x) for x in axis)} of an operation has a component "
            "above nine, which no crystallographic axis has. Report it, with the model "
            "and the call.")
    return "".join(str(int(x)) for x in axis)


def _parse_direction(text) -> tuple | None:
    """An integer direction read back from ``text``, as :func:`_direction_text` writes it.

    Returns three ints, or ``None`` where ``text`` holds anything but digits
    and minus signs or does not hold three digits.
    """
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
    """The sense of the turn of operation ``sym`` about ``axis``.

    ``sym`` is an operation of ``group``, of order above two, and ``axis`` the
    direction of its axis in the conventional cell used by the irreptables
    package (:func:`_rotation_axis`).  The turn is that of the proper part
    ``det·R``.  It is measured in the Cartesian axes in which the pythtb
    Lattice is specified, by :func:`.._sitechars.classify`, and its sign is
    taken against ``axis`` carried into those axes.  Returns ``+1`` for a
    counterclockwise turn about ``axis`` and ``-1`` for a clockwise one.
    """

    info = _sitechars.classify(sym)
    direction = ((np.asarray(group._refUC, dtype=float) @ np.asarray(axis, dtype=float))
                 @ np.asarray(group._lattice, dtype=float))
    return int(info["sense"]) * (1 if float(direction @ info["axis"]) >= 0 else -1)


def _fraction_mod_one(x) -> Fraction:
    """The translation component ``x`` as an exact fraction, modulo one.

    ``x`` is a component in the conventional cell used by the irreptables
    package.  It is assumed within :data:`_LATTICE_TOL` of a fraction of
    denominator :data:`_SEITZ_DENOMINATOR` or less, as every tabulated
    translation is.  Returns that fraction, in ``[0, 1)``.
    """
    value = Fraction(float(x)).limit_denominator(_SEITZ_DENOMINATOR)
    if abs(float(value) - float(x)) > _LATTICE_TOL:
        raise InternalError(                         # pragma: no cover - guard
            f"the translation component {float(x):.9g}, in the conventional cell used "
            "by the irreptables package, is no fraction of denominator "
            f"{_SEITZ_DENOMINATOR} or less. Report it, with the model and the call.")
    return value % 1


def _reduced_translation(t, centering) -> tuple:
    r"""One fixed representative of the translation ``t`` modulo the lattice.

    ``t`` is a translation in the conventional cell used by the irreptables
    package, and ``centering`` the centering vectors of that cell's lattice
    (:func:`_centering_of`).  The representatives are ``t`` and ``t`` plus
    each centering vector, each component modulo one
    (:func:`_fraction_mod_one`).  The one returned has the smallest sum of
    components, ties going to the smaller first component, then the smaller
    second.  A written symbol is matched on it, so that any representative
    names the same operation.  The symbol is written with the representative
    the tables give (:func:`_written_translation`).  Returns three fractions.
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
    """The centering vectors of the Bravais lattice of ``group``.

    ``group`` is a :class:`GroupData`, whose name opens on its lattice letter.
    Returns the vectors other than zero, in the conventional cell used by the
    irreptables package, none for a primitive lattice.
    """
    letter = str(group.name)[:1]
    if letter not in _CENTERINGS:
        raise InternalError(                         # pragma: no cover - guard
            f"the name {group.name!r} of group #{group.number_str} does not start with "
            "a Bravais lattice letter. Report it, with the model and the call.")
    return _CENTERINGS[letter]


def _tabulated_translations(group) -> dict:
    """The translations the tables give the operations of ``group``, by rotation.

    ``group`` is a :class:`GroupData`.  The tabulated operations are those the
    irrep package keeps on its record of the group, ``u_symmetries_tables``
    and ``au_symmetries_tables``.  Returns a mapping from
    ``(rotation, antiunitary)`` to the translation, an array of shape (3,).
    ``rotation`` is the bytes of the integer rotation matrix and ``antiunitary``
    a bool.  The matrix and the translation are in the conventional cell used
    by the irreptables package.  The mapping is empty where the record carries
    no tabulated operations.
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
    """The translation the Seitz symbol of operation ``sym`` is written with.

    ``sym`` is an operation of ``group``, ``tabulated`` the mapping
    :func:`_tabulated_translations` returns for ``group``, and ``shortest``
    the representative :func:`_reduced_translation` gives for the translation
    of ``sym``.  Returns the tabulated translation of the rotation and
    antiunitarity of ``sym``, each component modulo one, or ``shortest`` where
    none is tabulated; either is three fractions.
    """
    rotation = np.asarray(sym.rotation_refUC(np.asarray(group._refUC, dtype=float)),
                          dtype=int)
    entry = tabulated.get((rotation.tobytes(), bool(sym.time_reversal)))
    if entry is None:
        return shortest
    return tuple(_fraction_mod_one(x) for x in entry)


def _translation_text(parts) -> str:
    """Three translation components as a Seitz symbol writes them, ``0,1/2,1/4``.

    ``parts`` holds the components, each a fraction or, in a model's own
    reduced coordinates (:func:`_own_frame_translation`), a float.  A fraction
    is written ``n/d``, or ``n`` where it is an integer, and a float to four
    significant digits.  Returns the components joined by commas.
    """
    return ",".join(f"{p:.4g}" if isinstance(p, float)
                    else f"{p.numerator}" if p.denominator == 1
                    else f"{p.numerator}/{p.denominator}" for p in parts)


def _own_frame_translation(t) -> tuple:
    """The translation ``t`` modulo the lattice, in a model's own reduced coordinates.

    A component within :data:`_LATTICE_TOL` of a fraction of denominator
    :data:`_SEITZ_DENOMINATOR` or less is that fraction, modulo one.  Any other
    component is the float it is, modulo one; a model whose origin sits off
    every point its operations fix carries such translations.  Returns three
    components, three zeros for a lattice vector of the cell.
    """
    parts = []
    for x in np.asarray(t, dtype=float).reshape(-1):
        x = float(x) % 1.0
        value = Fraction(x).limit_denominator(_SEITZ_DENOMINATOR)
        parts.append(value % 1 if abs(float(value) - x) <= _LATTICE_TOL else x)
    return tuple(parts)


def _record_name(sym, lattice) -> str:
    """The name of operation ``sym`` as a record that lists operations writes it.

    ``lattice`` holds the cell's lattice vectors as rows, in Cartesian axes.
    The name is that of :func:`._sitechars.operation_name`, in the Cartesian
    axes in which the pythtb Lattice is specified.  Where the translation of
    ``sym``, in the cell's reduced coordinates (:func:`_own_frame_translation`),
    is not a lattice vector, the name is written in Seitz form with it:
    ``{E|1/2,0,0}`` for a half translation, which ``E`` alone would print as
    the identity.  A trailing ``'`` marks time reversal.  Returns the name.
    """

    name = _sitechars.operation_name(sym, np.asarray(lattice, dtype=float))
    parts = _own_frame_translation(sym.translation)
    if any(parts):
        name = "{" + name + "|" + _translation_text(parts) + "}"
    return name + ("'" if bool(sym.time_reversal) else "")


def _seitz_key(sym, group) -> tuple:
    """The quantities the Seitz symbol of operation ``sym`` is written from.

    ``sym`` is an operation of ``group``.  Returns
    ``(det, order, axis, sense, translation)``, read in the conventional cell
    used by the irreptables package: the determinant of the rotation ``R``;
    the order ``n`` of its proper part ``det·R``, the ``n`` of the symbol
    ``n`` or ``-n``; its axis (:func:`_rotation_axis`), empty for order one;
    its sense (:func:`_rotation_sense`), ``0`` for order two or one; and the
    representative of its translation :func:`_reduced_translation` gives.
    A written symbol is matched on the same quantities (:func:`_matches`).
    """
    refUC = np.asarray(group._refUC, dtype=float)
    rotation = np.asarray(sym.rotation_refUC(refUC), dtype=int)
    det = int(round(float(np.linalg.det(rotation))))
    proper = det * rotation
    trace = int(round(float(np.trace(proper))))
    if trace not in _ROTATION_ORDER:                   # pragma: no cover - guard
        raise InternalError(
            f"the rotation of operation #{int(sym.ind)}, in the conventional cell used "
            f"by the irreptables package, has determinant {det}, and its proper rotation "
            f"det(R)·R has trace {trace}, which is no crystallographic rotation. Report "
            "it, with the model and the call.")
    order = _ROTATION_ORDER[trace]
    axis = () if order == 1 else _rotation_axis(proper)
    sense = _rotation_sense(sym, group, axis) if order > 2 else 0
    t = sym.translation_refUC(refUC, np.asarray(group._shiftUC, dtype=float))
    return det, order, axis, sense, _reduced_translation(t, _centering_of(group))


def _seitz_text(key, time_reversal, written=None) -> str:
    """An operation's Seitz symbol, ``{-4+_001|0,0,1/2}``, from its quantities.

    ``key`` is what :func:`_seitz_key` returns for the operation, and
    ``time_reversal`` whether it is antiunitary, which adds a trailing ``'``.
    ``written`` is the translation to write, three fractions
    (:func:`_written_translation`); where it is ``None``, the translation in
    ``key`` is written.  Returns the symbol.
    """
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
    """The written Seitz symbol ``text`` read back, as ``(key, time_reversal)``.

    ``centering`` holds the centering vectors of the group's lattice
    (:func:`_centering_of`).  ``key`` is in the quantities :func:`_seitz_key`
    measures, with ``sense`` ``0`` where ``text`` gives none.  The axis is
    turned so that its first nonzero component is positive, and the sense with
    it, since ``n+`` about ``[uvw]`` and ``n-`` about ``[-u-v-w]`` are one
    operation.  The translation is reduced by :func:`_reduced_translation`.
    Spaces between the parts of the symbol are ignored.  The underscore
    before the axis may be left out, and the translation's components may be
    separated by spaces in place of commas.  Returns ``None`` where ``text``
    is not a Seitz symbol.  A translation component that is no fraction of
    denominator :data:`_SEITZ_DENOMINATOR` or less is not caught here: it
    raises in :func:`_fraction_mod_one`.
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
    """Whether a written symbol names the operation of ``key``.

    ``key`` and ``time_reversal`` are an operation's quantities
    (:func:`_seitz_key`) and whether it is antiunitary; ``wanted`` and
    ``wanted_tr`` are the same, read off the symbol (:func:`_parse_seitz`).
    They match where every quantity agrees, the sense compared only where the
    symbol gives one.
    """
    if bool(time_reversal) != bool(wanted_tr):
        return False
    det, order, axis, sense, parts = key
    w_det, w_order, w_axis, w_sense, w_parts = wanted
    if (det, order, tuple(axis), parts) != (w_det, w_order, tuple(w_axis), w_parts):
        return False
    return w_sense == 0 or w_sense == sense


@dataclass(frozen=True)
class Operation:
    """One operation of a group: its index, its Seitz symbol and its name.

    One row of an :class:`OperationTable`, ``r.operations[index]``.  Records
    with an entry for each operation of the group, such as the matrices of
    ``r.rep`` and the verdicts of ``r.numerical.validated``, name the
    operation by this index.  ``summary()`` prints it.

    Attributes
    ----------
    index : int
        Its index, in the order of the tables of the irreptables package.
    seitz : str
        Its Seitz symbol ``{R|t}``, written in the conventional cell used by
        the irreptables package: ``{3+_001|0,0,0}``.  A trailing ``'`` marks
        an operation combined with time reversal.
    name : str
        Its name in the Cartesian axes in which the pythtb Lattice is
        specified: ``C3(z)``, ``C3^-1(z)``, ``m(x)``.  These are the axes
        ``characters=`` is written in.  An axis, or a mirror's normal, along
        none of ``x``, ``y`` and ``z`` is named by its direction in the
        lattice vectors: ``C2([1-10])``, ``m([1-10])``.  A trailing ``'``
        marks time reversal, as in the Seitz symbol.
    rotation : ndarray of int, shape (3, 3)
        Its rotation, in the model's reduced coordinates.
    translation : ndarray, shape (3,)
        Its translation, in the model's reduced coordinates.
    antiunitary : bool
        Whether it is combined with time reversal.
    rotation_cartesian : ndarray, shape (3, 3)
        Its rotation in the Cartesian axes in which the pythtb Lattice is
        specified.  A layer whose Lattice is specified in two dimensions has
        its normal along ``z``.
    spin_lift : ndarray, shape (2, 2), or None
        If not ``None``, its matrix on a spin one-half.  For a rotation by
        ``θ`` about the unit vector ``n`` it is ``e^{-iθ n·σ/2}``, with ``n``
        in the same Cartesian axes and the spin along their ``z``.  An
        improper operation carries the matrix of its proper part.  An
        operation combined with time reversal carries the matrix of its
        unitary part.  ``None`` where the group is not a double group.
    """

    index: int
    seitz: str
    name: str
    rotation: np.ndarray
    translation: np.ndarray
    antiunitary: bool
    rotation_cartesian: np.ndarray | None = None
    spin_lift: np.ndarray | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operation):
            return NotImplemented
        return (self.index == other.index and self.seitz == other.seitz
                and self.name == other.name and self.antiunitary == other.antiunitary
                and np.array_equal(self.rotation, other.rotation)
                and np.array_equal(self.translation, other.translation))

    def summary(self) -> str:
        """The operation, a line for each part, with the attribute that holds it."""
        lines = [f"operation {int(self.index)}: {self.seitz} in the tables' setting, "
                 f"{self.name} in the model's frame   (.index, .seitz, .name)",
                 "  its rotation and translation in the model's reduced coordinates   "
                 "(.rotation, .translation)"]
        if self.rotation_cartesian is not None:
            lines.append("  its rotation in the model's Cartesian axes   (.rotation_cartesian)")
        if self.spin_lift is not None:
            lines.append("  its matrix on the spin, spin along the z of those axes   "
                         "(.spin_lift)")
        if self.antiunitary:
            lines.append("  combined with time reversal, which the ' marks   (.antiunitary)")
        return "\n".join(lines)


class OperationTable(dict):
    """The operations of a group, a mapping from each index to its operation.

    The record :func:`operations` returns, which ``r.operations`` also holds.
    It holds one operation for each coset of the lattice translations.
    ``table[2]`` is the :class:`Operation` of index 2, and ``table.values()``
    gives the operations in index order.  Printed, it is a table of each
    operation's index, Seitz symbol and name.  ``summary()`` prints it.

    Attributes
    ----------
    group : GroupData
        The group whose operations these are, ``r.group``.
    """

    def __init__(self, rows=(), *, group=None):
        super().__init__((row.index, row) for row in rows)
        #: The group these are the operations of.
        self.group = group

    def _named(self) -> str:
        """The group's name and number as the table prints them, ``P4mm1' #99.164``.

        Empty where the table has no group.
        """
        group = self.group
        return "" if group is None else f"{group.name} #{group.number_str}"

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"OperationTable({self._named()}, operations {_index_ranges(self)})"

    def summary(self) -> str:
        """The table as printed: each operation's index, Seitz symbol and name."""
        return str(self)

    def __str__(self) -> str:
        rows = list(self.values())
        width = max([len(row.seitz) for row in rows] + [5])
        head = (f"{self._named()}: one operation per coset of the lattice "
                "translations"
                + ("; the double group also holds each one's negative"
                   if any(row.spin_lift is not None for row in rows) else ""))
        lines = [head, f"{'index':>5}  {'Seitz':<{width}}  name"]
        lines += [f"{row.index:>5}  {row.seitz:<{width}}  {row.name}" for row in rows]
        return "\n".join(lines)


def _operation_rows(group, indices=None) -> OperationTable | Refusal:
    """The operations of ``group`` as an :class:`OperationTable`, all or some.

    ``group`` is a :class:`GroupData`, or a refusal, which is returned as it
    is.  ``indices`` holds the indices of the operations to list, in any
    order; with ``None`` every operation is listed.  Returns the table, its
    rows in index order.  Refuses ``lattice_not_descended`` where the group
    holds a pure translation that is not a lattice vector of the model's cell,
    and ``tables_not_ready`` where the model's cell could not be matched to
    the conventional cell used by the irreptables package.
    """
    if isinstance(group, Refusal):
        return group
    if not isinstance(group, GroupData):
        raise InternalError(                         # pragma: no cover - guard
            f"the operations were asked of a {type(group).__name__}, which is not a "
            "group. Report it, with the model and the call.")
    gate = _lattice_not_descended(group)
    if gate is not None:
        return gate
    if not group._tables_ready:
        return Refusal("tables_not_ready", group._setup_note or f"the cell of "
                       f"{group.name} #{group.number_str} could not be matched to the "
                       "setting the installed tables are written in. Report the model, "
                       "with this reason.", context={"number_str": group.number_str})

    lattice = np.asarray(group._lattice, dtype=float)
    tabulated = _tabulated_translations(group)
    wanted = (None if indices is None else {int(i) for i in indices})
    rows = []
    for sym in group._symmetries:
        if wanted is not None and int(sym.ind) not in wanted:
            continue
        prime = "'" if sym.time_reversal else ""
        key = _seitz_key(sym, group)
        rows.append(Operation(
            index=int(sym.ind),
            seitz=_seitz_text(key, bool(sym.time_reversal),
                              _written_translation(sym, group, tabulated, key[4])),
            name=_sitechars.operation_name(sym, lattice) + prime,
            rotation=np.rint(np.asarray(sym.rotation, dtype=float)).astype(int),
            translation=np.asarray(sym.translation, dtype=float).copy(),
            antiunitary=bool(sym.time_reversal),
            rotation_cartesian=np.asarray(sym.rotation_cart, dtype=float).copy(),
            spin_lift=(np.asarray(sym.spinor_rotation, dtype=complex).copy()
                       if group.spinor else None)))
    rows.sort(key=lambda row: row.index)
    return OperationTable(rows, group=group)


def operations(group) -> OperationTable | Refusal:
    """The operations of a group, one for each coset of the lattice translations.

    Each operation has an index, a Seitz symbol and a name.  The operations
    are listed in the order of the tables of the irreptables package.  In a
    double group each operation's product with the rotation by 2π is in the
    group too, and is not listed.  The Seitz symbol ``{R|t}`` is written in
    the conventional cell used by the irreptables package.  An underscore
    stands before the direction of the axis, and a minus sign for an overbar,
    as in ``{-4+_001|0,0,0}``.  The name is written in the Cartesian axes in
    which the pythtb Lattice is specified.

    Parameters
    ----------
    group : GroupData
        The group, ``r.group``.

    Returns
    -------
    OperationTable or Refusal
        A mapping from each index to its :class:`Operation`.  ``r.operations``
        holds the same table.  A ``Refusal`` in its place is falsy and names
        what stopped the call.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        the installed tables cannot be read for this group in this cell.

    Raises
    ------
    TypeError
        ``group`` is not a group: ``r`` or ``r.rep`` in its place, or
        anything else.

    See Also
    --------
    operation_index : the index of the operation a Seitz symbol names.
    """
    _require_group("operations", group)
    return _operation_rows(group)


def operation_index(group, seitz) -> int | Refusal:
    """The index of the operation a Seitz symbol names.

    The symbol is read in the conventional cell used by the irreptables
    package, the cell :func:`operations` writes the symbols in.  A rotation
    written about either direction of its axis names the same operation:
    ``3+_001`` and ``3-_00-1`` are one.  A symbol whose translation differs
    by a lattice vector names the same operation too.  Spaces between the
    parts of the symbol are ignored, and the underscore before the axis may
    be left out.

    Parameters
    ----------
    group : GroupData
        The group, ``r.group``.
    seitz : str
        A Seitz symbol ``{R|t}``, with a trailing ``'`` for an operation
        combined with time reversal: ``"{3+_001|0,0,0}'"``.

    Returns
    -------
    int or Refusal
        The operation's index.  A ``Refusal`` in its place is falsy and names
        what stopped the call.

    Refusals
    --------
    A refusal handed in as ``group`` is returned as it is.

    ``tables_not_ready``
        the installed tables cannot be read for this group in this cell.
    ``unknown_operation``
        ``seitz`` names no operation of the group.  The refusal names up to
        three of the group's symbols, the nearest in spelling where any is
        close.

    Raises
    ------
    TypeError
        ``group`` is not a group: ``r`` or ``r.rep`` in its place, or
        anything else.

    See Also
    --------
    operations : the operations of a group, each with its index, Seitz symbol
        and name.
    """
    _require_group("operation_index", group)
    rows = _operation_rows(group)
    if isinstance(rows, Refusal):
        return rows
    parsed = _parse_seitz(seitz, _centering_of(group))
    if parsed is not None:
        wanted, wanted_tr = parsed
        for sym in group._symmetries:
            if _matches(_seitz_key(sym, group), bool(sym.time_reversal),
                        wanted, wanted_tr):
                return int(sym.ind)
    carried = [row.seitz for row in rows.values()]
    near = difflib.get_close_matches(str(seitz), carried, n=3, cutoff=0.4) or carried[:3]
    return Refusal(
        "unknown_operation",
        f"`{str(seitz)}` names no operation of {rows._named()}. "
        f"The symbols it carries include `{'`, `'.join(near)}`. `operations(group)` "
        "lists every operation of the group with its index and Seitz symbol.",
        context={"number_str": str(rows.group.number_str), "seitz": str(seitz),
                 "nearest": tuple(near), "carried": tuple(carried)})
