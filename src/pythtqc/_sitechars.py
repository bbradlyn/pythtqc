r"""The character tables of measured site groups, and declarations on them.

A site group's irreducible characters are computed from the product table of
the operations that fix the site
(:meth:`.._spacegroup.GroupData._get_product_table`), by the class-sum
construction of :func:`_character_table`, so a table exists at every site of
every model.  Its rows come out under their Mulliken symbols
(:func:`_mulliken`), and :mod:`.._naming` renames them with the tables' names
for the site irreps, where the tables name them.  Each operation is named in
the Cartesian axes in which the pythtb Lattice is specified
(:func:`operation_name`), and a message on a declaration prints every
operation's name (:func:`generator_frame`).

A ``characters=`` entry is resolved on the table of its site
(:func:`resolve_spec`).  On a model whose states carry spin, a row is
returned under the double-valued site irrep, or sum of them, that its orbital
factor times the spin-1/2 lift is, and a declaration may name either
(:attr:`SiteGroupTable.declarable`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import factorial
from typing import Any

import numpy as np

from ._tolerance import ASSEMBLED, CHARACTER, ZERO

# --------------------------------------------------------------------------
# the declaration error, which the build and realized_group return as a Refusal
# --------------------------------------------------------------------------


class DeclarationError(ValueError):
    """A ``characters=`` declaration its site cannot read.

    Raised for a word that names no site irrep there (``unknown_label``), a
    declaration that does not say which orbitals hold a multiplet
    (``ambiguous_multiplet``), or a site whose character table cannot be built
    (``site_group``).  ``gate`` is that name, ``message`` the reason and
    ``context`` what was measured.  It never leaves the package:
    :func:`.._build._build_rep` and :func:`..groups.realized_group` return the
    three as a :class:`..refusal.Refusal`.  A declaration outside the grammar
    raises ``ValueError`` before any site is read
    (:func:`.._grammar._declaration_grammar`).
    """

    def __init__(self, gate, message, context=None):
        super().__init__(message)
        self.gate = gate
        self.message = message
        self.context = dict(context or {})


# --------------------------------------------------------------------------
# 1.  naming the operations of a measured group
# --------------------------------------------------------------------------

_AXIS_NAMES = {(1, 0, 0): "x", (0, 1, 0): "y", (0, 0, 1): "z"}


def _axis_name(v_cart, lattice) -> str:
    """The name of the direction ``v_cart``, a Cartesian vector.

    ``x``, ``y`` or ``z`` along a Cartesian axis, in either sense; otherwise
    ``[uvw]`` in the lattice vectors, the rows of ``lattice``, as integers with
    the first nonzero one positive, else as the three ratios to three figures.
    ``?`` for a zero vector.
    """
    v = np.asarray(v_cart, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return "?"
    v = v / n
    for key, name in _AXIS_NAMES.items():
        if np.allclose(np.abs(v), np.asarray(key, dtype=float), atol=1e-6):
            return name
    # lattice coordinates: v_cart = sum_i u_i a_i
    try:
        u = np.linalg.solve(np.asarray(lattice, dtype=float).T, v)
    except np.linalg.LinAlgError:  # pragma: no cover - a degenerate cell
        return "[?]"
    top = float(np.abs(u).max())
    if top > 1e-12:
        u = u / top                      # a direction, so only the ratios matter
    best = None
    for scale in range(1, 25):
        w = u * scale
        if np.abs(w - np.rint(w)).max() < 1e-6:
            best = np.rint(w).astype(int)
            break
    if best is None:
        return "[" + ",".join(f"{x:.3g}" for x in u) + "]"
    for x in best:                       # sign convention: first nonzero positive
        if x != 0:
            if x < 0:
                best = -best
            break
    return "[" + "".join(f"{x}" if x >= 0 else f"-{-x}" for x in best) + "]"


def classify(sym) -> dict:
    r"""The kind, order, turn and axis of one operation, read off its matrix.

    ``sym`` is an operation with ``rotation_cart``.  Returns a mapping with
    ``kind`` (``"E"``, ``"i"``, ``"m"``, ``"C"`` or ``"S"``), ``order``,
    ``angle``, ``axis``, ``det`` and ``name``, and ``sense``, ``+1`` or ``-1``,
    for ``"C"`` and ``"S"``.  There the angle is the turn ``φ`` of the proper
    part ``det·R``, from ``tr R = det·(1 + 2 cos φ)``, and the axis is that
    part's.  An improper operation is that turn followed by the inversion, the
    Seitz ``-n+``, and ``order`` holds the turn's ``n``: ``-3+`` is ``S3``,
    ``-4+`` is ``S4``.  A mirror's axis is its normal.  ``name`` is set for
    ``E`` and ``i`` alone.
    """
    return _classified(sym.rotation_cart)


def _classified(matrix) -> dict:
    """:func:`classify` of one operation given by its matrix alone."""
    R = np.asarray(matrix, dtype=float)
    det = int(round(float(np.linalg.det(R))))
    c = max(-1.0, min(1.0, (float(np.trace(R)) - det) / 2.0))
    theta = float(np.arccos(c))
    proper = det * R
    if abs(theta) < 1e-7:
        if det > 0:
            return {"kind": "E", "order": 1, "angle": 0.0, "axis": np.array([0.0, 0.0, 1.0]),
                    "det": det, "name": "E"}
        return {"kind": "m", "order": 2, "angle": 0.0, "axis": _eigen_axis(R, -1.0),
                "det": det, "name": None}
    if abs(theta - np.pi) < 1e-7 and det < 0:
        return {"kind": "i", "order": 2, "angle": np.pi, "axis": np.array([0.0, 0.0, 1.0]),
                "det": det, "name": "i"}
    axis = _eigen_axis(proper, 1.0)
    # sense of the rotation about `axis`, from the antisymmetric part
    A = (proper - proper.T) / 2.0
    vec = np.array([A[2, 1], A[0, 2], A[1, 0]])
    sense = 1 if float(vec @ axis) >= 0 else -1
    # the turn of the proper part, which names an improper operation as well:
    # `-n+` turns by 2π/n and then inverts
    phi = theta if det > 0 else np.pi - theta
    n = int(round(2 * np.pi / phi)) if phi > 1e-9 else 1
    if n == 2:
        # a half turn is its own inverse and has no sense: the antisymmetric
        # part it would be read off vanishes, and its sign is rounding
        sense = 1
    kind = "C" if det > 0 else "S"
    return {"kind": kind, "order": n, "angle": phi, "axis": axis, "det": det,
            "sense": sense, "name": None}


def _eigen_axis(M, value) -> np.ndarray:
    """The unit eigenvector of ``M`` for its eigenvalue nearest ``value``.

    Its sign makes the last nonzero component positive: ``+z`` first, then
    ``+y``, then ``+x``.
    """
    w, V = np.linalg.eig(np.asarray(M, dtype=float))
    k = int(np.argmin(np.abs(w - value)))
    v = np.real(V[:, k])
    nrm = float(np.linalg.norm(v))
    v = v / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
    for comp in v[::-1]:                 # prefer +z, then +y, then +x
        if abs(comp) > 1e-9:
            if comp < 0:
                v = -v
            break
    return v


def operation_name(sym, lattice) -> str:
    """The name of the operation ``sym`` in the model's Cartesian axes.

    ``E`` and ``i``; ``C{n}`` a counterclockwise turn by ``2π/n`` and
    ``C{n}^-1`` its inverse, a half turn being written ``C2`` alone; ``S{n}``
    that turn followed by the inversion, the Seitz ``-n+``, and ``S{n}^-1``
    the Seitz ``-n-``; ``m`` a mirror, named by its normal.  The axis follows
    in parentheses, ``C3(z)``, ``m(x)``: ``x``, ``y`` or ``z`` along a
    Cartesian axis, and otherwise ``[uvw]`` in the lattice vectors, the rows
    of ``lattice`` (:func:`_axis_name`).
    """
    info = classify(sym)
    if info["name"] is not None:
        return info["name"]
    ax = _axis_name(info["axis"], lattice)
    if info["kind"] == "m":
        return f"m({ax})"
    tag = f"{info['kind']}{info['order']}"
    if info.get("sense", 1) < 0:
        tag += "^-1"
    return f"{tag}({ax})"


def _key(text) -> str:
    """The operation's name ``text`` in one spelling, for comparison.

    It is lowercased, spaces and underscores are dropped, ``sigma_v``,
    ``sigma_d``, ``sigma_h`` and ``sigma`` read as ``m``, and ``mx`` reads as
    ``m(x)``.
    """
    s = str(text).strip().lower().replace(" ", "").replace("_", "")
    for pre in ("sigmav", "sigmad", "sigmah", "sigma"):
        if s.startswith(pre):
            s = "m" + s[len(pre):]
            break
    if s.startswith("m") and len(s) > 1 and s[1] not in "([":
        s = "m(" + s[1:] + ")"           # "mx" -> "m(x)"
    return s


# --------------------------------------------------------------------------
# 2.  conjugacy classes and the character table, from the product table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteGroupTable:
    r"""The character table of one site group, with the names of its rows.

    Operations are given by their ``.ind`` in the represented group.

    Attributes
    ----------
    indices : tuple of int
        The unitary operations of the represented group that fix the site.
    order : int
        How many operations :attr:`indices` holds.
    classes : tuple of tuple of int
        The conjugacy classes.
    class_names : tuple of str
        Each class's name, its size where above one and its representative's
        name: ``2C4(z)``.
    class_reps : tuple of int
        The representative each class is named by.
    class_of : dict
        The class number of each operation.
    characters : ndarray, shape (n_row, n_class)
        The irreducible characters, row by row and class by class.  A
        complex-conjugate pair is merged into one row where an antiunitary
        operation fixes the site.
    labels : tuple of str
        Each row's name, empty for a row nothing names.  On a spinful site it
        is the double-valued site irrep, or sum of them, that the row times the
        spin-1/2 lift is, so two rows can carry one label.
    dims : ndarray
        Each row's dimension.
    names : dict
        Each operation's name in the model's Cartesian axes
        (:func:`operation_name`).
    orthogonality : float
        The largest entry of :math:`|X \mathrm{diag}(|C|) X^\dagger - |G| 1|`
        for the characters :math:`X` before any pair is merged, zero for an
        exact table.
    merged : tuple of bool
        Whether each row is a merged complex-conjugate pair.
    antiunitary : tuple of int
        The antiunitary operations that fix the site.  They do not enter the
        characters.
    named_by : tuple of str
        Where each row's label comes from: ``"tables"``, ``"invariants"``,
        ``"point group"``, ``"Mulliken"`` or ``"unnamed"``, as
        :attr:`..symmetry._OrbitalRow.named_by` reads them.
    declarable : tuple of str
        On a spinful site, the single-valued name of each row, the word that
        declares its orbital factor (``A2``); empty elsewhere.  It is the name
        the group's single-valued entries give the row at the position, else
        the name the tables of the stable real-space invariants give it, else
        the row's point-group name or Mulliken symbol, where it has one.  It is
        never the name a row is returned under.
    """

    indices: tuple
    order: int
    classes: tuple
    class_names: tuple
    class_reps: tuple
    class_of: dict
    characters: np.ndarray
    labels: tuple
    dims: np.ndarray
    names: dict
    orthogonality: float
    merged: tuple
    antiunitary: tuple = ()
    named_by: tuple = ()
    declarable: tuple = ()

    def index_of(self, label) -> int:
        """The row named ``label``, whatever its case, or ``-1`` where none is.

        The labels are searched first, then the :attr:`declarable` names.
        ``"E"`` also names the one merged complex pair of dimension two,
        ``1E2E``, where no row is named ``E``.  Where several rows carry
        ``label``, as two rows that become one double-valued site irrep with
        spin do on a spinful site, the row returned is the one whose character
        differs from ``+1`` on the fewest operations, the first on a tie.  An
        empty label names no row.  :func:`resolve_spec` reads a declaration
        written in double-valued site irreps as a sum, and does not look it up
        here (:func:`named_in_coreps`).
        """
        want = str(label).strip().lower()
        for names in (self.labels, self.declarable):
            hits = [i for i, lab in enumerate(names) if lab and lab.lower() == want]
            if hits:
                return min(hits, key=lambda i: (
                    sum(len(self.classes[c]) for c in range(len(self.classes))
                        if abs(complex(self.characters[i, c]) - 1.0) > 1e-9), i))
        if want == "e":
            hits = [i for i in range(len(self.labels))
                    if self.merged[i] and int(self.dims[i]) == 2]
            if len(hits) == 1:
                return hits[0]
        return -1

    def character_on(self, row, ind) -> complex:
        """The character of one row on one operation.

        ``row`` is the row, and ``ind`` the operation's ``.ind``.
        """
        return complex(self.characters[row, self.class_of[int(ind)]])


def _local_table(prod, positions):
    """The multiplication table of the operations at ``positions``.

    ``prod`` is the group's product table, indexed by position in its list of
    operations, and ``positions`` sets the order of the rows.  Entry
    ``[i, j]`` is the index in ``positions`` of the product of the ``i``-th
    and the ``j``-th.  Returns ``None`` where a product lies outside
    ``positions``.
    """
    idx = {int(p): i for i, p in enumerate(positions)}
    n = len(positions)
    T = np.full((n, n), -1, dtype=int)
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            c = int(prod[a, b])
            if c not in idx:
                return None
            T[i, j] = idx[c]
    return T


def _classes(T):
    """The conjugacy classes of the group whose multiplication table is ``T``.

    Returns ``(classes, e, inverse)``: each class as its elements' indices in
    ``T``, sorted, the identity's index, and each element's inverse.
    """
    n = T.shape[0]
    e = next(i for i in range(n) if all(int(T[i, j]) == j for j in range(n)))
    inv = {i: next(j for j in range(n) if int(T[i, j]) == e) for i in range(n)}
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        cl = tuple(sorted({int(T[T[g, i], inv[g]]) for g in range(n)}))
        seen.update(cl)
        out.append(cl)
    return tuple(out), e, inv


def _character_table(T, seed=101):
    r"""The irreducible characters of the group whose product table is ``T``.

    The structure constants :math:`c_{ijk}` of the class sums give commuting
    matrices :math:`(M_i)_{kj} = c_{ijk}`, whose common eigenvectors are the
    central characters :math:`\omega_i = |C_i| \chi(C_i)/\chi(1)`.  One real
    combination of them, its coefficients drawn with ``seed``, is
    diagonalized, and each dimension follows from
    :math:`\sum_i |\omega_i|^2/|C_i| = |G|/\chi(1)^2`.  Returns
    ``(classes, class_of, e, chars, defect)``: the classes of :func:`_classes`
    and each element's class, the identity, the characters row by row in
    order of dimension, and the largest entry of
    :math:`|X \mathrm{diag}(|C|) X^\dagger - |G| 1|`.
    """
    cls, e, _ = _classes(T)
    n, r = T.shape[0], len(cls)
    cls_of = {g: i for i, c in enumerate(cls) for g in c}
    sizes = np.array([len(c) for c in cls], dtype=float)
    M = np.zeros((r, r, r))
    for i in range(r):
        for j in range(r):
            counts = np.zeros(r)
            for x in cls[i]:
                for y in cls[j]:
                    counts[cls_of[int(T[x, y])]] += 1.0
            M[i, :, j] = counts / sizes          # M[i][k, j] = c_{ijk}
    rng = np.random.default_rng(int(seed))
    A = np.tensordot(rng.normal(size=r), M, axes=(0, 0))
    _, V = np.linalg.eig(A)
    chars = np.zeros((r, r), dtype=complex)
    for p in range(r):
        v = V[:, p]
        k = int(np.argmax(np.abs(v)))
        omega = np.array([complex((M[i] @ v)[k] / v[k]) for i in range(r)])
        d = float(np.sqrt(n / float(np.sum(np.abs(omega) ** 2 / sizes).real)))
        chars[p] = omega * d / sizes
    order = sorted(range(r), key=lambda p: (round(float(chars[p, cls_of[e]].real), 6),
                                            [round(float(x), 6) for x in chars[p].real],
                                            [round(float(x), 6) for x in chars[p].imag]))
    chars = chars[order]
    gram = (chars * sizes) @ chars.conj().T
    defect = float(np.abs(gram - n * np.eye(r)).max())
    return cls, cls_of, e, chars, defect


def _merge_conjugate_pairs(chars, tol=1e-8):
    """The rows of ``chars`` with each complex-conjugate pair merged into one.

    A row is complex where an imaginary part exceeds ``tol``, and its partner
    is the row equal to its conjugate within ``tol``; the two are added.  A
    complex row with no partner is kept as it is.  Returns ``(rows, merged)``,
    ``merged`` saying which rows are pairs.
    """
    r = chars.shape[0]
    used, rows, merged = set(), [], []
    for p in range(r):
        if p in used:
            continue
        if np.abs(chars[p].imag).max() <= tol:
            rows.append(chars[p])
            merged.append(False)
            used.add(p)
            continue
        q = next((j for j in range(r)
                  if j not in used and j != p
                  and np.abs(chars[j] - np.conj(chars[p])).max() <= tol), None)
        if q is None:                       # pragma: no cover - no partner: keep as is
            rows.append(chars[p])
            merged.append(False)
            used.add(p)
            continue
        rows.append(chars[p] + chars[q])
        merged.append(True)
        used.update({p, q})
    return np.array(rows), tuple(merged)


# --------------------------------------------------------------------------
# 3.  Mulliken symbols, read off the operations' Cartesian matrices.  A site
#     irrep keeps one only where the tables' names leave its row open
#     (`.._naming`).
# --------------------------------------------------------------------------

def _class_info(group, cls):
    """The representative of the class ``cls`` of ``group``, and its members.

    ``cls`` holds ``.ind`` values.  The representative turns in the positive
    sense where it turns, and has its axis nearest a Cartesian axis.  Returns
    ``(ind, info, infos)``: the representative and its :func:`classify`, and
    ``(ind, classify)`` for every member.
    """
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    infos = [(ind, classify(group._symmetries[pos_of[int(ind)]])) for ind in cls]
    # representative: prefer a positive rotation sense, then an axis near +x/+y/+z
    def key(item):
        _, info = item
        ax = info["axis"]
        along = max(abs(float(ax[0])), abs(float(ax[1])), abs(float(ax[2])))
        return (-info.get("sense", 1), -round(along, 6), tuple(np.round(-ax, 6)))
    ind, info = sorted(infos, key=key)[0]
    return ind, info, infos


def _principal_axis(class_infos):
    """The proper rotation of highest order among the classes' representatives.

    ``class_infos`` holds :func:`_class_info` of each class.  A tie goes to the
    axis nearest ``z``, then nearest ``y``.  Returns the rotation's
    :func:`classify`, or ``None`` where no class holds a proper rotation.
    """
    best = None
    for _, info, _ in class_infos:
        if info["kind"] != "C" or info["order"] < 2:
            continue
        key = (info["order"], round(abs(float(info["axis"][2])), 6),
               round(abs(float(info["axis"][1])), 6))
        if best is None or key > best[0]:
            best = (key, info)
    return None if best is None else best[1]


def _secondary_reference(class_infos, principal):
    """The class that sets the subscript ``1`` or ``2`` of a Mulliken symbol.

    ``class_infos`` holds :func:`_class_info` of each class and ``principal``
    the principal rotation (:func:`_principal_axis`).  The class is the one of
    twofold axes perpendicular to the principal axis that holds the axis
    closest to ``x``.  With no such class, it is the one of mirrors containing
    the principal axis that holds the plane closest to containing ``x``.
    ``y`` stands for ``x`` where the principal axis lies along ``x``.  Every
    member of a class is compared, not its representative alone.  Returns the
    class's index, or ``None``.
    """
    if principal is None:
        return None
    ax = principal["axis"]
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ax @ ref)) > 0.99:                 # principal already along x
        ref = np.array([0.0, 1.0, 0.0])
    perp2 = [(i, max(abs(float(info["axis"] @ ref)) for _, info in members))
             for i, (_, _, members) in enumerate(class_infos)
             if all(info["kind"] == "C" and info["order"] == 2
                    and abs(float(info["axis"] @ ax)) < 1e-6 for _, info in members)]
    if perp2:                                        # C2' axis closest to `ref`
        return sorted(perp2, key=lambda t: (-t[1], t[0]))[0][0]
    vert = [(i, min(abs(float(info["axis"] @ ref)) for _, info in members))
            for i, (_, _, members) in enumerate(class_infos)
            if all(info["kind"] == "m" and abs(float(info["axis"] @ ax)) < 1e-6
                   for _, info in members)]
    if vert:                                         # mirror *plane* containing `ref`
        return sorted(vert, key=lambda t: (t[1], t[0]))[0][0]
    return None


def is_placeholder(label) -> bool:
    """Whether ``label``, written by :func:`_mulliken`, is a placeholder.

    :func:`_mulliken` writes a placeholder where its rules cannot tell rows
    apart, a subscript ``_a``, ``_b``, ..., and ``L`` with the row's
    dimension for a row that is no ``A``, ``B``, ``E`` or ``T``: ``L1`` for a
    complex one-dimensional row without time reversal.  Every other name is
    one a table prints.
    """
    label = str(label)
    head = label[1:2]
    return any(f"_{letter}" in label for letter in "abcdefgh") or (
        label.startswith("L") and head.isdigit())


def _mulliken(chars, merged, class_infos, cls_of_identity):
    """The Mulliken symbol of every row of a character table.

    ``chars`` holds the characters class by class, ``merged`` which rows are
    merged complex pairs, ``class_infos`` :func:`_class_info` of each class and
    ``cls_of_identity`` the identity's class.  A one-dimensional row is ``A``
    where the principal rotation (:func:`_principal_axis`) has character
    ``+1``, ``B`` where it has ``-1`` and ``L1`` otherwise; in a cubic group,
    which has several principal axes, it is ``A``.  A row of dimension two is
    ``E``, ``1E2E`` where it is a merged pair, and of dimension three ``T``;
    any other row is ``L`` and its dimension.  ``g`` or ``u`` follows the
    inversion's character, and in a group without the inversion ``'`` or
    ``''`` follows the horizontal mirror's, or in the group ``m`` its one
    mirror's.  Rows that would still share a name take the subscript ``1`` or
    ``2`` from the class :func:`_secondary_reference` gives, else ``1``,
    ``2``, ... in decreasing order of the principal rotation's character,
    else ``_a``, ``_b``, ....  Returns the names in the order of the rows.
    """
    principal = _principal_axis(class_infos)
    sec = _secondary_reference(class_infos, principal)
    prin_cls = None
    if principal is not None:
        for i, (_, info, _) in enumerate(class_infos):
            if info["kind"] == "C" and info["order"] == principal["order"] \
                    and np.allclose(info["axis"], principal["axis"], atol=1e-6):
                prin_cls = i
                break
    inv_cls = next((i for i, (_, info, _) in enumerate(class_infos) if info["kind"] == "i"), None)
    mirrors = [i for i, (_, info, _) in enumerate(class_infos) if info["kind"] == "m"]
    if principal is not None:
        hor_cls = next((i for i in mirrors
                        if abs(abs(float(class_infos[i][1]["axis"]
                                         @ principal["axis"])) - 1.0) < 1e-6), None)
    else:
        # With no rotation axis the group is `1`, `-1` or `m`.  In `m` the one
        # mirror is all the two rows differ on, so it sets the parity and the
        # rows read `A'` and `A''`; `-1` takes its parity from the inversion
        # above, and `1` has the one row `A`.
        hor_cls = mirrors[0] if len(mirrors) == 1 else None

    # A cubic group has several axes of its highest-order rotation, and then
    # Mulliken does not use A-versus-B at all: both one-dimensional irreps are
    # A, separated by the subscript (O: A1, A2).
    axes = set()
    if principal is not None:
        for _, info, members in class_infos:
            for _, mem in members:
                if mem["kind"] == "C" and mem["order"] == principal["order"]:
                    axes.add(tuple(np.round(np.abs(mem["axis"]), 6)))
    cubic = len(axes) > 1

    bases, parities = [], []
    for p in range(chars.shape[0]):
        dim = int(round(float(chars[p, cls_of_identity].real)))
        if merged[p]:
            base = "1E2E" if dim == 2 else f"L{dim}"
        elif dim == 1:
            x = chars[p, prin_cls] if prin_cls is not None else 1.0
            if cubic:
                base = "A"                    # cubic: A-versus-B is not used
            else:
                base = "A" if abs(x - 1.0) < 1e-6 else ("B" if abs(x + 1.0) < 1e-6 else "L1")
        elif dim == 2:
            base = "E"
        elif dim == 3:
            base = "T"
        else:
            base = f"L{dim}"
        parity = ""
        if inv_cls is not None:
            y = chars[p, inv_cls] / chars[p, cls_of_identity]
            parity = "g" if abs(y - 1.0) < 1e-6 else ("u" if abs(y + 1.0) < 1e-6 else "")
        elif hor_cls is not None:
            y = chars[p, hor_cls] / chars[p, cls_of_identity]
            parity = "'" if abs(y - 1.0) < 1e-6 else ("''" if abs(y + 1.0) < 1e-6 else "")
        bases.append(base)
        parities.append(parity)

    # A subscript is given only to rows that would otherwise share a name, so
    # the group `23` with time reversal reads `A`, `1E2E` and `T`, and `4mm`
    # reads `A1`, `A2`, `B1`, `B2` and `E`.
    subs = [""] * len(bases)
    groups = {}
    for p, key in enumerate(zip(bases, parities)):
        groups.setdefault(key, []).append(p)
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        assigned = None
        if sec is not None:                    # the C2'/sigma_v rule
            trial = {}
            for p in rows:
                y = chars[p, sec] / chars[p, cls_of_identity]
                trial[p] = "1" if abs(y - 1.0) < 1e-6 else ("2" if abs(y + 1.0) < 1e-6 else None)
            if None not in trial.values() and len(set(trial.values())) == len(rows):
                assigned = trial
        if assigned is None and prin_cls is not None:   # order by chi(principal)
            keyed = sorted(rows, key=lambda p: (-round(float(chars[p, prin_cls].real), 6),
                                                -round(float(chars[p, prin_cls].imag), 6)))
            vals = [round(complex(chars[p, prin_cls]).real, 6) for p in keyed]
            if len(set(vals)) == len(rows):
                assigned = {p: str(i + 1) for i, p in enumerate(keyed)}
        if assigned is None:
            assigned = {p: f"_{'abcdefgh'[i]}" for i, p in enumerate(rows)}
        for p, s in assigned.items():
            subs[p] = s
    return tuple(b + s + par for b, s, par in zip(bases, subs, parities))


# --------------------------------------------------------------------------
# 4.  building the table for one site
# --------------------------------------------------------------------------

def site_group_table(group, indices, antiunitary=(), seed=101) -> SiteGroupTable:
    """The character table of the site group whose operations are ``indices``.

    ``indices`` holds the ``.ind`` in ``group`` of the unitary operations that
    fix the site, and ``antiunitary`` of the antiunitary ones; ``seed`` seeds
    :func:`_character_table`.  The characters are computed from ``group``'s
    product table, and a complex-conjugate pair is merged where
    ``antiunitary`` is nonempty.  Returns the :class:`SiteGroupTable` with each
    row under its Mulliken symbol (:func:`_mulliken`), ``named_by`` reading
    ``"point group"`` and no :attr:`SiteGroupTable.declarable` names.
    :func:`.._naming._site_group_table` and :func:`.._naming._site_tables`
    rename the rows with the tables' names.

    Raises
    ------
    DeclarationError
        ``gate="site_group"`` where the operations are not closed under the
        product table, or where the table's deviation from orthogonality
        (:attr:`SiteGroupTable.orthogonality`) exceeds ``1e-6`` times the
        site group's order.
    """
    prod, _, _ = group._get_product_table(get_diff=True)
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    positions = [pos_of[int(i)] for i in indices]
    T = _local_table(prod, positions)
    if T is None:
        raise DeclarationError(
            "site_group",
            f"the site-symmetry operations {tuple(int(i) for i in indices)} are not closed "
            "under the group's product. Report it, with the model and the call.",
            {"indices": tuple(int(i) for i in indices)})
    cls_local, cls_of_local, e_local, chars, defect = _character_table(T, seed=seed)
    if defect > 1e-6 * max(1, len(positions)):
        raise DeclarationError(
            "site_group",
            f"the character table computed for the site group departs from orthogonality "
            f"by {defect:.3g}. Report it, with the model and the call.",
            {"indices": tuple(int(i) for i in indices), "orthogonality": defect})
    chars, merged = _merge_conjugate_pairs(chars) if antiunitary else (chars, (False,) * len(chars))

    lattice = np.asarray(group._lattice, dtype=float)
    ind_of = [int(group._symmetries[p].ind) for p in positions]
    classes = tuple(tuple(ind_of[g] for g in c) for c in cls_local)
    class_of = {ind_of[g]: cls_of_local[g] for g in range(len(positions))}
    names = {ind_of[g]: operation_name(group._symmetries[positions[g]], lattice)
             for g in range(len(positions))}
    infos = [_class_info(group, c) for c in classes]
    class_names = tuple((f"{len(c)}" if len(c) > 1 else "") + names[int(rep)]
                        for c, (rep, _, _) in zip(classes, infos))
    labels = _mulliken(chars, merged, infos, cls_of_local[e_local])
    return SiteGroupTable(
        indices=tuple(ind_of), order=len(ind_of), classes=classes,
        class_names=class_names, class_reps=tuple(int(rep) for rep, _, _ in infos),
        class_of=class_of, characters=chars, labels=labels,
        dims=np.rint(chars[:, cls_of_local[e_local]].real).astype(int),
        names=names, orthogonality=defect, merged=merged,
        antiunitary=tuple(int(i) for i in antiunitary),
        named_by=("point group",) * len(labels))


#: Every crystallographic point group, keyed by how many operations of each
#: kind and order it holds: ``E``, ``i``, a mirror ``m``, a proper rotation
#: ``Cn`` and a rotoinversion ``Sn`` (a turn by ``2π/n``, then the inversion:
#: ``-n``).  Those counts separate all thirty-two, so a measured site group is
#: named from :func:`classify` alone.  The names are Hermann-Mauguin symbols,
#: as a Wyckoff position's site symmetry is written.
_POINT_GROUP_SYMBOLS = {
    frozenset({("E", 1, 1)}): "1",
    frozenset({("E", 1, 1), ("i", 2, 1)}): "-1",
    frozenset({("E", 1, 1), ("C", 2, 1)}): "2",
    frozenset({("E", 1, 1), ("m", 2, 1)}): "m",
    frozenset({("E", 1, 1), ("C", 2, 1), ("i", 2, 1), ("m", 2, 1)}): "2/m",
    frozenset({("E", 1, 1), ("C", 2, 3)}): "222",
    frozenset({("E", 1, 1), ("C", 2, 1), ("m", 2, 2)}): "mm2",
    frozenset({("E", 1, 1), ("C", 2, 3), ("i", 2, 1), ("m", 2, 3)}): "mmm",
    frozenset({("E", 1, 1), ("C", 4, 2), ("C", 2, 1)}): "4",
    frozenset({("E", 1, 1), ("S", 4, 2), ("C", 2, 1)}): "-4",
    frozenset({("E", 1, 1), ("C", 4, 2), ("C", 2, 1), ("i", 2, 1),
               ("S", 4, 2), ("m", 2, 1)}): "4/m",
    frozenset({("E", 1, 1), ("C", 4, 2), ("C", 2, 5)}): "422",
    frozenset({("E", 1, 1), ("C", 4, 2), ("C", 2, 1), ("m", 2, 4)}): "4mm",
    frozenset({("E", 1, 1), ("S", 4, 2), ("C", 2, 3), ("m", 2, 2)}): "-42m",
    frozenset({("E", 1, 1), ("C", 4, 2), ("C", 2, 5), ("i", 2, 1),
               ("S", 4, 2), ("m", 2, 5)}): "4/mmm",
    frozenset({("E", 1, 1), ("C", 3, 2)}): "3",
    frozenset({("E", 1, 1), ("C", 3, 2), ("i", 2, 1), ("S", 3, 2)}): "-3",
    frozenset({("E", 1, 1), ("C", 3, 2), ("C", 2, 3)}): "32",
    frozenset({("E", 1, 1), ("C", 3, 2), ("m", 2, 3)}): "3m",
    frozenset({("E", 1, 1), ("C", 3, 2), ("C", 2, 3), ("i", 2, 1),
               ("S", 3, 2), ("m", 2, 3)}): "-3m",
    frozenset({("E", 1, 1), ("C", 6, 2), ("C", 3, 2), ("C", 2, 1)}): "6",
    frozenset({("E", 1, 1), ("S", 6, 2), ("C", 3, 2), ("m", 2, 1)}): "-6",
    frozenset({("E", 1, 1), ("C", 6, 2), ("C", 3, 2), ("C", 2, 1), ("i", 2, 1),
               ("S", 3, 2), ("S", 6, 2), ("m", 2, 1)}): "6/m",
    frozenset({("E", 1, 1), ("C", 6, 2), ("C", 3, 2), ("C", 2, 7)}): "622",
    frozenset({("E", 1, 1), ("C", 6, 2), ("C", 3, 2), ("C", 2, 1),
               ("m", 2, 6)}): "6mm",
    frozenset({("E", 1, 1), ("S", 6, 2), ("C", 3, 2), ("C", 2, 3),
               ("m", 2, 4)}): "-6m2",
    frozenset({("E", 1, 1), ("C", 6, 2), ("C", 3, 2), ("C", 2, 7), ("i", 2, 1),
               ("S", 3, 2), ("S", 6, 2), ("m", 2, 7)}): "6/mmm",
    frozenset({("E", 1, 1), ("C", 3, 8), ("C", 2, 3)}): "23",
    frozenset({("E", 1, 1), ("C", 3, 8), ("C", 2, 3), ("i", 2, 1),
               ("S", 3, 8), ("m", 2, 3)}): "m-3",
    frozenset({("E", 1, 1), ("C", 4, 6), ("C", 3, 8), ("C", 2, 9)}): "432",
    frozenset({("E", 1, 1), ("S", 4, 6), ("C", 3, 8), ("C", 2, 3),
               ("m", 2, 6)}): "-43m",
    frozenset({("E", 1, 1), ("C", 4, 6), ("C", 3, 8), ("C", 2, 9), ("i", 2, 1),
               ("S", 4, 6), ("S", 3, 8), ("m", 2, 9)}): "m-3m",
}


def point_group_symbol(group, indices) -> str:
    """The Hermann-Mauguin symbol of the operations ``indices`` of ``group``.

    ``indices`` holds the operations' ``.ind`` values, and their Cartesian
    matrices are read by :func:`point_group_of`.
    """
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    return point_group_of([group._symmetries[pos_of[int(ind)]].rotation_cart
                           for ind in indices])


def point_group_of(matrices) -> str:
    """The Hermann-Mauguin symbol of the point group of the ``matrices``.

    An operation's kind and order are read from its trace and determinant
    (:func:`_classified`), so the matrices may be written in any one basis,
    the Cartesian one or a lattice basis.  The counts of each kind and order
    are looked up in :data:`_POINT_GROUP_SYMBOLS`.  Returns
    ``"a group of order n"`` for a set that is none of the thirty-two.
    """
    counts: dict = {}
    matrices = list(matrices)
    for matrix in matrices:
        info = _classified(matrix)
        key = (str(info["kind"]), int(info["order"]))
        counts[key] = counts.get(key, 0) + 1
    sig = frozenset((kind, order, n) for (kind, order), n in counts.items())
    return _POINT_GROUP_SYMBOLS.get(sig, f"a group of order {len(matrices)}")


@dataclass(frozen=True)
class DeclaredContent:
    """A multiplet's declared site irreps, in the group they were read in.

    The declaration is carried as characters keyed by operation, and each
    later group re-keys them (:meth:`rekeyed`) rather than reading the
    declaration again.  Operations a smaller group lacks drop out, which
    restricts the declared irrep to the smaller site group, so a restriction
    that has become reducible, ``E`` of ``3m`` on ``3``, is still carried.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals that carry the declared site irreps together, one
        multiplet.
    spec : object
        The declaration's value as written.
    label : str
        The site irrep, or sum of them, the declaration resolved to, named in
        the site group it was read in.
    dim : int
        The multiplet's dimension, that of its orbital factor on a spinful
        site.
    declared_in : str
        The site group and the group the declaration was read in,
        ``"3m in P3m11' #156.50"``.
    constraints : dict
        The declared character on each operation that fixes the site, keyed
        by its ``.ind`` in the group the record is now on.
    note : str
        If nonempty, how the declaration was resolved, and which row
        Mulliken's convention calls each name of :attr:`mulliken`.
    broke : tuple of str
        The operations the declaration was measured to break, named in the
        model's Cartesian axes.  Empty until the group is settled.
    functions : tuple of dict
        If nonempty, the Cartesian polynomial of each orbital function
        declared by name, ``px`` or ``dxy``, as :data:`SHORTHANDS` writes it.
        A shell, a site irrep, a sum and generator characters or matrices
        leave it empty.
    reduced_in : str
        If nonempty, the site group and the group the declaration was resolved
        in after it lowered the cell's group, ``"m in Pm1' #6.4"``.  Empty
        where the declaration left the cell's group alone, and
        :attr:`declared_in` is then where it was both read and resolved.
    words : tuple
        If nonempty, ``(orbital, word)`` for each orbital of :attr:`orbitals`
        declared as an orbital function, the word in its canonical spelling
        (:func:`canonical_shorthand`): ``(0, "px")`` where ``"p_x"`` was
        written, which :attr:`written` holds.  A shell, ``"p"``, is the word
        of every orbital it was written on.
    written : tuple
        ``(orbital, spec)`` for every orbital of :attr:`orbitals`, what
        ``characters=`` wrote for it: ``"px"`` for one orbital and ``"py"``
        for its partner, where :attr:`spec` is one entry's.
    corep : bool
        Whether the declaration was written in the double-valued site irreps
        the rows are returned as (:func:`named_in_coreps`); such a
        declaration states their multiplicities alone.  :attr:`label` then
        names those double-valued site irreps.  :attr:`constraints` holds the
        orbital characters that become them with spin and rank lowest under
        the lowest-angular-momentum rule (:func:`_resolve_sum`).  Every
        orbital character that becomes them with spin is the same wherever
        the spin-1/2 lift has a nonzero trace.
    mulliken : tuple
        If nonempty, ``(name, Mulliken's name)`` for each name whose row
        Mulliken's convention names differently at this site
        (:func:`mulliken_names`): ``(("A1''", "A2'"),)`` at ``2c`` of
        ``P6/mmm``.
    matrices : dict
        If nonempty, the declared generator matrices: each operation's matrix
        on the orbital factor of :attr:`orbitals`, in their order, for every
        operation of the site group the declared ones generate
        (:func:`generator_matrices`), keyed as :attr:`constraints` is.
        :attr:`constraints` holds their traces.
    """

    orbitals: tuple
    spec: Any
    label: str
    dim: int
    declared_in: str
    constraints: dict
    note: str = ""
    broke: tuple = ()
    functions: tuple = ()
    reduced_in: str = ""
    words: tuple = ()
    written: tuple = ()
    corep: bool = False
    mulliken: tuple = ()
    matrices: dict = field(default_factory=dict)

    def rekeyed(self, pairs) -> "DeclaredContent":
        """The same declaration on another group, re-keyed through ``pairs``.

        ``pairs`` holds an ``(old, new)`` pair of ``.ind`` values for each
        operation the two groups share.  An operation with no pair drops out
        of :attr:`constraints` and :attr:`matrices`.
        """
        out, matrices = {}, {}
        for old, new in pairs:
            if int(old) in self.constraints:
                out[int(new)] = complex(self.constraints[int(old)])
            if int(old) in self.matrices:
                matrices[int(new)] = np.array(self.matrices[int(old)], dtype=complex)
        return replace(self, constraints=out, matrices=matrices)


def generator_frame(table) -> str:
    """Each operation of the site group whose table is ``table``, on one line.

    ``#1 = E; #2 = C3(z); #3 = C3^-1(z); ...``, each operation by its ``.ind``
    and its name.
    """
    return "; ".join(f"#{ind} = {table.names[ind]}" for ind in table.indices)


def _declared_name(table, row) -> str:
    """The name a declaration resolving to ``row`` of ``table`` is echoed as.

    The row's :attr:`SiteGroupTable.declarable` name where it has one, the
    orbital factor's word on a spinful site, and its label otherwise.
    """
    if row < len(table.declarable) and table.declarable[row]:
        return table.declarable[row]
    return table.labels[row]


def _listed(table) -> tuple:
    """The rows of ``table`` in the order the tables list a site's irreps.

    The tables list the site irreps of a Wyckoff position in the order of
    their names, ``A1, A2, B1, B2`` at an ``mm2`` site.  Each row is placed by
    its single-valued name, the :attr:`SiteGroupTable.declarable` one on a
    spinful site, and rows with no name come last.  Returns the row indices.
    """
    own = table.declarable if any(table.declarable) else table.labels
    return tuple(sorted(range(len(table.labels)),
                        key=lambda p: (not own[p], own[p] or "", p)))


def declaration_words(table) -> tuple:
    """Every name a declaration at the site group ``table`` may be written in.

    Each row's label, and beside it its :attr:`SiteGroupTable.declarable`
    name where it has one, the rows in the order :func:`_listed` gives and
    each name once.  A row with neither name is declared by its generator
    characters.
    """
    out: list = []
    for row in _listed(table):
        for name in (table.labels[row],
                     table.declarable[row] if row < len(table.declarable) else ""):
            if name and name not in out:
                out.append(name)
    return tuple(out)


def fitting_words(table, n) -> tuple:
    """The names of the site irreps that fit on a site holding ``n`` orbitals.

    ``table`` is the site's table.  ``n`` counts the site's orbitals, or on a
    spinful site its Wannier functions.  The site irreps of dimension at most
    ``n`` come first, by the name a declaration writes them in
    (:func:`_declared_name`), in order of dimension, the one nearest the
    trivial character first (:func:`_row_distance`), then in the order
    :func:`_listed` gives.  On a spinful site their labels follow, the
    double-valued site irrep, or sum of them, each irrep becomes with spin:
    ``A1``, ``A2`` and ``-E1`` at a ``3m`` site holding one Wannier function.
    Each name stands once.
    """
    rank = {p: i for i, p in enumerate(_listed(table))}
    rows = sorted((p for p in range(len(table.labels)) if int(table.dims[p]) <= int(n)),
                  key=lambda p: (int(table.dims[p]), _row_distance(table, p), rank[p]))
    out: list = []
    for p in rows:
        word = _declared_name(table, p)
        if word and word not in out:
            out.append(word)
    if any(table.declarable):
        # with spin, a row's label is its irrep times the spinor
        for p in rows:
            word = table.labels[p]
            if word and word not in out:
                out.append(word)
    return tuple(out)


def table_summary(table) -> str:
    """The character table ``table`` as text, for a refusal's reason.

    One line a row, in the order :func:`_listed` gives, under its
    :func:`_declared_name` and with its character on each class.  On a spinful
    site a last column, ``with spin``, gives the double-valued site irrep, or
    sum of them, the row becomes with spin.  A row with no name reads ``?``.
    """
    spinful = any(table.declarable)
    rows = _listed(table)
    names = [_declared_name(table, p) or "?" for p in rows]
    cells = [[_fmt(z) for z in table.characters[p]] for p in rows]
    widths = [max([len(c)] + [len(r[k]) for r in cells])
              for k, c in enumerate(table.class_names)]
    first = max(len(n) for n in names)
    lines = ["  " + " " * first + "".join(f"  {c:>{w}s}"
                                         for c, w in zip(table.class_names, widths))
             + ("  with spin" if spinful else "")]
    for p, name, cell in zip(rows, names, cells):
        lines.append(f"  {name:<{first}s}" + "".join(f"  {v:>{w}s}"
                                                    for v, w in zip(cell, widths))
                     + (f"  {table.labels[p] or '?'}" if spinful else ""))
    return "\n".join(lines)


def _fmt(z) -> str:
    """The character ``z`` as text, to three significant figures.

    A real or imaginary part below ``1e-8`` reads ``0``.
    """
    z = complex(z)
    re, im = (0.0 if abs(x) < 1e-8 else x for x in (z.real, z.imag))
    if im == 0.0:
        return f"{re:.3g}"
    return f"{re:.3g}{im:+.3g}j"


# --------------------------------------------------------------------------
# 5.  resolving a declaration against a site group
# --------------------------------------------------------------------------

#: Cartesian polynomials the physics shorthands stand for.  These are
#: *definitions of the orbital names*, not group data: the character each one
#: carries is computed by acting with the measured operations on the polynomial.
SHORTHANDS = {
    "s": {(0, 0, 0): 1.0},
    "px": {(1, 0, 0): 1.0}, "py": {(0, 1, 0): 1.0}, "pz": {(0, 0, 1): 1.0},
    "x": {(1, 0, 0): 1.0}, "y": {(0, 1, 0): 1.0}, "z": {(0, 0, 1): 1.0},
    "dxy": {(1, 1, 0): 1.0}, "dyz": {(0, 1, 1): 1.0},
    "dzx": {(1, 0, 1): 1.0}, "dxz": {(1, 0, 1): 1.0},
    "dx2-y2": {(2, 0, 0): 1.0, (0, 2, 0): -1.0},
    "dz2": {(0, 0, 2): 2.0, (2, 0, 0): -1.0, (0, 2, 0): -1.0},
    "x2-y2": {(2, 0, 0): 1.0, (0, 2, 0): -1.0},
    "z2": {(0, 0, 2): 2.0, (2, 0, 0): -1.0, (0, 2, 0): -1.0},
    "xy": {(1, 1, 0): 1.0}, "yz": {(0, 1, 1): 1.0}, "zx": {(1, 0, 1): 1.0},
}

#: The spellings worth printing when a label is not recognized (the rest of
#: :data:`SHORTHANDS` are aliases of these).
CANONICAL_SHORTHANDS = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "dz2")

#: The shells a declaration may name whole, each by its angular momentum ``l``:
#: the site irreps the real harmonics of degree ``l`` carry at the site,
#: whatever their orientation inside it.  ``"p"`` is ``A1 + E`` at a ``3m``
#: site and ``T2`` at a ``-43m`` one.  ``"s"`` is also the one ``s`` function,
#: and is read as that shorthand.
SHELLS = {"s": 0, "p": 1, "d": 2, "f": 3}


@lru_cache(maxsize=None)
def _monomials(degree):
    return tuple((a, b, degree - a - b)
                 for a in range(degree + 1) for b in range(degree + 1 - a))


@lru_cache(maxsize=None)
def _monomial_index(degree):
    """The column of each monomial of degree ``degree``, by its exponents."""
    return {m: i for i, m in enumerate(_monomials(degree))}


def _apply(R_inv, poly, degree):
    """The polynomial ``f(R^-1 r)``, for ``f`` the polynomial ``poly``.

    ``R_inv`` is the inverse of the operation's Cartesian matrix.  ``poly`` is
    of degree ``degree`` and maps exponents to coefficients, as
    :data:`SHORTHANDS` writes a polynomial.  Returns the coefficients in the
    columns :func:`_monomial_index` gives.
    """
    idx = _monomial_index(degree)
    out = np.zeros(len(idx))
    for expo, coef in poly.items():
        # product over variables of (sum_j R_inv[i, j] r_j) ** expo[i]
        terms = {(0, 0, 0): float(coef)}
        for i, k in enumerate(expo):
            for _ in range(k):
                new = {}
                for m, c in terms.items():
                    for j in range(3):
                        if abs(R_inv[i, j]) < 1e-12:
                            continue
                        mm = list(m)
                        mm[j] += 1
                        mm = tuple(mm)
                        new[mm] = new.get(mm, 0.0) + c * float(R_inv[i, j])
                terms = new
        for m, c in terms.items():
            out[idx[m]] += c
    return out


def shorthand_key(spec):
    """The :data:`SHORTHANDS` key the word ``spec`` names, or ``None``.

    The spelling is normalized as :func:`resolve_spec` normalizes it:
    lowercased, with spaces and underscores dropped.  ``None`` also for
    anything not a string.
    """
    if not isinstance(spec, str):
        return None
    key = str(spec).strip().lower().replace(" ", "").replace("_", "")
    return key if key in SHORTHANDS else None


def canonical_shorthand(spec) -> str:
    """The canonical spelling of the orbital function ``spec``, or ``""``.

    The spelling is the one :data:`CANONICAL_SHORTHANDS` gives the same
    polynomial: ``"p_x"`` and ``"x"`` are ``"px"``, and ``"dxz"`` is
    ``"dzx"``.  It is the word a returned row prints as its function.
    """
    key = shorthand_key(spec)
    if key is None:
        return ""
    poly = SHORTHANDS[key]
    return next((name for name in CANONICAL_SHORTHANDS if SHORTHANDS[name] == poly), key)


def span_basis(polys):
    """An orthonormal basis of the span of the Cartesian polynomials ``polys``.

    Each polynomial maps exponents to coefficients and is homogeneous, and an
    operation preserves the degree, so the span is the direct sum of its
    pieces of each degree.  Returns ``{degree: rows}``, the rows orthonormal
    in the monomial basis of their degree (:func:`_monomials`).
    """
    by_degree: dict = {}
    for poly in polys:
        by_degree.setdefault(max(sum(k) for k in poly), []).append(poly)
    out = {}
    for degree, group_of_polys in by_degree.items():
        basis = _monomials(degree)
        idx = {m: i for i, m in enumerate(basis)}
        V = np.zeros((len(group_of_polys), len(basis)))
        for r, poly in enumerate(group_of_polys):
            for m, c in poly.items():
                V[r, idx[m]] += float(c)
        _u, s, vh = np.linalg.svd(V, full_matrices=False)
        rank = int(np.sum(s > 1e-8 * max(1.0, float(s[0]) if len(s) else 1.0)))
        out[int(degree)] = np.asarray(vh[:rank], dtype=float)
    return out


def carries_span(group, ind, source, target, tol=1e-8) -> bool:
    r"""Whether one operation carries the span ``source`` onto ``target``.

    The operation is the one of ``group`` whose ``.ind`` is ``ind``, acting
    as ``f(r) -> f(R^-1 r)`` (:func:`_apply`).  ``source`` and ``target`` are
    spans as :func:`span_basis` returns them.  The operation carries one onto
    the other where the two have the same degrees and the same dimension in
    each, and the image of every basis function of ``source`` lies in
    ``target``, its residual at most ``tol`` times ``max(1, |image|)``.  Onto,
    not into, so that the operations it keeps are closed under products.
    """
    if set(source) != set(target):
        return False
    if any(source[d].shape[0] != target[d].shape[0] for d in source):
        return False
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    R_inv = np.linalg.inv(np.asarray(
        group._symmetries[pos_of[int(ind)]].rotation_cart, dtype=float))
    for degree, B in source.items():
        T = target[degree]
        for row in B:
            image = _apply(R_inv, _poly_of(row, degree), degree)
            residual = float(np.linalg.norm(image - T.T @ (T @ image)))
            if residual > tol * max(1.0, float(np.linalg.norm(image))):
                return False
    return True


def span_character(group, table, polys):
    """The character of the smallest invariant span holding the ``polys``.

    ``polys`` are Cartesian polynomials, each homogeneous and written as
    :data:`SHORTHANDS` writes one.  ``table`` is the site's table, and
    ``group`` the represented group its operations belong to.  Each operation
    of the site group is applied to each polynomial (:func:`_apply`), and the
    images span the smallest invariant space holding the ``polys``.  The
    polynomials of each degree are read apart and their characters added.
    Returns ``(chi, dim)``: ``chi`` each operation's trace on the span, one
    value per class of ``table``, and ``dim`` the span's dimension.
    """
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    mats = {int(ind): np.linalg.inv(np.asarray(
        group._symmetries[pos_of[int(ind)]].rotation_cart, dtype=float))
        for ind in table.indices}
    chi = np.zeros(len(table.classes), dtype=complex)
    rank = 0
    degrees: dict = {}
    for poly in polys:
        degrees.setdefault(max(sum(k) for k in poly), []).append(poly)
    for degree, graded in degrees.items():
        vecs = [_apply(mats[int(ind)], poly, degree)
                for poly in graded for ind in table.indices]
        _u, s, vh = np.linalg.svd(np.asarray(vecs))
        part = int(np.sum(s > 1e-8 * max(1.0, float(s[0]))))
        basis = vh[:part]                               # (part, n_monomial)
        rank += part
        for c, cls in enumerate(table.classes):
            ind = int(cls[0])
            images = np.asarray([_apply(mats[ind], _poly_of(basis[i], degree), degree)
                                 for i in range(part)])
            chi[c] += complex(np.trace(images @ basis.T))
    return chi, rank


def shorthand_character(group, table, name):
    """The character of the span the shorthand ``name`` generates at a site.

    ``name`` is a key of :data:`SHORTHANDS`.  ``group`` and ``table`` are as
    :func:`span_character` takes them, and it returns what that call returns.
    """
    return span_character(group, table, [SHORTHANDS[name]])


def shell_key(spec):
    """The :data:`SHELLS` key the word ``spec`` names, or ``None``.

    The spelling is normalized as :func:`shorthand_key` normalizes it.
    ``None`` also for anything not a string.
    """
    if not isinstance(spec, str):
        return None
    key = str(spec).strip().lower().replace(" ", "").replace("_", "")
    return key if key in SHELLS else None


def shell_character(group, table, degree) -> np.ndarray:
    """The character of the real harmonics of one degree at a site.

    The harmonics of degree ``degree`` are the polynomials of that degree
    less ``r^2`` times those of degree ``degree - 2``, so their character is
    the difference of the two spans' characters (:func:`span_character`).
    ``group`` and ``table`` are as that call takes them.  Returns the
    character, one value per class of ``table``: at a ``3m`` site, degree 1
    gives the character of ``A1 + E``.
    """
    def _degree(d):
        if d < 0:
            return np.zeros(len(table.classes), dtype=complex)
        chi, _rank = span_character(group, table, [{m: 1.0} for m in _monomials(int(d))])
        return chi

    return _degree(int(degree)) - _degree(int(degree) - 2)


def content_label(table, chi) -> str:
    """The name of a sum of site irreps, as a declaration writes it.

    ``chi`` is the sum's character, one value per class of ``table``.  Each
    irrep is named by :func:`_declared_name`, with the single-valued name on
    a spinful site.  Each appears once, its multiplicity before it where
    above one: ``"A1 + E"``, ``"2xE"``.  Returns ``""`` where ``chi`` is no
    sum of the table's rows, or holds a row with no name.
    """
    rows = _content_rows(table, chi)
    if rows is None:
        return ""
    parts = []
    for p, k in rows.items():
        name = _declared_name(table, p)
        if not name:
            return ""
        parts.append((f"{k}x" if k > 1 else "") + name)
    return " + ".join(parts)


def corep_label(table, chi) -> str:
    """The name of the double-valued site irreps a sum becomes with spin.

    ``chi`` is the orbital character of a sum of site irreps, one value per
    class of ``table``, a spinful site's table.  Each row's label is the
    double-valued site irrep, or sum of them, that the row becomes with spin,
    and the double-valued irreps are counted across the rows: ``A1 + E`` at a
    ``3m`` site is ``"2x-E1 + -1E2E"``, and ``A1`` and ``A2`` are both
    ``"-E1"``, as an orbital's row of the atomic-limit table writes them
    (:attr:`..symmetry._OrbitalRow.irrep`).  Returns ``""`` where ``chi`` is
    no sum of the table's rows, or holds a row with no label.
    """
    rows = _content_rows(table, chi)
    if rows is None:
        return ""
    counts = {}
    for p, k in rows.items():
        if not table.labels[p]:
            return ""
        for word, j in (_sum_terms(table.labels[p]) or [(table.labels[p], 1)]):
            counts[word] = counts.get(word, 0) + k * j
    return " + ".join(word if k == 1 else f"{k}x{word}" for word, k in counts.items())


def _content_rows(table, chi):
    """The rows of ``table`` a character holds, ``{row: copies}``, or ``None``.

    ``chi`` is the character, one value per class.  Returns ``None`` where
    its multiplicities (:func:`decompose`) are not non-negative integers.
    """
    rows = {}
    for p, n in enumerate(decompose(table, np.asarray(chi, dtype=complex))):
        n = complex(n)
        if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or round(n.real) < 0:
            return None
        if round(n.real):
            rows[p] = int(round(n.real))
    return rows


def _sum_terms(spec):
    """The terms of a declared sum of irreps, ``[(word, copies), ...]``.

    ``spec`` is the declared value.  ``"A1 + E"`` is two terms, ``"2xE"``
    one term of two copies, and ``"2x-E1 + -1E2E"``, as an orbital's row of
    the atomic-limit table writes it, ``-E1`` twice and ``-1E2E`` once.
    Returns ``None`` for one word of one copy, for a sum with an empty term,
    and for anything not a string.
    """
    if not isinstance(spec, str):
        return None
    terms = []
    for part in str(spec).split("+"):
        part = part.strip()
        if not part:
            return None
        copies, sep, rest = part.partition("x")
        if sep and copies.strip().isdigit() and rest.strip():
            terms.append((rest.strip(), int(copies)))
        else:
            terms.append((part, 1))
    if len(terms) == 1 and terms[0][1] == 1:
        return None
    return terms


def named_in_coreps(table, spec) -> bool:
    """Whether ``spec`` names the double-valued site irreps of ``table``.

    On a spinful site a declaration can name the orbital factor (``A1``,
    ``A1 + E``) or the double-valued site irreps the rows are returned as
    (``-E1``, ``2x-E1 + -1E2E``).  ``spec`` is of the second kind where a
    word of it is no single-valued name (:func:`_orbital_row`) and is a
    double-valued irrep some row becomes with spin.  Such a declaration
    states the multiplicities of the double-valued irreps alone: ``A1`` and
    ``A2`` of a ``3m`` site both become ``-E1``.  Returns ``False`` for
    anything not a string, and where the table has no single-valued names
    (:attr:`SiteGroupTable.declarable`), as on a spinless site.
    """
    if not isinstance(spec, str) or not any(table.declarable):
        return False
    words = _corep_words(table)
    terms = _sum_terms(spec) or [(str(spec).strip(), 1)]
    return any(_orbital_row(table, word) < 0 and str(word).strip().lower() in words
               for word, _k in terms)


def _corep_words(table) -> frozenset:
    """The double-valued site irreps the rows of ``table`` become with spin.

    Returns their names, in lower case.
    """
    out = set()
    for label in table.labels:
        if label:
            for word, _k in (_sum_terms(label) or [(label, 1)]):
                out.add(str(word).strip().lower())
    return frozenset(out)


def lift_squares(rotations) -> dict:
    """The squared trace of each operation's spin-1/2 lift.

    ``rotations`` maps each operation's ``.ind`` to its Cartesian matrix
    ``R``.  The lift of a turn by ``θ`` has trace ``±2 cos(θ/2)``, its sign a
    convention, and an improper operation's lift is its proper part's.  So
    ``|tr S(g)|^2 = 2 + 2 cos θ = 1 + tr(det(R) R)``.  It is ``4`` on the
    identity and the inversion, ``1`` on a threefold, and ``0`` on a twofold
    and on a mirror, where every orbital character times the lift vanishes.
    Returns ``{ind: |tr S(g)|^2}``, never below zero.
    """
    out = {}
    for ind, R in rotations.items():
        R = np.asarray(R, dtype=float)
        det = 1.0 if float(np.linalg.det(R)) > 0.0 else -1.0
        out[int(ind)] = max(0.0, 1.0 + float(np.trace(det * R)))
    return out


def _corep_matrix(table):
    """The double-valued site irreps each row of ``table`` holds with spin.

    They are read off the rows' labels, each a double-valued irrep or a sum
    of them.  Returns ``(names, N)``: ``names`` the double-valued irreps in
    lower case, in the order the rows first name them, and ``N[p, c]`` how
    often row ``p`` with spin holds ``names[c]``.  Returns ``None`` where a
    row of ``table`` has no label.
    """
    names, rows = [], []
    for label in table.labels:
        if not label:
            return None
        got = {}
        for word, k in (_sum_terms(label) or [(label, 1)]):
            key = str(word).strip().lower()
            got[key] = got.get(key, 0) + int(k)
            if key not in names:
                names.append(key)
        rows.append(got)
    return names, np.array([[got.get(n, 0) for n in names] for got in rows], dtype=float)


def corep_counts(table, rows):
    """The double-valued site irreps a sum of site irreps becomes with spin.

    ``rows`` is the sum as ``{row: copies}`` of ``table``, a spinful site's
    table.  Each row's double-valued irreps are read off its label
    (:func:`_corep_matrix`): a ``p`` shell at a ``3m`` site, ``A1 + E``,
    gives ``{"-e1": 2, "-1e2e": 1}``.  Returns ``{name: copies}``, the names
    in lower case.  Returns ``None`` where any row of ``table`` has no label,
    whether or not ``rows`` holds it.
    """
    got = _corep_matrix(table)
    if got is None:
        return None
    names, N = got
    out = {}
    for p, n in rows.items():
        if int(n) == 0:
            continue
        for c, name in enumerate(names):
            if N[int(p), c]:
                out[name] = out.get(name, 0) + int(n) * int(N[int(p), c])
    return out


def _resolve_sum(table, terms, *, rotations=None):
    """The rows of ``table`` a declared sum names, ``{row: copies}``.

    ``terms`` is a list of ``(word, copies)``, as :func:`_sum_terms` returns
    for a sum, and ``rotations`` maps each operation of the site group to its
    Cartesian matrix.  A sum written in single-valued names is read row by
    row.  On a spinful site a sum written in the double-valued site irreps,
    ``2x-E1 + -1E2E`` or ``-E1`` alone, is read by finding every sum of rows
    that becomes exactly those double-valued irreps with spin: ``A1 + E`` and
    ``A2 + E`` of a ``3m`` site both become ``2x-E1 + -1E2E``.  Of these, the
    sum returned is the lowest under the lowest-angular-momentum rule
    (:func:`_content_shells`, read only where ``rotations`` is given), then
    the one of fewest irreps counted with multiplicity, then the one whose
    rows, each counted with its copies, differ from ``+1`` on the fewest
    operations in all (:func:`_row_distance`), then the first by row.
    Returns ``{}`` where no sum of rows fits.
    """
    spinful = any(table.declarable)
    rows = {}
    for word, k in terms:
        p = _orbital_row(table, word) if spinful else table.index_of(word)
        if p < 0:
            rows = None
            break
        rows[p] = rows.get(p, 0) + int(k)
    if rows is not None or not spinful:
        return rows or {}
    want = {}
    for word, k in terms:
        want[str(word).strip().lower()] = want.get(str(word).strip().lower(), 0) + int(k)
    lifted = []
    for p, lab in enumerate(table.labels):
        got = {}
        for word, k in (_sum_terms(lab) or ([(lab, 1)] if lab else [])):
            got[str(word).strip().lower()] = got.get(str(word).strip().lower(), 0) + int(k)
        lifted.append(got)
    best = [None]
    shells, lowest = {}, {}

    def _shells(chosen):
        """The two numbers :func:`_content_shells` gives the rows ``chosen``.

        ``chosen`` is ``{row: copies}``.  Returns ``()`` where no
        ``rotations`` were given.
        """
        if rotations is None:
            return ()
        chi = sum(n * np.asarray(table.characters[q], dtype=complex) for q, n in chosen.items())
        theta = {int(cls[0]): complex(chi[c]) for c, cls in enumerate(table.classes)}
        return _content_shells(table, theta, rotations, shells=shells, lowest=lowest)

    def _search(p, left, chosen):
        if p == len(lifted):
            if not any(left.values()):
                key = (_shells(chosen), sum(chosen.values()),
                       sum(n * _row_distance(table, q) for q, n in chosen.items()),
                       sorted(chosen.items()))
                if best[0] is None or key < best[0][0]:
                    best[0] = (key, dict(chosen))
            return
        got = lifted[p]
        most = min((left.get(w, 0) // k for w, k in got.items()), default=0) if got else 0
        for n in range(most, -1, -1):
            if n:
                chosen[p] = n
            _search(p + 1, {w: left[w] - n * got.get(w, 0) for w in left}, chosen)
            chosen.pop(p, None)

    _search(0, want, {})
    return {} if best[0] is None else best[0][1]


def _row_distance(table, row) -> int:
    """How many operations a row's character is not ``+1`` on.

    ``row`` is a row of ``table``, and every operation of its site group is
    counted.
    """
    return sum(len(table.classes[c]) for c in range(len(table.classes))
               if abs(complex(table.characters[row, c]) - 1.0) > 1e-9)


def _orbital_row(table, word) -> int:
    """The row whose single-valued name is ``word``, or ``-1``.

    ``word`` is compared, without regard to case, with the
    :attr:`SiteGroupTable.declarable` names of ``table``.  Where several rows
    carry it, the row returned is the one whose character differs from
    ``+1`` on the fewest operations (:func:`_row_distance`), the first on a
    tie.  ``"E"`` also names the one merged complex-conjugate pair of
    dimension two, where no row is named ``E``.
    """
    want = str(word).strip().lower()
    hits = [p for p, name in enumerate(table.declarable) if name and name.lower() == want]
    if hits:
        return min(hits, key=lambda p: (_row_distance(table, p), p))
    if want == "e":
        hits = [p for p in range(len(table.labels))
                if table.merged[p] and int(table.dims[p]) == 2]
        if len(hits) == 1:
            return hits[0]
    return -1


def _poly_of(vec, degree):
    return {m: float(v) for m, v in zip(_monomials(degree), vec) if abs(v) > 1e-14}


@lru_cache(maxsize=None)
def _shell_weights(degree):
    r"""The weights of the inner product every rotation keeps, on one degree.

    Each monomial ``x^a y^b z^c`` of degree ``degree`` has the weight
    ``a! b! c! / (a+b+c)!``, in the order :func:`_monomials` lists them.  In
    the coordinates scaled by the square roots of these weights, every
    orthogonal operation's matrix on the polynomials of that degree is
    orthogonal.  Returns the weights, read only.
    """
    w = np.array([float(factorial(a) * factorial(b) * factorial(c))
                  / float(factorial(int(degree)))
                  for a, b, c in _monomials(int(degree))])
    w.flags.writeable = False
    return w


def _shell_vector(name, degree, scale):
    """The orbital function ``name`` as a unit vector in scaled coordinates.

    ``name`` is a key of :data:`SHORTHANDS` whose polynomial has degree
    ``degree``, and ``scale`` holds the square roots of
    :func:`_shell_weights` for that degree.  Returns the polynomial's
    coefficients times ``scale``, normalized.
    """
    idx = _monomial_index(int(degree))
    v = np.zeros(len(idx))
    for m, c in SHORTHANDS[name].items():
        v[idx[m]] += float(c)
    v = v * np.asarray(scale, dtype=float)
    return v / float(np.linalg.norm(v))


def shell_operator(group, ind, degree):
    r"""One operation's matrix on the polynomials of one degree.

    The operation is the one of ``group`` whose ``.ind`` is ``ind``, acting
    as ``(g f)(r) = f(R^-1 r)`` (:func:`_apply`) on the polynomials of degree
    ``degree``.  The matrix is written in the coordinates scaled by the
    square roots of :func:`_shell_weights`, where it is orthogonal: column
    ``j`` is the image of the ``j``-th scaled monomial, in the order
    :func:`_monomials` lists them.  An antiunitary operation has the matrix
    of its rotation, since complex conjugation leaves a real function as it
    is.
    """
    degree = int(degree)
    scale = np.sqrt(_shell_weights(degree))
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    R_inv = np.linalg.inv(np.asarray(
        group._symmetries[pos_of[int(ind)]].rotation_cart, dtype=float))
    n = len(_monomials(degree))
    M = np.zeros((n, n))
    for j in range(n):
        v = np.zeros(n)
        v[j] = 1.0
        M[:, j] = _apply(R_inv, _poly_of(v / scale, degree), degree) * scale
    return M


#: The functions of the shells the shorthands name, in the order they are listed
#: (:data:`CANONICAL_SHORTHANDS`).
_SHELL_FUNCTIONS = {0: ("s",), 1: ("px", "py", "pz"),
                    2: ("dxy", "dyz", "dzx", "dx2-y2", "dz2")}


@lru_cache(maxsize=None)
def _harmonic_functions(degree):
    r"""The real harmonics of one degree, as orthonormal columns.

    The columns are written in :func:`shell_operator`'s coordinates.  For
    ``degree`` 0, 1 and 2 they are the shorthands' own functions, in the
    order :data:`_SHELL_FUNCTIONS` lists them.  For a higher degree they are
    the monomials in their own order, each less its part along ``r^2`` times
    the polynomials of degree ``degree - 2``, made orthonormal one after
    another, the first ``2 degree + 1`` that are independent.  Returns them
    read only, one column per harmonic.
    """
    degree = int(degree)
    scale = np.sqrt(_shell_weights(degree))
    if degree in _SHELL_FUNCTIONS:
        out = np.asarray([_shell_vector(name, degree, scale)
                          for name in _SHELL_FUNCTIONS[degree]]).T
        out.flags.writeable = False
        return out
    idx = _monomial_index(degree)
    lower = _monomials(degree - 2)
    r2 = np.zeros((len(idx), len(lower)))
    for j, (a, b, c) in enumerate(lower):
        for da, db, dc in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
            r2[idx[(a + da, b + db, c + dc)], j] += 1.0
    q, _r = np.linalg.qr(r2 * scale[:, None])
    cols = []
    for j in range(len(idx)):
        v = np.zeros(len(idx))
        v[j] = 1.0
        v = v - q @ (q.T @ v)
        for u in cols:
            v = v - u * float(u @ v)
        norm = float(np.linalg.norm(v))
        if norm > ZERO:
            cols.append(v / norm)
        if len(cols) == 2 * degree + 1:
            break
    out = np.asarray(cols).T
    out.flags.writeable = False
    return out


def function_of(word):
    """The orbital function a shorthand names, ``(degree, unit column)``.

    ``word`` is the shorthand, spelled as :func:`shorthand_key` reads it.
    The column is in :func:`shell_operator`'s coordinates.  Returns ``None``
    where ``word`` names no shorthand.
    """
    key = shorthand_key(word)
    if key is None:
        return None
    degree = max(sum(m) for m in SHORTHANDS[key])
    return int(degree), _shell_vector(key, degree, np.sqrt(_shell_weights(degree)))


def irrep_functions(group, table, row, rotations):
    r"""The lowest-angular-momentum functions of one site irrep, or ``None``.

    ``row`` is a row of ``table``, ``group`` the represented group, and
    ``rotations`` maps each operation of the site group to its Cartesian
    matrix.  The functions are taken in the shell :func:`_lowest_l` finds the
    irrep in: the shell's own functions (:func:`_harmonic_functions`) are
    projected onto the irrep and kept in their order where independent, as
    many as its dimension.  At a ``4mm`` site ``E`` gets ``(px, py)``, and
    the row even under ``m(x)`` and odd under ``C4(z)`` gets ``dx2-y2``.
    Returns ``(degree, columns)``, the columns in :func:`shell_operator`'s
    coordinates.  Returns ``None`` where no shell up to ``_L_MAX`` holds the
    irrep, or where the projections span no invariant space of its
    dimension, as where the shell holds the irrep more than once and its
    first functions straddle two copies.
    """
    chi = {int(i): table.character_on(row, i) for i in table.indices}
    degree = _lowest_l(table, chi, rotations)
    if degree > _L_MAX:
        return None
    ops = {int(i): shell_operator(group, i, degree) for i in table.indices}
    dim = int(round(float(np.real(table.dims[row]))))
    each = dim // 2 if table.merged[row] else dim
    proj = sum(np.conj(chi[i]) * ops[i] for i in table.indices) * each / len(table.indices)
    cols = []
    for f in _harmonic_functions(degree).T:
        v = proj @ f
        for u in cols:
            v = v - u * np.vdot(u, v)
        norm = float(np.linalg.norm(v))
        if norm > ZERO:
            cols.append(v / norm)
        if len(cols) == dim:
            break
    if len(cols) != dim:
        return None
    basis = np.asarray(cols).T
    for M in ops.values():
        image = M @ basis
        if float(np.abs(image - basis @ (basis.conj().T @ image)).max()) > ASSEMBLED:
            return None
    return int(degree), basis


def decompose(table, chi):
    """The multiplicity of each row of ``table`` in the character ``chi``.

    ``chi`` is given one value per class.  The multiplicities are read by
    orthogonality, each row normalized by its own inner product, so a merged
    complex-conjugate pair such as ``1E2E``, whose inner product with itself
    is 2, counts once in a span that carries the pair once.  Returns one
    complex multiplicity per row.
    """
    sizes = np.array([len(c) for c in table.classes], dtype=float)
    n = float(table.order)
    out = []
    for p in range(table.characters.shape[0]):
        row = np.asarray(table.characters[p])
        norm = float(np.real(np.sum(sizes * row * np.conj(row)) / n))
        out.append(complex(np.sum(sizes * np.asarray(chi) * np.conj(row)) / n
                           / (norm if norm > 1e-9 else 1.0)))
    return np.array(out)


#: How far the search for the lowest degree of real harmonics carrying a site
#: irrep goes.  Every irrep of a crystallographic point group occurs by degree
#: nine (``A1u`` of ``m-3m`` is the last, at nine).  A sum of irreps that no
#: degree up to twelve holds ranks, under the lowest-angular-momentum rule,
#: after every sum that one does.
_L_MAX = 12


def _shell_trace(rotation_cart, degree) -> float:
    r"""The character of the degree-``l`` real harmonics under one operation.

    ``rotation_cart`` is the operation's Cartesian matrix ``R``, and ``l`` is
    ``degree``.  The character is ``det^l sin((2l+1)θ/2) / sin(θ/2)``, with
    ``θ`` the angle of the proper rotation ``det·R`` and the ratio ``2l + 1``
    at ``θ = 0``.  On an improper operation ``det^l`` is ``(-1)^l``, the
    parity of the harmonics under the inversion.  Returns the character, a
    float.
    """
    R = np.asarray(rotation_cart, dtype=float)
    det = 1.0 if float(np.linalg.det(R)) > 0.0 else -1.0
    c = max(-1.0, min(1.0, (float(np.trace(det * R)) - 1.0) / 2.0))
    half = float(np.arccos(c)) / 2.0
    s = float(np.sin(half))
    n = 2 * int(degree) + 1
    value = float(n) if abs(s) < 1e-9 else float(np.sin(n * half)) / s
    return (det ** int(degree)) * value


def _lowest_l(table, chi, rotations, *, cache=None, tol=1e-6) -> int:
    """The lowest degree of real harmonics that holds a sum of site irreps.

    ``chi`` maps each operation's ``.ind`` to the sum's orbital character,
    and ``rotations`` maps each to its Cartesian matrix, over the site group
    of ``table``.  The sum is held at degree ``l`` where the real harmonics
    of degree ``l`` hold every irrep of it at least as often
    (:func:`_shell_content`).  A multiplicity within ``tol`` of an integer is
    read as that integer, and ``cache`` is passed on to
    :func:`_shell_content`.  Returns ``_L_MAX + 1`` where no degree up to
    ``_L_MAX`` holds the sum, where its multiplicities are not non-negative
    integers, where it is empty, and where ``table`` is ``None`` or ``chi``
    misses a class.
    """
    if table is None or any(int(cls[0]) not in chi for cls in table.classes):
        return _L_MAX + 1
    want = decompose(table, np.asarray([complex(chi[int(cls[0])]) for cls in table.classes]))
    n = []
    for m in want:
        m = complex(m)
        if abs(m.imag) > tol or abs(m.real - round(m.real)) > tol or round(m.real) < 0:
            return _L_MAX + 1
        n.append(int(round(m.real)))
    if not any(n):
        return _L_MAX + 1
    cache = {} if cache is None else cache
    for degree in range(_L_MAX + 1):
        have = _shell_content(table, degree, rotations, cache=cache)
        if all(have[i] >= n[i] for i in range(len(n)) if n[i]):
            return int(degree)
    return _L_MAX + 1


def _shell_content(table, degree, rotations, *, cache=None) -> list:
    """How often the real harmonics of one degree hold each site irrep.

    The harmonics are those of degree ``degree``, and ``rotations`` maps each
    operation of the site group of ``table`` to its Cartesian matrix.  The
    counts are read off the shell's character (:func:`_shell_trace`) by
    :func:`decompose`: the ``d`` shell at a ``-3m`` site holds ``A1g`` once
    and ``Eg`` twice.  ``cache`` keeps the counts across calls, by table and
    degree.  Returns one count per row of ``table``.
    """
    cache = {} if cache is None else cache
    key = (id(table), int(degree))
    if key not in cache:
        shell = np.asarray([_shell_trace(rotations[int(cls[0])], degree)
                            for cls in table.classes], dtype=complex)
        cache[key] = [int(round(float(np.real(h)))) for h in decompose(table, shell)]
    return cache[key]


def _content_shells(table, theta, rotations, *, shells=None, lowest=None):
    """The angular momentum a sum of site irreps sits at, as two numbers.

    ``theta`` maps each operation's ``.ind`` to the sum's orbital character,
    and ``rotations`` maps each to its Cartesian matrix, over the site group
    of ``table``.  The first number is the lowest degree of real harmonics
    that holds the whole sum (:func:`_lowest_l`).  The second is the sum over
    its irreps of multiplicity times dimension times the lowest degree that
    holds the irrep.  The lowest-angular-momentum rule compares the first,
    then the second: five ``d`` orbitals at a ``-3m`` site, ``A1g + 2xEg``,
    give ``(2, 8)``, and a ``p`` shell with a second ``p`` pair,
    ``A2u + 2xEu``, gives ``(3, 5)`` and ranks after them.  ``shells`` and
    ``lowest`` keep what earlier calls measured, the shells' counts and each
    row's lowest degree.  Returns ``(_L_MAX + 1, 10**6)`` where the
    multiplicities are not non-negative integers, and ``(_L_MAX + 1, 0)``
    where ``table`` is ``None``, or ``theta`` is empty or misses a class.
    """
    if (table is None or not theta
            or any(int(cls[0]) not in theta for cls in table.classes)):
        return (_L_MAX + 1, 0)
    lowest = {} if lowest is None else lowest
    mult = decompose(table, np.asarray([complex(theta[int(cls[0])]) for cls in table.classes]))
    per = 0
    for p, n in enumerate(mult):
        n = complex(n)
        if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or n.real < -0.5:
            return (_L_MAX + 1, 10 ** 6)       # no character of the site group
        k = int(round(n.real))
        if k == 0:
            continue
        key = (id(table), p)
        if key not in lowest:
            row = {int(i): table.character_on(p, i) for i in table.indices}
            lowest[key] = _lowest_l(table, row, rotations, cache=shells)
        per += k * int(round(float(table.dims[p]))) * int(lowest[key])
    return (int(_lowest_l(table, theta, rotations, cache=shells)), int(per))


def _mulliken_name(group, table, row) -> str:
    """Mulliken's name of one row of a site's table, or ``""``.

    ``row`` is a row of ``table``, and ``group`` the represented group.  At a
    ``-6m2``, ``-42m``, ``6`` or ``6/m`` site the name is read off the row's
    characters on operations no choice of axes moves.  At ``-6m2``, ``1`` or
    ``2`` follows the twofolds perpendicular to the threefold axis, and
    ``'`` or ``''`` the horizontal mirror.  At ``-42m``, ``A`` or ``B``
    follows the ``-4``, and ``1`` or ``2`` the twofolds perpendicular to its
    axis.  At ``6`` and ``6/m``, ``A`` or ``B`` and ``E1`` or ``E2`` follow
    the twofold along the sixfold axis, ``-1`` on each state of ``E1``, and
    ``g`` or ``u`` the inversion.  A complex row, or a merged pair of them,
    is named as its pair.  Returns ``""`` at every other site group.
    """
    symbol = point_group_symbol(group, table.indices)
    if symbol not in ("-6m2", "-42m", "6", "6/m"):
        return ""
    infos = [_class_info(group, cls)[1] for cls in table.classes]
    chi = np.asarray(table.characters[row], dtype=complex)
    dim = float(chi[next(c for c, info in enumerate(infos) if info["kind"] == "E")].real)

    def find(kind, order, axis=None, along=True):
        """The first class of one kind and order, placed by its axis.

        ``kind`` and ``order`` are as :func:`classify` gives them.  With
        ``axis``, the class's axis lies along it, or with ``along=False``
        perpendicular to it; in these four groups every axis is one or the
        other.  Returns the class's index, or ``None`` where none fits.
        """
        for c, info in enumerate(infos):
            if info["kind"] != kind or int(info["order"]) != order:
                continue
            if axis is None or along == (abs(float(np.asarray(info["axis"], dtype=float)
                                                   @ axis)) > 0.5):
                return c
        return None

    def even(c):
        return float(chi[c].real) / dim > 0.0

    if symbol == "-6m2":
        axis = np.asarray(infos[find("C", 3)]["axis"], dtype=float)
        prime = "'" if even(find("m", 2, axis)) else "''"
        if round(dim) == 2:
            return "E" + prime
        return "A" + ("1" if even(find("C", 2, axis, along=False)) else "2") + prime
    if symbol == "-42m":
        if round(dim) == 2:
            return "E"
        s4 = find("S", 4)
        axis = np.asarray(infos[s4]["axis"], dtype=float)
        return (("A" if even(s4) else "B")
                + ("1" if even(find("C", 2, axis, along=False)) else "2"))
    c2 = find("C", 2)
    inversion = find("i", 2)
    parity = "" if inversion is None else ("g" if even(inversion) else "u")
    if round(dim) == 1 and float(np.abs(chi.imag).max()) < CHARACTER:
        return ("A" if even(c2) else "B") + parity
    return ("E2" if even(c2) else "E1") + parity


def _mulliken_spelling(name) -> str:
    """The site irrep ``name`` as Mulliken's convention spells it.

    A merged pair of complex rows, ``1E2E``, or one member of it, ``1E``, is
    spelled by its pair, ``E``.  Any other name is returned as it is.
    """
    pair = re.fullmatch(r"1(E.*)2\1", name)
    if pair is not None:
        return pair.group(1)
    member = re.fullmatch(r"[12](E.*)", name)
    return name if member is None else member.group(1)


def mulliken_names(group, table, spec) -> tuple:
    """The names ``spec`` writes that Mulliken's convention gives another row.

    ``spec`` is a declaration's value, ``table`` its site's table and
    ``group`` the represented group.  At some sites of four kinds, ``-6m2``,
    ``-42m``, ``6`` and ``6/m``, the tables' single-valued names exchange
    rows against Mulliken's convention, and a declaration written in such a
    name states the tables' row.  Each name ``spec`` writes, alone or in a
    sum, is compared with Mulliken's name of its row (:func:`_mulliken_name`).
    Returns ``((name, Mulliken's name), ...)``, each pair once, its first
    name as :func:`_declared_name` writes it: ``(("A1''", "A2'"),)`` for
    ``"A1''"`` at ``2c`` of ``P6/mmm``.  Returns ``()`` where no name
    differs, for a value written in the double-valued site irreps, and for
    anything not a string.
    """
    if not isinstance(spec, str) or named_in_coreps(table, spec):
        return ()
    out = []
    for word, _k in (_sum_terms(spec) or [(spec.strip(), 1)]):
        row = table.index_of(word)
        if row < 0:
            continue
        name = _declared_name(table, row)
        mulliken = _mulliken_name(group, table, row)
        if mulliken and _mulliken_spelling(name) != mulliken and (name, mulliken) not in out:
            out.append((name, mulliken))
    return tuple(out)


def mulliken_clause(names) -> str:
    """The clause a declaration's record carries for pairs of names.

    ``names`` holds ``(name, Mulliken's name)`` pairs, as
    :func:`mulliken_names` returns them.  Each pair reads "the tables' A1''
    at this site is the row Mulliken's convention calls A2'", with "a member
    of the pair" for "the row" where the name is one member of a merged
    pair.  Returns the clauses joined by ``"; "``.
    """
    parts = []
    for name, mulliken in names:
        member = (re.fullmatch(r"[12]E.*", name) is not None
                  and re.fullmatch(r"1(E.*)2\1", name) is None)
        what = "a member of the pair" if member else "the row"
        parts.append(f"the tables' {name} at this site is {what} Mulliken's convention "
                     f"calls {mulliken}")
    return "; ".join(parts)


def resolve_spec(group, table, spec, where):
    """The characters, name, dimension and note a ``characters=`` value fixes.

    ``spec`` is the value, ``table`` the table of its site, ``group`` the
    represented group, and ``where`` the words that open every message,
    ``"orbital 0"``.  The value is one of:

    - a site irrep by its name: its label on a spinless site, its
      single-valued name on a spinful one (:meth:`SiteGroupTable.index_of`);
    - a shorthand, which names the one site irrep the span of its function
      carries (:func:`shorthand_character`);
    - a shell (:data:`SHELLS`), the site irreps the real harmonics of one
      angular momentum carry here, ``A1 + E`` for ``"p"`` at a ``3m`` site;
    - a sum of site irreps, ``"A1 + E"``;
    - on a spinful site, a double-valued site irrep or a sum of them,
      ``"-E1"`` or ``"2x-E1 + -1E2E"``, which states their multiplicities
      alone (:func:`named_in_coreps`) and is read as the sum of rows
      :func:`_resolve_sum` chooses;
    - generator characters or generator matrices, a mapping from operations
      (:func:`_resolve_generator_dict`);
    - a :class:`DeclaredContent` read in a larger group, whose characters
      are kept on the operations of this site group alone.

    Returns ``(constraints, label, dim, note)``: ``constraints`` the
    character on each operation of the site group the value fixes, keyed by
    ``.ind``; ``label`` the name the value resolved to, ``""`` where no name
    fits; ``dim`` the dimension of its orbital factor, the number of orbitals
    it describes; ``note`` how it was resolved, ``""`` for a site irrep named
    directly.  A sum or a shell is named by :func:`content_label`, and a sum
    of double-valued irreps by :func:`corep_label`.  A
    :class:`DeclaredContent` keeps the label, dimension and note it was
    declared with.

    Raises
    ------
    DeclarationError
        ``unknown_label`` where the value names no site irrep, sum, shell,
        shorthand or operation of this site group, where a shorthand's span
        holds more than one site irrep or one more than once, and as
        :func:`_resolve_generator_dict` raises for a mapping.

    See Also
    --------
    .._declaration._label_character : the name a restricted declaration
        takes at this site.
    """
    if isinstance(spec, DeclaredContent):
        keep = {int(i) for i in table.indices}
        return ({int(k): complex(v) for k, v in spec.constraints.items() if int(k) in keep},
                str(spec.label), int(spec.dim), str(spec.note))
    if isinstance(spec, dict):
        return _resolve_generator_dict(table, spec, where, group)

    # A value written in the double-valued site irreps, one word or a sum, is
    # read as a sum below; any other word is first looked up as a row
    corep = named_in_coreps(table, spec)
    row = -1 if corep else table.index_of(spec)
    if row >= 0:
        chi = table.characters[row]
        return ({int(ind): complex(chi[table.class_of[int(ind)]]) for ind in table.indices},
                _declared_name(table, row), int(table.dims[row]), "")

    key = str(spec).strip().lower().replace(" ", "").replace("_", "")
    if key in SHORTHANDS:
        chi, dim = shorthand_character(group, table, key)
        mult = decompose(table, chi)
        rows = [p for p in range(len(mult)) if abs(mult[p]) > 1e-6]
        if len(rows) != 1 or abs(mult[rows[0]] - 1.0) > 1e-6:
            words = declaration_words(table)
            parts = " + ".join(
                f"{int(round(mult[p].real))}x"
                + (table.labels[p]
                   or (table.declarable[p] if p < len(table.declarable) else "")
                   or f"row {p}")
                for p in rows) or "nothing"
            raise DeclarationError(
                "unknown_label",
                f"{where}: the shorthand {spec!r} spans {parts} at this site. "
                f"Declare a site irrep by its name: {', '.join(words)}."
                f"\n  frame: {generator_frame(table)}",
                {"declared_as": spec, "decomposition": parts, "irreps": words})
        p = rows[0]
        return ({int(ind): complex(table.characters[p, table.class_of[int(ind)]])
                 for ind in table.indices},
                _declared_name(table, p), int(table.dims[p]),
                f"{spec!r} resolved to {_declared_name(table, p)} on this site group")

    # A shell: the site irreps the real harmonics of one degree carry here,
    # which no orientation of the functions inside the site moves
    shell = shell_key(spec)
    if shell is not None:
        chi = shell_character(group, table, SHELLS[shell])
        label = content_label(table, chi)
        return ({int(ind): complex(chi[table.class_of[int(ind)]]) for ind in table.indices},
                label, 2 * SHELLS[shell] + 1,
                f"{spec!r} resolved to {label} on this site group")

    # A sum of irreps, in single-valued or double-valued names, and a single
    # double-valued irrep as a sum of one
    terms = _sum_terms(spec) or ([(str(spec).strip(), 1)] if corep else None)
    rows = {}
    if terms:
        pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
        rows = _resolve_sum(table, terms, rotations={
            int(ind): np.asarray(group._symmetries[pos_of[int(ind)]].rotation_cart, dtype=float)
            for ind in table.indices})
    if rows:
        chi = sum(n * np.asarray(table.characters[p], dtype=complex) for p, n in rows.items())
        # a sum written in the double-valued irreps is named in them: the
        # rows standing for it are one of several sums that become it
        label = corep_label(table, chi) if corep else content_label(table, chi)
        return ({int(ind): complex(chi[table.class_of[int(ind)]]) for ind in table.indices},
                label, int(sum(n * int(round(float(np.real(table.dims[p]))))
                               for p, n in rows.items())),
                (f"{spec!r} states the corep content, and the orbital factor under it is "
                 "assigned as for an undeclared orbital" if corep else
                 f"{spec!r} resolved to {label} on this site group"))

    said = (f"{spec!r} is not what any sum of the site's single-valued irreps becomes "
            "with spin"
            if corep else
            f"{spec!r} is not an irrep of this site group, a sum of its irreps, "
            f"a shell or a shorthand")
    raise DeclarationError(
        "unknown_label",
        f"{where}: {said}. Declare one of the words listed below.\n"
        f"  site group: order {table.order}, classes "
        f"{', '.join(table.class_names)}\n"
        f"  valid labels: {', '.join(declaration_words(table)) or '(none: declare '
        f'generator characters instead)'}\n"
        f"  shells:       {', '.join(SHELLS)}\n"
        f"  shorthands:   {', '.join(CANONICAL_SHORTHANDS)}\n"
        f"  frame: {generator_frame(table)}\n"
        f"{table_summary(table)}",
        {"declared_as": spec, "irreps": declaration_words(table), "site_group": table.indices,
         "frame": dict(table.names)})


def _resolve_generator_dict(table, spec, where, group=None):
    """The characters, name, dimension and note a mapping of generators fixes.

    ``spec`` maps operations of the site group of ``table``, each by its
    name or its ``.ind``, to a character or to a matrix.  ``where`` opens
    every message, and ``group`` is the represented group, without which
    matrices are not read.  Generator matrices give their trace to every
    operation they generate (:func:`generator_matrices`), their size as the
    dimension, and a label (:func:`content_label`) where they reach the whole
    site group.  Generator characters give each value to every operation of
    the class its key names, and the label and dimension of the one row that
    matches every value within ``1e-6``; where not exactly one row matches,
    the label is ``""`` and the dimension 1.  Returns
    ``(constraints, label, dim, note)`` as :func:`resolve_spec` does.

    Raises
    ------
    DeclarationError
        ``unknown_label`` where a key names no operation of the site group
        (:func:`_match_operation`), where a character's key matches
        operations of more than one class, and as
        :func:`generator_matrices` raises.
    """
    matrices = generator_matrices(group, table, spec, where) if group is not None else {}
    if matrices:
        constraints = {int(ind): complex(np.trace(M)) for ind, M in matrices.items()}
        dim = int(next(iter(matrices.values())).shape[0])
        whole = all(int(i) in constraints for i in table.indices)
        label = (content_label(table, [constraints[int(cls[0])] for cls in table.classes])
                 if whole else "")
        named = ", ".join(str(k) if isinstance(k, str) else table.names[int(k)] for k in spec)
        note = (f"generator matrices on {named}"
                + (f" -> {label}" if label else
                   f", which generate {len(matrices)} of the site group's "
                   f"{len(table.indices)} operations"))
        return constraints, label, dim, note
    constraints, seen = {}, []
    for key, value in spec.items():
        inds = _match_operation(table, key, where)
        cls = {table.class_of[i] for i in inds}
        if len(cls) != 1:
            raise DeclarationError(
                "unknown_label",
                f"{where}: the key {key!r} matches operations in more than one conjugacy "
                f"class ({sorted(cls)}). Name one operation listed in the frame below."
                f"\n  frame: {generator_frame(table)}",
                {"key": key, "matched": inds})
        c = cls.pop()
        for ind in table.indices:
            if table.class_of[int(ind)] == c:
                constraints[int(ind)] = complex(value)
        seen.append((key, c, complex(value)))
    rows = [p for p in range(len(table.labels))
            if all(abs(table.characters[p, table.class_of[i]] - v) < 1e-6
                   for i, v in constraints.items())]
    label = _declared_name(table, rows[0]) if len(rows) == 1 else ""
    dim = int(table.dims[rows[0]]) if len(rows) == 1 else 1
    note = ("generator characters " +
            ", ".join(f"{k}={_fmt(v)}" for k, _, v in seen) +
            (f" -> {label}" if label else " (no single irrep matches them exactly)"))
    return constraints, label, dim, note


def generator_matrices(group, table, spec, where) -> dict:
    r"""The matrix of each operation the declared generator matrices generate.

    ``spec`` is one ``characters=`` value, ``table`` the site's table,
    ``group`` the represented group, and ``where`` the words that open every
    message.  Each key of ``spec`` names one operation of the site group, by
    its ``.ind`` or its name, and each matrix is that operation's on the
    orbital factor of the orbitals declared, in the order written: column
    ``j`` is the image of orbital ``j``, as
    :meth:`..symmetry.SpaceGroupRep.sewing_operator` writes a matrix.  The
    products of the declared matrices give the matrix of every operation the
    declared ones generate, the identity included.  A site group holds no
    translation, so each operation is found by its Cartesian rotation.
    Returns ``{ind: matrix}``, or ``{}`` where ``spec`` is not a nonempty
    mapping whose every value is a matrix.

    Raises
    ------
    DeclarationError
        ``unknown_label`` where a key names no operation of the site group or
        several, where a matrix is not unitary, and where two products
        reaching one operation give it two matrices.
    """
    if not isinstance(spec, dict) or not spec or not all(
            np.ndim(v) == 2 for v in spec.values()):
        return {}
    pos_of = {int(s.ind): i for i, s in enumerate(group._symmetries)}
    rotation = {int(i): np.asarray(group._symmetries[pos_of[int(i)]].rotation_cart, dtype=float)
                for i in table.indices}
    gens = []
    for key, value in spec.items():
        inds = _match_operation(table, key, where)
        if len(inds) != 1:
            raise DeclarationError(
                "unknown_label",
                f"{where}: the key {key!r} names {len(inds)} operations of this site group "
                f"({', '.join(table.names[int(i)] for i in inds)}). Name one operation "
                f"listed in the frame below.\n  frame: {generator_frame(table)}",
                {"key": key, "matched": inds})
        M = np.asarray(value, dtype=complex)
        if float(np.abs(M @ M.conj().T - np.eye(len(M))).max()) > CHARACTER:
            raise DeclarationError(
                "unknown_label",
                f"{where}: the matrix given {table.names[int(inds[0])]}, {_fmt_matrix(M)}, "
                "is not unitary. Declare a unitary matrix for it."
                f"\n  frame: {generator_frame(table)}",
                {"key": key, "matrix": M})
        gens.append((int(inds[0]), M))
    ident = next(int(i) for i in table.indices if np.allclose(rotation[int(i)], np.eye(3)))
    got = {ident: np.eye(len(gens[0][1]), dtype=complex)}
    frontier = [ident]
    while frontier:
        reached = []
        for a in frontier:
            for g, Mg in gens:
                R = rotation[g] @ rotation[a]
                c = next((int(i) for i in table.indices
                          if np.allclose(rotation[int(i)], R, atol=ASSEMBLED)), None)
                if c is None:                    # pragma: no cover (a site group closes)
                    continue
                M = Mg @ got[a]
                if c not in got:
                    got[c] = M
                    reached.append(c)
                elif float(np.abs(got[c] - M).max()) > CHARACTER:
                    raise DeclarationError(
                        "unknown_label",
                        f"{where}: the matrices declared are not a representation of "
                        f"this site group. Two of their products reach {table.names[c]} "
                        f"and give it {_fmt_matrix(got[c])} and {_fmt_matrix(M)}. "
                        "Declare matrices that multiply as their operations do.\n"
                        f"  frame: {generator_frame(table)}",
                        {"declared_as": spec, "operation": c})
        frontier = reached
    return got


def _fmt_matrix(M) -> str:
    """The matrix ``M`` as text, row by row.

    Each entry is written as :func:`_fmt` writes a character.
    """
    return "[" + ", ".join("[" + ", ".join(_fmt(x) for x in row) + "]"
                           for row in np.asarray(M)) + "]"


def _match_operation(table, key, where):
    """The operations of a site group that a declaration's key names.

    ``key`` is an operation's ``.ind`` or its name, ``table`` the site's
    table, and ``where`` the words that open the message.  A name is
    compared with each operation's name in the spelling :func:`_key` reads.
    Where none matches exactly and ``key`` holds no axis in parentheses,
    every operation whose name agrees before its parenthesis matches:
    ``"m"`` is every mirror.  Returns the ``.ind`` of each operation matched.

    Raises
    ------
    DeclarationError
        ``unknown_label`` where an ``.ind`` is no operation of the site
        group, and where a name matches none.
    """
    if isinstance(key, (int, np.integer)):
        if int(key) not in table.class_of:
            raise DeclarationError(
                "unknown_label",
                f"{where}: #{int(key)} names no operation of this site group, which holds "
                f"operations ({', '.join(str(i) for i in table.indices)}). Name one operation "
                "listed in the frame below.\n"
                f"  frame: {generator_frame(table)}",
                {"key": int(key), "site_group": table.indices})
        return [int(key)]
    want = _key(key)
    exact = [int(i) for i in table.indices if _key(table.names[int(i)]) == want]
    if exact:
        return exact
    loose = [int(i) for i in table.indices
             if _key(table.names[int(i)]).split("(")[0] == want.split("(")[0]
             and "(" not in want]
    if loose:
        return loose
    raise DeclarationError(
        "unknown_label",
        f"{where}: {key!r} names no operation of this site group. Name one operation "
        "listed in the frame below.\n"
        f"  frame: {generator_frame(table)}",
        {"key": key, "frame": dict(table.names)})
