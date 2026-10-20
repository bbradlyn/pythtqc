r"""Site-symmetry character mathematics, computed from the *measured* group.

A ``characters=`` declaration of :mod:`.symmetry` is resolved against these
tables.  The site groups come from
:attr:`..groups.OrbitalAssignment.site_group` (which
:func:`..groups._assign_orbitals` measured from the operations' action on the
model's own orbital positions), the multiplication comes from
``GroupData.get_product_table``, and the irreducible characters are computed
from that multiplication by the **class-sum** construction below: arithmetic on
a measured table.

A declaration is resolved against the character table of exactly the group that
was measured to fix that site, so a table exists for every site of every model.
The **names** of its rows are the tables' own: a row is named by the band
representation it induces, matched against the EBR entries at the site's Wyckoff
position (:mod:`..ebr`), which is physics and needs no orientation convention —
and it is the name a position is tabulated under, so a declaration may be
written in it.  For a model whose states carry spin the match is against the
group's **double-valued** entries, and what a row is named is the double-valued
site corep the states carry there.  Where the entries at the position do not
name the row — the tables list none there, or none whose band representation is
the row's, the row's being composite — it keeps the site point group's own
name: for a single-valued row the names of section 3 below, and for a
double-valued one the name the tables give that irrep of the point group at the
origin of a symmorphic space group, the one position where every point group is
a site group whole (:mod:`..ebr`).  A row nothing names has **no name** — an
empty label, :attr:`..symmetry.AtomicLimitRow.named_by` reading ``"unnamed"``,
and the generator characters as the whole statement.  Every label is echoed
together with those characters (:func:`generator_frame`), which no naming
convention touches.

The generator frame (read this before writing a declaration)
------------------------------------------------------------
Operations are named from their **Cartesian** rotation matrix
(``irrep``'s ``SymmetryOperation.rotation_cart``), measured, never typed:

* ``E`` identity, ``i`` inversion;
* ``C{n}({axis})`` a proper rotation by ``2π/n`` about ``{axis}``, and
  ``C{n}^-1({axis})`` its inverse -- a half turn ``C2({axis})`` is its own
  inverse and has none;
* ``S{n}({axis})`` an improper operation (``n > 2``), **named by its proper
  rotation part**: a counterclockwise turn by ``2π/n`` about ``{axis}``
  followed by the inversion, which is the Seitz ``-n+`` — ``S3(z)`` is
  ``-3+_001``, ``S4(z)`` is ``-4+_001`` and ``S6(z)`` is ``-6+_001``, and
  ``S{n}^-1({axis})`` is ``-n-``;
* ``m({axis})`` a mirror — and **the axis of a mirror is its normal**, the
  Hermann-Mauguin convention: ``m(x)`` is the operation
  ``(x, y, z) -> (-x, y, z)``, i.e. the plane ``x = 0``.  Mulliken notation
  names the same plane after the axes it *contains*, so

      ``m(x)`` here **is** Mulliken's ``sigma_v(yz)``,
      ``m(y)`` here **is** Mulliken's ``sigma_v(xz)``.

  Both spellings are accepted as keys (``"m(x)"``, ``"sigma_v(x)"``,
  ``"sigma(x)"``, ``"mx"``), and all of them mean *normal along x*.
* ``{axis}`` is ``x``, ``y`` or ``z`` when the direction is along a Cartesian
  axis, otherwise the direction in **lattice** coordinates, ``[uvw]``.

The **point group's own names** are read off the operations' Cartesian
geometry, Mulliken-style.  They are what the represented group's own table is
written in — its rows sit at no Wyckoff position and no EBR entry names them —
and what a single-valued site row keeps where neither the tables' entries at
its position nor the tables' names for the point group settle it, and the
symbol is a name rather than a placeholder
(:attr:`..symmetry.AtomicLimitRow.named_by` reading ``"Mulliken"``):

* dimension 1 gives ``A`` when ``chi`` of the principal rotation (the proper
  rotation of highest order; ties broken towards ``z``) is ``+1`` and ``B``
  when it is ``-1``; dimension 2 gives ``E``, dimension 3 ``T``, and anything
  larger ``L{dim}`` with an index;
* a complex-conjugate pair of one-dimensional irreps is **merged** into one
  physically irreducible character where time reversal is among the site's
  operations, since time reversal glues them, and is named ``1E2E``;
* the subscript ``1``/``2`` is the character of the **secondary reference
  element**: the ``C2'`` axis perpendicular to the principal axis that is
  closest to ``+x`` if the group has one, otherwise the vertical mirror whose
  *plane contains* ``+x`` (normal perpendicular to ``x``).  This is the
  convention that makes ``C2v`` come out as Mulliken's
  ``B1 = sigma_v(xz)-even``;
* ``g``/``u`` follows ``chi(i)`` when the group contains the inversion, and
  ``'``/``''`` follows ``chi`` of the horizontal mirror when it contains one
  and no inversion — and in the group ``m`` itself, which has no rotation axis
  for a mirror to be horizontal *to*, of its one reflection: ``m`` reads ``A'``
  and ``A''``, ``-1`` reads ``Ag`` and ``Au``, and the trivial group ``A``.

Where two characters would still collide the labels are suffixed ``_a``,
``_b``, … in a deterministic order.

None of this names a **site** row the tables reach.  A declaration at a site
of a model whose states carry spin states the *orbital* factor — ``s``,
``px``, ``A2`` — while the label that row is returned under is the
double-valued site corep the states carry, and the two are read from the two
table files of the one group:
the orbital factor is single-valued whether or not the states carry spin, so
it has a name at the position in the single-valued tables
(:attr:`SiteGroupTable.declarable`) exactly as the same orbital on a spinless
model does.  Both words are accepted; neither is ever returned in place of the
other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from math import factorial
from typing import Any

import numpy as np

from ._tolerance import ASSEMBLED, CHARACTER, ZERO

# --------------------------------------------------------------------------
# the declaration error, turned into a Refusal by `symmetry._build_rep`
# --------------------------------------------------------------------------


class DeclarationError(ValueError):
    """A ``characters=`` declaration its site cannot read.

    A word that names no site irrep there, a multiplet an ordered list cannot
    express, or a site whose character table could not be built to declare
    against.  It never leaves
    the package: the ``gate``, ``message`` and ``context`` it carries are what
    :func:`..symmetry._build_rep` and :func:`..groups.realized_group` return as
    a :class:`..refusal.Refusal`.  A declaration outside the grammar raises
    ``ValueError`` where the call starts
    (:func:`..symmetry._declaration_grammar`).
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
    """Name a direction: ``x``/``y``/``z`` when Cartesian, else ``[uvw]``."""
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
    r"""``{kind, order, angle, axis, det, name}`` of one operation, from its matrix.

    The angle is the turn ``φ`` of the proper part ``det·R``, from the trace
    (``tr R = det·(1 + 2 cos φ)``), and the axis is that part's own — so the
    classification is a measurement of ``rotation_cart``, with nothing assumed
    about the setting.  An improper operation is that turn followed by the
    inversion, the Seitz ``-n+``, and its ``order`` is the turn's ``n``:
    ``-3+`` is ``S3``, ``-4+`` is ``S4`` and ``-6+`` is ``S6``.
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
    """Unit eigenvector of ``M`` for eigenvalue ``value``, sign-canonicalized."""
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
    """Canonical name of one operation in the frame documented at module top."""
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
    """Normalize a user-supplied operation key: case, spaces, mirror spellings."""
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
    """Everything a declaration is resolved against, for one site group.

    Attributes
    ----------
    indices : tuple of int
        ``.ind`` of the **unitary, represented** operations that fix the site —
        the group the labels below belong to.
    order : int
    classes : tuple of tuple of int
        Conjugacy classes, as ``.ind`` tuples.
    class_names : tuple of str
        ``"2C4(z)"``-style names (multiplicity + representative).
    class_reps : tuple of int
        ``.ind`` of the representative each class name was built from.
    class_of : dict
        ``{ind: class number}``.
    characters : ndarray, shape (n_irrep, n_class)
        The physically irreducible characters — complex-conjugate pairs merged
        where the site carries time reversal, which glues them.
    labels : tuple of str
        The rows' names, unique within the table where present: the tables'
        own names for the point group's irreps wherever a site table is read
        (``..symmetry._site_group_table``), the Mulliken-style names of section
        3 where it is not, and empty for a row nothing names.
    dims : ndarray
        ``characters[:, class of identity]``, real.
    names : dict
        ``{ind: operation name}`` — the generator frame, echoed everywhere.
    orthogonality : float
        ``max |X diag(|C|) X† − |G| 1|`` — the table's own self-check.
    merged : tuple of bool
        Which rows came from merging a complex-conjugate pair.
    antiunitary : tuple of int
        ``.ind`` of the antiunitary operations that also fix the site (they do
        not enter the characters; time reversal among them is what merges a
        complex-conjugate pair).
    named_by : tuple of str
        Where each row's label comes from: ``"tables"`` when the band
        representation it induces is the EBR entries at the site's own Wyckoff
        position, ``"point group"`` when those entries do not name the row and
        the name is the point group's own (the tables' names at the origin of a
        symmorphic group, single- or double-valued), ``"Mulliken"`` when those
        names leave a single-valued row open and its Mulliken symbol of section
        3 names it, and ``"unnamed"`` when nothing reaches the row and the label
        is empty.
    declarable : tuple of str
        A second spelling of each row a declaration may also be written in,
        empty where there is only one.  It holds the **orbital-factor** name of
        a site whose states carry spin: :attr:`labels` is then the
        double-valued site corep the states carry and a declaration states what
        the orbital is (``A2``), so both have to resolve.  It is the name the
        group's **single-valued** entries give that row at the position —
        the same word the same orbital would be returned under on a spinless
        model — and for a row those entries do not reach, the tables' name for
        that irrep of the point group, which a row nothing names lacks.
        Never a returned name — :attr:`..symmetry.AtomicLimitRow.irrep` is always
        from :attr:`labels`, and what the caller wrote is kept verbatim in
        :attr:`..symmetry.AtomicLimitRow.declared_as`.
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
        """Row of ``label`` (case-insensitive), or ``-1``.

        ``"E"`` also answers for a time-reversal-glued complex pair
        (``1E2E``) when the table has no row of its own by that name — the
        friendly spelling of the same two-dimensional object.

        :attr:`declarable` answers too, where a row has a second spelling a
        declaration may be written in.

        Several rows can carry **one** label, and legitimately: where the
        states carry spin the label is the double-valued site corep, and two
        orbital characters differing by one the lift's trace vanishes on are
        the same corep.  The row answered is then the one closest to the
        trivial character — the fewest operations it is not ``+1`` on.  That is
        this lookup's own order and no reading of a declaration: a declaration
        written in a corep's name states the corep content alone, and
        :func:`resolve_spec` reads it so (:func:`named_in_coreps`) before it asks
        here.  An empty label names nothing and is never answered.
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
        """``chi_row(g)`` for the operation ``.ind = ind``."""
        return complex(self.characters[row, self.class_of[int(ind)]])


def _local_table(prod, positions):
    """Multiplication of ``positions`` in local indices, or ``None`` if not closed."""
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
    """Conjugacy classes of a group given by its multiplication table."""
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
    r"""Irreducible characters from the multiplication table (class-sum method).

    The class sums span the centre of the group algebra; their structure
    constants :math:`c_{ijk}` give commuting matrices :math:`(M_i)_{kj} =
    c_{ijk}` whose simultaneous eigenvectors are the central characters
    :math:`\omega_i = |C_i| \chi(C_i)/\chi(1)`.  A random real combination is
    diagonalized once (a genuine group has a simple spectrum for generic
    coefficients), the eigenvalues of each :math:`M_i` are read back off the
    eigenvector, and the dimension follows from
    :math:`\sum_i |\omega_i|^2/|C_i| = |G|/\chi(1)^2`.  Nothing is tabulated,
    and the row-orthogonality defect is returned as the self-check.
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
    """Glue complex-conjugate character pairs into physically irreducible rows."""
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
# 3.  Mulliken naming -- for the represented group's own one-dimensional
#     characters only.  A SITE irrep is named by the band representation it
#     induces (`..ebr`), which is what the tables key a Wyckoff position by;
#     the geometric reconstruction below reads a principal axis and a secondary
#     reference off the operations' Cartesian matrices, and that is an
#     orientation convention the tables carry per position and a measurement
#     cannot supply.
# --------------------------------------------------------------------------

def _class_info(group, cls):
    """Classification of a class, plus its preferred representative."""
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    infos = [(ind, classify(group.symmetries[pos_of[int(ind)]])) for ind in cls]
    # representative: prefer a positive rotation sense, then an axis near +x/+y/+z
    def key(item):
        _, info = item
        ax = info["axis"]
        along = max(abs(float(ax[0])), abs(float(ax[1])), abs(float(ax[2])))
        return (-info.get("sense", 1), -round(along, 6), tuple(np.round(-ax, 6)))
    ind, info = sorted(infos, key=key)[0]
    return ind, info, infos


def _principal_axis(class_infos):
    """Axis of the highest-order proper rotation; ties broken towards ``+z``."""
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
    """Class index of the ``C2'``/``sigma_v`` the ``1``/``2`` subscript follows.

    The comparison runs over **every member** of a class, not its representative:
    in ``C4v`` the class ``{m(x), m(y)}`` contains the mirror whose plane holds
    ``x`` while the diagonal class contains none, and it is that class
    Mulliken's subscript follows.
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
    """Whether a name of this section holds a place rather than naming an irrep.

    :func:`_mulliken` writes a placeholder where its rules cannot tell rows
    apart: a ``_a``, ``_b``, ... subscript, and an ``L`` base for a row that is
    no ``A``, ``B``, ``E`` or ``T`` (a complex one-dimensional row without time
    reversal, ``L1``).  Every other name is one a table prints.
    """
    label = str(label)
    head = label[1:2]
    return any(f"_{letter}" in label for letter in "abcdefgh") or (
        label.startswith("L") and head.isdigit())


def _mulliken(chars, merged, class_infos, cls_of_identity):
    """Names for every row of the character table (see the module docstring)."""
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
        # A GROUP WITH NO ROTATION AXIS AT ALL, which for a site group means `m`:
        # the identity and one reflection.  There is no principal axis for that
        # reflection to be *horizontal* to, and yet it is the only thing the two
        # rows differ on, so it is the parity reference and the group reads `A'`
        # and `A''` as every table prints it.  (`-1` is named by `inv_cls` above
        # and the trivial group has one row, `A`.)  Without this the subscript
        # rule below has nothing to separate two rows with and falls back to a
        # generic `_a`/`_b`, which names no irrep anyone could look up.
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

    # The subscript exists only to separate rows that would otherwise collide —
    # which is why `T` of a tetrahedral site comes out plain `A`, `1E2E`, `T`
    # while `C4v` needs the full `A1 A2 B1 B2 E`.
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
    """Character table of the measured site group ``indices`` (``.ind`` values).

    The rows come back with the Mulliken names of section 3, which only the
    **represented group's own** table keeps, for its characters.  A **site**
    table is renamed by :func:`..symmetry._site_group_table`, each row under the
    tables' own name for that irrep of the point group, and then by
    :func:`..symmetry._site_tables`, which reads every name a row has off the
    tables: the one it is returned under, and — where the states carry spin, so
    that the returned one is the double-valued corep — the single-valued name
    of its orbital factor, which is what a declaration is written in
    (:attr:`SiteGroupTable.declarable`).

    Raises
    ------
    DeclarationError
        ``gate="site_group"`` when the operations are not closed under the
        group's own product table (they always are for a measured site group;
        the check is a self-check, not a policy), or when the computed table
        fails its orthogonality self-check.
    """
    prod, _, _ = group.get_product_table(get_diff=True)
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    positions = [pos_of[int(i)] for i in indices]
    T = _local_table(prod, positions)
    if T is None:
        raise DeclarationError(
            "site_group",
            f"the site-symmetry operations {tuple(int(i) for i in indices)} are not closed "
            "under the group's own product table, so they have no character table.",
            {"indices": tuple(int(i) for i in indices)})
    cls_local, cls_of_local, e_local, chars, defect = _character_table(T, seed=seed)
    if defect > 1e-6 * max(1, len(positions)):
        raise DeclarationError(
            "site_group",
            f"the computed character table of the site group failed its own orthogonality "
            f"self-check (defect {defect:.3g}); labels built on it would be meaningless.",
            {"indices": tuple(int(i) for i in indices), "orthogonality": defect})
    chars, merged = _merge_conjugate_pairs(chars) if antiunitary else (chars, (False,) * len(chars))

    lattice = np.asarray(group.lattice, dtype=float)
    ind_of = [int(group.symmetries[p].ind) for p in positions]
    classes = tuple(tuple(ind_of[g] for g in c) for c in cls_local)
    class_of = {ind_of[g]: cls_of_local[g] for g in range(len(positions))}
    names = {ind_of[g]: operation_name(group.symmetries[positions[g]], lattice)
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


#: Every crystallographic point group, keyed by the multiset of operation kinds
#: it holds — ``E``, ``i``, a mirror ``m``, a proper rotation ``Cn``, a
#: rotoinversion ``Sn`` (a turn by ``2π/n``, then the inversion: ``-n``).
#: That multiset separates all thirty-two, so naming a
#: measured site group is arithmetic on :func:`classify`'s verdicts and not a
#: table of groups: nothing here says which operations a group *has*, only what
#: a group holding these is called.  The spelling is Hermann-Mauguin, which is
#: what a Wyckoff position's site symmetry is written in.
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
    """Hermann-Mauguin name of the measured point group ``indices`` fix.

    The operations are classified from their own Cartesian matrices
    (:func:`classify`) and the multiset of kinds is looked up: that multiset
    separates all thirty-two crystallographic point groups, so the name is
    arithmetic on the measurement rather than a stored group.  A set of
    operations that is not one of the thirty-two — which a measured site group
    never is — reads as its order instead.
    """
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    return point_group_of([group.symmetries[pos_of[int(ind)]].rotation_cart
                           for ind in indices])


def point_group_of(matrices) -> str:
    """Hermann-Mauguin name of the point group whose operations are ``matrices``.

    An operation's kind and order are its trace and determinant, which no
    change of basis moves, so the matrices may be written in any one basis —
    the Cartesian one, or the lattice basis a table writes its operations in —
    and a set that is not one of the thirty-two reads as its order.
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
    """What one multiplet of orbitals was declared to **be**.

    A ``characters=`` entry, resolved once against the site tables of the group
    it was written in, and carried from there: the constraint it puts on the
    representation is ``{ind: chi(g)}`` over the operations that fixed the site
    *there*, and every later re-identification re-keys those ``.ind`` rather
    than resolving the label again.  Operations a smaller group no longer has
    drop out of :attr:`constraints`, and that dropping **is** the restriction of
    the declared irrep to the smaller site group — which is why a restriction
    that has become reducible (``E`` of ``3m`` subduced to ``3``) is still
    carried and named as a sum, where a fresh resolution would refuse it.

    Attributes
    ----------
    orbitals : tuple of int
        The orbitals this content is carried by jointly — one multiplet.
    spec : object
        Exactly what the caller wrote.
    label : str
        The irrep it resolved to, in the site group it was read in.
    dim : int
        That irrep's dimension, i.e. the multiplet's.
    declared_in : str
        ``"3m in P3m11' #156.50"`` — the site group and the group it was read
        in, so a label the returned group no longer has a name for still says
        where it came from.
    constraints : dict
        ``{ind: chi(g)}`` over the operations that fix the site, keyed by the
        ``.ind`` of the group this record is currently on.
    note : str
        How a shorthand or a generator mapping was resolved, echoed back, and
        which row Mulliken's convention calls each name of :attr:`mulliken`.
    broke : tuple of str
        The operations this declaration was measured to break, named in the
        generator frame — empty until a verdict has been made on it.
    functions : tuple of dict
        The Cartesian polynomials the declaration was made **of**, as
        :data:`SHORTHANDS` writes them, and empty for an irrep label.  A
        shorthand names a *function*; a label names an abstract content.  The
        difference is what step 1 reads: a function's span at a site need not
        be invariant under that site's symmetry, and where it is not, the
        operations that carry it out of itself are not symmetries of the
        crystal that was declared (:func:`..symmetry._content_compatible`).
    reduced_in : str
        ``"m in Pm1' #6.4"`` — the site group this content was **resolved** in
        after a step 1 reduction, and the group it was reduced to.  Empty where
        the declared content left the cell alone, in which case
        :attr:`declared_in` is where it was both read and resolved.
    words : tuple
        ``(orbital, word)`` for every orbital of :attr:`orbitals` a function
        word was written for, spelled as ``characters=`` spells it
        (``(0, "px")``): the declared function each of them is, whichever entry
        of ``characters=`` wrote it, so a multiplet declared orbital by orbital
        holds one per orbital, and a shell (``"p"``) is the word of every
        orbital it was written on.  Empty where no orbital of it was given a
        function word, as for an irrep label, a sum of irreps or generator
        characters alone.
    written : tuple
        ``(orbital, spec)`` for every orbital of :attr:`orbitals`: exactly what
        ``characters=`` wrote for it, whatever it was written as -- ``"px"``
        for one orbital and ``"py"`` for its partner, where :attr:`spec` is one
        entry's.
    corep : bool
        Whether the content was written in the site coreps the rows are
        returned as, one corep or a sum of them (:func:`named_in_coreps`).  It
        then states the corep content and nothing finer: several orbital
        contents can lift to it, and which of them the orbitals hold is chosen
        by the rules that assign anything undeclared.  :attr:`constraints` and
        :attr:`label` hold the one the lowest-angular-momentum rule puts lowest,
        a representative for the record to print: every comparison reads it
        where the spin-1/2 lift has a nonzero trace, where every orbital
        content lifting to the same corep content has its characters, or as the
        coreps it lifts to (:func:`corep_counts`).  Under an operation whose
        lift has zero trace every one of them lifts to zero, so the content
        states nothing there.
    mulliken : tuple
        ``((name, Mulliken's name), ...)`` for each name the declaration wrote
        that the tables give a row Mulliken's convention calls by another
        (:func:`mulliken_names`): ``(("A1''", "A2'"),)`` at ``2c`` of
        ``P6/mmm``.  Empty everywhere else.
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

    def rekeyed(self, pairs) -> "DeclaredContent":
        """The same content on a re-identified group, through ``(old, new)`` pairs.

        Operations the new group does not carry drop out: a declaration
        is a statement about the orbitals, and a group with fewer operations
        asks it fewer questions.
        """
        out = {}
        for old, new in pairs:
            if int(old) in self.constraints:
                out[int(new)] = complex(self.constraints[int(old)])
        return replace(self, constraints=out)


def generator_frame(table) -> str:
    """One-line, human-checkable statement of what every operation is called."""
    return "; ".join(f"#{ind} = {table.names[ind]}" for ind in table.indices)


def _declared_name(table, row) -> str:
    """What a declaration resolving to ``row`` is echoed as.

    The row's **orbital-factor** spelling where it has one, because that is what
    a declaration states and the word it was written in; the returned label
    otherwise.  :attr:`..symmetry.AtomicLimitRow.irrep` is the row's returned
    name either way, so the two sit side by side in the record and a spinful
    declaration can be read back in the words it was written in.
    """
    if row < len(table.declarable) and table.declarable[row]:
        return table.declarable[row]
    return table.labels[row]


def _listed(table) -> tuple:
    """The rows in the order the tables list a site's irreps, the order of their names.

    The tables' own EBR entries list the site irreps of every Wyckoff position
    in the order of their names, ``A1, A2, B1, B2`` at an ``mm2`` site.  Each
    row is placed by the single-valued name a declaration writes it in, and a
    row nothing names comes after the named ones.
    """
    own = table.declarable if any(table.declarable) else table.labels
    return tuple(sorted(range(len(table.labels)),
                        key=lambda p: (not own[p], own[p] or "", p)))


def declaration_words(table) -> tuple:
    """Every label a declaration at this site group may be written in.

    The rows' returned names, and — where a row has a second spelling a
    declaration may use (:attr:`SiteGroupTable.declarable`) — that one beside
    it, the rows in the order the tables list them.  A row nothing named
    contributes only its declarable spelling, and a row with neither
    contributes nothing: it is then reachable by its generator characters
    alone.
    """
    out: list = []
    for row in _listed(table):
        for name in (table.labels[row],
                     table.declarable[row] if row < len(table.declarable) else ""):
            if name and name not in out:
                out.append(name)
    return tuple(out)


def fitting_words(table, n) -> tuple:
    """The words for one irrep each that a site holding ``n`` orbitals can carry.

    ``n`` counts the site's orbitals, or, where the states carry spin, its
    Wannier functions, two states each.  The site irreps of dimension at most
    ``n``, by the name a declaration writes them in -- the orbital factor's
    where the states carry spin -- and there each of them times the spinor
    too, a site corep or a sum: at a ``3m`` site holding one Wannier function,
    ``A1``, ``A2`` and ``-E1``.  The irreps come in order of dimension, the one
    nearest the trivial character first and otherwise in the order the tables
    list them, and the coreps after them.
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
    """The site group's character table as text, for refusal messages.

    One row per irrep, in the order the tables list them, named as a
    declaration names it -- the orbital factor's irrep where the states carry
    spin -- with its own characters on each class; where the states carry
    spin, a last column says which site corep, or sum of them, the row becomes
    with spin.
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
    """A character as text: a value within rounding of zero reads ``0``."""
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
#: the content the real harmonics of degree ``l`` carry at the site, whatever
#: their orientation inside it -- ``"p"`` is ``A1 + E`` at a ``3m`` site and
#: ``T2`` at a ``-43m`` one.  ``"s"`` is also the one ``s`` function, and is
#: read as that shorthand.
SHELLS = {"s": 0, "p": 1, "d": 2, "f": 3}


@lru_cache(maxsize=None)
def _monomials(degree):
    return tuple((a, b, degree - a - b)
                 for a in range(degree + 1) for b in range(degree + 1 - a))


@lru_cache(maxsize=None)
def _monomial_index(degree):
    """``{exponents: column}`` of the monomial basis of one degree."""
    return {m: i for i, m in enumerate(_monomials(degree))}


def _apply(R_inv, poly, degree):
    """``(g.f)(r) = f(R^-1 r)`` expanded in the monomial basis of ``degree``."""
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
    """The :data:`SHORTHANDS` key ``spec`` names, or ``None``.

    The one place a declared word is tested for being a **function** rather
    than a label: spelling is normalized exactly as :func:`resolve_spec`
    normalizes it, so the two never disagree about what a shorthand is.
    """
    if not isinstance(spec, str):
        return None
    key = str(spec).strip().lower().replace(" ", "").replace("_", "")
    return key if key in SHORTHANDS else None


def canonical_shorthand(spec) -> str:
    """The spelling :data:`CANONICAL_SHORTHANDS` gives the orbital ``spec`` names, or ``""``.

    ``"p_x"`` and ``"x"`` are ``"px"``, ``"dxz"`` is ``"dzx"``: one function, one
    word, the one a returned row prints.
    """
    key = shorthand_key(spec)
    if key is None:
        return ""
    poly = SHORTHANDS[key]
    return next((name for name in CANONICAL_SHORTHANDS if SHORTHANDS[name] == poly), key)


def span_basis(polys):
    """``{degree: orthonormal row basis}`` of the span of Cartesian polynomials.

    Every shorthand is **homogeneous**, and an operation of O(3) preserves the
    degree, so a span of several of them is the direct sum of its
    degree-graded pieces and every question about it is asked degree by
    degree.  Rows are orthonormal in the monomial basis of their own degree
    (:func:`_monomials`).
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
    r"""Does operation ``ind`` carry the span ``source`` **onto** ``target``?

    ``(g·f)(r) = f(R⁻¹r)`` on every basis function of ``source``
    (:func:`_apply`), and it does when every image lies in ``target`` and the
    two have the same dimension in every degree — onto, not merely into, so
    that the relation is symmetric and the operations it keeps are closed
    under products.  Both spans come from :func:`span_basis`.
    """
    if set(source) != set(target):
        return False
    if any(source[d].shape[0] != target[d].shape[0] for d in source):
        return False
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    R_inv = np.linalg.inv(np.asarray(
        group.symmetries[pos_of[int(ind)]].rotation_cart, dtype=float))
    for degree, B in source.items():
        T = target[degree]
        for row in B:
            image = _apply(R_inv, _poly_of(row, degree), degree)
            residual = float(np.linalg.norm(image - T.T @ (T @ image)))
            if residual > tol * max(1.0, float(np.linalg.norm(image))):
                return False
    return True


def span_character(group, table, polys):
    """Character of the smallest site-group-invariant span containing ``polys``.

    The site group's operations are applied to the Cartesian polynomials
    (:func:`_apply`), the span of the whole orbit is orthonormalized, and the
    character is the trace of each operation on that span — so the resolution
    is a *measurement* on the site group, not a lookup.  Degrees are summed:
    an operation preserves each graded piece, so the character of the whole is
    the sum of the characters of the pieces.

    Returns ``(chi over the table's classes, dimension)``.
    """
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    mats = {int(ind): np.linalg.inv(np.asarray(
        group.symmetries[pos_of[int(ind)]].rotation_cart, dtype=float))
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
    """Character of the span the shorthand ``name`` generates at this site."""
    return span_character(group, table, [SHORTHANDS[name]])


def shell_key(spec):
    """The :data:`SHELLS` key ``spec`` names, or ``None``, spelled as :func:`shorthand_key` reads it."""
    if not isinstance(spec, str):
        return None
    key = str(spec).strip().lower().replace(" ", "").replace("_", "")
    return key if key in SHELLS else None


def shell_character(group, table, degree) -> np.ndarray:
    """Character of the degree-``l`` real harmonics at this site, over the table's classes.

    The harmonics of degree ``l`` are the polynomials of that degree less
    ``r²`` times those of degree ``l − 2``, so their character is the
    difference of the two spans' own (:func:`span_character`, measured on the
    site group's operations): ``A1 + E`` for ``l = 1`` at a ``3m`` site.
    """
    def _degree(d):
        if d < 0:
            return np.zeros(len(table.classes), dtype=complex)
        chi, _rank = span_character(group, table, [{m: 1.0} for m in _monomials(int(d))])
        return chi

    return _degree(int(degree)) - _degree(int(degree) - 2)


def content_label(table, chi) -> str:
    """A site content spelled as the sum of its irreps, in the words a declaration uses.

    ``chi`` is over the table's classes.  Each irrep is named as a declaration
    names it (:func:`_declared_name`, the orbital factor's word where the
    states carry spin), once with its multiplicity (``"A1 + E"``,
    ``"2xE"``).  ``""`` where the character is no sum of the table's rows, or
    holds a row nothing names.
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
    """A site content spelled as the site coreps it becomes with spin, as a returned row spells them.

    ``chi`` is an orbital content over the table's classes.  Each row's corep
    content is read off its label, and the coreps are counted across the rows:
    ``A1 + E`` at a ``3m`` site is ``2x-E1 + -1E2E``, and ``A1`` and ``A2`` are
    both ``-E1``.  ``""`` where the character is no sum of the table's rows, or
    holds a row nothing names.
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
    """``{row: copies}`` of a site content given over the table's classes, or ``None`` where it is no sum of rows."""
    rows = {}
    for p, n in enumerate(decompose(table, np.asarray(chi, dtype=complex))):
        n = complex(n)
        if abs(n.imag) > 1e-6 or abs(n.real - round(n.real)) > 1e-6 or round(n.real) < 0:
            return None
        if round(n.real):
            rows[p] = int(round(n.real))
    return rows


def _sum_terms(spec):
    """``[(word, copies), ...]`` for a declared sum of irreps, or ``None`` for one word.

    ``"A1 + E"`` is two terms, and ``"2x-E1 + -1E2E"`` -- the spelling a
    returned row is written in -- is ``-E1`` twice and ``-1E2E`` once.
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
    """Whether ``spec`` is written in the site coreps the rows are returned as.

    Where the states carry spin a content can be written in the orbital
    factor's names (``A1``, ``A1 + E``) or in the coreps the rows are returned
    as (``-E1``, ``2x-E1 + -1E2E``).  The second states the corep content and
    nothing finer, one word or a sum alike: several orbital contents can lift
    to it -- ``A1`` and ``A2`` of a ``3m`` site both lift to ``-E1`` -- and
    which of them the orbitals hold is chosen by the rules that assign anything
    undeclared.
    """
    if not isinstance(spec, str) or not any(table.declarable):
        return False
    words = _corep_words(table)
    terms = _sum_terms(spec) or [(str(spec).strip(), 1)]
    return any(_orbital_row(table, word) < 0 and str(word).strip().lower() in words
               for word, _k in terms)


def _corep_words(table) -> frozenset:
    """Every site corep a row of ``table`` lifts to, by name and in lower case."""
    out = set()
    for label in table.labels:
        if label:
            for word, _k in (_sum_terms(label) or [(label, 1)]):
                out.add(str(word).strip().lower())
    return frozenset(out)


def lift_squares(rotations) -> dict:
    """``{ind: |tr S(g)|²}`` -- how much of each operation a site corep's character sees.

    ``rotations`` is ``{ind: Cartesian matrix}``.  The spin-1/2 lift of a turn
    by ``θ`` has trace ``2 cos(θ/2)``, and an improper operation's lift is its
    proper part's, so ``|tr S(g)|² = 2 + 2 cos θ = 1 + tr(det(R) R)``: ``4`` on
    the identity and the inversion, ``1`` on a threefold, and ``0`` on a
    twofold and on a mirror, where every orbital content lifts to a zero
    character.  The sign of a lift is a convention, and nothing read here
    depends on it.
    """
    out = {}
    for ind, R in rotations.items():
        R = np.asarray(R, dtype=float)
        det = 1.0 if float(np.linalg.det(R)) > 0.0 else -1.0
        out[int(ind)] = max(0.0, 1.0 + float(np.trace(det * R)))
    return out


def _corep_matrix(table):
    """``(names, N)`` -- the site coreps, and how often row ``p``'s lift holds corep ``c``.

    Read off the rows' own labels, each a corep or a sum of them; ``None``
    where a row has no name, which leaves its lift unspelled.
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
    """``{corep: copies}`` -- the site coreps an orbital content lifts to, or ``None``.

    ``rows`` is the content as ``{row: copies}`` of ``table``; each row's lift
    is spelled by its label, so a ``p`` shell at a ``3m`` site, ``A1 + E``,
    lifts to ``{-e1: 2, -1e2e: 1}``.  Two contents lifting to one corep content
    count alike, which is what a content written in the coreps states.
    ``None`` where a row the content holds has no name.
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
    """``{row: copies}`` -- a sum of irreps, as rows of the site table; ``{}`` where none.

    A sum written in the orbital factor's names is read row by row.  Where the
    states carry spin a sum can instead be written in the site coreps the rows
    are returned as -- ``2x-E1 + -1E2E`` for a ``p`` shell at a ``3m`` site,
    whose ``-1E2E`` is half of the ``E`` row's ``-1E2E + -E1``, or ``-E1``
    alone, a sum of one -- and then the coreps are counted and the rows that
    lift to exactly those counts are found.  Several orbital contents can lift
    to one sum (``A1 + E`` and ``A2 + E`` above, ``A1`` and ``A2`` for ``-E1``,
    and at a ``-43m`` site the ``E + T2`` of a ``d`` shell and ``A2 + 2xE``).
    Such a sum states the corep content alone
    (:func:`named_in_coreps`), which every comparison with the Hamiltonian
    reads and the record names (:func:`corep_label`), and the orbital content
    read here stands for it: the one the lowest-angular-momentum rule puts
    lowest -- the lowest single shell of real harmonics holding it, then the
    per-orbital sum (:func:`_content_shells`, over ``rotations``, the Cartesian
    matrix of each operation of the site group) -- then the one of the fewest
    irreps, then the one closest to the trivial character.
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
        """``(single shell, per orbital)`` of one orbital content (:func:`_content_shells`)."""
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
    """How many operations of the site group a row's character is not ``+1`` on."""
    return sum(len(table.classes[c]) for c in range(len(table.classes))
               if abs(complex(table.characters[row, c]) - 1.0) > 1e-9)


def _orbital_row(table, word) -> int:
    """The row an orbital factor's own name (:attr:`SiteGroupTable.declarable`) names, or ``-1``."""
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
    r"""The inner product a rotation preserves, on the polynomials of one degree.

    Two Cartesian monomials of the same degree are not orthonormal as functions
    on the sphere, and a rotation written in the monomial basis is therefore not
    an orthogonal matrix there: measured, a turn by ``theta`` about ``z`` takes
    ``xy`` to a vector of plain monomial norm ``sqrt(cos^2 2t + sin^2 2t / 2)``.
    The inner product it does preserve gives ``x^a y^b z^c`` the weight
    ``a! b! c! / (a+b+c)!``, which is the one real spherical harmonics of a
    degree are orthonormal in.  Working in ``sqrt(weight)``-scaled coordinates
    turns it into the ordinary Euclidean one, and that is what makes an
    operation's matrix on them orthogonal.
    """
    w = np.array([float(factorial(a) * factorial(b) * factorial(c))
                  / float(factorial(int(degree)))
                  for a, b, c in _monomials(int(degree))])
    w.flags.writeable = False
    return w


def _shell_vector(name, degree, scale):
    """One orbital function as a unit vector in the scaled monomial coordinates."""
    idx = _monomial_index(int(degree))
    v = np.zeros(len(idx))
    for m, c in SHORTHANDS[name].items():
        v[idx[m]] += float(c)
    v = v * np.asarray(scale, dtype=float)
    return v / float(np.linalg.norm(v))


def shell_operator(group, ind, degree):
    r"""One operation's matrix on the polynomials of one degree.

    ``(g.f)(r) = f(R^-1 r)`` (:func:`_apply`) on every monomial, written in the
    ``sqrt(weight)``-scaled coordinates of :func:`_shell_weights`, where a
    rotation is orthogonal, so that everything an orbital function does under an
    operation is read off this one matrix.  It is the same matrix for an
    antiunitary operation as for its unitary partner, because complex
    conjugation does nothing to a real function.
    """
    degree = int(degree)
    scale = np.sqrt(_shell_weights(degree))
    pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
    R_inv = np.linalg.inv(np.asarray(
        group.symmetries[pos_of[int(ind)]].rotation_cart, dtype=float))
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
    r"""The real harmonics of one degree, one unit column each, in :func:`shell_operator`'s coordinates.

    ``s``, ``p`` and ``d`` are the shorthands' own functions, in the order they
    are listed.  A higher shell names none, and its functions are the monomials
    in their own order with the part ``r²`` times the degree below removed, made
    orthonormal one after another.
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
    """``(degree, unit column)`` -- the orbital a shorthand names, in :func:`shell_operator`'s coordinates."""
    key = shorthand_key(word)
    if key is None:
        return None
    degree = max(sum(m) for m in SHORTHANDS[key])
    return int(degree), _shell_vector(key, degree, np.sqrt(_shell_weights(degree)))


def irrep_functions(group, table, row, rotations):
    r"""``(degree, columns)`` -- the functions the tables write one site irrep on, or ``None``.

    They are the irrep's lowest-angular-momentum real harmonics (the shell
    :func:`_lowest_l` finds it in): the shell's own functions
    (:func:`_harmonic_functions`) projected onto the irrep, kept in their order
    where independent, as many as its dimension -- ``(p_x, p_y)`` for ``E`` at a
    ``4mm`` site, ``d_{x²−y²}`` for its ``B1``.  ``None`` where the projections
    span no invariant space of that dimension, as where the shell holds the irrep
    more than once and its first functions straddle two copies.
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
    """Multiplicity of every irrep in a character (``n_p`` by orthogonality).

    Each row is normalized by **its own** inner product rather than by
    ``table.order`` alone.  For an ordinary irrep that norm is 1 and nothing
    changes; for a **merged** time-reversal pair (``1E2E``) the row is the
    *sum* of two inequivalent characters, so its self inner product is 2, and
    dividing by it is what makes a span carrying the pair once come back as
    multiplicity 1 instead of 2.  Measured on P4_232's ``2a`` site (group
    ``23``): the ``d_z2`` span has ``chi = (2, 2, -1, -1)``, which *is* the
    ``1E2E`` row -- without the normalization it reported ``2x1E2E`` and
    :func:`resolve_spec` refused the paper's own orbital name.
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
#: nine (``A1u`` of ``m-3m`` is the last, at nine), so a content still missing at
#: twelve is no character of the site group and ranks after every one that is.
_L_MAX = 12


def _shell_trace(rotation_cart, degree) -> float:
    r"""The character of the degree-``l`` real harmonics under one operation.

    ``det^l · sin((2l+1)θ/2) / sin(θ/2)``, with ``θ`` the rotation angle of
    ``det·R`` — the proper rotation an improper operation is followed by the
    inversion of — and ``2l + 1`` at ``θ = 0``.  The harmonics of degree ``l``
    have parity ``(−1)^l``, which is the ``det^l``.
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
    """The lowest degree of real harmonics among which a site content occurs.

    ``chi`` is the content's orbital character, ``{ind: chi(g)}`` over the site
    group ``table`` is the table of, and ``rotations`` the Cartesian matrix of
    each of those operations.  The content occurs at degree ``l`` when every
    irrep it holds is held by the degree-``l`` harmonics at least as often,
    read from the two characters by :func:`decompose`.  A content that is no
    character of the site group — multiplicities that are not non-negative
    integers — returns ``_L_MAX + 1``, and so does one no degree up to
    ``_L_MAX`` holds.
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
    """How often the degree-``l`` real harmonics hold each irrep of a site group.

    One count per row of ``table``, read off the shell's own character
    (:func:`_shell_trace`) by :func:`decompose`: the ``d`` shell at a ``-3m``
    site holds ``A1g`` once and ``Eg`` twice.  ``rotations`` is the Cartesian
    matrix of each operation of the site group.
    """
    cache = {} if cache is None else cache
    key = (id(table), int(degree))
    if key not in cache:
        shell = np.asarray([_shell_trace(rotations[int(cls[0])], degree)
                            for cls in table.classes], dtype=complex)
        cache[key] = [int(round(float(np.real(h)))) for h in decompose(table, shell)]
    return cache[key]


def _content_shells(table, theta, rotations, *, shells=None, lowest=None):
    """``(single shell, per orbital)`` — the angular momentum a site content sits at.

    The two numbers the lowest-angular-momentum rule compares contents by, in
    that order: the lowest single shell of real harmonics holding the whole
    content (:func:`_lowest_l`), and the sum over its irreps of multiplicity ×
    dimension × the lowest shell holding each.  A lone ``d`` shell at a
    ``-3m`` site is ``2`` taken whole, where one ``p`` shell and a second ``p``
    pair would be ``1`` orbital by orbital, and a content of mixed parity,
    which no single shell holds, is scored orbital by orbital alone.
    ``shells`` and ``lowest`` keep what earlier comparisons measured, the
    shells' own contents and each irrep's lowest shell.
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
    """Mulliken's name of one row at a ``-6m2``, ``-42m``, ``6`` or ``6/m`` site, or ``""``.

    Mulliken's letters and marks, read off the row's characters on operations
    no choice of axes moves.  At ``-6m2``, ``1``/``2`` follow the twofolds
    perpendicular to the threefold axis and ``'``/``''`` the horizontal
    mirror; at ``-42m``, ``A``/``B`` follow the ``-4`` and ``1``/``2`` the
    twofolds perpendicular to its axis; at ``6`` and ``6/m``, ``A``/``B`` and
    ``E1``/``E2`` follow the twofold along the sixfold axis, ``-1`` on each
    state of ``E1``, and ``g``/``u`` the inversion.  A complex row, or a pair
    of them glued by time reversal, is named by its pair.  Every other site
    group returns ``""``.
    """
    symbol = point_group_symbol(group, table.indices)
    if symbol not in ("-6m2", "-42m", "6", "6/m"):
        return ""
    infos = [_class_info(group, cls)[1] for cls in table.classes]
    chi = np.asarray(table.characters[row], dtype=complex)
    dim = float(chi[next(c for c, info in enumerate(infos) if info["kind"] == "E")].real)

    def find(kind, order, axis=None, along=True):
        """The class of this kind and order, along ``axis`` or perpendicular to it.

        In these four groups an operation's axis lies along the principal axis
        or perpendicular to it, so the cosine between the two is one or zero.
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
    """``name`` as Mulliken's convention spells it.

    A pair of complex rows glued by time reversal, or one member of it, is
    spelled by its pair.
    """
    pair = re.fullmatch(r"1(E.*)2\1", name)
    if pair is not None:
        return pair.group(1)
    member = re.fullmatch(r"[12](E.*)", name)
    return name if member is None else member.group(1)


def mulliken_names(group, table, spec) -> tuple:
    """``((name, Mulliken's name), ...)``: names ``spec`` writes that the tables give another row.

    At four kinds of site the tables' single-valued names break a mark no
    choice of axes moves: at a ``-6m2`` site whose band-representation key
    writes ``-6m2`` their ``A1''`` is the row Mulliken's convention calls
    ``A2'``, and the reverse; at a ``-42m`` site keyed ``-4m2`` their ``B2``,
    ``A2`` and ``B1`` are Mulliken's ``A2``, ``B1`` and ``B2``; and at ``6``
    and ``6/m`` sites ``E1`` and ``E2`` are exchanged.  A declaration written
    in such a name states the tables' row.  Each name ``spec`` writes, alone
    or in a sum, is compared with Mulliken's name of the row it names
    (:func:`_mulliken_name`); a content written in the site coreps, a
    function and generator characters name no row by its name.
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
    """The clause a declaration's record carries for :func:`mulliken_names`' pairs."""
    parts = []
    for name, mulliken in names:
        member = (re.fullmatch(r"[12]E.*", name) is not None
                  and re.fullmatch(r"1(E.*)2\1", name) is None)
        what = "a member of the pair" if member else "the row"
        parts.append(f"the tables' {name} at this site is {what} Mulliken's convention "
                     f"calls {mulliken}")
    return "; ".join(parts)


def resolve_spec(group, table, spec, where):
    """Turn one declaration value into ``(constraints, label, dim, note)``.

    ``constraints`` is ``{ind: required character value}`` over the site
    group's operations; ``label`` is what the echo record reports.

    A value is an irrep of the site group, by its name or, where the states
    carry spin, by its orbital factor's (:meth:`SiteGroupTable.index_of`); a
    shorthand, which names one function and so one irrep; a **shell**
    (:data:`SHELLS`), the content the real harmonics of one angular momentum
    carry here, ``A1 + E`` for ``"p"`` at a ``3m`` site; a **sum of irreps**,
    ``"A1 + E"``; a content written in the site coreps the rows are returned
    as, one corep or a sum of them, ``"-E1"`` or a returned row's own
    ``"2x-E1 + -1E2E"``, which states the corep content alone
    (:func:`named_in_coreps`), is named in the coreps (:func:`corep_label`) and
    is represented by the orbital content lifting to it that the
    lowest-angular-momentum rule puts lowest (:func:`_resolve_sum`); or
    generator characters.  A shell or a sum is a content of several irreps,
    and its ``dim`` is theirs together.

    A :class:`DeclaredContent` passes through unchanged but **restricted**: its
    constraints were resolved in a larger group and the operations this site
    group no longer has are dropped, which is the subduction of the declared
    irrep.  The label it carries is the one it was declared as, in the group it
    was read in; naming what the restriction is *here* belongs to
    (:func:`..symmetry._label_character`), since a subduced irrep can be
    reducible and has a sum for a name rather than a row of this table.

    Raises
    ------
    DeclarationError
        ``gate="unknown_label"`` when the value names no irrep of *this* site
        group, no sum of them, no shell, no shorthand and no operation.
    """
    if isinstance(spec, DeclaredContent):
        keep = {int(i) for i in table.indices}
        return ({int(k): complex(v) for k, v in spec.constraints.items() if int(k) in keep},
                str(spec.label), int(spec.dim), str(spec.note))
    if isinstance(spec, dict):
        return _resolve_generator_dict(table, spec, where)

    # A CONTENT WRITTEN IN THE SITE COREPS, one word or a sum, states the corep
    # content alone and is read as a sum below; any other word names a row
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
                f"{where}: the shorthand {spec!r} spans {parts} at this site "
                f"({', '.join(words)}), so it does not name one site irrep. "
                f"Name the irrep directly.\n  frame: {generator_frame(table)}",
                {"declared_as": spec, "decomposition": parts, "irreps": words})
        p = rows[0]
        return ({int(ind): complex(table.characters[p, table.class_of[int(ind)]])
                 for ind in table.indices},
                _declared_name(table, p), int(table.dims[p]),
                f"{spec!r} resolved to {_declared_name(table, p)} on this site group")

    # A SHELL: the content the real harmonics of one degree carry here, which
    # no orientation of the functions inside the site moves
    shell = shell_key(spec)
    if shell is not None:
        chi = shell_character(group, table, SHELLS[shell])
        label = content_label(table, chi)
        return ({int(ind): complex(chi[table.class_of[int(ind)]]) for ind in table.indices},
                label, 2 * SHELLS[shell] + 1,
                f"{spec!r} resolved to {label} on this site group")

    # A SUM OF IRREPS, written in either vocabulary: the content they add up to,
    # and a single corep as the sum of one
    terms = _sum_terms(spec) or ([(str(spec).strip(), 1)] if corep else None)
    rows = {}
    if terms:
        pos_of = {int(s.ind): i for i, s in enumerate(group.symmetries)}
        rows = _resolve_sum(table, terms, rotations={
            int(ind): np.asarray(group.symmetries[pos_of[int(ind)]].rotation_cart, dtype=float)
            for ind in table.indices})
    if rows:
        chi = sum(n * np.asarray(table.characters[p], dtype=complex) for p, n in rows.items())
        # a content written in the site coreps is named in them: the orbital
        # content standing for it is one of several that lift to it
        label = corep_label(table, chi) if corep else content_label(table, chi)
        return ({int(ind): complex(chi[table.class_of[int(ind)]]) for ind in table.indices},
                label, int(sum(n * int(round(float(np.real(table.dims[p]))))
                               for p, n in rows.items())),
                (f"{spec!r} states the corep content, and the orbital factor under it is "
                 "assigned as for an undeclared orbital" if corep else
                 f"{spec!r} resolved to {label} on this site group"))

    said = (f"{spec!r} is not a site irrep times the spinor, or a sum of them" if corep else
            f"{spec!r} is not an irrep of this site group, a sum of its irreps, "
            f"a shell or a known shorthand")
    raise DeclarationError(
        "unknown_label",
        f"{where}: {said}.\n"
        f"  site group: order {table.order}, classes "
        f"{', '.join(table.class_names)}\n"
        f"  valid labels: {', '.join(declaration_words(table)) or '(none: name the '
        f'generator characters instead)'}\n"
        f"  shells:       {', '.join(SHELLS)}\n"
        f"  shorthands:   {', '.join(CANONICAL_SHORTHANDS)}\n"
        f"  frame: {generator_frame(table)}\n"
        f"{table_summary(table)}",
        {"declared_as": spec, "irreps": declaration_words(table), "site_group": table.indices,
         "frame": dict(table.names)})


def _resolve_generator_dict(table, spec, where):
    """The naming-free path: ``{operation: character}`` straight to constraints."""
    constraints, seen = {}, []
    for key, value in spec.items():
        inds = _match_operation(table, key, where)
        cls = {table.class_of[i] for i in inds}
        if len(cls) != 1:
            raise DeclarationError(
                "unknown_label",
                f"{where}: the key {key!r} matches operations in more than one conjugacy "
                f"class ({sorted(cls)}), and a character is a class function. Name one "
                f"operation.\n  frame: {generator_frame(table)}",
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


def _match_operation(table, key, where):
    """Operations of the site group a user key names (canonical name or ``.ind``)."""
    if isinstance(key, (int, np.integer)):
        if int(key) not in table.class_of:
            raise DeclarationError(
                "unknown_label",
                f"{where}: operation #{int(key)} does not fix this site, so it is not part "
                f"of its site group ({', '.join(str(i) for i in table.indices)}).\n"
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
        f"{where}: {key!r} names no operation of this site group.\n"
        f"  frame: {generator_frame(table)}",
        {"key": key, "frame": dict(table.names)})
