"""The integer arithmetic of the elementary band representations: the Smith normal
form of their matrix, a content in its coordinates, and the compatibility
relations read off it.
"""

from __future__ import annotations

import numpy as np

from .refusal import InternalError, Refusal


# --------------------------------------------------------------------------
# the Smith path: exact integer arithmetic, shared with the future witness
# --------------------------------------------------------------------------

def _snf(matrix):
    """Smith normal form over ℤ, by elimination: ``(U, D, V)`` with ``U M V = D``.

    Deterministic and exact — plain Python integers throughout, no floats and no
    pivoting choices left to chance, so two runs on the same table return the
    same matrices.  ``U`` and ``V`` are unimodular and ``D`` is diagonal with
    ``d_i | d_{i+1}``.  Written out here rather than delegated because it is the
    *verifier*: its whole job is to be an independent computation of the Smith
    form ``irreptables`` ships.
    """
    A = [[int(x) for x in row] for row in matrix]
    m = len(A)
    n = len(A[0]) if m else 0
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def rswap(i, j):
        A[i], A[j] = A[j], A[i]
        U[i], U[j] = U[j], U[i]

    def cswap(i, j):
        for r in A:
            r[i], r[j] = r[j], r[i]
        for r in V:
            r[i], r[j] = r[j], r[i]

    def radd(dst, src, c):
        A[dst] = [a + c * b for a, b in zip(A[dst], A[src])]
        U[dst] = [a + c * b for a, b in zip(U[dst], U[src])]

    def cadd(dst, src, c):
        for r in A:
            r[dst] += c * r[src]
        for r in V:
            r[dst] += c * r[src]

    t = 0
    while t < min(m, n):
        piv = None
        for i in range(t, m):
            for j in range(t, n):
                if A[i][j] and (piv is None or abs(A[i][j]) < abs(A[piv[0]][piv[1]])):
                    piv = (i, j)
        if piv is None:
            break
        rswap(t, piv[0])
        cswap(t, piv[1])
        while True:
            for i in range(t + 1, m):
                if A[i][t]:
                    radd(i, t, -(A[i][t] // A[t][t]))
            bad = next((i for i in range(t + 1, m) if A[i][t]), None)
            if bad is not None:
                rswap(t, bad)
                continue
            for j in range(t + 1, n):
                if A[t][j]:
                    cadd(j, t, -(A[t][j] // A[t][t]))
            bad = next((j for j in range(t + 1, n) if A[t][j]), None)
            if bad is not None:
                cswap(t, bad)
                continue
            off = next(((i, j) for i in range(t + 1, m) for j in range(t + 1, n)
                        if A[i][j] % A[t][t]), None)
            if off is None:
                break
            radd(t, off[0], 1)                 # pull a non-divisible entry into the row
        if A[t][t] < 0:
            A[t] = [-a for a in A[t]]
            U[t] = [-a for a in U[t]]
        t += 1
    return U, A, V


def _matmul(a, b):
    """Exact integer matrix product of two lists of lists."""
    n = len(b)
    return [[sum(row[k] * b[k][j] for k in range(n)) for j in range(len(b[0]))]
            for row in a]


def _unimodular_inverse(u):
    """The (integer) inverse of a unimodular integer matrix, by exact elimination."""
    from fractions import Fraction

    n = len(u)
    a = [[Fraction(int(x)) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(u)]
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c]), None)
        if p is None:
            raise InternalError(
                "the tabulated Smith factor u is singular, so it is not a change of "
                "basis at all; the shipped Smith form is not one.")
        a[c], a[p] = a[p], a[c]
        piv = a[c][c]
        a[c] = [x / piv for x in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    out = []
    for r in range(n):
        row = []
        for x in a[r][n:]:
            if x.denominator != 1:
                raise InternalError(
                    "the tabulated Smith factor u has a non-integer inverse, so it is "
                    "not unimodular; the shipped Smith form is not one.")
            row.append(int(x))
        out.append(row)
    return out


def _smith_system(data, number_str):
    """``(u, v, d, matrix)`` from :func:`.groups.ebr_data`, as exact integers.

    ``d`` is the diagonal of ``R`` extended with zeros to one entry per EBR row, so
    ``d[i] == 0`` says exactly "row ``i`` of ``R`` is empty", which is the
    compatibility-relation half of ``y'``.  ``irreptables`` stores a degenerate
    Smith factor as a bare scalar where the group has a single EBR column
    (measured on 63 of the 920 unitary-and-grey systems); both shapes are
    normalized here.
    """
    u, r, v = data["smith"]
    matrix = np.asarray(data["matrix"], dtype=int)
    n_row, n_col = matrix.shape
    u = np.asarray(u, dtype=int)
    r = np.asarray(r, dtype=int)
    v = np.atleast_2d(np.asarray(v, dtype=int))
    if u.shape != (n_row, n_row) or v.shape != (n_col, n_col):
        raise InternalError(
            f"#{number_str}: the tabulated Smith factors have shapes u{u.shape}, "
            f"v{v.shape} against an EBR matrix {matrix.shape}. The Smith form the "
            "tables ship is what fixes this package's indicator basis, so a shape "
            "that cannot be that form is a bug here, not a property of the model.")
    if r.ndim == 1:                            # single EBR column: R is stored flat
        d = [int(x) if i == 0 else 0 for i, x in enumerate(r)]
    else:
        diag = [int(r[i, i]) for i in range(min(r.shape))]
        d = diag + [0] * (n_row - len(diag))
    return ([[int(x) for x in row] for row in u], [[int(x) for x in row] for row in v],
            d, [[int(x) for x in row] for row in matrix])


def _filled_content(counts, restriction, labels):
    """``(y, reconstructed)``: the content on every row of the basis, a dropped row read off the row it repeats.

    A dropped row that duplicates a kept one carries that row's multiplicity
    in any sum of the band representations it was compared on, so its value is
    arithmetic and nothing is guessed: :func:`..ebr.symmetry_indicator` and
    :func:`..ebr.decompose` complete the vector this way before the compatibility
    relations are read, each once every dropped row repeats a kept one.  A
    value supplied for a dropped row is left as it is, and the relations say
    whether it agrees.
    """
    filled = dict(counts)
    reconstructed = {}
    for dropped, sources in restriction.copies.items():
        if not sources or dropped in counts:
            continue
        filled[dropped] = int(counts.get(sources[0], 0))
        reconstructed[str(dropped)] = str(sources[0])
    return [int(filled.get(label, 0)) for label in labels], reconstructed


def _compatibility(data, number_str, labels, y):
    """``(smith, y_prime, refusal)``: the content in the Smith coordinates of the EBR matrix.

    ``smith`` is :func:`_smith_system`'s ``(u, v, d, matrix)`` and ``y_prime``
    is ``u y``.  The coordinates where ``d`` is zero vanish for every band
    structure of the group -- they are its compatibility relations -- and
    ``refusal`` is ``"not_a_band_structure"`` where one does not, ``None``
    otherwise.  :func:`..ebr.symmetry_indicator` reads its class off the same
    coordinates, and :func:`..ebr.decompose` asks the same question before it solves.
    """
    u, v, d, matrix = _smith_system(data, number_str)
    y_prime = [sum(u[i][j] * y[j] for j in range(len(y))) for i in range(len(u))]
    off = [(i, labels[i] if i < len(labels) else i, y_prime[i])
           for i in range(len(y_prime)) if d[i] == 0 and y_prime[i]]
    if not off:
        return (u, v, d, matrix), y_prime, None
    return (u, v, d, matrix), y_prime, Refusal(
        "not_a_band_structure",
        f"the content violates {len(off)} of #{number_str}'s compatibility "
        f"relations: the Smith coordinates {[(i, int(v2)) for i, _l, v2 in off]} must "
        "vanish for any band structure of this group and do not. These multiplicities "
        "are those of no set of bands separated from the rest at every k-point, as when "
        "the bands counted cut through a crossing between two high-symmetry points, so "
        "they have no class in {BS}/{AI} and no decomposition. Check that a gap separates "
        "the bands counted from the others at every k-point and that the labels came "
        "from this group.",
        value=float(max(abs(int(v2)) for _i, _l, v2 in off)), threshold=0.0,
        context={"violations": tuple((int(i), str(l), int(v2)) for i, l, v2 in off),
                 "y_prime": tuple(int(x) for x in y_prime),
                 "number_str": str(number_str)})


def _structure(divisors) -> str:
    return " x ".join(f"Z{d}" for d in divisors) or "trivial"


def _divisors_of(data, number_str) -> tuple:
    """The nontrivial elementary divisors of a group's EBR matrix — its ``X_BS``."""
    return tuple(int(x) for x in _smith_system(data, number_str)[2] if int(x) > 1)


def _verify_smith(number_str, matrix, u, v, d, y, slots, divisors, values,
                  order) -> dict:
    """The de-novo Smith computation, and what it is allowed to disagree about.

    The elementary divisors and the order of the class are properties of the
    *matrix* and of the *content*: no choice of Smith basis can move them, so a
    disagreement is this package misapplying a convention the tables fix, and
    raises :class:`~.refusal.InternalError`.  The individual components can
    legitimately differ between two Smith bases — that freedom is an
    automorphism of ``⊕ᵢ ℤ_{d_i}`` — so the change of basis is *recorded*
    instead, and whether it is the identity is reported rather than assumed.
    """
    where = (f"#{number_str}: the Smith form of the EBR matrix is what fixes this "
             "package's indicator basis, and irreptables ships it with the rows; "
             "failing to apply it consistently is a bug here, not a property of any "
             "model")
    if _matmul(_matmul(u, matrix), v) != [[int(x) for x in row] for row in
                                          _diagonal_like(d, matrix)]:
        raise InternalError(
            f"{where}: the shipped factors do not satisfy u M v = R on the transported "
            "EBR matrix, so what was transported is not a Smith decomposition of it.")

    u0, d0, _v0 = _snf(matrix)
    divisors0 = tuple(int(d0[i][i]) for i in range(min(len(d0), len(d0[0])))
                      if int(d0[i][i]) > 1)
    if sorted(divisors0) != sorted(divisors):
        raise InternalError(
            f"{where}: an independent integer Smith computation of the same matrix gives "
            f"elementary divisors {sorted(divisors0)} against the shipped "
            f"{sorted(divisors)}. The divisors are a property of the matrix; two "
            "computations of them cannot disagree unless one is wrong.")

    slots0 = [i for i in range(min(len(d0), len(d0[0]))) if int(d0[i][i]) > 1]
    y0 = [sum(u0[i][j] * y[j] for j in range(len(y))) for i in range(len(u0))]
    values0 = tuple(int(y0[i]) % int(d0[i][i]) for i in slots0)
    order0 = 1
    for val, div in zip(values0, (int(d0[i][i]) for i in slots0)):
        step = div // np.gcd(int(val), int(div)) if val else 1
        order0 = int(np.lcm(order0, int(step)))
    if order0 != order or bool(any(values0)) != bool(any(values)):
        raise InternalError(
            f"{where}: the de-novo computation puts this content in a class of order "
            f"{order0} against the shipped basis's {order}. The order of a class is "
            "basis-independent, so the two Smith bases are not identifying the same "
            "quotient.")

    change = _matmul(u0, _unimodular_inverse(u))
    restricted = tuple(tuple(int(change[a][b]) % int(divisors[k])
                             for k, b in enumerate(slots))
                       for a in slots0)
    identity = all(restricted[i][j] == (1 % int(divisors[j]) if i == j else 0)
                   for i in range(len(slots0)) for j in range(len(slots)))
    return {
        "performed": True,
        "divisors_denovo": tuple(sorted(divisors0)),
        "agrees": True,
        "shipped_is_smith": True,
        "denovo_is_smith": True,
        "values_denovo": values0,
        "order_denovo": int(order0),
        "change_of_basis": restricted,
        "change_of_basis_is_identity": bool(identity),
        "note": ("an independent exact-integer Smith normal form of the transported EBR "
                 "matrix reproduces the shipped elementary divisors and the class order; "
                 "the two Smith bases differ by the recorded automorphism of the "
                 "indicator group, and the returned values are the shipped basis's"),
    }


def _diagonal_like(d, matrix):
    """``R`` rebuilt from its diagonal, in the shape of ``matrix``."""
    n_row, n_col = len(matrix), len(matrix[0])
    return [[int(d[i]) if i == j and i < len(d) else 0 for j in range(n_col)]
            for i in range(n_row)]
