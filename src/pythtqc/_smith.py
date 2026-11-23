"""The integer arithmetic of a group's matrix of elementary band representations.

The Smith normal form of the matrix is the one the irreptables package ships.
A symmetry data vector is written in its coordinates, and the compatibility
relations are the coordinates that vanish for every band structure of the
group.  The Smith normal form is also computed again, exactly, and the
shipped one is checked against it.
"""

from __future__ import annotations

import numpy as np

from .refusal import InternalError, Refusal


# --------------------------------------------------------------------------
# The Smith normal form and what is read off it, in exact integer arithmetic
# --------------------------------------------------------------------------

def _snf(matrix):
    """The Smith normal form of an integer matrix, by exact elimination.

    ``matrix`` is the integer matrix ``M``, a sequence of rows.  Returns
    ``(U, D, V)``, each a list of rows of Python integers, with
    ``U M V = D``.  ``U`` and ``V`` are unimodular.  ``D`` is diagonal with
    nonnegative entries, the nonzero ones first, each dividing the next.
    Each diagonal position starts from the nonzero entry of least magnitude,
    the first in row order, and every step is fixed by the matrix alone, so
    a matrix always gives the same ``U`` and ``V``.
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
    """The exact product ``a b`` of two integer matrices.

    ``a`` and ``b`` are lists of rows, ``a`` with as many columns as ``b``
    has rows.  Returns the product as a list of rows.
    """
    n = len(b)
    return [[sum(row[k] * b[k][j] for k in range(n)) for j in range(len(b[0]))]
            for row in a]


def _unimodular_inverse(u):
    """The inverse of a unimodular integer matrix, by exact elimination.

    ``u`` is a square integer matrix, a sequence of rows: the left factor of
    a Smith form the tables ship.  Returns its inverse as a list of rows of
    integers.  Raises :class:`~.refusal.InternalError` where ``u`` is
    singular or its inverse is not an integer matrix.
    """
    from fractions import Fraction

    n = len(u)
    a = [[Fraction(int(x)) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(u)]
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c]), None)
        if p is None:
            raise InternalError(
                "the left factor of the Smith normal form the irreptables package ships "
                "is singular. Report it, with the model and the call.")
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
                    "the left factor of the Smith normal form the irreptables package "
                    "ships has an inverse that is not an integer matrix. Report it, "
                    "with the model and the call.")
            row.append(int(x))
        out.append(row)
    return out


def _smith_system(data, number_str):
    """The Smith form the irreptables package ships for a group's EBR matrix.

    ``data`` is the mapping :func:`.._tables._ebr_data` returns, of which
    ``"matrix"``, the EBR matrix ``M``, and ``"smith"``, the factors ``u``
    and ``v`` with the diagonal matrix ``R``, as ``(u, R, v)`` with
    ``u M v = R``, are read.  ``number_str`` is the group's number, as
    ``"166.98"``, for the message.  Returns ``(u, v, d, matrix)``, each a
    list of Python integers or of rows of them.  ``d`` is the diagonal of
    ``R`` extended with zeros to one entry per row of ``M``, so ``d[i]`` is
    zero exactly where row ``i`` of ``R`` is zero.  A group with a single
    EBR column, whose ``R`` the tables store as a vector and ``v`` as a
    number, is returned in the same form: ``v`` as a 1 x 1 matrix, ``d``
    with one entry per row of ``M``.  Raises
    :class:`~.refusal.InternalError` where the factors' shapes do not fit
    ``M``.
    """
    u, r, v = data["smith"]
    matrix = np.asarray(data["matrix"], dtype=int)
    n_row, n_col = matrix.shape
    u = np.asarray(u, dtype=int)
    r = np.asarray(r, dtype=int)
    v = np.atleast_2d(np.asarray(v, dtype=int))
    if u.shape != (n_row, n_row) or v.shape != (n_col, n_col):
        raise InternalError(
            f"#{number_str}: the factors u and v of the Smith normal form the "
            f"irreptables package ships have shapes {u.shape} and {v.shape}, and the "
            f"EBR matrix M has shape {matrix.shape}. Report it, with the model and "
            "the call.")
    if r.ndim == 1:                            # single EBR column: R is stored flat
        d = [int(x) if i == 0 else 0 for i, x in enumerate(r)]
    else:
        diag = [int(r[i, i]) for i in range(min(r.shape))]
        d = diag + [0] * (n_row - len(diag))
    return ([[int(x) for x in row] for row in u], [[int(x) for x in row] for row in v],
            d, [[int(x) for x in row] for row in matrix])


def _filled_content(counts, restriction, labels):
    """The symmetry data vector on every row, each dropped row read off a kept one.

    A row of the EBR matrix is a little-group representation at a
    high-symmetry point.  ``labels`` lists the rows' labels, in order.
    ``counts`` maps a label to its multiplicity.  ``restriction`` is the
    :class:`..ebr.Restriction` of the points measured.  A row is kept where
    its point is one of those, and dropped where it is not.
    ``restriction.copies`` maps each dropped row to the kept rows it
    repeats, those that occur as often as it in every elementary band
    representation they were compared on, on a layer, the layer's own.  A
    dropped row absent from ``counts`` takes the multiplicity of the first
    kept row it repeats.  A multiplicity ``counts`` gives is kept, and the
    compatibility relations say whether it agrees.  Any other row absent
    from ``counts`` has multiplicity zero.  :func:`..ebr.symmetry_indicator`
    and :func:`..ebr.decompose` call it only where every dropped row repeats
    a kept one.  Returns ``(y, reconstructed)``: ``y``, the multiplicities
    in the order of ``labels``; ``reconstructed``, the label of each dropped
    row so filled, mapped to the label it was read off.
    """
    filled = dict(counts)
    reconstructed = {}
    for dropped, sources in restriction.copies.items():
        if not sources or dropped in counts:
            continue
        filled[dropped] = int(counts.get(sources[0], 0))
        reconstructed[str(dropped)] = str(sources[0])
    return [int(filled.get(label, 0)) for label in labels], reconstructed


def _compatibility(data, number_str, y):
    """The symmetry data vector in Smith coordinates, checked against the compatibility relations.

    ``data`` and ``number_str`` are as :func:`_smith_system` takes them.
    ``y`` is the symmetry data vector on every row of the EBR matrix, as
    :func:`_filled_content` returns it.  With ``u`` and ``d`` as
    :func:`_smith_system` returns them, the coordinates of ``y' = u y`` where
    ``d`` is zero vanish for every band structure of the group: they are its
    compatibility relations.
    Returns ``(smith, y_prime, refusal)``: ``smith``, the
    ``(u, v, d, matrix)`` of :func:`_smith_system`; ``y_prime``, ``y'`` as a
    list; ``refusal``, ``not_a_band_structure`` naming each of those
    coordinates that does not vanish, as ``(index, value)``, or ``None``
    where all do.
    """
    u, v, d, matrix = _smith_system(data, number_str)
    y_prime = [sum(u[i][j] * y[j] for j in range(len(y))) for i in range(len(u))]
    off = [(i, y_prime[i]) for i in range(len(y_prime)) if d[i] == 0 and y_prime[i]]
    if not off:
        return (u, v, d, matrix), y_prime, None
    return (u, v, d, matrix), y_prime, Refusal(
        "not_a_band_structure",
        f"the symmetry data vector violates #{number_str}'s compatibility relations: "
        f"{len(off)} of its Smith coordinates that vanish for every band structure, "
        f"{[(i, int(v2)) for i, v2 in off]} as (index, value), are nonzero. Check "
        "that a gap separates the bands counted from the others at every k-point and "
        "that the labels came from this group.",
        value=float(max(abs(int(v2)) for _i, v2 in off)), threshold=0.0,
        context={"violations": tuple((int(i), int(v2)) for i, v2 in off),
                 "y_prime": tuple(int(x) for x in y_prime),
                 "number_str": str(number_str)})


def _structure(divisors) -> str:
    return " x ".join(f"Z{d}" for d in divisors) or "trivial"


def _divisors_of(data, number_str) -> tuple:
    """The elementary divisors above 1 of a group's EBR matrix, as the tables ship them.

    ``data`` and ``number_str`` are as :func:`_smith_system` takes them.
    Returns them as integers in the order of the Smith diagonal, ``()``
    where the indicator group is trivial.  The indicator group is the sum
    of ``Z_d`` over them, which :func:`_structure` names.
    """
    return tuple(int(x) for x in _smith_system(data, number_str)[2] if int(x) > 1)


def _verify_smith(number_str, matrix, u, v, d, y, slots, divisors, values,
                  order) -> dict:
    """The Smith form of the EBR matrix computed again, checked against the tables'.

    ``matrix`` is the EBR matrix ``M``, ``u`` and ``v`` its factors and
    ``d`` the diagonal of the tables' Smith form ``R``, as
    :func:`_smith_system` returns them.  ``number_str`` is the group's
    number, for the messages.  ``y`` is the symmetry data vector on every
    row.  ``slots`` are the indices ``i`` with ``d[i] > 1``, ``divisors``
    those ``d[i]``, ``values`` the class, ``(u y)_i mod d[i]`` at each
    slot, and ``order`` the order of the class.  Raises
    :class:`~.refusal.InternalError` where ``u M v`` is not the diagonal
    matrix ``d`` gives, where :func:`_snf` of ``M`` gives other elementary
    divisors, where the class of ``y`` has another order in the form
    computed again, or where ``u`` is not unimodular.  The components of the
    class may differ between the two bases, and are not compared.  Returns
    the fields of :class:`..ebr.SmithVerification` as a mapping, with
    ``"performed"`` added.  Its ``change_of_basis`` is ``u0 u^(-1)``, for
    ``u0`` the left factor computed again, its rows on the slots of that
    form and its columns on ``slots``, each entry reduced modulo the divisor
    of its row, the divisor of the cyclic factor it maps into.
    """
    where = (f"#{number_str}")
    if _matmul(_matmul(u, matrix), v) != [[int(x) for x in row] for row in
                                          _diagonal_like(d, matrix)]:
        raise InternalError(
            f"{where}: the Smith normal form R the irreptables package ships and its "
            "factors u and v do not satisfy u M v = R for the EBR matrix M. Report it, "
            "with the model and the call.")

    u0, d0, _v0 = _snf(matrix)
    divisors0 = tuple(int(d0[i][i]) for i in range(min(len(d0), len(d0[0])))
                      if int(d0[i][i]) > 1)
    if sorted(divisors0) != sorted(divisors):
        raise InternalError(
            f"{where}: the Smith normal form computed again from the EBR matrix has "
            f"elementary divisors above 1 {sorted(divisors0)}, and the form the "
            f"irreptables package ships has {sorted(divisors)}. Report it, with the "
            "model and the call.")

    slots0 = [i for i in range(min(len(d0), len(d0[0]))) if int(d0[i][i]) > 1]
    y0 = [sum(u0[i][j] * y[j] for j in range(len(y))) for i in range(len(u0))]
    values0 = tuple(int(y0[i]) % int(d0[i][i]) for i in slots0)
    order0 = 1
    for val, div in zip(values0, (int(d0[i][i]) for i in slots0)):
        step = div // np.gcd(int(val), int(div)) if val else 1
        order0 = int(np.lcm(order0, int(step)))
    if order0 != order or bool(any(values0)) != bool(any(values)):
        raise InternalError(
            f"{where}: the class of the symmetry data vector has order {order0} in the "
            f"Smith normal form computed again from the EBR matrix, and order {order} in "
            "the form the irreptables package ships. Report it, with the model and the "
            "call.")

    # an entry maps Z_(d[b]) into Z_(d0[a]), so it is fixed modulo d0[a], its row's
    change = _matmul(u0, _unimodular_inverse(u))
    restricted = tuple(tuple(int(change[a][b]) % int(d0[a][a]) for b in slots)
                       for a in slots0)
    identity = all(restricted[i][j] == (1 % int(d0[a][a]) if i == j else 0)
                   for i, a in enumerate(slots0) for j in range(len(slots)))
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
    """The diagonal matrix ``R`` rebuilt from its diagonal, in the shape of ``matrix``.

    ``d`` is the diagonal as :func:`_smith_system` returns it, and
    ``matrix`` a list of rows.  Returns ``R`` as a list of rows of integers.
    """
    n_row, n_col = len(matrix), len(matrix[0])
    return [[int(d[i]) if i == j and i < len(d) else 0 for j in range(n_col)]
            for i in range(n_row)]
