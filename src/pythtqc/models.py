"""Tight-binding crystals to measure on, each chosen to make one question sharp.

Every builder returns a new PythTB 2.0 model. Its docstring says what the
crystal is, where it comes from, and what pythTQC measures on it. A group named
is the one :func:`.groups.realized_group` returns.

:func:`i41_atomic`
    A spinful body-centered tetragonal crystal of 12 bands in I4_11' (#80.30),
    with ``s`` orbitals at 4a below a gap and at the general position 8b above.
    At ``P``, which is not time-reversal invariant, the coreps of the little
    group are ``-P3`` and ``-P4P4``. The table the irreptables package ships for
    ``P`` is that of ``-P``: its ``-P4`` and ``-P3P3`` carry the complex
    conjugate characters of ``-P3`` and ``-P4P4``.
:func:`i41_soc`
    :func:`i41_atomic` with spin-orbit coupling, in I4_11'. With
    ``soc="generic"``, the four lowest bands at ``P`` split into the levels
    ``-P3``, ``-P4P4`` and ``-P3``. With ``soc="axial"``, ``s_z`` is conserved.
:func:`p62_atomic`
    A spinful hexagonal crystal of 18 bands in P6_21' (#171.122), with ``s``
    orbitals at 3a below a gap and at the general position 6c above. At ``H``
    the coreps returned are those of the irreptables package's table for ``H``.
:func:`i213_flicker2018`
    A spinless body-centered cubic crystal of 4 bands in I2_131' (#199.13),
    from Flicker et al., Phys. Rev. B 98, 155145 (2018), Appendix F.1. It has
    spin-1 fermions at ``GM`` and ``H``. Its four bands carry the elementary
    band representation of ``s`` orbitals at 8a. No gap separates them.
:func:`p4232_cano2018`
    A spinless simple cubic crystal of 4 bands in P4_2321' (#208.45), from Cano
    et al., Phys. Rev. Lett. 120, 266401 (2018). Its four bands carry one
    elementary band representation. A gap splits them into two halves. The
    symmetry data vector of neither half is a sum of elementary band
    representations.
:func:`cubic_3c_axis_p`
    A spinless simple cubic crystal of 3 bands in Pm-3m1' (#221.93), with one
    ``p`` orbital at each 3c site along the site's fourfold axis. Every site
    carries ``A2u``.
:func:`cubic_3c_perp_doublet`
    A spinless simple cubic crystal of 6 bands in P-43m1' (#215.71), with the
    ``p`` pair perpendicular to each 3c site's axis. Every site carries ``E``
    of ``-42m1'``, which induces a composite band representation.
:func:`lin3g`
    The spinful square layer of Lin et al., Nat. Commun. 15, 550 (2024), of 8
    bands in P4mm1' (#99.164). Its four lowest bands carry the elementary band
    representation of ``s`` orbitals at 2c. The two above them are a fragile
    pair.

:func:`orbit_of`, :func:`closed_bonds`, :func:`atomic_model`,
:func:`spin_bonds`, :func:`spinor_of`, :func:`cartesian_rotation` and
:func:`primitive_ops` are the constructions the builders use. Together they
build a crystal symmetric under a group by construction.
"""

from __future__ import annotations

import itertools

import numpy as np
from pythtb import Lattice, TBModel

__all__ = [
    # the shared constructions: a crystal symmetric by construction
    "orbit_of", "closed_bonds", "atomic_model", "primitive_ops",
    "spin_bonds", "spinor_of", "cartesian_rotation", "PAULI",
    # the eight builders the module docstring lists
    "i41_atomic", "i41_soc", "p62_atomic",
    "i213_flicker2018", "p4232_cano2018",
    "cubic_3c_axis_p", "cubic_3c_perp_doublet",
    "lin3g",
]


# ---------------------------------------------------------------------------
# 1.  The shared constructions
# ---------------------------------------------------------------------------
#
# orbit_of, closed_bonds and atomic_model build a model of s orbitals on
# Wyckoff orbits, spinless or with both spin states, by closing a few seed
# hoppings under a group, so that the model obeys the group exactly by
# construction. spin_bonds does the same for hoppings that depend on spin, and
# primitive_ops carries {R|t} from a centered conventional cell to the
# primitive cell a model is built on.

def orbit_of(ops, seed, tol=1e-9):
    """The orbit of a point under a set of operations, in reduced coordinates.

    The orbit is the point's distinct images modulo the lattice, each brought
    into ``[0, 1)``. They are listed in the order of the operations that first
    reach them, which fixes the orbital order of the models here.

    Parameters
    ----------
    ops : sequence of (R, t)
        The operations. Pass each as a pair: ``R`` the rotation on reduced
        coordinates, a (3, 3) array or its nine entries row by row, and ``t``
        the translation in reduced coordinates.
    seed : (3,) array_like
        The point, in reduced coordinates.
    tol : float, default 1e-9
        The distance in reduced coordinates within which two images are one
        point.

    Returns
    -------
    (n, 3) numpy.ndarray
        The orbit, a point to a row.
    """
    out = []
    for R, t in ops:
        q = (np.asarray(R, dtype=float).reshape(3, 3) @ np.asarray(seed, dtype=float)
             + np.asarray(t, dtype=float)) % 1.0
        if not any(np.abs(((q - p) + 0.5) % 1.0 - 0.5).max() < tol for p in out):
            out.append(q)
    return np.array(out)


def closed_bonds(sites, ops, seeds, tol=1e-6):
    """Hoppings symmetric under a group: every image of a few seed hoppings.

    An operation ``{R|t}`` carries site ``i``, at ``q_i``, to site ``p(i)`` in
    the cell ``s_i``, where ``R q_i + t = q_p(i) + s_i``. It carries the bond
    ``(i, j, L)`` to ``(p(i), p(j), s_j + R L - s_i)``, with the seed's own
    amplitude. A bond and its reverse are one hopping. A bond reached more than
    once keeps the amplitude it is first given. :func:`spin_bonds` builds
    hoppings that depend on spin.

    Parameters
    ----------
    sites : (nsite, 3) array_like
        Every site, in reduced coordinates and in orbital order.
    ops : sequence of (R, t)
        The operations, as :func:`orbit_of` takes them.
    seeds : sequence of (int, int, tuple, float)
        The seed hoppings. Pass each as ``(i, j, L, amp)`` for the term
        ``amp c_i(0)† c_j(L)``, ``L`` a lattice vector in integer reduced
        components.
    tol : float, default 1e-6
        The distance in reduced coordinates within which an image of a site is
        that site.

    Returns
    -------
    dict
        ``{(i, j, L): amp}`` over every image, ``L`` a tuple of integers. Of a
        bond and its reverse only one stands, since pythtb's ``set_hop`` adds
        the reverse.

    Raises
    ------
    AssertionError
        An operation of ``ops`` carries a site to no site within ``tol``.
    """
    perm, shift = [], []
    for R, t in ops:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float)
        p, s = [], []
        for q in sites:
            img = R @ q + t
            for j, q2 in enumerate(sites):
                d = img - q2
                if np.abs(d - np.rint(d)).max() < tol:
                    p.append(j)
                    s.append(np.rint(d).astype(int))
                    break
            else:                                        # pragma: no cover
                raise AssertionError("closed_bonds(): an operation of ops= carries a "
                                     "site onto none of the sites. Pass every site of "
                                     "each Wyckoff orbit, at the coordinates orbit_of "
                                     "returns.")
        perm.append(p)
        shift.append(s)
    out = {}
    for (i, j, L, amp) in seeds:
        for (R, _), p, s in zip(ops, perm, shift):
            R = np.asarray(R, dtype=float).reshape(3, 3)
            i2, j2 = p[i], p[j]
            L2 = s[j] + np.rint(R @ np.asarray(L, dtype=float)).astype(int) - s[i]
            if (i2, j2, tuple(L2)) in out or (j2, i2, tuple(-L2)) in out:
                continue
            out[(i2, j2, tuple(L2))] = amp
    return out


def atomic_model(lattice, ops, orbits, onsites, seeds, spinful=False):
    """A crystal of one ``s`` orbital per site, symmetric under a group.

    The sites are those of the Wyckoff orbits given. Each orbital has its own
    on-site energy. The hoppings are the seed hoppings closed under the
    operations by :func:`closed_bonds`. An operation carries each hopping onto
    its image with the same amplitude, as it does for ``s`` orbitals.

    Parameters
    ----------
    lattice : (3, 3) array_like
        The lattice vectors as rows, in Cartesian components.
    ops : sequence of (R, t)
        The operations, as :func:`orbit_of` takes them.
    orbits : sequence of (n, 3) array_like
        The Wyckoff orbits, in reduced coordinates. Their sites, stacked in the
        order given, are the orbitals in order.
    onsites : sequence of float
        The on-site energy of each orbital, in that order.
    seeds : sequence of (int, int, tuple, float)
        The seed hoppings, as :func:`closed_bonds` takes them, indexed into the
        stacked sites.
    spinful : bool, default False
        Whether each orbital carries two spin states.

    Returns
    -------
    pythtb.TBModel
        A three-dimensional model of one orbital per site, symmetric under
        ``ops``. With ``spinful=True``, both spin states have the same
        hoppings. The band representation is then the spinless one times spin
        1/2, and every level is at least twofold degenerate.
    """
    sites = np.vstack(list(orbits))
    lat = Lattice(np.asarray(lattice, dtype=float).tolist(), sites.tolist(),
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=spinful)
    m.set_onsite(list(onsites))
    for (i, j, L), amp in closed_bonds(sites, ops, seeds).items():
        m.set_hop(float(amp), i, j, list(L), mode="add")
    return m


#: The Pauli matrices ``(sigma_x, sigma_y, sigma_z)``, in which the spin-orbit
#: seeds are written. A (2, 2) hopping matrix of a spinful pythtb model acts on
#: the spin of one pair of orbitals, so a seed written in them is passed as it
#: stands.
PAULI = (np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]], dtype=complex),
         np.array([[1, 0], [0, -1]], dtype=complex))


def cartesian_rotation(R_red, lattice):
    """A rotation on reduced coordinates, written in Cartesian coordinates.

    The matrix returned is ``lattice.T @ R_red @ inv(lattice.T)``.

    Parameters
    ----------
    R_red : (3, 3) array_like
        The rotation on reduced coordinates, or its nine entries row by row.
    lattice : (3, 3) array_like
        The lattice vectors as rows, in Cartesian components.

    Returns
    -------
    (3, 3) numpy.ndarray
        The rotation in Cartesian components.
    """
    lat = np.asarray(lattice, dtype=float)
    R = np.asarray(R_red, dtype=float).reshape(3, 3)
    return lat.T @ R @ np.linalg.inv(lat.T)


def spinor_of(R_cart, tol=1e-9):
    """The SU(2) matrix of a proper rotation given in Cartesian coordinates.

    For a rotation by the angle θ in ``[0, π]`` about the unit axis ``n``, the
    matrix returned is ``S = exp(-i θ n·σ/2)``. At ``θ = π``, ``cos(θ/2) = 0``,
    and the sign of ``S`` is that of the unit vector along the axis taken for
    ``n``, which is either one. A spin block carried as ``S T S†`` is the same
    for either sign.

    Parameters
    ----------
    R_cart : (3, 3) array_like
        A proper rotation, in Cartesian components.
    tol : float, default 1e-9
        The tolerance on the determinant being 1, and on the angle being 0 or π.

    Returns
    -------
    (2, 2) numpy.ndarray of complex
        The matrix ``S``.

    Raises
    ------
    AssertionError
        The determinant of ``R_cart`` differs from 1 by more than ``tol``.
    """
    Rc = np.asarray(R_cart, dtype=float)
    if abs(np.linalg.det(Rc) - 1.0) > tol:             # pragma: no cover - guard
        raise AssertionError("spinor_of(): R_cart= has a determinant other than 1. Pass a "
                             "proper rotation; for an improper rotation R, pass -R.")
    theta = np.arccos(np.clip((np.trace(Rc) - 1.0) / 2.0, -1.0, 1.0))
    if abs(theta) < tol:
        return np.eye(2, dtype=complex)
    if abs(theta - np.pi) < tol:
        M = Rc + np.eye(3)                             # axis = any nonzero column
        n = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
    else:
        n = np.array([Rc[2, 1] - Rc[1, 2], Rc[0, 2] - Rc[2, 0], Rc[1, 0] - Rc[0, 1]])
    n = n / np.linalg.norm(n)
    ns = n[0] * PAULI[0] + n[1] * PAULI[1] + n[2] * PAULI[2]
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * ns


def spin_bonds(sites, ops, lattice, seeds, tol=1e-6):
    """The group average of a few seed hoppings that depend on spin.

    The hoppings returned are symmetric under the group. Each operation carries
    a seed's bond as :func:`closed_bonds` does, and its spin block ``T`` to
    ``S T S†``. Here ``S`` is the matrix :func:`spinor_of` returns for the
    operation's rotation in Cartesian coordinates. An improper rotation ``R``
    is taken as its proper part ``-R``, since spin is even under inversion.
    The images landing on one bond are added, an image landing on the reversed
    bond as its adjoint. The sum is divided by the number of operations.

    Parameters
    ----------
    sites : (nsite, 3) array_like
        Every site, in reduced coordinates and in orbital order.
    ops : sequence of (R, t)
        The operations, as :func:`orbit_of` takes them.
    lattice : (3, 3) array_like
        The lattice vectors as rows, in Cartesian components.
    seeds : sequence of (int, int, tuple, ndarray)
        The seed hoppings. Pass each as ``(i, j, L, T)`` for the term
        ``c_i(0)† T c_j(L)``, ``T`` a (2, 2) matrix on spin and ``L`` a lattice
        vector in integer reduced components.
    tol : float, default 1e-6
        The distance in reduced coordinates within which an image of a site is
        that site.

    Returns
    -------
    dict
        ``{(i, j, L): T}`` over every image, each ``T`` a (2, 2) complex matrix.
        Of a bond and its reverse only one stands.

    Raises
    ------
    AssertionError
        A seed is an on-site term, which pythtb's ``set_onsite`` takes instead,
        or an operation of ``ops`` carries a site to no site within ``tol``.
    """
    sites = np.asarray(sites, dtype=float)
    perm, shift, spins = [], [], []
    for R, t in ops:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float)
        p, s = [], []
        for q in sites:
            img = R @ q + t
            for j, q2 in enumerate(sites):
                d = img - q2
                if np.abs(d - np.rint(d)).max() < tol:
                    p.append(j)
                    s.append(np.rint(d).astype(int))
                    break
            else:                                        # pragma: no cover
                raise AssertionError("spin_bonds(): an operation of ops= carries a site of "
                                     "sites= onto none of the sites. Pass every site of each "
                                     "Wyckoff orbit, at the coordinates orbit_of returns.")
        perm.append(p)
        shift.append(s)
        # an improper operation acts on the spin as its proper part, since spin
        # is even under inversion: S(I·R) = S(R), as in the tables of the
        # irreptables package, so that a spinful s orbital at an inversion
        # center is even
        Rc = cartesian_rotation(R, lattice)
        spins.append(spinor_of(np.sign(float(np.linalg.det(Rc))) * Rc))
    out = {}
    for (i, j, L, T) in seeds:
        T = np.asarray(T, dtype=complex)
        L = np.asarray(L, dtype=int)
        for (R, _), p, s, S in zip(ops, perm, shift, spins):
            R = np.asarray(R, dtype=float).reshape(3, 3)
            i2, j2 = p[i], p[j]
            L2 = tuple(s[j] + np.rint(R @ L).astype(int) - s[i])
            T2 = S @ T @ S.conj().T
            key, rkey = (i2, j2, L2), (j2, i2, tuple(-x for x in L2))
            if key == rkey:                              # pragma: no cover - guard
                raise AssertionError("spin_bonds(): a seed of seeds= is an on-site term. Set "
                                     "it with pythtb's set_onsite instead.")
            if rkey in out:
                out[rkey] = out[rkey] + T2.conj().T
            elif key in out:
                out[key] = out[key] + T2
            else:
                out[key] = T2
    return {k: v / len(ops) for k, v in out.items()}


def primitive_ops(ops, centering_rows):
    """The operations of a centered conventional cell, in its primitive cell.

    With ``M`` the matrix ``centering_rows``, the point at reduced coordinates
    ``u`` of the primitive cell is at ``M.T @ u`` in the conventional cell. The
    operation ``x -> R @ x + t`` there is
    ``u -> inv(M.T) @ R @ M.T @ u + inv(M.T) @ t`` in the primitive cell. A
    model is built on the primitive cell, since
    :func:`.groups.realized_group` refuses a Hamiltonian that obeys a pure
    translation of its cell (``supercell``).

    Parameters
    ----------
    ops : sequence of (R, t)
        The operations in the conventional cell, as :func:`orbit_of` takes them.
    centering_rows : (3, 3) array_like
        The primitive lattice vectors as rows, in the conventional cell's
        reduced coordinates. For a body-centered cell, they are
        ``(-1/2, 1/2, 1/2)``, ``(1/2, -1/2, 1/2)`` and ``(1/2, 1/2, -1/2)``.

    Returns
    -------
    tuple of (tuple of int, tuple of float)
        The operations in the primitive cell, each a pair: the rotation as its
        nine integer entries row by row, and the translation in primitive
        reduced coordinates.

    Raises
    ------
    AssertionError
        A rotation is not an integer matrix in the primitive cell.
    """
    mt = np.asarray(centering_rows, dtype=float).T
    inv = np.linalg.inv(mt)
    out = []
    for R, t in ops:
        rp = inv @ np.asarray(R, dtype=float).reshape(3, 3) @ mt
        if np.abs(rp - np.rint(rp)).max() > 1e-9:      # pragma: no cover - guard
            raise AssertionError("primitive_ops(): a rotation of ops= is not an integer "
                                 "matrix in the basis of the rows of centering_rows=. Pass "
                                 "as centering_rows= the primitive lattice vectors of the "
                                 "centered cell of ops=, in its reduced coordinates; for a "
                                 "body-centered cell, the rows (-1/2, 1/2, 1/2), "
                                 "(1/2, -1/2, 1/2) and (1/2, 1/2, -1/2).")
        out.append((tuple(np.rint(rp).astype(int).ravel()),
                    tuple(inv @ np.asarray(t, dtype=float))))
    return tuple(out)


# ---------------------------------------------------------------------------
# 2.  Two spinful atomic insulators, in I4_1 (#80) and P6_2 (#171)
# ---------------------------------------------------------------------------
#
# Both are crystals of s orbitals whose hoppings do not depend on spin, so
# their band representation is the spinless one times spin 1/2. Each has a
# point that is not time-reversal invariant, where time reversal joins irreps
# of the little group into coreps: P of the body-centered I4_1, joined by C4
# times T, and H of the hexagonal P6_2, joined by a twofold rotation times T.
# At H the coreps returned are those of the irreptables package's table. At P
# they are -P3 and -P4P4, whose characters are the complex conjugates of the
# table's -P4 and -P3P3. i41_soc shows the same on a spectrum with spin-orbit
# coupling.

#: ``{R|t}`` of I4_1 (#80) in the conventional body-centered cell of the
#: International Tables, in the order of the irreptables package.
I41_OPS = (
    ((1, 0, 0, 0, 1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),          # 1  E
    ((-1, 0, 0, 0, -1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),        # 2  2 0,0,z
    ((0, -1, 0, 1, 0, 0, 0, 0, 1), (0.0, 0.5, 0.25)),        # 3  4+(0,0,1/4)
    ((0, 1, 0, -1, 0, 0, 0, 0, 1), (0.0, 0.5, 0.25)),        # 4  4-(0,0,1/4)
)

#: The primitive vectors of the body-centered cell as rows, in reduced
#: coordinates of the conventional cell: the standard ``I`` setting.
I41_CENTERING = np.array([[-0.5, 0.5, 0.5],
                          [0.5, -0.5, 0.5],
                          [0.5, 0.5, -0.5]])

#: ``{R|t}`` of I4_1 on the primitive cell, which is what pythtb is handed.
I41_PRIM_OPS = primitive_ops(I41_OPS, I41_CENTERING)

#: Body-centered tetragonal, ``c/a = 1.6``: the primitive vectors as rows, in
#: Cartesian components.
I41_LATTICE = I41_CENTERING @ np.diag([1.0, 1.0, 1.6])

#: ``{R|t}`` of P6_2 (#171), primitive hexagonal, in the order of the
#: irreptables package.
P62_OPS = (
    ((1, 0, 0, 0, 1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),          # 1  E
    ((0, -1, 0, 1, -1, 0, 0, 0, 1), (0.0, 0.0, 2 / 3)),      # 2  3+(0,0,2/3)
    ((-1, 1, 0, -1, 0, 0, 0, 0, 1), (0.0, 0.0, 1 / 3)),      # 3  3-(0,0,1/3)
    ((-1, 0, 0, 0, -1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),        # 4  2 0,0,z
    ((0, 1, 0, -1, 1, 0, 0, 0, 1), (0.0, 0.0, 2 / 3)),       # 5  6-(0,0,2/3)
    ((1, -1, 0, 1, 0, 0, 0, 0, 1), (0.0, 0.0, 1 / 3)),       # 6  6+(0,0,1/3)
)

#: Primitive hexagonal (gamma = 120 deg), ``c/a = 1.3``.
P62_LATTICE = np.array([[1.0, 0.0, 0.0],
                        [-0.5, np.sqrt(3) / 2, 0.0],
                        [0.0, 0.0, 1.3]])

#: The default parameters of :func:`i41_atomic`, with generic values for the 4a
#: orbit ``(0, 0, z)`` and the general orbit 8b of I4_1. The 4a orbit alone is
#: a point set of group I4_1/amd (#141); the general orbit lowers it to #80.
I41_PARAMS = dict(z_a=0.13, seed_b=(0.21, 0.34, 0.07), delta=2.5,
                  t_a=0.09, t_b=0.05, t_ab=0.03)

#: The default parameters of :func:`p62_atomic`, chosen likewise. The 3a orbit
#: alone is a point set of group P6/mmm (#191); the general orbit 6c lowers it
#: to #171.
P62_PARAMS = dict(z_a=0.11, seed_c=(0.21, 0.34, 0.07), delta=2.5,
                  t_a=0.09, t_b=0.05, t_ab=0.03)


def i41_sites(z_a=None, seed_b=None):
    """The 4a and 8b orbits of I4_1, in reduced coordinates of the primitive cell.

    ``z_a`` is the free parameter of the 4a orbit ``(0, 0, z)``, and ``seed_b``
    a point of the 8b orbit, both in reduced coordinates of the conventional
    cell; ``None`` takes the value in :data:`I41_PARAMS`. Returns the pair
    ``(orbit_4a, orbit_8b)``, arrays of 2 and 4 points, a point to a row, each
    in the order :func:`orbit_of` lists them.
    """
    z_a = I41_PARAMS["z_a"] if z_a is None else float(z_a)
    seed_b = I41_PARAMS["seed_b"] if seed_b is None else tuple(seed_b)
    to_prim = np.linalg.inv(I41_CENTERING.T)
    return (orbit_of(I41_PRIM_OPS, to_prim @ np.array([0.0, 0.0, z_a])),
            orbit_of(I41_PRIM_OPS, to_prim @ np.asarray(seed_b, dtype=float)))


def i41_atomic(z_a=None, seed_b=None, delta=None, t_a=None, t_b=None, t_ab=None):
    """A spinful atomic insulator in I4_11' (#80.30), ``s`` orbitals at 4a and 8b.

    Six ``s`` orbitals stand in the primitive body-centered tetragonal cell, two
    on the 4a orbit ``(0, 0, z)`` and four on a generic 8b orbit, each with two
    spin states: 12 bands. The four lowest bands, below a gap, carry
    ``-1E2E at 4a (21')``, the only double-valued elementary band
    representation of the group. At ``P``, which is not time-reversal
    invariant, they carry ``-P3`` twice and ``-P4P4`` once. These are the two
    coreps of the little group at ``P``, as :func:`pythtqc.corep_table` returns
    them: ``-P3``, with the characters (1, -i) on the identity and ``C2(z)``,
    and ``-P4P4``, with (2, 2i). The table the irreptables package ships for
    ``P``, that of the Bilbao Crystallographic Server, is the table of ``-P``:
    its ``-P4`` and ``-P3P3`` carry the complex conjugate characters, (1, i)
    and (2, -2i).

    Parameters
    ----------
    z_a : float, default 0.13
        The free parameter of the 4a orbit ``(0, 0, z)``, in conventional
        reduced coordinates.
    seed_b : (3,) array_like, default (0.21, 0.34, 0.07)
        A point of the 8b orbit, in conventional reduced coordinates.
    delta : float, default 2.5
        Half the difference of the on-site energies, ``-delta`` at 4a and
        ``+delta`` at 8b.
    t_a : float, default 0.09
        The hopping between the two 4a sites of the primitive cell, and on the
        images of that bond.
    t_b : float, default 0.05
        The hopping between the first two 8b sites of the cell, in the order
        :func:`orbit_of` lists them, and on the images of that bond.
    t_ab : float, default 0.03
        The hopping between the first 4a site and the first 8b site of the
        cell, in that order, and on the images of that bond.

    Returns
    -------
    pythtb.TBModel
        A spinful three-dimensional model of 6 orbitals and 12 bands.
    """
    p = dict(I41_PARAMS)
    for key, val in dict(z_a=z_a, seed_b=seed_b, delta=delta, t_a=t_a, t_b=t_b,
                         t_ab=t_ab).items():
        if val is not None:
            p[key] = val if key == "seed_b" else float(val)
    orbits = i41_sites(p["z_a"], p["seed_b"])
    n_a = len(orbits[0])
    return atomic_model(
        I41_LATTICE, I41_PRIM_OPS, orbits,
        [-p["delta"]] * n_a + [p["delta"]] * len(orbits[1]),
        [(0, 1, (0, 0, 0), p["t_a"]),
         (n_a, n_a + 1, (0, 0, 0), p["t_b"]),
         (0, n_a, (0, 0, 0), p["t_ab"])],
        spinful=True)


#: The two sets of spin-orbit seeds :func:`i41_soc` names, each a pair
#: ``(T_a, T_ab)`` for the ``t_a`` and ``t_ab`` bonds of :func:`i41_atomic`.
#: ``"generic"`` seeds ``i b·σ`` with a generic ``b`` and a second block of
#: another orientation, so that only the multiples of the identity commute
#: with ``H(k)`` at every ``k``. ``"axial"`` seeds ``i σ_z`` on both bonds,
#: which conserves ``s_z``, so that the matrices commuting with ``H(k)`` at
#: every ``k`` form a space of dimension two. Both obey I4_11' exactly.
_I41_SOC_B = (0.31, -0.52, 0.44)
I41_SOC_SEEDS = {
    "generic": (1j * (_I41_SOC_B[0] * PAULI[0] + _I41_SOC_B[1] * PAULI[1]
                      + _I41_SOC_B[2] * PAULI[2]),
                0.7 * 1j * (_I41_SOC_B[0] * PAULI[0] + _I41_SOC_B[1] * PAULI[1]
                            + _I41_SOC_B[2] * PAULI[2]) + 0.2j * PAULI[0]),
    "axial": (1j * PAULI[2], 1j * PAULI[2]),
}


def i41_soc(soc="generic", lam=0.4, **kwargs):
    """:func:`i41_atomic` with spin-orbit coupling, in I4_11' (#80.30).

    The hoppings of :func:`i41_atomic` are kept. Spin-orbit hoppings are added
    on its ``t_a`` and ``t_ab`` bonds, from the seeds ``lam * T_a`` and
    ``lam * T_ab`` averaged over the operations of I4_1 by :func:`spin_bonds`.
    The names ``soc`` takes give these seeds:

    * ``"generic"``: ``T_a = i b·σ`` with ``b = (0.31, -0.52, 0.44)``, and
      ``T_ab = i (0.7 b·σ + 0.2 σ_x)``. No spin component is conserved. At the
      defaults, the four lowest bands at ``P`` split into three levels,
      ``-P3``, ``-P4P4`` and ``-P3`` from the lowest.
    * ``"axial"``: ``T_a = T_ab = i σ_z``, which every operation of I4_1 leaves
      unchanged, so ``s_z`` is conserved.

    Parameters
    ----------
    soc : {"generic", "axial"} or sequence, default "generic"
        The spin-orbit seeds. Pass a name above, or a pair ``(T_a, T_ab)`` of
        (2, 2) spin matrices without the factor ``lam``.
    lam : float, default 0.4
        The spin-orbit strength.
    **kwargs
        The parameters of :func:`i41_atomic`, by name: ``z_a``, ``seed_b``,
        ``delta``, ``t_a``, ``t_b`` and ``t_ab``.

    Returns
    -------
    pythtb.TBModel
        A spinful three-dimensional model of 6 orbitals and 12 bands.

    Raises
    ------
    KeyError
        ``soc`` is a name other than ``"generic"`` or ``"axial"``.
    TypeError
        A keyword is not one :func:`i41_atomic` takes.
    ValueError
        ``soc`` is a sequence of other than two matrices.
    """
    p = dict(I41_PARAMS)
    for key, val in kwargs.items():
        if key not in p:                                 # pragma: no cover - guard
            raise TypeError(f"i41_soc(): {key}= is not a parameter of i41_soc. Pass soc=, "
                            "lam=, or any of z_a=, seed_b=, delta=, t_a=, t_b= and t_ab=.")
        if val is not None:
            p[key] = val if key == "seed_b" else float(val)
    seeds = I41_SOC_SEEDS[soc] if isinstance(soc, str) else tuple(soc)
    if len(seeds) != 2:                                  # pragma: no cover - guard
        raise ValueError('i41_soc(): soc= is a sequence whose length is not two. Pass '
                         '"generic", "axial", or a pair (T_a, T_ab) of (2, 2) spin matrices.')

    orbits = i41_sites(p["z_a"], p["seed_b"])
    sites = np.vstack(list(orbits))
    n_a = len(orbits[0])
    scalar = closed_bonds(sites, I41_PRIM_OPS,
                          [(0, 1, (0, 0, 0), p["t_a"]),
                           (n_a, n_a + 1, (0, 0, 0), p["t_b"]),
                           (0, n_a, (0, 0, 0), p["t_ab"])])
    spin = spin_bonds(sites, I41_PRIM_OPS, I41_LATTICE,
                      [(0, 1, (0, 0, 0), float(lam) * np.asarray(seeds[0], dtype=complex)),
                       (0, n_a, (0, 0, 0), float(lam) * np.asarray(seeds[1], dtype=complex))])

    bonds = {key: float(amp) * np.eye(2, dtype=complex) for key, amp in scalar.items()}
    for (i, j, L), T in spin.items():
        rkey = (j, i, tuple(-x for x in L))
        if rkey in bonds:
            bonds[rkey] = bonds[rkey] + T.conj().T
        else:
            bonds[(i, j, L)] = bonds.get((i, j, L), 0.0 * T) + T

    lat = Lattice(np.asarray(I41_LATTICE, dtype=float).tolist(), sites.tolist(),
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=True)
    m.set_onsite([-p["delta"]] * n_a + [p["delta"]] * len(orbits[1]))
    for (i, j, L), T in bonds.items():
        m.set_hop(T, i, j, list(L), mode="add")
    return m


def p62_sites(z_a=None, seed_c=None):
    """The 3a and 6c orbits of P6_2, in reduced coordinates.

    ``z_a`` is the free parameter of the 3a orbit ``(0, 0, z)``, and ``seed_c``
    a point of the 6c orbit, both in reduced coordinates; ``None`` takes the
    value in :data:`P62_PARAMS`. Returns the pair ``(orbit_3a, orbit_6c)``,
    arrays of 3 and 6 points, a point to a row, each in the order
    :func:`orbit_of` lists them.
    """
    z_a = P62_PARAMS["z_a"] if z_a is None else float(z_a)
    seed_c = P62_PARAMS["seed_c"] if seed_c is None else tuple(seed_c)
    return (orbit_of(P62_OPS, (0.0, 0.0, z_a)),
            orbit_of(P62_OPS, seed_c))


def p62_atomic(z_a=None, seed_c=None, delta=None, t_a=None, t_b=None, t_ab=None):
    """A spinful atomic insulator in P6_21' (#171.122), ``s`` orbitals at 3a and 6c.

    Nine ``s`` orbitals stand in the primitive hexagonal cell, three on the 3a
    orbit ``(0, 0, z)`` and six on a generic 6c orbit, each with two spin
    states: 18 bands. The six lowest bands, below a gap, carry the elementary
    band representation ``-1E2E at 3a (21')``. At ``H = (1/3, 1/3, 1/2)``,
    which is not time-reversal invariant, they carry ``-H4`` twice and
    ``-H5H6`` twice. These are the two coreps of the little group at ``H``, as
    :func:`pythtqc.corep_table` returns them: ``-H4``, with the characters
    (1, -1, 1) on the identity, ``C3(z)`` and ``C3^-1(z)``, and ``-H5H6``, with
    (2, 1, -1). They are the coreps of the irreptables package's table for
    ``H``, character for character. ``-H5H6`` pairs two different irreps under
    time reversal, where ``-P4P4`` of :func:`i41_atomic` pairs two copies of
    one.

    Parameters
    ----------
    z_a : float, default 0.11
        The free parameter of the 3a orbit ``(0, 0, z)``, in reduced
        coordinates.
    seed_c : (3,) array_like, default (0.21, 0.34, 0.07)
        A point of the 6c orbit, in reduced coordinates.
    delta : float, default 2.5
        Half the difference of the on-site energies, ``-delta`` at 3a and
        ``+delta`` at 6c.
    t_a : float, default 0.09
        The hopping between the first two 3a sites of the cell, in the order
        :func:`orbit_of` lists them, and on the images of that bond.
    t_b : float, default 0.05
        The hopping between the first two 6c sites of the cell, in that order,
        and on the images of that bond.
    t_ab : float, default 0.03
        The hopping between the first 3a site and the first 6c site of the
        cell, in that order, and on the images of that bond.

    Returns
    -------
    pythtb.TBModel
        A spinful three-dimensional model of 9 orbitals and 18 bands.
    """
    p = dict(P62_PARAMS)
    for key, val in dict(z_a=z_a, seed_c=seed_c, delta=delta, t_a=t_a, t_b=t_b,
                         t_ab=t_ab).items():
        if val is not None:
            p[key] = val if key == "seed_c" else float(val)
    orbits = p62_sites(p["z_a"], p["seed_c"])
    n_a = len(orbits[0])
    return atomic_model(
        P62_LATTICE, P62_OPS, orbits,
        [-p["delta"]] * n_a + [p["delta"]] * len(orbits[1]),
        [(0, 1, (0, 0, 0), p["t_a"]),
         (n_a, n_a + 1, (0, 0, 0), p["t_b"]),
         (0, n_a, (0, 0, 0), p["t_ab"])],
        spinful=True)


# ---------------------------------------------------------------------------
# 3.  The connected multifold model of I2_13 (#199)
# ---------------------------------------------------------------------------
#
# The model of Flicker, de Juan, Bradlyn, Morimoto, Vergniory and Grushin,
# "Chiral Optical Response of Multifold Fermions", Phys. Rev. B 98, 155145
# (2018), arXiv:1806.09642, Appendix F.1 ("SGs 199 and 214"), with a = 1:
#
# * Eq. (88), the primitive vectors R_1 = (a/2)(-x + y + z),
#   R_2 = (a/2)(x - y + z) and R_3 = (a/2)(x + y - z).
# * Eq. (89), the generators {C_2x | 1/2 1/2 0} and {C_3,111 | 000}, with the
#   translations in reduced coordinates (Eq. (90)).
# * Eq. (92), the four sites of 8a in the primitive cell, in reduced
#   coordinates: q_1 = (u, u, u), q_2 = (1/2 - u, 1/2, 0),
#   q_3 = (0, 1/2 - u, 1/2) and q_4 = (1/2, 0, 1/2 - u), for -1/2 < u < 1/2.
#   At u = 1/4 the group is #214, so u is generic here.
# * Eq. (93), |q_1 - q_j|^2 = 1/4 - u + 2u^2, the same for every bond the
#   group relates.
# * Eq. (94), the nearest-neighbor Hamiltonian, six bonds and their conjugates.
# * Eqs. (95) to (98), its Fourier transform V†(u, k) H_0(k) V(u, k), which is
#   pythtb's convention I.
#
# The orbitals are spinless s orbitals, on which every operation of the site
# group of 8a acts trivially. The paper's Fig. 4(b) is this model's band
# structure, with spin-1 fermions at GM and H: the threefold GM4 at -t and H4
# at +t.

#: Eq. (88) with ``a = 1``: the primitive body-centered cubic basis.
I213_LATTICE = 0.5 * np.array([[-1.0, 1.0, 1.0],
                                [1.0, -1.0, 1.0],
                                [1.0, 1.0, -1.0]])

#: The free parameter of the 8a orbit. Any ``u`` in ``(-1/2, 1/2)`` but
#: ``1/4`` gives #199; at ``u = 1/4`` the group is #214. At ``u = -1/4`` the
#: sites allow #214 and the hoppings keep #199.
I213_U = 0.12

#: Eq. (94) rewritten so every creation operator sits in cell 0, in pythtb's
#: ``set_hop(t, i, j, R)`` order (``t c^dag_{i,0} c_{j,R}`` + h.c.).
I213_BONDS = (
    (1, 0, (0, 0, 0)),      # c^dag_2 c_1
    (2, 0, (0, 0, 0)),      # c^dag_3 c_1
    (3, 0, (0, 0, 0)),      # c^dag_4 c_1
    (3, 2, (1, 0, 0)),      # c^dag_{4,R-R_1} c_{3,R}
    (3, 1, (0, -1, 0)),     # c^dag_{4,R+R_2} c_{2,R}
    (2, 1, (0, 0, 1)),      # c^dag_{3,R-R_3} c_{2,R}
)


def i213_sites(u=I213_U):
    """The 8a orbit of I2_13, in reduced coordinates of the primitive cell.

    ``u`` is the free parameter of the orbit, whose first site is
    ``(u, u, u)``. The four sites are those of Eq. (92) of Flicker et al.,
    Phys. Rev. B 98, 155145 (2018), in its order. At ``u = 1/4`` the group is
    I4_132 (#214). Returns a (4, 3) array, a site to a row.
    """
    u = float(u)
    return np.array([[u, u, u], [0.5 - u, 0.5, 0.0],
                     [0.0, 0.5 - u, 0.5], [0.5, 0.0, 0.5 - u]])


def i213_bloch_hamiltonian(k_cart, u=I213_U, t=1.0):
    """The Bloch Hamiltonian of :func:`i213_flicker2018`, as the paper writes it.

    The matrix returned is ``H(k) = t V†(u, k) H_0(k) V(u, k)``, Eqs. (96) to
    (98) of Flicker et al., Phys. Rev. B 98, 155145 (2018), with
    ``V = diag(exp(i k·q_j))`` for the sites ``q_j`` of :func:`i213_sites` in
    Cartesian components. It equals the model's ``H(k)`` entry for entry. Only
    ``V`` depends on ``u``, so the spectrum does not. ``k_cart`` is ``k`` in
    Cartesian components, for a lattice constant of 1;
    ``numpy.linalg.solve(I213_LATTICE, 2*pi*k_red)`` gives it from reduced
    coordinates ``k_red``. ``u`` is the free parameter of the 8a orbit, and
    ``t`` the nearest-neighbor hopping. Returns a (4, 4) complex array.
    """
    k = np.asarray(k_cart, dtype=float).reshape(-1)[:3]
    a, b, c = (float(k @ r) for r in I213_LATTICE)
    e = np.exp
    H0 = np.array([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, e(-1j * c), e(1j * b)],
        [1.0, e(1j * c), 0.0, e(-1j * a)],
        [1.0, e(-1j * b), e(1j * a), 0.0]], dtype=complex)
    V = np.diag(np.exp(1j * (i213_sites(u) @ I213_LATTICE @ k)))
    return float(t) * V.conj().T @ H0 @ V


def i213_flicker2018(u=I213_U, t=1.0):
    """The I2_13 model of Flicker et al., Phys. Rev. B 98, 155145 (2018).

    Four spinless ``s`` orbitals stand at Wyckoff position 8a, in the primitive
    cell of the body-centered cubic lattice, ``a = 1``. They are joined by a
    nearest-neighbor hopping ``t``. The model is that of the paper's Appendix
    F.1 (arXiv:1806.09642). Its group is I2_131' (#199.13). For ``t > 0``, the
    three lowest bands at ``GM`` carry the threefold ``GM4``. The three highest
    at ``H`` carry the threefold ``H4``. These are the spin-1 fermions. No gap
    separates any of the four bands from the others.

    Parameters
    ----------
    u : float, default 0.12
        The free parameter of the 8a orbit ``(u, u, u)``, in reduced
        coordinates of the primitive cell.
    t : float, default 1.0
        The nearest-neighbor hopping, which sets the energy scale.

    Returns
    -------
    pythtb.TBModel
        A spinless three-dimensional model of 4 orbitals and 4 bands. With
        ``u=1/4``, the group returned is I4_1321' (#214.68).
    """
    lat = Lattice(I213_LATTICE.tolist(), i213_sites(u).tolist(),
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=False)
    for i, j, R in I213_BONDS:
        m.set_hop(float(t), i, j, list(R), mode="add")
    return m


def i213_conventional(u=I213_U, t=1.0):
    """The crystal of :func:`i213_flicker2018` on the conventional cubic cell.

    ``u`` and ``t`` are the free parameter of the 8a orbit and the
    nearest-neighbor hopping, as :func:`i213_flicker2018` takes them. The cell,
    of lattice constant 1, holds the four sites of the primitive cell and their
    images under the centering translation ``(1/2, 1/2, 1/2)``. The Hamiltonian
    obeys that translation, so :func:`.groups.realized_group` refuses it
    ``supercell``. Returns a spinless three-dimensional model of 8 orbitals and
    8 bands, whose spectrum is that of :func:`i213_flicker2018`, folded.
    """
    q = i213_sites(u)
    q_cart = q @ I213_LATTICE
    pos = np.vstack([q_cart, q_cart + 0.5]) % 1.0        # conventional a = 1 cube
    m = TBModel(Lattice(np.eye(3), pos.tolist(), periodic_dirs=[0, 1, 2]),
                spinful=False)
    for i, j, R in I213_BONDS:
        d = (q[j] + np.asarray(R, dtype=float)) @ I213_LATTICE - q_cart[i]
        for copy in (0, 4):
            target = pos[i + copy] + d
            for jj in range(8):
                shift = target - pos[jj]
                if np.abs(shift - np.rint(shift)).max() < 1e-9:
                    m.set_hop(float(t), i + copy, jj,
                              np.rint(shift).astype(int).tolist(), mode="add")
                    break
            else:                                        # pragma: no cover
                raise AssertionError("bond does not close in the conventional cell")
    return m


# ---------------------------------------------------------------------------
# 4.  The disconnected elementary band representation of P4_232 (#208)
# ---------------------------------------------------------------------------
#: Parameter values of Fig. 3a of arXiv:1711.11045.
P4232_PARAMS = dict(t1=0.2, t2=0.3, t3=0.1, t4=0.08, t5=0.05, t6=0.02)

#: The four nearest-neighbor vectors ``delta_{1,2,3,4}`` of the paper, in
#: Cartesian components for a cubic lattice constant of 1. They are the
#: tetrahedral half of the eight half body diagonals joining the two 2a sites,
#: the four the paper draws in the inset to Fig. 3a. The ``t2`` term, which
#: multiplies the sum of ``sin(k·delta_i)`` over these four, breaks the
#: inversion through each site. With ``t3`` to ``t6`` zero, an inversion
#: exchanging the two sites remains, and the group is Pn-3m1' (#224.111).
P4232_NN_VECTORS = 0.5 * np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
)

#: Orbital positions in reduced coordinates: two 2a sites, two orbitals each.
#: Basis order is ``(site, orbital)`` = (A,dz2), (A,dx2-y2), (B,dz2), (B,dx2-y2),
#: so a paper term ``tau_i (x) sigma_j`` is ``numpy.kron(tau_i, sigma_j)``.
_P4232_ORB = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])


def p4232_bloch_hamiltonian(k_cart, t1=None, t2=None, t3=None, t4=None, t5=None, t6=None):
    r"""The Bloch Hamiltonian of :func:`p4232_cano2018`, as the paper writes it.

    The matrix returned is :math:`H_\mathbf{k} + H^1_\mathbf{k}` of Eqs. (5) to
    (7) of Cano et al., Phys. Rev. Lett. 120, 266401 (2018):

    .. math::
        H_\mathbf{k} = t_1 f_1 \tau_x\sigma_0 + t_2 f_2 \tau_y\sigma_0
                     + t_3 (g_1 \tau_z\sigma_z + g_2 \tau_z\sigma_x)

        H^1_\mathbf{k} = t_4 f_1 \tau_y\sigma_y
                       + t_5 \tau_0 (g_2\sigma_z - g_1\sigma_x)
                       + t_6 g_3 \tau_z\sigma_0

    Here :math:`\tau` are the Pauli matrices on the two sites, and
    :math:`\sigma` those on the orbitals ``d_z2`` and ``d_x2-y2`` of a site.
    The :math:`\delta_i` are the four vectors of :data:`P4232_NN_VECTORS`, and
    :math:`f_1 = \sum_i \cos(\mathbf{k}\cdot\delta_i)`,
    :math:`f_2 = \sum_i \sin(\mathbf{k}\cdot\delta_i)`,
    :math:`g_1 = \cos k_x - \cos k_y`,
    :math:`g_2 = (\cos k_x + \cos k_y - 2\cos k_z)/\sqrt3`, and
    :math:`g_3 = \cos 2k_x \cos k_y - \cos 2k_y \cos k_x + \text{perm.}`
    (cyclic in :math:`k_x \to k_y \to k_z`). The matrix's phases carry the
    orbital positions, as pythtb's do, so it equals the model's ``H(k)`` entry
    for entry. ``k_cart`` is ``k`` in Cartesian components for a lattice
    constant of 1, which is ``2*pi*k_red`` for reduced coordinates ``k_red``.
    ``t1``, ``t2``, ``t3``, ``t4``, ``t5`` and ``t6`` are the hoppings, as
    :func:`p4232_cano2018` takes them; ``None`` takes the value in
    :data:`P4232_PARAMS`. Returns a (4, 4) complex array, in the model's
    orbital order.
    """
    p = dict(P4232_PARAMS)
    for name, val in zip(("t1", "t2", "t3", "t4", "t5", "t6"), (t1, t2, t3, t4, t5, t6)):
        if val is not None:
            p[name] = val
    kx, ky, kz = (float(c) for c in k_cart)

    s0 = np.eye(2)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = np.diag([1.0, -1.0])

    ph = P4232_NN_VECTORS @ np.array([kx, ky, kz])
    f1 = np.cos(ph).sum()
    f2 = np.sin(ph).sum()
    g1 = np.cos(kx) - np.cos(ky)
    g2 = (np.cos(kx) + np.cos(ky) - 2.0 * np.cos(kz)) / np.sqrt(3.0)
    g3 = sum(np.cos(2 * a) * np.cos(b) - np.cos(2 * b) * np.cos(a)
             for a, b in ((kx, ky), (ky, kz), (kz, kx)))

    H = p["t1"] * f1 * np.kron(sx, s0) + p["t2"] * f2 * np.kron(sy, s0)
    H = H + p["t3"] * (g1 * np.kron(sz, sz) + g2 * np.kron(sz, sx))
    H = H + p["t4"] * f1 * np.kron(sy, sy)
    H = H + p["t5"] * np.kron(s0, g2 * sz - g1 * sx)
    H = H + p["t6"] * g3 * np.kron(sz, s0)
    return H.astype(complex)


def p4232_cano2018(t1=None, t2=None, t3=None, t4=None, t5=None, t6=None):
    """The spinless model of Cano et al., Phys. Rev. Lett. 120, 266401 (2018).

    A simple cubic lattice, ``a = 1``, holds the two Wyckoff 2a sites
    ``(0, 0, 0)`` and ``(1/2, 1/2, 1/2)``, each with a ``d_z2`` and a
    ``d_x2-y2`` orbital: the model of arXiv:1711.11045. The paper's generators,
    ``{C_2x|0}``, ``{C_3,111|0}`` and ``{C_2,110|1/2 1/2 1/2}``, are those of
    P4_232, and the group measured is P4_2321' (#208.45). The four bands carry
    the elementary band representation ``1E2E at 2a (231')``. At the defaults,
    which are the values of the paper's Fig. 3a, a gap splits them into two
    halves of two bands. The symmetry data vector of neither half is a sum of
    elementary band representations.

    Parameters
    ----------
    t1 : float, default 0.2
        A hopping between like orbitals of the two sites, ``t1/2`` from the
        site at the origin along each of the eight vectors
        ``(±1/2, ±1/2, ±1/2)``.
    t2 : float, default 0.3
        A hopping between like orbitals of the two sites, ``-t2/2`` from the
        site at the origin along each of ``(1/2, 1/2, 1/2)``,
        ``(1/2, -1/2, -1/2)``, ``(-1/2, 1/2, -1/2)`` and ``(-1/2, -1/2, 1/2)``,
        and ``+t2/2`` along each of their negatives.
    t3 : float, default 0.1
        A hopping from each site to its images in the six nearest cells, of
        opposite sign on the two sites. Along x and y it joins like orbitals,
        with opposite signs for ``d_z2`` and ``d_x2-y2`` and for x and y. Along
        all three axes it joins ``d_z2`` to ``d_x2-y2``, along z with twice the
        amplitude and the opposite sign.
    t4 : float, default 0.08
        The hopping between unlike orbitals of the two sites, along the same
        eight vectors: ``-t4/2`` from ``d_z2`` at the origin to ``d_x2-y2``, and
        ``+t4/2`` from ``d_x2-y2`` at the origin to ``d_z2``.
    t5 : float, default 0.05
        A hopping from each site to its images in the six nearest cells, the
        same on the two sites. Along all three axes it joins like orbitals,
        with opposite signs for ``d_z2`` and ``d_x2-y2``, along z with twice
        the amplitude and the opposite sign. Along x and y it joins ``d_z2`` to
        ``d_x2-y2``, with opposite signs for x and y.
    t6 : float, default 0.02
        A hopping between like orbitals of one site and its images in the 24
        cells at ``(2, 1, 0)`` with every permutation and choice of signs, of
        opposite sign on the two sites. At the site at the origin it is
        ``t6/4`` where the 2 stands one component before the 1, cyclically, as
        in ``(2, 1, 0)`` and ``(0, 2, 1)``, and ``-t6/4`` where it stands one
        after, as in ``(1, 2, 0)`` and ``(2, 0, 1)``.

    Returns
    -------
    pythtb.TBModel
        A spinless three-dimensional model of 4 orbitals and 4 bands. With
        ``t2=0``, the hoppings obey inversion, and the group is Pm-3n1'
        (#223.105). With ``t4=0``, ``t5=0`` and ``t6=0``, the model is the
        paper's bare Hamiltonian, gapped and twofold degenerate at every ``k``.
    """
    p = dict(P4232_PARAMS)
    for name, val in zip(("t1", "t2", "t3", "t4", "t5", "t6"), (t1, t2, t3, t4, t5, t6)):
        if val is not None:
            p[name] = val

    m = TBModel(Lattice(np.eye(3), _P4232_ORB.tolist(), periodic_dirs=[0, 1, 2]),
                spinful=False)

    def bond(amp, i, j, d):
        """Add the hopping ``amp`` from orbital ``i`` to orbital ``j`` along ``d``.

        ``d = r_j + R - r_i`` is the bond vector, ``R`` the lattice vector
        pythtb's ``set_hop`` takes; a zero ``amp`` adds nothing. Raises
        ``AssertionError`` where ``d`` joins the orbitals by no lattice vector.
        """
        if amp == 0.0:
            return
        R = np.asarray(d, float) - (_P4232_ORB[j] - _P4232_ORB[i])
        Ri = np.rint(R).astype(int)
        assert np.allclose(R, Ri, atol=1e-12), f"non-lattice bond {d!r}"
        m.set_hop(amp, i, j, Ri.tolist(), mode="add")

    def cos_diag(coef, i, R):
        """Add ``coef * cos(k·R)`` to the entry ``(i, i)`` of H.

        ``R`` is a lattice vector.
        """
        bond(coef / 2.0, i, i, R)

    def cos_offdiag(coef, i, j, R):
        """Add ``coef * cos(k·R)`` to the entries ``(i, j)`` and ``(j, i)`` of H.

        ``R`` is a lattice vector, and ``i`` and ``j`` are orbitals of one site.
        """
        bond(coef / 2.0, i, j, R)
        bond(coef / 2.0, i, j, -np.asarray(R, float))

    # Inter-site terms: t1 and t2 (sigma_0), and t4 (-i sigma_y).
    a_amp = (p["t1"] - p["t2"]) / 2.0
    b_amp = (p["t1"] + p["t2"]) / 2.0
    for d in P4232_NN_VECTORS:
        for i, j in ((0, 2), (1, 3)):          # sigma_0: orbital-preserving
            bond(a_amp, i, j, d)
            bond(b_amp, i, j, -d)
        for dd in (d, -d):                     # t4 * f1 * (-i sigma_y) block
            bond(-p["t4"] / 2.0, 0, 3, dd)
            bond(+p["t4"] / 2.0, 1, 2, dd)

    # Intra-site terms: t3 (tau_z), t5 (tau_0) and t6 (tau_z).
    ex, ey, ez = np.eye(3)
    r3 = np.sqrt(3.0)
    for site, tz in ((0, +1.0), (2, -1.0)):    # site A block at 0,1; site B at 2,3
        # g1 = cos kx - cos ky  ->  t3 * tau_z * sigma_z  and  -t5 * sigma_x
        for e, sg in ((ex, +1.0), (ey, -1.0)):
            cos_diag(+tz * p["t3"] * sg, site, e)
            cos_diag(-tz * p["t3"] * sg, site + 1, e)
            cos_offdiag(-p["t5"] * sg, site, site + 1, e)
        # g2 = (cos kx + cos ky - 2 cos kz)/sqrt3  ->  t3 * tau_z * sigma_x  and  t5 * sigma_z
        for e, sg in ((ex, 1.0 / r3), (ey, 1.0 / r3), (ez, -2.0 / r3)):
            cos_offdiag(tz * p["t3"] * sg, site, site + 1, e)
            cos_diag(+p["t5"] * sg, site, e)
            cos_diag(-p["t5"] * sg, site + 1, e)
        # g3 = cos(2ka) cos(kb) - cos(2kb) cos(ka) + cyclic perms  ->  t6 * tau_z * sigma_0
        # cos(2ka)cos(kb) = [cos(2ka + kb) + cos(2ka - kb)] / 2
        for u, v in ((0, 1), (1, 2), (2, 0)):
            for pp, qq, sg in ((u, v, +1.0), (v, u, -1.0)):
                for sign in (+1.0, -1.0):
                    R = np.zeros(3)
                    R[pp], R[qq] = 2.0, sign
                    cos_diag(tz * p["t6"] * sg / 2.0, site, R)
                    cos_diag(tz * p["t6"] * sg / 2.0, site + 1, R)
    return m


def p4232_operations():
    """The 24 operations of P4_232, one per coset of the lattice translations.

    They are the closure of the three generators of P4_232 (#208) that Cano et
    al. give, ``{C_2x|0}``, ``{C_3,111|0}`` and ``{C_2,110|1/2 1/2 1/2}``, on
    the simple cubic cell of :func:`p4232_cano2018`. Returns a list of pairs
    ``(R, t)``, ``R`` an integer (3, 3) array on reduced coordinates and ``t``
    the translation in reduced coordinates, in ``[0, 1)``.
    """
    gens = [
        (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.zeros(3)),          # C_2x
        (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), np.zeros(3)),            # C_3,111
        (np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), np.full(3, 0.5)),       # {C_2,110|1/2 1/2 1/2}
    ]
    ops = [(np.eye(3, dtype=int), np.zeros(3))]

    def _key(R, t):
        return (tuple(np.asarray(R).astype(int).ravel()),
                tuple(np.round(np.mod(t, 1.0), 9) % 1.0))

    seen = {_key(*ops[0])}
    changed = True
    while changed:
        changed = False
        for R1, t1 in list(ops):
            for R2, t2 in gens:
                R = (R2 @ R1).astype(int)
                t = np.mod(R2 @ t1 + t2, 1.0)
                k = _key(R, t)
                if k not in seen:
                    seen.add(k)
                    ops.append((R, t))
                    changed = True
    return ops


def p4232_symmetry_cell(generic_site=(0.13, 0.27, 0.41)):
    """A crystal of space group P4_232 (#208): the 2a pair, decorated.

    The 2a pair of :func:`p4232_cano2018` alone is a body-centered cubic point
    set, of space group Im-3m (#229). Cano et al. add atoms that give it a
    crystal field of P4_232 symmetry, and do not give their positions. The
    decoration here is the orbit of ``generic_site`` under P4_232, 24 points;
    ``generic_site`` is a point on no special position, in reduced
    coordinates. Returns the keywords ``real_lattice``, ``positions`` and
    ``typat`` of ``irrep.spacegroup.SpaceGroup.from_cell``, with their values
    for this cell. In ``typat``, type 1 is the 2a pair and type 2 the
    decoration.
    """
    p = np.asarray(generic_site, dtype=float)
    orbit = []
    for R, t in p4232_operations():
        q = np.mod(R @ p + t, 1.0)
        q = np.where(np.abs(q - 1.0) < 1e-9, 0.0, q)
        if not any(np.allclose(q, o, atol=1e-8) for o in orbit):
            orbit.append(q)
    orbit = np.asarray(orbit)
    positions = np.vstack([_P4232_ORB[[0, 2]], orbit])
    typat = np.array([1, 1] + [2] * len(orbit), dtype=int)
    return {"real_lattice": np.eye(3), "positions": positions, "typat": typat}


# ---------------------------------------------------------------------------
# 5.  lin3g (wallpaper group p4mm): a fragile pair above an elementary band
#     representation
# ---------------------------------------------------------------------------

def lin3g():
    """The spinful square layer of Lin et al., Nat. Commun. 15, 550 (2024).

    The two orbitals at 1a, ``(0, 0)``, give four bands. The two orbitals at
    2c, ``(1/2, 0)`` and ``(0, 1/2)``, are spinful ``s`` orbitals. The four
    lowest bands carry their elementary band representation,
    ``-E at 2c (mm21')``. Bands 4 and 5, counted from 0, are two of the four
    bands of the 1a orbitals. They are a fragile pair with a gap above them.
    The model is that of the paper's SM Sec. 3G, SEq. (3.55) and (3.59)
    (arXiv:2207.10099, (C55) and (C60)), with the values of the SFig. 7
    caption: ``t1 = 5.0``, ``t2 = 1.5``, ``vm = -1.5``, ``tPH = 0.1``,
    ``vs = 1.3``, ``vMz = 0.4``, ``vmu = 8.25 vm``, ``vC = 4.0`` and
    ``vCS = 0.45 vC``. :func:`.groups.realized_group` returns P4mm1'
    (#99.164), a flat layer being read on a substrate by default. With
    ``environment="free-standing"``, the orbital positions allow P4/mmm1'
    (#123.340), and the hoppings lower it to P4mm1'.

    Returns
    -------
    pythtb.TBModel
        A spinful two-dimensional model of 4 orbitals and 8 bands.
    """
    sx = np.array([0, 1, 0, 0])
    sy = np.array([0, 0, 1, 0])
    sz = np.array([0, 0, 0, 1])
    I = 1j
    t1, t2, vm, tPH, vs, vMz = 5.0, 1.5, -1.5, 0.1, 1.3, 0.4
    vmu = 8.25 * vm
    vC = 4.0
    vCS = 0.45 * vC

    lat = Lattice([[1, 0], [0, 1]],
                  [[0, 0], [0, 0], [0.5, 0], [0, 0.5]],
                  periodic_dirs=[0, 1])
    m = TBModel(lat, spinful=True)

    # nearest-neighbor quartet hoppings (1a)
    m.set_hop(t1 / 2, 0, 0, [1, 0], mode="add")
    m.set_hop(-t1 / 2, 1, 1, [1, 0], mode="add")
    m.set_hop(t1 / 2, 0, 0, [0, 1], mode="add")
    m.set_hop(-t1 / 2, 1, 1, [0, 1], mode="add")
    m.set_hop(t2 / 2, 0, 1, [1, 0], mode="add")
    m.set_hop(t2 / 2, 1, 0, [1, 0], mode="add")
    m.set_hop(-t2 / 2, 0, 1, [0, 1], mode="add")
    m.set_hop(-t2 / 2, 1, 0, [0, 1], mode="add")

    # on-site terms: quartet mass and the s@2c level
    m.set_onsite([vm, -vm, vmu, vmu], mode="add")

    # particle-hole breaking
    m.set_hop(tPH / 2, 0, 0, [1, 0], mode="add")
    m.set_hop(tPH / 2, 1, 1, [1, 0], mode="add")
    m.set_hop(tPH / 2, 0, 0, [0, 1], mode="add")
    m.set_hop(tPH / 2, 1, 1, [0, 1], mode="add")

    # spin-orbit
    m.set_hop(-I * vs / ((2 * I) ** 2) * sz, 0, 1, [1, 1], mode="add")
    m.set_hop(I * vs / ((2 * I) ** 2) * sz, 1, 0, [1, 1], mode="add")
    m.set_hop(I * vs / ((2 * I) ** 2) * sz, 0, 1, [1, -1], mode="add")
    m.set_hop(-I * vs / ((2 * I) ** 2) * sz, 1, 0, [1, -1], mode="add")
    m.set_hop(sy * vMz / (2 * I), 0, 0, [1, 0], mode="add")
    m.set_hop(-sy * vMz / (2 * I), 1, 1, [1, 0], mode="add")
    m.set_hop(-sx * vMz / (2 * I), 0, 0, [0, 1], mode="add")
    m.set_hop(sx * vMz / (2 * I), 1, 1, [0, 1], mode="add")

    # coupling of the quartet to the s orbitals at 2c
    m.set_hop(vC / 2, 1, 2, [0, 0], mode="add")
    m.set_hop(vC / 2, 1, 2, [-1, 0], mode="add")
    m.set_hop(vC / 2, 1, 3, [0, 0], mode="add")
    m.set_hop(vC / 2, 1, 3, [0, -1], mode="add")
    m.set_hop(0.5 * sy * I * vCS, 1, 2, [0, 0], mode="add")
    m.set_hop(-0.5 * sy * I * vCS, 1, 2, [-1, 0], mode="add")
    m.set_hop(-0.5 * sx * I * vCS, 1, 3, [0, 0], mode="add")
    m.set_hop(0.5 * sx * I * vCS, 1, 3, [0, -1], mode="add")
    return m


# ---------------------------------------------------------------------------
# 6.  The 3c orbit of a simple cubic cell, whose sites carry one another's
#     orbitals
# ---------------------------------------------------------------------------
#
# The three 3c sites of a simple cubic cell, (0, 1/2, 1/2), (1/2, 0, 1/2) and
# (1/2, 1/2, 0), are the centers of the cube's faces. Each has a fourfold or a
# -4 axis along a different Cartesian direction, and the operations of the
# cell permute them. So an operation carrying one site onto the next carries
# p_x at the first onto p_y at the second: the functions at one site are the
# image of those at another, not a copy of them. Both models are spinless
# group averages of random seeds, so each obeys its group exactly.

#: The three ``3c`` sites of a simple cubic cell; site ``s`` has its fourfold
#: axis along the Cartesian direction ``s``.
_CUBIC_3C = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _cubic_rotations(proper_only=True):
    """The 24 operations of ``O`` or of ``T_d``, as signed permutation matrices.

    With ``proper_only=True`` they are the matrices of determinant 1, the
    rotations of ``O``. With ``proper_only=False`` they are the ones with an
    even number of minus signs, the operations of ``T_d``. Returns a list of
    (3, 3) integer arrays, on Cartesian coordinates.
    """
    out = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3), dtype=int)
        for i, j in enumerate(perm):
            P[j, i] = 1
        for signs in itertools.product([1, -1], repeat=3):
            R = P @ np.diag(signs)
            if (int(round(float(np.linalg.det(R)))) == 1 if proper_only
                    else list(signs).count(-1) % 2 == 0):
                out.append(R)
    return out


def _rotated_in_pairs(hops, n, angle):
    """The hoppings ``hops`` in a basis rotated inside each pair of orbitals.

    ``hops`` maps ``(i, j, R)`` to a real hopping, among ``n`` orbitals of
    which ``2s`` and ``2s + 1`` are the pair of site ``s``. The new pair is
    ``q = V.T @ p``, with ``V = [[cos, -sin], [sin, cos]]`` of ``angle`` at
    every site, so that ``H'(R) = V.T @ H(R) @ V`` at every lattice vector
    ``R``. Returns the mapping from ``(i, j, R)`` to the hopping of ``H'``, for
    every ``i`` and ``j`` at each ``R`` of ``hops``.
    """
    c, s = np.cos(angle), np.sin(angle)
    V = np.kron(np.eye(n // 2), np.array([[c, -s], [s, c]]))
    by_R = {}
    for (i, j, R), t in hops.items():
        by_R.setdefault(R, np.zeros((n, n)))[i, j] += t
    out = {}
    for R, H in by_R.items():
        Hp = V.T @ H @ V
        for i in range(n):
            for j in range(n):
                out[(i, j, R)] = float(Hp[i, j])
    return out


def _cubic_3c_p_model(orbitals, ops, seed, angle=0.0):
    """A spinless crystal of ``p`` orbitals on the 3c sites, obeying ``ops``.

    ``orbitals`` lists the orbitals in model order as pairs ``(s, a)``: the
    site ``s``, a row of :data:`_CUBIC_3C`, and the Cartesian direction ``a``
    of its ``p`` orbital. ``ops`` are operations that fix the origin, (3, 3)
    integer matrices on Cartesian coordinates. Every two orbitals at most one
    lattice constant apart, an orbital and itself included, get a random real
    hopping, drawn by the generator seeded with ``seed``. The hoppings are
    averaged over ``ops``, each ``R`` carrying ``p_a`` to ``sum_j R[j, a] p_j``
    at the image site, so the model obeys ``ops`` exactly where they carry the
    orbitals listed onto combinations of one another. A nonzero ``angle``
    writes each site's pair in a basis rotated inside the pair by that angle
    (:func:`_rotated_in_pairs`), for orbitals listed two to a site. Returns the
    model, on the simple cubic cell of lattice constant 1.
    """
    pos = np.array([_CUBIC_3C[s] for s, _ in orbitals])
    index = {pair: o for o, pair in enumerate(orbitals)}

    def site_image(R, s):
        r = R @ _CUBIC_3C[s]
        for s2 in range(3):
            L = r - _CUBIC_3C[s2]
            if np.allclose(L, np.round(L)):
                return s2, np.round(L).astype(int)
        raise AssertionError("the 3c orbit is not closed under this operation")

    def orbital_image(R, o):
        s, a = orbitals[o]
        s2, L = site_image(R, s)
        return [(index[(s2, j)], float(R[j, a]), L)
                for j in range(3) if (s2, j) in index and R[j, a] != 0]

    rng = np.random.default_rng(int(seed))
    seeds, shifts = {}, [np.array(v) for v in itertools.product([-1, 0, 1], repeat=3)]
    for i in range(len(orbitals)):
        for j in range(len(orbitals)):
            for L in shifts:
                if np.linalg.norm(pos[j] + L - pos[i]) <= 1.01 \
                        and (i, j, tuple(L)) not in seeds:
                    t = float(rng.normal())
                    seeds[(i, j, tuple(L))] = t
                    seeds[(j, i, tuple(-L))] = t
    hops = {}
    for (i, j, L), t in seeds.items():
        for R in ops:
            for i2, wi, Li in orbital_image(R, i):
                for j2, wj, Lj in orbital_image(R, j):
                    key = (i2, j2, tuple((Lj + R @ np.array(L) - Li).astype(int)))
                    hops[key] = hops.get(key, 0.0) + wi * wj * t / len(ops)
    if angle:
        hops = _rotated_in_pairs(hops, len(orbitals), angle)

    m = TBModel(Lattice(np.eye(3).tolist(), pos.tolist(), periodic_dirs=[0, 1, 2]),
                spinful=False)
    entered = set()
    for (i, j, L), t in hops.items():
        if abs(t) < 1e-12 or (j, i, tuple(-np.array(L))) in entered:
            continue
        entered.add((i, j, L))
        if i == j and L == (0, 0, 0):
            m.set_onsite(float(t), i)
        else:
            m.set_hop(float(t), i, j, list(L))
    return m


def cubic_3c_axis_p(seed=11):
    """One ``p`` orbital per 3c site of a simple cubic cell, along the site's axis.

    The orbitals are ``p_x`` at ``(0, 1/2, 1/2)``, ``p_y`` at ``(1/2, 0, 1/2)``
    and ``p_z`` at ``(1/2, 1/2, 0)``, each along its site's fourfold axis.
    Every two orbitals at most one lattice constant apart are joined by a
    random real hopping, averaged over the rotations of ``O``, under which the
    three ``p`` orbitals transform as the components of a vector. An operation
    carrying one site onto another carries its ``p`` onto a different ``p``.
    Averaged over ``O`` alone, the hoppings obey the whole of Pm-3m1'
    (#221.93). Every site carries ``A2u`` of its site group ``4/mmm1'``.
    ``A2u`` is not the restriction to ``4/mmm1'`` of any one-dimensional
    representation of Pm-3m1', so the hoppings rule out an ``s`` orbital. Of
    the site irreps they leave, the lowest-angular-momentum rule assigns
    ``A2u``.

    Parameters
    ----------
    seed : int, default 11
        The seed of the random generator that draws the hoppings.

    Returns
    -------
    pythtb.TBModel
        A spinless three-dimensional model of 3 orbitals and 3 bands.
    """
    return _cubic_3c_p_model([(s, s) for s in range(3)],
                             _cubic_rotations(proper_only=True), seed)


def cubic_3c_perp_doublet(angle=0.0, seed=7):
    """The ``p`` pair perpendicular to each 3c site's axis, in a simple cubic cell.

    The orbitals are ``(p_y, p_z)`` at ``(0, 1/2, 1/2)``, ``(p_x, p_z)`` at
    ``(1/2, 0, 1/2)`` and ``(p_x, p_y)`` at ``(1/2, 1/2, 0)``. Every two
    orbitals at most one lattice constant apart are joined by a random real
    hopping, averaged over the operations of ``T_d``. The sites alone allow
    Pm-3m1' (#221.93), and the hoppings lower it to P-43m1' (#215.71). Its site
    group at 3c is ``-42m1'``, with the ``-4`` axis along the site's own. Each
    pair carries ``E`` of that site group. An operation carrying one site onto
    another carries its pair onto the pair at the image site, which is made of
    different ``p`` orbitals. The band representation induced from ``E`` at 3c
    is equivalent to ``T1 at 1b (-43m1') + T2 at 1b (-43m1')``. So
    :func:`pythtqc.ebr_data` lists no elementary band representation induced
    from ``E`` at 3c.

    Parameters
    ----------
    angle : float, default 0.0
        An angle in radians. Each site's pair ``p`` is rotated within itself to
        ``q = V.T @ p``, with ``V = [[cos, -sin], [sin, cos]]`` of the angle.
    seed : int, default 7
        The seed of the random generator that draws the hoppings.

    Returns
    -------
    pythtb.TBModel
        A spinless three-dimensional model of 6 orbitals and 6 bands. With a
        nonzero ``angle``, the orbital basis inside each pair is changed and
        nothing else: the group, the site irreps and the symmetry data vector
        are those at ``angle=0``.
    """
    return _cubic_3c_p_model([(s, a) for s in range(3) for a in range(3) if a != s],
                             _cubic_rotations(proper_only=False), seed, angle=angle)
