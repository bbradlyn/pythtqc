"""Tight-binding crystals to measure on, each chosen to make one question sharp.

Every builder returns a freshly built PythTB v2.0 model; its docstring says what
the model is, where it comes from, and what pythtqc measures on it.

==============================  ===========================================================
builder                         what it is
==============================  ===========================================================
:func:`i41_atomic`              3D **spinful** body-centred tetragonal, 12 bands, SG **80**
                                (I4_1): ``s`` at 4a below a gap, ``s`` at the general 8b
                                above.  At ``P``, a point that is not time-reversal
                                invariant, its corep list is ``{-P3 (1), -P4P4 (2)}``,
                                where the Bilbao Crystallographic Server's convention
                                gives ``{-P3P3 (2), -P4 (1)}``.
:func:`i41_soc`                 :func:`i41_atomic` with spin-orbit coupling, exactly
                                I4_11'-symmetric.  ``soc="generic"`` splits the valence
                                level at ``P`` into ``[1, 2, 1]``, the doublet ``-P4P4``;
                                ``soc="axial"`` conserves ``s_z``.
:func:`p62_atomic`              3D **spinful** hexagonal, 18 bands, SG **171** (P6_2):
                                ``s`` at 3a below, ``s`` at the general 6c above.  At
                                ``H`` its corep list is the server's own.
:func:`i213_flicker2018`        3D **spinless** body-centred cubic, 4 bands, SG **199**
                                (I2_1 3), Flicker *et al.*, PRB **98**, 155145 (2018),
                                App. F.1: spin-1 fermions at ``GM`` and ``H``, and the
                                8a elementary band representation **connected**.
:func:`p4232_cano2018`          3D **spinless** cubic, 4 bands, SG **208** (P4_232),
                                Cano *et al.*, PRL **120**, 266401 (2018): a
                                **disconnected** elementary band representation, each
                                two-band branch a nonnegative sum of none.
:func:`cubic_3c_axis_p`         3D **spinless** simple cubic, 3 bands: one ``p`` at each
                                ``3c`` site, along that site's own fourfold axis.
                                ``Pm-3m1'`` (SG **221**), ``A2u`` at every site.
:func:`cubic_3c_perp_doublet`   3D **spinless** simple cubic, 6 bands: the ``p`` pair
                                perpendicular to each ``3c`` site's own axis.
                                ``P-43m1'`` (SG **215**), ``E`` of ``-42m`` at every
                                site, a **composite** band representation.
:func:`lin3g`                   2D **spinful** square layer, 8 bands, wallpaper group
                                **p4mm**: a fragile pair at ``1a`` above the ``s``
                                elementary band representation at ``2c``, six bands
                                occupied.  Lin et al., Nat. Commun. **15**, 550 (2024).
==============================  ===========================================================

:func:`orbit_of`, :func:`closed_bonds`, :func:`atomic_model`, :func:`spin_bonds`
and :func:`primitive_ops` build a crystal symmetric under a group by
construction, as the builders do.
"""

from __future__ import annotations

import itertools

import numpy as np
from pythtb import Lattice, TBModel

__all__ = [
    # the shared constructions: a crystal symmetric by construction
    "orbit_of", "closed_bonds", "atomic_model", "primitive_ops",
    "spin_bonds", "spinor_of", "cartesian_rotation", "PAULI",
    # the eight builders of the table above
    "i41_atomic", "i41_soc", "p62_atomic",
    "i213_flicker2018", "p4232_cano2018",
    "cubic_3c_axis_p", "cubic_3c_perp_doublet",
    "lin3g",
]


# ---------------------------------------------------------------------------
# 1.  the shared constructions
# ---------------------------------------------------------------------------
#
# :func:`orbit_of`, :func:`closed_bonds` and :func:`atomic_model` build a
# spinless (or spin-doubled) atomic-limit model on Wyckoff orbits by closing a
# handful of seed bonds under a group, so the model is exactly symmetric BY
# CONSTRUCTION rather than by inspection -- the detector then has something to
# confirm, not to discover.  :func:`spin_bonds` is the
# matrix-valued (spin-orbit) counterpart, and :func:`primitive_ops` carries
# ``{R|t}`` from a centred conventional cell into the primitive one that pythtb
# is handed.

def orbit_of(ops, seed, tol=1e-9):
    """The orbit of ``seed`` under ``ops``: its distinct images modulo the lattice, in reduced coordinates.

    They come in the order the operations generate them, which fixes the orbital
    order of the models here.

    Parameters
    ----------
    ops : sequence of (R, t)
        Operations as :data:`I41_OPS` writes them: a flat nine-entry rotation on
        reduced coordinates and a translation in the same basis.
    seed : (3,) array_like
        The point, in reduced coordinates.
    tol : float, optional
        How close two images are, in reduced coordinates, to be one point.
        Default ``1e-9``.

    Returns
    -------
    (n, 3) numpy.ndarray
        The orbit, in generation order.
    """
    out = []
    for R, t in ops:
        q = (np.asarray(R, dtype=float).reshape(3, 3) @ np.asarray(seed, dtype=float)
             + np.asarray(t, dtype=float)) % 1.0
        if not any(np.abs(((q - p) + 0.5) % 1.0 - 0.5).max() < tol for p in out):
            out.append(q)
    return np.array(out)


def closed_bonds(sites, ops, seeds, tol=1e-6):
    """Every image of a few scalar seed hoppings under ``ops``: hoppings exactly symmetric under the group.

    An operation carries the bond ``(i, j, L)`` to ``(σ(i), σ(j), a_j + R L - a_i)``
    with the seed's own amplitude, and a bond and its hermitian partner are
    counted once.  :func:`spin_bonds` is the counterpart for spin-dependent
    hoppings.

    Parameters
    ----------
    sites : (nsite, 3) array_like
        Every site, in reduced coordinates and orbital order; ``ops`` carries the
        set onto itself.
    ops : sequence of (R, t)
        The operations, as :data:`I41_OPS` writes them.
    seeds : sequence of (int, int, tuple, float)
        ``(i, j, L, t)`` per seed, for ``t c_i(0)^dag c_j(L)``, ``L`` a lattice
        vector in integer reduced components.
    tol : float, optional
        How close an image lands to a site, in reduced coordinates, to be it.
        Default ``1e-6``.

    Returns
    -------
    dict
        ``{(i, j, L): amplitude}`` over every image.
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
                raise AssertionError("the site set is not closed under the group")
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
    """An ``s``-orbital crystal on ``orbits``, exactly symmetric under ``ops``.

    One orbital per site, the on-site energies given, and the seed hoppings closed
    under ``ops`` by :func:`closed_bonds`.  ``spinful=True`` carries the same
    spin-independent hoppings on both spin states: the band representation is the
    spinless one times spin ½, induced from the double-valued site
    representation, and every level is doubly degenerate.

    Parameters
    ----------
    lattice : (3, 3) array_like
        Lattice vectors as rows, in Cartesian components.
    ops : sequence of (R, t)
        The operations, as :data:`I41_OPS` writes them.
    orbits : sequence of (n, 3) array_like
        The Wyckoff orbits, in reduced coordinates, stacked in the order given:
        the orbital order.
    onsites : sequence of float
        One on-site energy per site.
    seeds : sequence of (int, int, tuple, float)
        Seed hoppings for :func:`closed_bonds`, indexed into the stacked sites.
    spinful : bool, optional
        Two spin states per orbital.  Default ``False``.

    Returns
    -------
    pythtb.TBModel
        One orbital per site, obeying ``ops`` exactly.
    """
    sites = np.vstack(list(orbits))
    lat = Lattice(np.asarray(lattice, dtype=float).tolist(), sites.tolist(),
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=spinful)
    m.set_onsite(list(onsites))
    for (i, j, L), amp in closed_bonds(sites, ops, seeds).items():
        m.set_hop(float(amp), i, j, list(L), mode="add")
    return m


#: ``(sigma_x, sigma_y, sigma_z)`` -- the spin matrices the SOC seeds are
#: written in.  Spin is the **fast** state index in pythtb, so a ``2x2`` bond
#: matrix here is the spin block of one orbital pair and nothing needs
#: transposing.
PAULI = (np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]], dtype=complex),
         np.array([[1, 0], [0, -1]], dtype=complex))


def cartesian_rotation(R_red, lattice):
    """A rotation given on reduced coordinates, in Cartesian coordinates: ``latticeᵀ R lattice⁻ᵀ``.

    An operation's spin rotation is read off this matrix.

    Parameters
    ----------
    R_red : (3, 3) array_like
        The rotation on reduced coordinates; a flat nine entries is reshaped.
    lattice : (3, 3) array_like
        Lattice vectors as rows, in Cartesian components.

    Returns
    -------
    (3, 3) numpy.ndarray
    """
    lat = np.asarray(lattice, dtype=float)
    R = np.asarray(R_red, dtype=float).reshape(3, 3)
    return lat.T @ R @ np.linalg.inv(lat.T)


def spinor_of(R_cart, tol=1e-9):
    """The SU(2) matrix ``exp(-i θ n·σ/2)`` of a proper Cartesian rotation.

    The bonds built with it, ``S T S†``, are the same for either of its two lifts.

    Parameters
    ----------
    R_cart : (3, 3) array_like
        A proper rotation, ``det = +1``; an improper one raises
        :class:`AssertionError`.
    tol : float, optional
        Slack on ``det = +1`` and on the angles ``0`` and ``π``.  Default
        ``1e-9``.

    Returns
    -------
    (2, 2) numpy.ndarray of complex
    """
    Rc = np.asarray(R_cart, dtype=float)
    if abs(np.linalg.det(Rc) - 1.0) > tol:             # pragma: no cover - guard
        raise AssertionError("spinor_of handles proper rotations only")
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
    """Spin-dependent hoppings averaged over ``ops``: exactly symmetric under the group.

    An operation carries a seed's bond as :func:`closed_bonds` does and its
    ``2x2`` spin block as ``T -> S T S†``, ``S`` the SU(2) lift of its proper part,
    so that spin is even under inversion.  Different operations carry one seed onto
    one bond with different blocks, so every image is added, one landing on the
    reversed bond as its adjoint, and the sum is divided by the number of
    operations: the group average.

    Parameters
    ----------
    sites : (nsite, 3) array_like
        Every site, in reduced coordinates and orbital order; ``ops`` carries the
        set onto itself.
    ops : sequence of (R, t)
        The operations, as :data:`I41_OPS` writes them.
    lattice : (3, 3) array_like
        Lattice vectors as rows, in Cartesian components: the spin rotation is
        read off an operation's Cartesian form.
    seeds : sequence of (int, int, tuple, ndarray)
        ``(i, j, L, T)`` per seed, for ``c_i(0)† T c_j(L)``, ``T`` a ``2x2`` spin
        matrix.  An on-site seed raises :class:`AssertionError`; it belongs to
        ``set_onsite``.
    tol : float, optional
        How close an image lands to a site, in reduced coordinates, to be it.
        Default ``1e-6``.

    Returns
    -------
    dict
        ``{(i, j, L): (2, 2) ndarray}``, the group average.
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
                raise AssertionError("the site set is not closed under the group")
        perm.append(p)
        shift.append(s)
        # an improper operation acts on the spin as its proper part: the spin
        # does not transform under inversion, S(I·R) = S(R) -- the convention
        # the tables' lifts use, and the one that makes a spinful s orbital the
        # gerade spinor at an inversion centre
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
                raise AssertionError("spin_bonds does not take onsite seeds")
            if rkey in out:
                out[rkey] = out[rkey] + T2.conj().T
            elif key in out:
                out[key] = out[key] + T2
            else:
                out[key] = T2
    return {k: v / len(ops) for k, v in out.items()}


def primitive_ops(ops, centering_rows):
    """Operations ``{R|t}`` of a centred conventional cell, written in its primitive cell.

    With ``x_conv = Mᵀ u``, ``x -> R x + t`` becomes ``u -> (M⁻ᵀ R Mᵀ) u + M⁻ᵀ t``.
    A model is built on the primitive cell, a centred conventional cell being a
    supercell, which :func:`..groups.realized_group` refuses.

    Parameters
    ----------
    ops : sequence of (R, t)
        The operations in the conventional cell, as :data:`I41_OPS` writes them.
    centering_rows : (3, 3) array_like
        The primitive lattice vectors in the conventional cell's reduced
        coordinates, as rows: :data:`I41_CENTERING` for a body-centred cell.

    Returns
    -------
    tuple of (tuple of int, tuple of float)
        The operations in the primitive cell: rotations as flat nine-entry
        integer tuples, translations in primitive reduced coordinates.
    """
    mt = np.asarray(centering_rows, dtype=float).T
    inv = np.linalg.inv(mt)
    out = []
    for R, t in ops:
        rp = inv @ np.asarray(R, dtype=float).reshape(3, 3) @ mt
        if np.abs(rp - np.rint(rp)).max() > 1e-9:      # pragma: no cover - guard
            raise AssertionError("the centering rows do not span a sublattice")
        out.append((tuple(np.rint(rp).astype(int).ravel()),
                    tuple(inv @ np.asarray(t, dtype=float))))
    return tuple(out)


# ---------------------------------------------------------------------------
# 2.  the re-gluing pair -- I4_1 (#80) and P6_2 (#171), **spinful**
# ---------------------------------------------------------------------------
#
# The fourth face of the convention artifact: at these two k-points the
# time-reversal GLUING is convention-sensitive, so the physical corep list is a
# different partition of the same unitary irreps than the tables'.  The pair
# are the two SIDES of that face, and the contrast is the point of keeping both:
#
#   * P6_2's ``H`` is the EXCHANGE side.  The moved gluing produces a corep
#     that is character-identical to one the tables carry, so the returned name
#     follows the character back onto the tables' own label and the returned page
#     IS the tables' page.  Nothing is left in `reglued_points`, and #171.122's
#     transport is a no-op end to end.
#   * I4_1's ``P`` is the IRREDUCIBLE side.  Conjugating the shipped rows there
#     does not land on the shipped rows at all -- the tables' page tabulates the
#     other star member -P -- so no relabelling exists and the returned basis
#     carries names irreptables does not ship (`{-P3, -P4P4}` against the
#     tables' `{-P3P3, -P4}`).  The irrep the BCS convention doubles is not the
#     one the physical convention doubles, and `i41_soc` measures that on a real
#     spectrum.
#
# Both models are spinful deep atomic limits with spin-independent hoppings, so their content is the
# spinless induced character times the SU(2) trace and can be written down by
# hand; the point of interest is I4_1's ``P`` (non-TRIM of a body-centred
# lattice, pairing antiunitary ``C4·T``) and P6_2's ``H`` (non-TRIM of a
# hexagonal lattice, pairing antiunitary a two-fold times ``T``).

#: ``{R|t}`` of I4_1 (#80) in the ITA / BCS **conventional** (body-centred)
#: setting, in ``irreptables``' index order.
I41_OPS = (
    ((1, 0, 0, 0, 1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),          # 1  E
    ((-1, 0, 0, 0, -1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),        # 2  2 0,0,z
    ((0, -1, 0, 1, 0, 0, 0, 0, 1), (0.0, 0.5, 0.25)),        # 3  4+(0,0,1/4)
    ((0, 1, 0, -1, 0, 0, 0, 0, 1), (0.0, 0.5, 0.25)),        # 4  4-(0,0,1/4)
)

#: Primitive vectors of the body-centred cell, in conventional reduced
#: coordinates (rows) -- the standard ``I`` setting.
I41_CENTERING = np.array([[-0.5, 0.5, 0.5],
                          [0.5, -0.5, 0.5],
                          [0.5, 0.5, -0.5]])

#: ``{R|t}`` of I4_1 on the primitive cell, which is what pythtb is handed.
I41_PRIM_OPS = primitive_ops(I41_OPS, I41_CENTERING)

#: Body-centred tetragonal, ``c/a = 1.6``; the primitive vectors in Cartesian.
I41_LATTICE = I41_CENTERING @ np.diag([1.0, 1.0, 1.6])

#: ``{R|t}`` of P6_2 (#171), primitive hexagonal, in ``irreptables``' order.
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

#: 4a ``(0,0,z)`` and the general 8b of I4_1, generic free parameters.  ``4a``
#: alone is far too symmetric for an ``s``-orbital model (spglib reads
#: I4_1/amd, #141); the general orbit is what pins the group down to #80.
I41_PARAMS = dict(z_a=0.13, seed_b=(0.21, 0.34, 0.07), delta=2.5,
                  t_a=0.09, t_b=0.05, t_ab=0.03)

#: 3a ``(0,0,z)`` and the general 6c of P6_2, likewise (3a + 3b alone reads
#: P6_222, #180).
P62_PARAMS = dict(z_a=0.11, seed_c=(0.21, 0.34, 0.07), delta=2.5,
                  t_a=0.09, t_b=0.05, t_ab=0.03)


def i41_sites(z_a=None, seed_b=None):
    """``(orbit_4a, orbit_8b)`` of I4_1 in **primitive** reduced coordinates.

    Parameters
    ----------
    z_a : float, optional
        The free ``(0, 0, z)`` parameter of the ``4a`` orbit, in conventional
        reduced coordinates.  Default :data:`I41_PARAMS`'s ``z_a``.
    seed_b : (3,) array_like, optional
        A generic seed for the ``8b`` orbit, in conventional reduced
        coordinates.  Default :data:`I41_PARAMS`'s ``seed_b``.

    Returns
    -------
    ((n, 3) ndarray, (m, 3) ndarray)
        The two orbits, in primitive reduced coordinates, in the order the
        operations generate them.
    """
    z_a = I41_PARAMS["z_a"] if z_a is None else float(z_a)
    seed_b = I41_PARAMS["seed_b"] if seed_b is None else tuple(seed_b)
    to_prim = np.linalg.inv(I41_CENTERING.T)
    return (orbit_of(I41_PRIM_OPS, to_prim @ np.array([0.0, 0.0, z_a])),
            orbit_of(I41_PRIM_OPS, to_prim @ np.asarray(seed_b, dtype=float)))


def i41_atomic(z_a=None, seed_b=None, delta=None, t_a=None, t_b=None, t_ab=None):
    """A spinful atomic insulator in I4_1 (SG 80): ``s`` orbitals at 4a below a gap, at the general 8b above.

    Six ``s`` orbitals in the primitive body-centred cell, two on the ``4a`` orbit
    ``(0, 0, z)`` and four on a generic ``8b`` orbit, carried spinful: 12 bands, the
    four occupied ones ``-1E2E↑G(4)@4a``, the group's only elementary band
    representation.  At ``P``, a point that is not time-reversal invariant, time
    reversal glues the irreps differently from the Bilbao Crystallographic
    Server's convention: the corep list returned is ``{-P3 (1), -P4P4 (2)}``, where
    the server's is ``{-P3P3 (2), -P4 (1)}``.

    Parameters
    ----------
    z_a, seed_b : float and (3,) array_like, optional
        The ``4a`` parameter and a generic ``8b`` seed, in conventional reduced
        coordinates.  Default :data:`I41_PARAMS`.
    delta, t_a, t_b, t_ab : float, optional
        The on-site energies ``-delta`` at 4a and ``+delta`` at 8b, and the
        hoppings within 4a, within 8b and between them.  Default
        :data:`I41_PARAMS`.

    Returns
    -------
    pythtb.TBModel
        Spinful 3D model, 6 orbitals, 12 bands.
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


#: The two SOC seed sets of :func:`i41_soc`, both on the ``t_a`` and ``t_ab``
#: bonds of :func:`i41_atomic`.  ``"generic"`` seeds ``i b·σ`` with a generic
#: ``b`` (plus a second, differently-oriented block) and leaves the commutant
#: of ``{H(k)}`` at ``C·1``; ``"axial"`` seeds ``i σ_z`` on both, which
#: **conserves** ``s_z`` and makes the commutant two-dimensional.  Both are
#: exactly symmetric -- every operation of I4_11' validates on the default
#: 8**3 mesh, at 1.5e-14 (generic) and 2.1e-14 (axial) -- so what separates
#: them is the commutant dimension alone, which is the point.
_I41_SOC_B = (0.31, -0.52, 0.44)
I41_SOC_SEEDS = {
    "generic": (1j * (_I41_SOC_B[0] * PAULI[0] + _I41_SOC_B[1] * PAULI[1]
                      + _I41_SOC_B[2] * PAULI[2]),
                0.7 * 1j * (_I41_SOC_B[0] * PAULI[0] + _I41_SOC_B[1] * PAULI[1]
                            + _I41_SOC_B[2] * PAULI[2]) + 0.2j * PAULI[0]),
    "axial": (1j * PAULI[2], 1j * PAULI[2]),
}


def i41_soc(soc="generic", lam=0.4, **kwargs):
    """:func:`i41_atomic` with spin-orbit coupling, averaged over I4_1 so that it obeys I4_11' exactly.

    The hoppings of :func:`i41_atomic` do not depend on spin, so all of SU(2)
    commutes with its ``H(k)``; the spin-orbit bonds, on the ``t_a`` and ``t_ab``
    hoppings, leave less.

    * ``soc="generic"``: ``i λ b·σ`` with a generic ``b``.  Nothing beyond the
      crystal's symmetry is conserved, and the four valence states at ``P`` split
      ``[1, 2, 1]``: two singlets ``-P3`` and a doublet whose characters are
      ``-P4P4``'s, the corep list of :func:`i41_atomic` seen in a spectrum.
    * ``soc="axial"``: ``i λ σ_z`` on both bonds, which every operation of I4_1
      keeps, so ``s_z`` is conserved.

    Parameters
    ----------
    soc : str or sequence, optional
        A key of :data:`I41_SOC_SEEDS`, or a pair ``(T_a, T_ab)`` of ``2x2`` spin
        matrices without the factor ``lam``.  Default ``"generic"``.
    lam : float, optional
        The spin-orbit strength.  Default ``0.4``.
    **kwargs
        The parameters :func:`i41_atomic` takes.

    Returns
    -------
    pythtb.TBModel
        Spinful 3D model, 6 orbitals, 12 bands.
    """
    p = dict(I41_PARAMS)
    for key, val in kwargs.items():
        if key not in p:                                 # pragma: no cover - guard
            raise TypeError(f"i41_soc got an unexpected keyword {key!r}")
        if val is not None:
            p[key] = val if key == "seed_b" else float(val)
    seeds = I41_SOC_SEEDS[soc] if isinstance(soc, str) else tuple(soc)
    if len(seeds) != 2:                                  # pragma: no cover - guard
        raise ValueError("soc must name a key of I41_SOC_SEEDS or be a (T_a, T_ab) pair")

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
    """``(orbit_3a, orbit_6c)`` of P6_2 in reduced coordinates.

    Parameters
    ----------
    z_a : float, optional
        The free ``(0, 0, z)`` parameter of the ``3a`` orbit, in reduced
        coordinates.  Default :data:`P62_PARAMS`'s ``z_a``.
    seed_c : (3,) array_like, optional
        A generic seed for the ``6c`` orbit, in reduced coordinates.  Default
        :data:`P62_PARAMS`'s ``seed_c``.

    Returns
    -------
    ((n, 3) ndarray, (m, 3) ndarray)
        The two orbits, in reduced coordinates, in the order the operations
        generate them.
    """
    z_a = P62_PARAMS["z_a"] if z_a is None else float(z_a)
    seed_c = P62_PARAMS["seed_c"] if seed_c is None else tuple(seed_c)
    return (orbit_of(P62_OPS, (0.0, 0.0, z_a)),
            orbit_of(P62_OPS, seed_c))


def p62_atomic(z_a=None, seed_c=None, delta=None, t_a=None, t_b=None, t_ab=None):
    """A spinful atomic insulator in P6_2 (SG 171): ``s`` orbitals at 3a below a gap, at the general 6c above.

    The hexagonal companion of :func:`i41_atomic`: nine ``s`` orbitals, three on
    the ``3a`` orbit ``(0, 0, z)`` and six on a generic ``6c``, carried spinful, 18
    bands, the lowest six ``-1E2E↑G(6)@3a``.  At ``H = (1/3, 1/3, 1/2)``, not
    time-reversal invariant, the antiunitary operation that pairs the irreps is a
    twofold rotation with time reversal, and the gluing differs from the Bilbao
    Crystallographic Server's convention; the corep it makes is one the tables
    carry, so the list returned there is the server's own,
    ``{-H4 (1,-1,1), -H5H6 (2,1,-1)}``.

    Parameters
    ----------
    z_a, seed_c : float and (3,) array_like, optional
        The ``3a`` parameter and a generic ``6c`` seed, in reduced coordinates.
        Default :data:`P62_PARAMS`.
    delta, t_a, t_b, t_ab : float, optional
        The on-site energies ``-delta`` at 3a and ``+delta`` at 6c, and the
        hoppings within 3a, within 6c and between them.  Default
        :data:`P62_PARAMS`.

    Returns
    -------
    pythtb.TBModel
        Spinful 3D model, 9 orbitals, 18 bands.
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
# 3.  I2_1 3 (SG 199) -- the CONNECTED multifold model
# ---------------------------------------------------------------------------
#
# PROVENANCE.  Flicker, de Juan, Bradlyn, Morimoto, Vergniory & Grushin,
# "Chiral Optical Response of Multifold Fermions", Phys. Rev. B **98**, 155145
# (2018) = arXiv:1806.09642, **Appendix F.1** ("SGs 199 and 214"), transcribed
# from the arXiv LaTeX source:
#
# * Eq. (88)  R_1 = (a/2)(-x + y + z), R_2 = (a/2)(x - y + z),
#             R_3 = (a/2)(x + y - z);  a = 1 throughout.
# * Eq. (89)  G_199 = < {C_2x | 1/2 1/2 0}, {C_3,111 | 000}, translations >,
#             translations written in **reduced** coordinates (Eq. (90)).
# * Eq. (92)  the 8a orbit, reduced: q_1 = (u,u,u), q_2 = (1/2-u, 1/2, 0),
#             q_3 = (0, 1/2-u, 1/2), q_4 = (1/2, 0, 1/2-u), multiplicity 4 in
#             the primitive cell.  "-1/2 < u < 1/2" for SG 199; u = +-1/4
#             promotes the group to SG 214, so this model uses a generic u.
# * Eq. (93)  |q_1 - q_j|^2 = 1/4 - u + 2u^2, equal for all symmetry-related
#             bonds (asserted in the test suite).
# * Eq. (94)  the nearest-neighbour Hamiltonian, six bonds + h.c.
# * Eqs. (95)-(98)  its Fourier transform, ``V^dag(u,k) H_0(k) V(u,k)``, which
#             is pythtb's convention I on the nose (measured at 4.7e-16).
#
# Spinless s orbitals: "By using s-orbitals as our basis functions, all
# symmetries in the stabilizer group of the 8a position are represented
# trivially."  The paper's Fig. 4(b) is this model's band structure and its
# text states that SG 199 without spin-orbit coupling "realizes spin-1 fermions
# at Gamma and H" -- measured here as the threefold GM4 at -t and H4 at +t.
#
# Why this model is worth having: it is pathological in three independent ways at
# once, each of which stresses a different part of the machinery.
#   (i)   the Bravais lattice is body-centred, so the conventional cubic cell is
#         a supercell: `_group_of` returns it as the index-2 lattice supergroup
#         (`lattice_index == 2`, the centring translation among the operations)
#         and `realized_group` MUST refuse it (gate="supercell"), since this
#         Hamiltonian obeys that translation exactly;
#   (ii)  the group is nonsymmorphic (Hermann-Mauguin I2_1 3) and yet **not one
#         of its rotations is intrinsically a screw** -- the body-centring
#         translation absorbs every axial component, measured operation by
#         operation in the test suite;
#   (iii) the four sites of the single Wyckoff orbit permute *between unit
#         cells*: under the group's unitary operations, one per coset of the
#         lattice translations, 18 of the 48 images of the four sites lie in
#         another cell, and the test suite checks that the symmetry matrices
#         measured from the Hamiltonian alone, which assume nothing about how an
#         operation acts on the orbitals, put their support exactly there -- so
#         the permutation is a measurement, not a position-table lookup.

#: Eq. (88) with ``a = 1``: the primitive body-centred cubic basis.
I213_LATTICE = 0.5 * np.array([[-1.0, 1.0, 1.0],
                                [1.0, -1.0, 1.0],
                                [1.0, 1.0, -1.0]])

#: The free Wyckoff-8a parameter.  Any ``u`` in ``(-1/2, 1/2) \\ {+-1/4}`` gives
#: SG 199; ``u = +-1/4`` promotes the group to SG 214 (paper, App. F.1).
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
    """Eq. (92): the four 8a positions in reduced (primitive) coordinates.

    Parameters
    ----------
    u : float, optional
        The free Wyckoff-8a parameter, in reduced coordinates.  Default
        :data:`I213_U`; ``±1/4`` promotes the group to SG 214.

    Returns
    -------
    (4, 3) numpy.ndarray
        The orbit in primitive reduced coordinates, in the paper's order.
    """
    u = float(u)
    return np.array([[u, u, u], [0.5 - u, 0.5, 0.0],
                     [0.0, 0.5 - u, 0.5], [0.5, 0.0, 0.5 - u]])


def i213_bloch_hamiltonian(k_cart, u=I213_U, t=1.0):
    """The paper's Bloch Hamiltonian, Eqs. (96)-(98): ``H(k) = t V†(u,k) H_0(k) V(u,k)``.

    ``V = diag(e^{i k·q_i})`` places the orbitals, and ``H_0`` does not depend on
    ``u``: the spectrum does not depend on where the orbitals sit.

    Parameters
    ----------
    k_cart : (3,) array_like
        ``k`` in Cartesian units with ``a = 1``;
        ``numpy.linalg.solve(I213_LATTICE, 2*pi*k_red)`` converts a reduced one.
    u : float, optional
        The 8a parameter.  Default :data:`I213_U`.
    t : float, optional
        The nearest-neighbour hopping.  Default ``1.0``.

    Returns
    -------
    (4, 4) numpy.ndarray of complex
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
    """The SG 199 (I2_1 3) model of Flicker et al., arXiv:1806.09642, App. F.1.

    Four spinless ``s`` orbitals at Wyckoff 8a of the primitive body-centred cubic
    cell, with nearest-neighbour hopping.  The four bands are connected: the
    threefold ``GM4`` is at the top of the spectrum at ``GM`` and its partner
    ``H4`` at the bottom at ``H``, so no one- or three-band set can be isolated.

    Parameters
    ----------
    u : float, optional
        The 8a parameter; ``±1/4`` makes the group SG 214.  Default
        :data:`I213_U`.
    t : float, optional
        The nearest-neighbour hopping, the energy scale.  Default ``1.0``.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 4 orbitals, 4 bands.
    """
    lat = Lattice(I213_LATTICE.tolist(), i213_sites(u).tolist(),
                  periodic_dirs=[0, 1, 2])
    m = TBModel(lat, spinful=False)
    for i, j, R in I213_BONDS:
        m.set_hop(float(t), i, j, list(R), mode="add")
    return m


def i213_conventional(u=I213_U, t=1.0):
    """The crystal of :func:`i213_flicker2018` in the conventional cubic cell: a supercell.

    Eight sites, and a Hamiltonian that obeys the centring translation
    ``(1/2, 1/2, 1/2)`` exactly, so :func:`.groups.realized_group` refuses it
    ``supercell``; its spectrum is the primitive cell's, folded.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 8 orbitals, 8 bands.
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
# 4.  P4_232 (SG 208) -- the DISCONNECTED elementary band representation
# ---------------------------------------------------------------------------
#: Parameter values of Fig. 3a of arXiv:1711.11045.
P4232_PARAMS = dict(t1=0.2, t2=0.3, t3=0.1, t4=0.08, t5=0.05, t6=0.02)

#: The four nearest-neighbour vectors ``delta_{1,2,3,4}`` of the paper, in
#: cartesian units of the cubic lattice constant a = 1.  They are the
#: *tetrahedral* half of the eight body-diagonal half-vectors joining the two
#: 2a sites; the paper draws exactly four of them in the inset to Fig. 3a, and
#: picking four rather than all eight is what removes the inversion centre.
P4232_NN_VECTORS = 0.5 * np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
)

#: Orbital positions in reduced coordinates: two 2a sites, two orbitals each.
#: Basis order is ``(site, orbital)`` = (A,dz2), (A,dx2-y2), (B,dz2), (B,dx2-y2),
#: so a paper term ``tau_i (x) sigma_j`` is ``numpy.kron(tau_i, sigma_j)``.
_P4232_ORB = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])


def p4232_bloch_hamiltonian(k_cart, t1=None, t2=None, t3=None, t4=None, t5=None, t6=None):
    r"""The paper's Bloch Hamiltonian, Eqs. (5)-(7): the ``H(k)`` of :func:`p4232_cano2018`.

    .. math::
        H_\mathbf{k} = t_1 f_1 \tau_x\sigma_0 + t_2 f_2 \tau_y\sigma_0
                     + t_3 (g_1 \tau_z\sigma_z + g_2 \tau_z\sigma_x)

        H^1_\mathbf{k} = t_4 f_1 \tau_y\sigma_y
                       + t_5 \tau_0 (g_2\sigma_z - g_1\sigma_x)
                       + t_6 g_3 \tau_z\sigma_0

    with :math:`f_1 = \sum_i \cos(\mathbf{k}\cdot\delta_i)`,
    :math:`f_2 = \sum_i \sin(\mathbf{k}\cdot\delta_i)`,
    :math:`g_1 = \cos k_x - \cos k_y`,
    :math:`g_2 = (\cos k_x + \cos k_y - 2\cos k_z)/\sqrt3`, and
    :math:`g_3 = \cos 2k_x \cos k_y - \cos 2k_y \cos k_x + \text{perm.}`
    (cyclic in :math:`k_x \to k_y \to k_z`).  Its phases carry the orbital
    positions, as pythtb's do, so it is :func:`p4232_cano2018`'s ``H(k)`` matrix
    for matrix.

    Parameters
    ----------
    k_cart : (3,) array_like
        ``k`` in Cartesian units with lattice constant ``a = 1``,
        ``2*pi*k_reduced``.
    t1, t2, t3, t4, t5, t6 : float, optional
        Default :data:`P4232_PARAMS`.

    Returns
    -------
    (4, 4) numpy.ndarray of complex
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
    """The spinless model of Cano et al., arXiv:1711.11045: a disconnected elementary band representation of P4_232 (SG 208).

    A simple cubic lattice, ``a = 1``, with the two Wyckoff 2a sites ``(0,0,0)`` and
    ``(1/2,1/2,1/2)``, each carrying ``d_z2`` and ``d_x2-y2``, together one irrep of
    the site group with time reversal; four bands.  The
    paper's generators, ``{C_2x|0}``, ``{C_3,111|0}`` and
    ``{C_2,110|1/2 1/2 1/2}``, and its 2a pair are those of SG 208.  The
    paper's terms in real space:

    * ``t1 f1 τ_x + t2 f2 τ_y``: hoppings between the two sites along all eight
      body-diagonal half-vectors, ``(t1+t2)/2`` along ``-δ_i`` and
      ``(t1-t2)/2`` along ``+δ_i``; ``t2 = 0`` restores inversion.
    * ``t4 f1 τ_y σ_y``: the same eight bonds with the orbital matrix
      ``-iσ_y``, amplitude ``t4/2``.
    * ``t3`` and ``t5``: hoppings within one sublattice to the six nearest cells,
      with ``σ_z`` and ``σ_x``.
    * ``t6 g3 τ_z``: hoppings to the twelve ``(±2, ±1, 0)`` cells.

    Parameters
    ----------
    t1, t2, t3, t4, t5, t6 : float, optional
        Default :data:`P4232_PARAMS`, the paper's Fig. 3a; ``t4 = t5 = t6 = 0``
        is the paper's bare ``H_k``, gapped and doubly degenerate at every ``k``.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 4 orbitals, 4 bands.
    """
    p = dict(P4232_PARAMS)
    for name, val in zip(("t1", "t2", "t3", "t4", "t5", "t6"), (t1, t2, t3, t4, t5, t6)):
        if val is not None:
            p[name] = val

    m = TBModel(Lattice(np.eye(3), _P4232_ORB.tolist(), periodic_dirs=[0, 1, 2]),
                spinful=False)

    def bond(amp, i, j, d):
        """set_hop with the *bond vector* d = r_j + R - r_i (pythtb convention I)."""
        if amp == 0.0:
            return
        R = np.asarray(d, float) - (_P4232_ORB[j] - _P4232_ORB[i])
        Ri = np.rint(R).astype(int)
        assert np.allclose(R, Ri, atol=1e-12), f"non-lattice bond {d!r}"
        m.set_hop(amp, i, j, Ri.tolist(), mode="add")

    def cos_diag(coef, i, R):
        """Add coef*cos(k.R) to H_ii."""
        bond(coef / 2.0, i, i, R)

    def cos_offdiag(coef, i, j, R):
        """Add coef*cos(k.R) to H_ij and H_ji (a real, symmetric sigma_x-type term)."""
        bond(coef / 2.0, i, j, R)
        bond(coef / 2.0, i, j, -np.asarray(R, float))

    # --- inter-site terms: t1, t2 (sigma_0) and t4 (-i*sigma_y) ---------------
    a_amp = (p["t1"] - p["t2"]) / 2.0
    b_amp = (p["t1"] + p["t2"]) / 2.0
    for d in P4232_NN_VECTORS:
        for i, j in ((0, 2), (1, 3)):          # sigma_0: orbital-preserving
            bond(a_amp, i, j, d)
            bond(b_amp, i, j, -d)
        for dd in (d, -d):                     # t4 * f1 * (-i sigma_y) block
            bond(-p["t4"] / 2.0, 0, 3, dd)
            bond(+p["t4"] / 2.0, 1, 2, dd)

    # --- intra-site terms: t3 (tau_z), t5 (tau_0), t6 (tau_z) ----------------
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
    """The unitary operations of P4_232 (SG 208), one per coset of the lattice translations.

    The closure of the paper's three generators, ``{C_2x|0}``, ``{C_3,111|0}``
    and ``{C_2,110|1/2 1/2 1/2}``.

    Returns
    -------
    list of (ndarray(3,3) int, ndarray(3,) float)
        ``(rotation, translation)`` of each: an integer rotation on reduced
        coordinates, and a translation in ``[0, 1)``.
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
    """A crystal whose space group is exactly P4_232 (SG 208): the 2a pair, decorated.

    The 2a pair alone is a body-centred cubic point set, of space group Im-3m
    (SG 229).  The paper adds atoms that make "a crystal field with the symmetry
    of P4_232" without giving them; these are the orbit of a generic point under
    SG 208, the smallest orbit no larger cubic group carries onto itself.

    Parameters
    ----------
    generic_site : (3,) array_like, optional
        A point on no special position, in reduced coordinates.

    Returns
    -------
    dict
        ``{"real_lattice", "positions", "typat"}``, for spglib or
        ``irrep.spacegroup.SpaceGroup.from_cell``: type 1 the 2a pair, type 2
        the decoration.
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
# 5.  lin3g (wallpaper p4mm) -- a fragile pair over an elementary band rep
# ---------------------------------------------------------------------------

def lin3g():
    """A spinful square layer, wallpaper group p4mm: a fragile pair above the ``s`` elementary band representation at 2c.

    Two orbitals at ``1a`` ``(0, 0)`` carry a four-band quartet, and two at ``2c``,
    ``(1/2, 0)`` and ``(0, 1/2)``, the spinful ``s`` elementary band
    representation; half the quartet lies below the gap, so the six occupied bands
    are the ``2c`` band representation and a fragile pair.  The model is Lin et
    al., Nat. Commun. **15**, 550 (2024), SM Sec. 3G, SEq. (3.55) and (3.59)
    (arXiv:2207.10099, (C55) and (C60)), with the values of the SFig. 7
    caption: ``t1 = 5.0``, ``t2 = 1.5``, ``vm = -1.5``, ``tPH = 0.1``,
    ``vs = 1.3``, ``vMz = 0.4``, ``vmu = 8.25 vm``, ``vC = 4.0`` and
    ``vCS = 0.45 vC``.  The orbital positions alone have the mirror through
    the layer, which the spin-orbit hoppings break, so the group is the
    hoppings' to measure.

    Returns
    -------
    pythtb.TBModel
        Spinful 2D model, 4 orbitals, 8 bands.
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

    # nearest-neighbour quartet hoppings (1a)
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
# 6.  the 3c orbit of a simple cubic cell -- sites that carry one another's
#     orbitals
# ---------------------------------------------------------------------------
#
# The three 3c sites of a simple cubic cell -- (0,1/2,1/2), (1/2,0,1/2) and
# (1/2,1/2,0), the centres of the cube's faces -- each carry a fourfold (or a
# -4) axis, along a different Cartesian direction, and the cell's own
# operations permute them.  So an operation carrying one site onto the next
# carries p_x at the first onto p_y at the second: the functions at one site
# are the image of those at another, not a copy of them.  Both models are
# group averages of random seeds, so each obeys its group exactly, and both are
# spinless.

#: The three ``3c`` sites of a simple cubic cell; site ``s`` has its fourfold
#: axis along the Cartesian direction ``s``.
_CUBIC_3C = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _cubic_rotations(proper_only=True):
    """The signed permutation matrices of ``O`` (24) or of ``T_d`` (24).

    ``proper_only`` keeps the determinant ``+1`` matrices, which are the
    rotations of ``O``; otherwise the ones with an even number of minus signs,
    which are ``T_d``.
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
    """``hops`` written in a basis rotated by ``angle`` inside every pair of orbitals.

    The orbitals ``2s`` and ``2s + 1`` are one site's pair, and the new pair is
    ``q = Vᵀ p`` with ``V`` one ``[[cos, -sin], [sin, cos]]`` per site, so
    ``H'(R) = Vᵀ H(R) V`` at every lattice vector: a change of orbital basis
    and nothing else.
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
    """``p`` orbitals at the three ``3c`` sites, averaged over ``ops``.

    ``orbitals`` is ``[(site, cartesian direction)]`` in model order.  Every
    pair within one lattice constant gets a random real hopping, and the whole
    set is averaged over ``ops`` with the orbitals carried as the vectors they
    are (``g p_a = sum_j R[j, a] p_j`` at the image site), so the model obeys
    ``ops`` exactly.  A nonzero ``angle`` then writes every site's pair in a
    basis rotated inside the pair by that angle (:func:`_rotated_in_pairs`),
    the same at every site; ``angle = 0`` leaves the hoppings untouched.
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
    """One ``p`` orbital per ``3c`` site of a simple cubic cell, along that site's own fourfold axis.

    ``p_x`` at ``(0,½,½)``, ``p_y`` at ``(½,0,½)`` and ``p_z`` at ``(½,½,0)``, with a
    random real hopping between every two orbitals within one lattice constant,
    averaged over the rotations of ``O``, the orbitals carried as the vectors
    they are.  An operation carrying one site onto another carries its ``p`` onto
    a different ``p``.  Averaged over ``O`` alone, the hoppings obey the whole of
    ``Pm-3m1'`` (#221.93), a ``p`` along the axis being odd under the inversion
    that fixes its site, and every site carries the ``A2u`` of its ``4/mmm``: the
    restriction of no one-dimensional representation of the whole group, which
    the hoppings therefore tell from an ``s`` orbital.

    Parameters
    ----------
    seed : int, optional
        The seed of the random hoppings.  Default ``11``.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 3 orbitals, 3 bands.
    """
    return _cubic_3c_p_model([(s, s) for s in range(3)],
                             _cubic_rotations(proper_only=True), seed)


def cubic_3c_perp_doublet(angle=0.0, seed=7):
    """The ``p`` pair perpendicular to each ``3c`` site's own axis, in a simple cubic cell: a composite band representation.

    ``(p_y, p_z)`` at ``(0,½,½)``, ``(p_x, p_z)`` at ``(½,0,½)`` and ``(p_x, p_y)``
    at ``(½,½,0)``, with a random real hopping between every two orbitals within
    one lattice constant, averaged over the operations of ``T_d``.  The three
    sites allow ``Pm-3m1'``; the hoppings obey ``T_d`` and no more, and the
    Hamiltonian obeys ``P-43m1'`` (#215.71), whose site group at ``3c`` is
    ``-42m``, its ``-4`` axis along the site's own.  Each pair is the ``E`` of
    that ``-42m``, carried by an operation onto a pair of different functions at
    the image site.  ``E`` at ``3c`` induces a composite band representation,
    ``T1 + T2`` at ``1b``, so the tables list no elementary band representation
    for it.

    Parameters
    ----------
    angle : float, optional
        A rotation of every site's pair within itself, ``q = Vᵀ p`` with
        ``V = [[cos, -sin], [sin, cos]]``, in radians: a change of orbital basis,
        which moves no group, site irrep or band content, and away from right
        angles writes the pairs in no real-harmonic basis.  Default ``0.0``.
    seed : int, optional
        The seed of the random hoppings.  Default ``7``.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 6 orbitals, 6 bands.
    """
    return _cubic_3c_p_model([(s, a) for s in range(3) for a in range(3) if a != s],
                             _cubic_rotations(proper_only=False), seed, angle=angle)
