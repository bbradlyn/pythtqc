"""A layer's presentations and its two environments.

The slab cell a layer is read on; its decoupled stack, which carries the
layer's bands at every ``k_z``; the substrate put under every site; and the
comparison of the group on a substrate with the part of the free-standing group
that preserves the layer's normal.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from . import _bloch
from .refusal import InternalError, Refusal
from ._tolerance import _CARRY_TOL, _PRESENTATION_RELATIVE
from ._text import _PER_COSET
from ._spacegroup import _OWN_ORBITALS
from ._seitz import _record_name
from ._subgroups import _operation_key, _side_preserving

__all__ = ["SLAB_C", "DECORATION_DZ", "STACK_C", "stack_presentation"]


# --------------------------------------------------------------------------
# a layer's presentations: the slab it is read on, and its decoupled stack
# --------------------------------------------------------------------------

#: Vacuum between a layer and its periodic image in the slab presentation, in
#: the model's own length units: only the cell sees it, never ``H(k)``.
SLAB_C = 30.0

#: How far under each site the substrate sits, in reduced coordinates of the
#: non-periodic lattice vector: any depth off the plane removes the same
#: operations, the ones that flip the layer's side.
DECORATION_DZ = 0.05

#: The vacuum between the copies of the decoupled stack, in the model's own
#: length units: the period is the layer's extent along its normal plus this.
#: No hopping crosses it, so neither the bands nor the group depend on it.
STACK_C = 8.0


def _is_presented_as_a_slab(model, c=SLAB_C) -> bool:
    """Whether ``model`` is already its own slab presentation.

    ``c`` is the least length of a slab's non-periodic lattice vector, in the
    units of the model's lattice vectors.  True for a three-dimensional cell
    not periodic in exactly two directions, and for one whose non-periodic
    lattice vector is exactly perpendicular to the periodic ones and at least
    ``c`` long.  False otherwise: for a two-dimensional cell, and for a
    non-periodic vector tilted off the normal or shorter than ``c``.
    :func:`.._realized.realized_group` reads a model for which it is
    true as it stands, and presents any other with :func:`_slab_presentation`.
    """
    if int(model.lattice.dim_r) != 3:
        return False
    per = tuple(int(d) for d in model.periodic_dirs)
    if len(per) != 2:
        return True
    off = next(d for d in range(3) if d not in per)
    lat = np.asarray(model.lat_vecs, dtype=float)
    if np.any(lat[list(per)] @ lat[off] != 0.0):
        return False
    length = float(np.linalg.norm(lat[off]))
    return length <= 0.0 or length >= float(c)


def _slab_presentation(model, *, c=SLAB_C, check=True, n_check=_bloch.CHECK_K,
                       seed=_bloch.CHECK_SEED, record=None):
    r"""A layer on a three-dimensional slab cell, with the same Hamiltonian.

    A two-dimensional cell gains the third lattice vector ``c·ẑ``, which is
    not periodic.  Every orbital is put at height zero and every hopping's
    lattice vector is given a zero third component, so ``H(k)`` is unchanged
    bit for bit.  A three-dimensional cell keeps every orbital's Cartesian
    position, and its non-periodic lattice vector is put along the layer's
    normal and stretched to the length ``c`` where it is shorter.  Where that
    vector was tilted off the normal, each orbital's reduced coordinates in the
    plane move by its reduced coordinate along the old vector times the
    vector's in-plane part, in reduced coordinates of the periodic directions.
    ``H(k)`` then becomes ``Λ(k)† H(k) Λ(k)``, with
    ``Λ(k) = diag(exp(2πi k·Δ))`` and ``Δ`` each orbital's move.  A new model
    is returned in every case.

    Parameters
    ----------
    model : pythtb.TBModel
        A layer: periodic in exactly two directions of a two- or
        three-dimensional cell.
    c : float, default 30.0
        The least length of the non-periodic lattice vector, in the units of
        the model's lattice vectors.
    check : bool, default True
        Whether the new ``H(k)`` is compared with the model's: bit for bit
        where no orbital moved, and through ``Λ(k)`` to
        ``_PRESENTATION_RELATIVE`` times ``max|H|`` where some did.
    n_check : int, default 8
        The number of random ``k`` the comparison is made at.
    seed : int, default 1729
        The seed the random ``k`` are drawn with.
    record : dict, optional
        Filled, where one is handed in, with ``residual``, the largest
        difference the comparison found (``0.0`` without ``check``), and
        ``tilted``: ``None``, or for a tilted vector a mapping of ``tilt``,
        the vector's in-plane part in reduced coordinates of the periodic
        directions; ``max_move``, the largest move of an orbital in reduced
        coordinates; ``n_moved``, the number of orbitals moved;
        ``spectrum_residual``, the largest difference of the two spectra
        (``0.0`` without ``check``); and a ``note``.

    Returns
    -------
    pythtb.TBModel
        The slab, periodic in the model's two periodic directions.

    Raises
    ------
    ..refusal.InternalError
        Where the model is not periodic in exactly two directions of a two- or
        three-dimensional cell, which :func:`.._realized.realized_group` never
        hands in, or where the comparison finds ``H(k)`` changed.
    """
    from pythtb import Lattice, TBModel

    record = {} if record is None else record
    record["residual"], record["tilted"] = 0.0, None
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    if dim_r not in (2, 3) or len(per) != 2:
        raise InternalError(
            f"a slab cell was asked for a model periodic in {len(per)} "
            f"direction{'' if len(per) == 1 else 's'} of a {dim_r}-dimensional cell, "
            "which is not a layer. Report it, with the model and the call.")
    if dim_r == 3:
        # Already a slab, so this is a copy, its vacuum stretched to `c` where
        # it is thinner.  That moves no atom, since the reduced coordinate
        # shrinks by the factor the vector grows, and it leaves `H(k)` alone,
        # since a dim_k = 2 Bloch phase never reads the non-periodic direction.
        # A cell decorated on it is given the model's own vacuum whatever the
        # stretch (`_pad_cell`).  The copy is compared with the model below, as
        # every presentation built here is, the plain copy (`grow = 1`)
        # included.
        lat = np.asarray(model.lat_vecs, dtype=float)
        orb = np.asarray(model.orb_vecs, dtype=float)
        orb3 = orb.copy()
        lat3 = lat.copy()
        off = next(d for d in range(3) if d not in per)
        length = float(np.linalg.norm(lat[off]))
        grow = float(c) / length if 0.0 < length < float(c) else 1.0
        plane = lat[list(per)]
        overlap = plane @ lat[off]
        move = None
        if np.any(overlap != 0.0):
            # The vector tilted off the layer's normal is put on it.  The layer
            # is the same layer whichever non-periodic vector it was written
            # with, and the cell a symmetry finder reads must not stack its
            # copies aslant.  Every orbital stays where it is: the vector's
            # in-plane part, `tilt` on the periodic directions, is carried into
            # each orbital's in-plane coordinates in proportion to its height,
            # and the height is re-expressed on the normal, which the vector
            # now is.
            normal = np.cross(plane[0], plane[1])
            normal = normal / np.linalg.norm(normal)
            rise = float(np.dot(lat[off], normal))
            if rise < 0.0:
                normal, rise = -normal, -rise       # keep the model's own sense of up
            tilt = np.linalg.solve(plane @ plane.T, overlap)
            depth = length * grow
            lat3[off] = normal * depth
            move = np.outer(orb[:, off], tilt)
            for j, d in enumerate(per):
                orb3[:, d] = orb[:, d] + move[:, j]
            orb3[:, off] = orb[:, off] * (rise / depth)
        elif grow != 1.0:
            lat3[off] = lat[off] * grow
            orb3[:, off] = orb3[:, off] / grow
        out = TBModel(Lattice(lat3, orb3.tolist(), periodic_dirs=list(per)),
                      spinful=(int(model.nspin) == 2))
        out.set_onsite(model.onsite, mode="set")
        # the model's own hopping table, row for row and in one pass, so the
        # copy sums exactly the terms the model sums
        out._append_hops(*_bloch.hopping_arrays(model))
        moved = move is not None and bool(np.any(move))
        worst = spectrum = 0.0
        gate = 0.0
        if check:
            k = np.random.default_rng(seed).uniform(-0.5, 0.5,
                                                    size=(int(n_check), len(per)))
            a = np.asarray(_bloch.hamiltonian(model, k))
            b = np.asarray(_bloch.hamiltonian(out, k))
            if moved:
                # the model's H(k) carried through the move, Λ(k)† H(k) Λ(k):
                # the presentation's, to the rounding of the moved phases
                phase = np.exp(2j * np.pi * (k @ move.T))
                if int(model.nspin) == 2:
                    phase = np.repeat(phase, 2, axis=1)
                carried = np.conj(phase)[:, :, None] * a * phase[:, None, :]
                worst = float(np.abs(carried - b).max())
                gate = _PRESENTATION_RELATIVE * max(float(np.abs(a).max()), 1e-300)
                spectrum = float(np.abs(np.linalg.eigvalsh(a) - np.linalg.eigvalsh(b)).max())
            else:
                worst = float(np.abs(a - b).max())
            if worst > gate:                        # pragma: no cover, a defect that raises
                raise InternalError(
                    ("the slab cell's Hamiltonian, carried back through the move of its "
                     "orbitals, is not the model's: "
                     f"max|Lambda H_slab(k) Lambda^dagger - H(k)| = {worst:.3g} over "
                     f"{n_check} random k, with Lambda(k) = diag(e^{{2 pi i k.Delta}}) "
                     "and Delta each orbital's move," if moved else
                     "the slab cell's Hamiltonian is not the model's: "
                     f"max|H_slab(k) - H(k)| = {worst:.3g} over {n_check} random k,")
                    + f" held to {gate:.3g}. Report it, with the model and the call.")
        record["residual"] = worst
        if move is not None:
            n_moved = int(np.count_nonzero(np.abs(move).max(axis=1) > 0.0))
            biggest = float(np.abs(move).max(initial=0.0))
            record["tilted"] = {
                "tilt": tuple(float(x) for x in tilt),
                "max_move": biggest, "n_moved": n_moved,
                "spectrum_residual": spectrum,
                "note": (
                    "the slab's non-periodic lattice vector was tilted off the layer's "
                    "normal, its in-plane part "
                    f"({', '.join(f'{float(x):.6g}' for x in tilt)}) on the periodic "
                    "directions, and it is put on the normal, every orbital kept where "
                    "it is: a layer's group cannot depend on how that vector was "
                    "chosen, since there is no translation along it. "
                    + (f"{n_moved} of {int(model.norb)} orbitals lie off the plane "
                       "through the origin, and each one's reduced coordinates in the "
                       "plane moved by its reduced height times that tilt, at most "
                       f"{biggest:.3g}: a k-dependent gauge transformation of H(k), "
                       "which moves no energy"
                       + (f" (measured, {spectrum:.3g})." if check else ".")
                       if n_moved else
                       "Every orbital lies in the plane through the origin, so none "
                       "moved and H(k) is unchanged bit for bit."))}
        return out

    lat2 = np.asarray(model.lat_vecs, dtype=float)
    lat3 = np.zeros((3, 3))
    lat3[:2, :2] = lat2
    lat3[2] = [0.0, 0.0, float(c)]
    orb3 = np.zeros((int(model.norb), 3))
    orb3[:, :2] = np.asarray(model.orb_vecs, dtype=float)

    out = TBModel(Lattice(lat3, orb3.tolist(), periodic_dirs=[0, 1]),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    # the model's own hopping table, row for row and in one pass, each lattice
    # vector given a zero third component
    amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    R3 = np.zeros((R.shape[0], 3), dtype=int)
    R3[:, :dim_r] = R
    out._append_hops(amps, i_idx, j_idx, R3)

    if check:
        k = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(int(n_check), 2))
        a = np.asarray(_bloch.hamiltonian(model, k))
        b = np.asarray(_bloch.hamiltonian(out, k))
        worst = float(np.abs(a - b).max())
        if worst != 0.0:
            raise InternalError(
                "the slab cell's Hamiltonian is not the two-dimensional model's bit for "
                f"bit: max|H_slab(k) - H(k)| = {worst:.3g} over {n_check} random k. "
                "Report it, with the model and the call.")
        record["residual"] = worst
    return out


def stack_presentation(model, *, c=STACK_C, params=None):
    """A layer repeated along its normal, with no hopping between the copies.

    The crystal's lattice vectors are the layer's two and a third along the
    layer's normal.  The third has the length of the layer's extent along the
    normal, from its lowest orbital to its highest, plus ``c``.  Each orbital
    keeps its height along the normal.  The crystal's Bloch Hamiltonian is
    ``H(k, k_z) = Λ(k_z)† H(k) Λ(k_z)``, with ``H(k)`` the layer's.
    ``Λ(k_z)`` is diagonal, with the entry ``exp(2πi k_z z)`` for an orbital
    at height ``z`` from the layer's middle plane, in units of the third
    lattice vector.  So the crystal's bands at every ``k_z`` are the layer's
    bands at ``k``, whatever ``c``.

    Parameters
    ----------
    model : pythtb.TBModel
        A model periodic in two directions, in a cell of two or three
        dimensions.
    c : float, default 8.0
        The width of the vacuum between the copies, in the length units of the
        model's lattice vectors.  The default is ``pythtqc.groups.STACK_C``.
    params : dict, optional
        The scalar value of each parameter of the model, keyed by name:
        ``{"lam": 0.3}``.

    Returns
    -------
    pythtb.TBModel or Refusal
        The crystal, periodic in three directions, with each parameter of the
        model set to its value in ``params``.  A ``Refusal`` in its place is
        falsy and names what stopped the call.

    Examples
    --------
    Kane-Mele without its Rashba term, repeated along its normal, is periodic
    in three directions:

    >>> from pythtb.models import kane_mele                       # doctest: +SKIP
    >>> st = stack_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))   # doctest: +SKIP
    >>> st.lattice.dim_k, tuple(st.periodic_dirs)                  # doctest: +SKIP
    (3, (0, 1, 2))

    Refusals
    --------
    ``not_a_layer``
        the model is not periodic in exactly two directions of a two- or
        three-dimensional cell.
    ``hopping_across_the_stack``
        a hopping's lattice vector has a component along the non-periodic
        lattice vector of a three-dimensional cell, so the layer is coupled to
        its images along that vector.

    Raises
    ------
    ValueError
        A name in ``params`` that is not a parameter of the model, a parameter
        of the model that ``params`` leaves out, or a value that is not a
        number.

    See Also
    --------
    symmetry_indicator : the symmetry-indicator class of a band set.  A
        layer's little-group representations at the high-symmetry points with
        ``k_z = 1/2``, which the layer's own Bloch Hamiltonian does not reach,
        are measured on this crystal.
    """
    from pythtb import Lattice, TBModel

    model = _bloch.bound(model, params)
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    if int(model.lattice.dim_k) != 2 or len(per) != 2 or dim_r not in (2, 3):
        return Refusal(
            "not_a_layer",
            f"the model has dim_r = {dim_r}, dim_k = {int(model.lattice.dim_k)} and "
            f"periodic directions {per}. Pass a layer, a model periodic in two "
            "directions of a two- or three-dimensional cell.",
            value=float(int(model.lattice.dim_k)), threshold=2.0,
            context={"dim_r": dim_r, "dim_k": int(model.lattice.dim_k),
                     "periodic_dirs": per})

    lat = np.asarray(model.lat_vecs, dtype=float)
    plane = np.zeros((2, 3))
    plane[:, :dim_r] = lat[list(per)]
    # the normal that makes the stack's cell right-handed, as pythtb requires of
    # every cell, whichever of a slab's axes is its vacuum: a slab whose vacuum is
    # its middle axis has that vector opposite to this normal, and a cell built
    # along the vacuum vector would be left-handed
    normal = np.cross(plane[0], plane[1])
    normal = normal / np.linalg.norm(normal)
    stack_dir = None
    if dim_r == 3:
        stack_dir = next(d for d in range(3) if d not in per)

    # every hopping joins two orbitals of one copy
    amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    n_hop = int(R.shape[0])
    worst_hop = int(np.abs(R[:, stack_dir]).max()) if dim_r == 3 and n_hop else 0
    if worst_hop:
        return Refusal(
            "hopping_across_the_stack",
            f"a hopping's lattice vector has a component of {worst_hop} along the "
            "non-periodic lattice vector. Check the model's hoppings along that "
            "direction.",
            value=float(worst_hop), threshold=0.0,
            context={"n_hoppings": n_hop, "stacking_direction": stack_dir})

    # every orbital at its own height along the normal: the layer's extent
    # there, plus the vacuum c, is the period
    cart = np.asarray(model.orb_vecs, dtype=float) @ lat
    if cart.shape[1] == 2:
        cart = np.column_stack([cart, np.zeros(cart.shape[0])])
    height = cart @ normal
    top = float(height.max()) if height.size else 0.0
    bottom = float(height.min()) if height.size else 0.0
    extent = top - bottom
    period = extent + float(c)

    lat3 = np.zeros((3, 3))
    lat3[:2] = plane
    lat3[2] = normal * period
    # The new a1 and a2 are the model's own, so the in-plane reduced
    # coordinates are copied rather than recomputed: a projection would round
    # them at 1e-17, and the comparison below holds k_z = 0 bit for bit.  The
    # one correction is for the in-plane part of a slab's non-periodic third
    # vector, zero for every slab this package builds: each orbital's in-plane
    # coordinates move by its reduced coordinate along that vector times the
    # vector's in-plane part.  The comparison below does not carry H(k)
    # through that move, so where an orbital off the plane moves it raises.
    orb = np.asarray(model.orb_vecs, dtype=float)
    orb3 = np.zeros((int(model.norb), 3))
    orb3[:, :2] = orb[:, list(per)]
    if stack_dir is not None:
        overlap = plane @ np.asarray(lat[stack_dir], dtype=float)
        if np.any(overlap != 0.0):
            orb3[:, :2] += np.outer(orb[:, stack_dir],
                                    np.linalg.solve(plane @ plane.T, overlap))
    if extent > 0.0:
        # each orbital's height from the layer's middle plane, the frame the
        # layer's group is identified in; a flat layer's are all exactly zero
        orb3[:, 2] = (height - 0.5 * (top + bottom)) / period

    out = TBModel(Lattice(lat3, orb3.tolist(), periodic_dirs=[0, 1, 2]),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    # the model's own hopping table, row for row and in one pass, each lattice
    # vector written on the two in-plane directions; the component dropped is
    # zero on every row (refused above), so no two rows become one term
    R3 = np.zeros((n_hop, 3), dtype=int)
    R3[:, 0] = R[:, per[0]]
    R3[:, 1] = R[:, per[1]]
    out._append_hops(amps, i_idx, j_idx, R3)

    # H_stack(k, k_z) = Lambda^dagger H(k) Lambda, measured on every call, on
    # the k-set every presentation here is measured on
    rng = np.random.default_rng(_bloch.CHECK_SEED)
    k2 = rng.uniform(-0.5, 0.5, size=(_bloch.CHECK_K, 2))
    a = np.asarray(_bloch.hamiltonian(model, k2))
    stack = _bloch.evaluator(out)
    phase = 2j * np.pi * orb3[:, 2]
    if int(model.nspin) == 2:
        phase = np.repeat(phase, 2)
    # the phases round where the heights are not zero, and nowhere else
    gate = 0.0 if extent == 0.0 else _PRESENTATION_RELATIVE * float(np.abs(a).max())
    for kz in (0.0, 0.25, 0.5, *rng.uniform(-0.5, 0.5, size=3)):
        k3 = np.concatenate([k2, np.full((_bloch.CHECK_K, 1), float(kz))], axis=1)
        b = np.asarray(stack(k3))
        lam = np.exp(phase * float(kz))
        d = float(np.abs(np.conj(lam)[None, :, None] * a * lam[None, None, :] - b).max())
        held = 0.0 if float(kz) == 0.0 else gate
        if d > held:
            raise InternalError(
                "the decoupled stack is not the layer: max|H_stack(k, k_z) - "
                f"Lambda^dagger H_layer(k) Lambda| = {d:.3g} at k_z = {float(kz)}, "
                f"held to {held:.3g}, over {_bloch.CHECK_K} random k, with "
                "Lambda = diag(e^{2 pi i k_z z}) and z each orbital's reduced height. "
                "Report it, with the model and the call.")
    return out


def _layer_on_stack(group, assignment, stack) -> tuple:
    r"""A layer's group and orbital sites, carried onto the layer's decoupled stack.

    ``group`` is the group a layer's representation was built in, on its slab
    cell, ``assignment`` the orbitals' sites in it, and ``stack`` that layer's
    decoupled stack (:func:`stack_presentation`).  Nothing is identified
    again.  Each operation keeps its Cartesian rotation and translation and is
    written in the stack's cell, whose axes are the slab cell's up to their
    order, their sense and the length of the normal.  Each site keeps its
    height from the layer's middle plane, and the lattice vector each operation
    moves a site by is read again in the stack's cell.  The tables' setting is
    carried with the same order of the axes, so a k-point at ``k_z = 1/2`` of
    the layer's group is one of the stack's zone.  Returns
    ``(group, assignment)`` on the stack's cell, periodic in three directions.

    Raises
    ------
    ..refusal.InternalError
        Where the stack's cell is not the slab cell with its normal at another
        length or reversed, or where an operation, a site or a lattice vector
        does not read as one of the stack's cell to ``_CARRY_TOL``.
    """
    old = np.asarray(group._lattice, dtype=float)
    new = np.asarray(stack.lat_vecs, dtype=float)
    # reduced coordinates of the layer's cell, to Cartesian, to the stack's
    carry = np.linalg.solve(new.T, old.T)
    # the order of the axes alone, the normal's length divided out
    order = carry / np.linalg.norm(carry, axis=0)[None, :]
    axes = np.rint(order)
    if (float(np.abs(order - axes).max()) > _CARRY_TOL
            or not np.array_equal(np.abs(axes).sum(axis=0), np.ones(3))
            or not np.array_equal(np.abs(axes).sum(axis=1), np.ones(3))
            or any(abs(float(carry[:, int(d)] @ axes[:, int(d)]) - 1.0) > _CARRY_TOL
                   for d in group.periodic_dirs)):
        raise InternalError(
            f"{group.name} #{group.number_str}: the decoupled stack's lattice "
            f"{np.round(new, 6).tolist()} is not the layer's lattice "
            f"{np.round(old, 6).tolist()} with its normal at another length or "
            "reversed. Report it, with the model and the call.")
    inverse = np.linalg.inv(carry)
    padded = [int(d) for d in group._vacuum_dirs]

    def _miss(x):
        x = np.asarray(x, dtype=float)
        return float(np.abs(x - np.rint(x)).max()) if x.size else 0.0

    def _centered(x):
        """``x`` with each coordinate along the vacuum brought into [-1/2, 1/2)."""
        x = np.array(x, dtype=float, copy=True)
        if padded and x.size:
            # a whole cell subtracted, so a coordinate already there is untouched
            x[..., padded] -= np.floor(x[..., padded] + 0.5)
        return x

    def _placed(x):
        """The slab's reduced positions ``x`` in the stack's cell, at their heights.

        The third coordinate is taken modulo 1.
        """
        y = _centered(x) @ carry.T
        if y.size:
            y[..., 2] = y[..., 2] % 1.0
        return y

    moved, worst = [], 0.0
    for sym in group._symmetries:
        rotation = carry @ np.asarray(sym.rotation, dtype=float) @ inverse
        own = _centered(np.asarray(sym.translation, dtype=float))
        translation = carry @ own
        # an operation of a layer moves nothing along its normal, so its
        # translation reads alike whatever length the normal has
        worst = max(worst, _miss(rotation), _miss(translation - axes @ own))
        new_sym = copy.copy(sym)
        for cached in ("rotation_cart", "translation_cart", "lattice_inv",
                       "reciprocal_lattice", "det_cart", "det", "rotation_inv",
                       "is_identity"):
            new_sym.__dict__.pop(cached, None)
        new_sym.rotation = np.rint(rotation).astype(int)
        new_sym.translation = translation
        new_sym.real_lattice = new
        moved.append(new_sym)
    sites = _placed(group._site_positions)
    placed_sites = (None if assignment.site_positions is None
                    else _placed(assignment.site_positions))
    if placed_sites is None:
        shifts = np.asarray(assignment.lattice_shift, dtype=float) @ carry.T
    else:
        # the lattice vector each operation moves a site by, R tau_i + t - tau_g(i),
        # read again in the stack's cell, where the sites sit at their heights
        perm = np.asarray(assignment.permutations, dtype=int)
        shifts = np.stack([placed_sites @ np.asarray(s.rotation, dtype=float).T
                           + np.asarray(s.translation, dtype=float) - placed_sites[perm[p]]
                           for p, s in enumerate(moved)])
    worst = max(worst, _miss(shifts))
    if worst > _CARRY_TOL:
        raise InternalError(
            f"{group.name} #{group.number_str}: an operation of the layer, or a lattice "
            "vector it moves a site by, written in the decoupled stack's cell, is "
            f"{worst:.3g} from the nearest operation or lattice vector of that cell. "
            "Report it, with the model and the call.")
    ref = axes @ np.asarray(group._refUC, dtype=float)
    shift = axes @ np.asarray(group._shiftUC, dtype=float)
    sg = copy.copy(group._sg)
    for cached in ("lattice_inv", "reciprocal_lattice", "translations_cart"):
        sg.__dict__.pop(cached, None)
    sg.real_lattice, sg.symmetries, sg.positions = new, moved, sites
    sg.refUC, sg.shiftUC = ref, shift
    on_stack = replace(group, _symmetries=list(moved), _lattice=new, _site_positions=sites,
                       periodic_dirs=(0, 1, 2), _vacuum_dirs=(), _dim_k=3, _dim_r=3,
                       _refUC=ref, _shiftUC=shift, _sg=sg)
    placed = replace(assignment, lattice_shift=np.rint(shifts).astype(int),
                     site_positions=placed_sites)
    return on_stack, placed


def _off_plane_direction(model3d) -> int:
    """The index of the non-periodic lattice vector of the 3D cell ``model3d``.

    The first is returned where several are non-periodic, and ``2`` where
    none is.
    """
    per = tuple(int(d) for d in model3d.periodic_dirs)
    return next((d for d in range(3) if d not in per), 2)


def _substrate_decorations(model3d, typat=None, *, dz=DECORATION_DZ) -> tuple:
    r"""The substrate under a layer: one atom of a species of its own below every site.

    ``model3d`` is the layer's slab and ``typat`` the species of each orbital,
    ``None`` for one species.  Each distinct orbital position gets one atom
    ``dz`` below it, in reduced coordinates of the non-periodic lattice vector.
    The atoms' species is one more than the largest integer species, or
    ``"__substrate__"`` where the species are not integers.  Every operation
    that turns the layer over carries such an atom above the layer, so the
    cell keeps the subgroup of the free-standing group that preserves the
    layer's normal.  Returns ``((position, label), ...)``, the decorations
    :func:`_decorated_cell` takes.
    """
    norb = int(model3d.norb)
    labels = [0] * norb if typat is None else list(typat)
    if all(isinstance(t, (int, np.integer)) for t in labels):
        shadow = max(int(t) for t in labels) + 1
    else:                                # labels are not numeric: use a reserved name
        shadow = "__substrate__"
    off = _off_plane_direction(model3d)
    step = np.zeros(3)
    step[off] = float(dz)
    seen, out = [], []
    for row in np.asarray(model3d.orb_vecs, dtype=float):
        p = np.asarray(row, dtype=float).reshape(-1).copy()
        if any(float(np.abs((p - q + 0.5) % 1.0 - 0.5).max()) <= 1e-9 for q in seen):
            continue
        seen.append(p)
        out.append(([float(x) for x in (p - step)], shadow))
    return tuple(out)


def _decorated_cell(model3d, typat=None, *, dz=DECORATION_DZ, environment=None,
                    decorations=None):
    r"""The cell a group is identified on: the model's orbitals and any added species.

    ``model3d`` is the model presented in 3D, a slab or a crystal, and
    ``typat`` the species of each orbital, ``None`` for one species.  On a
    slab, ``environment`` sets the base: ``"substrate"``, or ``None``, puts
    the atoms of :func:`_substrate_decorations` ``dz`` below every site, and
    ``"free-standing"`` puts nothing.  A crystal periodic in three directions
    has no base.  ``decorations``, a sequence of ``(position, label)`` in
    reduced coordinates, is added on top of the base.  The cell carries no
    Hamiltonian, and ``model3d`` is not changed.  Returns ``(cell, species)``:
    the cell, its orbitals the model's followed by the added atoms, and the
    species of each point.  :func:`.._tables._assign_orbitals` takes the
    model's own ``typat``, not these species.

    Raises
    ------
    ValueError
        Where ``typat`` does not hold one entry per orbital.
    """
    from pythtb import Lattice, TBModel

    norb = int(model3d.norb)
    labels = [0] * norb if typat is None else list(typat)
    if len(labels) != norb:
        raise ValueError(f"realized_group(): the length of typat= is {len(labels)}, and "
                         f"the model has {norb} orbitals. Pass one entry per orbital.")
    extra_atoms = () if decorations is None else tuple(decorations)
    # A fully periodic cell has no side to make different: it is its own cell in
    # every environment, and only `decorations=` adds to it.
    periodic = tuple(int(d) for d in model3d.periodic_dirs)
    is_slab = any(d not in periodic for d in range(3))
    base_environment = "substrate" if environment is None else environment
    base = (_substrate_decorations(model3d, labels, dz=dz)
            if is_slab and base_environment == "substrate" else ())

    pos = np.asarray(model3d.orb_vecs, dtype=float).tolist()
    extra = []
    for p, lab in base + extra_atoms:
        pos.append([float(x) for x in np.asarray(p, dtype=float).reshape(-1)])
        extra.append(lab)
    cell = TBModel(Lattice(np.asarray(model3d.lat_vecs, dtype=float), pos,
                           periodic_dirs=list(model3d.periodic_dirs)), spinful=False)
    # the model's own orbitals lead the cell, and the count of them is kept,
    # since the vacuum is measured from them alone; a cell built on a cell
    # keeps the count of the model it was built on
    setattr(cell, _OWN_ORBITALS, int(getattr(model3d, _OWN_ORBITALS, norb)))
    return cell, labels + extra


def _environment_record(free_group, sub_group) -> dict:
    r"""The substrate group held against the free-standing one, operation by operation.

    ``free_group`` and ``sub_group`` are one layer's groups, identified
    free-standing and on a substrate.  A substrate removes every operation
    that turns the layer over and nothing else, so the substrate group is the
    subgroup of the free-standing group that preserves the layer's normal.
    Operations are matched by rotation, by translation modulo the lattice to
    ``1e-6``, and by whether they are combined with time reversal
    (:func:`.._subgroups._operation_key`).

    Returns a mapping.  ``free_standing`` and ``substrate`` hold each group's
    number, name and size.  ``rows`` holds each substrate operation that
    preserves the normal, with the free-standing one it matches, or ``None``.
    ``missing`` holds the substrate operations with no match, and ``extra``
    the free-standing operations that preserve the normal and have no match.
    ``flipping`` holds the substrate operations that turn the layer over, and
    ``unmapped`` the operations of either group that mix the plane with the
    non-periodic direction.  ``agrees`` is ``True`` where those four are
    empty, with ``outcome`` ``"agree"`` and a ``note`` saying so, and
    ``False`` otherwise.  It is ``None``, nothing compared, where the two
    cells do not share a single non-periodic direction.
    """
    lp = tuple(int(d) for d in free_group._vacuum_dirs)
    wp = tuple(int(d) for d in sub_group._vacuum_dirs)
    head = {"free_standing": {"number_str": str(free_group.number_str),
                              "name": str(free_group.name),
                              "size": int(free_group._size)},
            "substrate": {"number_str": str(sub_group.number_str),
                          "name": str(sub_group.name), "size": int(sub_group._size)}}
    if len(lp) != 1 or lp != wp:
        return dict(head, agrees=None, outcome=None, rows=(), missing=(), extra=(),
                    flipping=(), unmapped=(),
                    note=("the two cells do not share a single non-periodic direction "
                          f"(free-standing {lp}, substrate {wp}), so which operations "
                          "preserve the layer's side is not defined and the two "
                          "environments were not compared"))
    padded = int(lp[0])
    free_lat = np.asarray(free_group._lattice, dtype=float)
    sub_lat = np.asarray(sub_group._lattice, dtype=float)

    unmapped, expected = [], {}
    for s in free_group._symmetries:
        side = _side_preserving(s, padded)
        if side is None:
            unmapped.append({"group": "free-standing", "ind": int(s.ind),
                             "name": _record_name(s, free_lat)})
        elif side:
            expected[_operation_key(s)] = s

    got, flipping = {}, []
    for s in sub_group._symmetries:
        side = _side_preserving(s, padded)
        if side is None:
            unmapped.append({"group": "substrate", "ind": int(s.ind),
                             "name": _record_name(s, sub_lat)})
            continue
        if not side:
            flipping.append({"ind": int(s.ind), "name": _record_name(s, sub_lat)})
            continue
        got[_operation_key(s)] = s

    # each substrate operation paired with the free-standing one it is: the same
    # rotation and coset, and the same translation modulo the lattice to the
    # key's own tolerance.  The translations are compared as numbers, so two a
    # rounding apart that the keys' six decimals would put on two sides of a
    # boundary are still one operation; a group written about an origin fitted
    # to noisy centers has such translations
    def _same(a, b):
        if a[0] != b[0] or a[2] != b[2]:
            return False
        d = (np.asarray(got[a].translation, dtype=float).reshape(-1)
             - np.asarray(expected[b].translation, dtype=float).reshape(-1))
        return float(np.abs(d - np.rint(d)).max()) < 1e-6

    paired, free_keys = {}, sorted(expected)
    for key in sorted(got):
        match = next((k for k in free_keys if k not in paired.values() and _same(key, k)),
                     None)
        if match is not None:
            paired[key] = match

    rows = []
    for key in sorted(got):
        s = got[key]
        f = expected.get(paired[key]) if key in paired else None
        rows.append({"substrate": int(s.ind),
                     "free_standing": None if f is None else int(f.ind),
                     "name": _record_name(s, sub_lat),
                     "rotation": key[0], "translation": key[1],
                     "time_reversal": bool(s.time_reversal)})
    missing = tuple({"ind": int(got[k].ind), "name": _record_name(got[k], sub_lat)}
                    for k in sorted(got) if k not in paired)
    extra = tuple({"ind": int(expected[k].ind), "name": _record_name(expected[k], free_lat)}
                  for k in sorted(set(expected) - set(paired.values())))
    agrees = not (missing or extra or flipping or unmapped)
    return dict(head, agrees=bool(agrees), outcome=("agree" if agrees else None),
                rows=tuple(rows), missing=missing, extra=extra,
                flipping=tuple(flipping), unmapped=tuple(unmapped),
                note=("the substrate group is exactly the subgroup of the free-standing "
                      "layer group that preserves the layer's normal -- every operation with "
                      "R[2,2] = +1 and no other, in both cosets")
                if agrees else "")


def _raise_environment_contradiction(record, held=1):
    """Raise the two environments' contradiction of the subgroup statement.

    The statement is that the substrate group is the subgroup of the
    free-standing group that preserves the layer's normal.  ``record`` is what
    :func:`_environment_record` returns with ``agrees`` ``False``.  ``held``
    is how many pairs of the two environments' candidate groups were held to
    the statement, the pair returned among them; it exceeds one where the tie
    rule chose among candidates.  It is called only where no pair satisfies
    the statement: where another pair does, the disagreement is the tie
    rule's choice, and the cross-check records it.

    Raises
    ------
    ..refusal.InternalError
        Always, naming up to four operations of each kind that breaks the
        statement.
    """
    free, sub = record["free_standing"], record["substrate"]

    def _list(rows):
        return ", ".join(f"{r['name']} (#{r['ind']})" for r in rows[:4]) + (
            ", and others" if len(rows) > 4 else "")

    bits = []
    if record["missing"]:
        bits.append(f"{len(record['missing'])} substrate operations are not "
                    "operations of the free-standing group that preserve the layer's "
                    "normal: "
                    + _list(record["missing"]))
    if record["extra"]:
        bits.append(f"{len(record['extra'])} operations of the free-standing group "
                    "that preserve the layer's normal are missing from the substrate "
                    "group: "
                    + _list(record["extra"]))
    if record["flipping"]:
        bits.append(f"{len(record['flipping'])} operations of the substrate group flip "
                    "the layer's side (R[2,2] = -1): " + _list(record["flipping"]))
    if record.get("unmapped"):
        bits.append(f"{len(record['unmapped'])} operations mix the layer's plane with "
                    "its non-periodic direction: "
                    + _list(record["unmapped"]))
    raise InternalError(
        f"the layer's free-standing group {free['name']} #{free['number_str']} and its "
        f"group on a substrate {sub['name']} #{sub['number_str']} disagree, the "
        f"operations counted as {_PER_COSET}: "
        + "; ".join(bits)
        + "."
        + (f" The other candidates of a tie were compared too, {int(held) - 1} more "
           f"pair{'' if int(held) == 2 else 's'}, and no pair agrees." if int(held) > 1 else "")
        + " Neither is returned. Report it, with the model and the call. "
        "realized_group(..., cross_check=False) returns the group of the environment "
        "asked for alone.")
