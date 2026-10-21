"""A layer's presentations and its two environments: the slab a two-dimensional
model is read on, the decoupled stack that carries its bands at every ``k_z``,
the substrate under every site, and the check that the group on a substrate is
the part of the free-standing group that preserves the layer's normal.
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
#: No hopping crosses it, so no symmetry content can depend on it.
STACK_C = 8.0


def _is_presented_as_a_slab(model, c=SLAB_C) -> bool:
    """Whether ``model`` already *is* its own slab presentation.

    True for a ``dim_r = 3`` model whose non-periodic lattice vector lies along
    the layer's normal and is at least ``c`` long — deep enough for the model
    and a decorated cell to share one lattice, its vacuum included, which
    :func:`.._tables._assign_orbitals` requires — and for a ``dim_r = 3`` model with no
    non-periodic direction to measure.  A vector tilted off the normal is put
    on it by :func:`_slab_presentation`, however long it is.
    :func:`.._realized.realized_group` asks this before calling
    :func:`_slab_presentation`, which builds a new model in every case: the
    decision is made outside, so that a model handed back untouched is the
    original object and not a copy that happens to compare equal.
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
                       seed=_bloch.CHECK_SEED, params=None, record=None):
    r"""Internal to :func:`.._realized.realized_group`: a layer on a 3D slab lattice.

    A ``dim_r = 2`` model: the third lattice vector becomes ``c·ẑ`` and stays
    non-periodic, every orbital keeps ``z = 0``, and every hopping is copied
    with a zero third component.  A ``dim_k = 2`` Bloch phase never sees that
    direction, so ``H(k)`` comes out bit-identical — measured here rather
    than assumed.

    A ``dim_r = 3`` slab is copied with its non-periodic lattice vector along
    the layer's normal, stretched to ``c`` where it is shorter, every orbital
    kept where it is.  Stretching moves no atom: the reduced coordinate along
    the vector shrinks by exactly the factor the vector grows.  A vector tilted
    off the normal -- a slab cut from a crystal along one of the crystal's own
    lattice vectors has one -- is put on the normal at the same length,
    because a layer's group cannot depend on how that vector was chosen: there
    is no translation along it, while a symmetry finder reads the cell as a
    three-dimensional lattice on it, a lattice of copies of the layer stacked
    aslant.  Every orbital keeps its Cartesian position, so its reduced
    coordinates along the periodic directions move by ``Δ = h·t``, its reduced
    height ``h`` along the old vector times the vector's tilt ``t``, the
    in-plane part of the vector in those coordinates.  In pythtb's convention
    the Bloch phase carries ``e^{2πi k·τ}`` per orbital, so ``H(k)`` becomes
    ``Λ(k)† H(k) Λ(k)`` with ``Λ(k) = diag(e^{2πi k·Δ})``: a k-dependent gauge
    transformation, which moves no energy.  A layer whose orbitals all lie in
    the plane through the origin moves none and comes out bit-identical.

    A new model comes back every time.  ``record``, where one is handed in, is
    filled with what was measured: ``residual``, the largest difference between
    the presentation's ``H(k)`` and the model's, carried back through the move,
    over ``n_check`` random ``k`` (``0.0`` where ``check`` is off), and
    ``tilted``, ``None`` unless the vector was tilted off the normal, and then
    its ``tilt``, the largest move of an orbital's reduced coordinates
    (``max_move``), how many orbitals moved (``n_moved``), the largest
    difference of the two spectra over the same ``k`` (``spectrum_residual``)
    and a ``note``.

    Raises
    ------
    InternalError
        The model is not periodic in exactly two directions of a two- or
        three-dimensional cell: :func:`.._realized.realized_group` refuses a model periodic
        in fewer before it presents anything, and reads one periodic in three
        as its own cell.  Or the presentation changed the Hamiltonian: not
        bit-identical where no orbital moved, or beyond rounding once carried
        back through the move where some did.  A presentation changes nothing
        in exact arithmetic, so that is the package disagreeing with itself.
    """
    from pythtb import Lattice, TBModel

    params = dict(params or {})
    record = {} if record is None else record
    record["residual"], record["tilted"] = 0.0, None
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    if dim_r not in (2, 3) or len(per) != 2:
        raise InternalError(
            f"a slab presentation was asked for a model periodic in {len(per)} "
            f"direction{'' if len(per) == 1 else 's'} of a {dim_r}-dimensional cell. "
            "realized_group refuses a model periodic in fewer than two directions "
            "before it presents anything, and a fully periodic one is its own cell, so "
            "only a layer is presented as a slab.")
    if dim_r == 3:
        # Already a slab -- so this is a copy, with its vacuum stretched to `c`
        # where it is thinner.  That moves no atom (the reduced coordinate
        # shrinks by exactly the factor the vector grows) and cannot touch
        # `H(k)`, since a dim_k = 2 Bloch phase never sees the non-periodic
        # direction; a cell decorated on it is given the model's own
        # vacuum whatever the stretch (`_pad_cell`).  Measured below like every
        # other presentation built here, `grow = 1` -- the plain copy --
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
            # THE VECTOR TILTED OFF THE LAYER'S NORMAL, put on it.  The layer is
            # the same layer whichever non-periodic vector it was written with,
            # and the cell a symmetry finder reads must not stack its copies
            # aslant.  Every orbital stays where it is: the vector's in-plane
            # part, `tilt` on the periodic directions, is carried into each
            # orbital's in-plane coordinates in proportion to its height, and
            # the height is re-expressed on the normal, which the vector now is.
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
            a = np.asarray(_bloch.hamiltonian(model, k, params))
            b = np.asarray(_bloch.hamiltonian(out, k, params))
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
            if worst > gate:                        # pragma: no cover - a defect, raised
                raise InternalError(
                    ("this slab presentation, carried back through the move of its "
                     "orbitals, is not the model's Hamiltonian to rounding: "
                     f"max|Lambda H_slab(k) Lambda^dagger - H(k)| = {worst:.3g} over "
                     f"{n_check} random k. The hoppings were copied in order, the "
                     "non-periodic lattice vector was put on the layer's normal and "
                     "every orbital kept where it was, so in pythtb's convention H(k) "
                     "changed by the k-dependent gauge transformation "
                     "Lambda(k) = diag(e^{2 pi i k.Delta}) alone, Delta each orbital's "
                     "move." if moved else
                     "this slab presentation is not bit-identical to the model: "
                     f"max|H_slab(k) - H(k)| = {worst:.3g} over {n_check} random k. The "
                     "hoppings and the orbitals were copied in order, and at most the "
                     "non-periodic lattice vector and the reduced coordinate along it "
                     "were scaled, by reciprocal factors, or the vector put on the "
                     "layer's normal with every orbital in the plane through the "
                     "origin, so no atom moved and a dim_k = 2 Hamiltonian cannot "
                     "see it.")
                    + f" It is held to {gate:.3g}, and a presentation changes nothing, "
                    "so this is the package disagreeing with itself, not a property "
                    "of the model.")
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
        a = np.asarray(_bloch.hamiltonian(model, k, params))
        b = np.asarray(_bloch.hamiltonian(out, k, params))
        worst = float(np.abs(a - b).max())
        if worst != 0.0:
            raise InternalError(
                "the 3D presentation is not bit-identical to the 2D model: "
                f"max|H3(k) - H2(k)| = {worst:.3g} over {n_check} random k. The "
                "presentation is a relabelling of the same Hamiltonian and changes "
                "nothing, so this is the package disagreeing with itself, not a "
                "property of the model.")
        record["residual"] = worst
    return out


def stack_presentation(model, *, c=STACK_C, params=None):
    """The layer repeated along its normal with no hopping between the copies: a crystal with the layer's bands at every ``k_z``.

    The three-dimensional crystal a layer's rows at ``k_z = ½`` are measured
    on: the layer's own lattice and a third vector along its normal, every
    orbital at its own height, every hopping within one copy.  Its
    ``H(k, k_z)`` is the layer's ``H(k)`` up to the phases of the orbitals'
    heights, ``Λ(k_z)† H(k) Λ(k_z)`` with ``Λ`` diagonal, which each call
    measures, so its spectrum and symmetry are the layer's at every ``k_z``
    and ``c`` changes neither.

    Parameters
    ----------
    model : pythtb.TBModel
        A layer: periodic in two directions, in a cell of two or three.
    c : float, optional
        The vacuum between the copies, in the model's length units; the period
        is the layer's extent along its normal plus ``c``.  Default
        :data:`STACK_C`, ``8.0``.
    params : dict, optional
        Values for a model whose terms are given by parameter names.

    Returns
    -------
    pythtb.TBModel or Refusal
        The stack, periodic in three directions.  A ``Refusal`` in its place is
        falsy.

    Raises
    ------
    ..refusal.InternalError
        The stack's ``H`` is not the layer's up to the heights' phases: the
        package disagreeing with itself.

    Refusals
    --------
    ``not_a_layer``
        The model is not periodic in exactly two directions of a two- or
        three-dimensional cell.
    ``hopping_across_the_stack``
        A hopping runs along the non-periodic direction, so the model is not
        one decoupled layer.

    Examples
    --------
    >>> from pythtb.models import kane_mele                       # doctest: +SKIP
    >>> st = stack_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))   # doctest: +SKIP
    >>> st.lattice.dim_k, tuple(st.periodic_dirs)                  # doctest: +SKIP
    (3, (0, 1, 2))
    """
    from pythtb import Lattice, TBModel

    params = dict(params or {})
    dim_r = int(model.lattice.dim_r)
    per = tuple(int(d) for d in model.periodic_dirs)
    if int(model.lattice.dim_k) != 2 or len(per) != 2 or dim_r not in (2, 3):
        return Refusal(
            "not_a_layer",
            f"the decoupled stack is built from a layer -- two periodic directions in a "
            f"2D or 3D cell; this model has dim_r = {dim_r}, dim_k = "
            f"{int(model.lattice.dim_k)} and periodic directions {per}.",
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

    # every hopping must live inside the plane, structurally
    amps, i_idx, j_idx, R = _bloch.hopping_arrays(model)
    n_hop = int(R.shape[0])
    worst_hop = int(np.abs(R[:, stack_dir]).max()) if dim_r == 3 and n_hop else 0
    if worst_hop:
        return Refusal(
            "hopping_across_the_stack",
            f"{worst_hop} lattice units of hopping run along the non-periodic direction, "
            "so this is not one decoupled layer. Stacking it would either drop those "
            "hoppings or give H(k) a k_z it does not have; neither is the same "
            "Hamiltonian in another presentation.",
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
    # The new a1, a2 ARE the model's own, so the in-plane reduced coordinates
    # are copied rather than recomputed -- a projection would round them at
    # 1e-17 and the bit-identity gate below would (rightly) see it.  The only
    # correction is the in-plane part of a non-periodic third vector, which is
    # exactly zero for every presentation this package builds and is measured
    # rather than assumed.
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
    # zero on every row (gated above), so no two rows become one term
    R3 = np.zeros((n_hop, 3), dtype=int)
    R3[:, 0] = R[:, per[0]]
    R3[:, 1] = R[:, per[1]]
    out._append_hops(amps, i_idx, j_idx, R3)

    # the identity, measured on every call, on the k-set every presentation
    # here is measured on
    rng = np.random.default_rng(_bloch.CHECK_SEED)
    k2 = rng.uniform(-0.5, 0.5, size=(_bloch.CHECK_K, 2))
    a = np.asarray(_bloch.hamiltonian(model, k2, params))
    stack = _bloch.evaluator(out, params)
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
                f"against {held:.3g}, over {_bloch.CHECK_K} random k, Lambda the phases "
                "of the orbitals' heights. The presentation is the same Hamiltonian and "
                "changes nothing but that gauge, so this is the package disagreeing with "
                "itself, not a property of the model.")
    return out


def _layer_on_stack(group, assignment, stack) -> tuple:
    r"""A layer's group and orbital sites, carried onto the layer's decoupled stack.

    :func:`stack_presentation` repeats the layer along its normal with no
    hopping between the copies, and an operation of the layer carries each copy
    onto itself as it carries the layer, so the stack's group is the layer's:
    the same operations, each the same Cartesian rotation and translation,
    written in the stack's cell, and the same sites, each operation moving them
    as it moves the layer's.  Nothing is identified again.  The stack's cell
    keeps the layer's in-plane lattice vectors and puts its normal at another
    length, reversed where the slab's vacuum is its middle axis (the stack's
    cell is right-handed), so every operation keeps its reduced form, up to the
    order and the sense in which the two cells list their axes, and the tables'
    setting is carried in that same order: a k-point at ``k_z = ½`` of the
    layer's group is one of the stack's own zone.  Both cells are written about
    the layer's middle plane, and a site off it is carried at its height from
    that plane, so the lattice vector each operation moves a site by is read
    again in the stack's cell.

    ``group`` and ``assignment`` are the group a layer's representation was
    built in, on its slab cell, and the orbital sites in it; ``stack`` is that
    layer's decoupled stack.  Returns ``(group, assignment)`` on the stack's
    cell: three periodic directions and none along a vacuum.

    Raises
    ------
    ..refusal.InternalError
        Where the two cells are not one layer's, or an operation, a site or a
        lattice shift does not read as one of the stack's cell.  The stack is
        built from the layer's own lattice and orbitals, so either is a defect
        here.
    """
    old = np.asarray(group.lattice, dtype=float)
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
            f"#{group.number_str}: the decoupled stack's cell is not the layer's cell "
            "with its normal at another length or reversed -- the cell the group was "
            f"measured on has lattice {np.round(old, 6).tolist()} and the stack's "
            f"{np.round(new, 6).tolist()} -- so its operations cannot be carried "
            "onto the stack. The stack is built from the layer's own lattice, so "
            "this is a defect here, not a property of the model.")
    inverse = np.linalg.inv(carry)
    padded = [int(d) for d in group.vacuum_dirs]

    def _miss(x):
        x = np.asarray(x, dtype=float)
        return float(np.abs(x - np.rint(x)).max()) if x.size else 0.0

    def _centred(x):
        """Reduced coordinates, the vacuum one taken about the middle plane, in [-½, ½)."""
        x = np.array(x, dtype=float, copy=True)
        if padded and x.size:
            # a whole cell subtracted, so a coordinate already there is untouched
            x[..., padded] -= np.floor(x[..., padded] + 0.5)
        return x

    def _placed(x):
        """Positions carried onto the stack's cell at their heights, wrapped into the cell."""
        y = _centred(x) @ carry.T
        if y.size:
            y[..., 2] = y[..., 2] % 1.0
        return y

    moved, worst = [], 0.0
    for sym in group.symmetries:
        rotation = carry @ np.asarray(sym.rotation, dtype=float) @ inverse
        own = _centred(np.asarray(sym.translation, dtype=float))
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
    sites = _placed(group.site_positions)
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
            f"#{group.number_str}: an operation of the layer, or a lattice vector it "
            f"moves a site by, reads in the decoupled stack's cell {worst:.3g} away from "
            "one of that cell. The stack shares the layer's in-plane lattice and every "
            "operation of a layer keeps its plane, so this is a defect here, not a "
            "property of the model.")
    ref = axes @ np.asarray(group._refUC, dtype=float)
    shift = axes @ np.asarray(group._shiftUC, dtype=float)
    sg = copy.copy(group._sg)
    for cached in ("lattice_inv", "reciprocal_lattice", "translations_cart"):
        sg.__dict__.pop(cached, None)
    sg.real_lattice, sg.symmetries, sg.positions = new, moved, sites
    sg.refUC, sg.shiftUC = ref, shift
    on_stack = replace(group, symmetries=list(moved), lattice=new, site_positions=sites,
                       periodic_dirs=(0, 1, 2), vacuum_dirs=(), dim_k=3, dim_r=3,
                       _refUC=ref, _shiftUC=shift, _sg=sg)
    placed = replace(assignment, lattice_shift=np.rint(shifts).astype(int),
                     site_positions=placed_sites)
    return on_stack, placed


def _off_plane_direction(model3d) -> int:
    """The one non-periodic lattice direction of a slab cell (``2`` if none is)."""
    per = tuple(int(d) for d in model3d.periodic_dirs)
    return next((d for d in range(3) if d not in per), 2)


def _substrate_decorations(model3d, typat=None, *, dz=DECORATION_DZ) -> tuple:
    r"""Internal to :func:`.._realized.realized_group`: the substrate under every site.

    A layer lying on something has one side different from the other, and the
    way to say that to a symmetry finder is to put the substrate in the cell:
    one non-orbital atom of its own species at the site's own height minus
    ``dz``, under every one of the model's sites.  Every operation with
    ``R[2,2] = −1`` would then carry a shadow onto the layer, so every such
    operation is gone — ``σ_h``, the in-plane twofolds, ``S_n`` and
    three-dimensional inversion with them — while the in-plane group and the
    layer's own geometry, buckling included, are untouched.  What is left is
    the subgroup of the free-standing layer group that preserves the layer's
    normal.  Under
    *every* site rather than once at the origin, because a single shadow would
    also remove every operation that does not fix the origin, which is a
    statement about the coordinates the model was written in.

    Returns
    -------
    tuple
        ``((position, label), ...)`` — the decoration sequence
        :func:`_decorated_cell` takes, one entry per distinct site.
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
    r"""Internal to :func:`.._realized.realized_group`: the cell carrying the crystal field.

    The point set the group is identified on: the model's orbitals, plus any
    species that state a distortion the positions do not show.  Geometry only —
    it never carries a Hamiltonian, and the model is not modified.

    ``environment`` chooses the base: ``"substrate"`` (the default) puts one
    shadow species under every site (:func:`_substrate_decorations`), leaving
    the subgroup that preserves the layer's normal, and ``"free-standing"``
    puts nothing under
    the layer.  ``decorations`` adds on top of that base, and in 3D — where a
    model is its own cell and there is no base — it is the whole mechanism: one
    atom at the strengthened bond's midpoint lowers diamond to ``R-3m``, which
    is the group the distorted Fu-Kane-Mele Hamiltonian obeys.

    Raises
    ------
    ValueError
        When ``typat`` does not carry one entry per orbital.

    Returns
    -------
    (pythtb.TBModel, list)
        The decorated cell and the species list to identify it with.  Pass the
        *model's own* ``typat`` to :func:`.._tables._assign_orbitals`, not this one.
    """
    from pythtb import Lattice, TBModel

    norb = int(model3d.norb)
    labels = [0] * norb if typat is None else list(typat)
    if len(labels) != norb:
        raise ValueError(f"typat must have one entry per orbital ({norb}), "
                         f"got {len(labels)}")
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
    # the model's own orbitals lead the cell, and a cell built on a cell keeps
    # the count of the model it was built on: the vacuum is theirs
    setattr(cell, _OWN_ORBITALS, int(getattr(model3d, _OWN_ORBITALS, norb)))
    return cell, labels + extra


def _environment_record(free_group, sub_group) -> dict:
    r"""The two environments of one layer, compared — the consistency gate.

    **A substrate means a subgroup.**  It removes every operation that flips the
    layer's side and nothing else, so the group of a layer on a substrate is the
    ``R[2,2] = +1`` **subgroup** of the free-standing layer group of the *same*
    crystal — the same point set, the same coordinates, one species added below
    it.  That is a statement two independent selections can be held to, and this
    holds them to it, operation for operation:

    (a) every operation of the substrate group is an operation of the
        free-standing group that preserves the layer's normal, matched by rotation and translation modulo the
        lattice (:func:`_operation_key`) — anything else is ``missing``;
    (b) every operation of the free-standing group that preserves the layer's
        normal is in the
        substrate group — anything else is ``extra``;
    (c) both cosets, unitary and antiunitary, are compared the same way, since
        the key carries ``time_reversal``;
    (d) no operation of the substrate group flips the side (``flipping``), and
        no operation of either group mixes the plane with the vacuum direction
        (``unmapped``), which would not be a layer operation at all.

    Nothing is projected and every operation carries its own lift, which is what
    makes the subgroup statement exact rather than approximate: a buckled
    honeycomb on a substrate keeps precisely the six operations of its
    ``p-3m1`` that preserve the layer's normal, and reads ``p3m1``, not the ``p6mm`` that the in-plane image
    of a point set flattened onto one plane would report.

    Returns the record; ``agrees`` is ``False`` on any failure and ``None``
    where the two cells cannot be compared at all (no single vacuum direction).
    """
    lp = tuple(int(d) for d in free_group.vacuum_dirs)
    wp = tuple(int(d) for d in sub_group.vacuum_dirs)
    head = {"free_standing": {"number_str": str(free_group.number_str),
                              "name": str(free_group.name),
                              "size": int(free_group.size)},
            "substrate": {"number_str": str(sub_group.number_str),
                          "name": str(sub_group.name), "size": int(sub_group.size)}}
    if len(lp) != 1 or lp != wp:
        return dict(head, agrees=None, outcome=None, rows=(), missing=(), extra=(),
                    flipping=(), unmapped=(),
                    note=("the two cells do not share a single non-periodic direction "
                          f"(free-standing {lp}, substrate {wp}), so which operations "
                          "preserve the layer's side is not defined and the two "
                          "environments were not compared"))
    padded = int(lp[0])
    free_lat = np.asarray(free_group.lattice, dtype=float)
    sub_lat = np.asarray(sub_group.lattice, dtype=float)

    unmapped, expected = [], {}
    for s in free_group.symmetries:
        side = _side_preserving(s, padded)
        if side is None:
            unmapped.append({"group": "free-standing", "ind": int(s.ind),
                             "name": _record_name(s, free_lat)})
        elif side:
            expected[_operation_key(s)] = s

    got, flipping = {}, []
    for s in sub_group.symmetries:
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
    # key's own tolerance -- compared as numbers, so two translations a rounding
    # apart that the keys' six decimals would put on two sides of a boundary
    # (a group written about an origin fitted to noisy centres has such
    # translations) are still one operation
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
    """The two environments contradicting the subgroup statement — raised, never returned.

    A substrate removes every operation that flips the layer's side and nothing
    else, so the substrate group of a layer is the subgroup of its
    free-standing group that preserves the layer's normal, and two selections of one Hamiltonian that
    contradict it are this package disagreeing with itself.  Where one of them
    returned the candidate the tie rule keeps among several its measurement
    could not choose between, a failure that another pair of the candidates
    does not have is that choice and not a contradiction, and it is an outcome
    of the cross-check rather than this.  ``held`` is how many pairs of the two
    environments' candidates were held to the statement, the pair returned
    among them.

    Raises
    ------
    ..refusal.InternalError
        Always, naming the operations each side carries that the other does not.
    """
    free, sub = record["free_standing"], record["substrate"]

    def _list(rows):
        return ", ".join(f"{r['name']} (#{r['ind']})" for r in rows[:4]) + (
            ", ..." if len(rows) > 4 else "")

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
                    "the layer's side (R[2,2] = -1), which a substrate removes by "
                    "construction: " + _list(record["flipping"]))
    if record.get("unmapped"):
        bits.append(f"{len(record['unmapped'])} operations mix the plane with the "
                    "vacuum direction and are not layer operations at all: "
                    + _list(record["unmapped"]))
    raise InternalError(
        f"the free-standing realized group ({free['name']} #{free['number_str']}) and the "
        f"one on a substrate ({sub['name']} #{sub['number_str']}) do not describe the "
        f"same Hamiltonian, the operations counted as {_PER_COSET}: "
        + "; ".join(bits)
        + ". A substrate removes every operation that flips the layer's side and "
        "nothing else, so the substrate group is the R[2,2] = +1 subgroup of the "
        "free-standing layer group of the same crystal -- the two selections agreeing "
        "is not a coincidence to be checked politely, it is what makes either of them "
        "the group of this model."
        + (f" Where the tie rule chose, the subgroups it did not keep were held to it "
           f"too, {int(held) - 1} more pair{'' if int(held) == 2 else 's'} of the two "
           "environments' candidates, and it holds for none." if int(held) > 1 else "")
        + " Neither is returned: two selections of one "
        "Hamiltonian contradicting it is a defect here, not a measurement about the "
        "model, and `realized_group(..., cross_check=False)` shows each environment's "
        "realized group on its own.")
