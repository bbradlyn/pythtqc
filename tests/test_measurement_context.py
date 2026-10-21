"""The machinery wave P1 added, and the one thing it may not do: change an answer.

Two objects are measured here.

:class:`pythtqc.groups._MeasurementContext`
    ``H(k)`` on the detection mesh and the detector's solution space per
    **in-plane k-action**, shared by every measurement one
    :func:`~pythtqc.groups.realized_group` call makes.  The tests are: a
    ``_validate_group`` with a context returns what one without a context
    returns, field for field and array for array; the context is rebuilt by
    exactly the things the answer depends on — the Hamiltonian, the k-set, the
    model parameters, ``tol``, ``window``, ``solver`` — and by nothing else; and
    the two steps of the pipeline that change the Hamiltonian, the position pass
    and the hopping projection, both invalidate it in a real call.

``pythtqc.groups``' identification memo
    a bounded, process-level LRU of :func:`~pythtqc.groups._group_of` answers.
    The tests are its hit/miss accounting, that a hit is the object the first
    call built, that every flag the answer depends on is in the key, and that
    the bound is honoured.

Nothing here is a performance assertion: wall clock is not a property of the
package.  What is asserted is that the sharing is invisible in the answer.
"""

import numpy as np
import pytest

from conftest import patch_everywhere
from pythtb import Lattice, TBModel
from pythtb.models import kane_mele

import tqc_models
from pythtqc import groups as tqg
from pythtqc.refusal import Refusal
from pythtqc import _layers
from pythtqc import _detector
from pythtqc import _measure
from pythtqc import _realized
from pythtqc import _spacegroup


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _copy_model(model):
    """A bit-identical **new object**: the cross-check's other reading, in one line."""
    out = TBModel(model.lattice, spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    for hop in model.hoppings:
        out.set_hop(hop["amplitude"], hop["from_orbital"], hop["to_orbital"],
                    list(hop.get("lattice_vector", [0] * int(model.lattice.dim_r))),
                    mode="add")
    return out


def _nudged_hopping(model, delta=1e-3):
    """The same model with its first hopping amplitude moved."""
    out = TBModel(model.lattice, spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    for i, hop in enumerate(model.hoppings):
        amp = np.asarray(hop["amplitude"], dtype=complex)
        out.set_hop(amp + (delta if i == 0 else 0.0), hop["from_orbital"],
                    hop["to_orbital"],
                    list(hop.get("lattice_vector", [0] * int(model.lattice.dim_r))),
                    mode="add")
    return out


def _moved_orbital(model, delta=1e-3):
    """The same hoppings on centres one of which has been displaced."""
    orb = np.asarray(model.orb_vecs, dtype=float).copy()
    orb[0, 0] += delta
    out = TBModel(Lattice(model.lat_vecs, orb.tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(model.onsite, mode="set")
    for hop in model.hoppings:
        out.set_hop(hop["amplitude"], hop["from_orbital"], hop["to_orbital"],
                    list(hop.get("lattice_vector", [0] * int(model.lattice.dim_r))),
                    mode="add")
    return out


def _two_site_cell(shift=0.0):
    """Two spinless sites at general positions of an oblique layer: ``P2/m1' #10.43``.

    A cell of this little symmetry is identified in milliseconds, so every
    question the memo is asked below costs next to nothing; ``shift`` moves the
    second site, which is a different cell.
    """
    lat = Lattice([[1.0, 0.0], [0.3, 1.2]], [[0.0, 0.0], [0.37 + shift, 0.21]],
                  periodic_dirs=[0, 1])
    return TBModel(lat, spinful=False)


#: ``name: (builder, group keywords, operations refuted)`` — the goldens the two
#: routes are compared on.  The first two obey every operation of their cell; the
#: staggered Kane-Mele slab, its two sublattices read as one species, refutes half
#: of its ``P6/mmm1'``, so the two routes' refuted operations are compared too.
GOLDENS = {
    "lin3g": (lambda: _layers._slab_presentation(tqc_models.lin3g()),
              dict(typat=[0, 0, 1, 1]), 0),
    "kane_mele": (lambda: _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.0)), {}, 0),
    "km_staggered": (lambda: _layers._slab_presentation(tqc_models.kane_mele_z2("odd")),
                     {}, 24),
}


# ---------------------------------------------------------------------------
# 1.  a context changes nothing about what _validate_group measures
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_a_context_changes_nothing_validate_group_measures(name):
    """Same survivors, same residuals, same solution spaces — array for array.

    The context serves a k-action class the space some earlier call solved, and
    the claim that licenses it is that the solve is a deterministic function of
    the model, the k-set and ``tol``/``window``/``solver``.  This measures that
    claim on the goldens: every field of :class:`~pythtqc.groups.ValidatedGroup`
    is compared between a call with a context and one without, including the
    per-operation record, the whole stored basis of every kept operation and,
    on the staggered slab, every operation the Hamiltonian refutes.
    """
    build, kw, n_refuted = GOLDENS[name]
    model = build()
    assert not isinstance(model, Refusal), str(model)
    group = _spacegroup._group_of(model, **kw)
    assert group, str(group)

    plain = _measure._validate_group(model, group, nk=4)
    ctx = _measure._MeasurementContext()
    shared = _measure._validate_group(model, group, nk=4, context=ctx)

    assert shared.indices == plain.indices
    assert shared.closed == plain.closed
    assert shared.k_action_classes == plain.k_action_classes
    assert shared.verdicts == plain.verdicts
    assert shared._nullspace_dims == plain._nullspace_dims
    assert shared.tol == plain.tol and shared.n_kpts == plain.n_kpts
    assert set(shared.residuals) == set(plain.residuals)
    for ind, value in plain.residuals.items():
        assert shared.residuals[ind] == value or (
            np.isinf(value) and np.isinf(shared.residuals[ind]))
    assert len(shared.discrepancies) == len(plain.discrepancies) == n_refuted
    for a, b in zip(shared.discrepancies, plain.discrepancies):
        assert a["ind"] == b["ind"] and a["refuted_by"] == b["refuted_by"]
    assert set(shared._records) == set(plain._records)
    for ind, rec in plain._records.items():
        got = shared._records[ind]
        for field, want in rec.items():
            assert got[field] == want, f"{ind}.{field}"
    assert set(shared._bases) == set(plain._bases)
    for ind, basis in plain._bases.items():
        assert np.array_equal(np.asarray(shared._bases[ind]), np.asarray(basis)), ind
    assert set(shared._intertwiners) == set(plain._intertwiners)
    for ind, rep in plain._intertwiners.items():
        assert np.array_equal(np.asarray(shared._intertwiners[ind]), np.asarray(rep)), ind

    # a FRESH context serves nothing: every class this call poses is a miss, and
    # the grid is built once.  The sharing *within* one call is
    # `_validate_group`'s own `solved_class` dict and needs no context at all —
    # the context's serving is measured by the next test, where a second call
    # can hit.  (Until 2026-09-09 these two lines were read as "solved once and
    # served to the rest"; `hits` is 0 here, so nothing was served.)
    assert (ctx.hits, ctx.misses) == (0, len(plain.k_action_classes))
    assert ctx.rebuilds == 1
    assert len(ctx.spaces) == len(plain.k_action_classes)


def test_a_second_measurement_of_the_same_group_is_all_hits(monkeypatch):
    """The descent's own case: the same model and mesh measured twice.

    A descent re-measures a *subgroup* on the same model and the same mesh, so
    every class it poses was already posed by the group above it.  Measured here
    in its simplest form — the identical call, twice — where the second one
    solves nothing and returns what the first one did.

    "Solves nothing" is a **counted** :func:`~pythtqc.symmetry._find_intertwiners`:
    the second call adds zero solves.  "Returns what the first one did" is object
    identity on the stored bases — the served space *is* the stored one, because
    ``bases[ind] = space.basis`` keeps it by reference.  Neither claim can rest on
    ``hits``/``misses``: they are incremented by the dict lookup itself and say
    nothing about whether what it returned was used (found 2026-09-09 by the
    honesty audit — with the lookup's result thrown away the counters do not
    move and a ``np.array_equal`` on the bases still passes).
    """
    model = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))
    group = _spacegroup._group_of(model)
    assert group, str(group)
    ctx = _measure._MeasurementContext()

    solves = []
    _find_intertwiners = _detector._find_intertwiners

    def counted(*args, **kwargs):
        solves.append(1)
        return _find_intertwiners(*args, **kwargs)

    patch_everywhere(monkeypatch, "_find_intertwiners", counted)

    first = _measure._validate_group(model, group, nk=4, context=ctx)
    solved = int(ctx.misses)
    assert solved == len(first.k_action_classes) > 1
    assert len(solves) == solved, "the first call solves each class exactly once"
    hits_before = int(ctx.hits)

    second = _measure._validate_group(model, group, nk=4, context=ctx)
    assert len(solves) == solved, (
        f"the second measurement solved {len(solves) - solved} problems again")
    assert ctx.misses == solved, "nothing was solved a second time"
    assert ctx.hits == hits_before + len(first.k_action_classes)
    assert ctx.rebuilds == 1, "the model did not move, so nothing was rebuilt"
    assert second.indices == first.indices
    for ind, basis in first._bases.items():
        assert second._bases[ind] is basis, f"#{ind}: a new array, so a new solve"


# ---------------------------------------------------------------------------
# 2.  invalidation: what rebuilds the context, and what does not
# ---------------------------------------------------------------------------

def test_the_context_is_rebuilt_by_exactly_what_the_answer_depends_on():
    """Everything the solve is a function of is in the key, and nothing else is.

    The Hamiltonian, the k-set, the parameters, ``tol``, ``cut``, ``window``,
    ``solver``, the ceiling of the detector's memory and the floor below which
    its nullspace keeps a candidate -- each checked by moving it alone.
    ``cut`` is one of them: two calls differing only in where the nullspace is
    cut have different solution spaces, so a key without it would serve the
    first call's spaces to the second.  So is the ceiling: an operation the
    detector's own budget refuses on an exactly degenerate spectrum is solved
    under a higher one.  So is the floor, which a declared precision sets at the
    gate while its cut is relative to the Hamiltonian's magnitude.

    A bit-identical **copy** of the model is deliberately *not* a rebuild: that
    is the cross-check's other reading, whose slab presentation is a different
    object with the same ``H(k)`` at every ``k``, and sharing with it is the
    point of keying on content rather than on identity.  The other side of the
    same key: the *same* object changed in place is a rebuild, and the grid
    bound to it is the Hamiltonian it now has.
    """
    model = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))
    grid = _measure._detection_grid(int(model.lattice.dim_k), 4)
    ctx = _measure._MeasurementContext()

    first = ctx.bind(model, grid, {}, tol=1e-8, cut=None, window=None, solver="eigen")
    assert ctx.rebuilds == 1
    ctx.remember(("probe",), "a solution space")

    # the same request, again: nothing moves
    assert ctx.bind(model, grid, {}, tol=1e-8, cut=None, window=None,
                    solver="eigen") is first
    assert ctx.rebuilds == 1 and ctx.spaces

    # a bit-identical copy: still nothing moves
    twin = _copy_model(model)
    assert twin is not model
    assert ctx.bind(twin, grid, {}, tol=1e-8, cut=None, window=None,
                    solver="eigen") is first
    assert ctx.rebuilds == 1, "a copy of the same Hamiltonian is the same Hamiltonian"
    assert ctx.spaces, "and its solution spaces are the same spaces"

    # everything the solve depends on, one at a time
    changed = [
        ("a moved hopping", dict(model=_nudged_hopping(model))),
        ("a moved centre", dict(model=_moved_orbital(model))),
        ("a different mesh", dict(k_pts=_measure._detection_grid(
            int(model.lattice.dim_k), 6))),
        ("a different tol", dict(tol=1e-6)),
        ("a declared cut", dict(cut=1e-6)),
        ("a declared window", dict(window=1e-3)),
        ("a different solver", dict(solver="stacked")),
        ("a different parameter", dict(params={"p": 0.1})),
        ("a raised ceiling on the detector's memory", dict(max_bytes=2e9)),
        ("a floor a declared precision sets at the gate", dict(floor=1e-8)),
    ]
    for what, override in changed:
        ctx = _measure._MeasurementContext()
        base = ctx.bind(model, grid, {}, tol=1e-8, cut=None, window=None,
                        solver="eigen")
        ctx.remember(("probe",), "a solution space")
        call = dict(model=model, k_pts=grid, params={}, tol=1e-8, cut=None,
                    window=None, solver="eigen", max_bytes=None, floor=None)
        call.update(override)
        again = ctx.bind(call["model"], call["k_pts"], call["params"],
                         tol=call["tol"], cut=call["cut"], window=call["window"],
                         solver=call["solver"], max_bytes=call["max_bytes"],
                         floor=call["floor"])
        assert again is not base, what
        assert ctx.rebuilds == 2, what
        assert ctx.spaces == {}, f"{what}: the spaces went with the grid"

    # the same object, its first hopping moved in place: its content is new, so
    # the grid is rebuilt and sums the Hamiltonian as it now is
    same = _copy_model(model)
    ctx = _measure._MeasurementContext()
    base = ctx.bind(same, grid, {}, tol=1e-8, cut=None, window=None, solver="eigen")
    before = base.full().copy()
    hop = same.hoppings[0]
    same.set_hop(np.asarray(hop["amplitude"]) + 1e-3, hop["from_orbital"],
                 hop["to_orbital"], list(hop.get("lattice_vector", [0, 0, 0])),
                 mode="set")
    again = ctx.bind(same, grid, {}, tol=1e-8, cut=None, window=None, solver="eigen")
    assert again is not base and ctx.rebuilds == 2, "a model changed in place is new content"
    assert np.array_equal(again.full(), same.hamiltonian(k_pts=grid, flatten_spin_axis=True))
    assert not np.array_equal(again.full(), before)

    # the parameter case `repr` could not see: numpy truncates an array's repr past
    # 1000 elements, so two arrays differing only beyond that point repr alike.  Found
    # 2026-09-09 -- on the repr-keyed context they bound to ONE grid.  The key hashes
    # dtype, shape and bytes now; an equal copy still shares, a different value rebuilds.
    a = np.zeros(2000)
    b = a.copy()
    b[1000] = 1.0
    assert repr(a) == repr(b), "the premise: repr cannot tell them apart"
    ctx = _measure._MeasurementContext()
    one = ctx.bind(model, grid, {"p": a}, tol=1e-8, cut=None, window=None,
                   solver="eigen")
    assert ctx.bind(model, grid, {"p": a.copy()}, tol=1e-8, cut=None, window=None,
                    solver="eigen") is one, "an equal array is the same request"
    assert ctx.bind(model, grid, {"p": b}, tol=1e-8, cut=None, window=None,
                    solver="eigen") is not one
    assert ctx.rebuilds == 2, "two parameter arrays that differ past index 1000 are two H(k)"


def test_the_grid_is_only_served_for_the_mesh_it_was_built_on():
    """``grid_for`` checks the k-set byte for byte rather than trusting the caller.

    The lattice measurement reads ``H`` off the detector's own grid, and the
    thing that makes that sound is that the two k-sets are the same one.
    """
    model = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.0))
    grid = _measure._detection_grid(2, 4)
    ctx = _measure._MeasurementContext()
    assert ctx.grid_for(model, grid) is None, "nothing is bound yet"
    bound = ctx.bind(model, grid, {}, tol=1e-8, cut=None, window=None, solver="eigen")
    assert ctx.grid_for(model, grid) is bound
    assert ctx.grid_for(model, grid.copy()) is bound, \
        "the same points, a different array"
    assert ctx.grid_for(model, _measure._detection_grid(2, 6)) is None
    assert ctx.grid_for(model, grid[:-1]) is None
    # and it is the model's own grid, not merely a grid on the same points: a
    # bit-identical copy shares the SPACES (same H(k)) but is not the object the
    # lattice measurement is about
    assert ctx.grid_for(_copy_model(model), grid) is None


@pytest.mark.slow
def test_the_position_pass_and_the_projection_both_invalidate_the_context():
    """The two steps of the chain that change ``H``, in one real call.

    A model whose centres are off by 1e-9 and whose hoppings are off by 1e-9
    goes down both branches: the position pass moves the centres (a new model),
    and the hopping projection projects the Hamiltonian (another new model).
    The record says both ran, and the context was rebuilt for each of them.

    The position pass runs **before** the first measurement, so on a fresh
    context its adjusted cell would arrive as the first bind and not as an
    invalidation — a ``rebuilds >= 2`` would then be satisfied by one bind plus
    one rebuild and could not tell the two steps apart (found 2026-09-09 by the
    honesty audit).  The context is therefore bound on the Hamiltonian **as
    handed in** first, which makes both steps invalidations: three binds exactly,
    with the ends named — the last Hamiltonian bound is the projected one the
    answer is about, and it is not the one handed in.
    """
    rng = np.random.default_rng(20260907)
    clean = kane_mele(0.0, -1.0, 0.3, 0.0)
    orb = np.asarray(clean.orb_vecs, dtype=float) + rng.normal(scale=1e-9, size=(2, 2))
    dirty = TBModel(Lattice(clean.lat_vecs, orb.tolist(), periodic_dirs=[0, 1]),
                    spinful=True)
    dirty.set_onsite(clean.onsite, mode="set")
    for hop in clean.hoppings:
        amp = np.asarray(hop["amplitude"], dtype=complex)
        jitter = rng.normal(scale=1e-9, size=amp.shape)
        dirty.set_hop(amp + jitter + jitter.conj().T if amp.ndim else amp + 1e-9,
                      hop["from_orbital"], hop["to_orbital"],
                      list(hop.get("lattice_vector", [0, 0])), mode="add")

    ctx = _measure._MeasurementContext()
    # the input Hamiltonian, on the mesh the detector will use (`realized_group`'s
    # own `nk=6`): the first bind is spent here, so the pass and the projection
    # are the two that follow
    bound = ctx.bind(dirty, _measure._detection_grid(2, 6), {}, tol=1e-8, cut=None,
                     window=None, solver="eigen")
    assert bound is not None and ctx.rebuilds == 1
    shared = _realized._realize(dirty, precision=1e-8, characters={0: "pz"},
                          cross_check=False, context=ctx)
    assert not isinstance(shared, Refusal), str(shared)
    assert shared.positions is not None and shared.positions["adjusted"] is True
    assert shared.symmetrization is not None, "the projection ran"
    assert ctx.rebuilds == 3, (
        "the input, the moved centres and the projected hoppings are three "
        f"different Hamiltonians: {ctx.record()}")
    assert _measure._model_key(ctx.grid.model) == _measure._model_key(shared.model), \
        "the last Hamiltonian bound is the projected one the answer is about"
    assert _measure._model_key(dirty) != _measure._model_key(shared.model), \
        "and it is not the one handed in"


# ---------------------------------------------------------------------------
# 3.  the identification memo
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_memo():
    """The memo, emptied before and after — it is process-level state."""
    _spacegroup._identification_memo_clear()
    yield
    _spacegroup._identification_memo_clear()


def test_the_identification_memo_answers_a_repeated_cell_once(clean_memo):
    """One miss, then hits — and a hit is the object the first call built."""
    first = _spacegroup._group_of(_two_site_cell())
    assert first, str(first)
    assert _spacegroup._identification_memo_stats()["misses"] == 1
    assert _spacegroup._identification_memo_stats()["hits"] == 0

    # a different object, the same cell: the *positions* are what is keyed on
    again = _spacegroup._group_of(_two_site_cell())
    assert again is first, "a repeated cell is identified once per process"
    stats = _spacegroup._identification_memo_stats()
    assert (stats["hits"], stats["misses"], stats["stores"]) == (1, 1, 1)

    # and the answer is the same answer, not merely the same object
    assert (again.number_str, again.name, again.size) == ("10.43", "P2/m1'", 8)


@pytest.mark.slow
def test_every_flag_the_answer_depends_on_is_in_the_memo_key(clean_memo):
    """A flag left out of the key would serve a stale answer; none is left out.

    What is measured is the key: a second question is a second **miss** and its
    answer is a different object -- for every flag the identification takes,
    for the valuedness, and for the positions themselves.  Asked of a spinless
    cell, where ``spinor=True`` is a real choice: a spinful model resolves
    ``spinor=None`` to ``True`` already, so ``spinor=True`` there is the same
    question and rightly the same entry.  Three of the questions also change the
    answer on *this* input -- ``include_TR=False`` serves the unitary ``P2/m``
    #10 instead of the grey #10.43, ``typat=[0, 1]`` serves ``Pm1'`` #6.19, and
    ``spinor=True`` the double-valued tables -- while ``symprec=1e-4``,
    ``vacuum_length=25.0`` and ``magmom=[0, 0]`` serve the group the default does.
    Those three are in the key because they *could* change it on another cell,
    not because they do here.
    """
    questions = [
        ("include_TR", _two_site_cell(), dict(include_TR=False)),
        ("symprec", _two_site_cell(), dict(symprec=1e-4)),
        ("typat", _two_site_cell(), dict(typat=[0, 1])),
        ("vacuum_length", _two_site_cell(), dict(vacuum_length=25.0)),
        ("magmom", _two_site_cell(), dict(magmom=[0.0, 0.0])),
        ("spinor", _two_site_cell(), dict(spinor=True)),
        ("positions", _two_site_cell(shift=0.05), {}),
    ]
    for what, cell, call in questions:
        _spacegroup._identification_memo_clear()
        base = _spacegroup._group_of(_two_site_cell())
        assert base, str(base)
        other = _spacegroup._group_of(cell, **call)
        stats = _spacegroup._identification_memo_stats()
        assert (stats["hits"], stats["misses"], stats["size"]) == (0, 2, 2), \
            f"{what} was not in the key: {stats}"
        assert other is not base, what
        if not isinstance(other, Refusal):
            # asked again, the *other* question is memoized on its own terms
            assert _spacegroup._group_of(cell, **call) is other, what
            assert _spacegroup._identification_memo_stats()["hits"] == 1, what
        if what == "spinor":
            assert base.spinor is False and other.spinor is True


def test_the_memo_is_bounded_and_evicts_the_least_recently_used(clean_memo,
                                                               monkeypatch):
    """A memo, not a database: past its bound the least recently used goes.

    Each entry holds ``irrep``'s whole ``SpaceGroupIrreps`` object, so the bound
    is what keeps a long-running process from growing without limit.  *Which*
    entry goes is **recency**, not arrival order, and that is measured here by
    asking again: seven distinct cells through a memo of three, with one of the
    three survivors asked for in the middle.  It outlives the next two inserts
    and its untouched neighbour does not — the ``move_to_end`` on a hit is the
    only thing that makes that true, and a FIFO fails it.  (Until 2026-09-09 the
    eviction *counter* was the whole of the evidence and a FIFO passed.)
    """
    patch_everywhere(monkeypatch, "_IDENT_MEMO_MAX", 3)

    def cell(i):
        lat = Lattice([[1.0 + 0.1 * i, 0.0], [0.0, 1.3]], [[0.0, 0.0]],
                      periodic_dirs=[0, 1])
        m = TBModel(lat, spinful=False)
        m.set_onsite([0.0])
        m.set_hop(-1.0, 0, 0, [1, 0])
        return _layers._slab_presentation(m)

    cells = [cell(i) for i in range(7)]
    for c in cells[:5]:
        assert _spacegroup._group_of(c), "every one of them identifies"
    stats = _spacegroup._identification_memo_stats()
    assert stats["size"] == 3 and stats["evictions"] == 2, stats

    # cells[2:5] are what is left.  Ask for the oldest of the three: a hit, which
    # moves it to the young end of the order.
    assert _spacegroup._group_of(cells[2]) is not None
    assert _spacegroup._identification_memo_stats()["hits"] == 1

    # two more cells evict two more entries -- and under recency they are
    # cells[3] and cells[4], not the one just used
    for c in cells[5:]:
        assert _spacegroup._group_of(c), "the two further cells identify"
    stats = _spacegroup._identification_memo_stats()
    assert (stats["size"], stats["evictions"]) == (3, 4), stats
    hits, misses = stats["hits"], stats["misses"]
    assert _spacegroup._group_of(cells[2]) is not None
    assert _spacegroup._identification_memo_stats()["hits"] == hits + 1, \
        "the entry used most recently outlived the two inserts"
    assert _spacegroup._group_of(cells[3]) is not None
    assert _spacegroup._identification_memo_stats()["misses"] == misses + 1, \
        "and cells[3], never asked for again, is the one that went"
