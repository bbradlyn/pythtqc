"""Performance phase, wave P1: the eigen-sift detector, against its own oracle.

``_find_intertwiners`` no longer stacks the Sylvester operator over the grid and
takes its SVD.  It **parametrizes** the solution space at one k-point — where
the equation has an explicit nullspace, the outer products of matching-energy
eigenvector pairs — and **sifts** the remaining points through ``n**2 x m``
constraints.  The stacked route survives as ``solver="stacked"``: the audit
oracle, and the faithful instrument for a *near*-symmetry, where the eigen
route's window inflates the breaking.

What is regressed here, and the number that closes each:

* **the two solvers agree, operation by operation** — ``dim``, ``vacuous`` and a
  machine-level residual at *every* dimension, on the inline fixtures and on
  four goldens: the two whose every operation poses a degenerate problem and
  the two fully periodic ones (worst best-residual per golden, eigen against
  stacked: ``i41_soc`` axial 9.3e-15 / 2.3e-14, Kane-Mele Rashba-free 1.8e-14 /
  4.1e-14, Fu-Kane-Mele 1.8e-14 / 2.4e-14, P4_232 1.9e-15 / 4.7e-15);
* **the declared tolerance governs the cut, the window and the gate** — on a
  synthetic near-symmetry (one hopping moved by 1e-7, i.e. 1.6e-8 relative)
  both solvers refuse at ``tol = 1e-8`` and the eigen route *keeps* the
  operation at ``tol = 1e-6``, which is the wannier90 usage;
* **it scales** — a degeneracy-lifted 3x3x3 Fu-Kane-Mele supercell (n = 108)
  in 0.5 s and +150 MB, against the stacked route's extrapolated minutes and
  ~19 GB at n = 96 (a 96-orbital wannier90 model);
* **a supercell's own translations are read off it** — an L^3 supercell has
  L^3 lattice translations in the commutant of ``{H(k)}``, and the detector
  returns exactly ``dim = L**3`` with no ansatz;
* **a spectrum too degenerate to parametrize inside the memory bound is
  refused, not answered wrongly** — the failure mode that reached a wrong
  ``dim = 3`` at residual 0.93 before the valve was made honest;
* **``H(k)`` is the model's own Bloch sum, bit for bit pythtb's** — summed from
  lattice-vector blocks gathered once per model content, a mesh that fits the
  budget in one call and a larger one in bounded chunks.
"""

import functools
import gc
import sys
import threading
import time

import numpy as np
import pytest

from conftest import patch_everywhere
from pythtb import Lattice, Mesh, TBModel

from pythtqc import _bloch
from pythtqc import groups as G
from pythtqc import symmetry as S
from pythtqc import _detector
from pythtqc import _tolerance
from pythtqc import _layers
from pythtqc import _measure
from pythtqc import _spacegroup

sys.path.insert(0, __file__.rsplit("test_detector.py", 1)[0])
import tqc_models                                            # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ULP = np.finfo(float).eps          # one unit of double-precision roundoff, 2.2e-16


def _max_H(model, k):
    """``max |H|`` over the k-points given — the scale a residual is measured against.

    An intertwiner residual is an absolute quantity built out of ``H`` and its
    eigenvectors, so what makes it *machine-level* is ``n · eps · max|H|`` and
    not any stored decimal: the bounds below are written that way so that a
    different LAPACK, a different blocking or a rescaled Hamiltonian cannot move
    the statement.
    """
    return float(np.abs(np.asarray(model.hamiltonian(k_pts=np.asarray(k),
                                                     flatten_spin_axis=True))).max())


def _mesh(nk, dim_k):
    """pythtb's own Monkhorst-Pack hypercube, as ``_validate_group`` builds it."""
    ms = Mesh(["k"] * dim_k)
    ms.build_grid((int(nk),) * dim_k, gamma_centered=False, k_endpoints=False)
    return np.asarray(ms.flat)


def _peak_rss(fn):
    """``(result, seconds, peak RSS growth in MB)`` — ``None`` MB without psutil."""
    try:
        import psutil
    except ImportError:                       # pragma: no cover - optional
        t0 = time.perf_counter()
        return fn(), time.perf_counter() - t0, None
    proc = psutil.Process()
    gc.collect()
    base = proc.memory_info().rss
    high = [base]
    stop = threading.Event()

    def watch():
        while not stop.wait(0.005):
            high[0] = max(high[0], proc.memory_info().rss)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    t0 = time.perf_counter()
    try:
        out = fn()
        dt = time.perf_counter() - t0
    finally:
        stop.set()
        watcher.join()
    return out, dt, (high[0] - base) / 1e6


def _inversion_permutation(model):
    """The orbital permutation of ``r -> s - r``, with the centre ``s`` measured.

    The inversion centre of a pythtb cell is not at the coordinate origin in
    general (diamond's is at ``(1/8, 1/8, 1/8)``), so it is found from the
    orbital set rather than assumed.
    """
    pos = np.asarray(model.orb_vecs, dtype=float) % 1.0
    for j in range(pos.shape[0]):
        target = (pos[0] + pos[j] - pos) % 1.0
        d = np.abs((pos[None, :, :] - target[:, None, :] + 0.5) % 1.0 - 0.5).max(axis=2)
        if d.min(axis=1).max() < 1e-9:
            perm = d.argmin(axis=1)
            if len(set(perm.tolist())) == pos.shape[0]:
                return np.asarray(perm)
    raise AssertionError("no inversion centre in this orbital set")


def _supercell(sc, lift=0.0, seed=0):
    """A Fu-Kane-Mele supercell, optionally with its folding degeneracy lifted.

    An unperturbed ``L**3`` supercell is the *worst* case for the eigen route
    and the wrong instrument for a scaling measurement: its folded bands are
    exactly degenerate across the star of each folded momentum, so the
    parametrization at one k-point costs ``sum_i d_i d'_i`` = 11 n outer
    products at n = 108 rather than the 2 n a generic spectrum asks for.  A
    random **inversion-symmetric** on-site potential breaks the small-cell
    translations and the star degeneracy while leaving inversion exact, which
    is the generic large-n model the anchors want.  Both are measured below.
    """
    model = tqc_models.fu_kane_mele_z2("strong").make_supercell(sc, to_home=True)
    if lift:
        perm = _inversion_permutation(model)
        rng = np.random.default_rng(seed)
        e = rng.normal(size=model.norb)
        e = 0.5 * (e + e[perm])               # exactly symmetric under inversion
        for i in range(model.norb):
            model.set_onsite(lift * float(e[i]), i, mode="add")
    return model


@pytest.fixture(scope="module")
def sc108():
    """3x3x3 Fu-Kane-Mele with the folding degeneracy lifted: n = 108."""
    return _supercell([[3, 0, 0], [0, 3, 0], [0, 0, 3]], lift=0.3)


@pytest.fixture(scope="module")
def many_hopping():
    """A long-range one-orbital model in a 3x3x3 supercell: 29,646 hoppings.

    Built this way because it is the only route that is *fast*: pythtb's
    ``set_hop`` costs ~60 us per call and slows as the table grows (364,500
    calls did not finish in ten minutes), and ``make_supercell`` is superlinear
    (27 s to reach 70,272 hoppings).  The arithmetic at a real wannier90
    model's 723,408 hoppings is checked separately, without the model.
    """
    lat = Lattice(np.eye(3), [[0.0, 0.0, 0.0]], periodic_dirs=[0, 1, 2])
    base = TBModel(lat, spinful=False)
    rng = np.random.default_rng(1)
    for R in ((a, b, c) for a in range(-6, 7) for b in range(-6, 7)
              for c in range(-6, 7) if (a, b, c) > (0, 0, 0)):
        base.set_hop(float(1e-3 * rng.normal()), 0, 0, list(R))
    return base.make_supercell([[3, 0, 0], [0, 3, 0], [0, 0, 3]], to_home=True)


# ---------------------------------------------------------------------------
# 1.  the two solvers agree, operation by operation
# ---------------------------------------------------------------------------

def _agreement(model, group, k):
    """``(n_ops, n_measured, worst eigen, worst stacked)`` — asserts as it goes.

    **Every** operation that has a solution space is measured, at every ``dim``,
    on both routes, and two numbers are bounded over the candidate pool the
    detector returns -- the solution basis, plus, at ``dim >= 2``, the *searched*
    unitary, the polar factor of the best-conditioned combination the seeded
    probe found:

    * ``min(residuals)``, against ``1000 · n · eps · max|H|``: the space contains
      an operator that intertwines ``H`` over the whole grid at machine level;
    * ``min(unitarity)``, against ``1000 · n · eps``: that operator is an
      *element of the space* and not merely something near it.  ``unitarity`` is
      ``‖X†X − 1‖_max`` of the raw candidate, before polar extraction, and it is
      what separates the two.  A degenerate space's bare basis elements are
      routinely singular, sector-projector-like matrices: with the ``dim >= 2``
      search deleted the best of them reads 0.89 on ``i41_axial`` and 1.00 on
      ``km_rashba0`` (against floors of 2.7e-12 and 8.9e-13) while their polar
      factors can still intertwine ``H`` by accident.  Those two goldens pose
      ``dim == 2`` at 8 of 8 and 48 of 48 operations, so this is where the
      search is regressed.

    ``dim == 0`` is both solvers *proving* the operation broken; there is no
    candidate to measure and their agreeing verdict is the whole claim, so those
    operations are counted in ``n_ops`` and not in ``n_measured``.  Nothing
    measured returns ``None`` rather than a zero that was never written.

    Both bounds carry the same slack prefactor, and the worst case here is 39x
    below its floor (diamond's stacked unitarity 1.1e-14 against ``1000 n eps =
    4.4e-13``; then Kane-Mele Rashba-free's residual 4.1e-14 against
    ``1000 n eps max|H| = 2.7e-12``, 65x; every other fixture further below).
    The prefactor is slack on purpose: both
    quantities are eigenvector-error floors and a near-degenerate gap moves them
    by a factor of a few, while the breakage they are for -- a solver returning
    a near-solution, or a non-solution, as a solution -- is decades away.
    """
    u_floor = 1000.0 * int(model.nstate) * _ULP
    floor = u_floor * _max_H(model, k)
    n_ops, n_measured, worst_e, worst_s = 0, 0, None, None
    for sym in group.symmetries:
        action = group.k_action_of(sym)
        if action is None:
            continue
        n_ops += 1
        tr = bool(sym.time_reversal)
        eig = _detector._find_intertwiners(model, action, k, antiunitary=tr)
        old = _detector._find_intertwiners(model, action, k, antiunitary=tr, solver="stacked")
        assert eig.dim == old.dim, (
            f"operation {int(sym.ind)} (antiunitary={tr}): eigen dim {eig.dim} against "
            f"stacked {old.dim}; eigen basis sizes {eig.basis_size}, window {eig.window}")
        assert eig.vacuous == old.vacuous, f"operation {int(sym.ind)}: vacuity disagrees"
        assert eig.method == "eigen" and old.method in ("stacked", "stacked_subset")
        if eig.dim:
            re, rs = float(np.min(eig.residuals)), float(np.min(old.residuals))
            assert max(re, rs) <= floor, (
                f"operation {int(sym.ind)} at dim {eig.dim}: best residuals {re:.2e} "
                f"(eigen) / {rs:.2e} (stacked) against {floor:.2e} = 1000 n eps "
                "max|H| -- the solution space must contain an operator that "
                "intertwines H at machine level")
            ue, us = float(np.min(eig.unitarity)), float(np.min(old.unitarity))
            assert max(ue, us) <= u_floor, (
                f"operation {int(sym.ind)} at dim {eig.dim}: best unitarity {ue:.2e} "
                f"(eigen) / {us:.2e} (stacked) against {u_floor:.2e} = 1000 n eps -- "
                "that operator has to be a member of the space, which at dim >= 2 "
                "is what the seeded combination search is for")
            n_measured += 1
            worst_e = re if worst_e is None else max(worst_e, re)
            worst_s = rs if worst_s is None else max(worst_s, rs)
    assert n_ops, "the fixture posed no intertwiner problem at all"
    return n_ops, n_measured, worst_e, worst_s


@pytest.mark.slow
def test_the_two_solvers_agree_on_the_inline_fixtures():
    """One-band and two-band models, including the vacuous ones: 336 solves.

    The one-band models are the vacuity population: the eigen route reaches the
    absolute test by its own road -- a parametrization point degenerate to
    within the window carries no information, so the whole grid is scanned and
    the scan's per-point ``max_ab |E_b - E'_a|`` *is* the test -- and must land
    on the same verdicts.

    288 of the 336 operations carry a solution space and are measured; the other
    48 (16 of ``square(tx, ty)``, 32 of the ``tz``-split spinful cubic) are the
    two solvers agreeing on ``dim = 0``, the *proof* that the operation is
    broken.  Both halves of that census are pinned: a regression that turned
    solutions into refusals would keep the total and move the split.
    """
    from test_groups import cubic, diamond, square, tetragonal

    cases = [(square(tx=-1.0, ty=-0.6), True, 2, 8),
             (square(), False, 2, 8),
             (square(td=-0.3), False, 2, 8),
             (cubic(), True, 3, 4),
             (tetragonal(), True, 3, 4),
             (cubic(spinful=True), False, 3, 4),
             (cubic(spinful=True, tz=-0.4), False, 3, 4),
             (diamond(), False, 3, 4)]
    total, measured = 0, 0
    for model, with_tr, dim_k, nk in cases:
        group = _spacegroup._group_of(model, include_TR=with_tr)
        n_ops, n_measured, _, _ = _agreement(model, group, _mesh(nk, dim_k))
        total, measured = total + n_ops, measured + n_measured
    assert total == 336, "the census the agreement claim is made over"
    assert measured == 288, "and how many of them carried a residual to bound"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["km_rashba0", "i41_axial", "fkm", "p4232"])
def test_the_two_solvers_agree_on_the_golden_fixtures(name):
    """The goldens, including every degenerate-nullspace case in the suite.

    ``km_rashba0`` and ``i41_axial`` conserve ``s_z``, so *every* operation
    poses a two-dimensional problem -- the population that exposed both of the
    detector's historical soundness defects, and the one where an early stop
    can over-count if the sifted points are correlated.  These two are also
    where ``_agreement``'s residual bound rests entirely on the ``dim >= 2``
    unitary search: 48 of 48 and 8 of 8 operations, no dim-1 case to fall back
    on.  Fu-Kane-Mele and ``P4_232`` are the two fully periodic goldens, one
    spinful and one spinless.

    On ``i41_axial`` the search is also asked of the **worst basis** its plane
    admits.  The identity's solution space is the plane spanned by the two spin
    projectors ``P_up`` and ``P_dn`` (``s_z`` is conserved); the basis a solver
    returns is a rotation inside that plane which LAPACK chooses, so no bound on
    it is a statement about this code.  The projectors themselves, built from
    the model's conserved operator rather than read off a solver, are singular
    beyond anything a search could use (``sigma_min = 0`` exactly), and a
    singular member's polar factor acts arbitrarily on its own kernel.  The
    seeded combination search must still return a candidate whose polar factor
    is a unitary member of the plane that intertwines ``H`` at machine level --
    the commutant is a ``*``-algebra, closed under the polar decomposition of
    its nonsingular elements, and the residual is the arbiter.
    """
    from pythtb.models import kane_mele

    if name == "km_rashba0":
        model, group, k = kane_mele(0.0, -1.0, 0.3, 0.0), None, _mesh(6, 2)
    elif name == "i41_axial":
        model = tqc_models.i41_soc("axial")
        group, k = _spacegroup._group_of(model, spinor=True, include_TR=True), _mesh(4, 3)
    elif name == "fkm":
        model, group, k = tqc_models.fu_kane_mele_z2("strong"), None, _mesh(4, 3)
    else:
        model = tqc_models.p4232_cano2018()
        cell = tqc_models.p4232_symmetry_cell()
        decorated = TBModel(Lattice(np.eye(3), cell["positions"].tolist(),
                                    periodic_dirs=[0, 1, 2]), spinful=False)
        group, k = _spacegroup._group_of(decorated, typat=cell["typat"].tolist()), _mesh(5, 3)
    if group is None:
        group = _spacegroup._group_of(model)
    n_ops, n_measured, worst_e, worst_s = _agreement(model, group, k)
    assert n_ops >= 4
    assert n_measured, "every fixture keeps at least the identity, so at least one"
    print(f"\n{name}: {n_ops} operations, {n_measured} with a solution space, "
          f"worst best-residual {worst_e:.2e} (eigen) / {worst_s:.2e} (stacked)")

    if name != "i41_axial":
        return
    # the search on the worst basis the identity's plane admits
    n = int(model.nstate)
    H = np.asarray(model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    P_up = np.kron(np.eye(int(model.norb)), np.diag([1.0, 0.0]))    # spin is the fast index
    P_dn = np.kron(np.eye(int(model.norb)), np.diag([0.0, 1.0]))
    assert max(float(np.abs(P @ H - H @ P).max()) for P in (P_up, P_dn)) \
        <= 8 * _ULP * float(np.abs(H).max()), "s_z is conserved: the projectors commute with H(k)"
    plane = np.stack([P_up, P_dn]).astype(complex)
    plane *= (np.sqrt(n) / np.linalg.norm(plane, axis=(1, 2)))[:, None, None]
    Q, _ = np.linalg.qr(plane.reshape(2, -1).T)

    def outside_plane(X):
        """``‖X − ΠX‖ / ‖X‖``, ``Π`` the orthogonal projector onto the plane."""
        x = np.asarray(X, dtype=complex).ravel()
        return float(np.linalg.norm(x - Q @ (Q.conj().T @ x)) / np.linalg.norm(x))

    cands, search = _detector._candidates_and_search(plane, 2, n)
    assert len(cands) == 3, "the searched candidate is appended to the basis"
    assert search["sigma_min_basis"] <= 1e-20, \
        "the diagonal projectors are singular beyond anything the search could use"
    assert search["picked"] >= 2 and search["sigma_min"] > 0.9
    w_, _, vh_ = np.linalg.svd(cands[-1])
    U = w_ @ vh_                                           # the polar factor
    assert outside_plane(U) <= 1e-12, "the searched unitary is a member of the plane"
    assert float(np.abs(U.conj().T @ U - np.eye(n)).max()) <= 1e-12
    covariance = float(np.abs(np.einsum("ab,kbc,dc->kad", U, H, U.conj()) - H).max())
    assert covariance <= 1e-13, "and it intertwines H(k) with itself at machine level"


# ---------------------------------------------------------------------------
# 2.  the declared tolerance: one declaration, three roles -- the covariance
#     gate, the nullspace cut and the eigen route's matching window
# ---------------------------------------------------------------------------

def _axis_swapping(group):
    """``.ind`` of the operations whose rotation exchanges the two in-plane axes.

    The rotations are integer matrices in the reduced basis, so ``R[0, 0] == 0``
    is exactly "``a1`` is not carried onto ``±a1``" — no tolerance to choose.
    """
    return {int(sym.ind) for sym in group.symmetries
            if int(np.asarray(sym.rotation)[0, 0]) == 0}


@pytest.fixture(scope="module")
def near_symmetry():
    """lin3g with one hopping moved by 1e-7 — a symmetry broken at 1.6e-8 relative.

    The wannier90 situation in miniature: a Hamiltonian that is symmetric to
    its own numerical defect and not to machine precision.  ``max|H| = 12.375``
    here, and the perturbation moves ``H`` over the grid by ``delta``, measured
    against the unperturbed model and returned with the fixture: 2.0e-7, the
    hopping's 1e-7 counted at both ends of the bond.

    *Which* operations that breaks is read off the geometry rather than out of a
    solver.  The moved bond is ``orbital 0 -> orbital 0`` along ``a1``, and
    spin-independent (``1e-7 * 1_2``); an operation whose rotation exchanges the
    two in-plane axes carries it onto the ``a2`` bond of the same orbital, which
    was left alone, while one that preserves both axes carries it onto itself.
    So exactly the 16 of P4/mmm1''s 32 operations with ``R[0, 0] == 0`` are
    broken, and the detector is asked to reproduce that set in
    ``test_validate_group_forwards_the_one_declared_tolerance``.
    """
    import copy

    base = tqc_models.lin3g()
    model = copy.deepcopy(base)
    hop = model.hoppings[0]
    assert (int(hop["from_orbital"]), int(hop["to_orbital"])) == (0, 0) and \
        list(np.asarray(hop["lattice_vector"], dtype=int)) == [1, 0], \
        "the geometric prediction below names this bond"
    model.set_hop(1e-7 * np.eye(2), int(hop["from_orbital"]), int(hop["to_orbital"]),
                  list(np.asarray(hop["lattice_vector"], dtype=int)),
                  mode="add", allow_conjugate_pair=True)
    group = _spacegroup._group_of(model)
    k = _mesh(6, 2)
    delta = float(np.abs(np.asarray(model.hamiltonian(k_pts=k, flatten_spin_axis=True))
                         - np.asarray(base.hamiltonian(k_pts=k, flatten_spin_axis=True))
                         ).max())
    # 2e-7 by construction: the 1e-7 hopping enters H at both ends of the bond.
    # The two Hamiltonians are differenced at the scale of max|H|, so what the
    # equality is held to is the roundoff of *that* subtraction, 1000 eps max|H|
    # (2.7e-12 here, against an observed 5.6e-16 of it).
    assert delta == pytest.approx(2e-7, rel=0.0,
                                  abs=1000.0 * _ULP * _max_H(model, k)), \
        f"the perturbation's own size over the grid: {delta:.6e}"
    swapped = _axis_swapping(group)
    assert len(swapped) == 16 and len(group.symmetries) == 32
    broken = next(sym for sym in group.symmetries if int(sym.ind) in swapped)
    return model, group, broken, k, delta


def _split_copies(eps, turned):
    """Two copies of the square lattice's ``s`` band at one site, joined by ``eps`` on the site.

    ``turned`` writes the pair as ``(c1 ± c2)/√2``, where the join is a
    splitting of the two orbitals instead.  Either way the bands are ``h(k) ±
    eps`` at every k, one cluster whose magnitude about the mean on-site energy
    is ``eps`` itself wherever ``h(k) = 0``.
    """
    m = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]] * 2, periodic_dirs=[0, 1]),
                spinful=False)
    for o in (0, 1):
        m.set_hop(1.0, o, o, [1, 0])
        m.set_hop(1.0, o, o, [0, 1])
    if turned:
        m.set_onsite([eps, -eps])
    else:
        m.set_hop(eps, 0, 1, [0, 0])
    return m


def test_a_raised_tol_moves_the_window_with_it(near_symmetry):
    """The tolerance contract, measured: ``tol`` alone is not enough, and it knows it.

    A near-solution of relative residual ``rho`` has cross terms ``rho/gap``
    between bands, and a window ``delta`` represents it only to ``rho/delta``.
    So a raised ``tol`` with an unchanged window would still read ``dim = 0``,
    and the window follows ``tol``: ``clip(10*sqrt(tol), 1e-3, 0.3)``.

    Measured on this fixture (``rho = 1.6e-8`` relative):

    * ``tol = 1e-8``: **both** solvers refuse.  The stacked route finds
      ``dim = 1`` and refuses on the covariance residual, 3.9e-7 against the
      1e-8 gate; the eigen route escalates its window 1e-3 -> 1e-2 -> 0.1,
      still reads ``s_min/s_max = 1.1e-8`` against the 1e-8 cut, and refuses
      with ``dim = 0``.  Same verdict, different reason, and the record says
      which.
    * ``tol = 1e-6``: the window opens to 1e-2, the eigen route finds
      ``dim = 1`` at residual 4.0e-7, and ``_validate_group(tol=1e-6)`` keeps the
      operation.

    That residual is predicted, not stored.  This operation carries the moved
    bond onto the unmoved one, so its best near-solution misses by *both* of
    them and reads ``2 delta`` -- twice the perturbation's own size over the
    grid, which the fixture measures.  Both routes are held to it within 10 %:
    measured 2.000 delta by the eigen route, 1.933 by the stacked one, whose
    optimum is a least-squares compromise over the whole grid rather than one
    k-point's.

    No ``cut`` is declared anywhere here, so the cut is ``tol``: one number
    moves gate, cut and window together, and every value above is what that one
    declaration reads.  ``test_the_cut_finds_the_space_and_the_gate_judges_it``
    takes the three apart.

    And in the units of ``H`` the window is never narrower than the gate.  Two
    copies of one band split by ``2e-8``, read at a precision of ``1e-6`` (gate
    ``4e-5``, cut ``1e-5``), keep every matrix of the pair -- four, the
    copies' whole algebra -- whether the join is written between the copies or
    as a splitting of ``(c1 ± c2)/√2``: without the floor the sift is
    parametrized where ``h(k) = 0``, its window there is ``3e-10``, and the
    splitting reads as a gap (measured: two, in both bases).
    """
    model, group, sym, k, delta = near_symmetry
    action = group.k_action_of(sym)
    tr = bool(sym.time_reversal)

    for turned in (False, True):
        pair = _split_copies(1e-8, turned)
        scale = _max_H(pair, k)
        gate, cut = 10.0 * 1e-6 * scale, 10.0 * 1e-6
        held = _detector._find_intertwiners(pair, np.eye(2), k, tol=gate, cut=cut,
                                    window=_tolerance.eigen_window(cut), _floor=gate)
        assert held.dim == 4, ("a splitting below the declared precision is no gap",
                               turned, held.dim)

    strict_e = _detector._find_intertwiners(model, action, k, antiunitary=tr, tol=1e-8)
    strict_s = _detector._find_intertwiners(model, action, k, antiunitary=tr, tol=1e-8,
                                    solver="stacked")
    assert strict_e.dim == 0, "the eigen route refuses at the default tolerance"
    assert strict_e.window == pytest.approx(0.1), "escalated to the widest useful window"
    assert strict_e.escalations == (1e-3, 1e-2), "and the record says which it tried"
    assert 1e-9 < strict_e.singular_values[0] / strict_e.scale < 1e-7, \
        "the how-broken number, understated by at most 1/window"
    assert strict_s.dim == 1 and float(np.min(strict_s.residuals)) > 1e-8, \
        "the oracle refuses too, on the covariance residual instead"

    loose = _detector._find_intertwiners(model, action, k, antiunitary=tr, tol=1e-6)
    assert loose.dim >= 1, "a declared tolerance above the defect finds the operator"
    assert loose.window == pytest.approx(1e-2) and loose.escalations == ()
    for route, space in (("eigen", loose), ("stacked", strict_s)):
        residual = float(np.min(space.residuals))
        assert residual == pytest.approx(2.0 * delta, rel=0.1, abs=0.0), (
            f"{route} residual {residual:.3e} against the predicted 2 delta = "
            f"{2.0 * delta:.3e}: the operation moves the perturbed bond onto the "
            "unperturbed one, so the best near-solution misses by both")


@pytest.mark.slow
def test_validate_group_forwards_the_one_declared_tolerance(near_symmetry):
    """One declaration reaches gate, cut and window, and the record carries all three.

    ``tol`` is the covariance gate; ``cut`` defaults to it and ``window`` is
    derived from it, so a caller who declares one number gets the three moving
    together -- which is what a wannier90-grade model needs, where a raised gate
    against an unraised cut would read ``"no symmetry matrix"`` from a solve that was
    never given room to find the operator.  ``solver`` reaches the detector too,
    and every per-operation record says what the verdict rested on.

    The refusal set itself is checked against the fixture's geometric
    prediction: the 16 operations that exchange the two in-plane axes carry the
    perturbed bond onto an unperturbed one and must all be refused, and the 16
    that do not must all survive.

    That the default ``cut`` *is* ``tol`` is asserted rather than assumed:
    declaring it at the same number gives the same survivors, the same reason
    and the same window, down to the record.
    """
    model, group, sym, k, _ = near_symmetry
    ind = int(sym.ind)
    swapped = _axis_swapping(group)

    strict = _measure._validate_group(model, group, k_pts=k, tol=1e-8)
    assert set(strict.indices) == {int(s.ind) for s in group.symmetries} - swapped, \
        "the perturbed bond's geometry names the refusals, one for one"
    assert ind not in strict.indices, "refused at machine tolerance"
    reason = next(d["refuted_by"] for d in strict.discrepancies if d["ind"] == ind)
    assert reason == "no symmetry matrix"
    assert strict._records[ind]["window"] == pytest.approx(0.1)
    assert strict._records[ind]["escalations"] == (1e-3, 1e-2)
    assert strict._records[ind]["cut"] == strict._records[ind]["tol"] == 1e-8, \
        "the cut the solve ran at is the declared number itself"

    # the default is the declaration: spelling the cut out changes nothing
    spelled = _measure._validate_group(model, group, k_pts=k, tol=1e-8, cut=1e-8)
    assert spelled.indices == strict.indices
    assert spelled._records[ind] == strict._records[ind]

    loose = _measure._validate_group(model, group, k_pts=k, tol=1e-6)
    assert ind in loose.indices, "kept once the tolerance is declared above the defect"
    assert loose._records[ind]["window"] == pytest.approx(1e-2)
    assert loose._records[ind]["cut"] == 1e-6, "the cut followed the declaration"

    # the oracle stays reachable and refuses for its own (weaker) reason
    old = _measure._validate_group(model, group, k_pts=k, tol=1e-8, solver="stacked")
    assert ind not in old.indices
    assert next(d["refuted_by"] for d in old.discrepancies if d["ind"] == ind) \
        == "covariance"
    assert old._records[ind]["method"] in ("stacked", "stacked_subset"), \
        "solver= reached the detector and changed the algorithm, not only the record"


def test_the_cut_finds_the_space_and_the_gate_judges_it(near_symmetry):
    """Three roles under three names: which number moves decides what moves.

    ``cut`` is where the nullspace is taken -- relative, against the constraint
    operator's own norm.  ``tol`` is the covariance gate the answer is judged
    by -- absolute, in the units of H.  ``window`` is how wide a band of
    eigenvalue pairs the eigen route parametrizes in, and it follows ``tol``,
    because what has to be representable is a solution good to the *declared*
    tolerance.

    Measured on this fixture (broken at 1.6e-8 relative, best near-solution at
    residual ``2 delta = 4.0e-7``):

    * the cut raised alone (``tol = 1e-8``, ``cut = 1e-6``) finds the operator,
      ``dim = 1`` at the **unmoved** window 1e-3 and with no escalation: what
      the sift keeps a combination on is the cut;
    * the gate raised alone (``tol = 1e-6``, ``cut = 1e-8``) opens the window to
      1e-2 and escalates from there, and still reads ``dim = 0``: a wider window
      represents the near-solution, and the unmoved cut throws it away again.

    One layer up, the two halves are visible in the refusal itself: with the cut
    raised alone the operation is refused on its residual against the 1e-8 gate
    rather than for having no solution at all.
    """
    model, group, sym, k, delta = near_symmetry
    action = group.k_action_of(sym)
    tr = bool(sym.time_reversal)
    ind = int(sym.ind)

    raised_cut = _detector._find_intertwiners(model, action, k, antiunitary=tr,
                                      tol=1e-8, cut=1e-6)
    assert raised_cut.dim == 1, "the cut alone decides whether the space is found"
    assert raised_cut.window == pytest.approx(1e-3), \
        "and the window follows tol, which did not move"
    assert raised_cut.escalations == (), "a space found at the first window"
    assert (raised_cut.tol, raised_cut.cut) == (1e-8, 1e-6), "the record carries both"
    assert float(np.min(raised_cut.residuals)) == pytest.approx(2.0 * delta, rel=0.1,
                                                                abs=0.0)

    raised_gate = _detector._find_intertwiners(model, action, k, antiunitary=tr,
                                       tol=1e-6, cut=1e-8)
    assert raised_gate.dim == 0, "a raised gate does not move the cut"
    assert raised_gate.window == pytest.approx(0.1), "the window followed tol"
    assert raised_gate.escalations == (1e-2,), "starting one decade wider than before"

    refused = _measure._validate_group(model, group, k_pts=k, tol=1e-8, cut=1e-6)
    assert set(refused.indices) == {int(s.ind) for s in group.symmetries} \
        - _axis_swapping(group), "the gate still names the same survivors"
    entry = next(d for d in refused.discrepancies if d["ind"] == ind)
    assert entry["refuted_by"] == "covariance", \
        "a solution exists and misses the gate, rather than not existing"
    assert refused._nullspace_dims[ind] == 1
    assert entry["residual"] == pytest.approx(2.0 * delta, rel=0.1, abs=0.0)
    assert refused._records[ind]["cut"] == 1e-6 and refused._records[ind]["tol"] == 1e-8


def test_one_solve_per_k_action_class_and_the_record_names_it(monkeypatch):
    """Operations sharing a k-action pose one problem; it is solved once.

    A slab group's classes are the pairs ``{g, sigma_h g}``, so a 48-operation
    layer group poses 24 problems.  Counted, not inferred: ``_validate_group``
    reaches the detector through the name ``_find_intertwiners`` bound in
    ``groups``, and that binding is replaced by a counter for the call.  48
    operations, 24 calls -- dropping the sharing doubles it, and re-solving
    every member on top of the class solve makes it 72.

    That the sharing is *sound* is a separate claim and cannot be read off the
    bookkeeping, which writes ``solved_on`` from the same class map the
    partition is rebuilt from.  So each class's non-owner member is solved again
    here, directly and outside ``_validate_group``, and must return the verdict
    the class was given: 24 independent solves, same ``dim`` (1 throughout on
    this model) and residuals agreeing to machine level.
    """
    from pythtb.models import kane_mele

    layer = _layers._slab_presentation(kane_mele(0.0, -1.0, 0.3, 0.25))
    group = _spacegroup._group_of(layer)
    calls = []
    real = _detector._find_intertwiners

    def counted(*args, **kwargs):
        calls.append(kwargs.get("antiunitary"))
        return real(*args, **kwargs)

    patch_everywhere(monkeypatch, "_find_intertwiners", counted)
    validated = _measure._validate_group(layer, group, nk=4)
    assert len(validated.indices) == 48 == group.size
    assert len(calls) == 24 == len(validated.k_action_classes), \
        f"{len(calls)} solves for 48 operations in 24 k-action classes"

    k = _mesh(4, int(group.dim_k))
    floor = 1000.0 * int(layer.nstate) * _ULP * _max_H(layer, k)
    by_ind = {int(s.ind): s for s in group.symmetries}
    n_independent = 0
    for members in validated.k_action_classes:
        owner = validated._records[members[0]]["solved_on"]
        assert all(validated._records[i]["solved_on"] == owner for i in members)
        for i in members:
            if i == owner:
                continue
            sym = by_ind[i]
            space = _detector._find_intertwiners(layer, group.k_action_of(sym), k,
                                         antiunitary=bool(sym.time_reversal), tol=1e-8)
            n_independent += 1
            assert space.dim == validated._records[owner]["dim"], (
                f"operation {i} solved on its own reads dim {space.dim}, but was "
                f"given class owner {owner}'s dim {validated._records[owner]['dim']}")
            assert abs(float(np.min(space.residuals))
                       - validated._records[owner]["residual"]) <= floor, \
                f"operation {i}'s own residual against the one it was handed"
    assert n_independent == 24, "one non-owner solved independently per class"


# ---------------------------------------------------------------------------
# 3.  what it costs, at sizes the stacked route cannot reach
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_detector_at_a_hundred_orbitals(sc108):
    """n = 108, one operation: measured 0.53 s and +150 MB peak.

    The stacked route's own numbers at this size, from a 96-orbital wannier90 model:
    13-51 min and an extrapolated ~19 GB *per call*, on a machine with 17 GB.
    The bounds below carry more than an order of magnitude of slack in both
    quantities: what they are for is catching a regression to *minutes and
    gigabytes*, and both a loaded machine and the operating system's own
    working-set management move these numbers by factors of a few (measured:
    this test failed its 600 MB bound only while a second copy of the suite was
    competing for a 17 GB machine).

    The RSS half of that is measured only where ``psutil`` is installed (it is
    here).  Without it ``_peak_rss`` returns ``None`` MB and **no memory bound
    is asserted at all** -- the timing and the dimension still are.
    """
    k = _mesh(4, 3)
    (space,), seconds, peak = _peak_rss(
        lambda: (_detector._find_intertwiners(sc108, -np.eye(3), k),))
    assert sc108.nstate == 108
    assert space.dim == 1, "inversion is exact and nothing else is conserved"
    # machine level at n = 108: measured 2.4e-13 against n eps max|H| = 3.4e-14
    # (max|H| = 1.4, flat across this grid -- one k-point is taken rather than
    # all 64 only to avoid a 119 MB array inside a memory test)
    floor = 1000.0 * int(sc108.nstate) * _ULP * _max_H(sc108, k[:1])
    assert float(space.residuals[0]) < floor, \
        f"{float(space.residuals[0]):.2e} against {floor:.2e}"
    assert space.basis_size[0] == 216 == 2 * sc108.nstate, "Kramers pairs: m0 = 2n"
    assert space.n_kpts_solve <= 8 and space.n_kpts == 64
    print(f"\nn = 108: {seconds:.2f} s, peak +{peak} MB, m {space.basis_size}")
    assert seconds < 30.0, "measured 0.53 s"
    if peak is not None:
        assert peak < 1200.0, "measured +150 MB"


def test_a_supercells_own_translations_are_read_off_it():
    """``dim = L**3`` on an unperturbed ``L**3`` supercell — the answer is known.

    A supercell of a periodic model carries its ``L**3`` lattice translations
    as unitaries that commute with ``H(k)`` at every k, so the commutant is at
    least ``L**3``-dimensional and generically exactly that; the solution space
    of any operation is that commutant times one solution.  At ``L = 2``:
    ``dim = 8``, with 160 = 5n matching-energy pairs to parametrize it and a
    machine-level residual.  Nothing about supercells is typed into the solver.
    """
    model = _supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    k = _mesh(4, 3)
    space = _detector._find_intertwiners(model, -np.eye(3), k)
    assert model.nstate == 32
    assert space.dim == 8, "the 2x2x2 supercell's own translations"
    # machine level at n = 32: measured 2.8e-14 against n eps max|H| = 9.9e-15
    floor = 1000.0 * int(model.nstate) * _ULP * _max_H(model, k)
    assert float(np.min(space.residuals)) < floor, \
        f"{float(np.min(space.residuals)):.2e} against {floor:.2e}"
    assert space.basis_size[0] == 160 == 5 * model.nstate, "the parametrization's cost"


def test_a_spectrum_too_degenerate_to_parametrize_is_refused_not_guessed(monkeypatch):
    """The memory valve, and why it refuses instead of narrowing past degenerate pairs.

    A spectrum with exactly degenerate blocks has matching-energy pairs at
    *zero* gap, and narrowing the window to fit a budget cannot drop them: it
    collapses to machine precision, the basis can no longer represent an exact
    solution, and the returned space is **wrong** rather than coarse --
    measured ``dim = 3`` at residual 0.93 on an unperturbed 4x4x3 supercell
    (n = 192), whose 1,264 such pairs need 0.75 GB against the 0.51 GB default,
    before this was closed.  A resource limit is not a measurement about the
    physics, so it is refused, ``detector_too_large``, with what the honest
    solve needs and what runs it.

    Pinned at a size that costs nothing: inversion-symmetric Kane-Mele, every
    band a Kramers pair, matches 8 pairs at zero gap at its 4 states, so a
    budget of 7 refuses and names 9 -- the 8 and one more, which clear the
    window's floor.  The ceiling ``degenerate_max_bytes`` runs that solve
    exactly as the default budget does, and moves nothing else: on a nearly
    degenerate spectrum, three levels inside the window and a fourth apart,
    the valve narrows the window to the budget's nearest pairs with the
    ceiling raised as without it.  Through ``realized_group`` the refusal is
    returned, not raised, naming the cell's group as an upper bound, and
    ``max_bytes=`` is the ceiling that runs it -- the detector's own budget, cut
    to seven pairs by hand here, being the valve it does not move.
    """
    km = tqc_models.kane_mele_inversion()
    k = _mesh(4, 2)
    pair = 16 * km.nstate ** 2                   # one pair's matrix, in bytes
    default = _detector._find_intertwiners(km, -np.eye(2), k)
    tight = _detector._find_intertwiners(km, -np.eye(2), k, max_bytes=7 * pair)
    assert not tight and tight.gate == "detector_too_large"
    assert tight.context["n_pairs"] == 8
    assert (tight.value, tight.threshold) == (9 * pair, 7 * pair)
    assert "`realized_group(..., max_bytes=2.4e3)` runs it" in tight.reason
    ceiling = _detector._find_intertwiners(km, -np.eye(2), k, max_bytes=7 * pair,
                                   degenerate_max_bytes=tight.value)
    assert (ceiling.dim, ceiling.window) == (default.dim, default.window)
    assert np.array_equal(ceiling.residuals, default.residuals)

    # the valve, which the ceiling does not move: three levels within 5e-4 of
    # one another, a fourth at 1, and hoppings of 1e-5
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]],
                  [[0.0, 0.0], [0.31, 0.17], [0.62, 0.53], [0.83, 0.29]], periodic_dirs=[0, 1])
    flat = TBModel(lat, spinful=False)
    flat.set_onsite([1.0, 0.0, 3e-4, -2e-4])
    rng = np.random.default_rng(5)
    for i in range(4):
        for j in range(4):
            for R in ([1, 0], [0, 1]):
                flat.set_hop(1e-5 * rng.normal(), i, j, R, mode="add")
    five = 5 * 16 * flat.nstate ** 2
    valve = _detector._find_intertwiners(flat, np.eye(2), k, max_bytes=five)
    raised = _detector._find_intertwiners(flat, np.eye(2), k, max_bytes=five,
                                  degenerate_max_bytes=1e9)
    assert valve.window < _detector._find_intertwiners(flat, np.eye(2), k).window, "narrowed"
    assert (raised.window, raised.basis_size) == (valve.window, valve.basis_size)
    assert np.array_equal(raised.residuals, valve.residuals)

    # through realized_group: returned, the cell's group named, max_bytes the ceiling
    patch_everywhere(monkeypatch, "_find_intertwiners",
                        functools.partial(_detector._find_intertwiners, max_bytes=7 * pair))
    refused = G.realized_group(km, typat=[0, 1], cross_check=False, max_bytes=1e3)
    assert not refused and refused.gate == "detector_too_large"
    assert refused.reason.startswith(
        "The cell's group at this point is P3m11' #156.50, an upper bound")
    assert refused.context["group"].number_str == "156.50"
    runs = G.realized_group(km, typat=[0, 1], cross_check=False, max_bytes=2.4e3)
    monkeypatch.undo()
    plain = G.realized_group(km, typat=[0, 1], cross_check=False)
    assert runs.group.number_str == plain.group.number_str == "156.50"
    assert runs.rep.characters.irreps == plain.rep.characters.irreps


# ---------------------------------------------------------------------------
# 4.  the evaluation of H on a grid
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_grid_evaluator_is_bounded_and_streams_pythtbs_numbers(many_hopping):
    """``_GridHamiltonians``: pythtb's numbers, in one call where the mesh fits.

    One k-point of the Bloch sum holds one phase per lattice vector and per
    orbital and three ``nstate x nstate`` matrices, so on this model (29,646
    hoppings, 27 orbitals) the whole 512-point mesh fits the default budget:
    it is evaluated in one call, bit for bit ``TBModel.hamiltonian`` on the
    same points, and a point read from it costs no second evaluation.
    Squeezed to a budget of forty points, the same mesh streams in chunks of
    forty and returns the same numbers to the roundoff of the sum: a matrix
    product over forty points need not round as one over 512 does, and on
    Apple's Accelerate it does not.

    The numbers are asserted always; the memory only where ``psutil`` is
    installed, since that is what measures RSS at all.
    """
    k = _mesh(8, 3)
    assert many_hopping.nhops > 25000 and many_hopping.nstate == 27
    reference = np.asarray(many_hopping.hamiltonian(k_pts=k, flatten_spin_axis=True))

    whole = _detector._GridHamiltonians(many_hopping, k)
    assert whole.fits and whole.chunk >= len(k)
    (full,), seconds, peak = _peak_rss(lambda: (whole.full(),))
    assert np.array_equal(full, reference), "one call is pythtb's own sum, bit for bit"
    assert np.array_equal(whole.at(17), reference[17])
    assert whole.n_evaluated == len(k), "a point of a mesh that fits is read off the one call"

    squeezed = _detector._GridHamiltonians(many_hopping, k,
                                   max_bytes=40 * whole._sum().point_bytes)
    assert not squeezed.fits and squeezed.chunk == 40
    out = np.empty_like(reference)
    for lo, hi, blk in squeezed.chunks():
        assert hi - lo <= 40
        out[lo:hi] = blk
    # the roundoff of a sum at the scale of max|H|, 1000 eps max|H|, as the
    # perturbation test above holds its difference to
    streamed = float(np.abs(out - reference).max())
    assert streamed <= 1000.0 * _ULP * float(np.abs(reference).max()), \
        f"chunking changes the arithmetic beyond its rounding: {streamed:.3g}"
    assert squeezed.n_evaluated == len(k)
    print(f"\n512 points in one call, {many_hopping.nhops} hoppings: {seconds:.2f} s, "
          f"peak +{peak} MB")
    if peak is not None:
        assert peak < 100.0


def _parametrized_honeycomb():
    """A honeycomb whose site energies and bonds are the parameters ``delta``, ``t``, ``t2``."""
    lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]],
                  periodic_dirs=[0, 1])
    model = TBModel(lat, spinful=False)
    model.set_onsite([lambda delta: delta, lambda delta: -delta])
    for R in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(lambda t: t, 0, 1, R)
    model.set_hop("t2", 0, 0, [1, 0])
    model.set_hop(0.1j, 1, 1, [1, 0])
    return model


def test_the_grid_is_the_models_own_bloch_sum_bit_for_bit():
    """``H(k)`` from lattice-vector blocks gathered once, bit for bit pythtb's.

    ``TBModel.hamiltonian`` regroups every hopping by lattice vector on each
    call -- about a second a call on a wannier90 import, however few points
    are asked for.  The package gathers the blocks once per model content and
    sums them with pythtb's own arithmetic, so what it evaluates is
    ``np.array_equal`` to ``model.hamiltonian(k_pts=k, flatten_spin_axis=True,
    **params)`` on the same points: a spinless and a spinful three-dimensional
    model, a spinful layer and a parametrized one, their eigenpairs against
    ``solve_ham``, and the errors pythtb stops a request with.  Kept by
    content, the blocks follow a model changed in place.

    One point of the sum holds one phase per lattice vector and per orbital
    and the ``nstate x nstate`` matrix, and that sets the chunk: at a wannier90
    import's size -- 96 orbitals whose 723,408 hoppings reach 79 lattice
    vectors -- the 512-point detection mesh is one call inside the default
    256 MB, checked on a stand-in that reports what a point holds and is never
    asked for a Hamiltonian.  One grid serves every operation of a group, and
    its rotated grids share its sum.
    """
    honeycomb = _parametrized_honeycomb()
    values = {"delta": 0.2, "t": -1.0, "t2": 0.05j}
    rng = np.random.default_rng(11)
    for model, params in ((tqc_models.p4232_cano2018(), {}),
                          (tqc_models.fu_kane_mele_z2("strong"), {}),
                          (tqc_models.kane_mele_z2("odd"), {}),
                          (honeycomb, values)):
        for nk in (1, 5, 64):
            k = rng.uniform(-0.7, 0.7, size=(nk, int(model.lattice.dim_k)))
            want = model.hamiltonian(k_pts=k, flatten_spin_axis=True, **params)
            assert np.array_equal(_detector._GridHamiltonians(model, k, params).full(), want)
            evals, evecs = model.solve_ham(k, return_eigvecs=True, flatten_spin_axis=True,
                                           **params)
            got_evals, got_evecs = _bloch.solve(model, k, params)
            assert np.array_equal(got_evals, evals) and np.array_equal(got_evecs, evecs)

    # a request pythtb stops stops with pythtb's error
    k = rng.uniform(-0.5, 0.5, size=(3, 2))
    for params, pts in (({}, k), (dict(values, u=1.0), k), (values, None),
                        (values, k[:, :1])):
        with pytest.raises(ValueError) as theirs:
            honeycomb.hamiltonian(k_pts=pts, flatten_spin_axis=True, **params)
        with pytest.raises(ValueError) as ours:
            _bloch.hamiltonian(honeycomb, pts, params)
        assert str(ours.value) == str(theirs.value)

    # kept by content: the same object, changed in place, is summed as it now is
    km = tqc_models.kane_mele_z2("odd")
    first = _bloch.hamiltonian(km, k)
    amps, i, j, R = _bloch.hopping_arrays(km)
    km.set_hop(1.5 * np.asarray(amps[0]), int(i[0]), int(j[0]), R[0].tolist(), mode="set")
    moved = _bloch.hamiltonian(km, k)
    assert not np.array_equal(moved, first)
    assert np.array_equal(moved, km.hamiltonian(k_pts=k, flatten_spin_axis=True))
    km.set_onsite([0.3, 0.1, 0.0, 0.0], ind_i=0, mode="add")
    assert np.array_equal(_bloch.hamiltonian(km, k),
                          km.hamiltonian(k_pts=k, flatten_spin_axis=True))

    # the chunk at a wannier90 import's size, on a stand-in
    class _Model:
        nstate = 96

    class _Sum:
        point_bytes = _bloch.point_bytes(79, 96, 96)

    wannier = _detector._GridHamiltonians(_Model(), _mesh(8, 3), bloch=_Sum())
    assert wannier.fits and wannier.chunk >= 512, wannier.chunk

    # and the shared instance is what a group uses: one grid, every operation
    model = tqc_models.lin3g()
    k = _mesh(6, 2)
    shared = _detector._GridHamiltonians(model, k)
    group = _spacegroup._group_of(model)
    for sym in list(group.symmetries)[:4]:
        _detector._find_intertwiners(model, group.k_action_of(sym), k,
                             antiunitary=bool(sym.time_reversal), hamiltonians=shared)
    assert shared.n_evaluated == len(k), "the source grid is evaluated once, not once per op"
    assert shared.rotated(-k)._sum() is shared._sum(), "a rotated grid shares the blocks"

    with pytest.raises(ValueError, match="different k-grid"):
        _detector._find_intertwiners(model, np.eye(2), _mesh(4, 2), hamiltonians=shared)
