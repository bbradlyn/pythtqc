"""The spinor-basis declaration: ``tqc.spin`` and ``_build_rep(spin=...)``.

What each test is for, and the number that closes it.

* **the re-encoding theorem** -- the wave's anchor.  A native spinful fixture,
  re-presented as the spinless ``2N``-orbital model ``wannier90`` would deliver
  and re-labelled through the declaration, must give **the same physics**: the
  same space group, the same surviving operations, the same matrices under the
  basis permutation ``Pi``, the same site-character labels, the same corep
  content at every high-symmetry point and the same EBR verdict.  Measured on
  three fixtures (Kane-Mele in ``P3m11' #156.50``, lin3g in ``P4mm1' #99.164``,
  ``i41_soc`` in the unitary ``#80``) in the layouts that genuinely reorder
  the index (``blocked``, and a scrambled explicit ``pairs``):
  ``max |Pi^T D_flat Pi - D_native| <= 7.3e-16``.  (In the ``interleaved``
  layout ``Pi`` is the identity, and the two builds are literally the same
  arithmetic.)  The 3D grey case, Fu-Kane-Mele's strong phase, is measured
  below through its indicator class.  Each fixture is read in the group its
  *Hamiltonian* realizes, which is what makes the corep tables complete at
  every point; see ``_realized``.
* **the dialect** -- ``spin_operator`` is ``spin_operator_spinless``, bit
  for bit (``0.0``, and ``np.array_equal``), for the three kinds and two axes.
  One declaration therefore serves both this package and the ``P O P``
  machinery, which is the whole reason the vocabulary was copied rather than
  invented.
* **the presentation identity** -- ``max_k |H_flat(k) - Pi H_native(k) Pi^T| ==
  0.0`` exactly, on every fixture and layout.  The re-presentation is a
  relabelling of the state index and nothing else; the gate is ``0.0`` and is
  not to be weakened.
* **the unitary anchor survives the re-encoding** -- ``i41_soc`` with
  ``include_TR=False`` is the fixture whose gauge only the *tables* can fix
  (``#80``'s ``Z4`` has a complex linear character, so covariance, the
  canonical norm and the product table are all blind to it).  The declared
  basis carries the same argument verbatim, because its internal factor **is**
  the SU(2) lift: matrices equal to ``0.0`` under ``Pi`` and identical
  ``table_anchored`` flags.
* **the gates**, each a falsy ``Refusal`` naming its fix: a pair whose members
  sit ``0.57735`` apart (``spin_pair_split``, which ``realized_group`` returns
  from the centres alone, before any group is identified, and which a
  declared precision lifts for a pair apart only by noise), a declared basis
  against single-valued tables (a ``ValueError``, as for a native spinful
  model, since it is the same statement), spinor states
  read single-valued with no basis declared (time reversal measured broken by
  its square, ``-1``, and the unitary group built single-valued where the
  spinor class allows it; where it does not -- two commuting operations whose
  matrices anticommute -- not a gate but a descent, to the largest subgroup
  that carries an ordinary representation, a tie between such subgroups
  settled by a fixed rule), and ``spin=`` on a native spinful model
  (``ValueError`` -- a grammar mistake, not a measurement).
* **a pairing is read pair by pair** -- on a honeycomb carrying two spinors
  per site, the true pairing listed in another order reads the regular
  layout's group and labels, and a pairing that joins the states of two
  Wannier functions of one site descends, time reversal refuted by its own
  lift.
* **a declared pair is one Wannier function**: each site of the Kane-Mele
  flattening is one one-dimensional multiplet carrying the tables' own lift,
  and reads the native model's site irrep, ``-E1``, beside its corep content.
* **declarations on a declared basis** are about the pair's **spatial**
  character: the ordered form takes ``2N`` entries whose pair members must
  agree, the mapping form keys channels, and a pair given two different labels
  raises ``ValueError`` before anything is measured.
* **the passthroughs** -- ``realized_group(spin=...)`` measures time reversal
  as the native record does.  In 3D the literature anchor survives the
  re-encoding:
  Fu-Kane-Mele's strong phase, re-presented spinless and re-identified as
  ``R-3m1' #166.98`` through the decoration route, is class ``(0, 1)`` of
  order 4 in ``Z2 x Z4`` (``z4 = 3``, weak indices ``1, 1, 1``) --
  Fu-Kane's ``(1; 111)``.
* **the padded slab operations** -- ``include_vacuum_ops=True`` on the bare
  ``P6/mmm1' #191.234`` cell of Kane-Mele at ``rashba = 0`` represents all 48
  operations at covariance ``1.9e-15``, matching the native build under ``Pi``
  at ``0.0``.
"""

import numpy as np
import pytest
from pythtb.models import kane_mele

from pythtqc import ebr as tqe
from pythtqc import groups as tqg
from pythtqc import models as tqm
from pythtqc import sewing as tqw
from pythtqc import spin as tqsp
from pythtqc import symmetry as tqs
from pythtqc._tolerance import _PRESENTATION_RELATIVE
from pythtqc.refusal import InternalError, MagneticScopeWarning, Refusal

import tqc_models
from conftest import FKM_BOND, fkm_regime
# every 2x2 block of a spinful model conjugated by one SU(2) element, the
# recipe the layer tests turn a spin frame with
from test_layer import spin_blocks_turned

GATE = tqs.EXACT_TOL   # the exact floor of _build_rep's covariance gate

#: How far a re-encoded representation may differ from the native one.  It is a
#: *permutation* of the same arithmetic, so the honest expectation is machine
#: epsilon; measured worst over the fixture x layout pairs: 7.29e-16.
REENCODE_TOL = 1e-12

#: The ``(fixture, layout)`` pairs the theorem is measured on.  ``blocked`` and
#: the scrambled explicit ``pairs`` are the two layouts that genuinely reorder
#: the index (``interleaved`` is ``Pi = 1``), and ``i41`` needs ``blocked`` for
#: its table-anchored gauge.
CASES = (("km", "blocked"), ("km", "scrambled"), ("lin3g", "scrambled"),
         ("i41", "blocked"), ("i41", "scrambled"))


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------

def _scrambled_pairs(norb_spinful):
    """A deliberately irregular ``(up, dn)`` partition: ``(2N-1-i, i)``.

    Neither ``interleaved`` nor ``blocked``, and it puts the *down* member
    first in orbital order for every pair -- so the site's slot order carries
    ``up`` onto ``dn``, which the pair permutation has to survive.
    """
    return [(2 * norb_spinful - 1 - i, i) for i in range(norb_spinful)]


def _present(model, layout):
    """``(flat, basis)`` for one of the three layouts."""
    if layout == "scrambled":
        return tqsp.spinless_presentation(model, pairs=_scrambled_pairs(int(model.norb)))
    return tqsp.spinless_presentation(model, order=layout)


def _bundle(model, *, group=None, group_kw=None, typat=None, nk=6, **build_kw):
    """``(group, assignment, validated, rep, None)`` for one model, built once.

    The last slot is the time-reversal record ``realized_group`` measures, which
    a build by hand does not have.
    """
    if group is None:
        group = tqg._group_of(model, **(group_kw or {}))
    assert group, str(group)
    assignment = tqg._assign_orbitals(model, group, typat=typat)
    assert assignment, str(assignment)
    validated = tqg._validate_group(model, group, nk=nk)
    rep = tqs._build_rep(model, group, assignment, validated=validated, **build_kw)
    return group, assignment, validated, rep, None


def _realized(model, *, typat=None, nk=6, **build_kw):
    """The same five objects, through the whole pipeline instead.

    ``realized_group`` serves *the group the Hamiltonian realizes*; the bare
    ``_group_of`` chain above serves the **cell's** group, which is a strictly
    larger thing and, on three of these four fixtures, too large to label bands
    in.  Measured on that chain, with ``FIXTURES`` as it stood at ``9668d78``:

    ==========  ===================  ======================================
    fixture     cell group           ``hsp_table``
    ==========  ===================  ======================================
    km          ``191.234``, 48      refuses GM, K, M (``{}`` content)
    lin3g       ``123.340``, 32      refuses GM, M, X (``{}`` content)
    fkm         ``227.129``, 96      refuses GM, W, X; ``{'-L8': 1}`` at L
    i41         ``80``, 4            complete
    ==========  ===================  ======================================

    each refusal a representation short of its group's little group, which
    reading coreps now raises on.  Two different reasons, one symptom.  KM and
    lin3g are **layers**: ``_group_of`` sets them in a 3D cell,
    ``_validate_group`` cannot refute the operations on its vacuum direction
    (``H(k)`` has no out-of-plane ``k``), and ``_build_rep`` then leaves them
    out, so the little group at every point has columns the representation
    does not carry.  ``fkm`` is fully periodic, and the
    distorted bond simply breaks 72 of the cell's 96 operations.  Either way
    the labels a table would read are labels of a group ``H`` does not obey.

    ``realized_group`` answers the question that has an answer -- the layer on
    a substrate (``156.50``, ``99.164``) and the descent ``227.129 -> 166.98``
    -- and all three come back complete.  ``i41`` is not built this way because
    the reading it is here for, ``include_TR=False``, is a *choice of tables*
    and not something a Hamiltonian can realize: see
    ``test_the_declared_build_is_the_native_one_under_the_permutation``.
    """
    rg = tqg._realize(model, typat=typat, nk=nk, **build_kw)
    assert rg, str(rg)
    return rg.group, rg.assignment, rg.validated, rg.rep, rg.time_reversal


#: The three fixtures the theorem is measured on, with everything the native
#: build needs.  KM and lin3g are 2D (fast); ``i41_soc`` is the 3D unitary-only
#: anchor case.  ``route`` picks the builder above: ``"realized"`` for the two
#: whose cell group is bigger than the group their Hamiltonian obeys,
#: ``"_group_of"`` for ``i41``, whose unitary-only reading is asked for rather
#: than measured.  ``shared`` names the session fixture that holds the native
#: build already -- the same call, made once for every module.
FIXTURES = {
    "km": dict(build=lambda: tqc_models.kane_mele_z2("odd"), nk=6, bands=2,
               route="realized", group_kw={}, typat=[0, 1], characters=None,
               build_kw={}),
    "lin3g": dict(build=tqc_models.lin3g, nk=6, bands=4, route="realized",
                  group_kw={}, typat=[0, 0, 1, 1],
                  characters=["s", "dx2-y2", "s", "s"], build_kw={},
                  shared="lin3g_declared_cross_checked"),
    "i41": dict(build=lambda: tqc_models.i41_soc("generic", lam=0.4),
                nk=4, bands=4, route="_group_of", group_kw=dict(include_TR=False),
                typat=None, characters=None, build_kw=dict(n_battery=5)),
}

#: What each fixture's native ``hsp_table`` holds, on the route above.  Every
#: one of them is complete, which is what makes the corep comparison in
#: ``test_the_corep_content_and_the_ebr_verdict_are_the_native_ones`` a
#: measurement on every parameter rather than ``{} == {}`` on some.
NATIVE_COUNTS = {
    "km": {"-GM6": 1, "-K5": 1, "-K6": 1, "-M3M4": 1},
    "lin3g": {"-GM6": 1, "-GM7": 1, "-M6": 1, "-M7": 1, "-X5": 2},
    "i41": {"-GM5": 1, "-GM6": 1, "-GM7": 1, "-GM8": 1,
            "-M5": 1, "-M6": 1, "-M7": 1, "-M8": 1,
            "-P3": 2, "-P4": 2, "-X3": 2, "-X4": 2},
}


@pytest.fixture(scope="module")
def natives(request):
    """``{name: (model, group, assignment, validated, rep, time_reversal)}``.

    The native builds; ``time_reversal`` is ``realized_group``'s measured record
    where that is the route, and ``None`` for a build by hand.
    """
    out = {}
    for name, spec in FIXTURES.items():
        shared = request.getfixturevalue(spec["shared"]) if "shared" in spec else None
        model = spec["build"]() if shared is None else shared.source
        assert int(model.nspin) == 2, f"{name} must be natively spinful"
        if shared is not None:
            bundle = (shared.group, shared.assignment, shared.validated, shared.rep,
                      shared.time_reversal)
        elif spec["route"] == "realized":
            bundle = _realized(model, typat=spec["typat"], nk=spec["nk"],
                               characters=spec["characters"], **spec["build_kw"])
        else:
            bundle = _bundle(model, group_kw=spec["group_kw"], typat=spec["typat"],
                             nk=spec["nk"], characters=spec["characters"],
                             **spec["build_kw"])
        assert bundle[3], f"{name}: {bundle[3]}"
        out[name] = (model,) + tuple(bundle)
    return out


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def reencoded(request, natives):
    """One ``(fixture, layout)`` case, re-presented and re-built.

    Returns ``(name, layout, native_bundle, flat, basis, flat_bundle,
    presentation_residual)`` -- everything the theorem is read off, measured
    once per parameter and shared by every test below; the residual is
    relative to ``max|H|``.
    """
    name, layout = request.param
    spec = FIXTURES[name]
    model = natives[name][0]

    made = _present(model, layout)
    assert not isinstance(made, Refusal), str(made)
    flat, basis = made

    Pi = basis.permutation_matrix
    rng = np.random.default_rng(4242)
    k = rng.uniform(-0.5, 0.5, size=(8, int(model.lattice.dim_k)))
    a = np.asarray(model.hamiltonian(k_pts=k, flatten_spin_axis=True))
    b = np.asarray(flat.hamiltonian(k_pts=k, flatten_spin_axis=True))
    identity = float(np.abs(b - Pi @ a @ Pi.T).max()) / float(np.abs(a).max())

    chan = basis.pair_of_orbital
    typat = (None if spec["typat"] is None
             else [spec["typat"][int(chan[a_])] for a_ in range(int(flat.norb))])
    characters = spec["characters"]
    if isinstance(characters, (list, tuple)):
        characters = [characters[int(chan[a_])] for a_ in range(int(flat.norb))]

    if spec["route"] == "realized":
        # `spinor` is not passed: the declaration is what resolves it to True
        bundle = _realized(flat, typat=typat, nk=spec["nk"], spin=basis,
                           characters=characters,
                           reference_orbital=basis.states(0)[0], **spec["build_kw"])
    else:
        group = tqg._group_of(flat, spinor=True,
                              **{k_: v for k_, v in spec["group_kw"].items()
                                if k_ != "spinor"})
        bundle = _bundle(flat, group=group, typat=typat, nk=spec["nk"], spin=basis,
                         characters=characters,
                         reference_orbital=basis.states(0)[0], **spec["build_kw"])
    return name, layout, natives[name], flat, basis, bundle, identity


# ---------------------------------------------------------------------------
# 1.  the presentation: a relabelling of the state index and nothing else
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_spinless_presentation_is_the_same_hamiltonian(reencoded):
    """``H_flat(k) = Pi H_native(k) Pi^T`` to rounding, on every fixture and layout.

    The whole construction rests on this: the spinless model is the *same*
    Hamiltonian with the state index renamed, so every later disagreement is
    about the symmetry machinery and never about the model.  The re-presentation
    copies amplitudes, it does not recompute them; only the phases are
    evaluated anew, one per orbital -- ``2N`` against the native model's ``N``
    repeated for the spin, in a matrix product whose rounding can depend on its
    shape.  On the machines the suite was first measured on the two agree bit
    for bit; Apple's Accelerate rounds the two products differently, 5.2e-17 on
    ``i41``.  ``spinless_presentation``
    measures the identity itself and raises ``InternalError`` beyond
    ``_PRESENTATION_RELATIVE`` of ``max|H|``.  This test re-measures it on a
    *different* random k-set from the one the constructor used, so the number is
    not the constructor grading its own homework.

    Both members of a pair sit exactly where the spinful orbital sat.  That is
    the lossless principle: no model is ever inflated to ``nspin = 2`` inside
    this package, and ``bloch_operator`` works per orbital with ``nspin = 1``,
    so a pair with two centers would give two different Bloch phases to one
    Wannier function (``_build_rep`` refuses that anyway, ``spin_pair_split``).
    """
    name, layout, native, flat, basis, _bundle_, identity = reencoded
    model = native[0]
    assert identity <= _PRESENTATION_RELATIVE, \
        f"{name}/{layout}: max|H_flat - Pi H Pi^T| / max|H| = {identity:.3g}"
    assert int(flat.norb) == 2 * int(model.norb)
    assert int(flat.nspin) == 1 and not flat.spinful
    assert int(flat.nstate) == int(model.nstate), "the same number of states, renamed"
    assert len(basis.pairs) == int(model.norb)
    assert sorted(int(a) for pair in basis.pairs for a in pair) == list(range(flat.norb))

    orb = np.asarray(model.orb_vecs, dtype=float)
    flat_orb = np.asarray(flat.orb_vecs, dtype=float)
    for c, (up, dn) in enumerate(basis.pairs):
        assert np.array_equal(flat_orb[int(up)], orb[c])
        assert np.array_equal(flat_orb[int(dn)], orb[c])


def test_a_spinless_input_refuses_the_presentation():
    """It goes one way: a spinless model is already in the presentation.

    A falsy ``Refusal`` naming what to do instead -- ``spin_basis`` -- rather
    than an error, because "this model has no spin axis to take apart" is a
    measurement about the model.
    """
    from pythtb.models import haldane

    out = tqsp.spinless_presentation(haldane(0.0, -1.0, 0.15, 0.5))
    assert isinstance(out, Refusal) and not out, str(out)
    assert out.gate == "not_a_spinful_model"
    assert out.value == 1.0 and out.threshold == 2.0
    assert out.context["nspin"] == 1 and out.context["spinful"] is False
    assert "spin_basis" in out.reason


# ---------------------------------------------------------------------------
# 2.  the dialect: one declaration, two packages
# ---------------------------------------------------------------------------

def test_a_paired_operator_is_the_permuted_spin_fast_one():
    """``O = Pi (I_N (x) n.sigma) Pi^T`` for every paired kind -- ``0.0``.

    The declaration's content, stated independently of how it was written: a
    paired basis is exactly pythtb's own spin-innermost operator, carried
    through the permutation the basis names.  The three paired kinds differ
    only in that permutation.

    **``blocked`` is the kind that carries the claim.**  For ``interleaved``
    ``Pi`` is the identity and ``spin_operator`` *returns*
    ``kron(I_N, n.sigma)``, so the two sides are the same expression; for
    ``pairs=`` both sides scatter ``n.sigma`` onto the same ``(up, dn)`` tuple.
    Those two are one convention checked against itself -- worth having, since
    a permutation applied on the wrong side would still break them -- but only
    ``blocked``, where ``O = kron(n.sigma, I_N)`` is built spin-slowest and
    ``Pi`` is a genuine reordering, is an independent cross-check.  The default
    axis is ``z``.

    **The dialect.**  The vocabulary was copied from the sibling library rather
    than invented so that one declaration serves both the symmetry pipeline
    here and the projected-spin (``P s P``) machinery there, which is only true
    if the operators are the same object: ``0.0``, and ``np.array_equal``, over
    the three kinds and both axes.  That oracle is a **separate package in a
    separate checkout**, so it is measured where ``spin_resolved_analysis`` is
    importable, and the kinds above stand without it.
    """
    km = tqc_models.kane_mele_z2("odd")
    flat, _sb = tqsp.spinless_presentation(km)
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                     dtype=complex)
    kinds = ({"order": "interleaved"}, {"order": "blocked"}, {"pairs": [(1, 2), (3, 0)]})
    for kw in kinds:
        sb = tqsp.spin_basis(flat, **kw)
        Pi = sb.permutation_matrix
        for n in ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)):
            nvec = np.asarray(n, dtype=float)
            nsig = np.tensordot(nvec, sigma, axes=1) / np.linalg.norm(nvec)
            want = Pi @ np.kron(np.eye(len(sb.pairs)), nsig) @ Pi.T
            assert float(np.abs(tqsp.spin_operator(sb, n) - want).max()) == 0.0, \
                f"{kw} at n={n}"
        assert np.array_equal(tqsp.spin_operator(sb), tqsp.spin_operator(sb, (0, 0, 1)))

    try:
        import spin_resolved_analysis as sra
    except ImportError:
        return
    for kw in kinds:
        for n in ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)):
            mine = tqsp.spin_operator(tqsp.spin_basis(flat, **kw), n)
            theirs = sra.spin_operator_spinless(flat, n, **kw)
            assert np.array_equal(mine, theirs), f"{kw} at n={n}"
            assert np.abs(mine - mine.conj().T).max() == 0.0, "Hermitian"


def test_the_declaration_grammar_raises_rather_than_refusing():
    """Grammar mistakes are ``ValueError``s: none of them is a measurement.

    The refusal-vs-error law.  "This pairing splits a Wannier function" is a
    statement about the model and comes back as a ``Refusal``; "you wrote
    ``pairs`` that are not a partition" is a mistake in the sentence and there
    is no measurement to report.  So is a spin operator asked of anything but a
    declared basis, or along no axis.
    """
    from pythtb import Lattice, TBModel
    from pythtb.models import kane_mele

    km = kane_mele(1.0, 1.0, 0.6, 0.25)
    flat, sb = tqsp.spinless_presentation(km)

    with pytest.raises(ValueError, match="spinless model"):
        tqsp.spin_basis(km)                       # a native spinful model

    odd = TBModel(Lattice([[1, 0], [0, 1]], [[0, 0], [0.5, 0.0], [0.0, 0.5]],
                          periodic_dirs=[0, 1]), spinful=False)
    with pytest.raises(ValueError, match="even orbital count"):
        tqsp.spin_basis(odd, order="interleaved")
    with pytest.raises(ValueError, match="'interleaved' or 'blocked'"):
        tqsp.spin_basis(flat, order="spin-fast")

    with pytest.raises(ValueError, match="partition all orbital indices"):
        tqsp.spin_basis(flat, pairs=[(0, 1), (1, 2)])
    with pytest.raises(ValueError, match=r"shape \(M, 2\)"):
        tqsp.spin_basis(flat, pairs=[0, 1, 2, 3])

    with pytest.raises(ValueError, match="takes a SpinBasis"):
        tqsp.spin_operator("interleaved")
    with pytest.raises(ValueError, match="must be nonzero"):
        tqsp.spin_operator(sb, n=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="length-3 vector"):
        tqsp.spin_operator(sb, n=(0.0, 1.0))


def test_the_basis_reads_back_what_it_declared():
    """The record a caller checks before trusting a label.

    ``summary()`` names each pair's two orbitals; ``pair_of_orbital`` and
    ``spin_of_orbital`` are the arrays the machinery uses, and
    ``interleaved_index`` the permutation onto pythtb's spin-fast layout.
    """
    km = tqc_models.kane_mele_z2("odd")
    flat, _sb = tqsp.spinless_presentation(km)

    sb = tqsp.spin_basis(flat, pairs=[(1, 2), (3, 0)])
    assert sb.kind == "pairs" and len(sb.pairs) == 2
    assert sb.states(0) == (1, 2) and sb.states(1) == (3, 0)
    assert list(sb.pair_of_orbital) == [1, 0, 0, 1]
    assert list(sb.spin_of_orbital) == [-1, 1, -1, 1]
    assert list(sb.interleaved_index) == [1, 2, 3, 0]
    assert "pair 0: orbital 1 up, orbital 2 down" in sb.summary()


# ---------------------------------------------------------------------------
# 3.  the re-encoding theorem
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_declared_build_is_the_native_one_under_the_permutation(reencoded):
    """**The theorem.**  ``Pi^T D_flat Pi == D_native``, operation by operation.

    Not "the labels agree" -- the *matrices*.  ``Pi`` is the permutation of
    ``SpinBasis.interleaved_index``, the same object the presentation identity
    was measured with, so there is one permutation in play and no room for a
    second convention to hide in.  Measured worst over the fixture x layout
    pairs: ``7.29e-16`` (km, blocked).

    Its prerequisites are measurements too, made before any matrix is built:
    ``_group_of`` sees ``2N`` orbitals at ``N`` positions and reads the same
    space group, ``_validate_group`` measures the same operations surviving on
    the same Hamiltonian, and where ``realized_group`` read the crystal it
    measured time reversal the same way.

    **The unitary anchor survives the re-encoding.**  ``i41_soc`` in ``#80``
    with ``include_TR=False`` is the case where the gauge is fixed by **the
    tables alone** -- ``Z4`` has a complex linear character, so covariance, the
    canonical norm and the product table are all blind to it, and only
    ``RepAnchor.internal`` (the operation's own ``sym.spinor_rotation``,
    pseudo-real because SU(2) is) removes it.  The declared basis inherits that
    argument verbatim, because its internal factor **is** that lift, scattered
    onto the pair: identical ``table_anchored`` flags, and in the blocked layout
    matrices equal at ``0.0``.  Every other fixture here is grey and anchors on
    time reversal instead, which is asserted alongside so the boundary is
    stated rather than assumed.
    """
    name, layout, native, _flat, basis, bundle, _identity = reencoded
    _model, gN, asgN, valN, repN, trN = native
    group, assignment, validated, rep, tr = bundle
    assert rep, f"{name}/{layout}: {rep}"

    # the same crystal, read before any matrix is built
    assert group.number_str == gN.number_str, f"{name}/{layout}"
    assert group.name == gN.name and int(group.size) == int(gN.size)
    assert bool(group.spinor) is True
    assert tuple(validated.indices) == tuple(valN.indices)
    assert tuple(validated.broken) == tuple(valN.broken)
    # each spinful orbital's site now carries a whole (up, dn) pair
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    for _c, (up, dn) in enumerate(basis.pairs):
        assert int(site_of[int(up)]) == int(site_of[int(dn)])
    assert len(assignment.site_orbitals) == len(asgN.site_orbitals)
    assert assignment.wyckoff_spglib == asgN.wyckoff_spglib
    if FIXTURES[name]["route"] == "realized":
        assert tr["outcome"] == trN["outcome"] == "kept"

    assert rep._search["n_survivors"] == 1
    assert max(rep.residuals.values()) <= GATE
    assert tuple(rep.indices) == tuple(repN.indices)

    Pi = basis.permutation_matrix
    worst = max(float(np.abs(Pi.T @ np.asarray(rep.matrices[i]) @ Pi
                             - np.asarray(repN.matrices[i])).max())
                for i in repN.indices)
    assert worst <= REENCODE_TOL, f"{name}/{layout}: {worst:.3g}"

    assert rep.spin is basis and rep.spin.kind == basis.kind
    assert rep._reference["spin_pair"] == int(basis.pair_of_orbital[
        rep._reference["orbital"]])
    assert f"spinor basis ({basis.kind})" in repr(rep)

    # the gauge, fixed the same way: by the tables on #80, by time reversal elsewhere
    flags = {int(i): bool(r.table_anchored) for i, r in rep._normalization.items()}
    native_flags = {int(i): bool(r.table_anchored)
                    for i, r in repN._normalization.items()}
    assert flags == native_flags, f"{name}/{layout}"
    if name == "i41":
        assert rep.antiunitary_operator[0] is None and repN.antiunitary_operator[0] is None
        assert any(flags.values()), "the tables are what fixed this gauge"
        for r in rep._normalization.values():
            assert r.anchor_used is False, "there is no antiunitary operation to use"
            if r.table_anchored:
                assert len(r.candidates) == 4 and len(r.accepted) == 2
        if layout == "blocked":
            assert worst == 0.0, "the anchored matrices are the native ones, exactly"
    else:
        assert rep.antiunitary_operator[0] is not None, "grey: the antiunitary anchor does it"
        assert not any(flags.values())


@pytest.mark.slow
def test_the_pair_carries_the_orbitals_own_site_character(reencoded):
    """Each pair reads the label its spinful orbital read -- twice, once per member.

    The character extraction divides out the internal factor and reads the
    single-valued **spatial** character, so a declared pair is one multiplet
    with one label and the echo record repeats it on both members, each flagged
    ``up``/``dn``.  The function, the site symmetry and ``determined_by`` agree
    with the native build too: the declarations leave exactly the same
    labellings open.
    """
    name, layout, native, flat, basis, bundle, _identity = reencoded
    repN = native[4]
    rep = bundle[3]
    chan = basis.pair_of_orbital

    assert len(rep.characters.irreps) == int(flat.norb)
    for a in range(int(flat.norb)):
        row, native_row = rep.characters.orbitals[a], repN.characters.orbitals[int(chan[a])]
        assert row.irrep == native_row.irrep, f"{name}/{layout} orbital {a}"
        assert row.determined_by == native_row.determined_by
        assert row.function == native_row.function
        assert set(row.multiplet) == set(basis.pairs[int(chan[a])])
        assert row.site_symmetry == native_row.site_symmetry
        assert len(row.site_group) == len(native_row.site_group)
    assert rep.characters.spin is basis

    # the two members of a pair are one channel, and the summary says which is which
    for c, (up, dn) in enumerate(basis.pairs):
        assert rep.characters.irreps[int(up)] == rep.characters.irreps[int(dn)]
        assert rep.characters.orbitals[int(up)].multiplet \
            == rep.characters.orbitals[int(dn)].multiplet
        assert f"spin up of pair {c}" in rep.characters.orbitals[int(up)].note
        assert f"spin down of pair {c}" in rep.characters.orbitals[int(dn)].note
    text = rep.characters.summary()
    assert "spin" in text.splitlines()[0] and " up " in text and " down " in text


@pytest.mark.slow
def test_the_corep_content_and_the_ebr_verdict_are_the_native_ones(reencoded):
    """Downstream, unchanged: sewing, coreps and the EBR arithmetic.

    Nothing below ``_build_rep`` was taught anything about a spinor basis, and
    this is the measurement that says nothing had to be: the same corep labels,
    the same phase-invariant partitions at every point, the same aggregated
    symmetry vector and the same ``decompose`` verdict and terms.

    **The comparison is only a comparison if there is content on both sides**,
    so the native table's completeness and its non-empty containers are
    asserted first, against :data:`NATIVE_COUNTS` -- the labels themselves,
    typed out, so that a table that quietly emptied could not pass by agreeing
    with an equally empty one.  That is why the fixtures are built through
    ``realized_group``; ``_realized``'s docstring has the measurement that made
    it necessary.  ``decompose`` is called unconditionally and its verdict and
    terms read off the object, not through ``getattr`` defaults: it answers on
    all three fixtures (``"symmetry-indicated decomposable"`` for km, lin3g and
    i41), so a refusal on either side is a failure rather than two agreeing
    ``None`` values.
    """
    name, layout, native, _flat, _basis, bundle, _identity = reencoded
    gN, repN = native[1], native[4]
    group, rep = bundle[0], bundle[3]
    bands = FIXTURES[name]["bands"]

    tabN = tqw.hsp_table(repN, bands=bands)
    tab = tqw.hsp_table(rep, bands=bands)

    # the native side carries content -- otherwise the three comparisons below
    # are `{} == {}` and the docstring's claim is not measured at all
    assert tabN.complete and not tabN.refusals, f"{name}: {dict(tabN.refusals)}"
    assert dict(tabN.counts) == NATIVE_COUNTS[name], f"{name}"
    resolved = set(tabN.points) - set(tabN.skipped)
    assert set(tabN.content) == set(tabN.partitions) == resolved and len(resolved) >= 3
    assert all(tabN.content[p] and tabN.partitions[p] for p in resolved)

    assert sorted(tab.points) == sorted(tabN.points), f"{name}/{layout}"
    assert tab.complete is tabN.complete
    assert dict(tab.counts) == dict(tabN.counts)
    assert tab.content == tabN.content
    assert tab.partitions == tabN.partitions
    assert sorted(tab.refusals) == sorted(tabN.refusals)

    got = tqe.decompose(tab, group, rep=rep)
    want = tqe.decompose(tabN, gN, rep=repN)
    assert got, f"{name}/{layout}: {got}"
    assert want, f"{name}: {want}"
    assert got.verdict == want.verdict
    assert got.terms == want.terms


# ---------------------------------------------------------------------------
# 4.  the gates: what a declaration can get wrong
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def km_flat():
    """The interleaved Kane-Mele flattening and everything up to ``_build_rep``.

    Sites ``(0, 1)`` and ``(2, 3)``: each honeycomb sublattice carries one
    spinor, i.e. one declared pair.
    """
    km = tqc_models.kane_mele_z2("odd")
    flat, basis = tqsp.spinless_presentation(km, order="interleaved")
    group = tqg._group_of(flat, spinor=True)
    assert group and group.number_str == "191.234", str(group)
    assignment = tqg._assign_orbitals(flat, group)
    assert assignment.site_orbitals == ((0, 1), (2, 3))
    validated = tqg._validate_group(flat, group, nk=6)
    return km, flat, basis, group, assignment, validated


#: ``Pmmm`` (#47): the eight operations of ``P4/mmm`` whose rotations are diagonal.
PMMM_OPS = tuple((R, t) for R, t in tqc_models.P4MMM_OPS
                 if np.array_equal(np.asarray(R), np.diag(np.diag(np.asarray(R)))))
assert len(PMMM_OPS) == 8


def _pxpy_spinor(c=1.3, soc=0.3, seed=5, b=1.0, ops=tqc_models.P4MMM_OPS,
                 bonds=((1, 0, 0), (1, 0, 1))):
    """A spin-orbit-coupled ``(p_x, p_y)`` doublet at ``1a`` of ``P4/mmm``, natively spinful.

    One site, two orbitals, four states, in 3D: the smallest crystal whose
    spinless presentation puts two declared pairs into one multiplet -- ``E_u``
    of ``4/mmm`` times the spin -- which is where the verdict stage lifts the
    multiplet permutation to the declared pairs.  Random complex blocks on the
    ``(1, 0, 0)`` and ``(1, 0, 1)`` bonds are averaged over the sixteen
    operations of ``4/mmm``, the orbitals carried as ``(x, y)`` and the spin by
    the operations' own SU(2) lift, then over time reversal, beside an on-site
    ``L_z s_z``: the model obeys ``P4/mmm1'`` exactly and nothing else commutes
    with it.  ``b``, ``ops`` and ``bonds`` build its orthorhombic sibling: with
    ``b`` off ``1``, ``PMMM_OPS`` and the ``(1, 1, 0)`` and ``(1, 0, 1)`` bonds,
    the same doublet obeys ``Pmmm1'`` and, again, nothing else commutes with it.
    """
    from pythtb import Lattice, TBModel

    lattice = np.diag([1.0, b, c])
    rng = np.random.default_rng(seed)
    seeds = {}
    for bond in bonds:
        block = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        seeds[bond], seeds[tuple(-x for x in bond)] = block, block.conj().T
    hops = {}
    for R, _t in ops:
        Rc = tqm.cartesian_rotation(R, lattice)
        D = np.kron(Rc[:2, :2], tqm.spinor_of(np.sign(np.linalg.det(Rc)) * Rc))
        for bond, block in seeds.items():
            image = tuple(int(x) for x in np.rint(np.asarray(R, dtype=float) @ bond))
            hops[image] = (hops.get(image, 0.0) + D @ block @ D.conj().T / len(ops))
    reversal = np.kron(np.eye(2), [[0.0, 1.0], [-1.0, 0.0]])        # 1 (x) i sigma_y
    model = TBModel(Lattice(lattice.tolist(), [[0.0, 0.0, 0.0]] * 2,
                            periodic_dirs=[0, 1, 2]), spinful=True)
    onsite = soc * np.kron([[0.0, -1j], [1j, 0.0]], tqm.PAULI[2])    # L_z s_z
    model.set_onsite([onsite[:2, :2], onsite[2:, 2:]], mode="set")
    model.set_hop(onsite[:2, 2:], 0, 1, [0, 0, 0], mode="add")
    for bond, block in hops.items():
        if bond < tuple(-x for x in bond):
            continue                          # set once, with its hermitian partner
        block = 0.5 * (block + reversal @ block.conj() @ reversal.T)
        for i in range(2):
            for j in range(2):
                model.set_hop(block[2 * i:2 * i + 2, 2 * j:2 * j + 2], i, j, list(bond),
                              mode="add")
    return model


@pytest.fixture(scope="module")
def pxpy_spinor():
    """``(native, flat, basis)``: :func:`_pxpy_spinor`'s own record, and its interleaved flattening."""
    model = _pxpy_spinor()
    native = tqg.realized_group(model)
    assert native, str(native)
    assert (native.group.number_str, len(native.validated.indices)) == ("123.340", 32)
    assert native.rep.multiplets.dims == ((2,),), "one orbital doublet"
    assert native.conserved["dim"] == 0, "nothing but the identity commutes with H"
    flat, basis = tqsp.spinless_presentation(model, order="interleaved")
    return native, flat, basis


@pytest.fixture(scope="module")
def pmmm_spinor():
    """The doublet's orthorhombic sibling in ``Pmmm1'``, as its interleaved flattening."""
    model = _pxpy_spinor(b=1.17, ops=PMMM_OPS, bonds=((1, 1, 0), (1, 0, 1)))
    return tqsp.spinless_presentation(model, order="interleaved")[0]


def test_a_pair_split_across_two_sites_refuses(km_flat):
    """``pairs=[(0, 2), (1, 3)]`` on this layout is one sublattice paired with the other.

    A pair whose two members have different centres is not one spinor Wannier
    function: ``bloch_operator`` would give them two different Bloch phases,
    and no amount of symmetrizing makes them one orbital.  Measured and named:
    the two sites, both reduced positions, and the ``0.57735`` between them --
    the honeycomb nearest-neighbour distance in this cell -- and the two things
    it means, a mislabelled pairing or a magnetic configuration.  ``_build_rep``
    refuses it on the group it is handed, and ``realized_group`` from the
    centres alone, before any group is identified: the same refusal, value,
    context and reason, no group named, for the pairs as given and for
    ``"blocked"``, which is the same pairing, with ``typat`` or without.

    A pair apart only by the noise of a fitted model is neither, and the
    reason says so: one member moved by ``2e-5`` is refused with nothing
    declared, and under ``precision=1e-4``, which reads the two centres as one,
    the model reads the native group and band labels.
    """
    from pythtb import Lattice, TBModel

    _km, flat, _basis, group, assignment, validated = km_flat
    split = tqsp.spin_basis(flat, pairs=[(0, 2), (1, 3)])
    out = tqs._build_rep(flat, group, assignment, validated=validated, spin=split)
    assert isinstance(out, Refusal) and not out, str(out)
    assert out.gate == "spin_pair_split"
    assert out.value == pytest.approx(0.5773502691896257, abs=1e-12)
    assert out.threshold == 0.0
    assert out.context["pair"] == 0 and out.context["orbitals"] == (0, 2)
    assert out.context["sites"] == (0, 1)
    assert "different sites" in out.reason and "share its centre" in out.reason
    assert "mislabelled" in out.reason and "magnetic configuration" in out.reason
    assert "(`precision=`) reads two centres closer than it as one" in out.reason
    for spin, typat in ((split, None), ("blocked", None), ("blocked", [0, 0, 1, 1])):
        got = tqg.realized_group(flat, spin=spin, typat=typat)
        assert isinstance(got, Refusal) and got.gate == "spin_pair_split", str(got)
        assert (got.value, got.threshold, got.reason) == (out.value, out.threshold,
                                                           out.reason)
        kind = spin if isinstance(spin, str) else "pairs"
        assert got.context == {**out.context, "spin_kind": kind}, "no group named"

    orb = np.asarray(flat.orb_vecs, dtype=float).copy()
    orb[1, 0] += 2e-5                       # the down state of pair 0, off its partner
    noisy = TBModel(Lattice(flat.lat_vecs, orb.tolist(),
                            periodic_dirs=list(flat.periodic_dirs)), spinful=False)
    noisy.set_onsite(np.real(np.asarray(flat.onsite)).tolist(), mode="set")
    for hop in flat.hoppings:
        noisy.set_hop(hop["amplitude"], hop["from_orbital"], hop["to_orbital"],
                      list(hop.get("lattice_vector", [0, 0])), mode="add")
    got = tqg.realized_group(noisy, typat=[0, 0, 1, 1], spin="interleaved")
    assert isinstance(got, Refusal) and got.gate == "spin_pair_split", str(got)
    assert got.context["orbitals"] == (0, 1)
    assert got.value == pytest.approx(2e-5, rel=1e-9)
    read = tqg.realized_group(noisy, typat=[0, 0, 1, 1], spin="interleaved",
                              precision=1e-4)
    assert read, str(read)
    assert read.group.number_str == "156.50"
    assert read.rep.characters.irreps == ("-E1",) * 4
    assert tqw.hsp_table(read.rep, bands=2).counts == NATIVE_COUNTS["km"]


@pytest.fixture(scope="module")
def two_spinors_per_site():
    """A honeycomb with **two** spinors on each site: the native model and its interleaved flattening.

    The one geometry in which a pairing can be read across an orbit in more
    than one way: a site carrying more than one declared pair, in an orbit
    with a partner site.  Each site then has four flat orbitals, and which of
    the partner's pairs a pair lands in is read pair by pair from the
    declaration, whatever order a site lists its orbitals in.  Returns
    ``(native, flat, group, assignment, validated)``, the last three the
    flattening's cell.
    """
    from pythtb import Lattice, TBModel

    lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]],
                  [[1 / 3, 1 / 3], [1 / 3, 1 / 3], [2 / 3, 2 / 3], [2 / 3, 2 / 3]],
                  periodic_dirs=[0, 1])
    model = TBModel(lat, spinful=True)
    model.set_onsite([0.3, -0.3, 0.3, -0.3], mode="add")
    for R in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(-1.0, 0, 2, R, mode="add")
        model.set_hop(-1.0, 1, 3, R, mode="add")
        model.set_hop(0.2, 0, 3, R, mode="add")
        model.set_hop(0.2, 1, 2, R, mode="add")

    flat, _basis = tqsp.spinless_presentation(model, order="interleaved")
    group = tqg._group_of(flat, spinor=True)
    assert group and group.number_str == "191.234", str(group)
    assignment = tqg._assign_orbitals(flat, group)
    assert assignment.site_orbitals == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert assignment.orbits == ((0, 1),), "the two sites are one orbit"
    validated = tqg._validate_group(flat, group, nk=4)
    assert len(validated.indices) == 48 == group.size
    return model, flat, group, assignment, validated


@pytest.mark.slow
def test_a_pairing_is_read_pair_by_pair_and_a_wrong_one_descends(two_spinors_per_site):
    """Which orbitals form a pair is the declaration's; a pairing that is not the Hamiltonian's descends.

    The true pairing with the second site listed ``(a up, b up, a dn, b dn)``
    -- ``pairs=[(0, 1), (2, 3), (4, 6), (5, 7)]``, as ``spinless_presentation``
    builds and declares it -- reads ``P6mm1' #183.186`` with the regular
    layout's site irreps and band labels: each site is read pair by pair, never
    by the order it lists its orbitals in.

    Crossing the second site's pairs -- ``(4, 7)`` and ``(5, 6)``, each joining
    the up state of one Wannier function to the down state of the other --
    declares a crystal with less symmetry.  Every operation whose action those
    pairs do not carry is refuted by tensor structure, time reversal among
    them by its own lift, ``iσ_y`` on every pair, which no member of its
    solution space carries; the group descends to ``P1``, its representation
    built, and ``m(x)`` combined with time reversal, which the pairs do carry,
    is named beyond grey.  ``_build_rep`` handed the cell's group raises: the
    operations it is handed are the cell's, which the declared crystal does
    not obey in their own form, and the operations a representation is built
    for are the ones measured to hold in it.

    The same crossing on a crystal of one site, which conserves the spin, keeps
    ``P2``, whose diagonal lift the crossed pairs carry, and its two mirrors
    combined with time reversal, named beyond grey (``Pm'm'2``).  Each
    operation's solution space is read in its own form, its spin factor its
    own lift on the declared pairs, so no spin rotation the crystal conserves
    is among the members the representation is read off.
    """
    from pythtb import Lattice, TBModel

    model, flat, group, assignment, validated = two_spinors_per_site

    regular = tqg.realized_group(flat, spin="interleaved")
    assert regular, str(regular)
    assert (regular.group.name, regular.group.number_str) == ("P6mm1'", "183.186")
    irregular, basis = tqsp.spinless_presentation(model, pairs=[(0, 1), (2, 3), (4, 6), (5, 7)])
    read = tqg.realized_group(irregular, spin=basis)
    assert read, str(read)
    assert read.group.number_str == regular.group.number_str
    assert len(read.validated.indices) == len(regular.validated.indices) == 24
    assert read.rep.characters.irreps == regular.rep.characters.irreps == ("-E1",) * 8
    assert (tqw.hsp_table(read.rep, bands=4).counts
            == tqw.hsp_table(regular.rep, bands=4).counts)

    crossed = tqsp.spin_basis(flat, pairs=[(0, 1), (2, 3), (4, 7), (5, 6)])
    site_of = np.asarray(assignment.site_of_orbital, dtype=int)
    for up, dn in crossed.pairs:
        assert site_of[up] == site_of[dn], "within a site: this is not a split pair"
    with pytest.raises(InternalError, match="not realized on this orbital content"):
        tqs._build_rep(flat, group, assignment, validated=validated, spin=crossed,
                       n_battery=4)

    with pytest.warns(MagneticScopeWarning):
        wrong = tqg.realized_group(flat, spin=crossed)
    assert wrong, str(wrong)
    assert (wrong.group.name, wrong.group.number_str) == ("P1", "1")
    broken = {b["name"]: b["refuted_by"] for b in wrong.descent["dropped"]}
    assert set(broken.values()) == {"tensor structure"} and "E'" in broken
    assert wrong.time_reversal["outcome"] == "measured broken"
    assert "refuted by its spin lift" in wrong.time_reversal["note"]
    assert wrong.beyond_grey["magnetic_type"] == "III"
    assert [e["name"] for e in wrong.beyond_grey["operations"]] == ["m(x)'"]

    one = TBModel(Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]],
                          periodic_dirs=[0, 1]), spinful=True)
    one.set_onsite([0.3, -0.3], mode="add")
    for R in ([1, 0], [0, 1]):
        one.set_hop(-1.0, 0, 0, R, mode="add")
        one.set_hop(-0.5, 1, 1, R, mode="add")
        one.set_hop(0.2, 0, 1, R, mode="add")
        one.set_hop(0.2, 1, 0, R, mode="add")
    one_flat, _basis = tqsp.spinless_presentation(one, order="interleaved")
    with pytest.warns(MagneticScopeWarning):
        site = tqg.realized_group(one_flat, spin={"pairs": [(0, 3), (1, 2)]})
    assert site, str(site)
    assert (site.group.name, site.group.number_str) == ("P2", "3")
    assert site.time_reversal["outcome"] == "measured broken"
    assert site.beyond_grey["magnetic_type"] == "III"
    assert sorted(e["name"] for e in site.beyond_grey["operations"]) == ["m(x)'", "m(y)'"]


@pytest.mark.slow
def test_spinor_states_against_single_valued_tables(km_flat, pxpy_spinor, pmmm_spinor):
    """Spinor states in single-valued tables: declared, a contradiction; undeclared, a spinless crystal.

    ``_group_of(flat, spinor=False)`` is a perfectly legitimate ask for a
    spinless model, and it is what a caller gets by default.  Declaring the
    spinor basis beside it is the contradiction: the orbitals were *declared*
    to carry spin-1/2, so their content is double-valued and there is no
    single-valued question left: the build raises ``ValueError``, as it does
    for the native mirror, because it is the same statement.

    Left undeclared, the model is read as spinless, and its states say
    otherwise through time reversal: the square ``X X̄`` of its intertwiner
    ``X`` is gauge-free up to a positive factor, and it measures ``-1``, a
    spinor's time reversal, which single-valued tables do not describe.  So
    time reversal is measured broken, its record says why, and the cell is
    re-identified as the unitary group, whose representation is then the
    spinor class's question.  The spinful ``s`` pair at ``6c`` of ``R-3m``:
    the class of ``-3m`` is trivial, and the unitary ``R-3m`` is returned with
    its single-valued representation, ``E`` of ``3m`` at every orbital; with
    ``spin=`` the same model reads its native ``R-3m1'``.  Of the eleven
    antiunitary operations the detector keeps there, none holds by time
    reversal's own rule: the three twofolds and the three mirrors with time
    reversal square to ``+1``, as the tables ask, and each makes time reversal
    with its own twofold or mirror, which holds; inversion with it squares to
    ``-1``, and a threefold with it cubes to time reversal.  So nothing beyond
    grey is named -- the larger group these states obey is the grey one, read
    with ``spin=`` -- and the record's count is zero.  The doublet and its
    sibling below do name what lies beyond grey: each operation the projective
    class leaves out, combined with time reversal, is read once against the
    group returned, after the descent, and with it the eight of the doublet
    make ``P4/mm'm'`` and the four of its sibling a magnetic group of type III
    whose representation these states carry.

    The flattened ``(p_x, p_y)`` doublet of ``P4/mmm1'``: ``C2(z)`` and
    ``C2(y)`` commute and their matrices anticommute -- the sign their spin-1/2
    lifts carry, and a phase on either cancels from it -- so no single-valued
    representation of a group holding both exists, and one of every such pair
    is not an operation of the spinless crystal.  The group descends to the
    largest subgroup that carries an ordinary representation, ``P4/m``: its
    fourfold kept, its eight perpendicular twofolds and mirrors each refuted by
    the states' projective class beside the ``C2(z)`` it anticommutes with, and
    a single-valued representation that multiplies, built with default
    arguments.  ``i41_soc``'s spinor states read spinless keep the unitary
    ``I4_1``, whose fourfold screw the hoppings pin at ``i`` between two orbits
    of multiplets, and ``_build_rep`` called again on that group reads that
    factor as the root of unity it is and builds the matrices
    ``realized_group`` returned.  With ``descend=False`` the eight are named
    instead.  Its orthorhombic sibling in ``Pmmm1'`` is the
    tie: three ``2/m`` subgroups carry one, and the rule keeps the one whose
    twofold lies along the conventional ``c``, then ``a``, then ``b``.  The
    sibling's bonds written on the tetragonal cell descend in two steps, the
    hoppings first and the class after time reversal is measured broken, and
    the descent's note and the record's one-line summary count what every step
    dropped, by cause, operations left out to close a group or by the tie rule
    among the causes.  A flat
    layer's class is read once, free-standing, where each operation and its
    product with the horizontal mirror act alike on every orbital and every
    in-plane ``k``: the product has its partner's matrix times the mirror's
    sign, the even one the lowest-angular-momentum rule assigns, which commutes
    with every matrix.  Kane–Mele with its Rashba term, read so, keeps ``P6/m``
    free-standing, the first of the three subgroups of order 12 that carry one
    (``6/m`` and the two ``-6m2``) by its sixfold axis, its in-plane twofolds
    and vertical mirrors left out; on a substrate the side-keeping half of that
    reading, ``P6``, read on ``P6/mmm`` and carried, and the two agree.  With its
    bonds along the axes alone the sibling conserves a pseudospin, and the
    commutant supplies a time reversal of square ``+1``: the spinless reading
    keeps ``Pmmm1'``, its time reversal carrying ``-1`` on the ``p_y`` pair
    against the ``p_x`` pair -- the sign its own solution space pins, where a
    ``+1`` assumed on every multiplet misses covariance at order one.  The
    tetragonal doublet with its bonds along the axes alone conserves the same
    pseudospin on one fourfold multiplet: every operation's matrix there is the
    factor the hoppings fix times any matrix on the two copies, so which
    representation the copies carry the hoppings do not say, and it moves the
    site content itself -- ``2A2u + 2B2u``, ``2Eg`` or ``2Eu``, among twelve.
    The lowest-angular-momentum rule assigns ``2Eu``, the two spin copies of
    ``(p_x, p_y)``, keeps ``P4/mmm1'`` whole, and says so on the atomic-limit
    table and beside the band content.  A content none of those
    representations carries, four copies of ``A1g``, is a crystal with less
    symmetry: the group descends to ``Pmmm1'``, where one carries it.  Where
    every representation of the copies gives one site content -- two copies of
    an ``s`` at the general position of ``P-1``, ``2xA`` under each -- they are
    labellings like any other: the lowest two bands read differently under
    another, and one sentence says so, naming the pseudospin.

    The mirror is a time reversal squaring to ``+1`` read in double-valued
    tables: a spinful model with real, spin-independent hoppings and a Zeeman
    field along ``z`` keeps complex conjugation, which squares to ``+1``, and
    loses the spinor time reversal the double-valued tables ask for -- broken
    the same way, and the unitary ``I4_1`` returned.
    """
    _km, flat, basis, _group, _assignment, _validated = km_flat
    single = tqg._group_of(flat, spinor=False)
    assert single and single.spinor is False, str(single)
    assignment = tqg._assign_orbitals(flat, single)
    validated = tqg._validate_group(flat, single, nk=6)

    with pytest.raises(ValueError, match="declared to carry spin"):
        tqs._build_rep(flat, single, assignment, validated=validated, spin=basis)

    flat_r, _basis_r = tqsp.spinless_presentation(tqc_models.r3m_s_6c())
    with pytest.warns(MagneticScopeWarning):
        spinless = tqg.realized_group(flat_r)
    assert spinless, str(spinless)
    assert (spinless.group.name, spinless.group.number_str, spinless.group.size) == (
        "R-3m", "166", 12)
    reversal = spinless.time_reversal
    assert reversal["outcome"] == "measured broken" and reversal["grey_number_str"] == "166.98"
    assert reversal["square"]["asked"] == 1
    assert reversal["square"]["measured"] == pytest.approx(-1.0, abs=1e-12)
    assert "a spinor's time reversal" in reversal["note"]
    assert spinless.rep.characters.irreps == ("E",) * 4
    assert spinless.rep.product_residual < 1e-12
    assert spinless.descent is None, "the class of -3m is trivial: nothing descends"
    # each primed twofold and mirror makes time reversal with its own, which
    # holds: nothing of the grey group counts, and nothing beyond grey is named
    assert spinless.beyond_grey is None
    assert (reversal["n_obeyed"], reversal["n_broken"]) == (0, 12)
    assert ("Of those, 11, coset representatives of the lattice translations, have a "
            "symmetry matrix") in reversal["note"]
    assert "generates time reversal with its own unitary part" in reversal["note"]
    assert repr(spinless).endswith(
        "; time reversal measured broken, descended from R-3m1' #166.98: its square is -1, so "
        "the states carry half-integer spin, and spin= declares which orbitals are spin "
        "partners)")
    native = tqg.realized_group(flat_r, spin="interleaved")
    assert native, str(native)
    assert (native.group.number_str, native.group.size) == ("166.98", 24)
    assert native.time_reversal["outcome"] == "kept"
    assert native.time_reversal["square"] == pytest.approx({"asked": -1, "measured": -1.0},
                                                           abs=1e-12)

    # the flattened doublet: the pairs that anticommute, and the descent they make
    _native, flat_p, _basis_p = pxpy_spinor
    with pytest.warns(MagneticScopeWarning):
        out = tqg.realized_group(flat_p)
    assert out, str(out)
    assert (out.group.name, out.group.number_str, out.group.size) == ("P4/m", "83", 8)
    assert len(out.validated.indices) == 8 and out.group.spinor is False
    reversal = out.time_reversal
    assert reversal["outcome"] == "measured broken" and reversal["grey_number_str"] == "123.340"
    assert reversal["square"]["asked"] == 1
    assert reversal["square"]["measured"] == pytest.approx(-1.0, abs=1e-12)
    assert "a spinor's time reversal" in reversal["note"] and "`spin=`" in reversal["note"]
    d = out.descent
    assert (d["from"]["number_str"], d["to"]["number_str"]) == ("123", "83")
    assert d["dropped_by"] == {"projective class": 8} and d["iterations"] == 1
    assert any(row.seitz.startswith("{4+_001|") for row in out.operations.values()), \
        "the fourfold is kept"
    for b in d["dropped"]:
        assert b["refuted_by"] == "projective class" and np.isnan(b["residual"])
        assert "C2(z)" in b["names"] and b["name"] in b["names"], "named beside the kept C2(z)"
        assert b["commutation_phase"] == pytest.approx(-1.0, abs=1e-9)
        assert b["spin_half"] is True and "spin-1/2" in b["note"]
    assert sorted(b["name"] for b in d["dropped"]) == sorted(
        ["C2(x)", "C2(y)", "C2([110])", "C2([1-10])", "m(x)", "m(y)", "m([110])", "m([1-10])"])
    step = d["steps"][0]["projective"]
    assert step["tied"] == (), "the largest subgroup of 4/mmm carrying one is unique"
    assert len(step["pairs"]) == 24 and all(w["spin_half"] for w in step["pairs"])
    assert "anticommute" in d["note"] and "the largest subgroup" in d["note"]
    # the note says the eight hold and carry no single-valued representation,
    # and what the operations kept carry
    assert ("8 of them, which the Hamiltonian obeys, carry no single-valued "
            "representation on these states, so the cell was lowered to the largest group "
            "of its operations on which these states carry a single-valued "
            "representation, and re-identified as P4/m #83.") in d["note"], d["note"]
    assert "measured to hold" not in d["note"]
    assert ("(8 of 16 coset representatives of the lattice translations dropped: 8 by the "
            "states' projective class)") in repr(out)
    assert out.rep.antiunitary_operator[0] is None and out.rep.product_residual < 1e-12
    assert out.rep.multiplets.dims == ((2, 2),), "the two spin-J_z pairs, now apart"
    # the eight left out, each combined with time reversal, measured on
    # P4/mmm1', the last group that asks them, and read once against the P4/m
    # returned: with it they make P4/mm'm', whose representation the states
    # carry; the seven others the detector kept there make time reversal with
    # their unitary part, which P4/m holds
    beyond = out.beyond_grey
    assert beyond is not None and beyond["magnetic_type"] == "III" and beyond["closed"]
    assert (beyond["n_operations"], beyond["n_measured"]) == (8, 16)
    assert {o["of"] for o in beyond["operations"]} == {"123.340"}
    assert sorted(o["name"] for o in beyond["operations"]) == sorted(
        b["name"] + "'" for b in d["dropped"])
    assert (reversal["n_obeyed"], reversal["n_broken"]) == (8, 8)
    assert "Of those, 7 have a symmetry matrix" in reversal["note"]
    assert "Each of them combines time reversal with an operation of the group" in (
        reversal["note"])
    assert "The other 8 hold by that rule" in reversal["note"]
    assert repr(out).endswith("(8 antiunitary operations measured to hold) but is beyond "
                              "the validated scope)")
    assert "descended from P4/mmm1'" not in repr(out)
    refused = tqg.realized_group(flat_p, descend=False)
    assert isinstance(refused, Refusal) and refused.gate == "cell_group_not_realized"
    assert refused.context["refuted_by"] == {"projective class": 8}
    assert sorted(refused.context["projective_class"]) == sorted(b["ind"] for b in d["dropped"])
    assert "projective class of the states" in refused.reason and "`spin=`" in refused.reason
    assert refused.context["group"].number_str == "123"
    # ... and _build_rep, handed the fixed group P4/mmm it cannot lower, still
    # names the pair and the sign no single-valued representation carries
    fixed = refused.context["group"]
    no_rep = tqs._build_rep(flat_p, fixed, tqg._assign_orbitals(flat_p, fixed),
                            validated=refused.context["validated"])
    assert isinstance(no_rep, Refusal) and no_rep.gate == "rep_not_multiplicative"
    assert no_rep.value == pytest.approx(2.0, abs=1e-9)
    assert no_rep.context["commutation_phase"] == pytest.approx(-1.0, abs=1e-9)
    assert no_rep.context["spin_half"] is True and "half-integer spin" in no_rep.reason

    # i41_soc's spinor states read spinless keep the unitary I4_1, whose fourfold
    # screw the hoppings pin at i between two orbits of multiplets: _build_rep,
    # called again on the group returned, reads the i off the solution spaces
    # and builds the matrices realized_group returned
    flat_s = tqsp.spinless_presentation(tqm.i41_soc("generic", lam=0.4),
                                        order="interleaved")[0]
    with pytest.warns(MagneticScopeWarning):
        screw = tqg.realized_group(flat_s)
    assert screw, str(screw)
    assert (screw.group.number_str, len(screw.validated.indices)) == ("80", 4)
    again = tqs._build_rep(screw.model, screw.group, screw.assignment,
                           validated=screw.validated)
    assert again, str(again)
    assert again._search["strategy"] == "readoff", again._search["fallback_reason"]
    assert sum(abs(z - 1j) < 1e-9 for z in again._search["ratios"].values()) == 2
    assert max(float(np.abs(np.asarray(again.matrices[i])
                            - np.asarray(screw.rep.matrices[i])).max())
               for i in screw.rep.indices) == 0.0
    assert again.characters.irreps == screw.rep.characters.irreps

    # the tie: three 2/m of mmm carry one, and the rule keeps the one along c
    flat_o = pmmm_spinor
    with pytest.warns(MagneticScopeWarning):
        tie = tqg.realized_group(flat_o)
    assert tie, str(tie)
    assert (tie.group.name, tie.group.number_str, tie.group.size) == ("P2/m", "10", 4)
    step = tie.descent["steps"][0]["projective"]
    parent = tqg._group_of(flat_o, include_TR=False, spinor=False)
    assert parent.number_str == "47" and tie.descent["from"]["number_str"] == "47"
    axes = tqg._setting_axes(parent)
    twofold = [next(axes[i][2] for i in s if axes[i][:2] == (1, 2)) for s in step["tied"]]
    assert twofold == [(0, 0, 1), (1, 0, 0), (0, 1, 0)], "c, then a, then b"
    assert tuple(step["kept"]) == step["tied"][0]
    assert tie.descent["dropped_by"]["projective class"] == 4 and "the first of 3" in tie.descent["note"]
    assert ("since its principal axis, [001], comes before [100] in the conventional "
            "cell") in tie.descent["note"], "the reason the first was kept, as every tie says"
    assert tie.rep.product_residual < 1e-12
    # ... and the four it leaves out, combined with time reversal, make with it
    # a magnetic group of type III
    beyond = tie.beyond_grey
    assert beyond is not None and beyond["magnetic_type"] == "III"
    assert (beyond["n_operations"], beyond["n_measured"]) == (4, 8)
    assert sorted(o["name"] for o in beyond["operations"]) == sorted(
        b["name"] + "'" for b in tie.descent["dropped"])
    assert (tie.time_reversal["n_obeyed"], tie.time_reversal["n_broken"]) == (4, 4)
    declared = tqg.realized_group(flat_o, spin="interleaved")
    assert declared, str(declared)
    assert (declared.group.number_str, len(declared.validated.indices)) == ("47.250", 16)
    assert declared.descent is None and declared.time_reversal["outcome"] == "kept"
    assert declared.conserved["dim"] == 0, "nothing but the identity commutes with H"
    # ... and the sibling's bonds on the tetragonal cell descend in two steps:
    # the hoppings keep Pmmm1', time reversal is measured broken there by its
    # square, and the class lowers the unitary Pmmm.  The note counts what the
    # whole descent dropped, by cause, and the counts add up to what it reached
    stepped_flat = tqsp.spinless_presentation(
        _pxpy_spinor(ops=PMMM_OPS, bonds=((1, 1, 0), (1, 0, 1))), order="interleaved")[0]
    with pytest.warns(MagneticScopeWarning):
        stepped = tqg.realized_group(stepped_flat)
    assert stepped, str(stepped)
    d = stepped.descent
    assert [(st["from"]["number_str"], st["to"]["number_str"]) for st in d["steps"]] == [
        ("123.340", "47.250"), ("47", "10")]
    assert ("the cell allowed P4/mmm1' #123.340 (32 coset representatives of the lattice "
            "translations) and the Hamiltonian broke 16 "
            "of them by its hoppings, and time reversal is measured broken, which leaves out "
            "the 8 operations combined with it, and 4 more, which the Hamiltonian obeys, carry "
            "no single-valued representation on these states, so the cell was lowered to the "
            "largest group of its unitary operations that the Hamiltonian obeys and on which "
            "these states carry a single-valued representation, and re-identified as P2/m "
            "#10.") in d["note"], d["note"]
    assert ("descended from #123.340 in 2 steps (28 of 32 coset representatives of the "
            "lattice translations dropped: 16 by the hoppings, 8 with time reversal, 4 by the "
            "states' projective class)") in repr(stepped)
    # ... and the operations a step leaves out to close a group, or that belong
    # only to the subgroups the tie rule did not keep, are causes of their own,
    # which no shipped model's descent meets: read on a step written for them
    steps = [{"from": {"number_str": "47", "size": 8}, "to": {"number_str": "3", "size": 2},
              "dropped": [{"refuted_by": why} for why in
                          ("no symmetry matrix",) * 3 + ("closure",) * 2 + ("tie",)]}]
    counts = tqg._descent_counts(steps, None)
    assert counts == {"centres": 0, "hoppings": 3, "declaration": 0, "closure": 2, "tie": 1,
                      "time reversal": 0, "projective class": 0}
    assert tqg._descent_clauses(counts) == (
        "the Hamiltonian broke 3 of them by its hoppings, and 2 more are left out so that the "
        "operations kept form a group, and 1 more belongs only to the subgroups the tie rule "
        "did not keep",
        "the largest group of its operations that the Hamiltonian obeys, the first by the tie "
        "rule")

    # a flat layer: Kane-Mele with its Rashba term, read without spin=.  The
    # class is read once, free-standing, where each operation composed with the
    # horizontal mirror has its partner's matrix times the mirror's sign, the
    # even one the lowest-angular-momentum rule assigns; on a substrate the
    # side-keeping part of that reading is carried, and the two agree
    from pythtb.models import kane_mele

    layer = tqsp.spinless_presentation(kane_mele(0.0, -1.0, 0.3, 0.25))[0]
    free = tqg.realized_group(layer, environment="free-standing", cross_check=False)
    assert free, str(free)
    assert (free.group.name, free.group.number_str, free.group.size) == ("P6/m", "175", 12)
    d = free.descent
    assert (d["from"]["number_str"], d["dropped_by"]) == ("191", {"projective class": 12})
    assert sorted(b["name"] for b in d["dropped"]) == sorted(
        f"{kind}({axis})" for kind in ("C2", "m")
        for axis in ("x", "y", "[010]", "[110]", "[1-10]", "[2-10]"))
    step = d["steps"][0]["projective"]
    assert step["horizontal_mirror"] == {"sign": 1, "determined_by": "lowest-l"}
    assert len(step["tied"]) == 3 and tuple(step["kept"]) == step["tied"][0], \
        "6/m before the two -6m2: the highest-order rotation"
    assert "+1, the even one the lowest-angular-momentum rule assigns" in d["note"]
    assert all(label.endswith("'") for label in free.rep.characters.irreps), "even under m(z)"
    assert free.rep.product_residual < 1e-12
    sub = tqg.realized_group(layer)
    assert sub, str(sub)
    assert (sub.group.name, sub.group.number_str, sub.environment) == ("P6", "168", "substrate")
    step = sub.descent["steps"][0]["projective"]
    assert step["read_on"] == {"number_str": "191", "name": "P6/mmm", "size": 24, "kept": 12}
    assert sorted(b["name"] for b in sub.descent["dropped"]) == sorted(
        f"m({axis})" for axis in ("x", "y", "[010]", "[110]", "[1-10]", "[2-10]"))
    assert "The class is read on the free-standing layer, P6/mmm #191" in sub.descent["note"]
    assert "since its highest rotation is 6-fold where the next one's is 3-fold" in (
        sub.descent["note"]), "the tie settled free-standing, and why"
    assert sub.cross_check["outcome"] == "agree"
    assert sub.cross_check["free_standing"]["number_str"] == "175"
    # undeclared, each environment names its orbitals by the rule in its own site
    # groups (p_z and p_- on a substrate, p_+ and p_- where m(z) keeps p_z apart);
    # declared, nothing is left to the rule, and the free-standing reading
    # restricts to the substrate's: site irreps, and the bands' traces at K
    pairs = {0: "px", 1: "py", 2: "px", 3: "py"}
    free_p = tqg.realized_group(layer, environment="free-standing", characters=pairs,
                                cross_check=False)
    sub_p = tqg.realized_group(layer, characters=pairs)
    assert free_p and sub_p, (str(free_p), str(sub_p))
    assert (free_p.group.number_str, sub_p.group.number_str) == ("175", "168")
    assert sub_p.cross_check["outcome"] == "agree"
    assert [(o.irrep, o.declared_irrep) for o in free_p.rep.characters.orbitals] == \
        [("2E'", "E'"), ("1E'", "E'")] * 2, \
        "each row its own multiplet's: the threefold carries a site's two orbitals as " \
        "the complex pair, and the p_x, p_y declared on them name the doublet they make"
    assert sub_p.rep.characters.irreps == ("2E", "1E") * 2
    k_free = tqw.hsp_table(free_p.rep, bands=2).records["K"].table
    k_sub = tqw.hsp_table(sub_p.rep, bands=2).records["K"].table
    assert tuple(k_free.columns[:3]) == tuple(k_sub.columns) == (1, 2, 3), "E and the threefolds"
    assert np.abs(np.asarray(k_free.characters)[:, :3] - np.asarray(k_sub.characters)).max() < 1e-9

    # ... and with its bonds along the axes alone, a pseudospin the hoppings
    # conserve: the commutant supplies a time reversal of square +1, so the
    # spinless reading keeps Pmmm1', and time reversal carries -1 on the p_y pair
    # relative to the p_x pair -- the one sign its own solution space pins
    axes = tqsp.spinless_presentation(
        _pxpy_spinor(b=1.17, ops=PMMM_OPS, bonds=((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        order="interleaved")[0]
    grey = tqg.realized_group(axes)
    assert grey, str(grey)
    assert (grey.group.number_str, len(grey.validated.indices)) == ("47.250", 16)
    assert grey.time_reversal["outcome"] == "kept" and grey.conserved["dim"] == 3
    assert grey.time_reversal["square"]["measured"] == pytest.approx(1.0, abs=1e-12)
    reversal = np.asarray(grey.rep.antiunitary_operator[1])
    assert np.allclose(reversal, reversal[0, 0] * np.diag([1, 1, -1, -1]), atol=1e-12)
    assert max(grey.rep.residuals.values()) < 1e-13 and grey.rep.product_residual < 1e-12

    # ... and the tetragonal doublet with its bonds along the axes alone: the same
    # pseudospin, on one fourfold multiplet, whose copies' representation moves
    # the site content and is assigned by the lowest-angular-momentum rule
    doublet = tqsp.spinless_presentation(
        _pxpy_spinor(bonds=((1, 0, 0), (0, 1, 0), (0, 0, 1))), order="interleaved")[0]
    pseudo = tqg.realized_group(doublet)
    assert pseudo, str(pseudo)
    assert (pseudo.group.number_str, len(pseudo.validated.indices)) == ("123.340", 32)
    assert pseudo.time_reversal["square"]["measured"] == pytest.approx(1.0, abs=1e-12)
    assert pseudo.rep.multiplets.dims == ((4,),) and pseudo.conserved["dim"] == 3
    assert [(s["orbitals"], s["copies"], s["dim"]) for s in pseudo.conserved["sectors"]] == \
        [((0, 1, 2, 3), 2, 2)], "the pseudospin: two identical copies of the p_x, p_y pair"
    (space,) = pseudo.rep.characters.multiplicity_spaces
    assert (space.orbitals, space.wyckoff, space.dim, space.n_representations,
            space.determined_by) == ((0, 1, 2, 3), "1a", 2, 38, "lowest-l")
    assert space.content == space.contents[0] == "2xEu" and len(space.contents) == 12
    assert {"2xEg", "2xB2u + 2xA2u"} <= set(space.contents)
    assert pseudo.rep.characters.irreps == ("Eu",) * 4
    assert "twofold pseudospin" in pseudo.rep.characters.summary()
    assert max(pseudo.rep.residuals.values()) < 1e-13 and pseudo.rep.product_residual < 1e-12
    lowest = tqw.hsp_table(pseudo.rep, bands=2)
    assert lowest.complete and lowest.counts == {"GM5-": 1, "A5-": 1, "M5-": 1, "R3-": 1,
                                                 "R4-": 1, "X3-": 1, "X4-": 1, "Z5-": 1}
    assert "pseudospin" in lowest.note and "2xEu" in lowest.note
    # ... and copies every representation of which gives one site content: a
    # labelling like any other, under the one disclaimer, which names the pseudospin
    note = tqw.hsp_table(tqg.realized_group(tqc_models.p1bar_s_copies()).rep, bands=2).note
    assert note.count("the labels of these") == 1, note
    assert ", on which the Hamiltonian conserves a pseudospin: another choice" in note, note
    # ... declared four copies of A1g, which no representation of the copies
    # carries on P4/mmm: the group descends to Pmmm1', every operation left
    # out broken by the declaration
    false = tqg.realized_group(doublet, characters={(0, 1, 2, 3): "4xA1g"})
    assert false, str(false)
    assert (false.group.number_str, len(false.validated.indices)) == ("47.250", 16)
    assert false.descent["steps"][0]["refuted_by"] == {"declaration": 16}
    assert {o.determined_by for o in false.rep.characters.orbitals} == {"declaration"}

    zeeman = tqm.i41_atomic()
    zeeman.set_onsite([0.3 * np.diag([1.0, -1.0])] * int(zeeman.norb), mode="add")
    with pytest.warns(MagneticScopeWarning):
        spin_free = tqg.realized_group(zeeman)
    assert spin_free, str(spin_free)
    assert (spin_free.group.number_str, spin_free.group.size) == ("80", 4)
    reversal = spin_free.time_reversal
    assert reversal["outcome"] == "measured broken" and reversal["square"]["asked"] == -1
    assert reversal["square"]["measured"] == pytest.approx(1.0, abs=1e-12)
    assert "a spin-free time reversal" in reversal["note"]


@pytest.mark.slow
def test_every_spelling_of_spin_builds_the_same_representation(km_flat, pxpy_spinor):
    """``spin=`` takes the object, the shorthand, or the constructor's keywords.

    The shorthand exists so the common case is one word; the dict so a
    caller can hand ``spin_basis``'s own arguments through without importing
    it.  All of them resolve to the same declaration and therefore to the same
    matrices, bit for bit.

    ``realized_group`` resolves the spelling once, where its measurement
    starts, and every step after that is handed the declaration itself -- the
    verdict stage included, which on the flattened ``(p_x, p_y)`` doublet lifts
    the multiplet permutation to the two declared pairs the doublet holds.  So
    the word returns the native model's group and its content as the
    declaration itself does.
    """
    _km, flat, basis, group, assignment, validated = km_flat
    spellings = ("interleaved", {"order": "interleaved"},
                 {"pairs": [(0, 1), (2, 3)]}, basis)
    built = []
    for spec in spellings:
        rep = tqs._build_rep(flat, group, assignment, validated=validated, spin=spec)
        assert rep, f"{spec!r}: {rep}"
        assert rep.spin.pairs == basis.pairs
        built.append(rep)
    for rep in built[1:]:
        assert all(float(np.abs(np.asarray(rep.matrices[i])
                                - np.asarray(built[0].matrices[i])).max()) == 0.0
                   for i in built[0].indices)
    assert built[2].spin.kind == "pairs", "the explicit form keeps its own name"

    native, flat_p, basis_p = pxpy_spinor
    want = tqw.hsp_table(native.rep, bands=2)
    assert want.complete, "the native content is read at every point"
    got = tqg.realized_group(flat_p, spin="interleaved")
    assert got, str(got)
    assert (got.group.number_str, len(got.validated.indices)) == ("123.340", 32)
    assert got.rep.spin.pairs == basis_p.pairs
    assert got.rep.multiplets.dims == ((2,),), "one doublet of declared pairs"
    assert dict(tqw.hsp_table(got.rep, bands=2).counts) == dict(want.counts)


def test_a_wrong_basis_size_is_a_grammar_error(km_flat):
    """A declaration belongs to the model it was written for.

    And a native spinful model takes none: it already has a spin axis and
    ``_build_rep`` already uses it, so there is nothing to declare and nothing to
    report as a result -- a grammar mistake, not a measurement.
    """
    km, flat, _basis, group, assignment, validated = km_flat
    other = tqsp.spin_basis(
        tqsp.spinless_presentation(tqc_models.lin3g())[0], order="interleaved")
    assert other.norb == 8 != int(flat.norb)
    with pytest.raises(ValueError, match="8 orbitals and this one has 4"):
        tqs._build_rep(flat, group, assignment, validated=validated, spin=other)
    with pytest.raises(ValueError, match="SpinBasis, the shorthand"):
        tqs._build_rep(flat, group, assignment, validated=validated, spin=17)

    # the native spinful model, in the group of the same cell
    with pytest.raises(ValueError, match="natively spinful"):
        tqs._build_rep(km, group, tqg._assign_orbitals(km, group), spin="interleaved")
    with pytest.raises(ValueError, match="spinor=False"):
        tqg.realized_group(km, spin="interleaved", spinor=False)


# ---------------------------------------------------------------------------
# 5.  the layout a declaration measures, and declarations on a declared basis
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_declared_pair_reads_the_native_content_and_site_irrep(km_flat):
    """A declared pair is one Wannier function, and reads what the native model reads.

    Structurally all a declaration does is lift the refusal a spinless model
    in double-valued tables gets; what it then decides is the layout.  A
    declared pair is one Wannier function, so each site of this flattening is
    one **one-dimensional** multiplet whose internal factor is the tables' own
    SU(2) lift: the native spinful model's arithmetic (the re-encoding theorem
    above), and its physics -- one group, one corep content and one partition
    of the bands, point by point ``-GM6``, ``-K5 + -K6``, ``-M3M4``, and beside
    them the site content they come from, ``-E1`` of ``3m`` on both members of
    each pair, the Hamiltonian's own and the native model's.

    Asked in the **wallpaper** reading, which is the group this comparison is
    made in: ``typat=[0, 1]`` distinguishes the two sublattices, and the
    in-plane group that survives is ``P3m11' #156.50``, 12 of 12.
    """
    _km, flat, basis, _group, _assignment, _validated = km_flat
    got = tqg.realized_group(flat, spin=basis, typat=[0, 0, 1, 1],
                             environment="substrate")
    assert got, f"{basis.kind}: {got}"
    # the native spinful model's own record, read on a substrate
    assert got.group.number_str == "156.50" and got.group.size == 12
    tab = tqw.hsp_table(got.rep, bands=2)
    assert tab.complete
    assert dict(tab.counts) == {"-GM6": 1, "-K5": 1, "-K6": 1, "-M3M4": 1}
    assert {p: dict(v) for p, v in tab.content.items()} == {
        "GM": {"-GM6": 1}, "K": {"-K5": 1, "-K6": 1}, "M": {"-M3M4": 1}}
    assert dict(tab.partitions) == {"GM": (0,), "K": (0, 1), "M": (0,)}
    assert got.rep.spin is basis
    assert got.rep._reference["spin_pair"] == 0
    # one pair on each site, a one-dimensional multiplet carrying the native site irrep
    assert got.rep.multiplets.by_site == (((0, 1),), ((2, 3),))
    assert got.rep.multiplets.dims == ((1,), (1,))
    assert got.rep.characters.irreps == ("-E1",) * 4
    assert [o.determined_by for o in got.rep.characters.orbitals] == ["hamiltonian"] * 4
    assert got.rep.characters.residual < 1e-12


@pytest.mark.slow
def test_a_declaration_on_a_declared_basis_is_about_the_pairs_spatial_character(km_flat):
    """``characters=`` names channels, so a pair takes one declaration.

    Written out (``2N`` entries, both members of a pair agreeing) or as a
    mapping keyed on either member, it is the same statement and it fixes the
    same labelling as the native model's own declaration.  A tuple key naming
    both members is one multiplet of dimension one -- not a doublet.
    """
    km, flat, basis, group, assignment, validated = km_flat
    gN = tqg._group_of(km)
    asgN = tqg._assign_orbitals(km, gN)
    valN = tqg._validate_group(km, gN, nk=6)

    def _flat_rep(**kw):
        return tqs._build_rep(flat, group, assignment, validated=validated, spin=basis,
                              **kw)

    native = tqs._build_rep(km, gN, asgN, validated=valN, characters=["s", "s"])
    assert native, str(native)
    ordered = _flat_rep(characters=["s", "s", "s", "s"])
    assert ordered, str(ordered)
    assert ordered.characters.irreps == ("-E1",) * 4
    assert native.characters.irreps == ("-E1", "-E1"), (
        "both are named by their point group: `_build_rep` was handed the group the "
        "cell allows and represents only what the flat Hamiltonian obeys, so the "
        "site group here is a proper subgroup of the one the tables key 2c by, and "
        "the entries there name none of its rows")
    assert {o.named_by for o in native.characters.orbitals + ordered.characters.orbitals} \
        == {"point group"}
    assert [o.determined_by for o in ordered.characters.orbitals] == ["declaration"] * 4
    assert [o.determined_by for o in native.characters.orbitals] == ["declaration"] * 2, \
        "every orbital declared on both sides, so no row rests on the lowest-l rule"

    native_one = tqs._build_rep(km, gN, asgN, validated=valN, characters={0: "s"})
    mapped = _flat_rep(characters={0: "s"})
    assert mapped, str(mapped)
    assert [o.determined_by for o in mapped.characters.orbitals[::2]] == \
        [o.determined_by for o in native_one.characters.orbitals]
    assert mapped.characters.irreps[0] == native_one.characters.irreps[0]
    # the declaration reaches both members of the pair, and only that pair
    assert mapped.characters.orbitals[0].determined_by == "declaration"
    assert mapped.characters.orbitals[1].determined_by == "declaration"
    assert mapped.characters.orbitals[1].declared_as == "s"

    joint = _flat_rep(characters={(0, 1): "s"})
    assert joint, str(joint)
    assert len(joint.characters.declarations) == 1
    decl = joint.characters.declarations[0]
    assert decl["orbitals"] == (0, 1) and decl["dim"] == 1 and decl["declared_irrep"] == "A1"
    assert max(float(np.abs(np.asarray(joint.matrices[i])
                            - np.asarray(mapped.matrices[i])).max())
               for i in mapped.indices) == 0.0, "one declaration, one representation"


def test_a_pair_given_two_different_characters_is_a_grammar_error(km_flat, monkeypatch):
    """One Wannier function, one character.

    The up and the down state of one spinor are one orbital function times
    the two spin states, so they carry one site irrep.  Declaring them
    differently is a contradiction inside the declaration, and it raises
    ``ValueError`` naming the pair, its orbitals and both words -- in
    ``realized_group`` before anything is identified or measured.
    """
    _km, flat, basis, group, assignment, validated = km_flat
    for characters in (["s", "pz", None, None], {0: "s", 1: "pz"}):
        with pytest.raises(ValueError, match="spin pair 0") as raised:
            tqs._build_rep(flat, group, assignment, validated=validated, spin=basis,
                           characters=characters)
        assert "one Wannier function" in str(raised.value)
        assert "'s' and 'pz'" in str(raised.value) and "Orbitals 0 and 1" in str(raised.value)
    # ... and the two spellings that are not a contradiction still build
    assert tqs._build_rep(flat, group, assignment, validated=validated, spin=basis,
                          characters=["s", None, None, None])
    assert tqs._build_rep(flat, group, assignment, validated=validated, spin=basis,
                          characters=["s", "s", None, None])

    def measured(*args, **kwargs):
        raise AssertionError("the declaration's grammar is read before any group")

    monkeypatch.setattr(tqg, "_group_of", measured)
    with pytest.raises(ValueError, match="spin pair 0"):
        tqg.realized_group(flat, spin="interleaved", characters={0: "s", 1: "pz"})
    with pytest.raises(ValueError, match="split between"):
        tqg.realized_group(flat, spin="interleaved", multiplets=[[(0,), (1,)], [(2, 3)]])


# ---------------------------------------------------------------------------
# 6.  the passthroughs
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_padded_slab_operations_are_represented_too():
    """The ``sigma_h`` presentation, on a declared basis: all 48 operations.

    Kane-Mele at ``rashba = 0`` on the bare slab cell ``P6/mmm1' #191.234`` is
    the one presentation where the horizontal mirror and everything composed
    with it can be represented -- ``include_vacuum_ops=True``, which the
    detector cannot refute and only a built representation can settle.  The
    pair permutation then has to be lifted for operations on the vacuum
    direction too.  Measured: all 48 operations are represented at the 1e-15
    level, identical to the native build under ``Pi`` at ``0.0``.
    """
    km = tqc_models.kane_mele_inversion()
    flat, basis = tqsp.spinless_presentation(km, order="interleaved")
    slab = tqg._slab_presentation(flat)
    group = tqg._group_of(slab, spinor=True)
    assert group and group.number_str == "191.234" and group.size == 48
    assignment = tqg._assign_orbitals(slab, group)
    validated = tqg._validate_group(slab, group, nk=4)
    assert len(validated.indices) == 48

    native_slab = tqg._slab_presentation(km)
    ngroup = tqg._group_of(native_slab, spinor=True)
    nasg = tqg._assign_orbitals(native_slab, ngroup)
    nval = tqg._validate_group(native_slab, ngroup, nk=4)

    got = tqs._build_rep(slab, group, assignment, validated=validated, spin=basis,
                         n_battery=4, include_vacuum_ops=True)
    assert got, str(got)
    assert len(got.indices) == 48
    assert max(got.residuals.values()) <= GATE

    want = tqs._build_rep(native_slab, ngroup, nasg, validated=nval, n_battery=4,
                          include_vacuum_ops=True)
    assert want, str(want)
    Pi = basis.permutation_matrix
    worst = max(float(np.abs(Pi.T @ np.asarray(got.matrices[i]) @ Pi
                             - np.asarray(want.matrices[i])).max()) for i in want.indices)
    assert worst == 0.0
    assert got.product_residual == pytest.approx(want.product_residual, abs=1e-15)


@pytest.mark.slow
def test_the_indicator_class_of_fu_kane_mele_survives_the_re_encoding(realized):
    """The 3D literature anchor: FKM's strong phase, read off a spinless model.

    Distorted diamond with one non-orbital species at the strengthened bond's
    midpoint re-identifies as ``R-3m1' #166.98``; the valence pair sits at
    ``(0, 1)`` of order 4 in ``X_BS = Z2 x Z4``, i.e. ``z4 = 3`` with weak
    indices ``1, 1, 1`` -- Fu-Kane's ``(1; 111)``.  Re-presented as four
    spinless orbitals in the **blocked** layout, a genuine reordering of the
    index, and re-labelled through the declaration, it is the same crystal, the
    same content and the same class, with the representation itself identical
    under ``Pi`` to machine precision.  It is the one 3D grey crystal the
    re-encoding is measured on.
    """
    # the native record: the session's shared build of this call
    native = realized(fkm_regime("strong"), decorations=FKM_BOND, characters={0: "s"})
    assert native and native.group.number_str == "166.98", str(native)
    tabN = tqw.hsp_table(native.rep, bands=2)
    want = tqe.symmetry_indicator(tabN, native.group, rep=native.rep, bands=2)
    assert want and want.values == (0, 1) and want.order == 4
    assert want.structure == "Z2 x Z4" and want.irreptables_named["z4"] == 3

    fkm = tqc_models.fu_kane_mele_z2("strong")
    flat, basis = tqsp.spinless_presentation(fkm, order="blocked")
    got_rg = tqg.realized_group(flat, spin=basis, decorations=FKM_BOND,
                                characters={0: "s"})
    assert got_rg, str(got_rg)
    assert got_rg.group.number_str == "166.98"
    tab = tqw.hsp_table(got_rg.rep, bands=2)
    assert dict(tab.counts) == dict(tabN.counts) and tab.partitions == tabN.partitions

    got = tqe.symmetry_indicator(tab, got_rg.group, rep=got_rg.rep, bands=2)
    assert got, str(got)
    assert got.values == want.values and got.order == want.order
    assert got.irreptables_named == want.irreptables_named
    assert got.verdict == want.verdict == "indicated nontrivial"

    Pi = basis.permutation_matrix
    worst = max(float(np.abs(Pi.T @ np.asarray(got_rg.rep.matrices[i]) @ Pi
                             - np.asarray(native.rep.matrices[i])).max())
                for i in native.rep.indices)
    assert worst <= REENCODE_TOL


# ---------------------------------------------------------------------------
# 7.  the quantization axis: pythtb's convention, stated and applied by hand
# ---------------------------------------------------------------------------

def _onto_the_papers_axis(model):
    """The fixture's ``2x2`` spin blocks conjugated by ``W = i sigma_x``.

    Which, applied to ``p41_flat_band``, is the conversion run **backwards**:
    the fixture types the supplement's Hamiltonian already conjugated onto
    pythtb's axis, and ``W^2 = -1`` makes the conjugation its own inverse, so
    this returns the paper's own blocks -- the model in the frame quantized
    along ``-z``.  It is written out here rather than imported because the
    package exposes no such helper -- a conversion between two papers' spin
    conventions is not the package's to hand an end user -- and these four
    lines are the whole of it.
    """
    from pythtb import Lattice, TBModel

    W = 1j * np.array([[0, 1], [1, 0]], dtype=complex)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)), spinful=True)
    blocks = np.asarray(model.onsite, dtype=complex).reshape(int(model.norb), 2, 2)
    out.set_onsite([W @ b @ W.conj().T for b in blocks], mode="set")
    for h in model.hoppings:
        amp = np.asarray(h["amplitude"], dtype=complex).reshape(2, 2)
        out.set_hop(W @ amp @ W.conj().T, int(h["from_orbital"]), int(h["to_orbital"]),
                    list(h.get("lattice_vector", [0] * int(model.lattice.dim_r))),
                    mode="add")
    return out


def test_the_shipped_fixture_is_the_papers_closed_form_conjugated_onto_pythtbs_axis():
    """``H_fixture(k) = U H_closed(k) U†`` with ``U = 1_2 (x) i sigma_x``.

    The tie that makes ``p41_flat_band`` trustworthy, and the only one: the
    fixture types a finite hopping list, the paper writes a closed form
    ``t (u+ u+† + u- u-†)``, and they are the same Hamiltonian in two spin
    frames, which differ by the *global* unitary ``U``.  Measured ``2.220e-16``
    over 64 random ``k``.
    """
    fixture = tqc_models.p41_flat_band()
    W = 1j * np.array([[0, 1], [1, 0]], dtype=complex)   # unitary by inspection
    U = np.kron(np.eye(int(fixture.norb)), W)

    k = np.random.default_rng(20260905).uniform(-0.5, 0.5, size=(64, 2))
    a = np.asarray(fixture.hamiltonian(k_pts=k, flatten_spin_axis=True))
    closed = np.asarray([tqc_models.p41_flat_bloch_hamiltonian(kk) for kk in k])
    assert float(np.abs(a - U @ closed @ U.conj().T).max()) < 1e-14, \
        "the fixture IS the paper's closed form, conjugated onto pythtb's axis"


@pytest.mark.slow
def test_the_transcription_in_pythtbs_convention_is_labelled_s_in_its_own_group():
    """The paper's ``s`` on the ``2c`` pair is the label each frame's own group serves.

    The SRSI paper writes its fourfold operator as ``tau_1 . e^{+i pi sigma_3 / 4}``,
    the opposite sign from pythtb's ``e^{-i theta n.sigma / 2}``.  Since wave R4d
    the library does not flip spin axes, so the paper's frame (``_onto_the_papers_axis``
    above, the same four lines the retired ``flip_quantization_axis`` ran) reads a
    *different group* -- free-standing ``P-4m21' #115.284`` where ours is
    ``P4mm1' #99.164`` -- and there is no page on which the two models' blocks can
    be compared operation by operation; tests/test_layer.py pins those groups and
    the closed internal-factor gate that refuses the cross-build.

    What is measured here is the label.  The transcription in pythtb's convention
    reads ``-E`` on both sites in ``#99.164`` and ``-1E2E`` on the descended
    ``P41' #75.2`` without any declaration (the stored baseline's
    ``p41_flat_band`` and ``p41_chiral`` records, on a substrate); the paper's own
    declaration, ``s`` on the ``2c`` pair, is accepted without a contradiction in
    every frame -- ours in both groups, and the paper's in its own -- and serves
    the same label there.

    A frame turned by less than a declared precision is the tables' frame, read
    within the declaration.  Kane-Mele with its Rashba term, every spin block
    turned by ``1e-6`` rad about a direction of no symmetry, under
    ``precision=1e-5``: the detector's members absorb the turn and read every
    operation exact, while the matrices the tables' spin lift anchors miss ``H``
    by the turn and read near, so the projection makes them exact and
    ``P6mm1' #183.186`` is returned, both sites ``-E1``, ``H`` moved by about the
    turn.  With ``symmetrize=False`` the exact gate refuses, inside the declared
    tolerance.
    """
    fixture = tqc_models.p41_flat_band()
    paper = _onto_the_papers_axis(fixture)            # the involution: back to the paper's

    declared = tqg.realized_group(fixture, environment="substrate", characters={0: "s"})
    assert declared, str(declared)
    assert (declared.group.number_str, len(declared.validated.indices)) == ("99.164", 16)
    assert declared.rep.characters.irreps == ("-E", "-E")

    # the chiral pattern, on a substrate
    declared_down = tqg.realized_group(tqc_models.p41_flat_band(chiral=True),
                                       environment="substrate", characters={0: "s"})
    assert declared_down and declared_down.group.number_str == "75.2"
    assert declared_down.rep.characters.irreps == ("-1E2E", "-1E2E")

    on_the_layer = tqg.realized_group(paper, environment="free-standing",
                                      characters={0: "s"})
    assert on_the_layer, str(on_the_layer)
    assert on_the_layer.group.number_str == "115.284"
    assert on_the_layer.rep.characters.irreps == ("-E", "-E")

    # a spin frame turned by 1e-6 rad, under a precision of 1e-5
    n = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
    sigma = (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))
    w = np.cos(5e-7) * np.eye(2) - 1j * np.sin(5e-7) * sum(a * s for a, s in zip(n, sigma))
    turned = spin_blocks_turned(kane_mele(0.0, -1.0, 0.3, 0.25), w)
    read = tqg.realized_group(turned, precision=1e-5)
    assert read, str(read)
    assert (read.group.name, read.group.number_str, read.group.size) == (
        "P6mm1'", "183.186", 24)
    assert read.descent is None and read.rep.characters.irreps == ("-E1", "-E1")
    rec = read.symmetrization
    assert rec.near and 1e-7 < rec.defect < 1e-5, "H moved by about the turn"
    kept = tqg.realized_group(turned, precision=1e-5, symmetrize=False)
    assert kept.gate == "rep_not_covariant", str(kept)
    assert 1e-7 < kept.value < read.precision["tol"]
    assert kept.context["group"].number_str == "183.186"


def test_a_wrong_but_well_formed_pairing_dies_before_anything_is_built(km_flat):
    """A `spin=` declaration that is grammatical and wrong is an *ansatz*.

    Exchanging up and down in every pair (``pairs=[(1, 0), (3, 2)]`` on the
    interleaved Kane-Mele flattening) declares a spinor basis that exists -- same
    sites, same partition -- and places the SU(2) lift on the wrong two-state
    frame.  No dedicated gate can see that; it is exactly what reading each
    operation's space in its own form measures, and handed the cell's group the
    build stops there, before anything is built: the matrix of its own form
    nearest any of the eight operations' spaces carries ``H(k)`` onto
    ``H(Rk)`` to ``1.3`` against the ``1e-8`` they were kept at, each named
    with its spin lift.  The operations a representation is built for are the
    ones measured to hold in their own form, so the build raises, and so it
    does for Kane-Mele itself with the padded slab operations forced in, none
    of which carries its own spin lift.  ``realized_group`` reads the same form
    first and descends instead.
    """
    km, flat, sb, group, assignment, validated = km_flat
    assert sb.kind == "interleaved" and sb.pairs == ((0, 1), (2, 3))

    good = tqs._build_rep(flat, group, assignment, validated=validated, spin=sb)
    assert good and good._search["n_survivors"] == 1, str(good)

    with pytest.raises(InternalError) as err:
        tqs._build_rep(flat, group, assignment, validated=validated,
                       spin={"pairs": [(1, 0), (3, 2)]})
    said = str(err.value)
    assert "8 of the" in said and said.count("its spin lift") == 8, said
    assert "H(Rk) to 1.3," in said, said

    g0 = tqg._group_of(km)
    with pytest.raises(InternalError, match="its spin lift") as err:
        tqs._build_rep(km, g0, tqg._assign_orbitals(km, g0),
                       validated=tqg._validate_group(km, g0, nk=6),
                       include_vacuum_ops=True)
    assert "site permutation" not in str(err.value)
