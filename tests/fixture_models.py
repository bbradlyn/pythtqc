"""Golden tight-binding fixtures for the ``tqc`` (topological quantum chemistry) suite.

Three models, all built with the PythTB **v2.0** API (:class:`pythtb.Lattice` /
:class:`pythtb.TBModel`).  Every builder is a plain deterministic function that
returns a freshly constructed model; nothing here touches the filesystem or the
network at import time.

===========================  ============================================================
fixture                      what it is
===========================  ============================================================
:func:`lin3g`                2D spinful fragile quartet + s@2c EBR, wallpaper p4mm,
                             8 bands.  Lin et al., Nat. Commun. **15**, 550
                             (2024), SM Sec. 3G.  *Primary golden.*
:func:`kane_mele_z2`         2D spinful honeycomb Kane-Mele, 4 bands, in both its
                             Z2-nontrivial (``"odd"``) and Z2-trivial (``"even"``)
                             parameter regimes.
:func:`p4232_cano2018`       3D spinless nonsymmorphic cubic model, 4 bands, the gapped
                             ("disconnected") EBR of Cano *et al.*, PRL **120**, 266401
                             (2018) = arXiv:1711.11045.
:func:`fu_kane_mele_z2`      3D spinful diamond (SG 227), 4 bands, in four regimes:
                             undistorted semimetal, strong TI (1;111), weak TI (0;111)
                             and the trivial dimerized limit.  Fu, Kane & Mele, PRL
                             **98**, 106803 (2007).
:func:`kane_mele_inversion`  2D spinful honeycomb Kane-Mele at ``delta = rashba = 0``,
                             4 bands; the slab cell reads P6/mmm (#191) and the
                             Hamiltonian realizes P6mm (#183).  QSH throughout.
:func:`pwv_fragile`          2D spinful honeycomb, 4 (or 6) bands, wallpaper p6mm1';
                             the fragile model of Po, Watanabe & Vishwanath, PRL **121**,
                             126402 (2018).  Its CONDUCTION pair is the fragile one.
:func:`pxpy_honeycomb`       2D **spinless** honeycomb with a ``(p_x, p_y)`` doublet per
                             site, 4 bands, in a gapped ("tci") and a semimetallic
                             regime.  Cano *et al.*, PRL **120**, 266401 (2018) =
                             arXiv:1711.11045 -- the same paper as the P4_232 fixture.
                             The only fixture whose sigma_h-composed operations are
                             GENUINE, so its realized slab group is the full #191.
:func:`i213_flicker2018`    3D **spinless** body-centred cubic, 4 bands, SG **199**
                             (I2_1 3).  Flicker *et al.*, PRB **98**, 155145 (2018) =
                             arXiv:1806.09642 App. F.1.  Nonsymmorphic with no
                             intrinsic screw, non-primitive conventional cell, and
                             sites that permute between unit cells.
:func:`p4122_atomic`         3D **spinless** tetragonal, 8 bands, SG **91** (P4_122):
                             ``s`` at 4a below a gap, ``s`` at 4b above it.  A
                             constructed-here atomic control for the 4_1 screw points
                             ``A``/``Z``, where the physical little-group table is the
                             whole-character conjugate of the tabulated one.
:func:`p3121_atomic`         3D **spinless** hexagonal, 6 bands, SG **152** (P3_121 --
                             quartz's group): ``s`` at 3a below, ``s`` at 3b above.  The
                             3_1 companion of the above, whose ``A``/``H``/``L`` labels
                             *discriminate* the two orbits.
:func:`i41_atomic`           3D **spinful** body-centred tetragonal, 12 bands, SG **80**
                             (I4_1): ``s`` at 4a below a gap, ``s`` at the general 8b
                             above.  The anchor of the **re-gluing** class: at the
                             non-TRIM ``P`` the physical corep list is ``{-P3 (1),
                             -P4P4 (2)}`` where BCS's is ``{-P3P3 (2), -P4 (1)}``.
:func:`i41_soc`              :func:`i41_atomic` plus one group-averaged spin-orbit bond
                             set.  The detector control: ``soc="generic"`` is exactly
                             I4_11'-symmetric with commutant ``C*1``, ``soc="axial"``
                             conserves ``s_z`` and has a **two**-dimensional commutant.
                             ``"generic"`` is also the re-gluing demonstration: it splits
                             ``P``'s valence level ``[4]`` into ``[1, 2, 1]`` and the
                             doublet measures as ``-P4P4``.
:func:`p62_atomic`           3D **spinful** hexagonal, 18 bands, SG **171** (P6_2): ``s``
                             at 3a below, ``s`` at the general 6c above.  The second
                             re-gluing member, from a different Bravais family; its
                             re-glued points are ``A`` and the non-TRIM ``H``.
:func:`p41_flat_band`        2D **spinful** square lattice, 4 bands, two Kramers pairs
                             at ``2c``: the SRSI paper's flat-band model
                             (10.1038/s41467-026-74844-w, SM Sec. 5A), transcribed from
                             its closed form.  ``chiral=True`` adds a C4- and
                             T-symmetric next-nearest pattern that breaks the vertical
                             mirrors, which is the fixture the **unitary descent**
                             ``#99.164 -> #75.2`` is measured on.
===========================  ============================================================

.. warning::

   **Space group of the third fixture.**  The group is 208 (P4_232), not 212
   (P4_332).  arXiv:1711.11045 says, verbatim, "*we pick a particular
   non-symmorphic space group, P4_232, and construct a tight-binding model*",
   and it lists the generators ``{C_2x|0}``, ``{C_3,111|0}``,
   ``{C_2,110|1/2 1/2 1/2}`` and the site pair ``(0,0,0)``,
   ``(1/2,1/2,1/2)`` = Wyckoff **2a**.  That is **P4_232 = space group 208**,
   not P4_332 = 212 (whose maximal Wyckoff orbit is the 4-site 8c/4a set at
   ``(1/8,1/8,1/8)``-type positions, and which cannot host a two-site 2a orbit
   at all).  The fixture is therefore built and validated as SG 208.  It is
   still nonsymmorphic (4_2 screw), still cubic, still 3D, and still carries
   the paper's disconnected-EBR physics.

Metadata (band sets, occupancies, group labels, regime names) rides along in
:class:`FixtureMeta` constants next to each builder.

.. note::

   **Promotion of 2026-08-25.**  The shared construction helpers and the model
   families the ``examples/tqc_*.py`` worked examples need -- SG 80 (I4_1),
   SG 171 (P6_2), SG 199 (I2_1 3), SG 208 (P4_232) and the ``lin3g`` layer --
   now live in the public
   :mod:`pythtqc.models`.  They are imported at the top of this module
   and re-exported through :data:`__all__`, so every
   ``tqc_models.<name>`` call site is unchanged and the tests and the examples
   build the **same** object.  What stays here is test-only scaffolding: the
   :class:`FixtureMeta` records with their measured locks, and the fixtures
   that exist only so the pipeline has something to refuse.

The two 2D fixtures also record ``slab_sg_number``: the space group that a
symmetry finder returns for the *slab cell* (the 2D cell embedded in 3D with
vacuum) built from the orbital positions alone.  That number describes the
**cell**, not the Hamiltonian -- see the notes on each fixture.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

import numpy as np
from pythtb import Lattice, TBModel

# Every model builder below the metadata lives in the package now (2026-08-25):
# the tests and the ``examples/tqc_*.py`` worked examples import the SAME
# object, so a fixture can never drift from what a user runs.  Imported here and
# re-exported through ``__all__`` so every existing ``tqc_models.<name>`` call
# site keeps working unchanged.
from pythtqc.models import (  # noqa: E402,F401
    PAULI, I41_CENTERING, I41_LATTICE, I41_OPS, I41_PARAMS, I41_PRIM_OPS,
    I41_SOC_SEEDS, P62_LATTICE, P62_OPS, P62_PARAMS, P4232_NN_VECTORS,
    P4232_PARAMS, I213_BONDS, I213_LATTICE, I213_U, atomic_model,
    cartesian_rotation, closed_bonds, i41_atomic, i41_sites, i41_soc, orbit_of,
    lin3g, p4232_bloch_hamiltonian, p4232_cano2018, p4232_operations,
    p4232_symmetry_cell, p62_atomic, p62_sites, primitive_ops,
    i213_bloch_hamiltonian, i213_conventional, i213_flicker2018, i213_sites,
    spin_bonds, spinor_of)


__all__ = [
    "FixtureMeta",
    "lin3g",
    "lin3g_golden", "PYTHTB18_GOLDEN",
    "spin_operator", "psp_eigenstates", "psp_sector_wfarrays",
    "LIN3G",
    "kane_mele_z2",
    "KANE_MELE",
    "p4232_cano2018",
    "p4232_bloch_hamiltonian",
    "p4232_operations",
    "p4232_symmetry_cell",
    "P4232",
    "P4232_PARAMS",
    "P4232_NN_VECTORS",
    "fu_kane_mele_z2", "FU_KANE_MELE",
    "kane_mele_inversion", "KANE_MELE_INVERSION",
    "pwv_fragile", "pwv_bloch_hamiltonian", "pwv_operations",
    "PWV", "PWV6", "PWV_BOND_PARAMS", "PWV_BOND_GEOMETRY", "PWV_LATTICE", "PWV_SITES",
    "pxpy_honeycomb", "pxpy_bloch_hamiltonian", "pxpy_slater_koster",
    "PXPY", "PXPY_PARAMS", "PXPY_LATTICE", "PXPY_SITES", "PXPY_DELTAS",
    "PXPY_E1", "PXPY_E2",
    "i213_flicker2018", "i213_bloch_hamiltonian", "i213_sites", "i213_conventional",
    "I213", "I213_LATTICE", "I213_U", "I213_BONDS",
    "orbit_of", "closed_bonds", "atomic_model",
    "spin_bonds", "spinor_of", "cartesian_rotation", "PAULI",
    "p4122_atomic", "p4122_sites", "P4122", "P4122_OPS", "P4122_LATTICE", "P4122_PARAMS",
    "p3121_atomic", "p3121_sites", "P3121", "P3121_OPS", "P3121_LATTICE", "P3121_PARAMS",
    "i41_atomic", "i41_sites", "i41_soc", "I41", "I41_SOC", "I41_OPS", "I41_PRIM_OPS",
    "I41_LATTICE", "I41_CENTERING", "I41_PARAMS", "I41_SOC_SEEDS", "primitive_ops",
    "p62_atomic", "p62_sites", "P62", "P62_OPS", "P62_LATTICE", "P62_PARAMS",
    "p41_flat_band", "p41_flat_bloch_hamiltonian", "P41_FLAT", "P41_FLAT_LATTICE",
    "P41_FLAT_PARAMS",
]


@dataclass(frozen=True)
class FixtureMeta:
    """Everything a test needs to know about a fixture besides the model itself.

    Attributes
    ----------
    name : str
        Short fixture key.
    provenance : str
        Literature source of the model *and* of the parameter values.
    params : dict
        The parameter values the builder uses by default.
    nbands : int
        Number of bands (``model.nstate``: orbitals x spin components).
    n_occupied : int
        Number of occupied bands at the filling the fixture is about.
    band_sets : dict
        Named band-index groups, e.g. ``{"P4": (0, 1, 2, 3), ...}``.
    group : str
        Hermann-Mauguin / wallpaper symbol of the group the fixture carries.
    sg_number : int or None
        3D space-group number (``None`` for the 2D fixtures).
    slab_sg_number : int or None
        Space group of the 2D fixture's slab cell built from orbital positions
        alone; ``None`` for the 3D fixture.
    regime : str or None
        Label of the parameter regime, when the fixture ships more than one.
    notes : str
        Anything an implementer would otherwise have to rediscover.
    """

    name: str
    provenance: str
    params: dict = field(default_factory=dict)
    nbands: int = 0
    n_occupied: int = 0
    band_sets: dict = field(default_factory=dict)
    group: str = ""
    sg_number: int | None = None
    slab_sg_number: int | None = None
    regime: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# 1.  lin3g -- the primary golden
# ---------------------------------------------------------------------------

LIN3G = FixtureMeta(
    name="lin3g",
    provenance=(
        "Lin et al., Nat. Commun. 15, 550 (2024), SM Sec. 3G, "
        "SEq. (3.55)+(3.59) [= arXiv:2207.10099 (C55)+(C60)]; parameters from "
        "the SFig. 7 caption.  Builder copied verbatim from fragile_v2() in "
        "nested_and_spin_resolved_Wilson_loop_v2/examples/models.py, itself "
        "lifted from that repo's validation/fragile_2D_spin_resolved.py; checked "
        "here against the pythtb-1.8 golden vendored at tests/data/lin3g_pythtb18.npz."
    ),
    params=dict(t1=5.0, t2=1.5, vm=-1.5, tPH=0.1, vs=1.3, vMz=0.4,
                vmu=8.25 * -1.5, vC=4.0, vCS=0.45 * 4.0),
    nbands=8,
    n_occupied=6,
    band_sets={
        "P4": (0, 1, 2, 3),   # the spinful s@2c EBR
        "P2": (4, 5),         # the fragile pair (Wieder quartet remnant at 1a)
        "P6": (0, 1, 2, 3, 4, 5),   # occupied
        "Q2": (6, 7),         # conduction
    },
    group="p4mm",
    sg_number=None,
    slab_sg_number=123,   # P4/mmm -- orbital positions only; see notes
    regime=None,
    notes=(
        "Spinful, 4 orbitals -> 8 bands.  Orbitals 0,1 sit at Wyckoff 1a (0,0) "
        "and carry the Wieder fragile quartet; orbitals 2,3 sit at 2c "
        "(1/2,0) and (0,1/2) and carry the s EBR.  The 4-band 1a EBR is split "
        "across valence and conduction: P4 and P2 are both occupied, Q2 is not. "
        "slab_sg_number=123 (P4/mmm) is the symmetry of the *orbital point set* "
        "in a vacuum-padded slab cell -- the spin-orbit terms in the Hamiltonian "
        "obey only a subgroup of it, which is exactly the 'cell suggests one "
        "group, Hamiltonian obeys another' situation phase A1 must detect."
    ),
)


# `lin3g` is `pythtqc.models.lin3g`, imported at the top of this module and
# re-exported here, so the suite and the worked example that loads this file
# build the same object.


# ---------------------------------------------------------------------------
# 1b.  the pythtb-1.8 golden, and the P± sectors the sewing tests read
# ---------------------------------------------------------------------------

#: pythtb-1.8 reference data for lin3g, vendored from
#: ``nested_and_spin_resolved_Wilson_loop_v2/validation/golden/fragile_2D.npz``
#: (written there by ``validation/gen_golden.py`` with pythtb 1.8 and the
#: original spin-resolved library, 2026-05-30).  It is the one oracle in this
#: suite that is a *different code*: ``k_test``/``e_test`` are energies at four
#: k-points; ``k_path``/``spin_bands`` the ``P s_z P`` spectrum of bands
#: ``occ = [4, 5]`` on a 300-point path; ``k1``/``gamma_plus``/``gamma_minus``
#: the P± Wilson phases of that pair along direction ``dir1 = 1`` on an
#: ``N1 x N2 = 81 x 41`` grid (the last ``k1`` sample is the periodic image of
#: the first).
PYTHTB18_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "lin3g_pythtb18.npz")


def lin3g_golden():
    """The vendored pythtb-1.8 golden for lin3g, as ``{key: array}``."""
    with np.load(PYTHTB18_GOLDEN) as data:
        return {key: np.array(data[key]) for key in data.files}


def spin_operator(model, n=(0.0, 0.0, 1.0)):
    """``n̂·s`` on a native spinful model, in pythtb's flattened basis (spin fast)."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                     dtype=complex)
    return np.kron(np.eye(int(model.norb)), np.tensordot(n, sigma, axes=1))


def psp_eigenstates(model, k_pts, bands, n=(0.0, 0.0, 1.0)):
    r"""The ``P Ô P`` spectrum and eigenstates over ``bands``, ``Ô = n̂·s``.

    ``P`` projects on the occupied Bloch states ``|m k>``, ``m in bands``; the
    reduced matrix ``<m k| Ô |n k>`` is diagonalized and its eigenvectors are
    rotated back into the orbital ⊗ spin basis (Lin et al., Nat. Commun. 15,
    550 (2024): the spin-resolved bands).  Returns
    ``(evals, states)`` with ``evals`` of shape ``(Nk, nocc)`` ascending and
    ``states`` of shape ``(Nk, nocc, nstate)`` as rows in that order -- the
    construction of ``spin_resolved_analysis.psp_eigenstates``, written here
    with pythtb and numpy alone so the suite does not import that library.
    """
    S = spin_operator(model, n)
    _, evecs = model.solve_ham(np.atleast_2d(np.asarray(k_pts, dtype=float)),
                               return_eigvecs=True)
    w = np.asarray(evecs)[:, np.asarray(bands, dtype=int), :]    # (Nk, nocc, nstate)
    reduced = np.conj(w) @ S @ np.swapaxes(w, -1, -2)
    evals, V = np.linalg.eigh(reduced)
    return evals, np.einsum("kmj,kmt->kjt", V, w)


def psp_sector_wfarrays(model, mesh, bands, n=(0.0, 0.0, 1.0), *, gap_tol=1e-8):
    r"""``(P_+, P_-)`` over ``mesh`` as pythtb ``WFArray``\ s: the sector states.

    ``P_+`` carries the upper half of the ``P Ô P`` spectrum and ``P_-`` the
    lower half, the feature gap at the split asserted open (``gap_tol``); each
    fiber holds ``len(bands) // 2`` states and no energies, which is what a
    sewing matrix on a sector has to cope with.
    """
    from pythtb import WFArray

    bands = [int(i) for i in bands]
    half, odd = divmod(len(bands), 2)
    assert not odd, "an equal P+/P- split needs an even number of bands"
    k = np.asarray(mesh.get_k_points())                          # (*shape_k, dim_k)
    evals, states = psp_eigenstates(model, k.reshape(-1, k.shape[-1]), bands, n)
    gap = evals[:, half] - evals[:, half - 1]
    assert gap.min() > gap_tol, f"feature gap closed at the split: {gap.min():.2e}"
    states = states.reshape(*k.shape[:-1], len(bands), states.shape[-1])
    sectors = []
    for sub in (states[..., half:, :], states[..., :half, :]):
        wfa = WFArray(model.lattice, mesh, nstates=half, spinful=True)
        wfa.set_states(sub, is_cell_periodic=True, is_spin_axis_flat=True)
        # the provenance ``pythtqc.sewing`` reads off a WFArray (a solved one
        # carries it); ``set_states`` alone leaves the array model-less
        wfa._model = model
        sectors.append(wfa)
    return tuple(sectors)


# ---------------------------------------------------------------------------
# 2.  Kane-Mele -- the hexagonal spinful golden, both Z2 regimes
# ---------------------------------------------------------------------------

#: ``pythtb.models.kane_mele(delta, t, soc, rashba)`` arguments per regime.
#:
#: These are the parameter values of the canonical pythtb Kane-Mele example
#: (``esite`` 1.0 / 2.5, ``spin_orb`` 0.3, ``rashba`` 0.25), with ``t = -1.0``
#: to match the sign already used in this repo's ``tests/conftest.py``.  On the
#: bipartite honeycomb ``t -> -t`` is the sublattice gauge transformation
#: ``diag(1, -1)``, so it changes neither the spectrum nor Z2.
#:
#: The regime is set by the competition between the staggered sublattice
#: potential ``delta`` and the intrinsic spin-orbit coupling: the gap at K
#: closes at ``delta = 3*sqrt(3)*soc = 1.559`` for ``soc = 0.3``, so
#: ``delta = 1.0`` is below it (Z2 = 1) and ``delta = 2.5`` above it (Z2 = 0).
_KM_ARGS = {
    "odd": dict(delta=1.0, t=-1.0, soc=0.3, rashba=0.25),
    "even": dict(delta=2.5, t=-1.0, soc=0.3, rashba=0.25),
}

_KM_COMMON = dict(
    provenance=(
        "Kane & Mele, Phys. Rev. Lett. 95, 226801 (2005); built by "
        "pythtb.models.kane_mele(delta, t, soc, rashba).  Parameter values are "
        "those of the canonical pythtb kane_mele example (esite 1.0 / 2.5, "
        "spin_orb 0.3, rashba 0.25)."
    ),
    nbands=4,
    n_occupied=2,
    band_sets={"P2": (0, 1), "Q2": (2, 3)},
    # nominal: the honeycomb cell with delta != 0 makes the two sublattices
    # inequivalent, so the wallpaper content is p3m1.  Phase A1's `groups`
    # adapter is what determines this from the model rather than by hand.
    group="p3m1",
    sg_number=None,
    slab_sg_number=187,   # P-6m2; the two sublattices are inequivalent (delta != 0)
)

KANE_MELE = {
    "odd": FixtureMeta(
        name="kane_mele_odd",
        params=dict(_KM_ARGS["odd"]),
        regime="Z2-nontrivial (Z2 = 1, quantum spin Hall)",
        notes=(
            "delta = 1.0 < 3*sqrt(3)*soc = 1.559: the K-point gap has not been "
            "inverted by the sublattice potential.  Measured half-filling gap "
            "0.877398 (direct = indirect); Z2 = 1 from the k1-Wilson-loop "
            "hybrid-Wannier-centre flow."
        ),
        **_KM_COMMON,
    ),
    "even": FixtureMeta(
        name="kane_mele_even",
        params=dict(_KM_ARGS["even"]),
        regime="Z2-trivial (Z2 = 0, band insulator)",
        notes=(
            "delta = 2.5 > 3*sqrt(3)*soc = 1.559: the sublattice potential has "
            "won and the K-point gap is inverted back.  Measured half-filling "
            "gap 2.008880 (direct = indirect); Z2 = 0."
        ),
        **_KM_COMMON,
    ),
}


def kane_mele_z2(regime: str = "odd"):
    """Build the Kane-Mele fixture in the requested Z2 regime.

    Parameters
    ----------
    regime : {"odd", "even"}
        ``"odd"``  -> Z2-nontrivial (QSH);
        ``"even"`` -> Z2-trivial.

    Returns
    -------
    pythtb.TBModel
        Spinful 2D honeycomb model, 2 orbitals / 4 bands.
    """
    from pythtb.models import kane_mele as _kane_mele

    if regime not in _KM_ARGS:
        raise ValueError(f"regime must be one of {sorted(_KM_ARGS)}, got {regime!r}")
    a = _KM_ARGS[regime]
    return _kane_mele(a["delta"], a["t"], a["soc"], a["rashba"])


# ---------------------------------------------------------------------------
# 3.  P4_232 (SG 208) -- the nonsymmorphic, 3D, spinless golden
# ---------------------------------------------------------------------------

# The builders and their constants -- ``P4232_PARAMS``, ``P4232_NN_VECTORS``,
# :func:`p4232_bloch_hamiltonian`, :func:`p4232_cano2018`,
# :func:`p4232_operations` and :func:`p4232_symmetry_cell` -- were promoted to
# the public :mod:`pythtqc.models` on 2026-08-25 and are imported
# (and re-exported) at the top of this module.  What stays here is
# the test-facing :class:`FixtureMeta` record.

P4232 = FixtureMeta(
    name="p4232_cano2018",
    provenance=(
        "Cano, Bradlyn, Wang, Elcoro, Vergniory, Felser, Aroyo & Bernevig, "
        "Phys. Rev. Lett. 120, 266401 (2018) = arXiv:1711.11045, section "
        "'Non-symmorphic gapped EBR': Eqs. (5), (6), (7); parameters from the "
        "Fig. 3a caption."
    ),
    params=dict(P4232_PARAMS),
    nbands=4,
    n_occupied=2,
    band_sets={"P2": (0, 1), "Q2": (2, 3), "EBR4": (0, 1, 2, 3)},
    group="P4_232",
    sg_number=208,
    slab_sg_number=None,
    regime="gapped disconnected EBR (Fig. 3a)",
    notes=(
        "Spinless, 2 sites x 2 orbitals -> 4 bands; the whole 4-band set is one "
        "time-reversal-symmetric EBR induced from the 2D (dz2, dx2-y2) irrep of "
        "the site-symmetry group of Wyckoff 2a.  It is gapped at half filling, "
        "which is the paper's counterexample to the Michel-Zak connectivity "
        "conjecture; the valence pair (0,1) is therefore topological.  Note "
        "the group is 208 (P4_232), not 212 (P4_332) -- "
        "see the module docstring.  Setting t4 = t5 = t6 = 0 leaves the paper's "
        "H_k alone, which is doubly degenerate at *every* k and still gapped; "
        "H_k^1 lifts that degeneracy away from the symmetry-protected lines. "
        "The bare 2a orbital point set alone reads as SG 229 (Im-3m) -- the "
        "orbitals form a bcc lattice -- so the crystal field of the paper's "
        "extra (non-orbital) atoms is what makes the cell P4_232; "
        "p4232_symmetry_cell() supplies such a decoration."
    ),
)


# ---------------------------------------------------------------------------
# 4.  Fu-Kane-Mele -- the 3D spinful diamond golden (phase A4)
# ---------------------------------------------------------------------------

#: Distortion patterns for :func:`fu_kane_mele_z2`, as ``pythtb.models``'
#: ``dt`` argument (offsets added to the four nearest-neighbour bonds, in the
#: order ``R = [0,0,0], [-1,0,0], [0,-1,0], [0,0,-1]``).  ``t = 1``,
#: ``soc = 0.125``.
_FKM_ARGS = {
    "uniform": dict(t=1.0, soc=0.125, dt=(0.0, 0.0, 0.0, 0.0)),
    "strong": dict(t=1.0, soc=0.125, dt=(0.4, 0.0, 0.0, 0.0)),
    "weak": dict(t=1.0, soc=0.125, dt=(-0.4, 0.0, 0.0, 0.0)),
    "trivial": dict(t=1.0, soc=0.125, dt=(0.0, -1.0, -1.0, -1.0)),
}

_FKM_COMMON = dict(
    provenance=(
        "Fu, Kane & Mele, Phys. Rev. Lett. 98, 106803 (2007); built by "
        "pythtb.models.fu_kane_mele(t, soc, dt).  The single-bond distortion "
        "delta_t is the paper's own symmetry-lowering perturbation of the "
        "diamond lattice (its Fig. 2 phase diagram); soc = 0.125 t is a "
        "conventional value, not one the paper fixes.  Every Z2 index quoted "
        "below was MEASURED here by hybrid-Wannier-centre partner switching on "
        "the four inequivalent TRIM planes, not taken from the paper."
    ),
    nbands=4,
    n_occupied=2,
    band_sets={"P2": (0, 1), "Q2": (2, 3), "P4": (0, 1, 2, 3)},
    group="Fd-3m",
    sg_number=227,
    slab_sg_number=None,
)

FU_KANE_MELE = {
    "uniform": FixtureMeta(
        name="fu_kane_mele_uniform",
        params=dict(_FKM_ARGS["uniform"]),
        regime="undistorted: a semimetal, and the only regime that realizes Fd-3m",
        notes=(
            "delta_t = 0.  This is the regime the *content* measurement "
            "uses, because it is the only one whose HAMILTONIAN obeys the full "
            "group the cell allows: _validate_group gives 96 of 96, and _build_rep "
            "closes at covariance 2.3e-15 with product 2.1e-15.  It is gapless: "
            "the four bands are degenerate at X, which is the whole reason the "
            "FKM paper distorts the lattice.  So a two-band block refuses at X "
            "with gate='multiplet_split', while the four-band block resolves at "
            "all four points -- GM, L, W and X -- and is exactly the "
            "irreptables row -E2^G(4)@8a.  (W refused with defect 0.5 until "
            "2026-08-21, when the adapter started conjugating irreptables' "
            "factor system at the boundary; the tqc_fkm_diamond example's "
            "golden prints the four-band content, W included.)"
        ),
        **_FKM_COMMON,
    ),
    "strong": FixtureMeta(
        name="fu_kane_mele_strong",
        params=dict(_FKM_ARGS["strong"]),
        regime="strong topological insulator (1;111)",
        notes=(
            "delta_t = +0.4 on the R = [0,0,0] bond.  Measured: gap 0.800000 at "
            "half filling; Z2 on the (k1=0, k1=1/2, k2=1/2, k3=1/2) planes = "
            "(0, 1, 1, 1), i.e. (nu0; nu1 nu2 nu3) = (1;111).  The distortion "
            "lowers the symmetry: the cell still reads Fd-3m1' (#227.129, 96 "
            "operations) but _validate_group finds only 24 of them survive."
        ),
        **_FKM_COMMON,
    ),
    "weak": FixtureMeta(
        name="fu_kane_mele_weak",
        params=dict(_FKM_ARGS["weak"]),
        regime="weak topological insulator (0;111)",
        notes=(
            "delta_t = -0.4, the other branch of the same distortion.  Measured: "
            "gap 0.800000; Z2 planes (1, 1, 1, 1) -> (0;111).  **This is not a "
            "trivial insulator**, and saying so would be wrong: nu0 = 0 but the "
            "weak indices are all 1.  A single-bond distortion of FKM has no "
            "(0;000) phase at all, which is why the 'trivial' regime below is "
            "the dimerized limit instead.  24 of 96 operations survive."
        ),
        **_FKM_COMMON,
    ),
    "trivial": FixtureMeta(
        name="fu_kane_mele_trivial",
        params=dict(_FKM_ARGS["trivial"]),
        regime="trivial insulator (0;000) -- the decoupled-dimer atomic limit",
        notes=(
            "dt = (0, -1, -1, -1) with t = 1 switches three of the four "
            "nearest-neighbour bonds off, leaving isolated dimers along the "
            "R = [0,0,0] bond: an atomic limit, hence trivial by construction "
            "and by adiabatic continuity.  Measured: gap 2.000000 (= 2|t|, the "
            "dimer bonding/antibonding splitting) and Z2 planes (0, 0, 0, 0) -> "
            "(0;000).  24 of 96 operations survive."
        ),
        **_FKM_COMMON,
    ),
}


def fu_kane_mele_z2(regime: str = "strong"):
    """Build the Fu-Kane-Mele fixture in the requested regime.

    Parameters
    ----------
    regime : {"uniform", "strong", "weak", "trivial"}
        See :data:`FU_KANE_MELE` for what each one is and what was measured.

    Returns
    -------
    pythtb.TBModel
        Spinful 3D diamond model, 2 orbitals / 4 bands.
    """
    from pythtb.models import fu_kane_mele as _fkm

    if regime not in _FKM_ARGS:
        raise ValueError(f"regime must be one of {sorted(_FKM_ARGS)}, got {regime!r}")
    a = _FKM_ARGS[regime]
    return _fkm(a["t"], a["soc"], list(a["dt"]))


# ---------------------------------------------------------------------------
# 5.  Inversion-symmetric Kane-Mele -- the #191 cell, the #183 Hamiltonian
# ---------------------------------------------------------------------------

KANE_MELE_INVERSION = FixtureMeta(
    name="kane_mele_inversion",
    provenance=(
        "Kane & Mele, Phys. Rev. Lett. 95, 226801 (2005), the original model "
        "with no staggered sublattice potential and no Rashba coupling; built "
        "by pythtb.models.kane_mele(0.0, t, soc, 0.0) with t = -1, soc = 0.3."
    ),
    params=dict(delta=0.0, t=-1.0, soc=0.3, rashba=0.0),
    nbands=4,
    n_occupied=2,
    band_sets={"P2": (0, 1), "Q2": (2, 3), "P4": (0, 1, 2, 3)},
    group="p6mm",
    sg_number=None,
    slab_sg_number=191,   # P6/mmm -- both sublattices are the same species
    regime="quantum spin Hall for every soc != 0 (there is no trivial regime here)",
    notes=(
        "delta = 0 makes the two sublattices equivalent, so the slab cell reads "
        "P6/mmm1' (#191.234, 48 operations) rather than the #187 of the "
        "delta != 0 fixture.  rashba = 0 keeps sigma_h unbroken -- but a 2D "
        "pythtb model still cannot represent it (no out-of-plane orbital "
        "structure), so the group whose labels apply is the decorated-cell one, "
        "**P6mm1' (#183.186)**, all 24 of whose operations _validate_group finds on "
        "the Hamiltonian (23 of 24 until 2026-08-24: rashba = 0 conserves s_z, so "
        "every operation here poses a two-dimensional intertwiner problem, and "
        "m(x)' was the one whose SVD basis landed on an exactly singular element "
        "-- refused at residual 3.4547 until _find_intertwiners searched the "
        "solution space for a unitary member instead).  "
        "There is no inversion in a 2D wallpaper group: what the DECORATED cell "
        "keeps is C2z (R[2,2] = +1), a ROTATION whose action on in-plane k is "
        "inversion's -- which is what makes the two easy to confuse, and it is "
        "NOT what this fixture is named for.  Genuine inversion (r -> -r, "
        "squaring to E and never Ebar on spinors) is a LAYER-GROUP operation "
        "here, and that is the name's warrant: at rashba = 0 sigma_h is real, "
        "P = sigma_h * C2z is in #191.234, "
        "and it is reachable by environment=\"free-standing\" with "
        "_build_rep(include_vacuum_ops=True) -- measured in "
        "examples/tqc_kane_mele_three_presentations.py, which represents all 48 "
        "operations at rashba = 0 (product residual 1.2e-15) and gets "
        "rep_not_covariant at rashba = 0.25.  P's lift squares to +E and C2z's "
        "to Ebar, so parity eigenvalues exist for one and not the other. "
        "Measured: half-filling gap 2.001348 on a 121x121 grid, direct = indirect "
        "(the K-point gap is 6*sqrt(3)*soc = 3.117691, so the minimum sits away "
        "from K).  There is no "
        "Z2-trivial regime at delta = 0: the gap at K never closes for soc != 0, "
        "so this fixture is QSH throughout -- which is exactly why it complements "
        "the #156 pair.  This is also the fixture that forced _build_rep's joint "
        "uniqueness gate: both orientation branches are covariant (1.3e-15 / "
        "1.8e-15) AND both reproduce the multiplication table (9.4e-16 / "
        "1.2e-15), so neither of those pins the natural one.  What does is the "
        "internal factor: the g^-1 branch carries S(C3)^-1 where the group says "
        "S(C3), site-block defect 0.866 against 5.5e-16.  The build needs no "
        "orientation_branch=+1 any more."
    ),
)


def kane_mele_inversion():
    """Build the inversion-symmetric (delta = 0, rashba = 0) Kane-Mele fixture.

    Returns
    -------
    pythtb.TBModel
        Spinful 2D honeycomb model, 2 orbitals / 4 bands.
    """
    from pythtb.models import kane_mele as _kane_mele

    p = KANE_MELE_INVERSION.params
    return _kane_mele(p["delta"], p["t"], p["soc"], p["rashba"])


# ---------------------------------------------------------------------------
# 6.  The hexagonal fragile golden -- Po/Watanabe/Vishwanath
# ---------------------------------------------------------------------------
#
# PROVENANCE WARNING, read before citing this fixture.
#
# The attribution "the hexagonal symmetry-indicated fragile model of
# arXiv:1807.09729 (Bouhon-Black-Schaffer-Slager)" pairs the wrong identifier
# with the wrong authors:
#
# * **arXiv:1807.09729** is Bradlyn, Wang, Cano & Bernevig, *Disconnected
#   Elementary Band Representations, Fragile Topology, and Wilson Loops as
#   Topological Indices: An Example on the Triangular Lattice*, Phys. Rev. B
#   **99**, 045140 (2019).
# * Bouhon, Black-Schaffer & Slager, *Wilson loop approach to fragile topology
#   …*, is **arXiv:1804.09719**, Phys. Rev. B **100**, 195135 (2019).
#
# Both of them use the *same* model, and neither of them is its source: it is
# the fragile model of **Po, Watanabe & Vishwanath, arXiv:1709.06551**, Phys.
# Rev. Lett. **121**, 126402 (2018).  Bouhon et al. state the reuse in a
# footnote ("When restricted to the seventh layer of neighbors, our
# tight-binding model matches with the model given in Ref. [Po et al.]") and
# print no numerical parameters at all; Bradlyn et al. print the parameter
# table and the construction.  So the fixture is built from Po et al.'s
# definition, with Bradlyn et al.'s Table I and Appendix A as the second
# source, and it is named for Po/Watanabe/Vishwanath.
#
# Three defects in the papers as printed, each measured here rather than assumed:
#
# 1. Bradlyn et al. Eqs. (9)-(10) are **not Hermitian as written**; Po et al.'s
#    Eq. (2), which they transcribe, carries an explicit ``+ h.c.``.  It is
#    required, and the builder adds it (pythtb's ``set_hop`` does).
# 2. Bradlyn et al. Eq. (3) writes the ancillary on-site energy as
#    ``36(1-2 mu)``; Po et al. have ``3(1-2 mu) t``.  **3 is right**: it puts the
#    flat ancillary bands at exactly +3 at mu = 0, which is what Bradlyn et al.'s
#    own Fig. 2(a) shows, and it produces the gap closing at mu ~ 0.6 that their
#    text describes (with 36 the gap never closes).
# 3. Bradlyn et al. give ``lambda^z_5 = 0.12`` where Po et al. give 0.  It makes
#    no difference: that component symmetrises to exactly zero on bond 5's orbit
#    (measured: largest matrix element 0.0), so both tables define one model.
#
# The bond *geometry* is printed in neither paper -- it exists only in Fig. 10 of
# arXiv:1807.09729 / Fig. 1(c) of arXiv:1709.06551.  It is therefore not taken on
# trust: :func:`pwv_bloch_hamiltonian` transcribes Po et al.'s **closed-form**
# Bloch Hamiltonian (their Appendix A, which needs no figure) and
# ``test_fixtures_smoke.py`` checks the built model against it at 5.6e-16.

#: The bond parameters ``(t, lambda_x, lambda_y, lambda_z)``: Table I
#: (``p6mm1'`` column) of arXiv:1807.09729 = the Appendix A table of
#: arXiv:1709.06551.  Bonds 1-5 build ``H_0``, bonds 6-7 build ``H_c``.
PWV_BOND_PARAMS = {
    1: (-0.7, -0.4, -0.2, 0.0),
    2: (0.0, -0.6, 0.0, -1.0),
    3: (-0.3, -0.7, -0.4, 0.0),
    4: (0.0, 0.9, 0.3, 0.0),
    5: (-0.2, 0.4, 0.3, 0.12),
    6: (-3.0, -2.1, 1.2, 0.0),
    7: (0.5, 1.2, -0.7, 0.0),
}

#: Bond geometry, ``(origin site, endpoint site, endpoint cell offset)``, read
#: off Fig. 10 of arXiv:1807.09729 and **verified** by the closed-form check.
#: Sites: ``"A"`` = ``r1`` = ``(1/3, 2/3)``, ``"B"`` = ``r2`` = ``(2/3, 1/3)``
#: (the honeycomb, Wyckoff 2b), ``"C"`` = ``(0, 0)`` (the hexagon centre, 1a).
#: The shells are 1st-NN A-B, 1st-NN A-A, 3rd-NN A-B (across the hexagon),
#: 4th-NN A-B, 2nd-NN A-A, then the two couplings to the hexagon centre.
PWV_BOND_GEOMETRY = {
    1: ("A", "B", (0, 0)),
    2: ("A", "A", (0, -1)),
    3: ("B", "A", (1, -1)),
    4: ("A", "B", (0, -1)),
    5: ("A", "A", (1, -1)),
    6: ("A", "C", (0, 0)),
    7: ("B", "C", (0, -1)),
}

#: ``a1``, ``a2`` of Po et al. App. A (= Eqs. (25)-(26) of arXiv:1807.09729).
PWV_LATTICE = np.array([[np.sqrt(3) / 2, -0.5], [0.0, 1.0]])

#: Reduced positions of the three sites.
PWV_SITES = {"A": (1 / 3, 2 / 3), "B": (2 / 3, 1 / 3), "C": (0.0, 0.0)}

_PWV_SIGMA = (np.eye(2, dtype=complex),
              np.array([[0, 1], [1, 0]], dtype=complex),
              np.array([[0, -1j], [1j, 0]], dtype=complex),
              np.array([[1, 0], [0, -1]], dtype=complex))


def pwv_operations():
    """The 12 operations of ``6mm`` about the hexagon centre.

    Returns ``[(R_cart, U_spin, name, fixes_A), ...]``: the 2x2 Cartesian
    matrix, the spin-1/2 matrix (``exp(-i theta sigma_z / 2)`` for a rotation,
    ``-i n.sigma`` for a mirror with in-plane normal ``n``), and whether the
    operation fixes sublattice ``A``.  The six that do are the paper's
    ``G_0 = <m_y, C_3>``; the other six are the coset ``m_x G_0``, which
    exchanges the sublattices and is the one weighted by ``x_i``.
    """
    lat_inv = np.linalg.inv(PWV_LATTICE)
    a_cart = np.asarray(PWV_SITES["A"]) @ PWV_LATTICE
    out = []
    for n in range(6):
        th = n * np.pi / 3
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        U = np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)])
        out.append((R, U, f"C6^{n}"))
    for n in range(6):
        phi = n * np.pi / 6
        nh = np.array([np.cos(phi), np.sin(phi)])
        R = np.eye(2) - 2 * np.outer(nh, nh)
        U = -1j * (nh[0] * _PWV_SIGMA[1] + nh[1] * _PWV_SIGMA[2])
        out.append((R, U, f"m({n * 30}deg)"))
    tagged = []
    for R, U, name in out:
        d = (R @ a_cart) @ lat_inv - np.asarray(PWV_SITES["A"])
        tagged.append((R, U, name, bool(np.abs(d - np.round(d)).max() < 1e-8)))
    return tagged


def _pwv_site_and_cell(pos_cart, tol=1e-8):
    red = pos_cart @ np.linalg.inv(PWV_LATTICE)
    for name, r0 in PWV_SITES.items():
        d = red - np.asarray(r0)
        if np.abs(d - np.round(d)).max() < tol:
            return name, tuple(int(x) for x in np.round(d))
    raise ValueError(f"position {pos_cart} lies on no site of the model")


def _pwv_accumulate(bonds, x=None):
    """The symmetrised sum of Eqs. (9)-(10), before the 1/12 and the ``+ h.c.``.

    ``{(site_from, site_to, cell_offset): 2x2 spin block}``.  The paper writes
    ``h_i = (t + i lambda.sigma) c^dag_f c_o``, i.e. the matrix element
    ``<f|h|o>``; the block is stored here as ``(row = o, col = f)`` -- the
    conjugate placement, hence the dagger.
    """
    acc = {}
    ops = pwv_operations()
    order = {"A": 0, "B": 1, "C": 2}
    for i in bonds:
        t, lx, ly, lz = PWV_BOND_PARAMS[i]
        weight_off = 1.0 if x is None else float(x[i])
        o_name, f_name, cell = PWV_BOND_GEOMETRY[i]
        o = np.asarray(PWV_SITES[o_name]) @ PWV_LATTICE
        f = (np.asarray(PWV_SITES[f_name]) + np.asarray(cell, dtype=float)) @ PWV_LATTICE
        M = (t * _PWV_SIGMA[0]
             + 1j * (lx * _PWV_SIGMA[1] + ly * _PWV_SIGMA[2] + lz * _PWV_SIGMA[3]))
        for R, U, _name, fixes_a in ops:
            w = 1.0 if fixes_a else weight_off
            if w == 0.0:
                continue
            so, co = _pwv_site_and_cell(R @ o)
            sf, cf = _pwv_site_and_cell(R @ f)
            key = (order[so], order[sf], tuple(np.asarray(cf) - np.asarray(co)))
            acc[key] = acc.get(key, 0) + w * (U @ M @ U.conj().T).conj().T
    return acc


def pwv_bloch_hamiltonian(k, mu=0.0, six_band=False):
    r"""The **closed-form** Bloch Hamiltonian of Po *et al.*, Appendix A.

    The independent oracle for :func:`pwv_fragile`: it is printed in full in
    arXiv:1709.06551 and needs none of the bond geometry that lives only in a
    figure.  ``H_0`` is expanded as ``(t/12) sum_ij h_ij sigma_i (x) sigma_j``
    with the first Pauli set the sublattice and the second the spin, and the
    coupling block ``H_c`` as the explicit ``C_13``/``C_23`` sub-matrices.

    Parameters
    ----------
    k : array, shape (2,)
        Reduced coordinates.  Po *et al.*'s ``(g_1, g_2)`` are ``2 pi k``.
    mu : float, optional
        The interpolation parameter of arXiv:1807.09729 Eq. (3): the coupling
        carries ``sin(pi mu)`` and the ancillary on-site energy is
        ``3 (1 - 2 mu)`` (**3**, not the 36 that equation misprints).
    six_band : bool, optional
        Include the two ancillary states at the hexagon centre.

    Returns
    -------
    ndarray
        ``(4, 4)`` or ``(6, 6)``, in the **lattice gauge** (basis order
        ``r1, r2, r3 = 0``, spin fastest).
    """
    g1, g2 = 2 * np.pi * np.asarray(k, dtype=float).reshape(-1)[:2]
    s, c, r3 = np.sin, np.cos, np.sqrt(3.0)
    h = {
        (0, 0): -4 / 5 * (c(g1 - g2) + c(2 * g1 + g2) + c(g1 + 2 * g2)),
        (0, 1): 3 / 5 * (2 * (s(g1 + g2) - s(g1)) - (4 + r3) * s(3 * g2 / 2) * c(g1 + g2 / 2)
                         + 4 * s(g2)),
        (0, 2): 1 / 5 * s(g1 + g2 / 2) * ((6 + 8 * r3) * (c(g1 + g2 / 2) + c(g2 / 2) * c(g2))
                                          - (3 + 16 * r3) * c(g2 / 2)),
        (1, 0): 2 / 5 * (3 * s(g1) * s(g2) - 9 * c(g1) * c(g2) - 7 * (c(g1) + c(g2) + 1)),
        (1, 1): ((3 - r3) / 10 * (3 * s(2 * g1) - 2 * s(g2) - 3 * s(2 * g1 + g2))
                 + (21 + 4 * r3) / 5 * c(g1) * s(g2)
                 - 3 * (3 + r3) / 5 * s(g2) * c(g1 + g2) - 9 / 5 * s(2 * g2)),
        (1, 2): (3 * (1 + 3 * r3) / 10 * (s(2 * g1 + g2) + s(2 * g1))
                 + 3 * (3 * r3 - 1) / 10 * s(g1 + 2 * g2)
                 - (4 + 7 * r3) / 5 * (2 * s(g1) * c(g2) + c(g1) * s(g2))
                 + (r3 - 7) / 10 * s(g1) - 2 * (2 + r3) / 5 * s(g2) + 3 / 5 * s(2 * g2)),
        (2, 0): 2 / 5 * (7 * (s(g2) - s(g1)) - 3 * s(g1 - g2)),
        (2, 1): (1 / 10 * s(g2) * (6 * (3 + r3) * s(g1 + g2) + (42 + 8 * r3) * s(g1)
                                   + 3 * (r3 - 3) * s(2 * g1))
                 + 1 / 5 * s(g2 / 2) ** 2 * ((r3 - 3) * (3 * c(2 * g1) - 2) + 36 * c(g2))),
        (2, 2): (-1 / 5 * c(g2 / 2) ** 2 * ((3 + 9 * r3) * c(2 * g1) + 4 * (r3 + 2 - 3 * c(g2)))
                 + 1 / 5 * s(g1) * s(g2) * ((3 - 9 * r3) * c(g2) + 7 * r3 + 4)
                 + 1 / 10 * c(g1) * (6 * (1 + 3 * r3) * s(g1) * s(g2)
                                     + (9 * r3 - 3) * c(2 * g2) + 17 * r3 + 1)),
        (3, 3): 4 * (s(g1) + s(g2) - s(g1 + g2)),
    }
    h0 = np.zeros((4, 4), dtype=complex)
    for (i, j), v in h.items():
        h0 += v / 12 * np.kron(_PWV_SIGMA[i], _PWV_SIGMA[j])
    if not six_band:
        return h0

    e = np.exp
    s0, s1, s2 = _PWV_SIGMA[0], _PWV_SIGMA[1], _PWV_SIGMA[2]
    c13 = ((2 * c(g1) - 6 * (e(1j * (g1 + g2)) + e(1j * g2) + 1) + e(1j * (g1 + 2 * g2))) * s0
           + 1 / 20 * (3 * (21 + 4 * r3) * (s(g2) - 1j * c(g2) + 1j)
                       - 2 * (36 + 7 * r3) * e(1j * (g1 + g2)) * s(g2)) * s1
           - 1 / 20 * 1j * e(-1j * g1) * ((7 + 12 * r3) * (e(2j * (g1 + g2)) + e(2j * g1) - 2)
                                          + 3 * (4 + 7 * r3) * e(1j * g1)
                                          * (1 + (1 - 2 * e(1j * g1)) * e(1j * g2))) * s2)
    c23 = ((2 * c(g2) - 6 * (e(1j * (g1 + g2)) + e(1j * g1) + 1) + e(2j * g1 + 1j * g2)) * s0
           + 1 / 20 * 1j * e(-1j * g2) * ((36 + 7 * r3) * (e(2j * g2) - 1)
                                          + 3 * (21 + 4 * r3) * (1 - e(1j * g2))
                                          * e(1j * (g1 + g2))) * s1
           + 1 / 20 * 1j * e(-1j * g2) * (3 * (4 + 7 * r3) * e(1j * g2)
                                          * (e(1j * (g1 + g2)) + e(1j * g1) - 2)
                                          + (7 + 12 * r3)
                                          * (1 + (1 - 2 * e(2j * g1)) * e(2j * g2))) * s2)
    out = np.zeros((6, 6), dtype=complex)
    out[:4, :4] = h0
    coupling = np.sin(np.pi * float(mu)) / 12
    out[0:2, 4:6] = coupling * c13
    out[2:4, 4:6] = coupling * c23
    out[4:6, 0:2] = (coupling * c13).conj().T
    out[4:6, 2:4] = (coupling * c23).conj().T
    out[4, 4] = out[5, 5] = 3.0 * (1 - 2 * float(mu))
    return out


PWV = FixtureMeta(
    name="pwv_fragile",
    provenance=(
        "Po, Watanabe & Vishwanath, Phys. Rev. Lett. 121, 126402 (2018) = "
        "arXiv:1709.06551, Eqs. (1)-(2) and the Appendix A parameter table and "
        "closed-form Bloch Hamiltonian; the same model, with the same parameter "
        "table (Table I, p6mm1' column) and the interpolation of its Eq. (3), is "
        "the running example of Bradlyn, Wang, Cano & Bernevig, Phys. Rev. B 99, "
        "045140 (2019) = arXiv:1807.09729, and of Bouhon, Black-Schaffer & "
        "Slager, arXiv:1804.09719.  Bond geometry from Fig. 10 of "
        "arXiv:1807.09729, VERIFIED against the closed form at 5.6e-16."
    ),
    params=dict(mu=0.0, bonds=dict(PWV_BOND_PARAMS)),
    nbands=4,
    n_occupied=2,
    band_sets={"P2": (0, 1), "Q2": (2, 3), "P4": (0, 1, 2, 3)},
    group="p6mm1'",
    sg_number=None,
    slab_sg_number=191,
    regime="four-band H_0: valence trivial (obstructed at 1a), conduction FRAGILE",
    notes=(
        "Spinful s (equivalently p_z) orbitals on both honeycomb sites -- Wyckoff "
        "2b of p6mm1' -- with hoppings out to the fifth shell.  **The fragile "
        "branch is the CONDUCTION pair, not the valence pair.**  Po et al.: 'the "
        "valence bands of our model are trivial, whereas the conduction bands "
        "feature fragile topology'.  Measured here: valence content "
        "{-GM9, -K6, -M5} decomposes uniquely as -E1^G(2)@1a -- the paper's "
        "(E1bar ^ G)_1a, an obstructed atomic limit at the EMPTY hexagon centre "
        "-- while the conduction content {-GM8, -K4, -K5, -M5} admits NO "
        "nonnegative-integer EBR decomposition (CP-SAT infeasible) and is a "
        "signed difference [-E1^G(4)@2b] - [-E1^G(2)@1a], which is what "
        "'symmetry-indicated fragile' means.  Measured spectra reproduce the "
        "papers: E(GM) = (-1.2, -1.2, 0.8, 0.8), E(K) = (-1.066, -1.066, 0.6062, "
        "0.7258), E(M) = (-0.6536 x2, 0.787 x2); indirect gap 0.390557 on a 121x121 "
        "grid (Po et al.: '0.39t'), direct gap 0.560386, valence bandwidth "
        "2.209161 ('~2t').  The "
        "slab cell reads P6/mmm1' (#191.234); the group whose labels apply is the "
        "decorated-cell P6mm1' (#183.186), all 24 operations of which the "
        "Hamiltonian obeys."
    ),
)

PWV6 = FixtureMeta(
    name="pwv_fragile_six_band",
    provenance=PWV.provenance + "  Interpolation: arXiv:1807.09729 Eq. (3) at mu = 0.3.",
    params=dict(mu=0.3, bonds=dict(PWV_BOND_PARAMS)),
    nbands=6,
    n_occupied=2,
    band_sets={"P2": (0, 1), "MID": (2, 3), "TOP": (4, 5), "P4": (0, 1, 2, 3)},
    group="p6mm1'",
    sg_number=None,
    slab_sg_number=191,
    regime="six-band interpolation at mu = 0.3 (the 'band teleportation' setup)",
    notes=(
        "Adds two ancillary states at the hexagon centre (Wyckoff 1a) coupled "
        "with strength sin(pi mu) and sitting at on-site energy 3(1-2mu).  At "
        "mu = 0 the coupling vanishes and the ancillary bands are flat at exactly "
        "+3, which is the check that the '36' printed in Eq. (3) is a misprint "
        "for '3'.  At mu = 0.3 the fragile pair is the MIDDLE two bands (Bradlyn "
        "et al. Fig. 4); measured gaps 1.1062 below it and 0.8084 above it."
    ),
)


def pwv_fragile(mu: float = 0.0, six_band: bool = False, x=None):
    """Build the Po-Watanabe-Vishwanath fragile honeycomb model.

    The construction is the papers' own: for each of the seven bonds, symmetrise
    ``h_i = (t_i + i lambda_i . sigma) c^dag_f c_o`` over the twelve operations
    of ``6mm``, divide by 12, and add the Hermitian conjugate (which pythtb's
    ``set_hop`` does).  Bonds 1-5 give ``H_0``; bonds 6-7 give the coupling
    ``H_c`` to the hexagon centre.

    Parameters
    ----------
    mu : float, optional
        Interpolation parameter; only meaningful with ``six_band=True``.
    six_band : bool, optional
        Include the two ancillary states at Wyckoff 1a.
    x : mapping, optional
        Per-bond weights ``x_i`` for the sublattice-exchanging coset
        ``m_x G_0``.  All 1 (the default) gives the full ``6mm`` symmetry;
        arXiv:1807.09729 Table I uses ``x != 1`` to descend to ``p3m1``.

    Returns
    -------
    pythtb.TBModel
        Spinful 2D model: 2 orbitals / 4 bands, or 3 orbitals / 6 bands.
    """
    norb = 3 if six_band else 2
    order = ["A", "B", "C"][:norb]
    lat = Lattice(PWV_LATTICE.tolist(),
                  [list(PWV_SITES[s]) for s in order], periodic_dirs=[0, 1])
    model = TBModel(lat, spinful=True)

    def _add(acc, scale):
        folded: dict = {}
        for (i, j, cell), block in acc.items():
            if i >= norb or j >= norb:
                continue
            key, rev = (i, j, tuple(cell)), (j, i, tuple(-np.asarray(cell)))
            mat = block / 12.0
            if rev in folded and key != rev:
                folded[rev] = folded[rev] + mat.conj().T
            else:
                folded[key] = folded.get(key, 0) + mat
        for (i, j, cell), block in folded.items():
            model.set_hop(scale * block, i, j, list(cell), mode="add")

    _add(_pwv_accumulate([1, 2, 3, 4, 5], x=x), 1.0)
    if six_band:
        _add(_pwv_accumulate([6, 7], x=x), np.sin(np.pi * float(mu)))
        model.set_onsite([0.0, 0.0, 3.0 * (1 - 2 * float(mu))], mode="add")
    return model


# ---------------------------------------------------------------------------
# 7.  The spinless p_x, p_y honeycomb TCI -- the 2D-multiplet golden (A3b(b))
# ---------------------------------------------------------------------------
#
# PROVENANCE.  Cano, Bradlyn, Wang, Elcoro, Vergniory, Felser, Aroyo &
# Bernevig, "Topology of Disconnected Elementary Band Representations", PRL
# **120**, 266401 (2018) = arXiv:1711.11045 -- the SAME paper as the P4_232
# fixture, section "Spinless TCI on the honeycomb lattice", Eqs. (2)-(4), with
# the supplement's Sec. S1 supplying the geometry the main text draws only in
# Fig. 1a.  Everything below is transcribed from the arXiv LaTeX source:
#
# * lattice basis (Supplement Sec. S1, first display):
#     e_1 = (sqrt3/2) x + (1/2) y,   e_2 = (sqrt3/2) x - (1/2) y
# * sites (second display):  r_A = (2/3) e_1 - (1/3) e_2,
#                            r_B = (1/3) e_1 - (2/3) e_2   -- Wyckoff 2b
# * nearest-neighbour vectors (the commented-out ``eq:defdelta`` block of the
#   supplement, and the only definition of ``delta_i`` anywhere in the paper):
#     d_1 = -(e_1 + e_2)/3,  d_2 = (2 e_1 - e_2)/3,  d_3 = (-e_1 + 2 e_2)/3
# * Eq. (2):  H^0_k = [[0, h_k], [h_k^dag, 0]]  (A/B sublattice blocks)
# * Eq. (3):  h_k = (1/2)(sum_i e^{-i k.d_i})(t_s + t_p) I
#                 + (1/2)(e^{-i k.d_1} - (1/2)e^{-i k.d_2} - (1/2)e^{-i k.d_3})
#                       (t_s - t_p) sigma_z
#                 + (sqrt3/4)(e^{-i k.d_2} - e^{-i k.d_3})(t_s - t_p) sigma_x
# * Eq. (4):  H^1_k = sin(k.e_1/2) sin(k.e_2/2) sin(k.(e_1-e_2)/2) tau_z (x) sigma_y
# * Fig. 1c parameters: t_sigma = .8, t_pi = 1.0, x = .6
#   Fig. 1b parameters: t_sigma = 1,  t_pi = -.5, x = 0  (the semimetal)
#
# Two defects measured here rather than assumed:
#
# 1. The supplement's Slater-Koster display (``eq:hopping``) prints
#    ``t_xx(d) = (1/3)[(d.x)^2 t_sigma + (d.y)^2 t_pi]`` with **non-unit** d
#    (|d|^2 = 1/3), which is off by a factor of nine from the main text's
#    Eq. (3): it gives ``t_sigma/9`` on the bond along x where Eq. (3) gives
#    ``t_sigma``.  The builder uses the standard Slater-Koster form
#    ``t_ab(d) = t_sigma n_a n_b + t_pi (delta_ab - n_a n_b)`` with
#    ``n = d/|d|``, which reproduces Eq. (3) term by term to 2.2e-16 -- so the
#    stray 1/3 (and the non-normalized d) is a misprint, not a convention.
# 2. The paper's Fourier convention (supplement, ``c_{k,L,a} = sum_R
#    e^{i k.(R + r_L)} c_{R+r_L,a}``, hence the ``e^{-i k.d}`` of Eq. (3)) is
#    the **complex conjugate** of pythtb's convention I.  So the built model
#    satisfies ``H_pythtb(k) = conj(H_paper(k)) = H_paper(-k)`` -- the same
#    crystal read with the opposite Bloch-phase sign, which changes no
#    spectrum, no degeneracy and no irrep multiplicity.  ``test_fixtures_smoke``
#    measures that identity at 1.3e-15 rather than declaring it.
#
# One right-handedness bookkeeping note: pythtb requires a right-handed basis
# and the paper's ``(e_1, e_2)`` is left-handed, so :data:`PXPY_LATTICE` is
# ``(e_2, e_1)``.  That is a relabelling of the two reduced indices and nothing
# else; :func:`pxpy_bloch_hamiltonian` takes **cartesian** k precisely so the
# comparison never has to care.

#: Paper ``e_1`` and ``e_2`` (Supplement Sec. S1), in cartesian units.
PXPY_E1 = np.array([np.sqrt(3) / 2, 0.5])
PXPY_E2 = np.array([np.sqrt(3) / 2, -0.5])

#: The builder's lattice rows: ``(e_2, e_1)`` -- see the right-handedness note.
PXPY_LATTICE = np.array([PXPY_E2, PXPY_E1])

#: Nearest-neighbour vectors ``delta_{1,2,3}`` of Fig. 1a, in cartesian units.
PXPY_DELTAS = np.array([-(PXPY_E1 + PXPY_E2) / 3,
                        (2 * PXPY_E1 - PXPY_E2) / 3,
                        (-PXPY_E1 + 2 * PXPY_E2) / 3])

#: Reduced site positions in the **builder's** basis: the honeycomb, Wyckoff 2b
#: of ``P6mm`` (2c of the ``P6/mmm`` slab cell).
PXPY_SITES = {"A": (2 / 3, 2 / 3), "B": (1 / 3, 1 / 3)}

#: Fig. 1c (gapped TCI) and Fig. 1b (semimetal) parameter sets.
PXPY_PARAMS = {
    "tci": dict(t_sigma=0.8, t_pi=1.0, x=0.6),
    "semimetal": dict(t_sigma=1.0, t_pi=-0.5, x=0.0),
}

_PXPY_SX = np.array([[0.0, 1.0], [1.0, 0.0]])
_PXPY_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]])
_PXPY_SZ = np.diag([1.0, -1.0])


def pxpy_slater_koster(delta, t_sigma, t_pi):
    """The 2x2 ``(p_x, p_y)`` hopping across one bond, Slater-Koster.

    ``t_ab = t_sigma n_a n_b + t_pi (delta_ab - n_a n_b)`` with ``n`` the unit
    bond vector.  Summed with the Bloch phases of Eq. (3) this *is* Eq. (3);
    the supplement's own display of the same thing carries a stray 1/3 (module
    provenance note 1).
    """
    n = np.asarray(delta, dtype=float)
    n = n / np.linalg.norm(n)
    return t_sigma * np.outer(n, n) + t_pi * (np.eye(2) - np.outer(n, n))


def pxpy_bloch_hamiltonian(k_cart, t_sigma=None, t_pi=None, x=None):
    r"""The paper's analytic 4x4 Bloch Hamiltonian, Eqs. (2)-(4), **verbatim**.

    Basis order ``(A p_x, A p_y, B p_x, B p_y)``, so a paper term
    ``tau_i (x) sigma_j`` is ``numpy.kron(tau_i, sigma_j)``.

    Parameters
    ----------
    k_cart : (2,) array_like
        k in **cartesian** units, i.e. ``k.e_i`` is what Eq. (4) calls
        ``k.e_i``.  Convert a reduced k of the built model with
        ``numpy.linalg.solve(PXPY_LATTICE, 2*pi*k_red)``.
    t_sigma, t_pi, x : float, optional
        Default to :data:`PXPY_PARAMS`\ ``["tci"]``.

    Returns
    -------
    (4, 4) numpy.ndarray of complex

    Notes
    -----
    This is the reference implementation and it is printed in the paper's own
    Fourier convention, which is the **conjugate** of pythtb's: the built model
    satisfies ``H_pythtb(k) = conj(H_paper(k))``, measured in
    ``test_fixtures_smoke.py`` at 1.3e-15.
    """
    p = dict(PXPY_PARAMS["tci"])
    for name, val in zip(("t_sigma", "t_pi", "x"), (t_sigma, t_pi, x)):
        if val is not None:
            p[name] = val
    k = np.asarray(k_cart, dtype=float).reshape(-1)[:2]
    ts, tp = p["t_sigma"], p["t_pi"]

    ph = np.exp(-1j * (PXPY_DELTAS @ k))
    h = 0.5 * ph.sum() * (ts + tp) * np.eye(2)
    h = h + 0.5 * (ph[0] - 0.5 * ph[1] - 0.5 * ph[2]) * (ts - tp) * _PXPY_SZ
    h = h + np.sqrt(3) / 4 * (ph[1] - ph[2]) * (ts - tp) * _PXPY_SX

    H = np.zeros((4, 4), dtype=complex)
    H[:2, 2:] = h
    H[2:, :2] = h.conj().T
    a, b = float(k @ PXPY_E1), float(k @ PXPY_E2)
    f = np.sin(a / 2) * np.sin(b / 2) * np.sin((a - b) / 2)
    return H + p["x"] * f * np.kron(_PXPY_SZ, _PXPY_SY)


def pxpy_honeycomb(t_sigma=None, t_pi=None, x=None, regime="tci", turn=0.0):
    """Build the spinless ``p_x, p_y`` honeycomb model of arXiv:1711.11045.

    Nearest-neighbour Slater-Koster hoppings (Eqs. (2)-(3)) plus the
    next-nearest-neighbour gap opener (Eq. (4)), which expands as

    ``H^1_k = (x/4) [sin(k.(e1-e2)) - sin(k.e1) + sin(k.e2)] tau_z (x) sigma_y``

    -- three imaginary same-sublattice bonds with a Haldane-like sign pattern
    and the orbital matrix ``sigma_y``.

    Parameters
    ----------
    t_sigma, t_pi, x : float, optional
        Override the regime's values.
    regime : {"tci", "semimetal"}, optional
        ``"tci"`` is Fig. 1c (gapped, the disconnected EBR); ``"semimetal"``
        is Fig. 1b (``x = 0``: the symmetry-required touchings at Gamma and K).
    turn : float, optional
        Write each site's pair in a basis turned inside itself by this angle,
        in radians, the same at both sites -- an orientation of the Wannier
        pairs, a change of basis and nothing else.  Default ``0.0``.

    Returns
    -------
    pythtb.TBModel
        Spinless 2D model, 4 orbitals / 4 bands, basis order
        ``(A p_x, A p_y, B p_x, B p_y)``, turned by ``turn``.
    """
    if regime not in PXPY_PARAMS:
        raise ValueError(f"regime must be one of {sorted(PXPY_PARAMS)}, got {regime!r}")
    p = dict(PXPY_PARAMS[regime])
    for name, val in zip(("t_sigma", "t_pi", "x"), (t_sigma, t_pi, x)):
        if val is not None:
            p[name] = val

    inv = np.linalg.inv(PXPY_LATTICE)
    rA = np.asarray(PXPY_SITES["A"], dtype=float)
    rB = np.asarray(PXPY_SITES["B"], dtype=float)
    lat = Lattice(PXPY_LATTICE.tolist(),
                  [list(rA), list(rA), list(rB), list(rB)], periodic_dirs=[0, 1])
    m = TBModel(lat, spinful=False)

    # the same turn at both sites leaves the same-sublattice sigma_y blocks as
    # they are and writes every bond block in the turned bases, V^T t V
    V = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    for d in PXPY_DELTAS:
        cell = np.rint((rA @ PXPY_LATTICE + d) @ inv - rB).astype(int)
        t = pxpy_slater_koster(d, p["t_sigma"], p["t_pi"])
        if turn:
            t = V.T @ t @ V
        for a in range(2):
            for b in range(2):
                if abs(t[a, b]) > 1e-15:
                    m.set_hop(t[a, b], a, 2 + b, cell.tolist(), mode="add")

    # The minus sign is the conjugate Fourier convention of provenance note 2:
    # conj(x f(k) tau_z (x) sigma_y) = -x f(k) tau_z (x) sigma_y.
    for r_cart, coef in ((PXPY_E1 - PXPY_E2, -0.125j),
                         (PXPY_E1, +0.125j), (PXPY_E2, -0.125j)):
        R = np.rint(r_cart @ inv).astype(int).tolist()
        for site, tau_z in ((0, +1.0), (2, -1.0)):
            block = -p["x"] * coef * tau_z * _PXPY_SY
            for a in range(2):
                for b in range(2):
                    if abs(block[a, b]) > 1e-15:
                        m.set_hop(block[a, b], site + a, site + b, R, mode="add")
    return m


#: The Pauli matrices on the spin of :func:`pxpy_honeycomb_spinful`.
_SIGMA = (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
          np.array([[0.0, -1.0j], [1.0j, 0.0]]),
          np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))


def pxpy_honeycomb_spinful(soc=None, lam=0.1, turn_a=0.0, turn_b=0.0):
    """The ``p_x, p_y`` honeycomb with spin, and at most one small spin-orbit term.

    Every hopping of :func:`pxpy_honeycomb` (the gapped ``"tci"`` regime) is
    carried on both spins alike, and one symmetric spin-orbit term is added:

    ``None``
        none: every spin rotation is conserved.
    ``"LzSz"``
        on-site ``lam · L_z S_z``, with ``L_z = sigma_y`` on each ``(p_x, p_y)``
        pair; ``S_z`` is conserved and the horizontal mirror holds.
    ``"sigma.d"``
        nearest-neighbour ``i lam (sigma . d)`` on each bond ``d``, the orbital
        carried to itself: the in-plane twofolds hold, the horizontal and the
        vertical mirrors do not, and no spin component is conserved.
    ``"rashba"``
        nearest-neighbour ``i lam (sigma x d)_z``: the vertical mirrors hold, the
        horizontal mirror and the in-plane twofolds do not.

    ``turn_a`` and ``turn_b`` write the ``A`` and ``B`` pairs in bases turned by
    their own angles, radians -- a change of basis and nothing else.

    Returns
    -------
    pythtb.TBModel
        Spinful 2D model, 4 orbitals / 8 bands, basis order
        ``(A p_x, A p_y, B p_x, B p_y)``, turned as asked.
    """
    if soc not in (None, "LzSz", "sigma.d", "rashba"):
        raise ValueError(f"soc must be None, 'LzSz', 'sigma.d' or 'rashba', got {soc!r}")
    base = pxpy_honeycomb()
    sx, sy, sz = _SIGMA
    inv = np.linalg.inv(PXPY_LATTICE)
    rA = np.asarray(PXPY_SITES["A"], dtype=float)
    rB = np.asarray(PXPY_SITES["B"], dtype=float)

    # every term as a 2x2 spin block per (i, j, R), both halves of each conjugate pair
    blocks = {}

    def add(i, j, R, block):
        key = tuple(int(x) for x in R)
        blocks.setdefault(key, np.zeros((4, 4, 2, 2), dtype=complex))[i, j] += block
        back = tuple(-int(x) for x in R)
        blocks.setdefault(back, np.zeros((4, 4, 2, 2), dtype=complex))[j, i] += block.conj().T

    for h in base.hoppings:
        add(h["from_orbital"], h["to_orbital"], h.get("lattice_vector", [0, 0]),
            complex(h["amplitude"]) * np.eye(2))
    if soc == "LzSz":
        for site in (0, 2):
            add(site, site + 1, (0, 0), -1j * lam * sz)
    elif soc in ("sigma.d", "rashba"):
        for d in PXPY_DELTAS:
            cell = np.rint((rA @ PXPY_LATTICE + d) @ inv - rB).astype(int)
            n = d / np.linalg.norm(d)
            spin = (n[0] * sx + n[1] * sy if soc == "sigma.d" else n[1] * sx - n[0] * sy)
            for a in range(2):
                add(a, 2 + a, cell, 1j * lam * spin)

    # the pairs written in turned bases: V^T H V on the orbital index
    V = np.zeros((4, 4))
    for s, th in ((0, turn_a), (2, turn_b)):
        V[s:s + 2, s:s + 2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    m = TBModel(Lattice(PXPY_LATTICE.tolist(), [list(rA), list(rA), list(rB), list(rB)],
                        periodic_dirs=[0, 1]), spinful=True)
    onsite, amps, ii, jj, RR = None, [], [], [], []
    for key, H in blocks.items():
        H = np.einsum("ai,abst,bj->ijst", V, H, V)
        R = np.asarray(key, dtype=int)
        nz = np.flatnonzero(R)
        if nz.size and R[nz[0]] < 0:
            continue                                  # the conjugate half; pythtb adds it
        for i in range(4):
            for j in range(4):
                if not nz.size and i > j:
                    continue
                if not nz.size and i == j:
                    continue
                if np.abs(H[i, j]).max() > 1e-14:
                    amps.append(H[i, j]); ii.append(i); jj.append(j); RR.append(R)
        if not nz.size:
            onsite = [0.5 * (H[i, i] + H[i, i].conj().T) for i in range(4)]
    m.set_onsite(onsite if onsite is not None else [np.zeros((2, 2))] * 4, mode="set")
    m._append_hops(np.asarray(amps), np.asarray(ii), np.asarray(jj), np.asarray(RR))
    return m


_PXPY_COMMON = dict(
    provenance=(
        "Cano, Bradlyn, Wang, Elcoro, Vergniory, Felser, Aroyo & Bernevig, "
        "Phys. Rev. Lett. 120, 266401 (2018) = arXiv:1711.11045, section "
        "'Spinless TCI on the honeycomb lattice': Eqs. (2), (3), (4); geometry "
        "(e_1, e_2, r_A, r_B, delta_i) from Supplemental Material Sec. S1; "
        "parameters from the Fig. 1b / 1c captions.  Two printing defects "
        "measured at build time and documented in the module provenance block: "
        "the supplement's Slater-Koster display carries a stray 1/3 with "
        "non-unit delta, and the paper's Fourier convention is the conjugate "
        "of pythtb's."
    ),
    nbands=4,
    n_occupied=2,
    band_sets={"P2": (0, 1), "Q2": (2, 3), "P4": (0, 1, 2, 3)},
    group="p6mm",
    sg_number=None,
    slab_sg_number=191,
)

PXPY = {
    "tci": FixtureMeta(
        name="pxpy_honeycomb_tci",
        params=dict(PXPY_PARAMS["tci"]),
        regime="gapped disconnected EBR (Fig. 1c)",
        notes=(
            "Spinless, 2 sites x (p_x, p_y) -> 4 bands.  **The first fixture "
            "kept on the UNDECORATED slab cell**: p_x and p_y are in-plane, "
            "hence even under the horizontal mirror, so the sigma_h-composed "
            "operations act trivially on the orbital space and the model can "
            "be worked in the FULL slab group P6/mmm1' (#191.234) -- all 48 "
            "operations represented -- instead of the #183 "
            "sub-group every other 2D fixture here is decorated down to.  "
            "Build it with realized_group(..., environment='free-standing'): "
            "the free-standing reading "
            "is what says 'sigma_h is real here', and the channel layout "
            "measured there glues p_x and p_y into one two-dimensional channel "
            "because C_3 MIXES them (one channel per orbital reaches a best "
            "covariance of 0.329 and is refused). "
            "The site group is -6m2 (D_3h) and the measured multiplet is E'; "
            "declaring E' (or the px / py shorthands) narrows the admissible "
            "global characters from 8 to the four sigma_h-even ones, while E'' "
            "flips every parity label wholesale -- which is the honest "
            "statement that a 2D pythtb model cannot supply z-parity by "
            "itself.  Measured: E(GM) = (-2.7 x2, +2.7 x2) = -+1.5(t_pi+t_sig) "
            "(paper Table S-II), E(M) = (-1.1, -0.7, 0.7, 1.1) (Table S-I), "
            "E(K) = (-0.389711 x2, 0.089711, 0.689711) (the supplement's "
            "closed forms); the direct gap is the exact K-point number "
            "0.4794228634, and the indirect gap is grid-dependent (0.47575 on "
            "90 divisions, 0.47515 on 120) because the valence maximum sits at "
            "no special point.  Content in the paper's own group P6mm1' (#183.186, "
            "the decorated cell): valence (GM5, K3, M3, M4) -- literally the "
            "first of the two options its Sec. S1B lists -- conduction (GM6, "
            "K1, K2, M1, M2), and the 4-band space is 1 x E^G(4)@2b exactly."
        ),
        **_PXPY_COMMON,
    ),
    "semimetal": FixtureMeta(
        name="pxpy_honeycomb_semimetal",
        params=dict(PXPY_PARAMS["semimetal"]),
        regime="x = 0: the symmetry-required semimetal (Fig. 1b)",
        notes=(
            "x = 0 removes the next-nearest-neighbour term, and the two middle "
            "bands touch: measured min(E_2 - E_1) = 1.0e-15 over a 90x90 grid, "
            "with E(GM) = (-0.75 x2, +0.75 x2) and E(K) = (-2.25, 0, 0, 2.25) "
            "-- the paper's 'degeneracies at K and Gamma are "
            "symmetry-required'.  Kept as the negative control for the gapped "
            "regime, not for content measurements."
        ),
        **_PXPY_COMMON,
    ),
}


# ---------------------------------------------------------------------------
# 8.  I2_1 3 (SG 199) -- the deliberately pathological golden (A3b(c))
# ---------------------------------------------------------------------------
#
# The builders and their constants -- ``I213_LATTICE``, ``I213_U``,
# ``I213_BONDS``, :func:`i213_sites`, :func:`i213_bloch_hamiltonian`,
# :func:`i213_flicker2018` and :func:`i213_conventional` (the deliberate
# conventional-cell supercell that ``_group_of`` must refuse) -- moved to
# :mod:`pythtqc.models` on 2026-08-25, provenance comment and all;
# they are imported and re-exported at the top of this module.


I213 = FixtureMeta(
    name="i213_flicker2018",
    provenance=(
        "Flicker, de Juan, Bradlyn, Morimoto, Vergniory & Grushin, 'Chiral "
        "Optical Response of Multifold Fermions', Phys. Rev. B 98, 155145 "
        "(2018) = arXiv:1806.09642, Appendix F.1 'SGs 199 and 214': Eqs. (88) "
        "(lattice), (89) (generators), (92) (the 8a orbit), (93) (the "
        "nearest-neighbour distance), (94) (the real-space Hamiltonian) and "
        "(96)-(98) (its Bloch form).  The band structure is the paper's "
        "Fig. 4(b); it is schematic in energy units, so t_NN = 1 here."
    ),
    params=dict(u=I213_U, t=1.0),
    nbands=4,
    n_occupied=1,
    band_sets={"EBR4": (0, 1, 2, 3), "LOW3": (0, 1, 2), "TOP1": (3,)},
    group="I2_13",
    sg_number=199,
    slab_sg_number=None,
    regime="nearest-neighbour minimal model, generic u (SG 199, not 214)",
    notes=(
        "Spinless s at Wyckoff 8a (multiplicity 4 in the primitive cell), site "
        "symmetry '.3.'.  Deliberately pathological, three ways at once: "
        "(i) BODY-CENTRED, so i213_conventional() is served as the index-2 "
        "LATTICE SUPERGROUP (I2_131' #199.13, 48 operations, pure translation "
        "[.5 .5 .5], no tables) and realized_group must refuse it with "
        "gate='supercell' -- this Hamiltonian obeys that translation, measured "
        "at a covariance residual of exactly 0.0 -- while the primitive build "
        "reads I2_131' #199.13 with 24 of 24 "
        "operations validated and every verdict 'obeyed'; (ii) NONSYMMORPHIC "
        "with NO intrinsic screw -- 216 of 576 product-table pairs carry a "
        "nonzero translation difference, yet every one of the twelve unitary "
        "operations has its axial translation component in the lattice of "
        "projections along its own axis, so each is a pure rotation about a "
        "shifted axis; (iii) the four sites PERMUTE BETWEEN CELLS -- 36 of the "
        "96 (operation, site) images carry a nonzero lattice shift, and the "
        "detector's own ansatz-free unitaries put their support exactly where "
        "_assign_orbitals says (worst off/on ratio 1.2e-15).  Measured spectra "
        "(t = 1): E(GM) = (-1, -1, -1, 3), E(H) = (-3, 1, 1, 1), "
        "E(P) = (-sqrt3 x2, +sqrt3 x2), E(N) = (-sqrt5, -1, 1, sqrt5).  The "
        "threefold nodes at GM and H are the paper's 'spin-1 fermions at Gamma "
        "and H'; because the singlet is at the TOP at GM (GM1) and at the "
        "BOTTOM at H (H1), the four-band EBR is CONNECTED -- the opposite of "
        "the P4_232 fixture.  Content: {GM1, GM4} / {H1, H4} / {N1: 2, N2: 2} / "
        "{P1, P3} / {PA1, PA3} reproduce the irreptables row A1^G(4)@8a "
        "exactly, at every one of the five points.  P and PA used to refuse "
        "with gate='integer_multiplicity' (defect exactly 0.5), a second "
        "instance of the Fd-3m W-point class; that was BCS's e^{+ik.t} "
        "convention for lattice translations meeting pythtb's physical "
        "e^{-ik.t}, fixed 2026-08-21 by conjugating the factor system at the "
        "boundary -- see test_multiorbital.py, which also asserts that P and "
        "PA keep their own (chirality-distinct) labels."
    ),
)


# ---------------------------------------------------------------------------
# 9.  screw-group atomic controls -- P4_122 (#91) and P3_121 (#152, quartz)
# ---------------------------------------------------------------------------
#
# Constructed here, not ported: the only two fixtures in the suite whose group
# has a k-point where the tabulated little-group table cannot be carried into
# pythtb's convention by re-attaching a phase, and whose physical table is
# therefore the whole-character conjugate of the tabulated one (see
# ``groups._table_conjugation``).  Both are deep atomic limits, so their
# content at *every* point -- the previously-refused ones included -- is
# closed-form arithmetic: an s orbital on a maximal Wyckoff orbit induces
# exactly one EBR row, and ``decompose`` must return that row and nothing else.

#: ``{R|t}`` of P4_122 (#91) in the ITA / BCS setting, acting on **reduced**
#: coordinates as ``q -> R q + t``, in ``irreptables``' own index order (the
#: smoke suite asserts that identity against the shipped table).
P4122_OPS = (
    ((1, 0, 0, 0, 1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),          # 1   E
    ((-1, 0, 0, 0, -1, 0, 0, 0, 1), (0.0, 0.0, 0.5)),        # 2   2(0,0,1/2) 0,0,z
    ((0, -1, 0, 1, 0, 0, 0, 0, 1), (0.0, 0.0, 0.25)),        # 3   4+(0,0,1/4)
    ((0, 1, 0, -1, 0, 0, 0, 0, 1), (0.0, 0.0, 0.75)),        # 4   4-(0,0,3/4)
    ((-1, 0, 0, 0, 1, 0, 0, 0, -1), (0.0, 0.0, 0.0)),        # 5   2 0,y,0
    ((1, 0, 0, 0, -1, 0, 0, 0, -1), (0.0, 0.0, 0.5)),        # 6   2 x,0,1/4
    ((0, 1, 0, 1, 0, 0, 0, 0, -1), (0.0, 0.0, 0.75)),        # 7   2 x,x,3/8
    ((0, -1, 0, -1, 0, 0, 0, 0, -1), (0.0, 0.0, 0.25)),      # 8   2 x,-x,1/8
)

#: ``{R|t}`` of P3_121 (#152) -- quartz's space group -- same conventions.
P3121_OPS = (
    ((1, 0, 0, 0, 1, 0, 0, 0, 1), (0.0, 0.0, 0.0)),          # 1   E
    ((0, -1, 0, 1, -1, 0, 0, 0, 1), (0.0, 0.0, 1 / 3)),      # 2   3+(0,0,1/3)
    ((-1, 1, 0, -1, 0, 0, 0, 0, 1), (0.0, 0.0, 2 / 3)),      # 3   3-(0,0,2/3)
    ((0, 1, 0, 1, 0, 0, 0, 0, -1), (0.0, 0.0, 0.0)),         # 4   2 x,x,0
    ((1, -1, 0, 0, -1, 0, 0, 0, -1), (0.0, 0.0, 2 / 3)),     # 5   2 x,0,1/3
    ((-1, 0, 0, -1, 1, 0, 0, 0, -1), (0.0, 0.0, 1 / 3)),     # 6   2 0,y,1/6
)

#: Primitive tetragonal, ``c/a = 1.6`` -- generic, so nothing is promoted.
P4122_LATTICE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.6]])

#: Primitive hexagonal (gamma = 120 deg), ``c/a = 1.1``.
P3121_LATTICE = np.array([[1.0, 0.0, 0.0],
                          [-0.5, np.sqrt(3) / 2, 0.0],
                          [0.0, 0.0, 1.1]])

#: 4a ``(0,y,0)`` and 4b ``(1/2,y,0)`` of P4_122, generic free parameters, and
#: the on-site splitting / hoppings of the atomic limit.
P4122_PARAMS = dict(y_a=0.30, y_b=0.17, delta=2.5, t_a=0.09, t_aa=0.07,
                    t_b=0.05, t_ab=0.03)

#: 3a ``(x,0,1/3)`` and 3b ``(x,0,5/6)`` of P3_121, likewise.
P3121_PARAMS = dict(x_a=0.45, x_b=0.20, delta=2.5, t_a=0.09, t_b=0.05,
                    t_ab=0.03)


# The shared construction helpers -- :func:`orbit_of`, :func:`closed_bonds`,
# :func:`atomic_model`, ``PAULI``, :func:`cartesian_rotation`,
# :func:`spinor_of`, :func:`spin_bonds` and :func:`primitive_ops` -- moved to
# :mod:`pythtqc.models` on 2026-08-25 (the #80/#171 models in the next
# section need them, and those are public now).  The #91 /
# #152 controls below still build on them, through the re-export at the top of
# this module.


def p4122_sites(y_a=None, y_b=None):
    """``(orbit_4a, orbit_4b)`` of P4_122 in reduced coordinates."""
    y_a = P4122_PARAMS["y_a"] if y_a is None else float(y_a)
    y_b = P4122_PARAMS["y_b"] if y_b is None else float(y_b)
    return (orbit_of(P4122_OPS, (0.0, y_a, 0.0)),
            orbit_of(P4122_OPS, (0.5, y_b, 0.0)))


def p4122_atomic(y_a=None, y_b=None, delta=None, t_a=None, t_aa=None,
                 t_b=None, t_ab=None):
    """Atomic insulator in P4_122 (#91): ``s`` at 4a below, ``s`` at 4b above.

    Eight spinless ``s`` orbitals -- one per site of the 4a orbit ``(0,y,0)``
    and one per site of the 4b orbit ``(1/2,y,0)`` -- split by ``+-delta`` and
    coupled by small real hoppings symmetrized over the group.  The four
    occupied bands are the band representation induced from ``s @ 4a`` and the
    four empty ones the one induced from ``s @ 4b``, with a gap of order
    ``2 delta``.

    The point of the fixture is ``A = (1/2,1/2,1/2)`` and ``Z = (0,0,1/2)``,
    where the 4_1 screw makes the tabulated factor system complex.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 8 orbitals / 8 bands.
    """
    p = dict(P4122_PARAMS)
    for key, val in dict(y_a=y_a, y_b=y_b, delta=delta, t_a=t_a, t_aa=t_aa,
                         t_b=t_b, t_ab=t_ab).items():
        if val is not None:
            p[key] = float(val)
    orbits = p4122_sites(p["y_a"], p["y_b"])
    return atomic_model(
        P4122_LATTICE, P4122_OPS, orbits,
        [-p["delta"]] * 4 + [p["delta"]] * 4,
        [(0, 1, (0, 0, 0), p["t_a"]), (0, 2, (0, 0, 0), p["t_aa"]),
         (4, 5, (0, 0, 0), p["t_b"]), (0, 4, (0, 0, 0), p["t_ab"])])


def p3121_sites(x_a=None, x_b=None):
    """``(orbit_3a, orbit_3b)`` of P3_121 in reduced coordinates."""
    x_a = P3121_PARAMS["x_a"] if x_a is None else float(x_a)
    x_b = P3121_PARAMS["x_b"] if x_b is None else float(x_b)
    return (orbit_of(P3121_OPS, (x_a, 0.0, 1 / 3)),
            orbit_of(P3121_OPS, (x_b, 0.0, 5 / 6)))


def p3121_atomic(x_a=None, x_b=None, delta=None, t_a=None, t_b=None, t_ab=None):
    """Atomic insulator in P3_121 (#152) -- **quartz's space group**.

    Six spinless ``s`` orbitals: the 3a orbit ``(x,0,1/3)`` below and the 3b
    orbit ``(x,0,5/6)`` above, as in :func:`p4122_atomic`.  The three occupied
    bands are ``A^G(3) @ 3a`` and the three empty ones ``A^G(3) @ 3b``.

    Here the previously-refused points ``A``, ``H`` (and ``HA``) and ``L``
    *discriminate*: the 3a row carries ``A1 + A3`` and the 3b row ``A2 + A3``,
    so the fixture tests the transported labels and not only their integrality.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 6 orbitals / 6 bands.
    """
    p = dict(P3121_PARAMS)
    for key, val in dict(x_a=x_a, x_b=x_b, delta=delta, t_a=t_a, t_b=t_b,
                         t_ab=t_ab).items():
        if val is not None:
            p[key] = float(val)
    orbits = p3121_sites(p["x_a"], p["x_b"])
    return atomic_model(
        P3121_LATTICE, P3121_OPS, orbits,
        [-p["delta"]] * 3 + [p["delta"]] * 3,
        [(0, 1, (0, 0, 0), p["t_a"]), (3, 4, (0, 0, 0), p["t_b"]),
         (0, 3, (0, 0, 0), p["t_ab"])])


_SCREW_PROVENANCE = (
    "Constructed here (2026-08-21), not ported: a minimal atomic-limit control "
    "for the convention boundary this suite could not otherwise reach.  The "
    "{R|t} are the ITA / BCS setting of the group, taken to be identical to "
    "irreptables' own shipped operations (asserted in test_fixtures_smoke.py); "
    "the Wyckoff orbits are the ITA special positions with generic free "
    "parameters; the hoppings are arbitrary small reals symmetrized over the "
    "group, so nothing about the parameter values is load-bearing beyond "
    "'small, real, generic'.  The oracle is closed form: an s orbital on a "
    "maximal Wyckoff orbit induces exactly one EBR row."
)

P4122 = FixtureMeta(
    name="p4122_atomic",
    provenance=_SCREW_PROVENANCE,
    params=dict(P4122_PARAMS),
    nbands=8,
    n_occupied=4,
    band_sets={"VALENCE_4a": (0, 1, 2, 3), "CONDUCTION_4b": (4, 5, 6, 7)},
    group="P4_122",
    sg_number=91,
    slab_sg_number=None,
    regime="deep atomic limit (delta = 2.5 against hoppings <= 0.09)",
    notes=(
        "The 4_1 family of the convention completion.  _group_of reads "
        "P4_1221' #91.104 with include_TR=True (16 operations, every verdict "
        "'obeyed') and #91 with include_TR=False; _assign_orbitals gives two "
        "orbits with Wyckoff letters ('a', 'b') and site symmetry '.2.'.  At "
        "A = (1/2,1/2,1/2) and Z = (0,0,1/2) the tabulated factor system "
        "omega_0(R,S) = e^{2pi i (R^T k - k).t_S} is a fourth root of unity "
        "(max |Im omega_0| = 1.0), so the physical table is the WHOLE-CHARACTER "
        "CONJUGATE of the tabulated one and ebr_data(group)['convention']"
        "['conjugated_points'] == ('A', 'Z').  Measured content: valence "
        "{GM1, GM2, GM5} / {A1, A2} / {M1, M2, M5} / {2 R1} / {2 X1, X3, X4} / "
        "{Z1, Z2} = exactly the A^G(4)@4a row, conduction = the A^G(4)@4b row, "
        "and decompose returns 1 x A^G(4) @ 4a (resp. 4b), claim 'exact'.  "
        "Before the completion both A and Z refused with "
        "gate='integer_multiplicity' and defect exactly 0.5 -- the Fd-3m W / "
        "SG 199 P signature -- and decompose refused with "
        "gate='incomplete_content'."
    ),
)

# ---------------------------------------------------------------------------
# 10.  re-gluing controls -- I4_1 (#80) and P6_2 (#171), **spinful**
# ---------------------------------------------------------------------------
#
# The builders -- ``I41_*``, ``P62_*``, :func:`i41_sites`, :func:`i41_atomic`,
# :func:`i41_soc`, :func:`p62_sites`, :func:`p62_atomic` -- moved to
# :mod:`pythtqc.models` on 2026-08-25, taking the exchange-versus-irreducible
# discussion with them; they are imported and
# re-exported at the top of this module.  What stays here is the test-facing
# :class:`FixtureMeta` records, whose notes carry the measured locks.


_REGLUE_PROVENANCE = (
    "Constructed here (2026-08-21), not ported: the atomic-limit controls for "
    "the k-points where the time-reversal GLUING is convention-sensitive, so "
    "that the physical corep list is a different partition of the unitary "
    "irreps than BCS's. The {R|t} are the ITA / BCS setting of the group "
    "(asserted against irreptables' own shipped operations in "
    "test_fixtures_smoke.py); I4_1's are carried onto the primitive "
    "body-centred cell by fixture_models.primitive_ops because the "
    "conventional cell is not primitive and _group_of refuses it. The orbits "
    "are ITA special positions plus one generic orbit, which is what pins the "
    "group down: an s-orbital model on the special orbit alone is far more "
    "symmetric than the chiral group (I4_1 -> I4_1/amd #141, P6_2 -> P6_222 "
    "#180). The hoppings are arbitrary small reals symmetrized over the group "
    "and spin-independent, so the band representation is the spinless one "
    "times spin 1/2 and the content at every k-point is closed form."
)

I41 = FixtureMeta(
    name="i41_atomic",
    provenance=_REGLUE_PROVENANCE,
    params=dict(I41_PARAMS),
    nbands=12,
    n_occupied=4,
    band_sets={"VALENCE_4a": (0, 1, 2, 3),
               "CONDUCTION_8b": (4, 5, 6, 7, 8, 9, 10, 11)},
    group="I4_1",
    sg_number=80,
    slab_sg_number=None,
    regime="deep atomic limit (delta = 2.5 against hoppings <= 0.09)",
    notes=(
        "The re-gluing anchor.  _group_of reads I4_11' #80.30 with "
        "include_TR=True and #80 with include_TR=False; _assign_orbitals gives "
        "two orbits with Wyckoff letters ('a', 'b').  At P = (1/2,1/2,1/2) "
        "conventional = (1/4,1/4,1/4) primitive -- a NON-TRIM of the "
        "body-centred lattice, since -P - P has odd conventional sum -- the "
        "unitary little group is {E, 2_z}, pure time reversal is not in the "
        "extended little group, and the pairing antiunitary is the C4.T coset. "
        "Every term of the Wigner-Herring sum there carries e^{s.2pi.i k.m} "
        "with k.m in {1/4, 3/4}, so the indicators swap between the two "
        "conventions: BCS glues -P3P3 and leaves -P4 alone (which is exactly "
        "the shipped table), the physical convention leaves -P3 alone and "
        "glues -P4P4.  ebr_data(group)['convention']['reglued_points'] == "
        "('P',) and the served corep table at P carries labels -P3 (dim 1) "
        "and -P4P4 (dim 2) that BCS's table does not contain.  Measured "
        "valence content: {-GM5GM7, -GM6GM8} / {-M5M7, -M6M8} / {2 -N2N2} / "
        "{2 -P3, -P4P4} / {2 -X3X4} = exactly the transported "
        "-1E2E^G(4)@4a row, and decompose returns 1 x -1E2E^G(4) @ 4a, claim "
        "'exact'.  Before the re-gluing, ebr_data refused the whole group "
        "with gate='ebr_convention'."
    ),
)

I41_SOC = FixtureMeta(
    name="i41_soc",
    provenance=(
        "Constructed here (2026-08-24), on top of i41_atomic: the same "
        "geometry, orbits and on-site splitting with one group-averaged "
        "spin-orbit bond set added (spin_bonds averages i lam b.sigma over the "
        "group, so the model is exactly symmetric by construction rather than "
        "by tuning). Built for two jobs at once -- the detector controls, "
        "where the commutant dimension is the point (the axial variant, whose "
        "conserved s_z makes it two, is test_detector.py's), and the "
        "SOC-doublet demonstration at the re-glued point P."
    ),
    params=dict(I41_PARAMS, lam=0.4, soc="generic"),
    nbands=12,
    n_occupied=4,
    band_sets={"VALENCE_4a": (0, 1, 2, 3),
               "CONDUCTION_8b": (4, 5, 6, 7, 8, 9, 10, 11)},
    group="I4_1",
    sg_number=80,
    slab_sg_number=None,
    regime="deep atomic limit (delta = 2.5) plus spin-orbit lam = 0.4",
    notes=(
        "Exactly I4_11' (#80.30)-symmetric: 8 of 8 operations validate on the "
        "default 8**3 mesh at worst residual 1.5e-14 ('generic') / 2.1e-14 "
        "('axial'), and the built rep closes at 1.4e-15.  As a DETECTOR "
        "control the two seed sets differ only in the commutant: 'generic' "
        "leaves it at C*1 (every nullspace dimension 1), 'axial' conserves "
        "s_z and makes it two-dimensional (the degenerate-nullspace case).  "
        "As the SOC-DOUBLET DEMONSTRATION ('generic', 2026-08-24): the point "
        "of the SOC is that i41_atomic cannot show the re-gluing at all -- "
        "spin-independent hoppings mean SU(2), so P's four valence states are "
        "ONE level (degeneracy pattern [4], spread 4.9e-15, characters "
        "(4, 0)) and both conventions' corep lists span four units of "
        "dimension.  With SOC the level splits to [1, 2, 1]: at lam = 0.4, "
        "-2.744031 / {-2.501964, -2.501964} / -2.259907 (doublet split "
        "4.4e-16, level gaps 0.242066 and 0.242057; at lam = 0.2 the same "
        "pattern with gaps 0.121060, i.e. linear in lam to 0.03%).  The block "
        "content is unmoved -- {-P3: 2, -P4P4: 1}, hsp_table complete, "
        "decompose 1 x -1E2E^G(4)@4a @ 4a claim 'exact' -- but now RESOLVED, "
        "partition (0, 1, 0): chi(2_z) = -i on each singlet -> -P3, +2i on "
        "the doublet -> -P4P4.  [1, 2, 1] alone is consistent with BOTH "
        "conventions (BCS would read it as one -P3P3 plus two lone -P4); what "
        "decides is WHICH irrep the doublet carries, and the sewing "
        "characters say it is the one the physical convention doubles.  With "
        "the re-gluing switched off, BCS's shipped {-P3P3 (2), -P4 (1)} "
        "cannot express the singlets at all: each refuses with "
        "gate='integer_multiplicity' at a defect of exactly 0.5 "
        "(multiplicities (0.5, 0)), the whole block refuses naming rows "
        "[0, 2], and the doublet is mislabelled {-P4: 2} -- an accidental "
        "degeneracy of two lone coreps, which is what an enforced pair is "
        "not.  The atomic fixture under the same switch refuses nothing."
    ),
)

P62 = FixtureMeta(
    name="p62_atomic",
    provenance=_REGLUE_PROVENANCE,
    params=dict(P62_PARAMS),
    nbands=18,
    n_occupied=6,
    band_sets={"VALENCE_3a": (0, 1, 2, 3, 4, 5),
               "CONDUCTION_6c": tuple(range(6, 18))},
    group="P6_2",
    sg_number=171,
    slab_sg_number=None,
    regime="deep atomic limit (delta = 2.5 against hoppings <= 0.09)",
    notes=(
        "The EXCHANGE member of the class, from a different Bravais family.  "
        "_group_of reads P6_21' #171.122 with include_TR=True and #171 with "
        "include_TR=False; letters ('a', 'c').  The moved point is "
        "H = (1/3,1/3,1/2), a non-TRIM whose antiunitary coset is the "
        "two-fold rotations times T; the unitary little group is {E, 3+, 3-}, "
        "and A = (0,0,1/2) moves with it.  The physical gluing there is not "
        "BCS's, but the corep it produces is character-identical to one BCS "
        "ships, so since the 2026-08-24 naming directive the served list is "
        "BCS's own: {-H4 (dim 1, (1,-1,1)), -H5H6 (dim 2, (2,1,-1))}, served "
        "as whole-character conjugates with the labels exchanged with the "
        "rows.  H and A are therefore in `conjugated_points` and in neither "
        "`divergent_points` nor `reglued_points`, and #171.122's EBR transport "
        "is a no-op end to end.  (Before 2026-08-24 the same rows were served "
        "as {-H6 (dim 1), -H4H5 (dim 2)}.)  Measured valence content at H: "
        "{2 -H4, 2 -H5H6} = the transported -1E2E^G(6)@3a row, and decompose "
        "returns claim 'exact' with TWO solutions, 1 x -1E2E^G(6) @ 3a and "
        "@ 3b -- #171's two special orbits carry *identical* EBR vectors, so "
        "the content cannot separate them and `unique` is False by "
        "construction, not by weakness."
    ),
)

P3121 = FixtureMeta(
    name="p3121_atomic",
    provenance=_SCREW_PROVENANCE,
    params=dict(P3121_PARAMS),
    nbands=6,
    n_occupied=3,
    band_sets={"VALENCE_3a": (0, 1, 2), "CONDUCTION_3b": (3, 4, 5)},
    group="P3_121",
    sg_number=152,
    slab_sg_number=None,
    regime="deep atomic limit (delta = 2.5 against hoppings <= 0.09)",
    notes=(
        "The 3_1 family of the convention completion, and the group of "
        "alpha-quartz.  _group_of reads P3_1211' #152.34 with include_TR=True "
        "(12 operations, every verdict 'obeyed') and #152 with "
        "include_TR=False; two orbits, letters ('a', 'b'), site symmetry "
        "'.2.'.  Conjugated points: ('A', 'H', 'HA', 'L') on the grey table, "
        "('A', 'H', 'L') on the unitary one -- max |Im omega_0| = 0.866, a cube "
        "root of unity from the 3_1 screw.  Unlike #91 these points "
        "DISCRIMINATE: valence {A1, A3} / {H1, H3} / {L1, 2 L2} is the "
        "A^G(3)@3a row while conduction {A2, A3} / {H2, H3} / {2 L1, L2} is "
        "A^G(3)@3b, so the labels and not merely their integrality are under "
        "test.  Hand-derivable oracle at A = (0,0,1/2): the induced character "
        "of s@3a is (3,0,0,1,1,-1) on operations 1..6, which the physical "
        "table decomposes as A1 + A3 with multiplicities (1, 0, 1).  Before "
        "the completion A, H and L all refused with defect exactly 0.5."
    ),
)


# ---------------------------------------------------------------------------
# 12.  the paper's P41' flat-band model, and the chiral pattern that lowers it
# ---------------------------------------------------------------------------

#: The square lattice the P41' models live on: two Kramers pairs of s-like
#: orbitals at Wyckoff ``2c``, ``(1/2, 0)`` and ``(0, 1/2)``, in the paper's own
#: order ``(x-up, x-down, y-up, y-down)``.
P41_FLAT_LATTICE = (np.array([[1.0, 0.0], [0.0, 1.0]]),
                    [[0.5, 0.0], [0.0, 0.5]])

#: ``t`` is the flat-band model's one scale; ``t_a``/``t_b`` are the *chiral*
#: next-nearest pattern's two, along the two diagonals.  They are unequal by
#: construction -- that inequality is what breaks the vertical mirrors while
#: keeping ``C4`` and time reversal, and it is the whole point of the fixture.
P41_FLAT_PARAMS = dict(t=1.0, t_a=0.12, t_b=0.05)


def p41_flat_bloch_hamiltonian(k, t=None):
    r"""The supplement's closed form, ``H(k) = t (u+ u+^dag + u- u-^dag)``, at one ``k``.

    ``k`` is reduced, and the Bloch states are written in the momentum basis
    ``e^{i k.(R + t_alpha)}`` -- pythtb's convention I -- so this is directly
    comparable with ``model.hamiltonian``.  The two states are **not**
    orthogonal, so their projector sum has rank 2: two bands sit at exactly zero
    (the orthogonal complement of the span) and two disperse.

    This is the **paper's** closed form, in the paper's own spin convention.
    :func:`p41_flat_band` transcribes it into pythtb's (see its docstring), so
    the two are compared through ``U H U†`` with ``U = 1_2 ⊗ iσ_x`` — which is
    exactly what ``examples/tqc_srsi_p4_flatband.py`` step 1 measures, at
    ``2.259e-16`` over 500 random ``k``.
    """
    t = P41_FLAT_PARAMS["t"] if t is None else float(t)
    kx, ky = 2.0 * np.pi * float(k[0]), 2.0 * np.pi * float(k[1])
    c, s = np.cos(kx / 2), np.sin(kx / 2)
    C, S = np.cos(ky / 2), np.sin(ky / 2)
    u_plus = np.array([c, -1j * s, C, -S]) / np.sqrt(2)
    u_minus = np.array([-1j * s, -c, S, -C]) / np.sqrt(2)
    return t * (np.outer(u_plus, u_plus.conj()) + np.outer(u_minus, u_minus.conj()))


def p41_flat_band(chiral=False, t=None, t_a=None, t_b=None):
    r"""The flat-band model of the SRSI paper's SM section 5A, optionally made chiral.

    The hopping list is the closed form's, derived in
    ``examples/tqc_srsi_p4_flatband.py`` and checked against
    :func:`p41_flat_bloch_hamiltonian` there and in the suite.  As transcribed
    the Hamiltonian obeys **more** than the paper's ``P41'``: the vertical
    mirrors of the square lattice survive it too, and ``realized_group`` reads
    ``P4mm1' #99.164``, 16 of 16.

    The paper's spin blocks are written for the **opposite** quantization axis,
    so they are conjugated by ``W = iσ_x`` here — pythtb's basis is spin along
    ``+z`` in ``(↑, ↓)`` order and a rotation acts on the spinor as
    ``e^{−iθ n̂·σ/2}`` (:mod:`pythtqc.spin`).  The conjugation is written out
    here, block by block, and again in ``examples/tqc_srsi_p4_flatband.py``:
    it is four lines wherever a source in the other frame is transcribed, and
    the package exposes no helper for it.  It is a unitary conjugation of
    ``H(k)``, so no band, gap, Wilson loop or SRSI moves — what moves is the
    ``2c`` site character, from ``B`` (``B2`` read in ``P4mm1'``) to ``A``.

    ``chiral=True`` adds one C4- and time-reversal-symmetric next-nearest
    pattern that breaks exactly those mirrors: ``t_a`` along ``(1, 1)`` and
    ``t_b`` along ``(1, -1)`` on the ``x/2`` site, the two swapped on the
    ``y/2`` site (which is what ``C4`` demands, since it exchanges the sites and
    carries ``(1, 1)`` to ``(1, -1)``), with identity spin blocks so time
    reversal is untouched.  ``t_a != t_b`` is the mirror breaking: ``m_x`` would
    have to exchange the two diagonals on one site, and ``m_[110]`` to carry one
    site's ``t_a`` bond to the other site's ``t_b`` bond.  The Hamiltonian then
    obeys ``P41' #75.2`` and nothing more, which is what ``realized_group``
    descends to.
    """
    from pythtb import Lattice, TBModel

    p = dict(P41_FLAT_PARAMS)
    for name, value in (("t", t), ("t_a", t_a), ("t_b", t_b)):
        if value is not None:
            p[name] = float(value)
    t, t_a, t_b = p["t"], p["t_a"], p["t_b"]
    pauli_0 = np.eye(2, dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    w = 1j * pauli_x                       # onto pythtb's quantization axis

    def flip(block):
        return w @ np.asarray(block, dtype=complex) @ w.conj().T

    model = TBModel(Lattice(P41_FLAT_LATTICE[0], [list(r) for r in P41_FLAT_LATTICE[1]],
                            periodic_dirs=[0, 1]), spinful=True)
    model.set_onsite([flip(0.5 * t * pauli_0), flip(0.5 * t * pauli_0)], mode="set")
    model.set_hop(flip(0.25j * t * pauli_y), 0, 0, [1, 0])   # (t/2) i sin kx
    model.set_hop(flip(0.25j * t * pauli_x), 1, 1, [0, 1])   # -(t/2) sin ky
    model.set_hop(flip(t / 8 * np.array([[1 - 1j, -(1 - 1j)],
                                         [1 + 1j, 1 + 1j]])), 0, 1, [0, 0])
    model.set_hop(flip(t / 8 * np.array([[1 + 1j, 1 + 1j],
                                         [-(1 - 1j), 1 - 1j]])), 0, 1, [1, 0])
    model.set_hop(flip(t / 8 * np.array([[1 + 1j, -(1 + 1j)],
                                         [1 - 1j, 1 - 1j]])), 0, 1, [0, -1])
    model.set_hop(flip(t / 8 * np.array([[1 - 1j, 1 - 1j],
                                         [-(1 + 1j), 1 + 1j]])), 0, 1, [1, -1])
    if chiral:
        model.set_hop(t_a * pauli_0, 0, 0, [1, 1])
        model.set_hop(t_b * pauli_0, 0, 0, [1, -1])
        model.set_hop(t_a * pauli_0, 1, 1, [1, -1])
        model.set_hop(t_b * pauli_0, 1, 1, [1, 1])
    return model


P41_FLAT = FixtureMeta(
    name="p41_flat_band",
    provenance=(
        "Hwang et al., 'Stable real-space invariants and topology beyond "
        "symmetry indicators', Nat. Commun. 17, 8564 (2026), "
        "10.1038/s41467-026-74844-w, supplementary section 5A: "
        "the flat-band model H(k) = t (u+ u+^dag + u- u-^dag).  The closed form is "
        "transcribed as a finite hopping list here and in "
        "examples/tqc_srsi_p4_flatband.py, which derives the list in its comments "
        "and measures it against the closed form at 500 random k (2.259e-16).  The "
        "paper's spin blocks are written for the opposite quantization axis and are "
        "conjugated by i sigma_x in both places, so that up is spin along +z as "
        "pythtb has it; that is a unitary conjugation of H(k), so no band and no "
        "SRSI moves and the 2c site character reads A rather than B.  The "
        "chiral next-nearest pattern is NOT the paper's: it is added here so that "
        "the mirrors the paper's own model happens to keep are broken by the "
        "HAMILTONIAN, which is what realized_group's descent is measured on."
    ),
    params=dict(P41_FLAT_PARAMS),
    nbands=4,
    n_occupied=2,
    band_sets={"FLAT": (0, 1), "DISPERSIVE": (2, 3)},
    group="p4mm as transcribed; p4 with the chiral pattern",
    sg_number=None,
    slab_sg_number=123,   # P4/mmm -- the orbital point set alone, before decoration
    regime=None,
    notes=(
        "Spinful, 2 orbitals -> 4 bands, both at Wyckoff 2c of the square "
        "lattice.  chiral=False: realized_group reads P4mm1' #99.164, 16 of 16 "
        "obeyed, in EITHER reading -- asked free-standing it is measured out of "
        "the P4/mmm1' #123.340 the bare two-site cell allows (a lattice "
        "supergroup of index 2, tau = (1/2,1/2,0)), 44 operations refuted by the "
        "tensor test and 2 -- S4(z), S4^-1(z) -- by closure.  "
        "chiral=True: the vertical mirrors are broken by the Hamiltonian and the "
        "paper's own reading is the SUBSTRATE one -- a flat model's default -- "
        "which DESCENDS #99.164 -> P41' #75.2, 8 of 8; asked free-standing the "
        "same model is returned P41' #75.2 as well, 8 of 8 -- a flat cell cannot "
        "say whether the surviving fourfold is C4 or S4 = sigma_h*C4 (the two have "
        "the same in-plane image and the same matrices), and the tie rule keeps "
        "the rotations, S4(z) and S4^-1(z) recorded under 'tie'.  The model's own "
        "SRSIs are the paper's "
        "theta_Z = (1, 1, 0), theta_Zn = {2: (0, 1)} either way, since those are "
        "read off the ORBITALS.  The chiral pattern is a real perturbation of the "
        "bands: at t_a = 0.12, t_b = 0.05 the two flat bands disperse over 0.34 "
        "and the INDIRECT gap min E2 - max E1 of the transcribed model (0.293412) goes negative (-0.06) while the DIRECT gap min_k (E2 - E1), the one the criterion and the Wilson loops need, stays open (+0.14; the example uses 0.03/0.0125, direct gap +0.258), so this fixture "
        "is the DESCENT's, not the flat band's -- the paper's own numbers are "
        "measured on chiral=False."
    ),
)


# ---------------------------------------------------------------------------
# spinful s orbitals OFF the mirror planes of a centrosymmetric 3D group
# ---------------------------------------------------------------------------
#
# The lift-then-subduce question in the place it matters most (owner,
# 2026-09-10): a space group with inversion and mirrors, and orbitals placed
# where no mirror passes -- so the mirror and the inversion are TRANSPORT
# operations of the induced representation, not site operations, and the
# orbital type cannot declare their sign.  Whether a spinful ``s`` orbital is
# the gerade or the ungerade spinor is then fixed by how the spin is chosen to
# transform under inversion (it does not: ``S(I·R) = S(R)``, the tables'
# convention and ``spin_bonds``'), and the served labels must follow.  Both
# models are group averages of a few seeds, so they obey their group exactly.

def space_group_closure(generators, tol=1e-9):
    """Close ``(R, t)`` generators, in reduced coordinates, under composition modulo the lattice."""
    def key(R, t):
        return (tuple(int(x) for x in np.asarray(R).ravel()),
                tuple(np.round(np.asarray(t) - np.floor(np.asarray(t) + tol), 6)))
    ops = [(np.eye(3, dtype=int), np.zeros(3))]
    seen = {key(*ops[0])}
    frontier = list(ops)
    while frontier:
        new = []
        for R, t in frontier:
            for Rg, tg in generators:
                R2 = np.asarray(Rg, dtype=int) @ np.asarray(R, dtype=int)
                t2 = np.asarray(Rg, dtype=float) @ np.asarray(t, dtype=float) + np.asarray(tg, dtype=float)
                k = key(R2, t2)
                if k not in seen:
                    seen.add(k)
                    op = (R2, t2 - np.floor(t2 + tol))
                    ops.append(op)
                    new.append(op)
        frontier = new
    return ops


#: ``P4/mmm`` (#123): ``4/mmm`` with no translations, 16 operations.
P4MMM_OPS = space_group_closure([
    (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), np.zeros(3)),      # C4(z)
    (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.zeros(3)),     # C2(x)
    (-np.eye(3, dtype=int), np.zeros(3))])                            # inversion
assert len(P4MMM_OPS) == 16

#: ``Pnma`` (#62, standard setting): generated by ``2_1(z)``, ``2_1(y)`` and inversion, 8 operations.
PNMA_OPS = space_group_closure([
    (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), np.array([0.5, 0.0, 0.5])),   # -x+1/2, -y, z+1/2
    (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]), np.array([0.0, 0.5, 0.0])),   # -x, y+1/2, -z
    (-np.eye(3, dtype=int), np.zeros(3))])                                         # -x, -y, -z
assert len(PNMA_OPS) == 8

#: ``4a`` of ``Pnma``: the inversion centres, site group ``-1``, on no mirror plane.
PNMA_4A = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.5, 0.5]])


def _spinful_s_model(lattice, sites, ops, seeds):
    """Spinful ``s`` orbitals on ``sites``, the seeds closed under ``ops`` by :func:`spin_bonds`."""
    from pythtb import Lattice, TBModel
    lattice = np.asarray(lattice, dtype=float)
    sites = np.asarray(sites, dtype=float)
    m = TBModel(Lattice(lattice.tolist(), sites.tolist(), periodic_dirs=[0, 1, 2]), spinful=True)
    m.set_onsite([np.zeros((2, 2), dtype=complex)] * len(sites), mode="set")
    for (i, j, L), T in spin_bonds(sites, ops, lattice, seeds).items():
        if float(np.abs(T).max()) > 1e-12:
            m.set_hop(T, i, j, list(L), mode="add")
    return m


def p4mmm_s_2g(z=0.21, c=1.7, t=-1.0, rashba=0.35, t_pair=-0.6, t_c=-0.25):
    """Spinful ``s`` at ``2g`` ``(0,0,±z)`` of ``P4/mmm``: off the inversion centre and the horizontal mirror.

    Site group ``4mm`` (the vertical mirrors pass through the sites; ``σ_h`` and
    the inversion exchange them).  One in-plane bond carrying a Rashba term on
    the upper site -- the group average puts the opposite Rashba term on the
    lower one -- plus the vertical pair bond and an inter-pair bond along ``c``.
    Measured 2026-09-10: ``realized_group`` reads ``P4/mmm1' #123.340``, 32 of
    32, and the four bands are ``-E1g↑G(2)@1a + -E1u↑G(2)@1a`` (``2g`` is not a
    maximal position; the ``1b`` pair is the same band representation), the
    ``Ē1`` doublet of ``4mm`` induced with both parities.
    """
    sx, sy, sz = PAULI
    seeds = [(0, 0, [1, 0, 0], t * np.eye(2) + 1j * rashba * sy),
             (0, 1, [0, 0, 0], t_pair * np.eye(2)),
             (1, 0, [0, 0, 1], t_c * np.eye(2))]
    return _spinful_s_model(np.diag([1.0, 1.0, c]), [[0, 0, z], [0, 0, -z]], P4MMM_OPS, seeds)


def pnma_s_4a(b=1.3, c=1.6):
    """Spinful ``s`` at ``4a`` of ``Pnma``: inversion centres that lie on no mirror plane.

    Site group ``-1``; every mirror, screw and the remaining inversions carry a
    site to another site, so the whole mirror and inversion content of the
    served representation is transport.  Four generic SOC-carrying seeds, group
    averaged.  Measured 2026-09-10: ``realized_group`` reads ``Pnma1' #62.442``,
    16 of 16, and the eight bands are the tables' row ``-AgAg↑G(8)@4a`` -- the
    *gerade* spinor pair, the statement "the spin does not transform under
    inversion" read off a band structure.
    """
    sx, sy, sz = PAULI
    seeds = [(0, 1, [0, 0, 0], -1.0 * np.eye(2) + 0.3j * sy + 0.15j * sx),
             (0, 2, [0, 0, 0], -0.7 * np.eye(2) + 0.2j * sz),
             (0, 0, [1, 0, 0], -0.4 * np.eye(2)),
             (0, 3, [0, 0, 0], -0.5 * np.eye(2) + 0.1j * sx)]
    return _spinful_s_model(np.diag([1.0, b, c]), PNMA_4A, PNMA_OPS, seeds)


# ---------------------------------------------------------------------------
# spin-1/2 states read in single-valued tables, where the spinor class is trivial
# ---------------------------------------------------------------------------
#
# A spinor Hamiltonian with its spin in the orbital index -- what a wannier90
# spinor import delivers -- asked for its unitary single-valued group: its
# intertwiners form a projective representation with the spinor factor system.
# In -3m that class is trivial (S3 has no Schur multiplier, and the spin does
# not transform under inversion), so a single-valued representation exists on
# the states.  The P4/mmm models above are the other side: two perpendicular
# twofolds, whose spinor matrices anticommute.

#: ``R-3m`` (#166) in its rhombohedral primitive setting, 12 operations:
#: ``3m`` about ``[111]`` and inversion.
R_3M_OPS = space_group_closure([
    (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), np.zeros(3)),       # C3 about [111]
    (np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]), np.zeros(3)),    # C2 about [1-10]
    (-np.eye(3, dtype=int), np.zeros(3))])                            # inversion
assert len(R_3M_OPS) == 12


def r3m_s_6c(u=0.18, alpha=55.0):
    """Spinful ``s`` at ``±(u, u, u)`` of a rhombohedral ``R-3m`` cell: Wyckoff ``6c``, site group ``3m``.

    The threefold and the three mirrors fix each site and inversion exchanges
    the two.  Three spin-orbit-carrying seeds -- the pair bond along the
    threefold axis, a pair bond and a same-site bond along a lattice vector --
    group averaged, so the model obeys ``R-3m1'`` exactly; ``alpha`` (degrees)
    keeps the lattice rhombohedral rather than cubic.  Measured 2026-09-26:
    flattened by ``spinless_presentation`` and asked for its unitary
    single-valued group, ``_group_of`` reads ``R-3m #166``, all 12 operations
    obeyed, each spinor one measured doublet, and the representation is ``E``
    of ``3m`` at both sites -- ``Ē1`` rephased by ``-1`` on the threefold.  The
    canonical norm of ``S3``, the rotoinversion of order six, leaves it six
    roots, and the reference orbital's positivity picks ``e^{i pi/3}``, a cube
    root of unity away from the two that multiply: ``(S3 C2')^2 = E`` then
    misses by ``sqrt(3)``.
    """
    c = np.cos(np.deg2rad(alpha))
    tx, ty, tz = np.sqrt((1 - c) / 2), np.sqrt((1 - c) / 6), np.sqrt((1 + 2 * c) / 3)
    lattice = np.array([[tx, -ty, tz], [0.0, 2 * ty, tz], [-tx, -ty, tz]])
    sites = np.array([[u, u, u], [-u, -u, -u]]) % 1.0
    sx, sy, sz = PAULI
    seeds = [(0, 1, [0, 0, 0], -1.0 * np.eye(2) + 0.3j * sx + 0.1j * sz),
             (0, 1, [1, 0, 0], -0.6 * np.eye(2) + 0.25j * sy),
             (0, 0, [1, 0, 0], -0.3 * np.eye(2) + 0.2j * sx - 0.15j * sy)]
    return _spinful_s_model(lattice, sites, R_3M_OPS, seeds)


# ---------------------------------------------------------------------------
# copies of one site irrep, written in bases that turn from site to site
# ---------------------------------------------------------------------------
#
# A Wyckoff orbit whose sites carry several copies of one site irrep, each site
# free to write them in its own basis: the operation carrying one site onto the
# next then carries each copy onto a combination of the copies there, and only
# the whole isotypic component is a multiplet.  Both families are group
# averages of random seeds with the orbitals carried as the functions they are
# (``_sitechars.shell_operator`` is an operation's matrix on one shell), so
# each obeys its group exactly; both are spinless.

def _shell_matrix(rotation_cart, degree):
    """``f -> f(R^-1 r)`` on the degree-``l`` real harmonics, as the package writes it."""
    from types import SimpleNamespace

    from pythtqc import _sitechars
    stand_in = SimpleNamespace(symmetries=(SimpleNamespace(
        ind=0, rotation_cart=np.asarray(rotation_cart, dtype=float)),))
    return _sitechars.shell_operator(stand_in, 0, int(degree))


def _harmonic(name, degree, frame=None):
    """One real harmonic as a unit vector of its shell, in the frame whose axes are ``frame``'s columns."""
    from pythtqc import _sitechars
    scale = np.sqrt(_sitechars._shell_weights(int(degree)))
    v = _sitechars._shell_vector(name, int(degree), scale)
    return v if frame is None else _shell_matrix(frame, degree) @ v


def _carried_model(lattice, sites, ops, orbitals, seed, cutoff, rotation=None, scale=1.0):
    """Orbitals carried as functions, random hoppings averaged over ``ops`` (Cartesian, about the origin).

    ``sites`` are Cartesian and ``orbitals`` is ``[(site, degree, vector)]``,
    every site's orbitals of one degree spanning what the operations carry onto
    it.  Every pair within ``cutoff`` gets a random real hopping times
    ``scale``, the whole set averaged with the orbitals carried as the
    functions they are, so the model obeys ``ops`` exactly; ``rotation`` then
    writes it in the orbital basis ``q = Vᵀ p``, ``H'(R) = Vᵀ H(R) V`` -- a
    change of basis and nothing else.
    """
    A = np.asarray(lattice, dtype=float)
    Ainv = np.linalg.inv(A)
    sites = [np.asarray(s, dtype=float) for s in sites]
    n = len(orbitals)

    def site_image(R, s):
        r = R @ sites[s]
        for s2 in range(len(sites)):
            L = (r - sites[s2]) @ Ainv
            if np.allclose(L, np.round(L), atol=1e-8):
                return s2, np.round(L).astype(int)
        raise AssertionError("the orbit is not closed under this operation")

    shells = {}

    def orbital_image(g, R, o):
        s, degree, v = orbitals[o]
        s2, L = site_image(R, s)
        if (g, degree) not in shells:
            shells[(g, degree)] = _shell_matrix(R, degree)
        w = shells[(g, degree)] @ v
        return [(k, float(vk @ w), L) for k, (sk, dk, vk) in enumerate(orbitals)
                if sk == s2 and dk == degree and abs(float(vk @ w)) > 1e-12]

    rng = np.random.default_rng(int(seed))
    pos = np.array([sites[s] for s, _d, _v in orbitals])
    seeds = {}
    for i in range(n):
        for j in range(n):
            for L in itertools.product([-1, 0, 1], repeat=3):
                L = np.array(L)
                if np.linalg.norm(pos[j] + L @ A - pos[i]) <= cutoff \
                        and (i, j, tuple(L)) not in seeds:
                    t = float(scale) * float(rng.normal())
                    seeds[(i, j, tuple(L))] = t
                    seeds[(j, i, tuple(-L))] = t
    hops = {}
    for (i, j, L), t in seeds.items():
        for g, R in enumerate(ops):
            RL = np.round((R @ (np.asarray(L, dtype=float) @ A)) @ Ainv).astype(int)
            for i2, wi, Li in orbital_image(g, R, i):
                for j2, wj, Lj in orbital_image(g, R, j):
                    key = (i2, j2, tuple(int(x) for x in Lj + RL - Li))
                    hops[key] = hops.get(key, 0.0) + wi * wj * t / len(ops)
    V = np.eye(n) if rotation is None else np.asarray(rotation, dtype=float)
    by_L = {}
    for (i, j, L), t in hops.items():
        by_L.setdefault(L, np.zeros((n, n)))[i, j] += t
    m = TBModel(Lattice(A.tolist(), [list(p @ Ainv) for p in pos], periodic_dirs=[0, 1, 2]),
                spinful=False)
    entered = set()
    for L, H in by_L.items():
        Hp = V.T @ H @ V
        for i in range(n):
            for j in range(n):
                t = float(Hp[i, j])
                if abs(t) < 1e-12 or (j, i, tuple(-np.array(L))) in entered:
                    continue
                entered.add((i, j, L))
                if i == j and L == (0, 0, 0):
                    m.set_onsite(t, i)
                else:
                    m.set_hop(t, i, j, list(L))
    return m


#: ``P2/m`` with its twofold along ``y`` on a genuinely monoclinic cell.
P2M_LATTICE = np.array([[1.0, 0.0, 0.0], [0.0, 1.3, 0.0], [0.31, 0.0, 1.12]])
#: The four operations of ``2/m`` (twofold along ``y``), Cartesian.
P2M_OPS = [np.diag(d).astype(float) for d in ([1, 1, 1], [-1, 1, -1], [-1, -1, -1], [1, -1, 1])]


def p2m_mirror_pair(angle=0.0, seed=3, functions=("px", "pz"), listed_at_b=None,
                    phases=None):
    """Two ``p`` orbitals at both sites of a ``2m`` orbit of ``P2/m``, site ``B``'s pair turned by ``angle``.

    The orbit is ``A = (0.21, 0, 0.33)`` and ``B = -A``, carried into one
    another by the twofold and by the inversion; each site's own group is the
    mirror ``y -> -y`` alone (``m``), under which ``p_x`` and ``p_z`` are both
    even -- two copies of ``A'`` -- and ``p_y`` is odd, ``A''``.  Every pair of
    orbitals within 1.45 gets a random real hopping, averaged over the four
    operations of ``2/m``, so the hoppings obey ``P2/m1'`` (#10.43).  With the
    default ``p_x`` and ``p_z`` each orbital is a multiplet of its own; a
    nonzero ``angle`` turns site ``B``'s pair inside itself -- ``q = Vᵀ p`` with
    ``V = [[cos, -sin], [sin, cos]]`` -- a change of basis among copies of one
    irrep, after which no orbital subset carries the twofold and the copies at
    each site are one multiplet.  ``listed_at_b`` lists site ``B``'s functions
    in another order than site ``A``'s, which changes nothing but the order,
    and ``phases`` writes orbitals as complex multiples of themselves, which
    changes nothing but the basis functions' phases.

    Parameters
    ----------
    angle : float, optional
        The turn of site ``B``'s pair, in radians.  Default ``0.0``.
    seed : int, optional
        Seed of the random hoppings.  Default ``3``.
    functions : tuple of str, optional
        The two functions at site ``A``.  Default ``("px", "pz")``.
    listed_at_b : tuple of str, optional
        The same two functions in site ``B``'s own order.  Default: ``functions``.
    phases : dict, optional
        ``{orbital: phase}``: each orbital listed is written as that phase
        times the function it was, ``H'(R) = U† H(R) U`` with ``U`` diagonal
        -- ``{2: 1j}`` makes site ``B``'s first orbital ``i`` times the image
        of site ``A``'s first.  Default: none.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 4 orbitals / 4 bands.
    """
    a = np.array([0.21, 0.0, 0.33]) @ P2M_LATTICE
    at_b = tuple(functions) if listed_at_b is None else tuple(listed_at_b)
    orbitals = ([(0, 1, _harmonic(name, 1)) for name in functions]
                + [(1, 1, _harmonic(name, 1)) for name in at_b])
    c, s = np.cos(angle), np.sin(angle)
    V = np.eye(4)
    V[2:, 2:] = [[c, -s], [s, c]]
    model = _carried_model(P2M_LATTICE, [a, -a], P2M_OPS, orbitals, seed, 1.45, rotation=V)
    return model if not phases else _rephased(model, phases)


def _rephased(model, phases):
    """``model`` with orbital ``a`` written as ``phases[a]`` times itself: ``H'(R) = U† H(R) U``."""
    u = np.ones(int(model.norb), dtype=complex)
    for a, phase in dict(phases).items():
        u[int(a)] = complex(phase)
    dim_r = int(model.lattice.dim_r)
    out = TBModel(Lattice(np.asarray(model.lat_vecs, dtype=float),
                          np.asarray(model.orb_vecs, dtype=float).tolist(),
                          periodic_dirs=list(model.periodic_dirs)),
                  spinful=(int(model.nspin) == 2))
    out.set_onsite(np.asarray(model.onsite), mode="set")
    for h in model.hoppings:
        i, j = int(h["from_orbital"]), int(h["to_orbital"])
        amp = np.conj(u[i]) * u[j] * np.asarray(h["amplitude"], dtype=complex)
        out.set_hop(amp if amp.ndim else complex(amp), i, j,
                    list(h.get("lattice_vector", [0] * dim_r)), mode="add")
    return out


#: The body-centred cubic cell of ``Im-3m`` (#229), primitive vectors as rows.
IM3M_LATTICE = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
#: The four body diagonals the ``8c`` sites of the primitive cell sit on.
IM3M_8C_AXES = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=float)
#: Each shell's functions, and its isotypic components under ``-3m`` with ``z``
#: along the threefold: ``p = A2u + Eu`` and ``d = A1g + 2 Eg``.
IM3M_8C_SHELLS = {
    "p": (1, ("px", "py", "pz"), (("px", "py"), ("pz",))),
    "d": (2, ("dxy", "dyz", "dzx", "dx2-y2", "dz2"),
          (("dxy", "dyz", "dzx", "dx2-y2"), ("dz2",))),
}


def im3m_8c(shells=("p", "d"), rotated=False, centre=False, seed=5, rotation_seed=11):
    """Real harmonics at the ``8c`` sites of ``Im-3m``, each written in its site's own frame.

    ``8c`` is an orbit of eight sites in the conventional cell, four in the
    primitive one, ``(1/4, 1/4, 1/4)`` and its images; each site's group is
    ``-3m`` with its threefold along the site's own body diagonal.  At each
    site the functions of ``shells`` are written in a frame with ``z`` along
    that diagonal and ``x`` along a twofold axis, where ``p`` is ``A2u + Eu``
    and ``d`` is ``A1g + 2 Eg`` -- two different two-dimensional irreps, and two
    copies of one.  Every pair of orbitals within 0.75 gets a random real
    hopping, averaged over the 48 operations of ``m-3m``, so the hoppings obey
    ``Im-3m1'``.  ``rotated`` writes each site's orbitals in a basis turned by a
    random rotation inside each isotypic component, a different one at every
    site: a change of basis and nothing else, after which the two copies of
    ``Eg`` are written in no basis that separates them.

    The ``8c`` sites alone make a simple cubic point set of half the cell's
    edge, whose own lattice is finer than the body-centred one, so the crystal
    is found by a descent from ``Pm-3m1'``.  ``centre`` puts an ``s`` orbital
    at ``2a``, the origin, as orbital 0: the point set then has the
    body-centred lattice and nothing finer, and the ``s`` is the reference the
    other orbitals are read against.

    Parameters
    ----------
    shells : tuple of str, optional
        ``"p"`` and/or ``"d"``, in the order they are listed at each site.
        Default ``("p", "d")``.
    rotated : bool, optional
        Turn every site's basis inside each isotypic component.  Default
        ``False``.
    centre : bool, optional
        Put an ``s`` orbital at ``2a`` first.  Default ``False``.
    seed : int, optional
        Seed of the random hoppings.  Default ``5``.
    rotation_seed : int, optional
        Seed of the random rotations.  Default ``11``.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, four sites times the functions of ``shells``, and
        the ``s`` at ``2a`` where ``centre``.
    """
    ops = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[j, i] = 1.0
        ops.extend(P @ np.diag(signs) for signs in itertools.product([1, -1], repeat=3))
    sites = [0.25 * axis for axis in IM3M_8C_AXES]
    orbitals, names = [], []
    if centre:
        sites = [np.zeros(3)] + sites
        orbitals.append((0, 0, _harmonic("s", 0)))
        names.append((0, "s"))
    first = 1 if centre else 0
    for s, axis in enumerate(IM3M_8C_AXES):
        z = axis / np.linalg.norm(axis)
        x = np.array([axis[0], -axis[1], 0.0]) / np.sqrt(2.0)
        frame = np.stack([x, np.cross(z, x), z], axis=1)
        for shell in shells:
            degree, functions, _components = IM3M_8C_SHELLS[shell]
            for name in functions:
                orbitals.append((first + s, degree, _harmonic(name, degree, frame)))
                names.append((first + s, name))
    V = np.eye(len(orbitals))
    if rotated:
        rng = np.random.default_rng(int(rotation_seed))
        for s in range(first, first + len(IM3M_8C_AXES)):
            for shell in shells:
                for component in IM3M_8C_SHELLS[shell][2]:
                    idx = [names.index((s, name)) for name in component]
                    Q, R = np.linalg.qr(rng.normal(size=(len(idx), len(idx))))
                    Q = Q @ np.diag(np.sign(np.diag(R)))
                    if np.linalg.det(Q) < 0:
                        Q[:, 0] = -Q[:, 0]
                    V[np.ix_(idx, idx)] = Q
    return _carried_model(IM3M_LATTICE, sites, ops, orbitals, seed, 0.75, rotation=V)


# ---------------------------------------------------------------------------
# a shell written in a basis turned inside its site
# ---------------------------------------------------------------------------
#
# A Wannier basis is any unitary mixture of the functions at a site, so which
# orbital is which function is a gauge: a ``p`` shell whose orbitals are turned
# inside their site by a rotation of no symmetry holds the same site irreps as
# the aligned shell, while no subset of the turned orbitals carries the site
# group.  The spinful model carries the spin by the operations' own SU(2) lift.

def _carried_spinful_model(lattice, sites, ops, orbitals, seed, cutoff, rotation=None,
                           scale=1.0):
    """:func:`_carried_model` with the electron's spin: random spin-orbit blocks, averaged.

    Every pair of orbitals within ``cutoff``, an orbital with itself on site
    included, gets a random complex ``2x2`` block times ``scale``; each
    operation carries the orbitals as the functions they are and the spin by
    its own SU(2) lift (an improper one by that of its proper part, as the
    tables lift it), so the average over ``ops`` obeys them exactly, and the
    average with its time-reversed image obeys time reversal as well.
    ``rotation`` then writes the orbital factor in the basis ``q = Vᵀ p`` at
    each spin, ``H'(R) = (V ⊗ 1)ᵀ H(R) (V ⊗ 1)`` -- a change of basis and
    nothing else.
    """
    A = np.asarray(lattice, dtype=float)
    Ainv = np.linalg.inv(A)
    sites = [np.asarray(s, dtype=float) for s in sites]
    n = len(orbitals)

    def site_image(R, s):
        r = R @ sites[s]
        for s2 in range(len(sites)):
            L = (r - sites[s2]) @ Ainv
            if np.allclose(L, np.round(L), atol=1e-8):
                return s2, np.round(L).astype(int)
        raise AssertionError("the orbit is not closed under this operation")

    shells = {}

    def orbital_image(g, R, o):
        s, degree, v = orbitals[o]
        s2, L = site_image(R, s)
        if (g, degree) not in shells:
            shells[(g, degree)] = _shell_matrix(R, degree)
        w = shells[(g, degree)] @ v
        return [(k, float(vk @ w), L) for k, (sk, dk, vk) in enumerate(orbitals)
                if sk == s2 and dk == degree and abs(float(vk @ w)) > 1e-12]

    rng = np.random.default_rng(int(seed))
    pos = np.array([sites[s] for s, _d, _v in orbitals])
    seeds = {}
    for i in range(n):
        for j in range(n):
            for L in itertools.product([-1, 0, 1], repeat=3):
                L = np.array(L)
                if np.linalg.norm(pos[j] + L @ A - pos[i]) <= cutoff \
                        and (i, j, tuple(L)) not in seeds:
                    T = scale * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
                    if i == j and not L.any():
                        T = T + T.conj().T
                    seeds[(i, j, tuple(L))] = T
                    seeds[(j, i, tuple(-L))] = T.conj().T
    hops = {}
    for (i, j, L), T in seeds.items():
        for g, R in enumerate(ops):
            S = spinor_of(np.sign(float(np.linalg.det(R))) * R)
            T2 = S @ T @ S.conj().T
            RL = np.round((R @ (np.asarray(L, dtype=float) @ A)) @ Ainv).astype(int)
            for i2, wi, Li in orbital_image(g, R, i):
                for j2, wj, Lj in orbital_image(g, R, j):
                    key = (i2, j2, tuple(int(x) for x in Lj + RL - Li))
                    hops[key] = hops.get(key, 0.0) + wi * wj * T2 / len(ops)
    by_L = {}
    for (i, j, L), T in hops.items():
        by_L.setdefault(L, np.zeros((2 * n, 2 * n), dtype=complex))[
            2 * i:2 * i + 2, 2 * j:2 * j + 2] += T
    reverse = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))      # 1 (x) i sigma_y
    V = np.kron(np.eye(n) if rotation is None else np.asarray(rotation, dtype=float), np.eye(2))
    blocks = {L: V.T @ (0.5 * (H + reverse @ H.conj() @ reverse.T)) @ V for L, H in by_L.items()}
    m = TBModel(Lattice(A.tolist(), [list(p @ Ainv) for p in pos], periodic_dirs=[0, 1, 2]),
                spinful=True)
    zero = (0, 0, 0)
    # hermitian to the last bit, so that the spin can be carried into the
    # orbital index and back without a rounding's difference
    H0 = blocks.get(zero, np.zeros((2 * n, 2 * n), dtype=complex))
    blocks[zero] = H0 = 0.5 * (H0 + H0.conj().T)
    m.set_onsite([H0[2 * i:2 * i + 2, 2 * i:2 * i + 2] for i in range(n)], mode="set")
    entered = set()
    for L, H in blocks.items():
        for i in range(n):
            for j in range(n):
                if (L == zero and i >= j) or (j, i, tuple(-np.array(L))) in entered:
                    continue
                block = H[2 * i:2 * i + 2, 2 * j:2 * j + 2]
                if float(np.abs(block).max()) < 1e-12:
                    continue
                entered.add((i, j, L))
                m.set_hop(block, i, j, list(L), mode="add")
    return m


#: A rhombohedral cell with its threefold along ``z`` and a twofold along ``x``.
R3M_LATTICE = np.array([[0.5, -0.5 / np.sqrt(3.0), 2.6 / 3.0],
                        [0.0, 1.0 / np.sqrt(3.0), 2.6 / 3.0],
                        [-0.5, -0.5 / np.sqrt(3.0), 2.6 / 3.0]])
_C3Z = np.array([[-0.5, -np.sqrt(3.0) / 2.0, 0.0],
                 [np.sqrt(3.0) / 2.0, -0.5, 0.0],
                 [0.0, 0.0, 1.0]])
#: The six operations of ``3m`` about the origin, Cartesian: the threefold along
#: ``z`` and the three mirrors that contain it, ``x -> -x`` among them.
R3M_OPS = [np.linalg.matrix_power(_C3Z, k) @ mirror
           for k in range(3) for mirror in (np.eye(3), np.diag([-1.0, 1.0, 1.0]))]
#: An orbital-less species up the threefold axis: it makes the cell polar --
#: ``R3m``, where a lone site on this lattice allows ``R-3m`` -- before a hopping
#: is read.
R3M_SHADOW = (([0.31, 0.31, 0.31], "X"),)
#: The direction :func:`r3m_p_shell` turns its orbitals about, one of no symmetry.
R3M_TURN_AXIS = np.array([0.31, 0.52, 0.80]) / np.linalg.norm([0.31, 0.52, 0.80])
#: Where :func:`r3m_p_shell`'s ``pair`` puts its two sites: ``±u (a1 + a2 + a3)``.
R3M_PAIR_U = 0.21


def r3m_p_shell(angle=0.0, spinful=False, seed=3, pair=False):
    """A ``p`` shell at the origin of ``R3m``, its orbitals turned inside the site by ``angle``.

    The three ``p`` orbitals in the order a ``wannier90`` projection lists them,
    ``p_z``, ``p_x``, ``p_y``, at the origin of :data:`R3M_LATTICE`, with a
    random hopping between every two orbitals within 1.25 averaged over the six
    operations of ``3m`` (:data:`R3M_OPS`), the orbitals carried as the
    functions they are.  ``spinful`` gives each orbital its spin and every
    hopping a random spin-orbit block (:func:`_carried_spinful_model`), at half
    the scale so that ``max|H|`` keeps the covariance gate at its floor.  The
    site group is ``3m``, under which the shell is ``A1 + E`` -- ``p_z`` and the
    ``(p_x, p_y)`` pair -- and with the spin ``2x-E1 + -1E2E``.  A lone site on
    this lattice allows ``R-3m``; the hoppings obey ``R3m1'`` (#160.66), and
    :data:`R3M_SHADOW` states the polar axis so that the cell is ``R3m`` before
    a hopping is read.

    ``angle`` turns every orbital through that angle about
    :data:`R3M_TURN_AXIS`: orbital ``i`` becomes the ``p`` orbital along the
    turned image of its own axis, the spin untouched.  That is a change of
    orbital basis and nothing else -- the group, the site irreps and the bands
    cannot move -- and away from the angles of symmetry no subset of the turned
    orbitals carries the site group, so the site's own operations bind the
    whole shell into one multiplet.

    ``pair`` puts the shell at the two sites ``±u (a1 + a2 + a3)`` instead,
    ``u`` = :data:`R3M_PAIR_U`, with the hoppings averaged over the twelve
    operations of ``-3m``: the inversion carries one site onto the other, the
    hoppings obey ``R-3m1'`` (#166.98) with no shadow, and the two sites are
    the ``6c`` orbit, each of site group ``3m`` -- a position that is not
    maximal, so the tables list no site irrep there.  Both sites are turned
    alike.  With the spin its blocks are at a fifth of the scale, so that the
    covariance gate of twice the states and twice the operations stays at its
    floor as the single shell's does.

    Parameters
    ----------
    angle : float, optional
        The turn, in radians.  Default ``0.0``: the aligned shell.
    spinful : bool, optional
        Give the orbitals their spin, with spin-orbit coupling.  Default ``False``.
    seed : int, optional
        Seed of the random hoppings.  Default ``3``.
    pair : bool, optional
        The shell at both sites of ``6c`` of ``R-3m``.  Default ``False``.

    Returns
    -------
    pythtb.TBModel
        3D model, 3 orbitals / 3 bands, or 6 bands with the spin; twice that
        with ``pair``.
    """
    order = ("pz", "px", "py")
    V = _turned_p_shell(angle, order)
    sites, ops = [np.zeros(3)], R3M_OPS
    if pair:
        centre = R3M_PAIR_U * R3M_LATTICE.sum(axis=0)
        sites, ops = [centre, -centre], list(R3M_OPS) + [-R for R in R3M_OPS]
        V = np.kron(np.eye(2), V)
    orbitals = [(s, 1, _harmonic(name, 1)) for s in range(len(sites)) for name in order]
    if spinful:
        return _carried_spinful_model(R3M_LATTICE, sites, ops, orbitals, seed,
                                      1.25, rotation=V, scale=0.2 if pair else 0.5)
    return _carried_model(R3M_LATTICE, sites, ops, orbitals, seed, 1.25, rotation=V)


def _turned_p_shell(angle, order):
    """``V``: each ``p`` orbital of ``order`` turned through ``angle`` about :data:`R3M_TURN_AXIS`."""
    axis = {"px": 0, "py": 1, "pz": 2}
    k = R3M_TURN_AXIS
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    turn = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K
    return np.array([[turn[axis[order[j]], axis[order[i]]] for i in range(3)]
                     for j in range(3)])


#: A tetragonal cell for :func:`p4mmm_p_shell`, its fourfold axis along ``z``.
P4MMM_LATTICE = np.diag([1.0, 1.0, 1.3])


def p4mmm_p_shell(angle=0.0, seed=3):
    """A ``p`` shell at the origin of ``P4/mmm``, listed ``p_z`` first, turned inside the site by ``angle``.

    The three ``p`` orbitals in the order a ``wannier90`` projection lists them,
    ``p_z``, ``p_x``, ``p_y``, at the origin of :data:`P4MMM_LATTICE`, with a
    random hopping between every two orbitals within 1.7 averaged over the
    sixteen operations of ``4/mmm`` (:data:`P4MMM_OPS`), the orbitals carried as
    the functions they are, at a quarter of the scale so that ``max|H|`` keeps
    the covariance gate at its floor.  The site group is ``4/mmm``, under which the shell
    is ``A2u + Eu`` -- ``p_z`` and the ``(p_x, p_y)`` pair -- and the hoppings
    obey ``P4/mmm1'`` (#123.340).  The site is centrosymmetric, so the shell's
    one orbital ``p_z`` taken on its own would be the ``s`` labelling's twin;
    the whole site's content is one ``p`` shell whichever orbital is listed first.

    ``angle`` turns every orbital through that angle about
    :data:`R3M_TURN_AXIS`, a direction of no symmetry: a change of orbital basis
    and nothing else, after which no subset of the turned orbitals carries the
    site group and the site's own operations bind the whole shell into one
    multiplet.

    Parameters
    ----------
    angle : float, optional
        The turn, in radians.  Default ``0.0``: the aligned shell.
    seed : int, optional
        Seed of the random hoppings.  Default ``3``.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 3 orbitals / 3 bands.
    """
    order = ("pz", "px", "py")
    orbitals = [(0, 1, _harmonic(name, 1)) for name in order]
    ops = [np.asarray(R, dtype=float) for R, _t in P4MMM_OPS]
    return _carried_model(P4MMM_LATTICE, [np.zeros(3)], ops, orbitals, seed, 1.7,
                          rotation=_turned_p_shell(angle, order), scale=0.25)


# ---------------------------------------------------------------------------
# a (p_x, p_y) pair at every site of an orbit whose site group is 3
# ---------------------------------------------------------------------------
#
# The pair is the doublet 1E2E of 3 at every site, and the operations that
# exchange the sites can carry it onto the next site's pair with either sign:
# two labellings of the crystal that give every site the same site irrep and
# the lower bands different labels.  Random hoppings averaged over the point
# group, the pairs carried as (p_x, p_y), so each model obeys its group exactly.

#: A hexagonal cell, ``c/a = 1.2``, lattice vectors as rows.
P321_LATTICE = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.2]])
#: The two sites ``(0, 0, ±0.2)``, reduced: one orbit of ``P321``, site group ``3``.
P321_SITES = (np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, -0.2]))


def _turn2(angle):
    """The rotation of a ``(p_x, p_y)`` pair through ``angle``."""
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def _pair_orbit(lattice, sites, point_ops, span, cutoff, seed, turns, soc):
    """``(p_x, p_y)`` at each of ``sites``, random hoppings averaged over ``point_ops``.

    ``lattice`` has the lattice vectors as rows and ``sites`` are reduced;
    ``point_ops`` are Cartesian, about the origin, and carry the pair by their
    in-plane block.  Every pair of sites within ``cutoff`` (cells up to
    ``span`` away) gets a random real 2x2 block, and the set is averaged over
    the operations with its Hermitian partner.  ``turns`` writes each site's
    pair in a basis turned inside itself -- a change of basis and nothing else
    -- and ``soc`` gives every orbital its spin, with an on-site
    ``soc L_z S_z`` on each pair.
    """
    L = np.asarray(lattice, dtype=float)
    Linv = np.linalg.inv(L)
    F = [np.asarray(s, dtype=float) for s in sites]
    dim = L.shape[0]

    def image(R, s, cell):
        f = ((F[s] + cell) @ L @ R.T) @ Linv
        for t, site in enumerate(F):
            n = f - site
            if np.allclose(n, np.rint(n), atol=1e-9):
                return t, np.rint(n).astype(int)
        raise AssertionError("the orbit is not closed under this operation")

    rng = np.random.default_rng(int(seed))
    keys = [(i, j) + cell for i in range(len(F)) for j in range(len(F))
            for cell in itertools.product(range(-span, span + 1), repeat=dim)
            if 1e-9 < np.linalg.norm((F[j] + np.array(cell) - F[i]) @ L) <= cutoff]
    raw = {k: rng.normal(size=(2, 2)) for k in keys}
    hop = {k: np.zeros((2, 2)) for k in keys}
    for (i, j, *cell), h in raw.items():
        for R in point_ops:
            r2 = np.asarray(R, dtype=float)[:2, :2]
            i2, ci = image(R, i, np.zeros(dim, dtype=int))
            j2, cj = image(R, j, np.array(cell))
            carried = r2 @ h @ r2.T
            hop[(i2, j2, *(cj - ci))] += carried
            hop[(j2, i2, *(ci - cj))] += carried.T
    V = [_turn2(t) for t in turns]
    lat = Lattice(L.tolist(), [list(F[s]) for s in range(len(F)) for _ in range(2)],
                  periodic_dirs=list(range(dim)))
    m = TBModel(lat, spinful=soc is not None)
    for (i, j, *cell), h in hop.items():
        if i > j or (i == j and tuple(cell) < tuple(-c for c in cell)):
            continue                                   # pythtb adds the conjugate itself
        t = V[i].T @ (h / (2 * len(point_ops))) @ V[j]
        for p in range(2):
            for q in range(2):
                if abs(t[p, q]) > 1e-13:
                    amp = t[p, q] if soc is None else t[p, q] * np.eye(2)
                    m.set_hop(amp, 2 * i + p, 2 * j + q, list(cell), mode="add")
    if soc is not None:
        # (L_z)_{xy} = -i on a (p_x, p_y) pair, and no turn of the pair moves it
        for a in range(0, 2 * len(F), 2):
            m.set_hop(float(soc) * (-1j) * np.diag([1.0, -1.0]), a, a + 1, [0] * dim,
                      mode="add")
    return m


def p321_pxpy_pair(turn_a=0.0, turn_b=0.0, soc=None, seed=7):
    """``(p_x, p_y)`` at the two sites ``(0, 0, ±0.2)`` of a hexagonal ``P321`` cell.

    Site group ``3``, the two sites exchanged only by the twofolds of ``32``:
    a random hopping block between every two sites within 1.25 averaged over
    the six operations of ``32``, so the hoppings obey ``P3211'`` (#150.26).
    Every site's pair is ``1E2E`` of ``3``, and the twofolds can carry one
    site's pair onto the other's with either sign -- two labellings with one
    site irrep that label the lower two bands differently.  ``turn_a`` and
    ``turn_b`` write each site's pair in a basis turned inside itself; ``soc``
    gives the orbitals their spin and each pair an on-site ``soc L_z S_z``,
    which ``32`` and time reversal keep.

    Returns
    -------
    pythtb.TBModel
        3D model, 4 orbitals / 4 bands, or 8 bands with the spin.
    """
    a = 2.0 * np.pi / 3.0
    c3 = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    twofolds = [2.0 * np.outer(u, u) - np.eye(3)
                for u in (np.array([np.cos(np.radians(d)), np.sin(np.radians(d)), 0.0])
                          for d in (0.0, 60.0, 120.0))]
    ops = [np.eye(3), c3, c3 @ c3] + twofolds
    return _pair_orbit(P321_LATTICE, P321_SITES, ops, 2, 1.25, seed, (turn_a, turn_b), soc)


def p31m_pxpy_honeycomb(turn_a=0.0, turn_b=0.0, seed=7):
    """The ``(p_x, p_y)`` honeycomb with site group ``3``: ``P31m`` on a substrate.

    The sites of :data:`PXPY_SITES` on :data:`PXPY_LATTICE`, a random hopping
    block between every two sites within 1.2 averaged over the threefold and
    the three mirrors through the hexagon centre that exchange the
    sublattices.  On a substrate the crystal is ``P31m1'`` (#157.54) and
    free-standing ``P-62m1'`` (#189.222); every pair is the doublet of its
    site, and the mirrors can carry one sublattice's pair onto the other's
    with either sign.  ``turn_a`` and ``turn_b`` write each sublattice's pair
    in a basis turned inside itself.

    Returns
    -------
    pythtb.TBModel
        Spinless 2D model, 4 orbitals / 4 bands.
    """
    def mirror(deg):
        a = np.radians(2.0 * deg)
        return np.array([[np.cos(a), np.sin(a)], [np.sin(a), -np.cos(a)]])

    ops = [_turn2(0.0), _turn2(2.0 * np.pi / 3.0), _turn2(4.0 * np.pi / 3.0),
           mirror(30.0), mirror(90.0), mirror(150.0)]
    sites = (np.asarray(PXPY_SITES["A"], dtype=float), np.asarray(PXPY_SITES["B"], dtype=float))
    return _pair_orbit(PXPY_LATTICE, sites, ops, 3, 1.2, seed, (turn_a, turn_b), None)


# ---------------------------------------------------------------------------
# a spinful p shell at the 3c sites of P-43m
# ---------------------------------------------------------------------------
#
# The 3c orbit of P-43m is three sites of site group -42m, each with its -4
# axis along its own cube axis, under which a p shell is B2 (the p along the
# axis) + E (the other two).  With one site's shell written in a turned basis,
# the operations that exchange the sites carry a turned shell onto aligned ones.

#: The three ``3c`` sites of a simple cubic cell; site ``s`` has its ``-4`` axis along ``s``.
P43M_3C = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
#: An orbital-less species on the ``4e`` tetrahedron ``(x, x, x)``: it makes the
#: cell ``P-43m`` before a hopping is read, where the ``3c`` sites alone allow
#: ``Pm-3m``.
P43M_SHADOW = tuple(([x, y, z], "X") for x, y, z in
                    ((0.19, 0.19, 0.19), (0.19, -0.19, -0.19),
                     (-0.19, 0.19, -0.19), (-0.19, -0.19, 0.19)))
#: The seed of the rotation of no symmetry that ``p43m_3c_p_shell(turned=True)``
#: writes site 0's shell in.
P43M_TURN_SEED = 11


def _td_operations():
    """The 24 operations of ``-43m``: the signed permutations with an even number of minus signs."""
    out = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3), dtype=int)
        for i, j in enumerate(perm):
            P[j, i] = 1
        for signs in itertools.product([1, -1], repeat=3):
            if list(signs).count(-1) % 2 == 0:
                out.append(P @ np.diag(signs))
    return out


def p43m_3c_p_shell(turned=False, seed=5, on_site=3.0, hopping=0.2, scale=0.08,
                    cutoff=1.01, decoupled=False, turned_site=0):
    """A ``p`` shell with spin at each ``3c`` site of ``P-43m``, one written turned on request.

    Every pair of the shells' states within ``cutoff`` -- on site, the
    ``3c``-``3c`` neighbours at ``a/sqrt(2)``, and the same site one lattice
    vector away -- gets a random complex block, ``on_site`` on site and
    ``hopping`` between sites; the blocks are averaged over the 24 operations
    of ``-43m``, the ``p`` orbitals carried as the vectors they are and the spin
    by the SU(2) lift of the operation's proper part, and over time reversal,
    so the Hamiltonian obeys ``P-43m1'`` (#215.71) exactly; ``scale`` keeps
    ``max|H|`` near 0.1 and the covariance gate at its floor.  The three ``3c``
    sites alone allow ``Pm-3m``, and :data:`P43M_SHADOW` states ``-43m`` in the
    geometry.  The site group is ``-42m``, under which the shell is ``B2``
    (the ``p`` along the site's own ``-4`` axis) ``+ E`` (the other two), and
    with the spin ``-E2 + (-E1 + -E2)``.

    ``decoupled`` sets to zero every block between the ``p`` along a site's own
    axis and the other two, on site and between sites, which the operations
    carry onto itself: the axes then couple only to the axes and the pairs only
    to the pairs, two orbits the hoppings leave apart, each with a
    one-dimensional character of its own that they do not fix.

    ``turned`` writes the shell of site ``turned_site`` in the basis
    ``q_a = sum_b Q[b, a] p_b``, ``Q`` a rotation of no symmetry drawn from
    :data:`P43M_TURN_SEED`, and the other two sites as they are: a change of
    basis and nothing else.

    Parameters
    ----------
    turned : bool, optional
        Write one site's shell in the turned basis.  Default ``False``.
    seed : int, optional
        Seed of the random blocks.  Default ``5``.
    on_site, hopping : float, optional
        Scales of the random on-site and inter-site blocks before the average.
    scale : float, optional
        Overall scale of the averaged blocks.  Default ``0.08``.
    cutoff : float, optional
        Largest distance between two coupled sites, in lattice constants.
    decoupled : bool, optional
        Uncouple each site's axis from its pair everywhere.  Default ``False``.
    turned_site : int, optional
        The site whose shell ``turned`` writes turned.  Default ``0``.

    Returns
    -------
    pythtb.TBModel
        3D model, 9 orbitals / 18 bands.
    """
    ops = _td_operations()
    sites = P43M_3C
    rng = np.random.default_rng(int(seed))
    raw = {}
    shifts = [np.array(v) for v in itertools.product([-1, 0, 1], repeat=3)]
    for s1 in range(3):
        for s2 in range(3):
            for L in shifts:
                if np.linalg.norm(sites[s2] + L - sites[s1]) > cutoff:
                    continue
                key = (s1, s2, tuple(int(x) for x in L))
                rkey = (s2, s1, tuple(int(-x) for x in L))
                if key in raw:
                    continue
                sc = on_site if (s1 == s2 and not any(L)) else hopping
                M = sc * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
                if key == rkey:
                    M = 0.5 * (M + M.conj().T)
                raw[key] = M
                raw[rkey] = M.conj().T

    def image(R, s):
        r = R @ sites[s]
        for s2 in range(3):
            L = r - sites[s2]
            if np.allclose(L, np.round(L), atol=1e-9):
                return s2, np.round(L).astype(int)
        raise AssertionError("the 3c orbit is not closed under this operation")

    acc = {}
    for R in ops:
        # the p shell carried as a vector, its spin by the lift of the proper part
        D = np.kron(R.astype(float), spinor_of(np.sign(np.linalg.det(R)) * R))
        img = [image(R, s) for s in range(3)]
        for (s1, s2, L), M in raw.items():
            t1, a1 = img[s1]
            t2, a2 = img[s2]
            L2 = tuple(int(x) for x in (R @ np.asarray(L) + a2 - a1))
            acc[(t1, t2, L2)] = acc.get((t1, t2, L2), 0) + D @ M @ D.conj().T
    T = np.kron(np.eye(3), PAULI[1])
    blocks = {}
    for k, M in acc.items():
        M = M / len(ops)
        blocks[k] = scale * 0.5 * (M + T @ M.conj() @ T.conj().T)
    if decoupled:
        # site s's own axis is its orbital s: nothing between it and the pair
        for (s1, s2, _L), M in blocks.items():
            for a in range(3):
                for b in range(3):
                    if (a == s1) != (b == s2):
                        M[2 * a:2 * a + 2, 2 * b:2 * b + 2] = 0.0
    if turned:
        A = np.random.default_rng(P43M_TURN_SEED).normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        W = [np.eye(6)] * 3
        W[int(turned_site)] = np.kron(Q, np.eye(2))
        blocks = {(s1, s2, L): W[s1].conj().T @ M @ W[s2] for (s1, s2, L), M in blocks.items()}
    m = TBModel(Lattice(np.eye(3).tolist(), [list(sites[s]) for s in range(3) for _ in range(3)],
                        periodic_dirs=[0, 1, 2]), spinful=True)
    done = set()
    for (s1, s2, L), M in sorted(blocks.items()):
        if (s2, s1, tuple(-x for x in L)) in done:
            continue
        done.add((s1, s2, L))
        for a in range(3):
            for b in range(3):
                i, j = 3 * s1 + a, 3 * s2 + b
                blk = M[2 * a:2 * a + 2, 2 * b:2 * b + 2]
                if float(np.abs(blk).max()) < 1e-13:
                    continue
                if s1 == s2 and not any(L):
                    if a == b:
                        m.set_onsite(0.5 * (blk + blk.conj().T), i, mode="add")
                        continue
                    if b < a:
                        continue
                m.set_hop(blk, i, j, list(L), mode="add")
    return m


# ---------------------------------------------------------------------------
# two identical copies of an s orbital at the general position of P-1
# ---------------------------------------------------------------------------

#: A triclinic cell, lattice vectors as rows, of no symmetry beyond the inversion.
P1BAR_LATTICE = np.array([[1.0, 0.0, 0.0], [0.23, 1.17, 0.0], [0.31, 0.19, 1.33]])
#: ``r0`` of the general position ``±r0`` of ``P-1``, reduced: site group ``1``.
P1BAR_SITE = np.array([0.13, 0.27, 0.31])


def p1bar_s_copies(seed=3, cutoff=1.6):
    """Two identical copies of an ``s`` orbital at the general position ``±r0`` of ``P-1``.

    A random real hopping between every two orbitals of one copy within
    ``cutoff``, averaged with its image under the inversion, so that the copy
    obeys ``P-11'`` (#2.5); the second copy repeats the first, and nothing
    couples the two.  The Hamiltonian conserves a twofold pseudospin, and every
    representation of its copies -- ``Ag + Ag``, ``Ag + Au``, ``Au + Au`` --
    gives the one site content ``2xA`` of the site group ``1``, while they
    label the bands differently at the momenta the inversion fixes.

    Returns
    -------
    pythtb.TBModel
        Spinless 3D model, 4 orbitals / 4 bands: orbitals 0 and 1 one copy at
        ``r0`` and ``-r0``, orbitals 2 and 3 the other.
    """
    A = P1BAR_LATTICE
    pos = [P1BAR_SITE, -P1BAR_SITE]
    rng = np.random.default_rng(int(seed))
    raw = {}
    for i in range(2):
        for j in range(2):
            for R in itertools.product(range(-2, 3), repeat=3):
                d = (pos[j] + np.array(R) - pos[i]) @ A
                if not 1e-9 < np.linalg.norm(d) <= cutoff or (i, j, R) in raw:
                    continue
                raw[(i, j, R)] = raw[(j, i, tuple(-x for x in R))] = rng.normal()
    # the inversion carries the bond (i, j, R) onto (1 - i, 1 - j, -R)
    hop = {(i, j, R): 0.5 * (t + raw[(1 - i, 1 - j, tuple(-x for x in R))])
           for (i, j, R), t in raw.items()}
    m = TBModel(Lattice(A.tolist(), [list(p) for p in pos + pos], periodic_dirs=[0, 1, 2]))
    m.set_onsite([0.0] * 4)
    done = set()
    for (i, j, R), t in sorted(hop.items()):
        if (j, i, tuple(-x for x in R)) in done:
            continue
        done.add((i, j, R))
        for c in (0, 2):
            m.set_hop(t, i + c, j + c, list(R), mode="add")
    return m
