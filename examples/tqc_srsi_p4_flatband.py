# Example of a splitting of bands into two sets that the stable real-space
# invariants (SRSIs) show cannot both be atomic limits, using pythTQC. Each
# set's symmetry data vector is that of an atomic limit, and the indicator
# group is trivial. The Wilson loop of each set then shows which sets wind.
# The models are the two P41' models of Hwang et al., "Stable real-space
# invariants and topology beyond symmetry indicators", Nat. Commun. 17, 8564
# (2026), 10.1038/s41467-026-74844-w, supplementary section 5A. Each has two
# Kramers pairs of s-like orbitals at Wyckoff 2c of the square lattice, at
# (1/2, 0) and (0, 1/2), in the order (x-up, x-down, y-up, y-down), and is
# spinful and two-dimensional.
# Model (i) is the flat-band model H(k) = t (u+ u+^dagger + u- u-^dagger), its
# Bloch states u+ and u- written out under section 1. It is a sum of two
# rank-one projectors onto Bloch states that are not orthogonal, so two bands
# sit at zero and two disperse. Model (ii) is the Z2 model, whose blocks are
# written out under section 1. Both Hamiltonians are transcribed from their
# closed forms. The comments derive the finite list of hoppings each one
# implies, and section 1 checks each typed model against its closed form at
# random k. The paper writes its spin blocks with spin quantized along -z. They
# are conjugated here by i sigma_x, so that up is spin along +z as pythtb has
# it.
#
# As typed, both models also obey the four vertical mirrors of the square
# lattice, and their group is P4mm1'. Each is therefore measured with a small
# chiral pattern of diagonal hoppings added, from each orbital to its own images
# displaced by the lattice vectors (1,1) and (1,-1), with unequal amplitudes on
# the two diagonals. The pattern removes the mirrors, so the group the hoppings
# obey is P41', the group the paper's analysis is written in. For the flat-band
# model, the gap between the lower two bands and the upper two is measured at
# 11 strengths of the pattern, from zero to full, and stays open at each. So its
# two sets of bands are the supplement's own, deformed without closing the gap.
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. the two Hamiltonians typed as hoppings, each checked against its own
#    closed form at 500 random k
# 2. the group the hoppings obey before and after the chiral pattern, where
#    the orbitals sit, and the gap, which stays open as the pattern is turned on
# 3. srsi_of_model(realized): the SRSIs, read off the orbitals
# 4. the spectrum, and for each of the two sets of bands its symmetry data
#    vector, the sums of elementary band representations (EBRs) with that
#    vector, and its indicator class: what momentum space says about the
#    splitting
# 5. splitting(realized, bands=2): bands 0 and 1 against bands 2 and 3, and
#    whether the splitting is what the paper calls non-symmetry-indicated
#    topological, printed nsit: each set has atomic realizations, and no pair
#    of them has invariants valued modulo n adding up to the whole model's
# 6. the k_x-directed Wilson loop of each set as a function of k_y
# 7. the Z2 model: the same group, nsit again, and both loops winding
# 8. the two spectra, the two Wilson loops, and the figure written out

# Author: the pythTQC examples

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb import Lattice, Mesh, TBModel, WFArray

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_coreps as fmt
from pythtqc.utils import save_figure

# Some lines print ±, which a Windows console cannot print by default, so the
# output is written as UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band path and nothing else. The energy and
# Wilson-loop meshes below, on which the printed numbers are measured, are the
# same in both modes.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_p4_flatband")
NK_PATH = 61 if QUICK else 241              # band-path density: the figure only
NKE = 41                                    # the energy mesh, fixed
NKX, NKY = 81, 121                          # the Wilson-loop mesh, fixed

T_FLAT = 1.0                                # the flat-band model's one scale
T_NN, T_NNN, LAMBDA = 2.0, 1.0, 1.0         # the Z2 model's, as the supplement sets them

# The chiral pattern of diagonal hoppings: two unequal amplitudes on the two
# diagonals, small on the scale of the transcribed model's own gap.
T_CHIRAL_A, T_CHIRAL_B = 0.03, 0.0125

PAULI_0 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

# The supplement writes its fourfold operator as tau_1 . e^{+i pi sigma_3 / 4}.
# pythtb has spin along +z in (up, dn) order. The tables pythTQC reads lift a
# rotation by theta about n onto the spinor as e^{-i theta n.sigma / 2}, with
# spin along +z, as the section "Spin" of docs/background/conventions.md
# states. The two frames differ by conjugating every 2x2 spin block by
# W = i sigma_x, so each model typed below is rewritten block by block onto
# pythtb's axis. The same W on every orbital is the global unitary
# SPIN_FLIP = 1_2 (x) W on H(k), which leaves every band, gap and Wilson loop
# unchanged. What it changes is which spinor rotations of the tables H is
# symmetric under, and so the group and the site irreps measured. In the
# paper's frame, the model with the chiral pattern obeys only P21'. SPIN_FLIP
# is applied to the closed forms below.
W = 1j * PAULI_X
SPIN_FLIP = np.kron(np.eye(2), W)

PATH_LABELS = (r"$\Gamma$", "X", "M", r"$\Gamma$")

os.makedirs(OUTDIR, exist_ok=True)

print("Hwang et al., Nat. Commun. 17, 8564 (2026), supplementary section 5A: two Kramers")
print("pairs at 2c of the square lattice, with the vertical mirrors broken in the hoppings.")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One printed row: a label of at most 16 characters, then the row's lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def values(theta):
    return "(" + ", ".join(str(v) for v in theta) + ")"


def mod_n(theta_zn, modulus):
    """The invariants valued modulo ``modulus``, in the tables' order."""
    return tuple(entry.value for entry in theta_zn.values() if entry.modulus == modulus)


def reading(split):
    """``nsit`` or ``not nsit``, the nsit verdict on a splitting."""
    return "nsit" if split.nsit else "not nsit"


def numbers(points):
    """Numbers written out, each rounded to six decimals."""
    return ", ".join(str(float(np.round(v, 6))) for v in points)


def atomic_limits(d):
    """The sums of elementary band representations whose symmetry data vector the band
    set carries, as a label and a value: the one sum, how many sums there are, or none."""
    if not d.decomposable:
        return "EBR sums", "none"
    sums = [s.label for s in d.candidate_sums]
    if len(sums) == 1:
        return "EBR sum", sums[0]
    return "EBR sums", f"{len(sums)} sums of elementary band representations"


def indicator_class(si):
    """The symmetry-indicator class of a band set, as a label and a value."""
    if not si.divisors:
        return "indicator group", "trivial"
    return "indicator class", f"{values(si.values)} in {si.indicator_group}"


def wilson_row(label, ky, trace, winding):
    """Where the two hybrid Wannier centers of a set meet, tr W / 2 there, and the winding."""
    row(label, f"the hybrid Wannier centers meet at k_y = {numbers(ky)},",
        f"where tr W / 2 = {numbers(trace)},",
        "and the loop does not wind" if int(round(winding)) == 0
        else f"and each center winds by ±{int(round(winding))}")


########### 1. the two Hamiltonians, typed and checked ###########

print("\n1. the supplement's two Hamiltonians, typed as hoppings and checked")

# --- the flat-band model ------------------------------------------------------
# The paper's Bloch states, in the momentum basis e^{i k.(R + t_alpha)}, which
# is pythtb's convention I. In this convention, a hopping <0,alpha|H|R,beta>
# enters H(k) with the phase e^{i k.(R + t_beta - t_alpha)}:
#
#   u+(k) = ( cos(kx/2), -i sin(kx/2),  cos(ky/2), -sin(ky/2) ) / sqrt(2)
#   u-(k) = (-i sin(kx/2), -cos(kx/2),  sin(ky/2), -cos(ky/2) ) / sqrt(2)
#   H(k)  = t ( u+ u+^dagger + u- u-^dagger )
#
# Writing c = cos(kx/2), s = sin(kx/2), C = cos(ky/2), S = sin(ky/2), H(k) is
# (t/2) times
#
#   [[      1,      i sin kx,   cC - i sS,  -cS + i sC ],
#    [ -i sin kx,          1,  -cS - i sC,   cC + i sS ],
#    [        .,         .,             1,     -sin ky ],
#    [        .,         .,       -sin ky,           1 ]]   (lower half by h.c.)
#
# and every entry of that is a finite hopping:
#  * the diagonal is a constant t/2 on both orbitals;
#  * sin kx on the x site and sin ky on the y site are same-site hops by one
#    lattice vector, to the next-nearest neighbors of the 2c orbit;
#  * the half-angle products are the four x-y bonds.  With t_y - t_x =
#    (-1/2, 1/2), the bond phases are P(R) = e^{i k.(R + t_y - t_x)}, so
#    P(0,0) = e^{i(-kx+ky)/2}, P(1,0) = e^{i(kx+ky)/2}, P(0,-1) = e^{-i(kx+ky)/2}
#    and P(1,-1) = e^{i(kx-ky)/2}; then
#      cC = [P(1,-1) + P(0,0) + P(1,0) + P(0,-1)] / 4
#      sS = [P(1,-1) + P(0,0) - P(1,0) - P(0,-1)] / 4
#      sC = [P(1,0) - P(0,-1) + P(1,-1) - P(0,0)] / 4i
#      cS = [P(1,0) - P(0,-1) - P(1,-1) + P(0,0)] / 4i
#    which gives the four 2x2 spin blocks typed below (rows = x spin,
#    columns = y spin).
lattice = Lattice([[1, 0], [0, 1]], [[0.5, 0.0], [0.0, 0.5]], periodic_dirs=[0, 1])
paper_flat = TBModel(lattice, spinful=True)
paper_flat.set_onsite([0.5 * T_FLAT * PAULI_0, 0.5 * T_FLAT * PAULI_0], mode="set")
paper_flat.set_hop(0.25j * T_FLAT * PAULI_Y, 0, 0, [1, 0])       # (t/2) i sin kx
paper_flat.set_hop(0.25j * T_FLAT * PAULI_X, 1, 1, [0, 1])       # -(t/2) sin ky
paper_flat.set_hop(T_FLAT / 8 * np.array([[1 - 1j, -(1 - 1j)],
                                          [1 + 1j, 1 + 1j]]), 0, 1, [0, 0])
paper_flat.set_hop(T_FLAT / 8 * np.array([[1 + 1j, 1 + 1j],
                                          [-(1 - 1j), 1 - 1j]]), 0, 1, [1, 0])
paper_flat.set_hop(T_FLAT / 8 * np.array([[1 + 1j, -(1 + 1j)],
                                          [1 - 1j, 1 - 1j]]), 0, 1, [0, -1])
paper_flat.set_hop(T_FLAT / 8 * np.array([[1 - 1j, 1 - 1j],
                                          [-(1 + 1j), 1 + 1j]]), 0, 1, [1, -1])

# The same model, with every 2x2 spin block conjugated onto pythtb's axis.
flat_model = TBModel(lattice, spinful=True)
flat_model.set_onsite(
    [W @ b @ W.conj().T for b in
     np.asarray(paper_flat.onsite, dtype=complex).reshape(int(paper_flat.norb), 2, 2)],
    mode="set")
for h in paper_flat.hoppings:
    flat_model.set_hop(
        W @ np.asarray(h["amplitude"], dtype=complex).reshape(2, 2) @ W.conj().T,
        int(h["from_orbital"]), int(h["to_orbital"]),
        list(h.get("lattice_vector", [0, 0])), mode="add")

rng = np.random.default_rng(7)
k_random = rng.uniform(-0.5, 0.5, size=(500, 2))
h_typed = np.asarray(flat_model.hamiltonian(k_pts=k_random, flatten_spin_axis=True))
worst_flat = 0.0
for index, (k1, k2) in enumerate(k_random):
    kx, ky = 2 * np.pi * k1, 2 * np.pi * k2
    c, s = np.cos(kx / 2), np.sin(kx / 2)
    C, S = np.cos(ky / 2), np.sin(ky / 2)
    u_plus = np.array([c, -1j * s, C, -S]) / np.sqrt(2)
    u_minus = np.array([-1j * s, -c, S, -C]) / np.sqrt(2)
    closed = T_FLAT * (np.outer(u_plus, u_plus.conj())
                       + np.outer(u_minus, u_minus.conj()))
    closed = SPIN_FLIP @ closed @ SPIN_FLIP.conj().T
    worst_flat = max(worst_flat, float(np.abs(h_typed[index] - closed).max()))
row("flat-band model", f"{flat_model.norb} orbitals, {flat_model.nstate} bands, "
    f"{len(flat_model.hoppings)} typed hoppings")

# --- the Z2 model -------------------------------------------------------------
# The supplement's blocks, in the same basis and the same convention:
#
#   A  = 2 t_NNN cos ky . 1        (the x site)
#   A' = 2 t_NNN cos kx . 1        (the y site)
#   B  = diag( 2(t_NN - i lam) cos((kx-ky)/2) + 2(t_NN + i lam) cos((kx+ky)/2),
#              2(t_NN + i lam) cos((kx-ky)/2) + 2(t_NN - i lam) cos((kx+ky)/2) )
#   H  = [[A, B], [B^dagger, A']],     t_NN = 2, t_NNN = 1, lam = 1
#
# A and A' are single same-site hops by one lattice vector; and with the same
# four bond phases as above, 2 cos((kx-ky)/2) = P(1,-1) + P(0,0) and
# 2 cos((kx+ky)/2) = P(1,0) + P(0,-1), so B is four x-y bonds, diagonal in spin.
paper_z2 = TBModel(lattice, spinful=True)
paper_z2.set_hop(T_NNN * PAULI_0, 0, 0, [0, 1])                  # A  = 2 t_NNN cos ky
paper_z2.set_hop(T_NNN * PAULI_0, 1, 1, [1, 0])                  # A' = 2 t_NNN cos kx
paper_z2.set_hop(np.diag([T_NN - 1j * LAMBDA, T_NN + 1j * LAMBDA]), 0, 1, [0, 0])
paper_z2.set_hop(np.diag([T_NN - 1j * LAMBDA, T_NN + 1j * LAMBDA]), 0, 1, [1, -1])
paper_z2.set_hop(np.diag([T_NN + 1j * LAMBDA, T_NN - 1j * LAMBDA]), 0, 1, [1, 0])
paper_z2.set_hop(np.diag([T_NN + 1j * LAMBDA, T_NN - 1j * LAMBDA]), 0, 1, [0, -1])

z2_model = TBModel(lattice, spinful=True)
z2_model.set_onsite(
    [W @ b @ W.conj().T for b in
     np.asarray(paper_z2.onsite, dtype=complex).reshape(int(paper_z2.norb), 2, 2)],
    mode="set")
for h in paper_z2.hoppings:
    z2_model.set_hop(
        W @ np.asarray(h["amplitude"], dtype=complex).reshape(2, 2) @ W.conj().T,
        int(h["from_orbital"]), int(h["to_orbital"]),
        list(h.get("lattice_vector", [0, 0])), mode="add")

h_typed_z2 = np.asarray(z2_model.hamiltonian(k_pts=k_random, flatten_spin_axis=True))
worst_z2 = 0.0
for index, (k1, k2) in enumerate(k_random):
    kx, ky = 2 * np.pi * k1, 2 * np.pi * k2
    block_a = 2 * T_NNN * np.cos(ky) * np.eye(2)
    block_a_prime = 2 * T_NNN * np.cos(kx) * np.eye(2)
    block_b = np.diag([
        2 * (T_NN - 1j * LAMBDA) * np.cos((kx - ky) / 2)
        + 2 * (T_NN + 1j * LAMBDA) * np.cos((kx + ky) / 2),
        2 * (T_NN + 1j * LAMBDA) * np.cos((kx - ky) / 2)
        + 2 * (T_NN - 1j * LAMBDA) * np.cos((kx + ky) / 2)])
    closed = np.block([[block_a, block_b], [block_b.conj().T, block_a_prime]])
    closed = SPIN_FLIP @ closed @ SPIN_FLIP.conj().T
    worst_z2 = max(worst_z2, float(np.abs(h_typed_z2[index] - closed).max()))
row("Z2 model", f"{z2_model.norb} orbitals, {z2_model.nstate} bands, "
    f"{len(z2_model.hoppings)} typed hoppings")
row("closed forms", f"the typed Hamiltonians differ from them by at most {worst_flat:.6f}",
    f"and {worst_z2:.6f} at {len(k_random)} random k")

########### 2. the group the hoppings obey ###########

print("\n2. the group the hoppings obey, and the chiral pattern that lowers it")
wide = tqc.realized_group(flat_model)
row("as transcribed", f"realized group #{wide.group.number_str} {wide.group.name}, larger than the "
    "P41' the paper", "analyzes: the typed hoppings also obey the four vertical mirrors",
    "of the square lattice")

# A pattern of diagonal hoppings that obeys C4 and time reversal, with unequal
# amplitudes on the two diagonals, removes the mirrors. It has t_a along (1,1)
# and t_b along (1,-1) on the x site, and the two swapped on the y site. C4
# demands the swap, since it exchanges the sites and carries (1,1) to (1,-1).
# m(x) would carry the x site's t_a bond to its t_b bond, and m([110]) the x
# site's t_a bond to the y site's t_b bond, so with t_a != t_b neither is a
# symmetry. The spin blocks are the identity, so time reversal is kept.
chiral_model = TBModel(lattice, spinful=True)
chiral_model.set_onsite(flat_model.onsite, mode="set")
for h in flat_model.hoppings:
    chiral_model.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                         list(h.get("lattice_vector", [0, 0])), mode="add")
chiral_model.set_hop(T_CHIRAL_A * PAULI_0, 0, 0, [1, 1], mode="add")
chiral_model.set_hop(T_CHIRAL_B * PAULI_0, 0, 0, [1, -1], mode="add")
chiral_model.set_hop(T_CHIRAL_A * PAULI_0, 1, 1, [1, -1], mode="add")
chiral_model.set_hop(T_CHIRAL_B * PAULI_0, 1, 1, [1, 1], mode="add")

realized = tqc.realized_group(chiral_model)
group = realized.group
row("chiral pattern", f"t_a = {T_CHIRAL_A} along (1,1) and t_b = {T_CHIRAL_B} along (1,-1) on "
    "the", "x site, the two swapped on the y site")
row("with the pattern", f"realized group #{group.number_str} {group.name}, with the orbitals at",
    f"Wyckoff {realized.wyckoff_orbits[0].wyckoff} as in the paper; nothing is declared")

# The gap between bands 1 and 2 is measured at 11 values of s along the ramp
# from the transcribed model, s = 0, to the one with the pattern, s = 1. If it
# stays open at all of them, the two sets of bands studied in sections 3 to 6
# are the transcribed model's own two, deformed without closing the gap.
mesh_e = np.stack(np.meshgrid(np.linspace(0, 1, NKE, endpoint=False),
                              np.linspace(0, 1, NKE, endpoint=False),
                              indexing="ij"), -1).reshape(-1, 2)
energies_typed = np.asarray(flat_model.solve_ham(mesh_e))
energies = np.asarray(chiral_model.solve_ham(mesh_e))
ramp, ramp_s = [], np.linspace(0.0, 1.0, 11)
ramp_model = TBModel(lattice, spinful=True)
ramp_model.set_onsite(flat_model.onsite, mode="set")
for h in flat_model.hoppings:
    ramp_model.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                       list(h.get("lattice_vector", [0, 0])), mode="add")
for s in ramp_s:
    ramp_model.set_hop(s * T_CHIRAL_A * PAULI_0, 0, 0, [1, 1], mode="set")
    ramp_model.set_hop(s * T_CHIRAL_B * PAULI_0, 0, 0, [1, -1], mode="set")
    ramp_model.set_hop(s * T_CHIRAL_A * PAULI_0, 1, 1, [1, -1], mode="set")
    ramp_model.set_hop(s * T_CHIRAL_B * PAULI_0, 1, 1, [1, 1], mode="set")
    e_ramp = np.asarray(ramp_model.solve_ham(mesh_e))
    ramp.append(float(np.min(e_ramp[:, 2] - e_ramp[:, 1])))
row("bands 1 and 2", f"over a {NKE}x{NKE} mesh, smallest direct gap "
    f"{np.min(energies_typed[:, 2] - energies_typed[:, 1]):.6f} as transcribed,",
    f"{np.min(energies[:, 2] - energies[:, 1]):.6f} with the pattern, and {min(ramp):.6f} at "
    "its smallest along the", f"{len(ramp)}-point ramp between them")
print("   the gap stays open at each point of the ramp, so the two sets of bands studied")
print("   next are the supplement's own, deformed without closing the gap")

########### 3. the model's own SRSIs ###########

print("\n3. srsi_of_model(realized): the stable real-space invariants (SRSIs), read off the orbitals")
m_srsi = tqc.srsi_of_model(realized)
print(textwrap.indent(m_srsi.summary(), "   "))

########### 4. the spectrum, and what momentum space says ###########

print("\n4. the spectrum, and what momentum space says about the two sets of bands")
flat_zero = float(np.abs(energies_typed[:, :2]).max())
row("as transcribed", f"bands 0 and 1 flat at zero, largest |E| {flat_zero:.6f} over the",
    f"{NKE}x{NKE} mesh; bands 2 and 3 from {energies_typed[:, 2].min():.6f} to "
    f"{energies_typed[:, 3].max():.6f}")
row("with the pattern", f"bands 0 and 1 from {energies[:, :2].min():.6f} to "
    f"{energies[:, :2].max():.6f}, a spread of first order",
    f"in the pattern, on its scale 2(t_a + t_b) = {2 * (T_CHIRAL_A + T_CHIRAL_B):.6f}; bands 2 "
    "and 3", f"from {energies[:, 2].min():.6f} to {energies[:, 3].max():.6f}, moved at second order")
row("bands 1 and 2", f"indirect gap {energies[:, 2].min() - energies[:, 1].max():.6f} with the pattern")

t_valence = tqc.hsp_table(realized, bands=[0, 1])
b_valence = tqc.srsi_of_bands(t_valence)
t_conduction = tqc.hsp_table(realized, bands=[2, 3])
b_conduction = tqc.srsi_of_bands(t_conduction)
d_valence = tqc.decompose(t_valence)
d_conduction = tqc.decompose(t_conduction)
row("bands 0 and 1", f"{fmt(t_valence.symmetry_data)}, Z-valued SRSIs {values(b_valence.theta_Z.values())}")
row(*atomic_limits(d_valence))
row("bands 2 and 3", f"{fmt(t_conduction.symmetry_data)}, Z-valued SRSIs {values(b_conduction.theta_Z.values())}")
row(*atomic_limits(d_conduction))
class_lower = indicator_class(b_valence.indicator)
class_upper = indicator_class(b_conduction.indicator)
if class_lower == class_upper:
    row(*class_lower)
else:
    row(class_lower[0], f"{class_lower[1]} for bands 0 and 1", f"{class_upper[1]} for bands 2 and 3")
print("   each set's symmetry data vector is that of one EBR, an atomic limit")

########### 5. the splitting ###########

print("\n5. splitting(realized, bands=2): bands 0 and 1 against bands 2 and 3")
split_flat = tqc.splitting(realized, bands=2)
print(textwrap.indent(split_flat.summary(), "   "))
(modulus,) = {entry.modulus for entry in split_flat.theta_Zn.values()}   # the one modulus of #75.2
print("   each set's one atomic realization induces the EBR section 4 prints for it")

########### 6. the Wilson loop of each set ###########

print("\n6. the k_x-directed Wilson loop W of each set, as a function of k_y")

# The Wilson loops are pythtb's: a WFArray on a (k_x, k_y) grid, wilson_loop()
# for the unitary, and berry_phase(berry_evals=True) for the phases of the
# hybrid Wannier centers along the k_x loop at each k_y. They are measured on
# the model with the chiral pattern, whose group gave the verdict above.
mesh = Mesh(["k", "k"])
mesh.build_grid([NKX, NKY], k_endpoints=[False, True])
wf_chiral = WFArray(lattice, mesh, spinful=True)
wf_chiral.solve_model(chiral_model)
ky_axis = mesh.points[0, :, 1]
row("mesh", f"{NKX} x {NKY}, k_y from {ky_axis[0]:g} to {ky_axis[-1]:g} inclusive")

w_lower = wf_chiral.wilson_loop(axis_idx=0, state_idx=[0, 1])
phases_lower = wf_chiral.berry_phase(axis_idx=0, state_idx=[0, 1], berry_evals=True,
                                     contin=False) / (2 * np.pi)
cos_lower = np.real(np.trace(w_lower, axis1=1, axis2=2)) / 2.0
psi_lower = np.arccos(np.clip(cos_lower, -1.0, 1.0))
winding_lower = float(np.abs(np.diff(psi_lower)).sum() / (2 * np.pi))
touch_lower = np.abs(cos_lower) > 1 - 1e-9
wilson_row("bands 0 and 1", ky_axis[touch_lower], cos_lower[touch_lower], winding_lower)

w_upper = wf_chiral.wilson_loop(axis_idx=0, state_idx=[2, 3])
phases_upper = wf_chiral.berry_phase(axis_idx=0, state_idx=[2, 3], berry_evals=True,
                                     contin=False) / (2 * np.pi)
cos_upper = np.real(np.trace(w_upper, axis1=1, axis2=2)) / 2.0
psi_upper = np.arccos(np.clip(cos_upper, -1.0, 1.0))
winding_upper = float(np.abs(np.diff(psi_upper)).sum() / (2 * np.pi))
center_upper = float(np.abs(phases_upper).max())
row("bands 2 and 3", f"both centers at zero at every k_y, largest |center| {center_upper:.6f},",
    "and the loop does not wind" if int(round(winding_upper)) == 0
    else f"and each center winds by ±{int(round(winding_upper))}")
print("   only bands 0 and 1 wind: they carry the obstruction section 5 diagnoses")

########### 7. the Z2 model ###########

print("\n7. the Z2 model: the same group, nsit again, and both loops winding")
z2_wide = tqc.realized_group(z2_model)

z2_chiral = TBModel(lattice, spinful=True)
for h in z2_model.hoppings:
    z2_chiral.set_hop(h["amplitude"], h["from_orbital"], h["to_orbital"],
                      list(h.get("lattice_vector", [0, 0])), mode="add")
z2_chiral.set_hop(T_CHIRAL_A * PAULI_0, 0, 0, [1, 1], mode="add")
z2_chiral.set_hop(T_CHIRAL_B * PAULI_0, 0, 0, [1, -1], mode="add")
z2_chiral.set_hop(T_CHIRAL_A * PAULI_0, 1, 1, [1, -1], mode="add")
z2_chiral.set_hop(T_CHIRAL_B * PAULI_0, 1, 1, [1, 1], mode="add")

z2_realized = tqc.realized_group(z2_chiral)
z2_srsi = tqc.srsi_of_model(z2_realized)
split_z2 = tqc.splitting(z2_realized, bands=2)
row("as transcribed", f"realized group #{z2_wide.group.number_str} {z2_wide.group.name}")
row("with the pattern", f"realized group #{z2_realized.group.number_str} "
    f"{z2_realized.group.name}")

energies_z2 = np.asarray(z2_chiral.solve_ham(mesh_e))
pair_01 = float(np.abs(energies_z2[:, 0] - energies_z2[:, 1]).max())
pair_23 = float(np.abs(energies_z2[:, 2] - energies_z2[:, 3]).max())
row("degenerate pairs", f"bands 0 and 1, and bands 2 and 3, over the {NKE}x{NKE} mesh: largest",
    f"|E0 - E1| {pair_01:.6f} and |E2 - E3| {pair_23:.6f}")
row("bands 1 and 2", f"gap {energies_z2[:, 2].min() - energies_z2[:, 1].max():.6f}")
row("Z-valued SRSIs", f"all 4 bands {values(z2_srsi.theta_Z.values())}, the valence bands "
    f"{values(split_z2.valence.theta_Z.values())}")
row(f"mod {modulus}", f"all 4 bands {values(mod_n(z2_srsi.theta_Zn, modulus))}")
row("splitting", "bands 0 and 1 against bands 2 and 3 is",
    "nsit" if split_z2.nsit else "not nsit")
if split_z2.consistent_pairs:
    row("consistent pairs", f"{len(split_z2.consistent_pairs)}, each a valence and a conduction "
        "realization whose", "invariants valued modulo n add up to those of all 4 bands")
else:
    row("consistent pairs", "none: no valence and conduction realization have invariants",
        "valued modulo n adding up to those of all 4 bands")

wf_z2 = WFArray(lattice, mesh, spinful=True)
wf_z2.solve_model(z2_chiral)

w_z2_v = wf_z2.wilson_loop(axis_idx=0, state_idx=[0, 1])
phases_z2_v = wf_z2.berry_phase(axis_idx=0, state_idx=[0, 1], berry_evals=True,
                                contin=False) / (2 * np.pi)
cos_z2_v = np.real(np.trace(w_z2_v, axis1=1, axis2=2)) / 2.0
psi_z2_v = np.arccos(np.clip(cos_z2_v, -1.0, 1.0))
winding_z2_v = float(np.abs(np.diff(psi_z2_v)).sum() / (2 * np.pi))
touch_z2_v = np.abs(cos_z2_v) > 1 - 1e-9
wilson_row("bands 0 and 1", ky_axis[touch_z2_v], cos_z2_v[touch_z2_v], winding_z2_v)

w_z2_c = wf_z2.wilson_loop(axis_idx=0, state_idx=[2, 3])
phases_z2_c = wf_z2.berry_phase(axis_idx=0, state_idx=[2, 3], berry_evals=True,
                                contin=False) / (2 * np.pi)
cos_z2_c = np.real(np.trace(w_z2_c, axis1=1, axis2=2)) / 2.0
psi_z2_c = np.arccos(np.clip(cos_z2_c, -1.0, 1.0))
winding_z2_c = float(np.abs(np.diff(psi_z2_c)).sum() / (2 * np.pi))
touch_z2_c = np.abs(cos_z2_c) > 1 - 1e-9
wilson_row("bands 2 and 3", ky_axis[touch_z2_c], cos_z2_c[touch_z2_c], winding_z2_c)
print("   both sets wind here, and in the flat-band model only bands 0 and 1 do: the SRSIs")
print("   show that the two sets cannot both be atomic limits, not which set is not one")

########### 8. the two spectra and the two Wilson loops ###########

print("\n8. the two spectra and the two Wilson loops, one panel each")
hsp = tqc.high_symmetry_points(group)
path = [hsp["GM"], hsp["X"], hsp["M"], hsp["GM"]]
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))

# --- the flat-band model's spectrum, with the mirrors broken ---
kv, kd, kn = chiral_model.k_path(path, NK_PATH, report=False)
bands = np.asarray(chiral_model.solve_ham(kv))
for i in range(4):
    axes[0, 0].plot(kd, bands[:, i], lw=2.0, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0, 0].axvline(x, color="0.82", lw=0.7)
axes[0, 0].set_xticks(kn)
axes[0, 0].set_xticklabels(PATH_LABELS, fontsize=13)
axes[0, 0].set_xlim(kd[0], kd[-1])
axes[0, 0].set_ylabel("$E$", fontsize=15)
axes[0, 0].set_title(f"(i) the flat-band model, mirrors broken, "
                     f"{group.name} #{group.number_str}\n"
                     fr"$\theta_\mathbb{{Z}}[V]="
                     fr"({', '.join(str(v) for v in b_valence.theta_Z.values())})$, "
                     fr"$\theta_\mathbb{{Z}}[C]="
                     fr"({', '.join(str(v) for v in b_conduction.theta_Z.values())})$, "
                     fr"$\theta_{{\mathbb{{Z}}_2}}[H]={mod_n(m_srsi.theta_Zn, modulus)}$"
                     f"\nsplitting: {reading(split_flat)}", fontsize=9.5)

# --- the same model's Wilson loops ---
for column in range(2):
    axes[0, 1].plot(ky_axis, phases_lower[:, column], "o", ms=2.6, color="tab:blue")
for column in range(2):
    axes[0, 1].plot(ky_axis, phases_upper[:, column], "s", ms=2.6, color="tab:red")
axes[0, 1].set_xlabel("$k_y / 2\\pi$", fontsize=12)
axes[0, 1].set_ylabel(r"$\phi_{k_x} / 2\pi$", fontsize=12)
axes[0, 1].set_ylim(-0.55, 0.55)
axes[0, 1].set_title(f"(i) $k_x$ Wilson loops: lower set (blue) winds "
                     f"±{int(round(winding_lower))},\nupper set (red) "
                     + ("does not wind" if int(round(winding_upper)) == 0
                        else f"winds ±{int(round(winding_upper))}"), fontsize=9.5)

# --- the Z2 model's spectrum, with the same mirrors broken ---
kv, kd, kn = z2_chiral.k_path(path, NK_PATH, report=False)
bands_z2 = np.asarray(z2_chiral.solve_ham(kv))
for i in range(4):
    axes[1, 0].plot(kd, bands_z2[:, i], lw=2.0, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[1, 0].axvline(x, color="0.82", lw=0.7)
axes[1, 0].set_xticks(kn)
axes[1, 0].set_xticklabels(PATH_LABELS, fontsize=13)
axes[1, 0].set_xlim(kd[0], kd[-1])
axes[1, 0].set_ylabel("$E$", fontsize=15)
axes[1, 0].set_title(f"(ii) the $\\mathbb{{Z}}_2$ model, mirrors broken, "
                     f"{z2_realized.group.name} #{z2_realized.group.number_str}\n"
                     fr"$\theta_{{\mathbb{{Z}}_2}}[H]={mod_n(z2_srsi.theta_Zn, modulus)}$"
                     f"   splitting: {reading(split_z2)}", fontsize=9.5)

# --- the Z2 model's Wilson loops ---
for column in range(2):
    axes[1, 1].plot(ky_axis, phases_z2_v[:, column], "o", ms=2.6, color="tab:blue")
for column in range(2):
    axes[1, 1].plot(ky_axis, phases_z2_c[:, column], "s", ms=2.6, color="tab:red")
axes[1, 1].set_xlabel("$k_y / 2\\pi$", fontsize=12)
axes[1, 1].set_ylabel(r"$\phi_{k_x} / 2\pi$", fontsize=12)
axes[1, 1].set_ylim(-0.55, 0.55)
axes[1, 1].set_title(f"(ii) $k_x$ Wilson loops: lower set (blue) winds "
                     f"±{int(round(winding_z2_v))},\nupper set (red) winds "
                     f"±{int(round(winding_z2_c))}", fontsize=9.5)

fig.suptitle("The P41' models of supplementary section 5A, with the vertical mirrors broken so that "
             "P41' is the group their own hoppings obey:\nthe SRSIs show that in neither "
             "model can both sets be atomic limits, and the Wilson loops show which "
             "sets wind.", fontsize=10)
fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_p4_flatband.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
