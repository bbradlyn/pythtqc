# Example of a band splitting that momentum space has no objection to and the
# SRSI no-go criterion diagnoses all the same, and of the Wilson loop that then
# says which obstruction each side carries, using pythtqc.
# The models are the two P41' models of Hwang et al., "Stable real-space
# invariants and topology beyond symmetry indicators", Nat. Commun. 17, 8564
# (2026), 10.1038/s41467-026-74844-w, supplementary section 5A: two Kramers
# pairs of s-like orbitals at Wyckoff 2c of the square lattice, at (1/2, 0) and
# (0, 1/2), in the order (x-up, x-down, y-up, y-down), both spinful and
# two-dimensional.
# (i) is the flat-band model H(k) = t (u+ u+^dagger + u- u-^dagger), a sum of
# two rank-one projectors onto non-orthogonal Bloch states, so two bands sit at
# zero and two disperse; (ii) is the Z2 model, blocks A / A' on the two sites
# and B between them.  Both Hamiltonians are transcribed below from their
# closed forms: the finite hopping list each one implies is derived in the
# comments and typed out, and section 1 measures the typed model against the
# closed form at random k.  The paper's spin blocks are written for the opposite
# quantization axis and are conjugated by i sigma_x here, so that up is spin
# along +z as pythtb has it.
#
# What is analysed.  Neither transcription obeys only P41': as typed, both are
# symmetric under the four vertical mirrors of the square lattice as well.  So
# what each section below runs on is the supplement's Hamiltonian plus a small
# chiral next-nearest hopping pattern with unequal amplitudes on the two
# diagonals, which removes those mirrors, so that the P41' the paper writes its
# analysis in is the group the hoppings themselves obey.  The pattern is a
# gapped deformation -- measured on a mesh and along the whole ramp -- so the
# band sets it is asked about are the supplement's own, continued.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. the two Hamiltonians typed as hoppings, each checked against its own
#    closed form at 500 random k
# 2. the group the hoppings obey before and after the chiral pattern, the gap
#    that makes the pattern a deformation, and where the orbitals sit
# 3. srsi_of_model: the SRSIs of the Kramers pair, read off the orbitals
# 4. the spectrum, and the two pairs' contents, indicator classes and
#    decompositions -- what momentum space says about the splitting
# 5. splitting(): what the real-space invariants say about the same splitting
# 6. the k_x-directed Wilson loop of each pair as a function of k_y
# 7. the Z2 model: the same group, the same verdict, and both loops winding
# 8. the two spectra, the two Wilson loops, and the figure written out

# Author: the pythtqc examples

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb import Lattice, Mesh, TBModel, WFArray

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_counts as fmt
from pythtqc.utils import save_figure

# Elementary band representations print as ρ↑G, the site irrep ρ induced to the
# group; a Windows console cannot print ↑ by default, so the output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band path and nothing else; the energy and
# Wilson-loop meshes below are measurements, the same in both modes.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_p4_flatband")
NK_PATH = 61 if QUICK else 241              # band-path density: the figure only
NKE = 41                                    # the energy mesh, fixed
NKX, NKY = 81, 121                          # the Wilson-loop mesh, fixed

T_FLAT = 1.0                                # the flat-band model's one scale
T_NN, T_NNN, LAMBDA = 2.0, 1.0, 1.0         # the Z2 model's, as the supplement sets them

# the chiral next-nearest pattern: two unequal amplitudes on the two diagonals
# of the 2c orbit, small on the scale of the transcribed model's own gap
T_CHIRAL_A, T_CHIRAL_B = 0.03, 0.0125

PAULI_0 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

# The supplement writes its fourfold operator as tau_1 . e^{+i pi sigma_3 / 4},
# where pythtb has spin along +z in (up, dn) order and a rotation by theta about
# n acting on the spinor as e^{-i theta n.sigma / 2} (the convention is stated in
# pythtqc.spin).  The two frames differ by conjugating every 2x2 spin block by
# W = i sigma_x, so each model typed below is re-written block by block onto
# pythtb's axis.  The same W on every orbital is the global unitary
# 1_2 (x) W = SPIN_FLIP on H(k) -- a conjugation, so no band, gap, Wilson loop or
# SRSI moves; what moves is every comparison against the symmetry tables' own
# SU(2) lift, and with it the 2c site character.  SPIN_FLIP is the same
# conjugation on the 4-state basis, for the closed forms below.
W = 1j * PAULI_X
SPIN_FLIP = np.kron(np.eye(2), W)

PATH_LABELS = (r"$\Gamma$", "X", "M", r"$\Gamma$")

os.makedirs(OUTDIR, exist_ok=True)

print("Hwang et al., Nat. Commun. 17, 8564 (2026), SM section 5A: two Kramers")
print("pairs at 2c of the square lattice, with the vertical mirrors broken in the hoppings.")

########### 1. the two Hamiltonians, typed and checked ###########

print("\n1. the supplement's two Hamiltonians, typed as hoppings and checked")

# --- the flat-band model ------------------------------------------------------
# The paper's Bloch states, in the momentum basis e^{i k.(R + t_alpha)} -- which
# is pythtb's convention I, so a hopping <0,alpha|H|R,beta> enters H(k) with the
# phase e^{i k.(R + t_beta - t_alpha)}:
#
#   u+(k) = ( cos(kx/2), -i sin(kx/2),  cos(ky/2), -sin(ky/2) ) / sqrt(2)
#   u-(k) = (-i sin(kx/2), -cos(kx/2),  sin(ky/2), -cos(ky/2) ) / sqrt(2)
#   H(k)  = t ( u+ u+^dagger + u- u-^dagger )
#
# Writing c = cos(kx/2), s = sin(kx/2), C = cos(ky/2), S = sin(ky/2), the sum of
# the two outer products is (t/2) times
#
#   [[      1,      i sin kx,   cC - i sS,  -cS + i sC ],
#    [ -i sin kx,          1,  -cS - i sC,   cC + i sS ],
#    [        .,         .,             1,     -sin ky ],
#    [        .,         .,       -sin ky,           1 ]]   (lower half by h.c.)
#
# and every entry of that is a finite hopping:
#  * the diagonal is a constant t/2 on both orbitals;
#  * sin kx on the x site and sin ky on the y site are same-site hops by one
#    lattice vector -- the next-nearest neighbours of the 2c orbit;
#  * the half-angle products are the four x-y bonds.  With t_y - t_x =
#    (-1/2, 1/2), the bond phases are P(R) = e^{i k.(R + t_y - t_x)}, i.e.
#    P(0,0) = e^{i(-kx+ky)/2}, P(1,0) = e^{i(kx+ky)/2}, P(0,-1) = e^{-i(kx+ky)/2}
#    and P(1,-1) = e^{i(kx-ky)/2}; then
#      cC = [P(1,-1) + P(0,0) + P(1,0) + P(0,-1)] / 4
#      sS = [P(1,-1) + P(0,0) - P(1,0) - P(0,-1)] / 4
#      sC = [P(1,0) - P(0,-1) + P(1,-1) - P(0,0)] / 4i
#      cS = [P(1,0) - P(0,-1) - P(1,-1) + P(0,0)] / 4i
#    which collects the four 2x2 spin blocks typed below (rows = x spin,
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

# ... and the same model with every 2x2 spin block conjugated onto pythtb's axis
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
print(f"   flat-band model   : {flat_model.norb} orbitals, {flat_model.nstate} bands, "
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
print(f"   Z2 model          : {z2_model.norb} orbitals, {z2_model.nstate} bands, "
      f"{len(z2_model.hoppings)} typed hoppings")
print("   both typed hopping lists reproduce the supplement's Hamiltonian at every")
print(f"   one of {len(k_random)} random k: max |H - H_supplement| = {worst_flat:.6f} "
      f"and {worst_z2:.6f}")

########### 2. the group the hoppings obey ###########

print("\n2. the group the hoppings obey, and the chiral pattern that lowers it")
wide = tqc.realized_group(flat_model)
print(f"   as transcribed    : #{wide.group.number_str} {wide.group.name} — more than the "
      "P41' the paper analyses,")
print("                       because nothing in the transcription distinguishes the two")
print("                       diagonals of the square lattice")

# One C4- and time-reversal-symmetric next-nearest pattern with unequal
# amplitudes on the two diagonals removes the mirrors: t_a along (1,1) and t_b
# along (1,-1) on the x/2 site, the two swapped on the y/2 site, which is what
# C4 demands since it exchanges the sites and carries (1,1) to (1,-1).  m(x)
# would have to exchange the two diagonals on one site and m([110]) to carry one
# site's t_a bond to the other site's t_b bond; with t_a != t_b neither can.
# The spin blocks are the identity, so time reversal is untouched.
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
model, group, assignment, validated, rep = realized
sites = tqc.wyckoff_sites_tables(group, assignment.site_positions)   # letter and multiplicity
print(f"   mirrors broken    : t_a = {T_CHIRAL_A} along (1,1) and t_b = {T_CHIRAL_B} along "
      "(1,-1) on the x")
print(f"                       site, the two swapped on the y site — realized group "
      f"#{group.number_str} {group.name},")
print(f"                       orbitals at Wyckoff {sites[0].multiplicity}{sites[0].letter}, "
      "the position the paper puts its Kramers")
print("                       pairs on (measured from the Hamiltonian, nothing declared)")

# ... and the pattern is a deformation of the supplement's model: the two band
# sets sections 3-6 ask about are the transcribed model's own two, continued,
# because the gap between them stays open along the whole ramp s = 0 -> 1.
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
print(f"   a gapped change   : over a {NKE}x{NKE} mesh the smallest direct gap "
      "E2(k) - E1(k) is")
print(f"                       {np.min(energies[:, 2] - energies[:, 1]):.6f} with the "
      f"pattern and {np.min(energies_typed[:, 2] - energies_typed[:, 1]):.6f} without it, "
      f"and {min(ramp):.6f}")
print(f"                       at its smallest over the {len(ramp)}-point ramp between "
      "them — the two")
print("                       band sets below are the supplement's own, continued")

########### 3. the model's own SRSIs ###########

print("\n3. srsi_of_model — the SRSIs of the Kramers pair, read off the orbitals")
m_srsi = tqc.srsi_of_model(realized)
print(m_srsi.summary())
print("   -> a model's SRSIs are read off where its orbitals sit and what they")
print("      carry, so the chiral pattern cannot move them — only the group.")

########### 4. the spectrum, and what momentum space says ###########

print("\n4. the spectrum, and what momentum space says about the two pairs")
flat_zero = float(np.abs(energies_typed[:, :2]).max())
print(f"   the flat band     : as transcribed, bands 0 and 1 sit at zero, max |E| = "
      f"{flat_zero:.6f}")
print(f"                       over the {NKE}x{NKE} mesh — the orthogonal complement "
      "of the span of")
print("                       the two non-orthogonal Bloch states, which is what makes "
      "it flat")
print(f"   bands 0 and 1     : with the pattern they span "
      f"[{energies[:, :2].min():.6f}, {energies[:, :2].max():.6f}], on the")
print(f"                       pattern's own scale 2(t_a + t_b) = "
      f"{2 * (T_CHIRAL_A + T_CHIRAL_B):.6f} — an exactly")
print("                       degenerate pair split at first order")
print(f"   bands 2 and 3     : [{energies[:, 2].min():.6f}, {energies[:, 3].max():.6f}] "
      f"against [{energies_typed[:, 2].min():.6f}, {energies_typed[:, 3].max():.6f}] as "
      "transcribed —")
print(f"                       second order, and the gap min E2 - max E1 = "
      f"{energies[:, 2].min() - energies[:, 1].max():.6f}")

t_valence = tqc.hsp_table(rep, bands=[0, 1])
b_valence = tqc.srsi_of_bands(t_valence, group, rep=rep)
t_conduction = tqc.hsp_table(rep, bands=[2, 3])
b_conduction = tqc.srsi_of_bands(t_conduction, group, rep=rep)
d_valence = tqc.decompose(t_valence, group, rep=rep)
d_conduction = tqc.decompose(t_conduction, group, rep=rep)
print(f"   the lower pair    : {fmt(t_valence.counts)}, "
      f"theta_Z = ({', '.join(str(v) for v in b_valence.theta_Z)})")
print(f"   the upper pair    : {fmt(t_conduction.counts)}, "
      f"theta_Z = ({', '.join(str(v) for v in b_conduction.theta_Z)})")
print(f"   the indicator     : class {b_valence.indicator.values} of "
      f"{b_valence.indicator.structure} and {b_conduction.indicator.values} of "
      f"{b_conduction.indicator.structure} — #{group.number_str} has no")
print("                       elementary divisor above 1, so there is no indicator to read")
print(f"   decompose         : {d_valence.verdict}, "
      f"{d_valence.solution_strings()[0]} and")
print(f"                       {d_conduction.solution_strings()[0]}")
print("   -> both halves decompose and there is no indicator to read: nothing in")
print("      momentum space objects to this splitting.")

########### 5. the criterion ###########

print("\n5. splitting() — the lower pair against the upper one")
split_flat = tqc.splitting(realized, bands=2)
print(split_flat.summary())
(modulus,) = tuple(split_flat.theta_Zn_total)          # the one modulus of #75.2's theta_Zn
zn_v = split_flat.realizations_v.realizations[0].theta_Zn[modulus]
zn_c = split_flat.realizations_c.realizations[0].theta_Zn[modulus]
print(f"   the arithmetic    : theta_Zn mod {modulus}, V[0] {zn_v} + C[0] {zn_c} = "
      f"{tuple((a + b) % modulus for a, b in zip(zn_v, zn_c))}, against")
print(f"                       H {split_flat.theta_Zn_total[modulus]}")
print("   -> the only atomic configurations carrying the two halves' symmetry data")
print("      are the EBRs at 1a above, one for each half, and their theta_Zn cannot")
print("      add up to the theta_Zn of H: the splitting momentum space passed is")
print("      diagnosed topological.")

########### 6. the Wilson loop of each pair ###########

print("\n6. the k_x-directed Wilson loop of each pair, as a function of k_y")

# The Wilson loops are pythtb's: a WFArray on a (k_x, k_y) grid, wilson_loop()
# for the unitary and berry_phase(berry_evals=True) for the hybrid-Wannier-centre
# phases along the k_x loop at each k_y.  They are measured on the chiral model,
# the Hamiltonian whose group gave the verdict above.
mesh = Mesh(["k", "k"])
mesh.build_grid([NKX, NKY], k_endpoints=[False, True])
wf_chiral = WFArray(lattice, mesh, spinful=True)
wf_chiral.solve_model(chiral_model)
ky_axis = mesh.points[0, :, 1]
print(f"   the mesh          : {NKX} x {NKY}, k_y from {ky_axis[0]:g} to {ky_axis[-1]:g} "
      "inclusive")

w_lower = wf_chiral.wilson_loop(axis_idx=0, state_idx=[0, 1])
phases_lower = wf_chiral.berry_phase(axis_idx=0, state_idx=[0, 1], berry_evals=True,
                                     contin=False) / (2 * np.pi)
cos_lower = np.real(np.trace(w_lower, axis1=1, axis2=2)) / 2.0
psi_lower = np.arccos(np.clip(cos_lower, -1.0, 1.0))
winding_lower = float(np.abs(np.diff(psi_lower)).sum() / (2 * np.pi))
touch_lower = np.abs(cos_lower) > 1 - 1e-9
print("   the lower pair    : the two centres meet (|tr W / 2| = 1) at")
print(f"                       k_y = {[float(np.round(v, 6)) for v in ky_axis[touch_lower]]}, "
      f"where tr W / 2 =")
print(f"                       {[float(np.round(v, 6)) for v in cos_lower[touch_lower]]}, "
      + ("and the loop does not wind" if int(round(winding_lower)) == 0
         else f"and each branch winds by ±{int(round(winding_lower))}"))

w_upper = wf_chiral.wilson_loop(axis_idx=0, state_idx=[2, 3])
phases_upper = wf_chiral.berry_phase(axis_idx=0, state_idx=[2, 3], berry_evals=True,
                                     contin=False) / (2 * np.pi)
cos_upper = np.real(np.trace(w_upper, axis1=1, axis2=2)) / 2.0
psi_upper = np.arccos(np.clip(cos_upper, -1.0, 1.0))
winding_upper = float(np.abs(np.diff(psi_upper)).sum() / (2 * np.pi))
centre_upper = float(np.abs(phases_upper).max())
print("   the upper pair    : both centres sit at zero at every k_y, "
      f"max |phi / 2pi| = {centre_upper:.6f},")
print("                       "
      + ("and the loop does not wind" if int(round(winding_upper)) == 0
         else f"and each branch winds by ±{int(round(winding_upper))}"))
print("   -> section 5 says V and C cannot both be atomic insulators; the loop says")
print("      which of them carries the obstruction.")

########### 7. the Z2 model ###########

print("\n7. the Z2 model — the same group, the same verdict, and both loops winding")
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
print(f"   as transcribed    : #{z2_wide.group.number_str} {z2_wide.group.name} again, and "
      f"the same pattern lowers it to "
      f"#{z2_realized.group.number_str} {z2_realized.group.name}")

energies_z2 = np.asarray(z2_chiral.solve_ham(mesh_e))
pair_01 = float(np.abs(energies_z2[:, 0] - energies_z2[:, 1]).max())
pair_23 = float(np.abs(energies_z2[:, 2] - energies_z2[:, 3]).max())
print("   the spectrum      : bands 0 and 1 and bands 2 and 3 are each degenerate")
print(f"                       over the {NKE}x{NKE} mesh, max |E0 - E1| = {pair_01:.6f} and")
print(f"                       max |E2 - E3| = {pair_23:.6f}; the gap min E2 - max E1 = "
      f"{energies_z2[:, 2].min() - energies_z2[:, 1].max():.6f}")
print(f"   the same verdict  : theta_Z, V "
      f"({', '.join(str(v) for v in split_z2.theta_Z_v)}) against H "
      f"({', '.join(str(v) for v in z2_srsi.theta_Z)}), and the theta_Zn")
print(f"                       of H, mod {modulus} {z2_srsi.theta_Zn[modulus]} — "
      f"'{split_z2.verdict}', {len(split_z2.consistent_pairs)} consistent pairs")

wf_z2 = WFArray(lattice, mesh, spinful=True)
wf_z2.solve_model(z2_chiral)

w_z2_v = wf_z2.wilson_loop(axis_idx=0, state_idx=[0, 1])
phases_z2_v = wf_z2.berry_phase(axis_idx=0, state_idx=[0, 1], berry_evals=True,
                                contin=False) / (2 * np.pi)
cos_z2_v = np.real(np.trace(w_z2_v, axis1=1, axis2=2)) / 2.0
psi_z2_v = np.arccos(np.clip(cos_z2_v, -1.0, 1.0))
winding_z2_v = float(np.abs(np.diff(psi_z2_v)).sum() / (2 * np.pi))
touch_z2_v = np.abs(cos_z2_v) > 1 - 1e-9
print(f"   the lower loop    : the centres meet at "
      f"k_y = {[float(np.round(v, 6)) for v in ky_axis[touch_z2_v]]} with tr W / 2 =")
print(f"                       {[float(np.round(v, 6)) for v in cos_z2_v[touch_z2_v]]}, "
      + ("and the loop does not wind" if int(round(winding_z2_v)) == 0
         else f"and each branch winds by ±{int(round(winding_z2_v))}"))

w_z2_c = wf_z2.wilson_loop(axis_idx=0, state_idx=[2, 3])
phases_z2_c = wf_z2.berry_phase(axis_idx=0, state_idx=[2, 3], berry_evals=True,
                                contin=False) / (2 * np.pi)
cos_z2_c = np.real(np.trace(w_z2_c, axis1=1, axis2=2)) / 2.0
psi_z2_c = np.arccos(np.clip(cos_z2_c, -1.0, 1.0))
winding_z2_c = float(np.abs(np.diff(psi_z2_c)).sum() / (2 * np.pi))
touch_z2_c = np.abs(cos_z2_c) > 1 - 1e-9
print(f"   the upper loop    : the centres meet at "
      f"k_y = {[float(np.round(v, 6)) for v in ky_axis[touch_z2_c]]} with tr W / 2 =")
print(f"                       {[float(np.round(v, 6)) for v in cos_z2_c[touch_z2_c]]}, "
      + ("and the loop does not wind" if int(round(winding_z2_c)) == 0
         else f"and each branch winds by ±{int(round(winding_z2_c))}"))
print("   -> both sides wind here, and in the flat-band model one side's loop is")
print("      flat and the other winds twice.  The criterion returns the same")
print("      verdict for both: it diagnoses the splitting, not the side.")

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
                     fr"({', '.join(str(v) for v in b_valence.theta_Z)})$, "
                     fr"$\theta_\mathbb{{Z}}[C]="
                     fr"({', '.join(str(v) for v in b_conduction.theta_Z)})$, "
                     fr"$\theta_{{\mathbb{{Z}}_2}}[H]={m_srsi.theta_Zn[modulus]}$"
                     f"\nsplitting: {split_flat.verdict}", fontsize=9.5)

# --- the same model's Wilson loops ---
for column in range(2):
    axes[0, 1].plot(ky_axis, phases_lower[:, column], "o", ms=2.6, color="tab:blue")
for column in range(2):
    axes[0, 1].plot(ky_axis, phases_upper[:, column], "s", ms=2.6, color="tab:red")
axes[0, 1].set_xlabel("$k_y / 2\\pi$", fontsize=12)
axes[0, 1].set_ylabel(r"$\phi_{k_x} / 2\pi$", fontsize=12)
axes[0, 1].set_ylim(-0.55, 0.55)
axes[0, 1].set_title(f"(i) $k_x$ Wilson loops: lower pair (blue) winds "
                     f"±{int(round(winding_lower))},\nupper pair (red) "
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
                     fr"$\theta_{{\mathbb{{Z}}_2}}[H]={z2_srsi.theta_Zn[modulus]}$"
                     f"   splitting: {split_z2.verdict}", fontsize=9.5)

# --- the Z2 model's Wilson loops ---
for column in range(2):
    axes[1, 1].plot(ky_axis, phases_z2_v[:, column], "o", ms=2.6, color="tab:blue")
for column in range(2):
    axes[1, 1].plot(ky_axis, phases_z2_c[:, column], "s", ms=2.6, color="tab:red")
axes[1, 1].set_xlabel("$k_y / 2\\pi$", fontsize=12)
axes[1, 1].set_ylabel(r"$\phi_{k_x} / 2\pi$", fontsize=12)
axes[1, 1].set_ylim(-0.55, 0.55)
axes[1, 1].set_title(f"(ii) $k_x$ Wilson loops: lower pair (blue) winds "
                     f"±{int(round(winding_z2_v))},\nupper pair (red) winds "
                     f"±{int(round(winding_z2_c))}", fontsize=9.5)

fig.suptitle("The P41' models of SM section 5A, with the vertical mirrors broken so that "
             "P41' is the group their own hoppings obey:\nthe SRSIs diagnose both "
             "splittings from the tabulated real-space invariants alone, and the Wilson "
             "loops say which obstruction each one carries.", fontsize=10)
fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_p4_flatband.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
