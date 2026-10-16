# Example of a Wyckoff position that moves with the regime, and of a degeneracy
# held by a conserved operator rather than by the little group, using pythtqc.
# The model is Kane and Mele's honeycomb model, PRL 95, 226801 (2005): four
# bands, two occupied, a staggered sublattice potential delta, an intrinsic
# spin-orbit coupling and a Rashba term.  It is pythtb's own
# pythtb.models.kane_mele(delta, t, soc, rashba).
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. delta = 1.0 with rashba on: the group, the content at GM, K and M, and the
#    two decompositions
# 2. delta = 2.5 with rashba on: the same four measurements
# 3. delta = 0, rashba = 0: the same four measurements
# 4. delta = 0, rashba = 0.25: the same four, and K level by level
# 5. the conserved operator, read off H(k)
# 6. what the lost operator does to the degeneracies at GM, M and K
# 7. the deep atomic limit: delta = 50, the weight and the letter
# 8. the bands along GM-K-M-GM, one panel per regime, and the figure written out

# The layer group this script sets aside -- the horizontal mirror, the inversion
# that comes with it, and the symmetry-indicator class of one monolayer -- is the
# subject of examples/tqc_kane_mele_three_presentations.py.

# Author: the pythtqc examples

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb.models import kane_mele

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_counts as fmt
from pythtqc import solve_at
from pythtqc.utils import save_figure

# Elementary band representations print as ρ↑G, the site irrep ρ induced to the
# group, and a difference of them with a minus sign −; a Windows console cannot
# print either by default, so the output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band path and nothing else.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures",
    "tqc_kane_mele_realized_group")
NK_PATH = 61 if QUICK else 301             # band-path density: the figure only

# kane_mele(delta, t, soc, rashba) — the canonical pythtb values with t = -1.  On
# the bipartite honeycomb t -> -t is the sublattice gauge diag(1, -1), so it
# changes neither the spectrum nor Z2.  The K-point gap closes at
# delta = 3*sqrt(3)*soc = 1.559, so delta = 1.0 is Z2-odd and delta = 2.5 is even.
T = -1.0                   # the nearest-neighbour hopping
SOC = 0.3                  # the intrinsic (Kane-Mele) spin-orbit strength

# A pythtb orbital carries no symmetry character, so the A1 here is a declaration.
DECLARE = {0: "A1"}
TYPAT = [0, 1]             # the two honeycomb sites declared inequivalent species

POINTS = ("GM", "K", "M")                                   # the group's k-points
PATH = [[0.0, 0.0], [1 / 3, 2 / 3], [0.5, 0.5], [0.0, 0.0]]     # GM - K - M - GM
NODES = ("GM", "K", "M", "GM")
TICKS = (r"$\Gamma$", "K", "M", r"$\Gamma$")

RASHBA_SWEEP = (0.05, 0.125, 0.25)         # the K-point splitting is measured here
NK_MESH = 6                                # realized_group's default mesh

os.makedirs(OUTDIR, exist_ok=True)

print("Kane-Mele on the honeycomb, PRL 95, 226801 (2005): four bands, two occupied,")
print("a staggered sublattice potential, spin-orbit coupling and a Rashba term.")

########### 1. delta = 1.0, rashba = 0.25 ###########

print("\n1. delta = 1.0, rashba = 0.25")
m1 = kane_mele(1.0, T, SOC, 0.25)

out1 = tqc.realized_group(m1, typat=TYPAT, characters=DECLARE)
model1, g1, asg1, val1, rep1 = out1
sites1 = tqc.wyckoff_sites_tables(g1, asg1.site_positions)   # letter and multiplicity
seats1 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites1))
print(f"   realized group   : #{g1.number_str} {g1.name}, orbitals at Wyckoff "
      f"{' and '.join(seats1)}, {model1.nstate} bands")
print("                      (measured from the Hamiltonian: the staggered potential")
print("                      puts the two honeycomb sites on inequivalent positions)")

t_val1 = tqc.hsp_table(rep1, bands=(0, 1))
t_con1 = tqc.hsp_table(rep1, bands=(2, 3))
d_val1 = tqc.decompose(t_val1, g1, rep=rep1)
d_con1 = tqc.decompose(t_con1, g1, rep=rep1)

# whether an orbital sits at that position, read off the measured Wyckoff letters
seat1 = d_val1.positions[0][0]
where1 = ("a site an orbital sits on" if seat1 in seats1
          else "a position no orbital sits on")
print("   valence content  : "
      + "   ".join(f"{p}: {fmt(t_val1.content[p])}" for p in POINTS))
print(f"      -> {d_val1.solution_strings()[0]}, {where1}")

# no EBR sum exists for the conduction pair; the signed combinations allow
# negative multiplicities
short1 = sorted({tuple(sorted(t)) for t in d_con1.signed_combinations["terms"]},
                key=len)[0]
terms1 = " ".join(f"{'+' if w > 0 else '−'}[{n}@{p.split('(')[0]}]"
                  for w, n, p in short1)
print("   conduction       : "
      + "   ".join(f"{p}: {fmt(t_con1.content[p])}" for p in POINTS))
print(f"      -> {d_con1.verdict}; as a difference {terms1}")

si_val1 = tqc.symmetry_indicator(t_val1, g1, rep=rep1)
print(f"   indicator group  : X_BS(#{si_val1.number_str}) = {si_val1.structure}, "
      "so these indicators do not distinguish")
print(f"                      the valence pair from an atomic limit at {seat1}")

########### 2. delta = 2.5, rashba = 0.25 ###########

print("\n2. delta = 2.5, rashba = 0.25")
m2 = kane_mele(2.5, T, SOC, 0.25)          # delta past the K-point gap closing

out2 = tqc.realized_group(m2, typat=TYPAT, characters=DECLARE)
_m, g2, asg2, val2, rep2 = out2
sites2 = tqc.wyckoff_sites_tables(g2, asg2.site_positions)
seats2 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites2))
print(f"   realized group   : #{g2.number_str} {g2.name}, orbitals at Wyckoff "
      f"{' and '.join(seats2)}")

t_val2 = tqc.hsp_table(rep2, bands=(0, 1))
t_con2 = tqc.hsp_table(rep2, bands=(2, 3))
d_val2 = tqc.decompose(t_val2, g2, rep=rep2)
d_con2 = tqc.decompose(t_con2, g2, rep=rep2)

seat2 = d_val2.positions[0][0]
where2 = ("a site an orbital sits on" if seat2 in seats2
          else "a position no orbital sits on")
print("   valence content  : "
      + "   ".join(f"{p}: {fmt(t_val2.content[p])}" for p in POINTS))
print(f"      -> {d_val2.solution_strings()[0]}, {where2}")
print("   conduction       : "
      + "   ".join(f"{p}: {fmt(t_con2.content[p])}" for p in POINTS))
print(f"      -> {d_con2.solution_strings()[0]}")
print(f"   the group is #{g2.number_str} in both regimes and the valence pair's "
      "position")
print(f"   moves from {seat1} to {seat2} with the regime")

########### 3. delta = 0, rashba = 0 ###########

print("\n3. delta = 0, rashba = 0")
m3 = kane_mele(0.0, T, SOC, 0.0)           # no staggering, no Rashba

out3 = tqc.realized_group(m3, characters=DECLARE)     # one species: delta = 0
_m, g3, asg3, val3, rep3 = out3
sites3 = tqc.wyckoff_sites_tables(g3, asg3.site_positions)
seats3 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites3))
print(f"   realized group   : #{g3.number_str} {g3.name}, orbitals at Wyckoff "
      f"{' and '.join(seats3)}")
print("                      (delta = 0 puts the two honeycomb sites on one orbit, and")
print(f"                      the group is the one the staggered potential lowered to "
      f"#{g1.number_str})")

t_val3 = tqc.hsp_table(rep3, bands=(0, 1))
t_con3 = tqc.hsp_table(rep3, bands=(2, 3))
d_val3 = tqc.decompose(t_val3, g3, rep=rep3)
d_con3 = tqc.decompose(t_con3, g3, rep=rep3)

seat3 = d_val3.positions[0][0]
where3 = ("a site an orbital sits on" if seat3 in seats3
          else "a position no orbital sits on")
short3 = sorted({tuple(sorted(t)) for t in d_con3.signed_combinations["terms"]},
                key=len)[0]
terms3 = " ".join(f"{'+' if w > 0 else '−'}[{n}@{p.split('(')[0]}]"
                  for w, n, p in short3)
print("   valence content  : "
      + "   ".join(f"{p}: {fmt(t_val3.content[p])}" for p in POINTS))
print(f"      -> {d_val3.solution_strings()[0]}, {where3}")
print("   conduction       : "
      + "   ".join(f"{p}: {fmt(t_con3.content[p])}" for p in POINTS))
print(f"      -> {d_con3.verdict}; as a difference {terms3}")

########### 4. delta = 0, rashba = 0.25 ###########

print("\n4. delta = 0, rashba = 0.25")
m4 = kane_mele(0.0, T, SOC, 0.25)          # the same crystal, Rashba turned on

out4 = tqc.realized_group(m4, characters=DECLARE)
_m, g4, asg4, val4, rep4 = out4
sites4 = tqc.wyckoff_sites_tables(g4, asg4.site_positions)
seats4 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites4))
print(f"   realized group   : #{g4.number_str} {g4.name}, orbitals at Wyckoff "
      f"{' and '.join(seats4)}")

t_val4 = tqc.hsp_table(rep4, bands=(0, 1))
t_con4 = tqc.hsp_table(rep4, bands=(2, 3))
d_val4 = tqc.decompose(t_val4, g4, rep=rep4)
d_con4 = tqc.decompose(t_con4, g4, rep=rep4)

short4 = sorted({tuple(sorted(t)) for t in d_con4.signed_combinations["terms"]},
                key=len)[0]
terms4 = " ".join(f"{'+' if w > 0 else '−'}[{n}@{p.split('(')[0]}]"
                  for w, n, p in short4)
print("   valence content  : "
      + "   ".join(f"{p}: {fmt(t_val4.content[p])}" for p in POINTS))
print(f"      -> {d_val4.solution_strings()[0]}")
print("   conduction       : "
      + "   ".join(f"{p}: {fmt(t_con4.content[p])}" for p in POINTS))
print(f"      -> {d_con4.verdict}; as a difference {terms4}")

# ... and at K level by level: multiplets groups the energies into degenerate
# levels, and corep_content names the corep of each whole level.
hsp4 = tqc.high_symmetry_points(g4)
states_k4 = solve_at(m4, hsp4["K"])
energies_k4 = np.asarray(states_k4[0], dtype=float).reshape(-1)
multiplets_k4 = tqc.multiplets(energies_k4, degeneracy_tol=1e-6)
levels_k4 = []
for multiplet in multiplets_k4:
    content = tqc.corep_content(rep4, hsp4["K"], states=states_k4, bands=list(multiplet))
    levels_k4.append((content, float(np.mean(energies_k4[list(multiplet)])),
                      len(multiplet)))
print("   K level by level : "
      + "   ".join(f"{fmt(c.counts)} @ {e:+.6f} (dim {n})"
                   for c, e, n in levels_k4[:2]))
print("                      "
      + "   ".join(f"{fmt(c.counts)} @ {e:+.6f} (dim {n})"
                   for c, e, n in levels_k4[2:]))

########### 5. the conserved operator, read off H(k) ###########

print(f"\n5. the conserved operator, read off H(k) on a {NK_MESH}x{NK_MESH} mesh "
      "of in-plane k")

# realized_group tests each operation g on H(k) over its 6 x 6 mesh, solving
# X H(k) = H(gk) X at every k at once.  For the identity that is X H(k) = H(k) X,
# whose solutions are the commutant of the Hamiltonian, and conserved["dim"]
# counts them beyond the identity; the call is told nothing about spin.  The
# lines below say which operator it is: s_z commutes with H(k) at every k.
kx = np.arange(NK_MESH) / NK_MESH
ks = np.stack(np.meshgrid(kx, kx, indexing="ij"), -1).reshape(-1, 2)
sz = np.kron(np.eye(2), np.diag([1.0, -1.0]))       # s_z on the four basis states

print(f"   delta 0, rashba 0     : the commutant of H(k) is the identity and "
      f"{out3.conserved['dim']} more")
h3 = np.asarray(m3.hamiltonian(k_pts=ks, flatten_spin_axis=True))
comm3 = float(np.abs(sz @ h3 - h3 @ sz).max())
print("                           operator, and s_z commutes with H(k) at every k,")
print(f"                           max |[s_z, H(k)]| = {comm3:.6f}: that operator is s_z")

h4 = np.asarray(m4.hamiltonian(k_pts=ks, flatten_spin_axis=True))
comm4 = float(np.abs(sz @ h4 - h4 @ sz).max())
print("   delta 0, rashba 0.25  : the commutant is the identity and "
      f"{out4.conserved['dim']} more,")
print(f"                           the Rashba term mixing the spins: "
      f"max |[s_z, H(k)]| = {comm4:.6f}")

print("   delta 1.0, rashba 0.25: the commutant is the identity and "
      f"{out1.conserved['dim']} more")

########### 6. what the lost operator does to the degeneracies ###########

print("\n6. what the lost operator does to the degeneracies")
print(f"   the group is #{g3.number_str} at both rashba values and the content "
      "above is the same")
print("   entry for entry; what differs is the commutant")

shapes = []
for point in POINTS:
    e_off = np.asarray(solve_at(m3, hsp4[point])[0], dtype=float).reshape(-1)
    e_on = np.asarray(solve_at(m4, hsp4[point])[0], dtype=float).reshape(-1)
    off = tuple(len(b) for b in tqc.multiplets(e_off, degeneracy_tol=1e-6))
    on = tuple(len(b) for b in tqc.multiplets(e_on, degeneracy_tol=1e-6))
    shapes.append(f"{point} {off} -> {on}")
print("   multiplets       : " + "   ".join(shapes))
print("                      at rashba = 0 and at rashba = 0.25")

# the two levels the conduction pair splits into, character row by character row;
# every character of either level here is real or purely imaginary.
(c_lo, e_lo, n_lo), (c_mid, e_mid, _n2), (c_hi, e_hi, _n3) = levels_k4
chi_mid = np.asarray(c_mid.table.characters).ravel()
chi_hi = np.asarray(c_hi.table.characters).ravel()
row_mid = ", ".join(f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else
                    (f"{z.imag:+.0f}i" if abs(z.real) < 1e-9 else f"{z:+.3f}")
                    for z in chi_mid)
row_hi = ", ".join(f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else
                   (f"{z.imag:+.0f}i" if abs(z.real) < 1e-9 else f"{z:+.3f}")
                   for z in chi_hi)
print("   the two levels K splits into, over the operations "
      f"({', '.join(out4.operations[i].name for i in c_mid.table.columns)}):")
print(f"      {fmt(c_mid.counts)} @ {e_mid:+.6f} : ({row_mid})")
print(f"      {fmt(c_hi.counts)} @ {e_hi:+.6f} : ({row_hi})")
print("      one row is the conjugate of the other: two distinct one-dimensional")
print("      coreps, so the little group holds neither of them to the other, and what")
print("      held them together at rashba = 0 was s_z")
print("      (BCS's labels attach to -k, so an operation taking k to -k interchanges")
print("      the two names; the characters are the observable)")
print(f"   {fmt(c_lo.counts)} is a corep of dimension {n_lo} and stays one level "
      "whatever the commutant does")

splits = []
for rashba in RASHBA_SWEEP:
    e = np.asarray(solve_at(kane_mele(0.0, T, SOC, rashba), hsp4["K"])[0],
                   dtype=float).reshape(-1)
    splits.append((rashba, float(e[3] - e[2])))
print("   the splitting    : rashba "
      + ", ".join(f"{r} -> {s:.6f}" for r, s in splits))
print("                      split / rashba = "
      + " / ".join(f"{s / r:.6f}" for r, s in splits))

########### 7. the deep atomic limit ###########

print("\n7. the deep atomic limit: delta = 50")
deep = kane_mele(50.0, T, SOC, 0.25)       # the even regime, delta driven to 50
_e, vectors = deep.solve_ham(k_pts=deep.k_uniform_mesh([12, 12]), return_eigvecs=True,
                             flatten_spin_axis=False)
weight = (np.abs(vectors[:, :2]) ** 2).sum(axis=(0, 1, -1))
weight = weight / weight.sum()             # the valence pair's weight per orbital

out_deep = tqc.realized_group(deep, typat=TYPAT, characters=DECLARE)
_m, g_deep, asg_deep, _v, rep_deep = out_deep
sites_deep = tqc.wyckoff_sites_tables(g_deep, asg_deep.site_positions)
t_deep = tqc.hsp_table(rep_deep, bands=2)
d_deep = tqc.decompose(t_deep, g_deep, rep=rep_deep)
# the assignment's site_of_orbital names the site each orbital sits on
heavy = sites_deep[int(asg_deep.site_of_orbital[int(np.argmax(weight))])]
light = sites_deep[int(asg_deep.site_of_orbital[int(np.argmin(weight))])]
print(f"   the valence pair carries {100 * weight.max():.2f} % of its weight on the "
      f"orbital at Wyckoff {heavy.multiplicity}{heavy.letter},")
print(f"   {100 * weight.min():.2f} % on the one at "
      f"{light.multiplicity}{light.letter}")
print(f"   decompose        : {d_deep.solution_strings()[0]} — the site the "
      "electrons are on")

########### 8. the bands along GM-K-M-GM ###########

print("\n8. the bands along GM-K-M-GM, one panel per regime")
fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.6), sharey=False)

# --- delta = 1.0, rashba = 0.25 ---
kv, kd, kn = m1.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m1.solve_ham(kv))
for i in range(4):
    axes[0].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0].axvline(x, color="0.75", lw=0.7)
# the valence pair is Kramers-degenerate at every one of these points, so one
# offset below it clears both bands; the conduction labels go above
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[0].annotate(fmt(t_val1.content[point]), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    axes[0].annotate(fmt(t_con1.content[point]), xy=(x, bands[j, 2]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:red")
axes[0].set_xticks(kn)
axes[0].set_xticklabels(TICKS, fontsize=13)
axes[0].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[0].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[0].set_ylabel("$E$", fontsize=16)
axes[0].set_title(f"delta = 1.0, rashba = 0.25\n#{g1.number_str} {g1.name}\n"
                  f"valence at {seat1}   |   conduction: no EBR sum", fontsize=9.5)

# --- delta = 2.5, rashba = 0.25 ---
kv, kd, kn = m2.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m2.solve_ham(kv))
for i in range(4):
    axes[1].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[1].axvline(x, color="0.75", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[1].annotate(fmt(t_val2.content[point]), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    axes[1].annotate(fmt(t_con2.content[point]), xy=(x, bands[j, 2]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:red")
axes[1].set_xticks(kn)
axes[1].set_xticklabels(TICKS, fontsize=13)
axes[1].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[1].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[1].set_title(f"delta = 2.5, rashba = 0.25\n#{g2.number_str} {g2.name}\n"
                  f"valence at {seat2}   |   conduction at "
                  f"{d_con2.positions[0][0]}", fontsize=9.5)

# --- delta = 0, rashba = 0 ---
kv, kd, kn = m3.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m3.solve_ham(kv))
for i in range(4):
    axes[2].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[2].axvline(x, color="0.75", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[2].annotate(fmt(t_val3.content[point]), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    axes[2].annotate(fmt(t_con3.content[point]), xy=(x, bands[j, 2]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:red")
axes[2].set_xticks(kn)
axes[2].set_xticklabels(TICKS, fontsize=13)
axes[2].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[2].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[2].set_title(f"delta = 0, rashba = 0\n#{g3.number_str} {g3.name}\n"
                  f"valence at {seat3}   |   conduction: no EBR sum", fontsize=9.5)

# --- delta = 0, rashba = 0.25: K resolved level by level ---
kv, kd, kn = m4.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m4.solve_ham(kv))
for i in range(4):
    axes[3].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[3].axvline(x, color="0.75", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[3].annotate(fmt(t_val4.content[point]), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    if point != "K":                     # K is resolved level by level instead
        axes[3].annotate(fmt(t_con4.content[point]), xy=(x, bands[j, 2]),
                         xytext=(0, 8), textcoords="offset points", ha="center",
                         fontsize=7.5, color="tab:red")
x_k = kn[NODES.index("K")]
axes[3].annotate(fmt(c_hi.counts), xy=(x_k, e_hi), xytext=(0, 7),
                 textcoords="offset points", ha="center", fontsize=7.5, color="tab:red")
axes[3].annotate(fmt(c_mid.counts), xy=(x_k, e_mid), xytext=(0, -14),
                 textcoords="offset points", ha="center", fontsize=7.5, color="tab:red")
axes[3].annotate("", xy=(x_k, e_mid), xytext=(x_k, e_hi),
                 arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.3))
axes[3].annotate(f"$\\Delta$ = {e_hi - e_mid:.3f}", xy=(x_k, e_mid), xytext=(0, -28),
                 textcoords="offset points", ha="center", va="top", fontsize=8,
                 color="0.25")
axes[3].set_xticks(kn)
axes[3].set_xticklabels(TICKS, fontsize=13)
axes[3].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[3].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[3].set_title(f"delta = 0, rashba = 0.25\n#{g4.number_str} {g4.name}\n"
                  f"{fmt(c_mid.counts)} and {fmt(c_hi.counts)} split at K",
                  fontsize=9.5)

fig.suptitle("Kane-Mele on the honeycomb: the four bands along GM-K-M-GM, the valence "
             "pair in blue and the conduction pair in red, with the measured corep "
             "content at each node.", fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.955))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "km_bands_and_coreps.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
