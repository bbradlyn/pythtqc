# Example of a Wyckoff position that moves with the regime, and of a degeneracy
# held by a conserved operator rather than by the little group, using pythtqc.
# The model is Kane and Mele's honeycomb model, PRL 95, 226801 (2005): four
# bands, two occupied, a staggered sublattice potential delta, an intrinsic
# spin-orbit coupling and a Rashba term.  It is pythtb's own
# pythtb.models.kane_mele(delta, t, soc, rashba).
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. delta = 1.0 with rashba on: the group, the content at GM, K and M of the
#    two occupied bands and of the two above them, the atomic limits of each,
#    and the symmetry-indicator group
# 2. delta = 2.5 with rashba on: the group, the two contents and their atomic
#    limits
# 3. delta = 0, rashba = 0: the same
# 4. delta = 0, rashba = 0.25: the same, and the coreps at K level by level
# 5. the conserved operators, read off H(k)
# 6. what the lost operator does to the degeneracies at GM, K and M
# 7. the deep atomic limit: delta = 50, the weight on each orbital and the
#    Wyckoff position of the atomic limit
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
# group; a Windows console cannot print ↑ by default, so the output is UTF-8.
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
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def listed(words):
    """Words as a sentence lists them: a, b and c."""
    words = [str(word) for word in words]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def by_point(table):
    """A band set's content at GM, K and M, each as format_counts writes it."""
    return "   ".join(f"{p}: {fmt(table.content[p])}" for p in POINTS)




def signed_sum(d):
    """The shortest sum with negative multiplicities that carries the content, in lines."""
    shortest = sorted({tuple(sorted(t)) for t in d.signed_combinations["terms"]},
                      key=len)[0]
    lines = []
    for m, name, where in shortest:
        term = f"{m} x {name}@{where}"
        if not lines:
            lines.append(term)
        elif len(lines[-1]) + len(term) + 3 > 64:
            lines.append(f"+ {term}")
        else:
            lines[-1] += f" + {term}"
    return lines


def limits(d, where=""):
    """The atomic limits of a band set, or a signed sum where there are none."""
    if d.verdict == "none":
        row("atomic limits", "none")
        row("a signed sum", *signed_sum(d))
        return
    sums = list(d.solution_strings())
    sums[0] += f", {where}" if where else ""
    row("atomic limit" if len(sums) == 1 else "atomic limits", *sums)


def indicator(si):
    """The symmetry-indicator class, as a label and a value, or that the group has none."""
    if si.verdict == "trivial indicator group":
        return "indicator group", "trivial"
    return "indicator class", f"({', '.join(str(v) for v in si.values)}) in {si.structure}, order {si.order}"


########### 1. delta = 1.0, rashba = 0.25 ###########

print("\n1. delta = 1.0, rashba = 0.25")
m1 = kane_mele(1.0, T, SOC, 0.25)

out1 = tqc.realized_group(m1, typat=TYPAT, characters=DECLARE)
model1, g1, asg1, val1, rep1 = out1
sites1 = tqc.wyckoff_sites_tables(g1, asg1.site_positions)   # letter and multiplicity
seats1 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites1))
row("realized group", f"#{g1.number_str} {g1.name}, orbitals at Wyckoff "
    f"{' and '.join(seats1)}, {model1.nstate} bands")

t_val1 = tqc.hsp_table(rep1, bands=(0, 1))
t_con1 = tqc.hsp_table(rep1, bands=(2, 3))
d_val1 = tqc.decompose(t_val1)
d_con1 = tqc.decompose(t_con1)

# whether an orbital sits at that position, read off the measured Wyckoff letters
seat1 = d_val1.positions[0][0]
where1 = ("a site an orbital sits on" if seat1 in seats1
          else "a position no orbital sits on")
row("bands 0 and 1", by_point(t_val1))
limits(d_val1, where1)

# no sum of elementary band representations carries the content of bands 2 and
# 3; the signed combinations allow negative multiplicities, and the shortest of
# them is printed
row("bands 2 and 3", by_point(t_con1))
limits(d_con1)

si_val1 = tqc.symmetry_indicator(t_val1)
row(*indicator(si_val1))
print("   the staggered potential makes the two honeycomb sites inequivalent, and no")
print(f"   symmetry indicator tells bands 0 and 1 from an atomic limit at {seat1}")

########### 2. delta = 2.5, rashba = 0.25 ###########

print("\n2. delta = 2.5, rashba = 0.25")
m2 = kane_mele(2.5, T, SOC, 0.25)          # delta past the K-point gap closing

out2 = tqc.realized_group(m2, typat=TYPAT, characters=DECLARE)
_m, g2, asg2, val2, rep2 = out2
sites2 = tqc.wyckoff_sites_tables(g2, asg2.site_positions)
seats2 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites2))
row("realized group", f"#{g2.number_str} {g2.name}, orbitals at Wyckoff "
    f"{' and '.join(seats2)}")

t_val2 = tqc.hsp_table(rep2, bands=(0, 1))
t_con2 = tqc.hsp_table(rep2, bands=(2, 3))
d_val2 = tqc.decompose(t_val2)
d_con2 = tqc.decompose(t_con2)

seat2 = d_val2.positions[0][0]
where2 = ("a site an orbital sits on" if seat2 in seats2
          else "a position no orbital sits on")
row("bands 0 and 1", by_point(t_val2))
limits(d_val2, where2)
row("bands 2 and 3", by_point(t_con2))
limits(d_con2)
print(f"   the group is #{g2.number_str} in both regimes, and the Wyckoff position of "
      "bands 0 and 1")
print(f"   moves from {seat1} to {seat2}")

########### 3. delta = 0, rashba = 0 ###########

print("\n3. delta = 0, rashba = 0")
m3 = kane_mele(0.0, T, SOC, 0.0)           # no staggering, no Rashba

out3 = tqc.realized_group(m3, characters=DECLARE)     # one species: delta = 0
_m, g3, asg3, val3, rep3 = out3
sites3 = tqc.wyckoff_sites_tables(g3, asg3.site_positions)
seats3 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites3))
row("realized group", f"#{g3.number_str} {g3.name}, orbitals at Wyckoff "
    f"{' and '.join(seats3)}")

t_val3 = tqc.hsp_table(rep3, bands=(0, 1))
t_con3 = tqc.hsp_table(rep3, bands=(2, 3))
d_val3 = tqc.decompose(t_val3)
d_con3 = tqc.decompose(t_con3)

seat3 = d_val3.positions[0][0]
where3 = ("a site an orbital sits on" if seat3 in seats3
          else "a position no orbital sits on")
row("bands 0 and 1", by_point(t_val3))
limits(d_val3, where3)
row("bands 2 and 3", by_point(t_con3))
limits(d_con3)
print("   at delta = 0 the two honeycomb sites are one orbit, and the staggered potential")
print(f"   lowers this group to #{g1.number_str}")

########### 4. delta = 0, rashba = 0.25 ###########

print("\n4. delta = 0, rashba = 0.25")
m4 = kane_mele(0.0, T, SOC, 0.25)          # the same crystal, Rashba turned on

out4 = tqc.realized_group(m4, characters=DECLARE)
_m, g4, asg4, val4, rep4 = out4
sites4 = tqc.wyckoff_sites_tables(g4, asg4.site_positions)
seats4 = list(dict.fromkeys(f"{s.multiplicity}{s.letter}" for s in sites4))
row("realized group", f"#{g4.number_str} {g4.name}, orbitals at Wyckoff "
    f"{' and '.join(seats4)}")

t_val4 = tqc.hsp_table(rep4, bands=(0, 1))
t_con4 = tqc.hsp_table(rep4, bands=(2, 3))
d_val4 = tqc.decompose(t_val4)
d_con4 = tqc.decompose(t_con4)

row("bands 0 and 1", by_point(t_val4))
limits(d_val4)
row("bands 2 and 3", by_point(t_con4))
limits(d_con4)

# the coreps at K level by level: multiplets groups the energies into degenerate
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
row("coreps at K", *(f"{fmt(c.counts)} at E = {e:+.6f}, dimension {n}"
                     for c, e, n in levels_k4))

########### 5. the conserved operators, read off H(k) ###########

print(f"\n5. the conserved operators, read off H(k) on a {NK_MESH}x{NK_MESH} mesh "
      "of in-plane k")

# realized_group tests each operation g on H(k) over its 6 x 6 mesh, solving
# X H(k) = H(gk) X at every k at once.  For the identity that is X H(k) = H(k) X,
# whose solutions are the operators the Hamiltonian conserves, and
# conserved["dim"] counts them beyond the identity; the call is told nothing
# about spin.  The commutator with s_z says which operator it is.
kx = np.arange(NK_MESH) / NK_MESH
ks = np.stack(np.meshgrid(kx, kx, indexing="ij"), -1).reshape(-1, 2)
sz = np.kron(np.eye(2), np.diag([1.0, -1.0]))       # s_z on the four basis states

h3 = np.asarray(m3.hamiltonian(k_pts=ks, flatten_spin_axis=True))
comm3 = float(np.abs(sz @ h3 - h3 @ sz).max())
h4 = np.asarray(m4.hamiltonian(k_pts=ks, flatten_spin_axis=True))
comm4 = float(np.abs(sz @ h4 - h4 @ sz).max())
row("delta, rashba", f"{'beyond the identity':<22}max |[s_z, H(k)]|")
row("0, 0", f"{out3.conserved['dim']:<22}{comm3:.6f}")
row("0, 0.25", f"{out4.conserved['dim']:<22}{comm4:.6f}")
row("1.0, 0.25", f"{out1.conserved['dim']:<22}not measured")
print("   at rashba = 0 the one conserved operator beyond the identity is s_z, whose")
print("   conservation the Rashba term breaks")

########### 6. what the lost operator does to the degeneracies ###########

print("\n6. what the lost operator does to the degeneracies")

# the dimensions of the levels at each point, from the bottom, without and with
# the Rashba term
shapes = []
for point in POINTS:
    e_off = np.asarray(solve_at(m3, hsp4[point])[0], dtype=float).reshape(-1)
    e_on = np.asarray(solve_at(m4, hsp4[point])[0], dtype=float).reshape(-1)
    off = tuple(len(b) for b in tqc.multiplets(e_off, degeneracy_tol=1e-6))
    on = tuple(len(b) for b in tqc.multiplets(e_on, degeneracy_tol=1e-6))
    shapes.append((point, " + ".join(str(n) for n in off),
                   " + ".join(str(n) for n in on)))
width = max(len(off) for _point, off, _on in shapes) + 3
row("rashba", f"{'0':<{width}}0.25")
for point, off, on in shapes:
    row(f"levels at {point}", f"{off:<{width}}{on}")

# the two levels bands 2 and 3 split into at K, character row by character row;
# every character of either level here is real or purely imaginary.  The tables
# attach these labels to -k, so an operation taking k to -k interchanges the two
# names; the characters are what is measured.
(c_lo, e_lo, n_lo), (c_mid, e_mid, _n2), (c_hi, e_hi, _n3) = levels_k4
chi_mid = np.asarray(c_mid.table.characters).ravel()
chi_hi = np.asarray(c_hi.table.characters).ravel()
row_mid = [f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else
           (f"{z.imag:+.0f}i" if abs(z.real) < 1e-9 else f"{z:+.3f}")
           for z in chi_mid]
row_hi = [f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else
          (f"{z.imag:+.0f}i" if abs(z.real) < 1e-9 else f"{z:+.3f}")
          for z in chi_hi]
names = [out4.operations[i].name for i in c_mid.table.columns]
widths = [max(len(a), len(b), len(c)) for a, b, c in zip(names, row_mid, row_hi)]
for label, cells in (("operations at K", names), (fmt(c_mid.counts), row_mid),
                     (fmt(c_hi.counts), row_hi)):
    row(label, "   ".join(f"{cell:<{w}}" for cell, w in zip(cells, widths)).rstrip())

splits = []
for rashba in RASHBA_SWEEP:
    e = np.asarray(solve_at(kane_mele(0.0, T, SOC, rashba), hsp4["K"])[0],
                   dtype=float).reshape(-1)
    splits.append((rashba, float(e[3] - e[2])))
row("bands 2 and 3", f"split at K by {listed(f'{s:.6f}' for _r, s in splits)}",
    f"at rashba = {listed(rashba for rashba, _s in splits)}")
row("split / rashba", listed(f"{s / r:.6f}" for r, s in splits))
print(f"   the {fmt(c_mid.counts)} and {fmt(c_hi.counts)} rows are complex conjugates, "
      "two distinct one-dimensional coreps")
print(f"   that s_z holds together at rashba = 0, while {fmt(c_lo.counts)}, of dimension "
      f"{n_lo}, stays one level")

########### 7. the deep atomic limit ###########

print("\n7. the deep atomic limit: delta = 50")
deep = kane_mele(50.0, T, SOC, 0.25)       # the even regime, delta driven to 50
_e, vectors = deep.solve_ham(k_pts=deep.k_uniform_mesh([12, 12]), return_eigvecs=True,
                             flatten_spin_axis=False)
weight = (np.abs(vectors[:, :2]) ** 2).sum(axis=(0, 1, -1))
weight = weight / weight.sum()             # the weight of bands 0 and 1 per orbital

out_deep = tqc.realized_group(deep, typat=TYPAT, characters=DECLARE)
_m, g_deep, asg_deep, _v, rep_deep = out_deep
sites_deep = tqc.wyckoff_sites_tables(g_deep, asg_deep.site_positions)
t_deep = tqc.hsp_table(rep_deep, bands=2)
d_deep = tqc.decompose(t_deep)
# the assignment's site_of_orbital names the site each orbital sits on
heavy = sites_deep[int(asg_deep.site_of_orbital[int(np.argmax(weight))])]
light = sites_deep[int(asg_deep.site_of_orbital[int(np.argmin(weight))])]
row("bands 0 and 1", by_point(t_deep))
limits(d_deep)
row("orbital weight", f"{100 * weight.max():.2f} % on the orbital at Wyckoff "
    f"{heavy.multiplicity}{heavy.letter}, {100 * weight.min():.2f} % on the one at "
    f"{light.multiplicity}{light.letter}")
print("   the atomic limit of bands 0 and 1 sits on the orbital that carries their weight")

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
