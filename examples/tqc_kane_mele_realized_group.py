# Example of two bands whose decomposition into elementary band representations
# changes Wyckoff position from the Z2-odd regime to the Z2-even one, and of a
# degeneracy that a conserved operator protects and the little group does not,
# using pythTQC.
# The model is the Kane-Mele model on the honeycomb lattice, PRL 95, 226801
# (2005), which is pythtb's own pythtb.models.kane_mele(delta, t, soc, rashba).
# It has four bands, two of them occupied, a staggered sublattice potential
# delta, an intrinsic spin-orbit coupling and a Rashba term.  With
# rashba = 0.25, delta = 1.0 is in the Z2-odd regime and delta = 2.5 in the
# Z2-even regime.
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. delta = 1.0, rashba = 0.25: the group, the corep multiplicities at GM, K
#    and M of the two occupied bands and of the two above them, the
#    decompositions of each into elementary band representations, and the
#    symmetry-indicator group
# 2. delta = 2.5, rashba = 0.25: the group, the corep multiplicities of both
#    pairs of bands, and their decompositions
# 3. delta = 0, rashba = 0: the same
# 4. delta = 0, rashba = 0.25: the same, and the coreps at K level by level
# 5. the operators that commute with H(k) on a 6 x 6 mesh of k
# 6. what the loss of one such operator does to the degeneracies at GM, K and M
# 7. the deep atomic limit, delta = 50: the weight of the occupied bands on each
#    orbital, and the Wyckoff position of their decomposition
# 8. the bands along GM-K-M-GM, one panel per parameter set, written to a figure

# The groups read here leave out the horizontal mirror.  The layer group, with
# that mirror and the inversion it brings, is the subject of
# examples/tqc_kane_mele_three_presentations.py, with the symmetry-indicator
# class of one monolayer.

# Author: the pythTQC examples

import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb.models import kane_mele

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_coreps as fmt
from pythtqc import solve_at
from pythtqc.utils import save_figure

# Write the output as UTF-8, which a Windows console does not use by default.
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

# kane_mele(delta, t, soc, rashba), with t = -1.  On the bipartite honeycomb
# lattice t -> -t is the sublattice gauge diag(1, -1), with a spin rotation by
# pi about z where the Rashba term is on, so it changes neither the spectrum
# nor Z2.  Without the Rashba term the gap at K closes at
# delta = 3*sqrt(3)*soc = 1.559.  With rashba = 0.25 it closes near
# delta = 1.47.  So delta = 1.0 is Z2-odd and delta = 2.5 is Z2-even.
T = -1.0                   # the nearest-neighbor hopping
SOC = 0.3                  # the intrinsic (Kane-Mele) spin-orbit strength

# A pythtb orbital has no stated site irrep, so the A1 here is declared.
DECLARE = {0: "A1"}
TYPAT = [0, 1]             # the two honeycomb sites declared inequivalent species

POINTS = ("GM", "K", "M")                                   # the high-symmetry points
PATH = [[0.0, 0.0], [1 / 3, 2 / 3], [0.5, 0.5], [0.0, 0.0]]     # GM - K - M - GM
NODES = ("GM", "K", "M", "GM")
TICKS = (r"$\Gamma$", "K", "M", r"$\Gamma$")

RASHBA_SWEEP = (0.05, 0.125, 0.25)         # the K-point splitting is measured here
NK_MESH = 6                                # realized_group's default mesh

os.makedirs(OUTDIR, exist_ok=True)

print("The Kane-Mele model on the honeycomb lattice, PRL 95, 226801 (2005): four bands,")
print("two occupied, a staggered sublattice potential, spin-orbit coupling and a Rashba term.")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One printed row: a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def listed(words):
    """Words as a sentence lists them: a, b and c."""
    words = [str(word) for word in words]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def coreps_of(table, point):
    """The coreps a table's bands carry at one point, as {corep: multiplicity}."""
    return Counter(name for m in table.multiplets[point] for name in m.corep_names)


def by_point(table):
    """The corep multiplicities of a band set at GM, K and M, as format_coreps writes them."""
    return "   ".join(f"{p}: {fmt(coreps_of(table, p))}" for p in POINTS)




def signed_sum(d):
    """The shortest signed sum of elementary band representations with the bands' symmetry
    data vector, as lines."""
    shortest = min(d.signed_combinations.sums, key=len)
    lines = []
    for t in sorted(shortest, key=lambda t: (t.multiplicity, t.irrep, t.wyckoff)):
        m = "" if t.multiplicity == 1 else f"{t.multiplicity}x"
        term = f"{m}{t.irrep} at {t.wyckoff} ({t.site_symmetry})"
        if not lines:
            lines.append(term)
        elif len(lines[-1]) + len(term) + 3 > 64:
            lines.append(f"+ {term}")
        else:
            lines[-1] += f" + {term}"
    return lines


def limits(d, where=""):
    """The decompositions of a band set, or the shortest signed sum where there are none."""
    if not d.decomposable:
        row("decompositions", "none")
        row("a signed sum", *signed_sum(d))
        return
    sums = [s.label for s in d.candidate_sums]
    sums[0] += f", {where}" if where else ""
    row("decomposition" if len(sums) == 1 else "decompositions", *sums)


def indicator(si):
    """The symmetry-indicator class, as a label and a value, or that the group has none."""
    if not si.divisors:
        return "indicator group", "trivial"
    return "indicator class", f"({', '.join(str(v) for v in si.values)}) in {si.indicator_group}, order {si.order}"


########### 1. delta = 1.0, rashba = 0.25 ###########

print("\n1. delta = 1.0, rashba = 0.25")
m1 = kane_mele(1.0, T, SOC, 0.25)

r1 = tqc.realized_group(m1, typat=TYPAT, characters=DECLARE)
g1 = r1.group
seats1 = list(dict.fromkeys(o.wyckoff for o in r1.wyckoff_orbits))   # each orbit's Wyckoff position
row("realized group", f"#{g1.number_str} {g1.name}, orbitals at Wyckoff "
    f"{' and '.join(seats1)}, {r1.model.nstate} bands")

t_val1 = tqc.hsp_table(r1, bands=(0, 1))
t_con1 = tqc.hsp_table(r1, bands=(2, 3))
d_val1 = tqc.decompose(t_val1)
d_con1 = tqc.decompose(t_con1)

# Whether an orbital sits at the Wyckoff position of the first decomposition.
seat1 = min(t.wyckoff for t in d_val1.candidate_sums[0])
where1 = ("a site an orbital sits on" if seat1 in seats1
          else "a position no orbital sits on")
row("bands 0 and 1", by_point(t_val1))
limits(d_val1, where1)

# The symmetry data vector of bands 2 and 3 has no decomposition.  A signed sum
# of elementary band representations allows negative multiplicities, and the
# shortest one is printed.
row("bands 2 and 3", by_point(t_con1))
limits(d_con1)

si_val1 = tqc.symmetry_indicator(t_val1)
row(*indicator(si_val1))
print("   the indicator group is trivial, so no symmetry indicator distinguishes")
print(f"   bands 0 and 1 from an atomic limit at {seat1}")

########### 2. delta = 2.5, rashba = 0.25 ###########

print("\n2. delta = 2.5, rashba = 0.25")
m2 = kane_mele(2.5, T, SOC, 0.25)          # delta past the K-point gap closing

r2 = tqc.realized_group(m2, typat=TYPAT, characters=DECLARE)
g2 = r2.group
seats2 = list(dict.fromkeys(o.wyckoff for o in r2.wyckoff_orbits))
row("realized group", f"#{g2.number_str} {g2.name}, orbitals at Wyckoff "
    f"{' and '.join(seats2)}")

t_val2 = tqc.hsp_table(r2, bands=(0, 1))
t_con2 = tqc.hsp_table(r2, bands=(2, 3))
d_val2 = tqc.decompose(t_val2)
d_con2 = tqc.decompose(t_con2)

seat2 = min(t.wyckoff for t in d_val2.candidate_sums[0])
where2 = ("a site an orbital sits on" if seat2 in seats2
          else "a position no orbital sits on")
row("bands 0 and 1", by_point(t_val2))
limits(d_val2, where2)
row("bands 2 and 3", by_point(t_con2))
limits(d_con2)
print(f"   the group is #{g2.number_str} in both regimes, and the Wyckoff position of "
      "the")
print(f"   decomposition of bands 0 and 1 changes from {seat1} to {seat2}")

########### 3. delta = 0, rashba = 0 ###########

print("\n3. delta = 0, rashba = 0")
m3 = kane_mele(0.0, T, SOC, 0.0)           # no staggering, no Rashba

r3 = tqc.realized_group(m3, characters=DECLARE)     # one species: delta = 0
g3 = r3.group
seats3 = list(dict.fromkeys(o.wyckoff for o in r3.wyckoff_orbits))
row("realized group", f"#{g3.number_str} {g3.name}, orbitals at Wyckoff "
    f"{' and '.join(seats3)}")

t_val3 = tqc.hsp_table(r3, bands=(0, 1))
t_con3 = tqc.hsp_table(r3, bands=(2, 3))
d_val3 = tqc.decompose(t_val3)
d_con3 = tqc.decompose(t_con3)

seat3 = min(t.wyckoff for t in d_val3.candidate_sums[0])
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

r4 = tqc.realized_group(m4, characters=DECLARE)
g4 = r4.group
seats4 = list(dict.fromkeys(o.wyckoff for o in r4.wyckoff_orbits))
row("realized group", f"#{g4.number_str} {g4.name}, orbitals at Wyckoff "
    f"{' and '.join(seats4)}")

t_val4 = tqc.hsp_table(r4, bands=(0, 1))
t_con4 = tqc.hsp_table(r4, bands=(2, 3))
d_val4 = tqc.decompose(t_val4)
d_con4 = tqc.decompose(t_con4)

row("bands 0 and 1", by_point(t_val4))
limits(d_val4)
row("bands 2 and 3", by_point(t_con4))
limits(d_con4)

# The coreps at K level by level.  multiplets groups the energies into
# degenerate levels, and coreps_at names the corep of each whole level.
hsp4 = tqc.high_symmetry_points(g4)
states_k4 = solve_at(m4, hsp4["K"])
energies_k4 = np.asarray(states_k4[0], dtype=float).reshape(-1)
multiplets_k4 = tqc.multiplets(energies_k4, degeneracy_tol=1e-6)
levels_k4 = []
for multiplet in multiplets_k4:
    content = tqc.coreps_at(r4, hsp4["K"], states=states_k4, bands=list(multiplet))
    levels_k4.append((content, float(np.mean(energies_k4[list(multiplet)])),
                      len(multiplet)))
row("coreps at K", *(f"{fmt(c.coreps)} at E = {e:+.6f}, dimension {n}"
                     for c, e, n in levels_k4))

########### 5. the conserved operators, read off H(k) ###########

print(f"\n5. the conserved operators, read off H(k) on a {NK_MESH}x{NK_MESH} mesh "
      "of in-plane k")

# realized_group tests each operation g on H(k) over its 6 x 6 mesh, solving
# X H(k) = H(gk) X at every k at once.  For the identity that is
# X H(k) = H(k) X, whose solutions are the operators the Hamiltonian conserves.
# conserved.dim counts them beyond the identity.  They are found from H alone,
# and the commutator with s_z below says which operator it is.
kx = np.arange(NK_MESH) / NK_MESH
ks = np.stack(np.meshgrid(kx, kx, indexing="ij"), -1).reshape(-1, 2)
sz = np.kron(np.eye(2), np.diag([1.0, -1.0]))       # s_z on the four basis states

h3 = np.asarray(m3.hamiltonian(k_pts=ks, flatten_spin_axis=True))
comm3 = float(np.abs(sz @ h3 - h3 @ sz).max())
h4 = np.asarray(m4.hamiltonian(k_pts=ks, flatten_spin_axis=True))
comm4 = float(np.abs(sz @ h4 - h4 @ sz).max())
row("delta, rashba", f"{'operators beyond 1':<22}max |[s_z, H(k)]|")
row("0, 0", f"{r3.conserved.dim:<22}{comm3:.6f}")
row("0, 0.25", f"{r4.conserved.dim:<22}{comm4:.6f}")
row("1.0, 0.25", f"{r1.conserved.dim:<22}not measured")
print("   at rashba = 0 the one conserved operator beyond the identity is s_z, whose")
print("   conservation the Rashba term breaks")

########### 6. what the lost operator does to the degeneracies ###########

print("\n6. what the lost operator does to the degeneracies")

# The dimensions of the levels at each point, from the bottom, without and with
# the Rashba term.
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

# The characters of the two levels bands 2 and 3 split into at K, one row for
# each level.  Every character of either level here is real or purely
# imaginary.
(c_lo, e_lo, n_lo), (c_mid, e_mid, _n2), (c_hi, e_hi, _n3) = levels_k4
# The characters of each level, on the unitary operations of the little group at K.
level_mid, level_hi = (tqc.characters(r4, hsp4["K"], states=states_k4,
                                      bands=list(multiplets_k4[i])).multiplets[0]
                       for i in (1, 2))
columns = tuple(level_mid.characters)
chi_mid = [level_mid.characters[i] for i in columns]
chi_hi = [level_hi.characters[i] for i in columns]
row_mid = [f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else
           (f"{z.imag:+.0f}i" if abs(z.real) < 1e-9 else f"{z:+.3f}")
           for z in chi_mid]
row_hi = [f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else
          (f"{z.imag:+.0f}i" if abs(z.real) < 1e-9 else f"{z:+.3f}")
          for z in chi_hi]
names = [r4.operations[i].name for i in columns]
widths = [max(len(a), len(b), len(c)) for a, b, c in zip(names, row_mid, row_hi)]
for label, cells in (("operations at K", names), (fmt(c_mid.coreps), row_mid),
                     (fmt(c_hi.coreps), row_hi)):
    row(label, "   ".join(f"{cell:<{w}}" for cell, w in zip(cells, widths)).rstrip())

splits = []
for rashba in RASHBA_SWEEP:
    e = np.asarray(solve_at(kane_mele(0.0, T, SOC, rashba), hsp4["K"])[0],
                   dtype=float).reshape(-1)
    splits.append((rashba, float(e[3] - e[2])))
row("bands 2 and 3", f"split at K by {listed(f'{s:.6f}' for _r, s in splits)}",
    f"at rashba = {listed(rashba for rashba, _s in splits)}")
row("split / rashba", listed(f"{s / r:.6f}" for r, s in splits))
print(f"   the {fmt(c_mid.coreps)} and {fmt(c_hi.coreps)} rows are complex conjugates: "
      "two distinct one-dimensional coreps")
print(f"   they are one level at rashba = 0, where s_z is conserved;\n   {fmt(c_lo.coreps)}, "
      f"of dimension {n_lo}, is one level at rashba = 0 and 0.25")

########### 7. the deep atomic limit ###########

print("\n7. the deep atomic limit: delta = 50")
deep = kane_mele(50.0, T, SOC, 0.25)       # the even regime, delta driven to 50
_e, vectors = deep.solve_ham(k_pts=deep.k_uniform_mesh([12, 12]), return_eigvecs=True,
                             flatten_spin_axis=False)
weight = (np.abs(vectors[:, :2]) ** 2).sum(axis=(0, 1, -1))
weight = weight / weight.sum()             # the weight of bands 0 and 1 per orbital

r_deep = tqc.realized_group(deep, typat=TYPAT, characters=DECLARE)
t_deep = tqc.hsp_table(r_deep, bands=2)
d_deep = tqc.decompose(t_deep)
# The Wyckoff orbits of the orbitals with the most and the least weight, read
# off r_deep.wyckoff_orbits.
heavy = next(o for o in r_deep.wyckoff_orbits if int(np.argmax(weight)) in o.orbitals)
light = next(o for o in r_deep.wyckoff_orbits if int(np.argmin(weight)) in o.orbitals)
row("bands 0 and 1", by_point(t_deep))
limits(d_deep)
row("orbital weight", f"{100 * weight.max():.2f} % on the orbital at Wyckoff "
    f"{heavy.wyckoff}, {100 * weight.min():.2f} % on the one at "
    f"{light.wyckoff}")
print("   the decomposition of bands 0 and 1 is at the position of the orbital with "
      "their weight")

########### 8. the bands along GM-K-M-GM ###########

print("\n8. the bands along GM-K-M-GM, one panel per parameter set")
fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.6), sharey=False)

# --- delta = 1.0, rashba = 0.25 ---
kv, kd, kn = m1.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m1.solve_ham(kv))
for i in range(4):
    axes[0].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0].axvline(x, color="0.75", lw=0.7)
# The coreps of the valence pair are written below the upper valence band, and
# those of the conduction pair above the lower conduction band.
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[0].annotate(fmt(coreps_of(t_val1, point)), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    axes[0].annotate(fmt(coreps_of(t_con1, point)), xy=(x, bands[j, 2]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:red")
axes[0].set_xticks(kn)
axes[0].set_xticklabels(TICKS, fontsize=13)
axes[0].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[0].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[0].set_ylabel("$E$", fontsize=16)
axes[0].set_title(f"delta = 1.0, rashba = 0.25\n#{g1.number_str} {g1.name}\n"
                  f"valence at {seat1}   |   conduction: no decomposition", fontsize=9.5)

# --- delta = 2.5, rashba = 0.25 ---
kv, kd, kn = m2.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m2.solve_ham(kv))
for i in range(4):
    axes[1].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[1].axvline(x, color="0.75", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[1].annotate(fmt(coreps_of(t_val2, point)), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    axes[1].annotate(fmt(coreps_of(t_con2, point)), xy=(x, bands[j, 2]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:red")
axes[1].set_xticks(kn)
axes[1].set_xticklabels(TICKS, fontsize=13)
axes[1].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[1].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[1].set_title(f"delta = 2.5, rashba = 0.25\n#{g2.number_str} {g2.name}\n"
                  f"valence at {seat2}   |   conduction at "
                  f"{min(t.wyckoff for t in d_con2.candidate_sums[0])}", fontsize=9.5)

# --- delta = 0, rashba = 0 ---
kv, kd, kn = m3.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m3.solve_ham(kv))
for i in range(4):
    axes[2].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[2].axvline(x, color="0.75", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[2].annotate(fmt(coreps_of(t_val3, point)), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    axes[2].annotate(fmt(coreps_of(t_con3, point)), xy=(x, bands[j, 2]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:red")
axes[2].set_xticks(kn)
axes[2].set_xticklabels(TICKS, fontsize=13)
axes[2].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[2].set_ylim(bands.min() - 0.30 * (bands.max() - bands.min()),
                 bands.max() + 0.22 * (bands.max() - bands.min()))
axes[2].set_title(f"delta = 0, rashba = 0\n#{g3.number_str} {g3.name}\n"
                  f"valence at {seat3}   |   conduction: no decomposition", fontsize=9.5)

# --- delta = 0, rashba = 0.25: K resolved level by level ---
kv, kd, kn = m4.k_path(PATH, NK_PATH, report=False)
bands = np.asarray(m4.solve_ham(kv))
for i in range(4):
    axes[3].plot(kd, bands[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[3].axvline(x, color="0.75", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[3].annotate(fmt(coreps_of(t_val4, point)), xy=(x, bands[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color="tab:blue")
    if point != "K":                     # K is resolved level by level instead
        axes[3].annotate(fmt(coreps_of(t_con4, point)), xy=(x, bands[j, 2]),
                         xytext=(0, 8), textcoords="offset points", ha="center",
                         fontsize=7.5, color="tab:red")
x_k = kn[NODES.index("K")]
axes[3].annotate(fmt(c_hi.coreps), xy=(x_k, e_hi), xytext=(0, 7),
                 textcoords="offset points", ha="center", fontsize=7.5, color="tab:red")
axes[3].annotate(fmt(c_mid.coreps), xy=(x_k, e_mid), xytext=(0, -14),
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
                  f"{fmt(c_mid.coreps)} and {fmt(c_hi.coreps)} split at K",
                  fontsize=9.5)

fig.suptitle("The Kane-Mele model on the honeycomb lattice: the four bands along "
             "GM-K-M-GM, the valence pair in blue and the conduction pair in red, with "
             "the measured coreps at each high-symmetry point.", fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.955))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "km_bands_and_coreps.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
