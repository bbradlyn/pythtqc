# Example of one monolayer presented two ways, as a layer in the plane and as a
# periodic decoupled stack, with one symmetry-indicator class in both, and of
# the inversion a layer carries, using pythTQC.
# The model is the Kane-Mele model on the honeycomb lattice at delta = 0 and
# rashba = 0, PRL 95, 226801 (2005), which is pythtb's own
# pythtb.models.kane_mele(0.0, -1.0, 0.3, 0.0).  Its horizontal mirror m(z) is
# unbroken.
# A wallpaper group has no inversion.  Where it has the twofold rotation C2z,
# that rotation acts on in-plane k as inversion does.  A layer group with both
# m(z) and C2z has their product, the inversion r -> -r.  The inversion squares
# to +E on spinors and has parity eigenvalues +-1, which C2z does not have.
# Declaring an s orbital in place of p_z changes the class.  Section D sets the
# three classes side by side.
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# A. the layer in the plane, declared free-standing: the group, the site irrep,
#    the corep multiplicities of the two occupied bands, the coreps at
#    k_z = 1/2, for which a plane has no coordinate, and the symmetry-indicator
#    class
# B. the same layer with the physical p_z orbital declared: the corep
#    multiplicities and the class again, the inversion the layer group carries,
#    and a Rashba term that breaks the mirror
# C. the layer as a periodic decoupled stack: a 3D crystal whose Hamiltonian
#    does not depend on k_z, the corep multiplicities at its six high-symmetry
#    points, all of them measured on the stack's own bands, and its class
# D. the three classes side by side, what is the same in all three and what
#    changes with the declared site irrep, and the class of all four bands, an
#    atomic limit
# E. the bands: the layer, and the stack at k_z = 0 and k_z = 1/2

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
from pythtqc.utils import save_figure

# Write the output as UTF-8, which a Windows console does not use by default.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band paths and nothing else.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures",
    "tqc_kane_mele_three_presentations")
NK_PATH = 61 if QUICK else 301             # band-path density: the figure only

# kane_mele(delta, t, soc, rashba).  delta = 0 keeps the two sublattices
# equivalent, and rashba = 0 keeps the mirror m(z) unbroken.
DELTA = 0.0
T = -1.0
SOC = 0.3
RASHBA = 0.0
RASHBA_BROKEN = 0.25       # a Rashba term that breaks m(z)

POINTS = ("GM", "K", "M")                  # the layer's high-symmetry points
STACK_POINTS = ("GM", "A", "K", "H", "M", "L")
KZ = (0.0, 0.25, 0.5)                      # where the stack's H is compared

PATH = [[0.0, 0.0], [1 / 3, 2 / 3], [0.5, 0.5], [0.0, 0.0]]     # GM - K - M - GM
NODES = ("GM", "K", "M", "GM")
TICKS = (r"$\Gamma$", "K", "M", r"$\Gamma$")

os.makedirs(OUTDIR, exist_ok=True)

layer = kane_mele(DELTA, T, SOC, RASHBA)

print("The Kane-Mele model on the honeycomb lattice, PRL 95, 226801 (2005), at delta = 0")
print("and rashba = 0: one monolayer with four bands, two occupied, and an unbroken "
      "mirror m(z),")
print("presented as a layer in the plane and as a periodic decoupled stack.")
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


def values(theta):
    return "(" + ", ".join(str(v) for v in theta) + ")"


def coreps_of(table, point):
    """The coreps a table's bands carry at one point, as {corep: multiplicity}."""
    return Counter(name for m in table.multiplets[point] for name in m.corep_names)


def by_point(table, points):
    """The corep multiplicities of a band set at each point, as format_coreps writes them."""
    return "   ".join(f"{p}: {fmt(coreps_of(table, p))}" for p in points)


def indicator(si):
    """The symmetry-indicator class, as a label and a value, or that the group has none."""
    if not si.divisors:
        return "indicator group", "trivial"
    return "indicator class", f"{values(si.values)} in {si.indicator_group}, order {si.order}"


########### A. the layer in the plane ###########

print("\nA. the layer in the plane, declared free-standing")

r_a = tqc.realized_group(layer, characters={0: "s"}, environment="free-standing")
g_a = r_a.group
orbit_a = r_a.wyckoff_orbits[0]          # the one Wyckoff orbit of the orbitals
row("realized group", f"#{g_a.number_str} {g_a.name}, {r_a.model.nstate} bands, "
    f"orbitals at Wyckoff {orbit_a.wyckoff}")
row("site irrep", f"{orbit_a.label}, of the declared s orbital")

table_a = tqc.hsp_table(r_a, bands=2)          # the coreps of bands 0 and 1
row("bands 0 and 1", by_point(table_a, POINTS))

# The elementary band representations of #191.234 have coreps at k_z = 1/2, for
# which a plane has no coordinate.  Handed the table, symmetry_indicator
# measures the bands at every point again on the layer's own decoupled stack,
# and reads the class from those multiplicities.  in_plane_restriction lists
# the coreps at k_z = 1/2 in .dropped.
rest = tqc.in_plane_restriction(g_a, require_exact=False)
row("at k_z = 1/2", f"{len(rest.dropped)} coreps, their multiplicities measured on the "
    "layer's own stack")

si_a = tqc.symmetry_indicator(table_a)
row(*indicator(si_a))
print("   free-standing is a declaration: a flat model has no third coordinate, and a")
print("   layer on a substrate has no mirror in z")

########### B. the same layer, with the physical p_z declared ###########

print("\nB. the same layer, with the physical p_z declared")

# The p_z orbitals of graphene are odd under the mirror m(z), and an s orbital
# is even.  Here m(z) is in the group, so the declaration says whether the
# orbital is odd or even under it.  The hoppings do not fix this.
full = tqc.realized_group(layer, environment="free-standing", characters={0: "pz"})
g191 = full.group
orbit_b = full.wyckoff_orbits[0]
# The character of orbital 0 under the mirror m(z), its orbital factor without
# the spin, found by the mirror's index.
mz_ind = tqc.operation_index(g191, "{m_001|0,0,0}")
mz = orbit_b.numerical.characters[0][mz_ind].real
row("realized group", f"#{g191.number_str} {g191.name}, orbitals at Wyckoff "
    f"{orbit_b.wyckoff}")
row("site irrep", f"{orbit_b.label}, of the declared p_z, with orbital character "
    f"{mz:+.0f} under m(z)")

table_b = tqc.hsp_table(full, bands=2)
row("bands 0 and 1", by_point(table_b, POINTS))

si_b = tqc.symmetry_indicator(table_b)
row(*indicator(si_b))

# The three operations compared below, each named by its Seitz symbol in the
# conventional cell used by the irreptables package: the inversion, the twofold
# rotation C2z and the mirror m(z).
p_ind = tqc.operation_index(g191, "{-1|0,0,0}")
c_ind = tqc.operation_index(g191, "{2_001|0,0,0}")
h_ind = tqc.operation_index(g191, "{m_001|0,0,0}")
P = full.operations[p_ind]
C2 = full.operations[c_ind]
SH = full.operations[h_ind]
# An operation's spin_lift is its SU(2) matrix on the spin, whose square is +E or Ebar.
p_sq = "+E" if np.allclose(P.spin_lift @ P.spin_lift, np.eye(2), atol=1e-9) else "Ebar"
c_sq = "+E" if np.allclose(C2.spin_lift @ C2.spin_lift, np.eye(2), atol=1e-9) else "Ebar"
h_sq = "+E" if np.allclose(SH.spin_lift @ SH.spin_lift, np.eye(2), atol=1e-9) else "Ebar"
squares = h_sq if h_sq == c_sq else f"{h_sq} and {c_sq}"
# The inversion's matrix maps the states on each site to the states on the site
# the inversion takes it to.  Its rows and columns run over the states, each
# orbital's spin up then down.  orbit_b.site_multiplets lists each site's
# orbitals.
M_p = np.asarray(full.rep.matrices[p_ind])
per = M_p.shape[0] // layer.norb          # the number of states on one orbital
site_of = {i: n for n, multiplet in enumerate(orbit_b.site_multiplets) for i in multiplet}
perm_p = [site_of[int(np.argmax(np.abs(M_p[:, per * multiplet[0]]))) // per]
          for multiplet in orbit_b.site_multiplets]
row("inversion", "r to -r, the product of the mirror m(z) and the rotation C2z,",
    "taking " + listed(f"site {i} to {j}" for i, j in enumerate(perm_p)))
row("spinor squares", f"{p_sq} for the inversion, so eigenvalues +-1; {squares} for m(z) "
    "and C2z,", "so eigenvalues +-i")

# A Rashba term breaks m(z), and the same call measures the group that is
# left.  R_zz is the zz entry of an operation's rotation.
broken = kane_mele(DELTA, T, SOC, RASHBA_BROKEN)
down = tqc.realized_group(broken, environment="free-standing", characters={0: "pz"})
g_down = down.group
r22_full = sorted({int(op.rotation[2, 2]) for op in full.operations.values()})
r22_down = sorted({int(op.rotation[2, 2]) for op in down.operations.values()})
table_down = tqc.hsp_table(down, bands=2)
si_down = tqc.symmetry_indicator(table_down)
row(f"rashba = {RASHBA_BROKEN}", f"#{g_down.number_str} {g_down.name}, a subgroup of "
    f"index {len(full.operations) // len(down.operations)};",
    f"{indicator(si_down)[0]}: {indicator(si_down)[1]}")
row("operations' R_zz", f"{listed(f'{v:+d}' for v in r22_full)} at rashba = {RASHBA:g}, "
    f"only {listed(f'{v:+d}' for v in r22_down)} at rashba = {RASHBA_BROKEN}:",
    "the mirror and every operation composed with it are gone")
print("   C2z acts on in-plane k as the inversion does, and is a rotation")
print("   the inversion and its parity eigenvalues belong to the layer group, not to a "
      "wallpaper group")

########### C. the layer as a periodic decoupled stack ###########

print("\nC. the layer as a periodic decoupled stack")

# stack_presentation repeats the layer along its normal with no hopping across
# the gap.  The stack has three periodic directions and no vacuum, and for this
# flat layer its H(k, k_z) does not depend on k_z.  So the coreps at k_z = 1/2
# of the elementary band representations of a 3D group can be measured for a
# monolayer.
stack = tqc.stack_presentation(layer)
row("the stack", f"{len(stack.periodic_dirs)} periodic directions and no vacuum, "
    f"period c = {np.asarray(stack.lat_vecs)[2, 2]:g}")

# The stack's H(k, k_z) against the layer's H(k), at six random in-plane k and
# three values of k_z.
rng = np.random.default_rng(7)
kk = rng.uniform(-0.5, 0.5, size=(6, 2))
h2d = np.asarray(layer.hamiltonian(k_pts=kk, flatten_spin_axis=True))
worst = 0.0
for kz in KZ:
    k3 = np.concatenate([kk, np.full((len(kk), 1), kz)], axis=1)
    h3d = np.asarray(stack.hamiltonian(k_pts=k3, flatten_spin_axis=True))
    worst = max(worst, float(np.abs(h2d - h3d).max()))
row("H(k, k_z) - H(k)", f"{worst:.6f} in its largest entry, over {len(kk)} random "
    "in-plane k", f"and k_z = {listed(f'{kz:g}' for kz in KZ)}")

piled = tqc.realized_group(stack, characters={0: "pz"})
g_stack = piled.group
hsp_stack = tqc.high_symmetry_points(g_stack)
row("realized group", f"#{g_stack.number_str} {g_stack.name}, orbitals at Wyckoff "
    f"{piled.wyckoff_orbits[0].wyckoff}, k-points "
    + " ".join(sorted(hsp_stack)))

six = tqc.hsp_table(piled, bands=2)      # all six points, measured directly
row("bands 0 and 1", by_point(six, STACK_POINTS))

si_c = tqc.symmetry_indicator(six)
row(*indicator(si_c))

########### D. the three classes side by side ###########

print("\nD. the three classes side by side")

# One row for each presentation, each column as wide as its widest entry.
lines = [("presentation", ("group", "indicator group", "class", "order", "z2w,3", "z4"))]
for name, si in (("the plane, s", si_a), ("the plane, p_z", si_b),
                 ("the stack, p_z", si_c)):
    named = si.indicators
    lines.append((name, (f"#{si.group.number_str}", si.indicator_group, values(si.values),
                         str(si.order), str(named["z2w,3"]), str(named["z4"]))))
widths = [max(len(cells[j]) for _name, cells in lines) for j in range(len(lines[0][1]))]
for name, cells in lines:
    row(name, "   ".join(f"{cell:<{w}}" for cell, w in zip(cells, widths)).rstrip())

forms = list(si_b.indicators.items())
row("named indicators", "of the p_z class, "
    + ", ".join(f"{k} = {v}" for k, v in forms[:4]) + ",",
    ", ".join(f"{k} = {v}" for k, v in forms[4:]))

support = si_b.numerical.two_dimensional.support
only = [n for n, s in support.items() if s == "out-of-plane only"]
part = [n for n, s in support.items() if s == "mixed"]
row("multiplicities", f"{listed(only)} depend only on those at k_z = 1/2,",
    f"{listed(part)} on those at k_z = 0 and k_z = 1/2")

# All four bands together carry the band representation of the orbitals, an
# atomic limit, whose class is zero.
table_all = tqc.hsp_table(piled, bands=4)
whole = tqc.symmetry_indicator(table_all)
row("all 4 bands", "the band representation of the orbitals at "
    f"{piled.wyckoff_orbits[0].wyckoff}, an atomic limit")
row(indicator(whole)[0], f"{indicator(whole)[1]}, every named indicator "
    + ", ".join(str(v) for v in sorted(set(whole.indicators.values()))))
print(f"   the stack and the plane with p_z have the same class, {values(si_c.values)}")
print(f"   declaring s in the plane instead gives {values(si_a.values)}, with the same "
      "order, z2w,3 and z4")

########### E. the bands ###########

print("\nE. the layer's bands, and the stack's at k_z = 0 and k_z = 1/2")
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)

# --- the layer, labeled in its own layer group ---
kv, kd, kn = layer.k_path(PATH, NK_PATH, report=False)
bands_2d = np.asarray(layer.solve_ham(kv))
for i in range(4):
    axes[0].plot(kd, bands_2d[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0].axvline(x, color="0.75", lw=0.7)
# The valence pair is degenerate at GM, K and M, so one label below the pair
# names the corep both bands carry.
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[0].annotate(fmt(coreps_of(table_b, point)), xy=(x, bands_2d[j, 1]),
                     xytext=(0, -16), textcoords="offset points", ha="center",
                     fontsize=8, color="tab:blue")
axes[0].set_xticks(kn)
axes[0].set_xticklabels(TICKS, fontsize=13)
axes[0].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[0].set_ylim(bands_2d.min() - 0.30 * (bands_2d.max() - bands_2d.min()),
                 bands_2d.max() + 0.22 * (bands_2d.max() - bands_2d.min()))
axes[0].set_ylabel("$E$", fontsize=16)
axes[0].set_title(f"the layer in the plane\n#{g191.number_str} {g191.name}   "
                  f"class {si_b.values} in {si_b.indicator_group}", fontsize=10)

# --- the stack, at k_z = 0 and at k_z = 1/2, overlaid ---
path_kz0 = [[k[0], k[1], 0.0] for k in PATH]
path_kz_half = [[k[0], k[1], 0.5] for k in PATH]
kv0, kd0, kn0 = stack.k_path(path_kz0, NK_PATH, report=False)
bands_kz0 = np.asarray(stack.solve_ham(kv0))
kvh, kdh, knh = stack.k_path(path_kz_half, NK_PATH, report=False)
bands_kzh = np.asarray(stack.solve_ham(kvh))
for i in range(4):
    axes[1].plot(kd0, bands_kz0[:, i], lw=3.2, color="0.72")
for i in range(4):
    axes[1].plot(kdh, bands_kzh[:, i], lw=1.4, ls="--",
                 color="tab:blue" if i < 2 else "tab:red")
for x in kn0:
    axes[1].axvline(x, color="0.75", lw=0.7)
for point, x in zip(("A", "H", "L", "A"), kn0):
    j = int(np.argmin(np.abs(kd0 - x)))
    axes[1].annotate(fmt(coreps_of(six, point)), xy=(x, bands_kzh[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="tab:blue")
axes[1].set_xticks(kn0)
axes[1].set_xticklabels((r"$A$", "H", "L", r"$A$"), fontsize=13)
axes[1].set_xlim(kd0[0] - 0.07 * kd0[-1], kd0[-1] * 1.07)
axes[1].set_title(f"the periodic stack at $k_z = 1/2$ (dashed)\nover $k_z = 0$ (gray)"
                  f", #{g_stack.number_str} {g_stack.name}   "
                  f"class {si_c.values} in {si_c.indicator_group}", fontsize=10)

fig.suptitle("One Kane-Mele monolayer, presented as a layer in the plane and as a "
             "periodic decoupled stack.  The symmetry-indicator class is the same in "
             "both presentations, and the labels are the measured coreps.",
             fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.93))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "km_three_presentations.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
