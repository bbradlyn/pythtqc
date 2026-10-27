# Example of presentation independence using pythtqc: one monolayer, asked more
# than one way, and the inversion a layer carries.
# The model is Kane and Mele's honeycomb at delta = 0 and rashba = 0, PRL 95,
# 226801 (2005) -- pythtb's own pythtb.models.kane_mele(0.0, -1.0, 0.3, 0.0) --
# the variant whose horizontal mirror sigma_h is unbroken.
# The physics: there is no inversion in a 2D wallpaper group.  C2z is a rotation
# that happens to act on in-plane k the way inversion would.  Genuine inversion --
# r -> -r, squaring to E and not Ebar on spinors -- needs the third direction, and
# a flat layer has it as a layer-group operation: where sigma_h is real,
# P = sigma_h * C2z is in the group and carries the parity eigenvalues +-1 that C2z
# does not have.  The class the layer occupies in {BS}/{AI} must not depend on how
# the crystal was presented, and section D is where that is read off.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# A. the layer in the plane, declared free-standing: the group, the site irrep,
#    the content of the two occupied bands, the coreps at k_z = 1/2 a plane has
#    no coordinate for, and the symmetry-indicator class
# B. the same layer with the physical p_z declared: the content and the class
#    again, the inversion the layer group carries, and the Rashba control that
#    breaks the mirror
# C. the layer as a periodic decoupled stack: a 3D crystal with no k_z, its six
#    k-points measured directly, and the class asked of it straight
# D. the three classes side by side, what is invariant and what moves with the
#    declared site irrep, and all four bands as the atomic-limit control
# E. the bands: the layer, and the stack at k_z = 0 and k_z = 1/2

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
from pythtqc.utils import save_figure

# The tables name some indicators with π (z3m,π); a Windows console cannot print
# π by default, so the output is UTF-8.
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
# equivalent; rashba = 0 keeps sigma_h unbroken, which is what this script is about.
DELTA = 0.0
T = -1.0
SOC = 0.3
RASHBA = 0.0
RASHBA_BROKEN = 0.25       # the control: a Rashba term breaks sigma_h

POINTS = ("GM", "K", "M")                  # the layer's k-points
STACK_POINTS = ("GM", "A", "K", "H", "M", "L")
KZ = (0.0, 0.25, 0.5)                      # where the stack's k_z-freedom is checked

PATH = [[0.0, 0.0], [1 / 3, 2 / 3], [0.5, 0.5], [0.0, 0.0]]     # GM - K - M - GM
NODES = ("GM", "K", "M", "GM")
TICKS = (r"$\Gamma$", "K", "M", r"$\Gamma$")

os.makedirs(OUTDIR, exist_ok=True)

layer = kane_mele(DELTA, T, SOC, RASHBA)

print("Kane-Mele on the honeycomb at delta = 0 and rashba = 0, PRL 95, 226801 (2005):")
print("one monolayer with four bands, two occupied, and an unbroken horizontal mirror,")
print("asked three ways.")
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


def values(theta):
    return "(" + ", ".join(str(v) for v in theta) + ")"


def by_point(table, points):
    """A band set's content at each point, as format_counts writes it."""
    return "   ".join(f"{p}: {fmt(table.content[p])}" for p in points)


def indicator(si):
    """The symmetry-indicator class, as a label and a value, or that the group has none."""
    if si.verdict == "trivial indicator group":
        return "indicator group", "trivial"
    return "indicator class", f"{values(si.values)} in {si.structure}, order {si.order}"


########### A. the layer in the plane ###########

print("\nA. the layer in the plane, declared free-standing")

out_a = tqc.realized_group(layer, characters={0: "s"}, environment="free-standing")
model_a, g_a, asg_a, val_a, rep_a = out_a
sites_a = tqc.wyckoff_sites_tables(g_a, asg_a.site_positions)   # letter, multiplicity
row("realized group", f"#{g_a.number_str} {g_a.name}, {model_a.nstate} bands, "
    f"orbitals at Wyckoff {sites_a[0].multiplicity}{sites_a[0].letter}")
row("site irrep", f"{rep_a.characters.irreps[0]}, of the declared s orbital")

table_a = tqc.hsp_table(rep_a, bands=2)          # the content of bands 0 and 1
row("bands 0 and 1", by_point(table_a, POINTS))

# The elementary band representations of #191.234 have coreps at k_z = 1/2,
# which a plane has no coordinate for; given the representation,
# symmetry_indicator measures them on the layer's own decoupled stack.
# in_plane_restriction lists those coreps in .dropped, whether or not each
# repeats one in the plane.
rest = tqc.in_plane_restriction(g_a, require_exact=False)
row("out of the plane", f"{len(rest.dropped)} coreps at k_z = 1/2, measured on the "
    "layer's own stack")

si_a = tqc.symmetry_indicator(table_a)
row(*indicator(si_a))
print("   free-standing is a declaration: a flat model has no third coordinate, and a")
print("   layer on a substrate has no mirror in z")

########### B. the same layer, with the physical p_z declared ###########

print("\nB. the same layer, with the physical p_z declared")

# A honeycomb monolayer is built from p_z orbitals: odd under the horizontal
# mirror, where an s orbital is even.  Here sigma_h is in the group, so the choice
# is expressible, and the hoppings cannot make it.
full = tqc.realized_group(layer, environment="free-standing", characters={0: "pz"})
_mf, g191, asg191, val191, rep191 = full
sites_b = tqc.wyckoff_sites_tables(g191, asg191.site_positions)
mz = rep191.characters.orbitals[0].generator_character["m(z)"].real
row("realized group", f"#{g191.number_str} {g191.name}, orbitals at Wyckoff "
    f"{sites_b[0].multiplicity}{sites_b[0].letter}")
row("site irrep", f"{rep191.characters.irreps[0]}, of the declared p_z, odd under the "
    f"mirror: m(z) = {mz:+.0f}")

table_b = tqc.hsp_table(rep191, bands=2)
row("bands 0 and 1", by_point(table_b, POINTS))

si_b = tqc.symmetry_indicator(table_b)
row(*indicator(si_b))

# The three operations compared below, each named by its Seitz symbol in the
# tables' setting: the inversion, the twofold rotation about z and the mirror
# normal to z.
p_ind = tqc.operation_index(g191, "{-1|0,0,0}")
c_ind = tqc.operation_index(g191, "{2_001|0,0,0}")
h_ind = tqc.operation_index(g191, "{m_001|0,0,0}")
P = rep191.operations[p_ind]
C2 = rep191.operations[c_ind]
SH = rep191.operations[h_ind]
# an operation's spin_lift is its SU(2) matrix on the spin, whose square is +E or Ebar
p_sq = "+E" if np.allclose(P.spin_lift @ P.spin_lift, np.eye(2), atol=1e-9) else "Ebar"
c_sq = "+E" if np.allclose(C2.spin_lift @ C2.spin_lift, np.eye(2), atol=1e-9) else "Ebar"
h_sq = "+E" if np.allclose(SH.spin_lift @ SH.spin_lift, np.eye(2), atol=1e-9) else "Ebar"
squares = h_sq if h_sq == c_sq else f"{h_sq} and {c_sq}"
# group.symmetries are the group's operations in the tables' order, and the
# assignment's permutations[k] the site each site is carried to by the k-th of them
position = [int(s.ind) for s in g191.symmetries].index(p_ind)
perm_p = np.asarray(asg191.permutations[position]).tolist()
row("inversion", "r to -r, the product of the mirror m(z) and the rotation C2z,",
    "taking " + listed(f"site {i} to {j}" for i, j in enumerate(perm_p)))
row("spinor squares", f"{p_sq} for the inversion, so eigenvalues +-1; {squares} for m(z) "
    "and C2z,", "so eigenvalues +-i")

# The control: a Rashba term breaks sigma_h, and the same call measures what is
# left.  R_zz is the zz entry of an operation's rotation.
broken = kane_mele(DELTA, T, SOC, RASHBA_BROKEN)
down = tqc.realized_group(broken, environment="free-standing", characters={0: "pz"})
_mb, g_down, _ab, _vb, rep_down = down
r22_full = sorted({int(np.rint(np.asarray(s.rotation)[2, 2])) for s in g191.symmetries})
r22_down = sorted({int(np.rint(np.asarray(s.rotation)[2, 2]))
                   for s in g_down.symmetries})
table_down = tqc.hsp_table(rep_down, bands=2)
si_down = tqc.symmetry_indicator(table_down)
row(f"rashba = {RASHBA_BROKEN}", f"#{g_down.number_str} {g_down.name}, a subgroup of "
    f"index {g191.size // g_down.size};", f"{indicator(si_down)[0]}: {indicator(si_down)[1]}")
row("operations' R_zz", f"{listed(f'{v:+d}' for v in r22_full)} at rashba = {RASHBA:g}, "
    f"only {listed(f'{v:+d}' for v in r22_down)} at rashba = {RASHBA_BROKEN}:",
    "the mirror and every operation composed with it are gone")
print("   C2z acts on in-plane k as inversion does but is a rotation: the inversion, with")
print("   its parity eigenvalues, is the layer group's and needs the third direction")

########### C. the layer as a periodic decoupled stack ###########

print("\nC. the layer as a periodic decoupled stack")

# stack_presentation repeats the layer along its normal with no hopping across the
# gap: three periodic directions, no vacuum, and an H(k, k_z) that does not depend
# on k_z at all.  That last property makes the coreps at k_z = 1/2 of a 3D group's
# elementary band representations measurable on a monolayer.
stack = tqc.stack_presentation(layer)
row("the stack", f"{len(stack.periodic_dirs)} periodic directions and no vacuum, "
    f"period c = {np.asarray(stack.lat_vecs)[2, 2]:g}")

# the stack's H(k, k_z) against the layer's H(k), at six random in-plane k and
# three values of k_z
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
_ms, g_stack, asg_stack, val_stack, rep_stack = piled
sites_c = tqc.wyckoff_sites_tables(g_stack, asg_stack.site_positions)
hsp_stack = tqc.high_symmetry_points(g_stack)
row("realized group", f"#{g_stack.number_str} {g_stack.name}, orbitals at Wyckoff "
    f"{sites_c[0].multiplicity}{sites_c[0].letter}, k-points "
    + " ".join(sorted(hsp_stack)))

six = tqc.hsp_table(rep_stack, bands=2)      # all six points, measured directly
row("bands 0 and 1", by_point(six, STACK_POINTS))

si_c = tqc.symmetry_indicator(six)
row(*indicator(si_c))

########### D. the three classes side by side ###########

print("\nD. the three classes side by side")

# one row per presentation, each column as wide as its widest entry
lines = [("presentation", ("group", "indicator group", "class", "order", "z2w,3", "z4"))]
for name, si in (("the plane, s", si_a), ("the plane, p_z", si_b),
                 ("the stack, p_z", si_c)):
    named = si.irreptables_named
    lines.append((name, (f"#{si.number_str}", si.structure, values(si.values),
                         str(si.order), str(named["z2w,3"]), str(named["z4"]))))
widths = [max(len(cells[j]) for _name, cells in lines) for j in range(len(lines[0][1]))]
for name, cells in lines:
    row(name, "   ".join(f"{cell:<{w}}" for cell, w in zip(cells, widths)).rstrip())

forms = list(si_b.irreptables_named.items())
row("named indicators", "of the p_z class, "
    + ", ".join(f"{k} = {v}" for k, v in forms[:4]) + ",",
    ", ".join(f"{k} = {v}" for k, v in forms[4:]))

support = si_b.two_dimensional["irreptables_support"]
only = [n for n, s in support.items() if s == "out-of-plane only"]
part = [n for n, s in support.items() if s == "mixed"]
row("out of the plane", f"{listed(only)} read only the coreps at k_z = 1/2,",
    f"{listed(part)} those of both planes")

# all four bands together: the orbitals' own band representation, an atomic limit
# by construction, and the control that says the class above is half-filling physics
table_all = tqc.hsp_table(rep_stack, bands=4)
whole = tqc.symmetry_indicator(table_all)
row("all 4 bands", "the band representation of the orbitals at "
    f"{sites_c[0].multiplicity}{sites_c[0].letter}, an atomic limit")
row(indicator(whole)[0], f"{indicator(whole)[1]}, every named indicator "
    + ", ".join(str(v) for v in sorted(set(whole.irreptables_named.values()))))
print(f"   the stack and the plane with p_z have one class, {values(si_c.values)}; "
      "declaring s instead")
print(f"   moves it to {values(si_a.values)} and leaves its order, z2w,3 and z4 as "
      "they are")

########### E. the bands ###########

print("\nE. the layer's bands, and the stack's at k_z = 0 and k_z = 1/2")
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)

# --- the layer, labelled in its own layer group ---
kv, kd, kn = layer.k_path(PATH, NK_PATH, report=False)
bands_2d = np.asarray(layer.solve_ham(kv))
for i in range(4):
    axes[0].plot(kd, bands_2d[:, i], lw=1.8, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0].axvline(x, color="0.75", lw=0.7)
# the valence pair is Kramers-degenerate at every node, so one offset below it
# clears both bands
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[0].annotate(fmt(table_b.content[point]), xy=(x, bands_2d[j, 1]),
                     xytext=(0, -16), textcoords="offset points", ha="center",
                     fontsize=8, color="tab:blue")
axes[0].set_xticks(kn)
axes[0].set_xticklabels(TICKS, fontsize=13)
axes[0].set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
axes[0].set_ylim(bands_2d.min() - 0.30 * (bands_2d.max() - bands_2d.min()),
                 bands_2d.max() + 0.22 * (bands_2d.max() - bands_2d.min()))
axes[0].set_ylabel("$E$", fontsize=16)
axes[0].set_title(f"the layer in the plane\n#{g191.number_str} {g191.name}   "
                  f"class {si_b.values} in {si_b.structure}", fontsize=10)

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
    axes[1].annotate(fmt(six.content[point]), xy=(x, bands_kzh[j, 1]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="tab:blue")
axes[1].set_xticks(kn0)
axes[1].set_xticklabels((r"$A$", "H", "L", r"$A$"), fontsize=13)
axes[1].set_xlim(kd0[0] - 0.07 * kd0[-1], kd0[-1] * 1.07)
axes[1].set_title(f"the periodic stack at $k_z = 1/2$ (dashed)\nover $k_z = 0$ (grey)"
                  f" — #{g_stack.number_str} {g_stack.name}   "
                  f"class {si_c.values} in {si_c.structure}", fontsize=10)

fig.suptitle("One Kane-Mele monolayer, presented as a layer in the plane and as a "
             "periodic decoupled stack.  The class in {BS}/{AI} is a property of the "
             "crystal, not of the presentation; the labels are measured corep content.",
             fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.93))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "km_three_presentations.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
