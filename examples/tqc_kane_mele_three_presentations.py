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
# A. the layer in the plane, declared free-standing: the group, the valence
#    content, the EBR rows a plane has no coordinate for, and the class
# B. the same layer with the physical p_z declared: the content and the class
#    again, the inversion the layer group carries, and the Rashba control that
#    breaks the mirror
# C. the layer as a periodic decoupled stack: a 3D crystal with no k_z, its six
#    k-points measured directly, and the class asked of it straight
# D. the three classes side by side, what is invariant and what is
#    convention-relative, and the four-band atomic-limit control
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

# The tables name some forms of the indicator with π (z3m,π); a Windows console
# cannot print π by default, so the output is UTF-8.
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
print("one monolayer, whose horizontal mirror is unbroken, asked three ways.")

########### A. the layer in the plane ###########

print("\nA. the layer in the plane, declared free-standing")

out_a = tqc.realized_group(layer, characters={0: "s"}, environment="free-standing")
model_a, g_a, asg_a, val_a, rep_a = out_a
sites_a = tqc.wyckoff_sites_tables(g_a, asg_a.site_positions)   # letter, multiplicity
print(f"   realized group   : #{g_a.number_str} {g_a.name}, {model_a.nstate} bands, "
      f"orbitals at Wyckoff {sites_a[0].multiplicity}{sites_a[0].letter}")
print(f"                      carrying the site irrep {rep_a.characters.irreps[0]}")
print("                      (free-standing is a declaration: a flat model has no "
      "third")
print("                      coordinate, and a layer on a substrate has no mirror "
      "in z)")

table_a = tqc.hsp_table(rep_a, bands=2)          # the valence pair's content
print("   valence content  : "
      + "   ".join(f"{p}: {fmt(table_a.content[p])}" for p in POINTS))

# #191.234's EBR basis carries rows at k_z = 1/2, which a plane has no coordinate
# for; given the representation, symmetry_indicator measures them on the layer's
# own decoupled stack.  in_plane_restriction names those rows, in .dropped,
# whether or not each repeats a row in the plane.
rest = tqc.in_plane_restriction(g_a, require_exact=False)
print(f"   the EBR rows     : the class of a layer needs the {len(rest.dropped)} "
      f"k_z = 1/2 rows the plane")
print("                      has no coordinate for; they are measured on the layer's "
      "own stack")

si_a = tqc.symmetry_indicator(table_a, g_a, rep=rep_a)
print(f"   class            : {si_a.values}, order {si_a.order} in "
      f"X_BS(#{si_a.number_str}) = {si_a.structure}")

########### B. the same layer, with the physical p_z declared ###########

print("\nB. the same layer, with the physical p_z declared")

# A honeycomb monolayer is built from p_z orbitals: odd under the horizontal
# mirror, where an s orbital is even.  Here sigma_h is in the group, so the choice
# is expressible, and the hoppings cannot make it.
full = tqc.realized_group(layer, environment="free-standing", characters={0: "pz"})
_mf, g191, asg191, val191, rep191 = full
sites_b = tqc.wyckoff_sites_tables(g191, asg191.site_positions)
mz = rep191.characters.orbitals[0].generator_character["m(z)"].real
print(f"   realized group   : #{g191.number_str} {g191.name}, orbitals at Wyckoff "
      f"{sites_b[0].multiplicity}{sites_b[0].letter}")
print(f"                      carrying the site irrep {rep191.characters.irreps[0]}")
print("                      (the declared p_z is odd under the mirror, "
      f"m(z) = {mz:+.0f}, and")
print(f"                      an even s orbital lifts to "
      f"{rep_a.characters.irreps[0]} instead)")

table_b = tqc.hsp_table(rep191, bands=2)
print("   valence content  : "
      + "   ".join(f"{p}: {fmt(table_b.content[p])}" for p in POINTS))

si_b = tqc.symmetry_indicator(table_b, g191, rep=rep191)
print(f"   class            : {si_b.values}, order {si_b.order} in "
      f"X_BS(#{si_b.number_str}) = {si_b.structure}")

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
print(f"   inversion        : #{g191.number_str} has r -> -r, the product of the mirror "
      f"m(z) and the")
print(f"                      twofold rotation C2z; its spinor lift squares to {p_sq}, "
      f"so its")
print("                      eigenvalues are +-1 and a parity count exists, while the "
      "lifts of")
print(f"                      m(z) and C2z square to {squares} with eigenvalues +-i.")
print("                      C2z acts on in-plane k as inversion does but is a "
      "rotation: this")
print("                      inversion is the layer group's and needs the third "
      "direction.")
print("                      It exchanges the two honeycomb sites: "
      + ", ".join(f"site {i} -> {j}" for i, j in enumerate(perm_p)) + ".")

# The control: a Rashba term breaks sigma_h, and the same call measures what is left.
broken = kane_mele(DELTA, T, SOC, RASHBA_BROKEN)
down = tqc.realized_group(broken, environment="free-standing", characters={0: "pz"})
_mb, g_down, _ab, _vb, rep_down = down
r22_full = sorted({int(np.rint(np.asarray(s.rotation)[2, 2])) for s in g191.symmetries})
r22_down = sorted({int(np.rint(np.asarray(s.rotation)[2, 2]))
                   for s in g_down.symmetries})
table_down = tqc.hsp_table(rep_down, bands=2)
si_down = tqc.symmetry_indicator(table_down, g_down, rep=rep_down)
print(f"   rashba = {RASHBA_BROKEN}    : the same call measures #{g_down.number_str} "
      f"{g_down.name}, a subgroup of index {g191.size // g_down.size}.")
print(f"                      #{g191.number_str} has operations with R[2,2] = "
      + " and ".join(f"{v:+d}" for v in r22_full) + "; what the Rashba")
print("                      term leaves has only R[2,2] = "
      + " and ".join(f"{v:+d}" for v in r22_down)
      + " -- the mirror and everything")
print("                      composed with it is gone, and "
      f"X_BS(#{si_down.number_str}) = {si_down.structure}")

########### C. the layer as a periodic decoupled stack ###########

print("\nC. the layer as a periodic decoupled stack")

# stack_presentation repeats the layer along its normal with no hopping across the
# gap: three periodic directions, no vacuum, and an H(k, k_z) that does not depend
# on k_z at all.  That last property is what makes a 3D EBR basis' k_z = 1/2 rows
# measurable on a monolayer.
stack = tqc.stack_presentation(layer)
print(f"   the stack        : {len(stack.periodic_dirs)} periodic directions and no "
      f"vacuum, period c = {np.asarray(stack.lat_vecs)[2, 2]:g}")

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
print("   no k_z           : the stack's H(k, k_z) equals the layer's H(k) at k_z = "
      + ", ".join(f"{kz:g}" for kz in KZ) + ",")
print(f"                      max |H(k, k_z) - H(k)| = {worst:.6f} over {len(kk)} "
      "random in-plane k")

piled = tqc.realized_group(stack, characters={0: "pz"})
_ms, g_stack, asg_stack, val_stack, rep_stack = piled
sites_c = tqc.wyckoff_sites_tables(g_stack, asg_stack.site_positions)
hsp_stack = tqc.high_symmetry_points(g_stack)
print(f"   realized group   : #{g_stack.number_str} {g_stack.name}, orbitals at "
      f"Wyckoff {sites_c[0].multiplicity}{sites_c[0].letter}, k-points "
      + " ".join(sorted(hsp_stack)))

six = tqc.hsp_table(rep_stack, bands=2)      # all six points, measured directly
print("   six-point content: "
      + "   ".join(f"{p}: {fmt(six.content[p])}" for p in STACK_POINTS))

si_c = tqc.symmetry_indicator(six, g_stack, rep=rep_stack)
print(f"   class            : {si_c.values}, order {si_c.order} in "
      f"X_BS(#{si_c.number_str}) = {si_c.structure}")

########### D. the three classes side by side ###########

print("\nD. the three classes side by side")
print(f"   A. the plane, s          : "
      f"#{si_a.number_str}, {si_a.structure}, class {si_a.values}, "
      f"order {si_a.order}")
print(f"   B. the plane, p_z        : "
      f"#{si_b.number_str}, {si_b.structure}, class {si_b.values}, "
      f"order {si_b.order}")
print(f"   C. the stack, p_z        : "
      f"#{si_c.number_str}, {si_c.structure}, class {si_c.values}, "
      f"order {si_c.order}")
print(f"   the invariant part: order {si_a.order}/{si_b.order}/{si_c.order}, "
      f"z2w,3 = {si_a.irreptables_named['z2w,3']}/{si_b.irreptables_named['z2w,3']}"
      f"/{si_c.irreptables_named['z2w,3']}, "
      f"z4 = {si_a.irreptables_named['z4']}/{si_b.irreptables_named['z4']}"
      f"/{si_c.irreptables_named['z4']}")
print("   B and C are one crystal presented two ways and their classes agree entry "
      "for")
print("   entry; A and B differ in the declared site character alone, and that moves")
print("   the component values and not the invariant part")

forms = list(si_b.irreptables_named.items())
print("   irreptables forms : "
      + ", ".join(f"{k} = {v}" for k, v in forms[:4]) + ",")
print("                       "
      + ", ".join(f"{k} = {v}" for k, v in forms[4:]))

support = si_b.two_dimensional["irreptables_support"]
only = [n for n, s in support.items() if s == "out-of-plane only"]
part = [n for n, s in support.items() if s == "mixed"]
print(f"   out of the plane  : {', '.join(only)} read only the k_z = 1/2 rows, and")
print(f"                       {', '.join(part)} both planes.  Those rows repeat the "
      "in-plane ones,")
print("                       so the class is the class of the layer's decoupled stack,")
print("                       and every form is an invariant of that stack, fixed by")
print("                       the layer's own content")

# the four bands together: the orbitals' own band representation, an atomic limit
# by construction, and the control that says the class above is half-filling physics
table_all = tqc.hsp_table(rep_stack, bands=4)
whole = tqc.symmetry_indicator(table_all, g_stack, rep=rep_stack)
print("   the control       : the four bands together are the band representation "
      "of the")
print(f"                       orbitals at "
      f"{sites_c[0].multiplicity}{sites_c[0].letter}, an atomic limit by "
      "construction, and their")
print(f"                       class is {whole.values}, every irreptables form "
      + ", ".join(str(v) for v in sorted(set(whole.irreptables_named.values()))))

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
