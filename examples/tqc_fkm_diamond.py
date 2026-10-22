# Example of a bond distortion measured as a symmetry lowering, and of the
# symmetry-indicator class of each phase, using pythtqc.
# The model is Fu, Kane and Mele's spin-orbit model on the diamond lattice,
# PRL 98, 106803 (2007): four bands, two occupied, with one of the four
# nearest-neighbour bonds detuned along [111].  It is pythtb's own
# pythtb.models.fu_kane_mele(t, soc, dt).
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. the undistorted crystal: the group, the bands at the high-symmetry points,
#    and the content of all four bands with its atomic limits
# 2. delta_t = +0.4 on the [111] bond: the group, the bands, the valence content
#    and its atomic limits
# 3. delta_t = -0.4 on the same bond: the same four measurements
# 4. three bonds cut -- decoupled dimers along [111]: the same four measurements
# 5. the content at the eight time-reversal-invariant momenta
# 6. the symmetry-indicator class of the three phases in {BS}/{AI}
# 7. the bands along GM-T-F-L-GM, and the figure written out

# Author: the pythtqc examples

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb.models import fu_kane_mele

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
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_fkm_diamond")
NK_PATH = 61 if QUICK else 241              # band-path density: the figure only

# fu_kane_mele(t, soc, dt): ``dt`` offsets the four nearest-neighbour bonds, in
# the order R = [0,0,0], [-1,0,0], [0,-1,0], [0,0,-1].
T = 1.0                    # the nearest-neighbour hopping
SOC = 0.125                # the spin-orbit strength

# A pythtb orbital carries no symmetry character, so the s here is a declaration.
DECLARE = {0: "s"}

POINTS = ("GM", "T", "F", "L")               # the distorted phases' k-points
NODES = ("GM", "T", "F", "L", "GM")          # the figure's path
TICKS = (r"$\Gamma$", "T", "F", "L", r"$\Gamma$")

os.makedirs(OUTDIR, exist_ok=True)

print("Fu-Kane-Mele on the diamond lattice, PRL 98, 106803 (2007): four bands, two")
print("occupied, one of the four nearest-neighbour bonds detuned along [111].")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def values(entries):
    return "(" + ", ".join(str(v) for v in entries) + ")"


def atomic_limits(d):
    """The atomic limits of a band set, as a label and a value: the one sum of elementary
    band representations with this content, how many sums there are, or none."""
    if d.verdict == "none":
        return "atomic limits", "none"
    sums = d.solution_strings()
    if len(sums) == 1:
        return "atomic limit", sums[0]
    return "atomic limits", f"{len(sums)} sums of elementary band representations"


########### 1. the undistorted crystal ###########

print("\n1. the undistorted crystal")
undistorted = fu_kane_mele(T, SOC, [0.0, 0.0, 0.0, 0.0])     # every bond equal

out = tqc.realized_group(undistorted, characters=DECLARE)
model, group, assignment, validated, rep = out
sites = tqc.wyckoff_sites_tables(group, assignment.site_positions)   # letter and multiplicity
row("realized group", f"#{group.number_str} {group.name}, s orbitals at Wyckoff "
    f"{sites[0].multiplicity}{sites[0].letter}, {model.nstate} bands")

# the four energies at each high-symmetry point, and the gap at half filling
hsp = tqc.high_symmetry_points(group)
for point in sorted(hsp):
    w = np.asarray(solve_at(undistorted, hsp[point])[0], dtype=float).reshape(-1)
    gap = w[2] - w[1]
    if w.max() - w.min() < 1e-9:
        row(f"E at {point}", f"{'the four bands meet':<42}   gap {gap:.6f}")
    else:
        row(f"E at {point}", ", ".join(f"{x:+.6f}" for x in w) + f"   gap {gap:.6f}")

table4 = tqc.hsp_table(rep, bands=4)          # the content of all four bands
d4 = tqc.decompose(table4, group, rep=rep)
row("all 4 bands", fmt(table4.counts))
row(*atomic_limits(d4))

# ... and the same table asked for the valence bands alone
table2 = tqc.hsp_table(rep, bands=2)
row("valence content", f"not defined at {', '.join(sorted(table2.refusals))}, where all "
    "four bands meet")
print("   the undistorted crystal is a semimetal")

########### 2. delta_t = +0.4 on the [111] bond ###########

print("\n2. delta_t = +0.4 on the [111] bond")
m_strong = fu_kane_mele(T, SOC, [0.4, 0.0, 0.0, 0.0])        # one bond strengthened

r_strong = tqc.realized_group(m_strong, characters=DECLARE)
_m, g_strong, asg_strong, val_strong, rep_strong = r_strong
sites_strong = tqc.wyckoff_sites_tables(g_strong, asg_strong.site_positions)
row("realized group", f"#{g_strong.number_str} {g_strong.name}, orbitals at Wyckoff "
    f"{sites_strong[0].multiplicity}{sites_strong[0].letter}: the detuned bond lowers",
    f"{group.name} to {g_strong.name}, with no atom moved")

hsp_strong = tqc.high_symmetry_points(g_strong)
for point in POINTS:
    w = np.asarray(solve_at(m_strong, hsp_strong[point])[0], dtype=float).reshape(-1)
    row(f"E at {point}", ", ".join(f"{x:+.6f}" for x in w) + f"   gap {w[2] - w[1]:.6f}")

t2_strong = tqc.hsp_table(rep_strong, bands=2)
row("valence content", fmt(t2_strong.counts))

dec_strong = tqc.decompose(t2_strong, g_strong, rep=rep_strong)
row(*atomic_limits(dec_strong))

########### 3. delta_t = -0.4 on the same bond ###########

print("\n3. delta_t = -0.4 on the same bond")
m_weak = fu_kane_mele(T, SOC, [-0.4, 0.0, 0.0, 0.0])         # the same bond weakened

r_weak = tqc.realized_group(m_weak, characters=DECLARE)
_m, g_weak, asg_weak, val_weak, rep_weak = r_weak
sites_weak = tqc.wyckoff_sites_tables(g_weak, asg_weak.site_positions)
row("realized group", f"#{g_weak.number_str} {g_weak.name}, orbitals at Wyckoff "
    f"{sites_weak[0].multiplicity}{sites_weak[0].letter}")

hsp_weak = tqc.high_symmetry_points(g_weak)
gaps_weak = []
for point in POINTS:
    w = np.asarray(solve_at(m_weak, hsp_weak[point])[0], dtype=float).reshape(-1)
    gaps_weak.append(w[2] - w[1])
row("gaps GM/T/F/L", " / ".join(f"{g:.6f}" for g in gaps_weak))

t2_weak = tqc.hsp_table(rep_weak, bands=2)
row("valence content", fmt(t2_weak.counts))

dec_weak = tqc.decompose(t2_weak, g_weak, rep=rep_weak)
row(*atomic_limits(dec_weak))

########### 4. three bonds cut: decoupled dimers along [111] ###########

print("\n4. three bonds cut: decoupled dimers along [111]")
m_dimers = fu_kane_mele(T, SOC, [0.0, -1.0, -1.0, -1.0])     # three bonds cut

r_dimers = tqc.realized_group(m_dimers, characters=DECLARE)
_m, g_dimers, asg_dimers, val_dimers, rep_dimers = r_dimers
sites_dimers = tqc.wyckoff_sites_tables(g_dimers, asg_dimers.site_positions)
row("realized group", f"#{g_dimers.number_str} {g_dimers.name}, orbitals at Wyckoff "
    f"{sites_dimers[0].multiplicity}{sites_dimers[0].letter}")

hsp_dimers = tqc.high_symmetry_points(g_dimers)
gaps_dimers = []
for point in POINTS:
    w = np.asarray(solve_at(m_dimers, hsp_dimers[point])[0], dtype=float).reshape(-1)
    gaps_dimers.append(w[2] - w[1])
row("gaps GM/T/F/L", " / ".join(f"{g:.6f}" for g in gaps_dimers))

t2_dimers = tqc.hsp_table(rep_dimers, bands=2)
row("valence content", fmt(t2_dimers.counts))

dec_dimers = tqc.decompose(t2_dimers, g_dimers, rep=rep_dimers)
row(*atomic_limits(dec_dimers))
print("   the valence bands are the antibonding pair of each dimer's two s orbitals,")
print(f"   centred at {dec_dimers.positions[0][0]} on the [111] axis")

########### 5. the content at the eight TRIM ###########

print("\n5. the content at the eight time-reversal-invariant momenta (TRIM)")

# The star of a point is the set of its distinct images under the group:
# group.symmetries are the group's operations, and group.k_action_of(op) is the
# action of one of them on the model's reduced k.
k_actions = []
for op in g_strong.symmetries:
    action = g_strong.k_action_of(op)
    if action is not None:
        k_actions.append(action)

stars = {}
for point in POINTS:
    orbit = set()
    for action in k_actions:
        q = np.mod(np.asarray(action(hsp_strong[point]), dtype=float).reshape(-1), 1.0)
        orbit.add(tuple(np.round(np.where(np.abs(q - 1.0) < 1e-9, 0.0, q), 6)))
    stars[point] = len(orbit)
n_trim = sum(stars.values())
row("k-orbits", ", ".join(f"{p} ({stars[p]})" for p in POINTS)
    + f", covering the cell's {n_trim} TRIM")

# the strong phase against the dimers, point by point, and then the weak one
changed_strong = sorted(p for p in POINTS
                        if t2_strong.content.get(p) != t2_dimers.content.get(p))
n_strong = sum(stars[p] for p in changed_strong)
row("strong vs dimers", f"the content changes at {' and '.join(changed_strong)}, "
    f"{n_strong} of the {n_trim} TRIM")

changed_weak = sorted(p for p in POINTS
                      if t2_weak.content.get(p) != t2_dimers.content.get(p))
n_weak = sum(stars[p] for p in changed_weak)
row("weak vs dimers", f"the content changes at {' and '.join(changed_weak)}, "
    f"{n_weak} of the {n_trim} TRIM")

print(f"   #{g_strong.number_str} has inversion, so nu_0 is the parity product over "
      "these eight points")

########### 6. the symmetry-indicator class in {BS}/{AI} ###########

# The quotient {BS}/{AI} is fixed by the group alone, so a class is a property of
# the content, one step beyond whether a sum of elementary band representations
# carries it.
print("\n6. the symmetry-indicator class in {BS}/{AI}")
si_strong = tqc.symmetry_indicator(t2_strong, g_strong, rep=rep_strong)
si_weak = tqc.symmetry_indicator(t2_weak, g_weak, rep=rep_weak)
si_dimers = tqc.symmetry_indicator(t2_dimers, g_dimers, rep=rep_dimers)

z4_strong = si_strong.irreptables_named["z4"]
z2w_strong = tuple(si_strong.irreptables_named[f"z2w,{i}"] for i in (1, 2, 3))
z4_weak = si_weak.irreptables_named["z4"]
z2w_weak = tuple(si_weak.irreptables_named[f"z2w,{i}"] for i in (1, 2, 3))
z4_dimers = si_dimers.irreptables_named["z4"]
z2w_dimers = tuple(si_dimers.irreptables_named[f"z2w,{i}"] for i in (1, 2, 3))

row("indicator group", si_strong.structure)
print(f"   {'phase':<18} {'class':<8} {'order':<8} {'z4':<4} {'z2w,1/2/3':<11} "
      "nu_0 = z4 mod 2")
print(f"   {'strong':<18} {values(si_strong.values):<8} {si_strong.order:<8} "
      f"{z4_strong:<4} {' '.join(str(v) for v in z2w_strong):<11} {z4_strong % 2}")
print(f"   {'weak':<18} {values(si_weak.values):<8} {si_weak.order:<8} "
      f"{z4_weak:<4} {' '.join(str(v) for v in z2w_weak):<11} {z4_weak % 2}")
print(f"   {'dimers':<18} {values(si_dimers.values):<8} {si_dimers.order:<8} "
      f"{z4_dimers:<4} {' '.join(str(v) for v in z2w_dimers):<11} {z4_dimers % 2}")

paper_strong = f"({z4_strong % 2};{z2w_strong[0]}{z2w_strong[1]}{z2w_strong[2]})"
paper_weak = f"({z4_weak % 2};{z2w_weak[0]}{z2w_weak[1]}{z2w_weak[2]})"
paper_dimers = f"({z4_dimers % 2};{z2w_dimers[0]}{z2w_dimers[1]}{z2w_dimers[2]})"
row("Z2 indices", f"strong {paper_strong}, weak {paper_weak}, dimers {paper_dimers}")

########### 7. the bands along GM-T-F-L-GM ###########

print("\n7. the bands along GM-T-F-L-GM, one panel per phase")
path = [hsp_strong["GM"], hsp_strong["T"], hsp_strong["F"], hsp_strong["L"],
        hsp_strong["GM"]]
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharey=False)

# --- the strong phase ---
kv, kd, kn = m_strong.k_path(path, NK_PATH, report=False)
bands = np.asarray(m_strong.solve_ham(kv))
for i in range(4):
    axes[0].plot(kd, bands[:, i], lw=1.7, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0].axvline(x, color="0.82", lw=0.7)
# the valence pair is Kramers-degenerate at every one of these points, so one
# offset below it clears both bands; the box keeps the label readable where the
# dispersion runs underneath it
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[0].annotate(fmt(t2_strong.content[point]), xy=(x, bands[j, 0]),
                     xytext=(0, -16), textcoords="offset points", ha="center",
                     fontsize=8.5, color="tab:blue", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                               alpha=0.85))
axes[0].set_xticks(kn)
axes[0].set_xticklabels(TICKS, fontsize=13)
axes[0].set_ylim(bands.min() - 0.24 * (bands.max() - bands.min()),
                 bands.max() + 0.08 * (bands.max() - bands.min()))
axes[0].set_ylabel("$E$", fontsize=15)
axes[0].set_title("strong: delta_t = +0.4\n"
                  f"#{g_strong.number_str} {g_strong.name}   class "
                  f"{si_strong.values} in {si_strong.structure}", fontsize=10)

# --- the weak phase ---
kv, kd, kn = m_weak.k_path(path, NK_PATH, report=False)
bands = np.asarray(m_weak.solve_ham(kv))
for i in range(4):
    axes[1].plot(kd, bands[:, i], lw=1.7, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[1].axvline(x, color="0.82", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[1].annotate(fmt(t2_weak.content[point]), xy=(x, bands[j, 0]),
                     xytext=(0, -16), textcoords="offset points", ha="center",
                     fontsize=8.5, color="tab:blue", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                               alpha=0.85))
axes[1].set_xticks(kn)
axes[1].set_xticklabels(TICKS, fontsize=13)
axes[1].set_ylim(bands.min() - 0.24 * (bands.max() - bands.min()),
                 bands.max() + 0.08 * (bands.max() - bands.min()))
axes[1].set_title("weak: delta_t = -0.4\n"
                  f"#{g_weak.number_str} {g_weak.name}   class "
                  f"{si_weak.values} in {si_weak.structure}", fontsize=10)

# --- the dimers ---
kv, kd, kn = m_dimers.k_path(path, NK_PATH, report=False)
bands = np.asarray(m_dimers.solve_ham(kv))
for i in range(4):
    axes[2].plot(kd, bands[:, i], lw=1.7, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[2].axvline(x, color="0.82", lw=0.7)
for point, x in zip(NODES, kn):
    j = int(np.argmin(np.abs(kd - x)))
    axes[2].annotate(fmt(t2_dimers.content[point]), xy=(x, bands[j, 0]),
                     xytext=(0, -16), textcoords="offset points", ha="center",
                     fontsize=8.5, color="tab:blue", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                               alpha=0.85))
axes[2].set_xticks(kn)
axes[2].set_xticklabels(TICKS, fontsize=13)
axes[2].set_ylim(bands.min() - 0.24 * (bands.max() - bands.min()),
                 bands.max() + 0.08 * (bands.max() - bands.min()))
axes[2].set_title("dimers: three bonds cut\n"
                  f"#{g_dimers.number_str} {g_dimers.name}   class "
                  f"{si_dimers.values} in {si_dimers.structure}", fontsize=10)

fig.suptitle("Fu-Kane-Mele on diamond, one bond detuned along [111]: the four "
             "bands along GM-T-F-L-GM, the valence pair in blue with its corep "
             "content at each node.", fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.93))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "fkm_diamond_phases.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
