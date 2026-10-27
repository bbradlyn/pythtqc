# Example of the connectivity of an elementary band representation, using
# pythtqc.  Michel and Zak conjectured that an elementary band representation
# is always connected -- that its bands can never be separated into isolated
# subsets anywhere in the zone.  Two four-band elementary band representations
# are measured here, and they answer in opposite directions.
# The connected one is App. F.1 of Flicker et al., PRB 98, 155145 (2018):
# spinless s orbitals at Wyckoff 8a of SG 199 (I2_1 3).
# The disconnected one is the counterexample of Cano et al., PRL 120, 266401
# (2018), at the Fig. 3a parameters: spinless (d_z2, d_x2-y2) at Wyckoff 2a of
# SG 208 (P4_2 3 2).  They are pythtqc.models' i213_flicker2018() and
# p4232_cano2018().
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. SG 199: the group, and the levels of the four bands at every high-symmetry
#    point
# 2. SG 199: the four bands as one elementary band representation, and the
#    three inter-band gaps minimised over a fixed mesh
# 3. SG 208: the group, with the crystal field the paper draws, and the levels
#    of the four bands at every high-symmetry point
# 4. SG 208: one elementary band representation, and the same gap census
# 5. SG 208: each half of the split on its own
# 6. the two band structures, and the figure written out

# Author: the pythtqc examples

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import models as tqm
from pythtqc import format_counts as fmt
from pythtqc import solve_at
from pythtqc.utils import save_figure

# Elementary band representations print as ρ↑G, the site irrep ρ induced to the
# group; a Windows console cannot print ↑ by default, so the output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band paths and nothing else.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures",
    "tqc_multifold_connectivity")
NK_PATH = 121 if QUICK else 601      # band-path density: the figures only
NK_MESH = 13                         # the gap census mesh, the same for both models

# A pythtb orbital carries no symmetry character, so the s here is a declaration.
DECLARE = {0: "s"}

# GM - H - N - GM - P - H in the primitive body-centred cell of SG 199.
PATH199 = [[0, 0, 0], [.5, .5, .5], [0, 0, .5], [0, 0, 0], [.25, .25, .25],
           [.5, .5, .5]]
NODES199 = ("GM", "H", "N", "GM", "P", "H")
TICKS199 = (r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H")

# GM - X - M - GM - R - X in the simple cubic cell of SG 208.
PATH208 = [[0, 0, 0], [0, .5, 0], [.5, .5, 0], [0, 0, 0], [.5, .5, .5],
           [0, .5, 0]]
NODES208 = ("GM", "X", "M", "GM", "R", "X")
TICKS208 = (r"$\Gamma$", "X", "M", r"$\Gamma$", "R", "X")

os.makedirs(OUTDIR, exist_ok=True)

print("Two four-band elementary band representations, and whether their bands can")
print("be separated: SG 199 (Flicker et al., PRB 98, 155145) and SG 208 (Cano et")
print("al., PRL 120, 266401).")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def atomic_limits(d):
    """The atomic limits of a band set, as a label and a value: the one sum of elementary
    band representations with this content, how many sums there are, or none."""
    if d.verdict == "none":
        return "atomic limits", "none"
    sums = d.solution_strings()
    if len(sums) == 1:
        return "atomic limit", sums[0]
    return "atomic limits", f"{len(sums)} sums of elementary band representations"


def direct_gaps(gaps):
    """The smallest direct gap between each two neighbouring bands over the mesh, a line each."""
    for i, gap in enumerate(gaps):
        row(f"bands {i} and {i + 1}", f"smallest direct gap {gap:.6f}"
            + (f" over a {NK_MESH}x{NK_MESH}x{NK_MESH} mesh" if i == 0 else ""))


########### 1. SG 199: the group and the levels ###########

print("\n1. SG 199: the group, and the levels of the four bands")
m199 = tqm.i213_flicker2018()

out199 = tqc.realized_group(m199, characters=DECLARE)
_m, g199, asg199, val199, rep199 = out199
sites199 = tqc.wyckoff_sites_tables(g199, asg199.site_positions)   # letter and multiplicity
row("realized group", f"#{g199.number_str} {g199.name}, s orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(f'{s.multiplicity}{s.letter}' for s in sites199))}, "
    f"{m199.nstate} bands")

# one diagonalization per high-symmetry point: corep_content reads the
# characters of each level off its eigenvectors and names the corep it carries;
# each level is printed as its corep, its dimension in parentheses, and its energy.
hsp199 = tqc.high_symmetry_points(g199)
levels199 = []                     # (k-point, energy, corep label, degeneracy)
for point in sorted(hsp199):
    k = hsp199[point]
    out = tqc.corep_content(rep199, k, states=solve_at(m199, k), bands=4)
    rows = []
    for multiplet, dim, counts in zip(out.table.multiplets, out.table.dims, out.labels):
        energy = float(out.table.spectrum[multiplet[0]])
        levels199.append((point, energy, fmt(counts), int(dim)))
        rows.append(f"{fmt(counts)}({int(dim)}) at {energy:+.6f}")
    row(point, ", ".join(rows))
threefold = sorted({lab for _p, _e, lab, dim in levels199 if dim == 3})
print(f"   {' and '.join(threefold)} are threefold: the spin-1 fermions")

########### 2. SG 199: one elementary band representation, and the gaps ###########

print("\n2. SG 199: the four bands together, and the three inter-band gaps")
t199 = tqc.hsp_table(rep199, bands=4)          # the content at every point, in one table
d199 = tqc.decompose(t199)
row("all 4 bands", fmt(t199.counts))
row(*atomic_limits(d199))

mesh199 = m199.k_uniform_mesh([NK_MESH] * 3)
ev199 = np.asarray(m199.solve_ham(mesh199))
gaps199 = [float((ev199[:, i + 1] - ev199[:, i]).min()) for i in range(3)]
direct_gaps(gaps199)
print("   every gap closes somewhere, so no subset of the four bands is isolated at any")
print("   filling: this elementary band representation is connected")

########### 3. SG 208: the group, the crystal field, and the levels ###########

print("\n3. SG 208: the group, the crystal field the paper draws, and the levels")
m208 = tqm.p4232_cano2018()

# The two 2a sites by themselves are a body-centred point set, whose symmetry is
# far above the group the hoppings obey.  The paper's own device is to draw
# extra, non-orbital atoms that "create a crystal field with the symmetry of
# P4_232"; p4232_symmetry_cell() is one such set, the 24-fold orbit of a generic
# point, and decorations= puts it in the cell the group is identified from.
cell = tqm.p4232_symmetry_cell()
field = [(p.tolist(), int(t)) for p, t in zip(cell["positions"], cell["typat"])
         if int(t) != cell["typat"][0]]            # the non-orbital species alone

# The two orbitals of each 2a site are carried into one another by its own
# site-symmetry group 23, so they are one two-dimensional multiplet: the
# doublet a d_z2, d_x2-y2 pair carries there, whose matrices are those two
# functions' own.
out208 = tqc.realized_group(m208, decorations=field)
_m, g208, asg208, val208, rep208 = out208
sites208 = tqc.wyckoff_sites_tables(g208, asg208.site_positions)
row("realized group", f"#{g208.number_str} {g208.name}, orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(f'{s.multiplicity}{s.letter}' for s in sites208))}, "
    f"{m208.nstate} bands")
row("crystal field", f"{len(field)} atoms that carry no orbital, the paper's own device")

hsp208 = tqc.high_symmetry_points(g208)
levels208 = []
for point in sorted(hsp208):
    k = hsp208[point]
    out = tqc.corep_content(rep208, k, states=solve_at(m208, k), bands=4)
    rows = []
    for multiplet, dim, counts in zip(out.table.multiplets, out.table.dims, out.labels):
        energy = float(out.table.spectrum[multiplet[0]])
        levels208.append((point, energy, fmt(counts), int(dim)))
        rows.append(f"{fmt(counts)}({int(dim)}) at {energy:+.6f}")
    row(point, ", ".join(rows))

########### 4. SG 208: one elementary band representation, and the gaps ###########

print("\n4. SG 208: the four bands together, and the three inter-band gaps")
t208 = tqc.hsp_table(rep208, bands=4)
d208 = tqc.decompose(t208)
row("all 4 bands", fmt(t208.counts))
row(*atomic_limits(d208))

mesh208 = m208.k_uniform_mesh([NK_MESH] * 3)
ev208 = np.asarray(m208.solve_ham(mesh208))
gaps208 = [float((ev208[:, i + 1] - ev208[:, i]).min()) for i in range(3)]
indirect208 = float(ev208[:, 2].min() - ev208[:, 1].max())
direct_gaps(gaps208)
row("half filling", f"indirect gap {indirect208:.6f}")
print("   the gap between bands 1 and 2 stays open, so this elementary band representation")
print("   splits at half filling into two isolated halves of two bands each")

########### 5. SG 208: each half on its own ###########

print("\n5. SG 208: each half of the split on its own")

# hsp_table takes the indices of the bands to read, so each half is asked
# exactly as the whole set was in section 4.
table_lo = tqc.hsp_table(rep208, bands=(0, 1))
d_lo = tqc.decompose(table_lo)
row("bands 0 and 1", fmt(table_lo.counts))
row(*atomic_limits(d_lo))

table_hi = tqc.hsp_table(rep208, bands=(2, 3))
d_hi = tqc.decompose(table_hi)
row("bands 2 and 3", fmt(table_hi.counts))
row(*atomic_limits(d_hi))
print("   each isolated half is topological")

########### 6. the two band structures ###########

print("\n6. the bands of both models, one panel each")
fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

# --- SG 199, along GM-H-N-GM-P-H ---
kv199, kd199, kn199 = m199.k_path(PATH199, NK_PATH, report=False)
bands199 = np.asarray(m199.solve_ham(kv199))
xof199 = {}                        # k-point name -> the first x on the path it sits at
for name, x in zip(NODES199, kn199):
    xof199.setdefault(name, x)
for i in range(4):
    axes[0].plot(kd199, bands199[:, i], lw=1.7, color="tab:blue")
for x in kn199:
    axes[0].axvline(x, color="0.78", lw=0.7)
for point, energy, label, dim in levels199:
    if point not in xof199:        # PA is a k-point of the group, not a node here
        continue
    x = xof199[point]
    first = abs(x - kn199[0]) < 1e-12
    if dim >= 2:                   # circle the measured degeneracies
        axes[0].plot([x], [energy], marker="o", ms=10 if dim > 2 else 6.5,
                     mfc="none", mec="tab:red", mew=2.0, zorder=5)
    axes[0].annotate(f"{label} ({dim}-fold)" if dim >= 2 else label, xy=(x, energy),
                     xytext=(6 if first else -6, 7), ha="left" if first else "right",
                     textcoords="offset points", fontsize=7.5,
                     color="tab:red" if dim >= 3 else "0.2", zorder=6)
axes[0].set_xticks(kn199)
axes[0].set_xticklabels(TICKS199, fontsize=13)
axes[0].set_xlim(kd199[0] - 0.10 * kd199[-1], kd199[-1] * 1.13)
axes[0].set_ylim(bands199.min() - 0.10 * (bands199.max() - bands199.min()),
                 bands199.max() + 0.13 * (bands199.max() - bands199.min()))
axes[0].set_ylabel("$E$", fontsize=16)
axes[0].set_title(f"SG 199, #{g199.number_str} {g199.name}\n"
                  f"{d199.solution_strings()[0]}\nconnected: every inter-band gap "
                  "closes somewhere", fontsize=10)

# --- SG 208, along GM-X-M-GM-R-X ---
kv208, kd208, kn208 = m208.k_path(PATH208, NK_PATH, report=False)
bands208 = np.asarray(m208.solve_ham(kv208))
xof208 = {}
for name, x in zip(NODES208, kn208):
    xof208.setdefault(name, x)
axes[1].axhspan(bands208[:, 1].max(), bands208[:, 2].min(), color="0.88", zorder=0)
axes[1].annotate(f"gap {gaps208[1]:.4f}", xy=(kd208[-1] * 0.5,
                 0.5 * (bands208[:, 1].max() + bands208[:, 2].min())),
                 ha="center", va="center", fontsize=9, color="0.25")
for i in range(4):
    axes[1].plot(kd208, bands208[:, i], lw=1.7, color="tab:blue")
for x in kn208:
    axes[1].axvline(x, color="0.78", lw=0.7)
for point, energy, label, dim in levels208:
    if point not in xof208:
        continue
    x = xof208[point]
    first = abs(x - kn208[0]) < 1e-12
    if dim >= 2:
        axes[1].plot([x], [energy], marker="o", ms=10 if dim > 2 else 6.5,
                     mfc="none", mec="tab:red", mew=2.0, zorder=5)
    axes[1].annotate(f"{label} ({dim}-fold)" if dim >= 2 else label, xy=(x, energy),
                     xytext=(6 if first else -6, 7), ha="left" if first else "right",
                     textcoords="offset points", fontsize=7.5,
                     color="tab:red" if dim >= 3 else "0.2", zorder=6)
axes[1].set_xticks(kn208)
axes[1].set_xticklabels(TICKS208, fontsize=13)
axes[1].set_xlim(kd208[0] - 0.10 * kd208[-1], kd208[-1] * 1.13)
axes[1].set_ylim(bands208.min() - 0.10 * (bands208.max() - bands208.min()),
                 bands208.max() + 0.13 * (bands208.max() - bands208.min()))
axes[1].set_title(f"SG 208, #{g208.number_str} {g208.name}\n"
                  f"{d208.solution_strings()[0]}\nsplit at half filling, and "
                  "neither half is an atomic limit", fontsize=10)

fig.suptitle("One elementary band representation each, the same measurement: the "
             "four bands of SG 199 cannot be separated anywhere, and those of "
             "SG 208 are split at half filling.", fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.955))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "multifold_connectivity.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
