# Example of the connectivity of an elementary band representation, using
# pythTQC.  Michel and Zak conjectured that an elementary band representation
# is always connected, that is, that its bands cannot be split into two sets
# with a gap between them at every k.  Two four-band elementary band
# representations are measured here: one is connected, and the other splits
# into two halves.
# The connected one is Appendix F.1 of Flicker et al., PRB 98, 155145 (2018):
# spinless s orbitals at Wyckoff 8a of space group 199 (I2_1 3).
# The disconnected one is the counterexample of Cano et al., PRL 120, 266401
# (2018), at the parameters of its Figure 3a: spinless d_z2 and d_x2-y2
# orbitals at Wyckoff 2a of space group 208 (P4_2 3 2).  They are
# pythtqc.models' i213_flicker2018() and p4232_cano2018().
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. space group 199: the group, and the levels of the four bands at every
#    high-symmetry point
# 2. space group 199: the four bands as one elementary band representation,
#    and the three inter-band gaps minimized over a fixed mesh
# 3. space group 208: the group, with the crystal field the paper adds, and the
#    levels of the four bands at every high-symmetry point
# 4. space group 208: one elementary band representation, and the same three
#    gaps
# 5. space group 208: each half of the split on its own
# 6. the two band structures, and the figure written out

# Author: the pythTQC examples

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import models as tqm
from pythtqc import format_coreps as fmt
from pythtqc import solve_at
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
    "tqc_multifold_connectivity")
NK_PATH = 121 if QUICK else 601      # band-path density: the figures only
NK_MESH = 13                         # the mesh the gaps are minimized over, both models

# A pythtb orbital does not say whether it is s, p or d, so the s here is declared.
DECLARE = {0: "s"}

# GM - H - N - GM - P - H in reduced coordinates of the primitive reciprocal basis
# of space group 199.
PATH199 = [[0, 0, 0], [.5, .5, .5], [0, 0, .5], [0, 0, 0], [.25, .25, .25],
           [.5, .5, .5]]
NODES199 = ("GM", "H", "N", "GM", "P", "H")
TICKS199 = (r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H")

# GM - X - M - GM - R - X in the simple cubic cell of space group 208.
PATH208 = [[0, 0, 0], [0, .5, 0], [.5, .5, 0], [0, 0, 0], [.5, .5, .5],
           [0, .5, 0]]
NODES208 = ("GM", "X", "M", "GM", "R", "X")
TICKS208 = (r"$\Gamma$", "X", "M", r"$\Gamma$", "R", "X")

os.makedirs(OUTDIR, exist_ok=True)

print("Two four-band elementary band representations, and whether their bands can be")
print("separated: space group 199 (Flicker et al., PRB 98, 155145) and space group 208")
print("(Cano et al., PRL 120, 266401).")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One line of the ledger: label, a heading of at most 16 characters, then first,
    then each line of rest beneath it."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def atomic_limits(d):
    """The sums of elementary band representations whose symmetry data vector is that
    of the bands decomposed in d, as a row's heading and its value: the one sum, how
    many there are, or none."""
    if not d.decomposable:
        return "atomic limits", "none"
    sums = [s.label for s in d.candidate_sums]
    if len(sums) == 1:
        return "atomic limit", sums[0]
    return "atomic limits", f"{len(sums)} sums of elementary band representations"


def direct_gaps(gaps):
    """Print gaps[i], the smallest direct gap between bands i and i + 1 over the
    mesh, a line each."""
    for i, gap in enumerate(gaps):
        row(f"bands {i} and {i + 1}", f"smallest direct gap {gap:.6f}"
            + (f" over a {NK_MESH}x{NK_MESH}x{NK_MESH} mesh" if i == 0 else ""))


########### 1. space group 199: the group and the levels ###########

print("\n1. space group 199: the group, and the levels of the four bands")
m199 = tqm.i213_flicker2018()

r199 = tqc.realized_group(m199, characters=DECLARE)
g199 = r199.group
row("realized group", f"#{g199.number_str} {g199.name}, s orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(o.wyckoff for o in r199.wyckoff_orbits))}, "
    f"{m199.nstate} bands")

# One diagonalization per high-symmetry point.  coreps_at names the corep each
# level carries, read off its eigenvectors.  A level is printed as its corep, its
# dimension in parentheses, and its energy.
hsp199 = tqc.high_symmetry_points(g199)
levels199 = []                     # (k-point, energy, corep label, degeneracy)
for point in sorted(hsp199):
    k = hsp199[point]
    out = tqc.coreps_at(r199, k, states=solve_at(m199, k), bands=4)
    rows = []
    for m in out.multiplets:
        levels199.append((point, m.energy, m.label, len(m.bands)))
        rows.append(f"{m.label}({len(m.bands)}) at {m.energy:+.6f}")
    row(point, ", ".join(rows))
threefold = sorted({lab for _p, _e, lab, dim in levels199 if dim == 3})
print(f"   {' and '.join(threefold)} are threefold: the spin-1 fermions")

########### 2. space group 199: one elementary band representation, and the gaps ###########

print("\n2. space group 199: the four bands together, and the three inter-band gaps")
t199 = tqc.hsp_table(r199, bands=4)          # the coreps at every point, in one table
d199 = tqc.decompose(t199)
row("all 4 bands", fmt(t199.symmetry_data))
row(*atomic_limits(d199))

mesh199 = m199.k_uniform_mesh([NK_MESH] * 3)
ev199 = np.asarray(m199.solve_ham(mesh199))
gaps199 = [float((ev199[:, i + 1] - ev199[:, i]).min()) for i in range(3)]
direct_gaps(gaps199)
print("   every gap closes somewhere, so no subset of the four bands is isolated at any")
print("   filling. This elementary band representation is connected")

########### 3. space group 208: the group, the crystal field, and the levels ###########

print("\n3. space group 208: the group, the crystal field the paper adds, and the levels")
m208 = tqm.p4232_cano2018()

# The two 2a sites by themselves are a body-centered point set, of space group
# Im-3m.  The paper adds atoms that carry no orbital and "create a crystal field
# with the symmetry of P4_232".  p4232_symmetry_cell() is one such set, the
# 24-fold orbit of a generic point.  decorations= puts it in the cell the group
# is identified from, so the cell's own group is P4_2321' and no operation is
# broken by the hoppings.  Without it, the hoppings lower Im-3m1' to the same
# group, and the levels and the decomposition below are the same.
cell = tqm.p4232_symmetry_cell()
field = [(p.tolist(), int(t)) for p, t in zip(cell["positions"], cell["typat"])
         if int(t) != cell["typat"][0]]            # the non-orbital species alone

# Each 2a site has the site-symmetry group 231'.  Its two orbitals carry the
# two-dimensional site irrep 1E2E, the one d_z2 and d_x2-y2 carry.  The
# operations act on them by the matrices of d_z2 and d_x2-y2.
r208 = tqc.realized_group(m208, decorations=field)
g208 = r208.group
row("realized group", f"#{g208.number_str} {g208.name}, orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(o.wyckoff for o in r208.wyckoff_orbits))}, "
    f"{m208.nstate} bands")
row("crystal field", f"{len(field)} atoms that carry no orbital, of the kind the paper adds")

hsp208 = tqc.high_symmetry_points(g208)
levels208 = []
for point in sorted(hsp208):
    k = hsp208[point]
    out = tqc.coreps_at(r208, k, states=solve_at(m208, k), bands=4)
    rows = []
    for m in out.multiplets:
        levels208.append((point, m.energy, m.label, len(m.bands)))
        rows.append(f"{m.label}({len(m.bands)}) at {m.energy:+.6f}")
    row(point, ", ".join(rows))

########### 4. space group 208: one elementary band representation, and the gaps ###########

print("\n4. space group 208: the four bands together, and the three inter-band gaps")
t208 = tqc.hsp_table(r208, bands=4)
d208 = tqc.decompose(t208)
row("all 4 bands", fmt(t208.symmetry_data))
row(*atomic_limits(d208))

mesh208 = m208.k_uniform_mesh([NK_MESH] * 3)
ev208 = np.asarray(m208.solve_ham(mesh208))
gaps208 = [float((ev208[:, i + 1] - ev208[:, i]).min()) for i in range(3)]
indirect208 = float(ev208[:, 2].min() - ev208[:, 1].max())
direct_gaps(gaps208)
row("half filling", f"indirect gap {indirect208:.6f}")
print("   the gap between bands 1 and 2 stays open over the mesh. This elementary band")
print("   representation splits at half filling into isolated halves of two bands each")

########### 5. space group 208: each half on its own ###########

print("\n5. space group 208: each half of the split on its own")

# hsp_table also takes the indices of the bands to read, so each half is read
# by the same call as the whole set in section 4.
table_lo = tqc.hsp_table(r208, bands=(0, 1))
d_lo = tqc.decompose(table_lo)
row("bands 0 and 1", fmt(table_lo.symmetry_data))
row(*atomic_limits(d_lo))

table_hi = tqc.hsp_table(r208, bands=(2, 3))
d_hi = tqc.decompose(table_hi)
row("bands 2 and 3", fmt(table_hi.symmetry_data))
row(*atomic_limits(d_hi))
print("   each isolated half is topological")

########### 6. the two band structures ###########

print("\n6. the bands of both models, one panel each")
fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

# Space group 199, along GM-H-N-GM-P-H
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
    if point not in xof199:        # PA, the point -P, is not on this path
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
axes[0].set_title(f"space group 199, #{g199.number_str} {g199.name}\n"
                  f"{d199.candidate_sums[0].label}\nconnected: every inter-band gap "
                  "closes somewhere", fontsize=10)

# Space group 208, along GM-X-M-GM-R-X
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
axes[1].set_title(f"space group 208, #{g208.number_str} {g208.name}\n"
                  f"{d208.candidate_sums[0].label}\nsplit at half filling, and "
                  "neither half is an atomic limit", fontsize=10)

fig.suptitle("One elementary band representation each, the same measurement: the "
             "four bands of space group 199 cannot be separated by a gap, and those "
             "of space group 208 are split at half filling.", fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.955))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "multifold_connectivity.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
