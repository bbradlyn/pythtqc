# Example of the SRSI no-go criterion applied to a composite band
# representation, using pythtqc.
# The model is the spinful fragile-plus-EBR model of Lin et al., Nat. Commun.
# 15, 550 (2024), 10.1038/s41467-024-44762-w, SM Sec. 3G --
# pythtqc.models.lin3g.  Four orbitals, eight bands, wallpaper group p4mm: two
# orbitals at the cell origin carrying the paper's fragile quartet and two at
# the edge centres carrying an s EBR, so the eight bands together are three
# elementary band representations at two Wyckoff positions rather than the
# single one whose splittings the paper's supplement tabulates.  The question
# is what the criterion says about two different gaps of that one Hamiltonian.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. the group, the two site orbits with their contents, the atomic limits of
#    all eight bands, and the stable real-space invariants of the model
# 2. bands 4 and 5: where the Z2-valued invariants of a band set can be read from
# 3. splitting(r, bands=6): the six occupied bands against the two above them
# 4. splitting(r, bands=4): the same model, the gap two bands lower
# 5. the two gaps, the eight bands along GM-X-M-GM, and the figure

# Author: the pythtqc examples

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_counts as fmt
from pythtqc.models import lin3g
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
    "tqc_srsi_lin3g_composite_splitting")
NK_PATH = 61 if QUICK else 241              # band-path density: the figure only
NK_MESH = 41                                # the gap mesh, fixed in both modes

# orbitals 0 and 1 are one species, orbitals 2 and 3 another; the two orbitals
# of the first carry different site irreps, which is a declaration (the hoppings
# cannot measure what an orbital is; they measure which operations are
# symmetries once the orbitals are stated).
TYPAT = [0, 0, 1, 1]
DECLARE = ["s", "dx2-y2", "s", "s"]

PATH_LABELS = (r"$\Gamma$", "X", "M", r"$\Gamma$")

os.makedirs(OUTDIR, exist_ok=True)

print("Lin et al., Nat. Commun. 15, 550 (2024), SM Sec. 3G: eight bands from four")
print("spinful orbitals, and the no-go criterion on two gaps of the one Hamiltonian.")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def values(theta):
    return "(" + ", ".join(str(v) for v in theta) + ")"


def moduli(theta_zn):
    return ", ".join(f"mod {n} {values(theta)}" for n, theta in sorted(theta_zn.items()))


def atomic_limits(d):
    """The atomic limits of a band set, as a label and a value: the one sum of elementary
    band representations with this content, how many sums there are, or none."""
    if d.verdict == "none":
        return "atomic limits", "none"
    sums = d.solution_strings()
    if len(sums) == 1:
        return "atomic limit", sums[0]
    return "atomic limits", f"{len(sums)} sums of elementary band representations"


########### 1. the model, its two orbits and all eight bands ###########

print("\n1. the model, its two site orbits and all eight bands")
r = tqc.realized_group(lin3g(), typat=TYPAT, characters=DECLARE)
model, group, assignment, validated, rep = r
sites = tqc.wyckoff_sites_tables(group, assignment.site_positions)   # letter and multiplicity
letters = []
for site in sites:
    label = f"{site.multiplicity}{site.letter}"
    if label not in letters:
        letters.append(label)
row("realized group", f"#{group.number_str} {group.name}, orbitals at Wyckoff "
    f"{' and '.join(letters)}, {model.nstate} bands")

tables = tqc.srsi_tables(group)
m_srsi = tqc.srsi_of_model(r)
first, second = m_srsi.orbits[0], m_srsi.orbits[1]
row(f"the {first.multiplicity}{first.letter} orbit", fmt(first.content))
row(f"the {second.multiplicity}{second.letter} orbit", fmt(second.content))

t8 = tqc.hsp_table(rep, bands=8)
d8 = tqc.decompose(t8, group, rep=rep)
own_terms = [" + ".join(f"{n} x {name}@{where}" for n, name, where in orbit.chosen["terms"])
             for orbit in (first, second)]
index = list(d8.solution_strings()).index(" + ".join(own_terms))   # the orbitals' own sum
row("all 8 bands", fmt(t8.counts))
row("atomic limits", f"{atomic_limits(d8)[1]}, at",
    ", ".join("+".join(where) for where in sorted(d8.positions)))
row("the model's own", f"{'+'.join(d8.positions[index])}, {own_terms[0]}",
    f"+ {own_terms[1]}")
row("srsi_of_model(r)", "the invariants of the model and of each site orbit")
print(textwrap.indent(m_srsi.summary(), "   "))
print("   the two orbits have one content and different Z2-valued invariants; the eight")
print("   bands are three elementary band representations at two Wyckoff positions, an")
print("   atomic insulator, not the one whose splittings the supplement tabulates")

########### 2. bands 4 and 5: where the Z2-valued invariants come from ###########

print("\n2. bands 4 and 5: the Z2-valued invariants come from the orbitals, not the bands")
p2 = tqc.hsp_table(rep, bands=[4, 5])
d2 = tqc.decompose(p2, group, rep=rep)
row("bands 4 and 5", fmt(p2.counts))
row(*atomic_limits(d2))

b2 = tqc.srsi_of_bands(p2, group, rep=rep)
row("srsi_of_bands", f"Z-valued {values(b2.theta_Z)}, and no Z_n-valued invariants")


def signed_sum(p):
    """A vector over the group's elementary band representations, in the invariants' notation."""
    parts = []
    for m, (irrep, letter) in zip(p, tables.ebrs):
        if m:
            sign = "- " if m < 0 else ("+ " if parts else "")
            parts.append(f"{sign}{abs(m)} ({irrep})_{letter}")
    return " ".join(parts)


# two vectors over the elementary band representations with this content: the
# one atomic limit, and the rational vector the band route evaluates, one of
# many with the same content
column = tables.theta_columns(list(d2.solutions[0]))
solved = tables.theta_columns(list(b2.p))
row("the atomic limit", f"{signed_sum(d2.solutions[0])}, Z-valued {values(column[0])}, "
    f"{moduli(column[1])}")
row("a signed sum", f"{signed_sum(b2.p)}, Z-valued {values(solved[0])}, "
    f"{moduli(solved[1])}")
print("   the Z-valued invariants agree and the Z2-valued ones differ: the Z_n-valued")
print("   invariants are not functions of the content, so a band set has them only through")
print("   a set of orbitals known to carry it, the model's own or an atomic limit")

########### 3. the gap between bands 5 and 6 ###########

print("\n3. splitting(r, bands=6): the six occupied bands against the two above them")
six = tqc.splitting(r, bands=6)
print(textwrap.indent(six.summary(), "   "))
print("   bands 4 and 5 are among the six: what their atomic limit in section 2 could not")
print("   show, the Z2-valued invariants of the two sides of this gap do")

########### 4. the gap between bands 3 and 4 ###########

print("\n4. splitting(r, bands=4): the same model, the gap two bands lower")
four = tqc.splitting(r, bands=4)
print(textwrap.indent(four.summary(), "   "))
print("   the same Hamiltonian and the same eight bands, one gap lower: the criterion")
print("   diagnoses the gap between bands 5 and 6, whose valence side holds bands 4 and 5,")
print("   and not the gap between bands 3 and 4")

########### 5. the two gaps and the eight bands along GM-X-M-GM ###########

print("\n5. the eight bands along GM-X-M-GM, with the two gaps marked")

# The two gaps are measured on a fixed mesh, the same in quick mode, which thins
# only the figure's path.
mesh = np.stack(np.meshgrid(np.linspace(0, 1, NK_MESH, endpoint=False),
                            np.linspace(0, 1, NK_MESH, endpoint=False),
                            indexing="ij"), -1).reshape(-1, 2)
energies = np.asarray(model.solve_ham(mesh))
gap4 = float(energies[:, 4].min() - energies[:, 3].max())
gap6 = float(energies[:, 6].min() - energies[:, 5].max())
row("bands 3 and 4", f"gap {gap4:.6f} over a {NK_MESH}x{NK_MESH} mesh")
row("bands 5 and 6", f"gap {gap6:.6f}")
print("   both gaps are open")

hsp = tqc.high_symmetry_points(group)
path = [hsp["GM"], hsp["X"], hsp["M"], hsp["GM"]]
kv, kd, kn = model.k_path(path, NK_PATH, report=False)
bands = np.asarray(model.solve_ham(kv))

fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.4))
for i in range(8):
    ax.plot(kd, bands[:, i], lw=1.8,
            color="tab:blue" if i < 4 else ("tab:purple" if i < 6 else "tab:red"))
for x in kn:
    ax.axvline(x, color="0.82", lw=0.7)
mid4 = 0.5 * (bands[:, 3].max() + bands[:, 4].min())
mid6 = 0.5 * (bands[:, 5].max() + bands[:, 6].min())
ax.axhline(mid4, color="0.35", ls="--", lw=1.1)
ax.axhline(mid6, color="0.35", ls=":", lw=1.1)
ax.annotate(f"4-band gap ({gap4:.3f}):  {four.verdict}\n"
            fr"$\theta_\mathbb{{Z}}[V]=({', '.join(str(v) for v in four.theta_Z_v)})$",
            xy=(0.995, mid4), xycoords=("axes fraction", "data"),
            xytext=(0, 6), textcoords="offset points",
            ha="right", fontsize=9, color="0.15",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9))
ax.annotate(f"6-band gap ({gap6:.3f}):  {six.verdict}\n"
            fr"$\theta_\mathbb{{Z}}[V]=({', '.join(str(v) for v in six.theta_Z_v)})$",
            xy=(0.995, mid6), xycoords=("axes fraction", "data"),
            xytext=(0, 6), textcoords="offset points",
            ha="right", fontsize=9, color="0.15",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9))
ax.set_xticks(kn)
ax.set_xticklabels(PATH_LABELS, fontsize=13)
ax.set_xlim(kd[0], kd[-1])
ax.set_ylabel("$E$", fontsize=15)
ax.set_title(f"lin3g in #{group.number_str} {group.name}: the eight bands are "
             f"{'+'.join(d8.positions[index])},\n"
             fr"$\theta_\mathbb{{Z}}[H]=({', '.join(str(v) for v in m_srsi.theta_Z)})$, "
             fr"$\theta_{{\mathbb{{Z}}_2}}[H]={m_srsi.theta_Zn[2]}$"
             f" — the six-band gap {six.verdict}, the four-band gap {four.verdict}",
             fontsize=9)
fig.tight_layout(pad=0.6)
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_lin3g_composite_splitting.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
