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
# 1. the group, the two site orbits with their contents and their SRSIs, and
#    the ways the eight bands decompose
# 2. bands 4 and 5: where the Z_n invariants of a band set can be read from
# 3. splitting(r, bands=6): the six occupied bands against the two above them
# 4. splitting(r, bands=4): the same model, the gap two bands lower
# 5. the two gaps, the eight bands along GM-X-M-GM, and the figure

# Author: the pythtqc examples

import os
import sys

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

print("Lin et al., Nat. Commun. 15, 550 (2024), SM Sec. 3G: eight bands from")
print("four spinful orbitals, and the no-go criterion on two different gaps of them.")

########### 1. the model, its two orbits and its total ###########

print("\n1. the model, its two site orbits and its total band representation")
r = tqc.realized_group(lin3g(), typat=TYPAT, characters=DECLARE)
model, group, assignment, validated, rep = r
sites = tqc.wyckoff_sites_tables(group, assignment.site_positions)   # letter and multiplicity
letters = []
for site in sites:
    label = f"{site.multiplicity}{site.letter}"
    if label not in letters:
        letters.append(label)
print(f"   realized group    : #{group.number_str} {group.name}, orbitals at Wyckoff "
      f"{' and '.join(letters)}, {model.nstate} bands")

tables = tqc.srsi_tables(group)
m_srsi = tqc.srsi_of_model(r)
print(m_srsi.summary())

# the two orbits' momentum-space contents, side by side: the SRSIs above
# separate them and the contents below do not
first, second = m_srsi.orbits[0], m_srsi.orbits[1]
head_first = f"the {first.multiplicity}{first.letter} orbit"
head_second = f"the {second.multiplicity}{second.letter} orbit"
print(f"   {head_first:<18}: {fmt(first.content)}")
print(f"   {head_second:<18}: {fmt(second.content)}")
print("                       — one content, told apart only by the mod 2 invariants above")

t8 = tqc.hsp_table(rep, bands=8)
d8 = tqc.decompose(t8, group, rep=rep)
own = (" + ".join(f"{n} x {name}@{where}" for n, name, where in first.chosen["terms"])
       + " + "
       + " + ".join(f"{n} x {name}@{where}" for n, name, where in second.chosen["terms"]))
index = list(d8.solution_strings()).index(own)   # the orbitals' own, among the solutions
print(f"   all eight bands   : {fmt(t8.counts)}")
print(f"   decompose         : {d8.verdict}, {len(d8.solutions)} solutions:")
print("                       "
      + ", ".join("+".join(where) for where in sorted(d8.positions)))
print(f"   -> the orbitals' own localization, {'+'.join(d8.positions[index])}, "
      "is one of the six: the eight")
print("      bands are three elementary band representations at two Wyckoff")
print("      positions, not one of the single-EBR splittings the supplement")
print("      tabulates.  The invariants of all eight, H in the splittings below,")
print("      are the model's own above: the whole Hilbert space is an atomic")
print("      insulator.")

########### 2. bands 4 and 5: where the Z_n invariants come from ###########

print("\n2. bands 4 and 5: the Z_n invariants come from the orbitals, not the bands")
p2 = tqc.hsp_table(rep, bands=[4, 5])
d2 = tqc.decompose(p2, group, rep=rep)
print(f"   bands 4 and 5     : {fmt(p2.counts)}")
print(f"   decompose         : {d2.verdict}, {d2.solution_strings()[0]}")
print("                       (that verdict is about symmetry indicators, nothing finer)")

b2 = tqc.srsi_of_bands(p2, group, rep=rep)
# the band route solves BR.p = y over the rationals; the solution is one of
# many, and the Z_n values differ across them where the Z-SRSIs do not
solved = tables.theta_columns(list(b2.p))
column = tables.theta_columns(list(d2.solutions[0]))
print(f"   srsi_of_bands     : theta_Z = ({', '.join(str(v) for v in b2.theta_Z)}), "
      "and no Z_n values at all")
print(f"   the solved-back p : ({', '.join(str(v) for v in b2.p)}) over the EBR columns, "
      "one rational")
print(f"                       solution of BR.p = y: theta_Z = "
      f"({', '.join(str(v) for v in solved[0])}), theta_Zn = {solved[1]}")
print("   the 1b EBR column : the configuration decompose names above:")
print(f"                       theta_Z = ({', '.join(str(v) for v in column[0])}), "
      f"theta_Zn = {column[1]}")
print("   -> the same Z-SRSIs, different Z2-SRSIs.  The Z_n invariants are not")
print("      functions of the symmetry-data vector, so read them only for a")
print("      configuration that is known: the model's own, an EBR column, or a")
print("      decompose solution.")

########### 3. the six-band gap ###########

print("\n3. splitting(r, bands=6): the six occupied bands against the two above them")
six = tqc.splitting(r, bands=6)
print(six.summary())
print("   -> bands 4 and 5 are among the six, and the obstruction their")
print("      decomposition verdict could not see is this arithmetic.")

########### 4. the four-band gap ###########

print("\n4. splitting(r, bands=4): the same model, the gap two bands lower")
four = tqc.splitting(r, bands=4)
print(four.summary())
print("   -> the same Hamiltonian, the same total, one gap lower: the criterion")
print("      diagnoses the gap that contains bands 4 and 5 and not the other.")

########### 5. the two gaps and the eight bands along GM-X-M-GM ###########

print("\n5. the eight bands along GM-X-M-GM, with the two splittings marked")

# The two gaps are measured on a fixed mesh, the same in quick mode, which thins
# only the figure's path.
mesh = np.stack(np.meshgrid(np.linspace(0, 1, NK_MESH, endpoint=False),
                            np.linspace(0, 1, NK_MESH, endpoint=False),
                            indexing="ij"), -1).reshape(-1, 2)
energies = np.asarray(model.solve_ham(mesh))
gap4 = float(energies[:, 4].min() - energies[:, 3].max())
gap6 = float(energies[:, 6].min() - energies[:, 5].max())
print(f"   the two gaps      : over a {NK_MESH}x{NK_MESH} mesh, "
      f"min E4 - max E3 = {gap4:.6f} and")
print(f"                       min E6 - max E5 = {gap6:.6f} — both splittings are gapped")

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
