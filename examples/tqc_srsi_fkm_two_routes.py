# Example of the stable real-space invariants (SRSIs) of a band set, read from
# the bands and read from the orbitals, using pythtqc.
# The model is Fu, Kane and Mele's spin-orbit model on the diamond lattice,
# PRL 98, 106803 (2007): four bands, two occupied, with one of the four
# nearest-neighbour bonds detuned along [111].  It is pythtb's own
# pythtb.models.fu_kane_mele(t, soc, dt), the model examples/tqc_fkm_diamond.py
# reads in momentum space.
# An SRSI is a linear function of the site-symmetry vector -- how many orbitals
# of each site irrep sit at each Wyckoff position -- that no adiabatic,
# symmetry-preserving rearrangement of the orbitals can change.
# The tables are those of Hwang et al., "Stable real-space invariants and
# topology beyond symmetry indicators", Nat. Commun. 17, 8564 (2026),
# 10.1038/s41467-026-74844-w, shipped as pythtqc package data.  The paper's
# theorem: a band set's Z-valued invariants are fractional if and only if its
# symmetry-indicator class is nonzero.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. delta_t = +0.4 on the [111] bond, from the bands: the group, the valence
#    content, its Z-valued invariants, their common denominator and the
#    indicator class
# 2. the same phase from the orbitals: the orbit, its content, the two
#    localizations on the closure of its position, and all four bands
# 3. delta_t = -0.4 and the decoupled dimers: the denominator against the order
#    of the indicator class in all three phases
# 4. splitting(r, bands=2): the no-go criterion on the gap above the valence
#    bands, in each phase
# 5. the bands along GM-T-F-L-GM, and the figure written out

# Author: the pythtqc examples

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb.models import fu_kane_mele

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_counts as fmt
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
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_fkm_two_routes")
NK_PATH = 61 if QUICK else 241              # band-path density: the figure only

# fu_kane_mele(t, soc, dt): ``dt`` offsets the four nearest-neighbour bonds, in
# the order R = [0,0,0], [-1,0,0], [0,-1,0], [0,0,-1].
T = 1.0                    # the nearest-neighbour hopping
SOC = 0.125                # the spin-orbit strength

# A pythtb orbital carries no symmetry character, so the s here is a declaration.
DECLARE = {0: "s"}

TICKS = (r"$\Gamma$", "T", "F", "L", r"$\Gamma$")   # the figure's path

os.makedirs(OUTDIR, exist_ok=True)

print("Fu-Kane-Mele on the diamond lattice, PRL 98, 106803 (2007): the valence bands'")
print("stable real-space invariants, read from the bands and from the orbitals.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def values(theta):
    return "(" + ", ".join(str(v) for v in theta) + ")"


########### 1. delta_t = +0.4 on the [111] bond, from the bands ###########

print("\n1. delta_t = +0.4 on the [111] bond, read from the bands")
m_strong = fu_kane_mele(T, SOC, [0.4, 0.0, 0.0, 0.0])        # one bond strengthened

r_strong = tqc.realized_group(m_strong, characters=DECLARE)
_m, g_strong, asg_strong, val_strong, rep_strong = r_strong
sites = tqc.wyckoff_sites_tables(g_strong, asg_strong.site_positions)
row("realized group", f"#{g_strong.number_str} {g_strong.name}, s orbitals at Wyckoff "
    f"{sites[0].multiplicity}{sites[0].letter}")

t2_strong = tqc.hsp_table(rep_strong, bands=2)               # the valence bands
row("valence content", fmt(t2_strong.counts))

# srsi_of_bands takes what decompose and symmetry_indicator take, the measured
# symmetry vector, and evaluates the invariants on a rational site-symmetry vector
# that induces it.
b_strong = tqc.srsi_of_bands(t2_strong)
si_strong = tqc.symmetry_indicator(t2_strong)
row("srsi_of_bands", f"Z-valued {values(b_strong.theta_Z)}")
row("denominator", b_strong.denominator)
row("indicator class", f"{values(si_strong.values)} in {si_strong.structure}, of order "
    f"{si_strong.order}")
print("   the Z-valued invariants are fractional: no integer site-symmetry vector")
print("   induces the valence content, so the valence bands are no atomic limit")

########### 2. the same phase, from the orbitals ###########

print("\n2. the same phase, read from the orbitals")

# srsi_of_model reads the invariants off the orbitals: for each site orbit it
# measures the content of the orbitals there, writes it as a sum of elementary
# band representations at the positions in the closure of the orbit's position,
# and evaluates the invariants on that sum.
m_srsi = tqc.srsi_of_model(r_strong)
orbit = m_srsi.orbits[0]
row("the orbit", f"orbitals {' and '.join(str(i) for i in orbit.orbitals)} at Wyckoff "
    f"{orbit.multiplicity}{orbit.letter}, not a maximal position")
row("its content", fmt(orbit.content))

# The orbit sits on a line whose closure carries two positions with elementary
# band representations, and its orbitals localize at each of them: one set of
# orbitals moved two ways, with the same invariants.
for loc in orbit.localizations:
    where = loc["terms"][0][2].split("(")[0]
    terms = " + ".join(f"{n} x {name}@{at}" for n, name, at in loc["terms"])
    row(f"localized at {where}", f"{terms},", f"Z-valued {values(loc['theta_Z'])}")

t4_strong = tqc.hsp_table(rep_strong, bands=4)
b4_strong = tqc.srsi_of_bands(t4_strong)
row("all 4 bands", f"Z-valued {values(b4_strong.theta_Z)}, read from the bands")
row("the whole model", f"Z-valued {values(m_srsi.theta_Z)}, read from the orbitals")
print("   all four bands together are the model's own orbitals, so the bands carry the")
print("   orbitals' integers")

########### 3. the other two phases, and the denominator ###########

print("\n3. delta_t = -0.4, and three bonds cut: decoupled dimers along [111]")
m_weak = fu_kane_mele(T, SOC, [-0.4, 0.0, 0.0, 0.0])         # the same bond weakened

r_weak = tqc.realized_group(m_weak, characters=DECLARE)
_m, g_weak, asg_weak, val_weak, rep_weak = r_weak
sites_weak = tqc.wyckoff_sites_tables(g_weak, asg_weak.site_positions)
t2_weak = tqc.hsp_table(rep_weak, bands=2)
b_weak = tqc.srsi_of_bands(t2_weak)
si_weak = tqc.symmetry_indicator(t2_weak)
row("weak,   realized", f"#{g_weak.number_str} {g_weak.name}, orbitals at Wyckoff "
    f"{sites_weak[0].multiplicity}{sites_weak[0].letter}")
row("weak,   valence", fmt(t2_weak.counts))

m_dimers = fu_kane_mele(T, SOC, [0.0, -1.0, -1.0, -1.0])     # three bonds cut

r_dimers = tqc.realized_group(m_dimers, characters=DECLARE)
_m, g_dimers, asg_dimers, val_dimers, rep_dimers = r_dimers
sites_dimers = tqc.wyckoff_sites_tables(g_dimers, asg_dimers.site_positions)
t2_dimers = tqc.hsp_table(rep_dimers, bands=2)
b_dimers = tqc.srsi_of_bands(t2_dimers)
si_dimers = tqc.symmetry_indicator(t2_dimers)
row("dimers, realized", f"#{g_dimers.number_str} {g_dimers.name}, orbitals at Wyckoff "
    f"{sites_dimers[0].multiplicity}{sites_dimers[0].letter}")
row("dimers, valence", fmt(t2_dimers.counts))

# one line per phase: the class on the left, the invariants on the right
print(f"   {'phase':<8} {'class':<8} {'order':<7} "
      f"{'Z-valued invariants of the valence bands':<42} denominator")
for phase, si, b in (("strong", si_strong, b_strong), ("weak", si_weak, b_weak),
                     ("dimers", si_dimers, b_dimers)):
    print(f"   {phase:<8} {values(si.values):<8} {si.order:<7} "
          f"{values(b.theta_Z):<42} {b.denominator}")
print("   the denominator equals the order of the class in all three phases; the theorem")
print("   of Hwang et al. is the equivalence behind it, a nonzero class if and only if a")
print("   fractional Z-valued invariant")

########### 4. the no-go criterion on the gap above the valence bands ###########

print("\n4. splitting(r, bands=2): the no-go criterion on the gap above the valence bands")

# splitting() reads the invariants of all the bands off the orbitals, the Z-valued
# and the Z_n-valued ones, the Z-valued invariants of the valence bands off the bands,
# takes the conduction bands' as the difference, and asks whether the two sides can
# both be atomic insulators.
s_strong = tqc.splitting(r_strong, bands=2)
s_weak = tqc.splitting(r_weak, bands=2)
s_dimers = tqc.splitting(r_dimers, bands=2)
row("strong", "delta_t = +0.4")
print(textwrap.indent(s_strong.summary(), "   "))
row("weak", "delta_t = -0.4")
print(textwrap.indent(s_weak.summary(), "   "))
row("dimers", "three bonds cut")
print(textwrap.indent(s_dimers.summary(), "   "))

########### 5. the bands along GM-T-F-L-GM ###########

print("\n5. the bands along GM-T-F-L-GM, one panel per phase")
hsp = tqc.high_symmetry_points(g_strong)
path = [hsp["GM"], hsp["T"], hsp["F"], hsp["L"], hsp["GM"]]
fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=False)

# --- the strong phase ---
kv, kd, kn = m_strong.k_path(path, NK_PATH, report=False)
bands = np.asarray(m_strong.solve_ham(kv))
for i in range(4):
    axes[0].plot(kd, bands[:, i], lw=1.7, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[0].axvline(x, color="0.82", lw=0.7)
axes[0].set_xticks(kn)
axes[0].set_xticklabels(TICKS, fontsize=13)
axes[0].set_ylabel("$E$", fontsize=15)
axes[0].set_title("strong: delta_t = +0.4\n"
                  f"#{g_strong.number_str} {g_strong.name}   class "
                  f"{si_strong.values} in {si_strong.structure}\n"
                  fr"$\theta_\mathbb{{Z}}[V] = "
                  fr"({', '.join(str(v) for v in b_strong.theta_Z)})$, "
                  f"denominator {b_strong.denominator}", fontsize=9.5)

# --- the weak phase ---
kv, kd, kn = m_weak.k_path(path, NK_PATH, report=False)
bands = np.asarray(m_weak.solve_ham(kv))
for i in range(4):
    axes[1].plot(kd, bands[:, i], lw=1.7, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[1].axvline(x, color="0.82", lw=0.7)
axes[1].set_xticks(kn)
axes[1].set_xticklabels(TICKS, fontsize=13)
axes[1].set_title("weak: delta_t = -0.4\n"
                  f"#{g_weak.number_str} {g_weak.name}   class "
                  f"{si_weak.values} in {si_weak.structure}\n"
                  fr"$\theta_\mathbb{{Z}}[V] = "
                  fr"({', '.join(str(v) for v in b_weak.theta_Z)})$, "
                  f"denominator {b_weak.denominator}", fontsize=9.5)

# --- the dimers ---
kv, kd, kn = m_dimers.k_path(path, NK_PATH, report=False)
bands = np.asarray(m_dimers.solve_ham(kv))
for i in range(4):
    axes[2].plot(kd, bands[:, i], lw=1.7, color="tab:blue" if i < 2 else "tab:red")
for x in kn:
    axes[2].axvline(x, color="0.82", lw=0.7)
axes[2].set_xticks(kn)
axes[2].set_xticklabels(TICKS, fontsize=13)
axes[2].set_title("dimers: three bonds cut\n"
                  f"#{g_dimers.number_str} {g_dimers.name}   class "
                  f"{si_dimers.values} in {si_dimers.structure}\n"
                  fr"$\theta_\mathbb{{Z}}[V] = "
                  fr"({', '.join(str(v) for v in b_dimers.theta_Z)})$, "
                  f"denominator {b_dimers.denominator}", fontsize=9.5)

fig.suptitle("Fu-Kane-Mele on diamond, one bond detuned along [111]: the four bands "
             "along GM-T-F-L-GM, the valence pair in blue, with its stable "
             "real-space invariants and the order of its indicator class.",
             fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.92))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_fkm_two_routes.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
