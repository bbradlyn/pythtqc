# Example of the stable real-space invariants (SRSIs) of a band set, read from
# the bands and read from the orbitals, using pythtqc.
# The model is Fu, Kane and Mele's spin-orbit model on the diamond lattice,
# PRL 98, 106803 (2007): four bands, two occupied, with one of the four
# nearest-neighbour bonds detuned along [111].  It is pythtb's own
# pythtb.models.fu_kane_mele(t, soc, dt), the model examples/tqc_fkm_diamond.py
# reads in momentum space.
# An SRSI is a linear form theta = Theta.p on the site-symmetry vector p -- how
# many orbitals of each site-symmetry irrep sit at each Wyckoff position -- that
# no adiabatic, symmetry-preserving rearrangement of the orbitals can change.
# The tables are those of Hwang et al., "Stable real-space invariants and
# topology beyond symmetry indicators", Nat. Commun. 17, 8564 (2026),
# 10.1038/s41467-026-74844-w, shipped as pythtqc package data.  The paper's
# theorem: a band set's Z-valued SRSIs are fractional if and only if its
# symmetry-indicator class is nonzero.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. delta_t = +0.4 on the [111] bond, from the bands: the group, the valence
#    content, its Z-valued SRSIs and their common denominator
# 2. the same phase from the orbitals: the orbit, its content, the two
#    localizations on the closure of its position, and all four bands
# 3. delta_t = -0.4 and the decoupled dimers: the denominator against the order
#    of the indicator class in all three phases
# 4. the no-go criterion on the valence/conduction gap of each phase
# 5. the bands along GM-T-F-L-GM, and the figure written out

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

print("Fu-Kane-Mele on the diamond lattice, PRL 98, 106803 (2007): the valence")
print("pair's stable real-space invariants, from the bands and from the orbitals.")

########### 1. delta_t = +0.4 on the [111] bond, from the bands ###########

print("\n1. delta_t = +0.4 on the [111] bond, read from the bands")
m_strong = fu_kane_mele(T, SOC, [0.4, 0.0, 0.0, 0.0])        # one bond strengthened

r_strong = tqc.realized_group(m_strong, characters=DECLARE)
_m, g_strong, asg_strong, val_strong, rep_strong = r_strong
sites = tqc.wyckoff_sites_tables(g_strong, asg_strong.site_positions)
print(f"   realized group   : #{g_strong.number_str} {g_strong.name}, s orbitals at "
      f"Wyckoff {sites[0].multiplicity}{sites[0].letter}")
print("                      (measured from the Hamiltonian: the detuned bond lowers")
print("                      Fd-3m to R-3m, although no atom moved)")

t2_strong = tqc.hsp_table(rep_strong, bands=2)               # the valence pair
print(f"   valence content  : {fmt(t2_strong.counts)}")

# srsi_of_bands takes what decompose and symmetry_indicator take -- the measured
# symmetry vector -- solves BR.p = y over the rationals and evaluates Theta.p.
b_strong = tqc.srsi_of_bands(t2_strong, g_strong, rep=rep_strong)
si_strong = tqc.symmetry_indicator(t2_strong, g_strong, rep=rep_strong)
print(f"   theta_Z          : ({', '.join(str(v) for v in b_strong.theta_Z)})")
print(f"   the denominator  : {b_strong.denominator}; the indicator class is "
      f"{si_strong.values} of {si_strong.structure}, order {si_strong.order}")
print("                      (fractional: no integer site-symmetry vector reproduces")
print("                      the valence pair, so it is no atomic limit)")

########### 2. the same phase, from the orbitals ###########

print("\n2. the same phase, read from the orbitals")

# srsi_of_model reads the configuration off the model: for each orbit of sites it
# measures the symmetry content of the orbitals there, decomposes it over the EBRs
# the tables list at the positions in the closure of the orbit's position, and
# evaluates Theta on the vector that comes out.
m_srsi = tqc.srsi_of_model(r_strong)
orbit = m_srsi.orbits[0]
print(f"   the orbit        : orbitals {list(orbit.orbitals)} at Wyckoff "
      f"{orbit.multiplicity}{orbit.letter}, not a maximal position")
print(f"   its content      : {fmt(orbit.content)}")

# The orbit sits on a line whose closure carries two positions with EBRs, and its
# orbitals localize at each of them: one set of orbitals moved two ways, with the
# same invariants.
for loc in orbit.localizations:
    where = loc["terms"][0][2].split("(")[0]
    terms = " + ".join(f"{n} x {name}" for n, name, _at in loc["terms"])
    print(f"   localized at {where:<4}: {terms}   "
          f"theta_Z = ({', '.join(str(v) for v in loc['theta_Z'])})")

t4_strong = tqc.hsp_table(rep_strong, bands=4)
b4_strong = tqc.srsi_of_bands(t4_strong, g_strong, rep=rep_strong)
print(f"   all four bands   : theta_Z = ({', '.join(str(v) for v in b4_strong.theta_Z)})"
      "   from the bands")
print(f"   the model's total: theta_Z = ({', '.join(str(v) for v in m_srsi.theta_Z)})"
      "   from the orbitals")
print("                      (the four bands together are the model's own orbitals,")
print("                      so the band route returns the integers read off them)")

########### 3. the other two phases, and the denominator ###########

print("\n3. delta_t = -0.4, and three bonds cut: decoupled dimers along [111]")
m_weak = fu_kane_mele(T, SOC, [-0.4, 0.0, 0.0, 0.0])         # the same bond weakened

r_weak = tqc.realized_group(m_weak, characters=DECLARE)
_m, g_weak, asg_weak, val_weak, rep_weak = r_weak
sites_weak = tqc.wyckoff_sites_tables(g_weak, asg_weak.site_positions)
t2_weak = tqc.hsp_table(rep_weak, bands=2)
b_weak = tqc.srsi_of_bands(t2_weak, g_weak, rep=rep_weak)
si_weak = tqc.symmetry_indicator(t2_weak, g_weak, rep=rep_weak)
print(f"   weak,   realized : #{g_weak.number_str} {g_weak.name}, orbitals at "
      f"Wyckoff {sites_weak[0].multiplicity}{sites_weak[0].letter}")
print(f"   weak,   valence  : {fmt(t2_weak.counts)}")

m_dimers = fu_kane_mele(T, SOC, [0.0, -1.0, -1.0, -1.0])     # three bonds cut

r_dimers = tqc.realized_group(m_dimers, characters=DECLARE)
_m, g_dimers, asg_dimers, val_dimers, rep_dimers = r_dimers
sites_dimers = tqc.wyckoff_sites_tables(g_dimers, asg_dimers.site_positions)
t2_dimers = tqc.hsp_table(rep_dimers, bands=2)
b_dimers = tqc.srsi_of_bands(t2_dimers, g_dimers, rep=rep_dimers)
si_dimers = tqc.symmetry_indicator(t2_dimers, g_dimers, rep=rep_dimers)
print(f"   dimers, realized : #{g_dimers.number_str} {g_dimers.name}, orbitals at "
      f"Wyckoff {sites_dimers[0].multiplicity}{sites_dimers[0].letter}")
print(f"   dimers, valence  : {fmt(t2_dimers.counts)}")

# one row per phase: the class on the left, the invariants on the right
print("   phase    class    order   theta_Z of the valence pair              denom")
row = (f"   {'strong':<8} {str(si_strong.values):<8} {si_strong.order:<7} "
       f"({', '.join(str(v) for v in b_strong.theta_Z)})")
print(f"{row:<69} {b_strong.denominator}")
row = (f"   {'weak':<8} {str(si_weak.values):<8} {si_weak.order:<7} "
       f"({', '.join(str(v) for v in b_weak.theta_Z)})")
print(f"{row:<69} {b_weak.denominator}")
row = (f"   {'dimers':<8} {str(si_dimers.values):<8} {si_dimers.order:<7} "
       f"({', '.join(str(v) for v in b_dimers.theta_Z)})")
print(f"{row:<69} {b_dimers.denominator}")
print("   measured here: the denominator equals the order of the class in all")
print("   three phases.  The paper's theorem is the equivalence behind it — a")
print("   nonzero indicator class if and only if a fractional Z-valued SRSI.")

########### 4. the no-go criterion on the valence/conduction gap ###########

print("\n4. the no-go criterion on the valence/conduction gap")

# splitting() reads theta[H] off the orbitals (both halves), theta_Z[V] off the
# bands, takes theta_Z[C] as the difference, and asks whether V and C can be two
# atomic insulators at once.
s_strong = tqc.splitting(r_strong, bands=2)
s_weak = tqc.splitting(r_weak, bands=2)
s_dimers = tqc.splitting(r_dimers, bands=2)
print(f"   strong / weak    : '{s_strong.verdict}' and '{s_weak.verdict}' — theta_Z[V] "
      "is fractional, so no")
print("                      atomic insulator has the valence pair's symmetry data")
print("                      and the criterion is not needed")
print(f"   dimers           : '{s_dimers.verdict}' — the SRSI group of "
      f"#{s_dimers.number_str} is")
print(f"                      {m_srsi.structure}, with no Z_n row, so there")
print("                      is no additivity for the criterion to test")

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
