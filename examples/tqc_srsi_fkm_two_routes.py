# Example of the stable real-space invariants (SRSIs) of a set of bands, read
# from the bands and from the orbitals, using pythTQC.
# The model is the spin-orbit model of Fu, Kane and Mele on the diamond
# lattice, PRL 98, 106803 (2007), pythtb.models.fu_kane_mele(t, soc, dt): four
# bands, two of them occupied. Its nearest-neighbor bond along [111] is
# strengthened or weakened, or its other three bonds are cut.
# examples/tqc_fkm_diamond.py reads the same model in momentum space.
# The site-symmetry vector holds the multiplicity of each site irrep at each
# Wyckoff position. An SRSI is a linear combination of its entries that no
# smooth deformation preserving the symmetry changes.
# The tables are those of Hwang et al., "Stable real-space invariants and
# topology beyond symmetry indicators", Nat. Commun. 17, 8564 (2026),
# 10.1038/s41467-026-74844-w, shipped with pythTQC. The paper proves that the
# integer-valued SRSIs of a set of bands, printed Z-valued, are fractional
# exactly when its symmetry-indicator class is nonzero.
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. delta_t = +0.4 on the [111] bond, from the bands: the group, the symmetry
#    data vector of the valence bands, their integer-valued SRSIs, the common
#    denominator of these and the symmetry-indicator class
# 2. the same phase from the orbitals: the Wyckoff orbit, its symmetry data
#    vector, its decompositions at the two positions its orbitals slide onto,
#    and all four bands
# 3. delta_t = -0.4 and the decoupled dimers: the denominator against the order
#    of the indicator class in all three phases
# 4. splitting(r, bands=2): the no-go criterion on the gap above the valence
#    bands, whether the splitting is non-symmetry-indicated topological (nsit),
#    in each phase
# 5. the bands along GM-T-F-L-GM, and the figure

# Author: the pythTQC examples

import math
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
from pythtqc import format_coreps as fmt
from pythtqc.utils import save_figure

# The output is written as UTF-8, whatever the console's own encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band path and nothing else.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_fkm_two_routes")
NK_PATH = 61 if QUICK else 241              # band-path density: the figure only

# fu_kane_mele(t, soc, dt): dt offsets the hoppings of the four nearest-neighbor
# bonds, written delta_t below. They are in the order of R, the cell of the
# bond's far end: [0,0,0], [-1,0,0], [0,-1,0], [0,0,-1]. The first is the bond
# along [111].
T = 1.0                    # the nearest-neighbor hopping
SOC = 0.125                # the spin-orbit strength

# A pythtb orbital carries no symmetry character, so the s here is a declaration.
DECLARE = {0: "s"}

TICKS = (r"$\Gamma$", "T", "F", "L", r"$\Gamma$")   # the figure's path

os.makedirs(OUTDIR, exist_ok=True)

print("Fu-Kane-Mele on the diamond lattice, PRL 98, 106803 (2007): the stable real-space")
print("invariants of its bands, read from the bands and from the orbitals.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def values(theta):
    return "(" + ", ".join(str(v) for v in theta) + ")"


def denominator(b):
    """The common denominator of a band set's Z-valued invariants."""
    return math.lcm(*(v.denominator for v in b.theta_Z.values()))


########### 1. delta_t = +0.4 on the [111] bond, from the bands ###########

print("\n1. delta_t = +0.4 on the [111] bond, the strong phase, read from the bands")
m_strong = fu_kane_mele(T, SOC, [0.4, 0.0, 0.0, 0.0])        # one bond strengthened

r_strong = tqc.realized_group(m_strong, characters=DECLARE)
g_strong = r_strong.group
row("realized group", f"#{g_strong.number_str} {g_strong.name}, s orbitals at Wyckoff "
    f"{r_strong.wyckoff_orbits[0].wyckoff}")

t2_strong = tqc.hsp_table(r_strong, bands=2)               # the valence bands
row("valence bands", fmt(t2_strong.symmetry_data))

# srsi_of_bands takes the table hsp_table returns, as decompose and
# symmetry_indicator do. It writes the symmetry data vector as a rational sum of
# elementary band representations, and evaluates the SRSIs on that sum.
b_strong = tqc.srsi_of_bands(t2_strong)
si_strong = tqc.symmetry_indicator(t2_strong)
row("srsi_of_bands", f"Z-valued {values(b_strong.theta_Z.values())}")
row("denominator", denominator(b_strong))
row("indicator class", f"{values(si_strong.values)} in {si_strong.indicator_group}, of order "
    f"{si_strong.order}")
print("   the Z-valued invariants are fractional, so no integer site-symmetry vector")
print("   induces the symmetry data vector of the valence bands, which are not an atomic limit")

########### 2. the same phase, from the orbitals ###########

print("\n2. the same phase, read from the orbitals")

# srsi_of_model reads the SRSIs off the orbitals. For each Wyckoff orbit it
# measures the symmetry data vector of the orbitals and writes it as a sum of
# elementary band representations at each position its orbitals can slide onto
# where the tables list some, its own among them. The SRSIs are evaluated on
# each sum, and agree.
m_srsi = tqc.srsi_of_model(r_strong)
orbit = m_srsi.wyckoff_orbits[0]
row("the orbit", f"orbitals {' and '.join(str(i) for i in orbit.orbitals)} at Wyckoff "
    f"{orbit.wyckoff}, not a maximal position")
row("its orbitals", fmt(orbit.numerical.symmetry_data))

# The orbit's position 6c lies on the threefold axis through 3a and 3b, where
# the tables list elementary band representations. The orbitals are moved onto
# each, and their symmetry data vector is decomposed there.
for loc in orbit.numerical.localizations:
    row(f"localized at {loc.wyckoff}", f"{loc.label},",
        f"Z-valued {values(loc.theta_Z.values())}")

t4_strong = tqc.hsp_table(r_strong, bands=4)
b4_strong = tqc.srsi_of_bands(t4_strong)
row("all 4 bands", f"Z-valued {values(b4_strong.theta_Z.values())}, read from the bands")
row("the whole model", f"Z-valued {values(m_srsi.theta_Z.values())}, read from the orbitals")
print("   all four bands together have the symmetry data vector of the model's orbitals,")
print("   so the two routes give the same integers")

########### 3. the other two phases, and the denominator ###########

print("\n3. delta_t = -0.4, the weak phase, and three bonds cut, the dimers along [111]")
m_weak = fu_kane_mele(T, SOC, [-0.4, 0.0, 0.0, 0.0])         # the same bond weakened

r_weak = tqc.realized_group(m_weak, characters=DECLARE)
g_weak = r_weak.group
t2_weak = tqc.hsp_table(r_weak, bands=2)
b_weak = tqc.srsi_of_bands(t2_weak)
si_weak = tqc.symmetry_indicator(t2_weak)
row("weak,   realized", f"#{g_weak.number_str} {g_weak.name}, orbitals at Wyckoff "
    f"{r_weak.wyckoff_orbits[0].wyckoff}")
row("weak,   valence", fmt(t2_weak.symmetry_data))

m_dimers = fu_kane_mele(T, SOC, [0.0, -1.0, -1.0, -1.0])     # three bonds cut

r_dimers = tqc.realized_group(m_dimers, characters=DECLARE)
g_dimers = r_dimers.group
t2_dimers = tqc.hsp_table(r_dimers, bands=2)
b_dimers = tqc.srsi_of_bands(t2_dimers)
si_dimers = tqc.symmetry_indicator(t2_dimers)
row("dimers, realized", f"#{g_dimers.number_str} {g_dimers.name}, orbitals at Wyckoff "
    f"{r_dimers.wyckoff_orbits[0].wyckoff}")
row("dimers, valence", fmt(t2_dimers.symmetry_data))

# one line per phase: the class on the left, the invariants on the right
print(f"   {'phase':<8} {'class':<8} {'order':<7} "
      f"{'Z-valued invariants of the valence bands':<42} denominator")
for phase, si, b in (("strong", si_strong, b_strong), ("weak", si_weak, b_weak),
                     ("dimers", si_dimers, b_dimers)):
    print(f"   {phase:<8} {values(si.values):<8} {si.order:<7} "
          f"{values(b.theta_Z.values()):<42} {denominator(b)}")
print("   the common denominator of the Z-valued invariants equals the order of the class")
print("   in all three phases. Hwang et al. prove that the Z-valued invariants are")
print("   fractional exactly when the class is nonzero")

########### 4. the no-go criterion on the gap above the valence bands ###########

print("\n4. splitting(r, bands=2): the no-go criterion on the gap above the valence bands")

# splitting reads the SRSIs of all the bands, of both kinds, off the orbitals.
# It reads the integer-valued SRSIs of the valence bands off their symmetry data
# vector, and takes the conduction bands' as the difference. An atomic
# realization of a side is a set of site irreps at maximal Wyckoff positions,
# with nonnegative multiplicities, that has its integer-valued SRSIs. nsit,
# non-symmetry-indicated topological, is true where both sides have atomic
# realizations and no pair of them has SRSIs valued modulo n adding up to those
# of all the bands. The two sides then cannot both be atomic limits, and no
# symmetry indicator shows it.
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

# the strong phase
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
                  f"{si_strong.values} in {si_strong.indicator_group}\n"
                  "valence bands, Z-valued "
                  f"({', '.join(str(v) for v in b_strong.theta_Z.values())}), "
                  f"denominator {denominator(b_strong)}", fontsize=9.5)

# the weak phase
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
                  f"{si_weak.values} in {si_weak.indicator_group}\n"
                  "valence bands, Z-valued "
                  f"({', '.join(str(v) for v in b_weak.theta_Z.values())}), "
                  f"denominator {denominator(b_weak)}", fontsize=9.5)

# the dimers
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
                  f"{si_dimers.values} in {si_dimers.indicator_group}\n"
                  "valence bands, Z-valued "
                  f"({', '.join(str(v) for v in b_dimers.theta_Z.values())}), "
                  f"denominator {denominator(b_dimers)}", fontsize=9.5)

fig.suptitle("Fu-Kane-Mele on diamond in three phases: the four bands "
             "along GM-T-F-L-GM, the valence pair in blue, with its stable "
             "real-space invariants and the order of its indicator class.",
             fontsize=10)
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.92))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_fkm_two_routes.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
