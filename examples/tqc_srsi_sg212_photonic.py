# Example of the stable real-space invariants of photonic bands, using pythTQC.
# The model is the space group 212 (P4_332) tight-binding model of Hwang et al.,
# arXiv:2601.06293, H_SG212,1 of its supplementary note: two orbitals at each 4a
# site, carrying the irrep E of the site group 32, with nearest-neighbour
# hoppings.  Its energies stand for ω², and its 8 bands for a photonic crystal's
# 4 lowest bands together with 4 auxiliary bands, which the paper adds so that
# the two transverse modes at Γ and ω = 0 have a symmetry data vector.
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. the symmetry group of the Hamiltonian
# 2. the coreps of all 8 bands at the high-symmetry points, and the band structure
# 3. the symmetry data vector of all 8 bands, that of the elementary band
#    representation (EBR) of E at 4a
# 4. the auxiliary bands, the EBR of A1 at 4a or at 4b
# 5. the stable real-space invariants of all 8 bands, of the auxiliary bands and
#    of the physical bands

# Author: the pythTQC examples

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pythtb import Lattice, TBModel

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc.utils import save_figure

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the band path and nothing else.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_sg212_photonic")
NK_PATH = 101 if QUICK else 401             # band-path density: the figure only

os.makedirs(OUTDIR, exist_ok=True)


def sg212_model(e, t1, t2, t3):
    """H_SG212,1 of the supplementary note: two orbitals at each 4a site of P4_332, nearest-neighbour hoppings.

    Orbitals (0, 1), (2, 3), (4, 5) and (6, 7) sit at the sites A, B, C and D of
    4a, in the note's order, and model.hamiltonian(k) is the note's H_SG212,1 at
    k = 2π times the reduced wavevector.
    """
    lat_vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    orb_vecs = [
        [1 / 8, 1 / 8, 1 / 8], [1 / 8, 1 / 8, 1 / 8],  # A
        [3 / 8, 7 / 8, 5 / 8], [3 / 8, 7 / 8, 5 / 8],  # B
        [7 / 8, 5 / 8, 3 / 8], [7 / 8, 5 / 8, 3 / 8],  # C
        [5 / 8, 3 / 8, 7 / 8], [5 / 8, 3 / 8, 7 / 8],  # D
    ]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1, 2])
    model = TBModel(lattice=lat, spinful=False)

    model.set_onsite([e] * 8)

    w = np.exp(2j * np.pi / 3)
    wb = w.conjugate()

    # A -> B, R = [0, -1, 0], bond vector = (1/4, -1/4, 1/2)
    model.set_hop(2*t1 + 2*t2*w, 0, 2, [0, -1, 0])
    model.set_hop(-t3, 0, 3, [0, -1, 0])
    model.set_hop(-t3, 1, 2, [0, -1, 0])
    model.set_hop(2*t1 + 2*t2*wb, 1, 3, [0, -1, 0])

    # A -> B, R = [0, -1, -1], bond vector = (1/4, -1/4, -1/2)
    model.set_hop(2*t1 + 2*t2*wb, 0, 2, [0, -1, -1])
    model.set_hop(-t3, 0, 3, [0, -1, -1])
    model.set_hop(-t3, 1, 2, [0, -1, -1])
    model.set_hop(2*t1 + 2*t2*w, 1, 3, [0, -1, -1])

    # A -> C, R = [-1, 0, 0], bond vector = (-1/4, 1/2, 1/4)
    model.set_hop(2*t1 + 2*t2*w, 0, 4, [-1, 0, 0])
    model.set_hop(-t3*w, 0, 5, [-1, 0, 0])
    model.set_hop(-t3*wb, 1, 4, [-1, 0, 0])
    model.set_hop(2*t1 + 2*t2*wb, 1, 5, [-1, 0, 0])

    # A -> C, R = [-1, -1, 0], bond vector = (-1/4, -1/2, 1/4)
    model.set_hop(2*t1 + 2*t2*wb, 0, 4, [-1, -1, 0])
    model.set_hop(-t3*w, 0, 5, [-1, -1, 0])
    model.set_hop(-t3*wb, 1, 4, [-1, -1, 0])
    model.set_hop(2*t1 + 2*t2*w, 1, 5, [-1, -1, 0])

    # A -> D, R = [0, 0, -1], bond vector = (1/2, 1/4, -1/4)
    model.set_hop(-t3, 0, 6, [0, 0, -1])
    model.set_hop(2*t1*wb + 2*t2, 0, 7, [0, 0, -1])
    model.set_hop(2*t1*w + 2*t2, 1, 6, [0, 0, -1])
    model.set_hop(-t3, 1, 7, [0, 0, -1])

    # A -> D, R = [-1, 0, -1], bond vector = (-1/2, 1/4, -1/4)
    model.set_hop(-t3, 0, 6, [-1, 0, -1])
    model.set_hop(2*t1*wb + 2*t2*w, 0, 7, [-1, 0, -1])
    model.set_hop(2*t1*w + 2*t2*wb, 1, 6, [-1, 0, -1])
    model.set_hop(-t3, 1, 7, [-1, 0, -1])

    # B -> C, R = [0, 0, 0], bond vector = (1/2, -1/4, -1/4)
    model.set_hop(2*t1 + 2*t2*wb, 2, 4, [0, 0, 0])
    model.set_hop(-t3*wb, 2, 5, [0, 0, 0])
    model.set_hop(-t3*w, 3, 4, [0, 0, 0])
    model.set_hop(2*t1 + 2*t2*w, 3, 5, [0, 0, 0])

    # B -> C, R = [-1, 0, 0], bond vector = (-1/2, -1/4, -1/4)
    model.set_hop(2*t1 + 2*t2*w, 2, 4, [-1, 0, 0])
    model.set_hop(-t3*wb, 2, 5, [-1, 0, 0])
    model.set_hop(-t3*w, 3, 4, [-1, 0, 0])
    model.set_hop(2*t1 + 2*t2*wb, 3, 5, [-1, 0, 0])

    # B -> D, R = [0, 0, 0], bond vector = (1/4, -1/2, 1/4)
    model.set_hop(-t3*wb, 2, 6, [0, 0, 0])
    model.set_hop(2*t1*wb + 2*t2, 2, 7, [0, 0, 0])
    model.set_hop(2*t1*w + 2*t2, 3, 6, [0, 0, 0])
    model.set_hop(-t3*w, 3, 7, [0, 0, 0])

    # B -> D, R = [0, 1, 0], bond vector = (1/4, 1/2, 1/4)
    model.set_hop(-t3*wb, 2, 6, [0, 1, 0])
    model.set_hop(2*t1*wb + 2*t2*w, 2, 7, [0, 1, 0])
    model.set_hop(2*t1*w + 2*t2*wb, 3, 6, [0, 1, 0])
    model.set_hop(-t3*w, 3, 7, [0, 1, 0])

    # C -> D, R = [0, 0, 0], bond vector = (-1/4, -1/4, 1/2)
    model.set_hop(-t3*w, 4, 6, [0, 0, 0])
    model.set_hop(2*t1*wb + 2*t2*w, 4, 7, [0, 0, 0])
    model.set_hop(2*t1*w + 2*t2*wb, 5, 6, [0, 0, 0])
    model.set_hop(-t3*wb, 5, 7, [0, 0, 0])

    # C -> D, R = [0, 0, -1], bond vector = (-1/4, -1/4, -1/2)
    model.set_hop(-t3*w, 4, 6, [0, 0, -1])
    model.set_hop(2*t1*wb + 2*t2, 4, 7, [0, 0, -1])
    model.set_hop(2*t1*w + 2*t2, 5, 6, [0, 0, -1])
    model.set_hop(-t3*wb, 5, 7, [0, 0, -1])

    return model


print("Hwang et al., arXiv:2601.06293: the space group 212 model of a photonic crystal's bands.")
print("Bands are counted from 0, as pythtb counts them.")

########### 1. the symmetry group of the Hamiltonian ###########

# The supplementary note's parameters: GM4 at zero energy, GM3 above it, GM5 below.
m212 = sg212_model(-1.2, 0.8, 0.6, 0.8)

# realized_group measures the symmetry group of the Hamiltonian and the site irrep
# of every orbital.  The hoppings fix the content of each site, E, and leave two
# labellings open, which exchange GM4 with GM5, X1 with X2 and M1M4 with M2M3.
# characters= declares the supplementary note's: the twofold rotation about
# [1-10] through the first site acts on orbitals 0 and 1 as [[0, 1], [1, 0]].
print("\n1. the symmetry group of the Hamiltonian")
r = tqc.realized_group(m212, characters={(0, 1): {"C2([1-10])": [[0, 1], [1, 0]]}})
print(r.summary())

########### 2. the coreps at the high-symmetry points, and the band structure ###########

# hsp_table reads the coreps of the 8 bands at every high-symmetry point; each
# point's record lists them by degenerate multiplet, lowest first.
print("\n2. the coreps of all 8 bands at the high-symmetry points")
table = tqc.hsp_table(r.rep, bands=8)
for record in table.records.values():
    print(record.summary())

# The bands along GM-X-M-GM-R-X-M-R, each multiplet at a high-symmetry point
# labelled with its corep: the 4 lowest bands, below zero away from Γ, are the
# auxiliary bands, and the 4 above them the physical bands.  A label sits above
# its multiplet, or below it where the next multiplet up is too close for two
# labels, as X1 and X2 are at X.
path = ["GM", "X", "M", "GM", "R", "X", "M", "R"]
hsp = tqc.high_symmetry_points(r.group)
k_vec, k_dist, k_node = m212.k_path([hsp[name] for name in path], NK_PATH, report=False)
evals = m212.solve_ham(k_vec)

fig, ax = plt.subplots()
ax.plot(k_dist, evals[:, :4], color="gray")
ax.plot(k_dist, evals[:, 4:], color="red")
ax.axhline(0, color="black", lw=0.5, ls="--")
for name, x in zip(path, k_node):
    energies = table.records[name].table.energies
    for i, label in enumerate(table.records[name].labels):
        below = i + 1 < len(energies) and energies[i + 1] - energies[i] < 0.5
        ax.annotate(tqc.format_counts(label), xy=(x, energies[i]), ha="center",
                    va="top" if below else "bottom", xytext=(0, -4 if below else 4),
                    textcoords="offset points", bbox=dict(fc="white", ec="none", pad=0.5))
ax.set_xticks(k_node, labels=["Γ", "X", "M", "Γ", "R", "X", "M", "R"])
ax.margins(x=0.05, y=0.08)
ax.set_ylabel("ε(k), standing for ω²")
ax.set_title("the auxiliary bands in gray, the physical bands in red")
save_figure(fig, os.path.join(OUTDIR, "sg212_bands.pdf"))
plt.close(fig)

########### 3. all 8 bands as an elementary band representation ###########

# ebr_rows gives the symmetry data vector of every EBR of the group; all 8 bands
# carry that of the EBR induced from E at 4a.
print("\n3. the symmetry data vector of all 8 bands, and of the elementary band "
      "representation of E at 4a")
ebrs = tqc.ebr_rows(r.group)
print("all 8 bands:       ", tqc.format_counts(table.counts))
print("E↑G(8)@4a(321',32):", tqc.format_counts(ebrs["E↑G(8)@4a(321',32)"]))

########### 4. the auxiliary bands ###########

# The auxiliary bands carry the symmetry data vector of an EBR induced from A1, an
# s orbital, at either the 4a or 4b position.  In the band structure they are the
# 4 lowest bands, with GM5 at Γ, X1 and X2 at X, M5 and M1M4 at M, and R3 at R.
# Their GM1 corresponds to the longitudinal subspace of the GM4 triplet at zero.
# The physical bands, all 8 less the auxiliary 4, carry GM4 - GM1 at Γ.
print("\n4. the auxiliary bands: the elementary band representation induced from A1 at either 4a or 4b")
print("A1↑G(4)@4a(321',32):", tqc.format_counts(ebrs["A1↑G(4)@4a(321',32)"]))
print("A1↑G(4)@4b(321',32):", tqc.format_counts(ebrs["A1↑G(4)@4b(321',32)"]))

########### 5. the stable real-space invariants ###########

# srsi_of_model reads the stable real-space invariants (SRSIs) of all 8 bands from
# the model's orbitals, 1 (E)_a.
print("\n5. the stable real-space invariants")
print(tqc.srsi_of_model(r).summary())

# srsi_tables holds the group's SRSIs as formulas in the site irreps, and theta()
# evaluates them on a band representation multiplicity vector m, the multiplicity
# of each site irrep: it returns the Z-valued SRSIs, as exact fractions, and the
# Z2-valued one as an integer mod 2.
srsi = tqc.srsi_tables(r.group)
print("1 (E)_a: ", srsi.theta({("E", "a"): 1}))
print("1 (A1)_a:", srsi.theta({("A1", "a"): 1}))
print("1 (A1)_b:", srsi.theta({("A1", "b"): 1}))

# The SRSIs are linear in m, so the SRSIs of the physical bands are a difference
# of the SRSIs of the model minus the SRSIs of the auxiliary bands.  Both choices
# of auxiliary bands give the Z-valued SRSIs (-1, 0, 1).  The Z2-valued SRSI
# differs between the two choices.
print("1 (E)_a - 1 (A1)_a:", srsi.theta({("E", "a"): 1, ("A1", "a"): -1}))
print("1 (E)_a - 1 (A1)_b:", srsi.theta({("E", "a"): 1, ("A1", "b"): -1}))

print("\nDone — figure written to", OUTDIR)
