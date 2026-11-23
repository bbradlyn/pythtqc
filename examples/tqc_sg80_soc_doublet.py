# Example of labeling a degenerate level at a point that is not a
# time-reversal-invariant momentum, using pythTQC.
# I4_1 (space group 80) is body-centered, and P = (1/4, 1/4, 1/4) in primitive
# reduced coordinates is not a time-reversal-invariant momentum, since 2P is
# not a reciprocal-lattice vector.  The antiunitary operations that fix P are
# the fourfold screws combined with time reversal, C4(z)' and C4^-1(z)'.  The
# two double-valued irreps of the unitary little group at P are
# one-dimensional and complex conjugates of one another, so a twofold
# degeneracy the symmetry enforces there is one of them doubled by an
# antiunitary operation.  The trace of the twofold rotation on the level says
# which one.
# The models are pythtqc.models' i41_atomic(), i41_soc("generic") and
# p62_atomic().
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. the spinful atomic limit: the group, and the fourfold level at P
# 2. the coreps at P, and the rows the irreptables package lists for P, which
#    are the coreps at -P
# 3. the spin-orbit model: the fourfold level at P split 1 + 2 + 1
# 4. each of the three levels labeled from one diagonalization
# 5. the symmetry data vector of the four valence bands, that of one elementary
#    band representation
# 6. the control: P6_2 (space group 171) at H, where an antiunitary operation
#    pairs two different irreps
# 7. the P-point levels against the spin-orbit strength, and the figure

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

# The characters print as χ(...) with a minus sign −, and the spin-orbit strength
# as λ; a Windows console cannot print these by default, so the output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK thins the figure's λ sweep and nothing else.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
QUICK = os.environ.get("TQC_EXAMPLE_QUICK", "0") not in ("0", "", "false")
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_sg80_soc_doublet")
LAM = 0.4                  # the spin-orbit strength every printed number is read at
NLAM = 5 if QUICK else 41  # λ sweep points: the figure's resolution only

# A pythtb orbital does not say whether it is s, p or d, so the s here is declared.
DECLARE = {0: "s"}

os.makedirs(OUTDIR, exist_ok=True)

print("I4_1 (space group 80) at P = (1/4, 1/4, 1/4), not a time-reversal-invariant")
print("momentum. A fourfold screw combined with time reversal doubles one of the two")
print("conjugate irreps at P, and the trace of the twofold rotation says which.")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One line of the ledger: label, a heading of at most 16 characters, then first,
    then each line of rest beneath it."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


# Each character is printed as a paper tabulates it, 4, −i or 2i, from its real
# part r and imaginary part m; the characters themselves stay complex
def character(z):
    """One character, 4, −i or 2i, its minus sign written −."""
    r, m = round(z.real, 6) + 0.0, round(z.imag, 6) + 0.0
    if not m:
        text = f"{r:g}"
    else:
        text = ((f"{r:g}{'+' if m > 0 else ''}" if r else "")
                + ("-" if m == -1 else "" if m == 1 else f"{m:g}") + "i")
    return text.replace("-", "−")


def characters(chi):
    """The characters on a list of operations, (1, −i)."""
    return "(" + ", ".join(character(complex(z)) for z in chi) + ")"


def band_names(indices):
    """Bands named by index: band 0, bands 1 and 2."""
    if len(indices) == 1:
        return f"band {indices[0]}"
    return "bands " + ", ".join(str(i) for i in indices[:-1]) + f" and {indices[-1]}"


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


########### 1. the spinful atomic limit ###########

print("\n1. the spinful atomic limit")
atomic = tqm.i41_atomic()

r = tqc.realized_group(atomic, characters=DECLARE)
group = r.group
row("realized group", f"#{group.number_str} {group.name}, s orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(o.wyckoff for o in r.wyckoff_orbits))}, "
    f"{atomic.nstate} bands")

# the operation each character below belongs to, by index, with its name in the
# Cartesian axes in which the pythtb Lattice is specified
operations = r.operations

hsp = tqc.high_symmetry_points(group)
P = hsp["P"]
row("P", f"({', '.join(str(x) for x in np.round(P, 4).tolist())}) in primitive reduced "
    "coordinates,", "and 2P is not a reciprocal-lattice vector")

# solve_at(model, k) diagonalizes H(k) at one k and returns the energies and the
# eigenvectors; every label below is read off that one diagonalization.
states = solve_at(atomic, P)
w = np.asarray(states[0], dtype=float).reshape(-1)
table_P = tqc.characters(r, P, states=states, bands=[0, 1, 2, 3])
columns = tuple(table_P.multiplets[0].characters)    # the unitary operations at P
chi = np.array([[m.characters[i] for i in columns] for m in table_P.multiplets]).ravel()
row("P valence levels", ", ".join(str(x) for x in np.round(w[:4], 6).tolist()))
row(f"χ({', '.join(operations[i].name for i in columns)})", characters(chi))
print("   one fourfold level. The hoppings do not depend on spin, so all of SU(2)")
print("   commutes with H. The level has both irreps and does not say which is doubled")

########### 2. the coreps at P, and the rows listed for P ###########

print("\n2. the coreps at P, and the rows the irreptables package lists for P")
page = tqc.corep_table(group, P)
ops = sorted(next(iter(page.values())))
print(f"   the coreps at P, χ({', '.join(operations[i].name for i in ops)}):")
for label in sorted(page):
    entry = page[label]
    print(f"      {label:<8} dim {int(round(abs(complex(entry[ops[0]])))):d}   "
          f"{characters(entry[i] for i in ops)}")

# The rows the irreptables package lists for P, read from it directly.  Its
# tables represent a lattice translation t by e^{+ik·t}, and the matrices
# returned here by e^{-ik·t}.  Since e^{+ik·t} = e^{-i(-k)·t}, the rows listed
# for P are the coreps at -P.
from irreptables.irreps import IrrepTable

table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic_type == "II")
shipped = {ir.name: {int(i): complex(c) for i, c in ir.characters.items()}
           for ir in table.irreps if ir.kpname == "P"}
print(f"   the rows listed for P, which are the coreps at -P, "
      f"χ({', '.join(operations[i].name for i in ops)}):")
for label in sorted(shipped):
    entry = shipped[label]
    print(f"      {label:<8} dim {int(round(abs(complex(entry[ops[0]])))):d}   "
          f"{characters(entry[i] for i in ops)}")
print("   each row of one table is the complex conjugate of a row of the other. A label")
print("   depends on whether the table is read at P or at -P. The trace of the twofold")
print("   rotation on a level does not")

########### 3. the spin-orbit model: the split at P ###########

print(f"\n3. i41_soc('generic', lam={LAM}): spin-orbit hoppings averaged over "
      "the group")
soc = tqm.i41_soc("generic", lam=LAM)

r_soc = tqc.realized_group(soc, characters=DECLARE)
g_soc = r_soc.group
row("realized group", f"#{g_soc.number_str} {g_soc.name}")

P_soc = tqc.high_symmetry_points(g_soc)["P"]
states = solve_at(soc, P_soc)
w = np.asarray(states[0], dtype=float).reshape(-1)
mult = tqc.multiplets(w[:4])                  # the degenerate levels, by energy
row("P valence levels", ", ".join(str(x) for x in np.round(w[:4], 6).tolist()))
row("degeneracies", f"{' + '.join(str(len(b)) for b in mult)}, level gaps "
    f"{w[1] - w[0]:.6f} and {w[3] - w[2]:.6f}")

########### 4. each level labeled from one diagonalization ###########

print("\n4. the three levels, each labeled from that one diagonalization")
labels, chis = [], []
for multiplet in mult:
    out_b = tqc.coreps_at(r_soc, P_soc, states=states, bands=list(multiplet))
    level = tqc.characters(r_soc, P_soc, states=states, bands=list(multiplet)).multiplets[0]
    columns_b = tuple(level.characters)
    chi_b = np.array([level.characters[i] for i in columns_b])
    labels.append(next(iter(out_b.coreps)))
    chis.append(chi_b)
    row(band_names(list(multiplet)), f"E = {w[multiplet[0]]:+.6f}   "
        f"χ({', '.join(operations[i].name for i in columns_b)}) = "
        f"{characters(chi_b)}   {labels[-1]}")

label_lo, label_mid, label_hi = labels
pair = complex(chis[1][-1])            # the twofold's trace on the doubled level
lg = tqc.little_group(g_soc, P_soc)
singlet = min(shipped, key=lambda lab: abs(complex(shipped[lab][ops[0]])))
twofold = operations[ops[-1]].name
row("doubled corep", f"{label_mid}, trace {character(pair)} on {twofold}, two copies of "
    f"the irrep with trace {character(pair / 2)}")
row("little group", f"unitary part of {len(lg.unitary)} operations, with one-dimensional irreps")
print(f"   an antiunitary operation doubles the irrep with χ({twofold}) = "
      f"{character(pair / 2)}, so the pair is")
print(f"   symmetry-enforced. It is not two copies of {singlet}, an accidental degeneracy, as")
print("   the rows listed for P would make it if they were read as the coreps at P")

########### 5. the symmetry data vector of the four valence bands ###########

print("\n5. the symmetry data vector of the four valence bands")
table_soc = tqc.hsp_table(r_soc, bands=[0, 1, 2, 3])   # the coreps at every point
d = tqc.decompose(table_soc)
row("valence bands", fmt(table_soc.symmetry_data))
row(*atomic_limits(d))
print(f"   with {label_mid} at P, the symmetry data vector is that of the atomic limit above")

########### 6. the control: P6_2 at H ###########

print("\n6. the control: P6_2 at H")
p62 = tqm.p62_atomic()

r62 = tqc.realized_group(p62, characters=DECLARE)
g62 = r62.group
operations62 = r62.operations
row("realized group", f"#{g62.number_str} {g62.name}, s orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(o.wyckoff for o in r62.wyckoff_orbits))}, "
    f"{p62.nstate} bands")

H = tqc.high_symmetry_points(g62)["H"]
page62 = tqc.corep_table(g62, H)
ops62 = sorted(next(iter(page62.values())))
print(f"   the coreps at H, χ({', '.join(operations62[i].name for i in ops62)}):")
for label in sorted(page62):
    entry = page62[label]
    print(f"      {label:<8} dim {int(round(abs(complex(entry[ops62[0]])))):d}   "
          f"{characters(entry[i] for i in ops62)}")

cH = tqc.coreps_at(r62, H, states=solve_at(p62, H), bands=list(range(6)))
row("lowest 6 bands", f"{fmt(cH.coreps)} at H")
doubled62 = max(page62, key=lambda lab: abs(complex(page62[lab][ops62[0]])))
print(f"   {doubled62} pairs two different irreps. At P, {label_mid} is one irrep doubled")

########### 7. the levels at P against the spin-orbit strength ###########

print("\n7. the same P-point levels against the spin-orbit strength λ")

# one model, one solve_ham at P and its four valence eigenvalues, per λ
lams = np.linspace(0.0, LAM, NLAM)
levels = np.empty((NLAM, 4))
for i, lam in enumerate(lams):
    m = tqm.i41_soc("generic", lam=float(lam))
    levels[i] = np.asarray(m.solve_ham(np.atleast_2d(P_soc)),
                           dtype=float).reshape(-1)[:4]

fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
ax.plot(lams, levels[:, 0], color="tab:blue", lw=1.6, marker="o", ms=3.5,
        label=f"singlet  $\\to$  {label_lo}")
ax.plot(lams, levels[:, 1], color="tab:red", lw=3.2, marker="s", ms=5.0,
        label=f"enforced doublet  $\\to$  {label_mid}")
ax.plot(lams, levels[:, 2], color="tab:red", lw=3.2, ls="--")
ax.plot(lams, levels[:, 3], color="tab:blue", lw=1.6, marker="o", ms=3.5)
ax.axhline(levels[0, 0], color="0.7", lw=0.8, ls=":")
ax.annotate(r"$\lambda = 0$: one level" "\n" r"(an SU(2) degeneracy)",
            xy=(0.0, levels[0, 0]), xytext=(0.035, levels[:, 0].min() + 0.02),
            fontsize=9, color="0.25",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.45"))
# the corep each level carries, in a clean right margin
for label, row in ((label_hi, 3), (label_mid, 1), (label_lo, 0)):
    ax.annotate(label, xy=(LAM, levels[-1, row]),
                xytext=(LAM + 0.012, levels[-1, row]), fontsize=11, ha="left",
                va="center", color="tab:red" if row == 1 else "tab:blue")
ax.set_xlim(-0.012, LAM * 1.19)
ax.set_ylim(levels.min() - 0.05 * (levels.max() - levels.min()),
            levels.max() + 0.16 * (levels.max() - levels.min()))
ax.set_xlabel(r"spin-orbit strength $\lambda$", fontsize=15)
ax.set_ylabel(r"$E$ at $P = (\frac{1}{4},\frac{1}{4},\frac{1}{4})$", fontsize=15)
ax.set_title(f"#{g_soc.number_str} {g_soc.name} at $P$, not time-reversal invariant: "
             f"{[len(m.bands) for m in table_P.multiplets]} $\\to$ "
             f"{[len(b) for b in mult]},\n"
             f"and the trace of {operations[ops[-1]].name} says that {label_mid} is the "
             "doubled corep", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
fig.tight_layout(pad=0.4)
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "sg80_P_levels_vs_lambda.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
