# Example of labelling a degenerate level at a non-TRIM point using pythtqc.
# I4_1 (SG 80) is body-centred, and P = (1/4, 1/4, 1/4) in primitive
# coordinates is not a time-reversal-invariant momentum -- 2P is no reciprocal-
# lattice vector -- so the antiunitary acting at P is the coset element
# {C4|0,1/2,1/4}·T.  Both double-valued irreps of the unitary little group at P
# are one-dimensional and complex conjugates of one another, so an enforced
# twofold degeneracy there can only be the antiunitary's doing, and the
# question is which of the two conjugate irreps it glues to itself.  The
# observable that answers is the trace of the twofold rotation on the level.
# The models are pythtqc.models' i41_atomic(), i41_soc("generic") and
# p62_atomic().
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. the spinful atomic limit: the group, and the fourfold level at P
# 2. the little group at P, and the same table read at the other star member -P
# 3. the spin-orbit model: the fourfold level at P split 1 + 2 + 1
# 4. each of the three levels labelled from one diagonalization
# 5. the four valence bands as an elementary band representation
# 6. the control: P6_2 (#171) at H, where the gluing pairs two different irreps
# 7. the P-point levels against the spin-orbit strength, and the figure

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

# The characters print as χ(...) with a minus sign −, the spin-orbit strength as λ,
# and elementary band representations as ρ↑G, the site irrep ρ induced to the
# group; a Windows console cannot print these by default, so the output is UTF-8.
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

# A pythtb orbital carries no symmetry character, so the s here is a declaration.
DECLARE = {0: "s"}

os.makedirs(OUTDIR, exist_ok=True)

print("I4_1 (SG 80) at P = (1/4, 1/4, 1/4), which is not a time-reversal-invariant")
print("momentum: time reversal glues one of the two conjugate coreps to itself, and")
print("the trace of the twofold says which.")
print("Bands are counted from 0, as pythtb counts them.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


# each character is printed the way a paper tabulates it -- 4, −i, 2i -- from its
# real part r and imaginary part m; the characters themselves stay complex
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
    """The atomic limits of a band set, as a label and a value: the one sum of elementary
    band representations with this content, how many sums there are, or none."""
    if d.verdict == "none":
        return "atomic limits", "none"
    sums = d.solution_strings()
    if len(sums) == 1:
        return "atomic limit", sums[0]
    return "atomic limits", f"{len(sums)} sums of elementary band representations"


########### 1. the spinful atomic limit ###########

print("\n1. the spinful atomic limit")
atomic = tqm.i41_atomic()

out = tqc.realized_group(atomic, characters=DECLARE)
model, group, assignment, validated, rep = out
sites = tqc.wyckoff_sites_tables(group, assignment.site_positions)   # letter and multiplicity
row("realized group", f"#{group.number_str} {group.name}, s orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(f'{s.multiplicity}{s.letter}' for s in sites))}, "
    f"{model.nstate} bands")

# the operation each character below belongs to, by index, with its name in the
# model's own frame
operations = out.operations

hsp = tqc.high_symmetry_points(group)
P = hsp["P"]
row("P", f"({', '.join(str(x) for x in np.round(P, 4).tolist())}) in the primitive cell,",
    "and 2P is no reciprocal-lattice vector")

# solve_at(model, k) diagonalizes H(k) at one k and returns the energies and the
# eigenvectors; every label below is read off that one diagonalization.
states = solve_at(atomic, P)
w = np.asarray(states[0], dtype=float).reshape(-1)
content = tqc.corep_content(rep, P, states=states, bands=[0, 1, 2, 3])
chi = np.asarray(content.table.characters).ravel()
row("P valence levels", ", ".join(str(x) for x in np.round(w[:4], 6).tolist()))
row(f"χ({', '.join(operations[i].name for i in content.table.columns)})", characters(chi))
print("   one fourfold level: spin-independent hoppings commute with every spin")
print("   operator, so this degeneracy is SU(2)'s and decides nothing here")

########### 2. the little group at P, and the same table read at -P ###########

print("\n2. what the little group at P allows, and the same table read at -P")
page = tqc.corep_table(group, P)
ops = sorted(next(iter(page.values())))
print(f"   the little group at P, χ({', '.join(operations[i].name for i in ops)}):")
for label in sorted(page):
    entry = page[label]
    print(f"      {label:<8} dim {int(round(abs(complex(entry[ops[0]])))):d}   "
          f"{characters(entry[i] for i in ops)}")

# The tables' own rows at P, read from irreptables directly.  The tables are
# written in the e^{+ik·t} translation convention, and since
# e^{+ik·t} = e^{-i(-k)·t}, the table printed for P is the table at -P.
from irreptables.irreps import IrrepTable

table = IrrepTable(group.number_str, group.spinor, magnetic=group.magnetic)
shipped = {ir.name: {int(i): complex(c) for i, c in ir.characters.items()}
           for ir in table.irreps if ir.kpname == "P"}
print(f"   the same table read at -P, the point it is printed for, "
      f"χ({', '.join(operations[i].name for i in ops)}):")
for label in sorted(shipped):
    entry = shipped[label]
    print(f"      {label:<8} dim {int(round(abs(complex(entry[ops[0]])))):d}   "
          f"{characters(entry[i] for i in ops)}")
print("   the two tables are entrywise conjugates: BCS's labels attach to -k, an")
print("   operation taking P to -P interchanges conjugate labels, and the trace is the")
print("   observable")

########### 3. the spin-orbit model: the split at P ###########

print(f"\n3. i41_soc('generic', lam={LAM}): one group-averaged spin-orbit bond set")
soc = tqm.i41_soc("generic", lam=LAM)

out_soc = tqc.realized_group(soc, characters=DECLARE)
_m, g_soc, asg_soc, val_soc, rep_soc = out_soc
row("realized group", f"#{g_soc.number_str} {g_soc.name}")

P_soc = tqc.high_symmetry_points(g_soc)["P"]
states = solve_at(soc, P_soc)
w = np.asarray(states[0], dtype=float).reshape(-1)
mult = tqc.multiplets(w[:4])                  # the degenerate levels, by energy
row("P valence levels", ", ".join(str(x) for x in np.round(w[:4], 6).tolist()))
row("degeneracies", f"{' + '.join(str(len(b)) for b in mult)}, level gaps "
    f"{w[1] - w[0]:.6f} and {w[3] - w[2]:.6f}")

########### 4. each level labelled from one diagonalization ###########

print("\n4. the three levels, each labelled from that one diagonalization")
labels, chis = [], []
for multiplet in mult:
    out_b = tqc.corep_content(rep_soc, P_soc, states=states, bands=list(multiplet))
    chi_b = np.asarray(out_b.table.characters).ravel()
    labels.append(next(iter(out_b.counts)))
    chis.append(chi_b)
    row(band_names(list(multiplet)), f"E = {w[multiplet[0]]:+.6f}   "
        f"χ({', '.join(operations[i].name for i in out_b.table.columns)}) = "
        f"{characters(chi_b)}   {labels[-1]}")

label_lo, label_mid, label_hi = labels
pair = complex(chis[1][-1])            # the twofold's trace on the doubled level
lg = tqc.little_group(g_soc, P_soc)
singlet = min(shipped, key=lambda lab: abs(complex(shipped[lab][ops[0]])))
twofold = operations[ops[-1]].name
row("doubled corep", f"{label_mid}, trace {character(pair)} on {twofold}, two copies of "
    f"{character(pair / 2)}")
row("little group", f"{len(lg.unitary)} unitary elements at P, so no two-dimensional irrep")
print(f"   the antiunitary glues the irrep with χ({twofold}) = {character(pair / 2)} to "
      "itself, so the pair is")
print(f"   symmetry-enforced; the -P table read at P makes it two copies of {singlet}, an")
print("   accidental degeneracy")

########### 5. the four valence bands as an elementary band representation ###########

print("\n5. the four valence bands as an elementary band representation")
table_soc = tqc.hsp_table(rep_soc, bands=[0, 1, 2, 3])   # the content at every point
d = tqc.decompose(table_soc)
row("valence content", fmt(table_soc.counts))
row(*atomic_limits(d))
print(f"   {label_mid} belongs to a labelling of the whole set, not a choice made at P")

########### 6. the control: P6_2 at H ###########

print("\n6. the control: P6_2 at H")
p62 = tqm.p62_atomic()

out62 = tqc.realized_group(p62, characters=DECLARE)
_m, g62, asg62, val62, rep62 = out62
sites62 = tqc.wyckoff_sites_tables(g62, asg62.site_positions)
operations62 = out62.operations
row("realized group", f"#{g62.number_str} {g62.name}, s orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(f'{s.multiplicity}{s.letter}' for s in sites62))}, "
    f"{p62.nstate} bands")

H = tqc.high_symmetry_points(g62)["H"]
page62 = tqc.corep_table(g62, H)
ops62 = sorted(next(iter(page62.values())))
print(f"   the little group at H, χ({', '.join(operations62[i].name for i in ops62)}):")
for label in sorted(page62):
    entry = page62[label]
    print(f"      {label:<8} dim {int(round(abs(complex(entry[ops62[0]])))):d}   "
          f"{characters(entry[i] for i in ops62)}")

cH = tqc.corep_content(rep62, H, states=solve_at(p62, H), bands=list(range(6)))
row("lowest 6 bands", f"{fmt(cH.counts)} at H")
doubled62 = max(page62, key=lambda lab: abs(complex(page62[lab][ops62[0]])))
print(f"   {doubled62} glues two different irreps; at P {label_mid} glues one to itself")

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
ax.set_title(f"#{g_soc.number_str} {g_soc.name} at the non-TRIM point $P$: "
             f"{[int(d) for d in content.table.dims]} $\\to$ "
             f"{[len(b) for b in mult]},\n"
             f"and the trace of {operations[ops[-1]].name} says that {label_mid} is the "
             "doubled corep", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
fig.tight_layout(pad=0.4)
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "sg80_P_levels_vs_lambda.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
