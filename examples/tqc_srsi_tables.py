# Example of the stable-real-space-invariant tables themselves, using pythtqc:
# no model anywhere in this script.
# The tables are those of Hwang et al., "Stable real-space invariants and
# topology beyond symmetry indicators", Nat. Commun. 17, 8564 (2026),
# 10.1038/s41467-026-74844-w, converted from the paper's supplement and shipped
# as pythtqc package data.
# An SRSI is a linear form theta = Theta.p on the site-symmetry vector p -- how
# many orbitals of each site-symmetry irrep sit at each Wyckoff position -- that
# annihilates every adiabatic, symmetry-preserving rearrangement of the
# orbitals.  Some rows are Z-valued and some are Z_n-valued; the Z_n rows carry
# information no symmetry vector does, and they are why these tables say more
# than a symmetry indicator.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. srsi_tables(211, spinor=False): one group's rows, its whole site-key basis,
#    which keys are maximal and which have an EBR, and its band-count form
# 2. two site irreps at one position that the invariants do not distinguish
# 3. two printed indicator rows of SG 147 that are not indicators on the group's
#    own EBR, and how many rows in all the tables behave that way
# 4. the supplement's symmetry-indicated split EBRs, re-asked with the full
#    enumeration of atomic realizations, and the unflagged cases one by one
# 5. the pairs the SRSIs do not distinguish, over every tabulated system
# 6. the figure: the split-EBR verdicts and the same-content tally

# Author: the pythtqc examples

import os
import sys
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import nsit_criterion
from pythtqc.utils import save_figure

# Elementary band representations print as ρ↑G, the site irrep ρ induced to the
# group; a Windows console cannot print ↑ by default, so the output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK changes nothing here: the sweeps below are over all the
# tables, and a thinned sweep would be a different measurement.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_tables")

os.makedirs(OUTDIR, exist_ok=True)

# srsi_tables takes a realized group with time reversal, or a space-group number
# with spinor=; no model is needed.
t211 = tqc.srsi_tables(211, spinor=False)
print(f"The SRSI tables of {t211.provenance['doi']}.")

########### 1. one group's tables, read end to end ###########

print("\n1. srsi_tables(211, spinor=False): one group's tables, end to end")
print(f"   the tables       : #{t211.number_str}, {len(t211.rows)} SRSI rows over "
      f"{len(t211.site_keys)} site keys and {len(t211.ebrs)} EBR columns, "
      f"{len(t211.indicators)} tabulated indicators, "
      f"{len(t211.split_ebrs)} symmetry-indicated split EBRs")
print(f"   the SRSI group   : {t211.structure}")

# The basis is the whole site-symmetry vector: p runs over the site-symmetry
# irreps of every Wyckoff position, maximal or not, because the adiabatic moves
# that define the SRSIs are the moves between them.  Only some keys have an EBR
# column, and the basis holds the others too.
keys = t211.site_keys
print(f"   the basis        : {len(keys)} site keys, "
      f"{len([k for k in keys if k.maximal])} of them at maximal positions, "
      f"{len([k for k in keys if k.ebr])} with an EBR column")
print("   with a column    : "
      + ", ".join(f"({k.irrep})_{k.letter}" for k in keys if k.ebr))
print("   without one      : "
      + ", ".join(f"({k.irrep})_{k.letter}" for k in keys if not k.ebr))
print("   the band count   : nu = "
      + " + ".join(f"{v}*theta_{i + 1}"
                   for i, v in enumerate(t211.band_count_form) if v))
print("                      (a stable invariant itself, and what bounds the")
print("                      enumeration of atomic realizations)")

########### 2. two site irreps the invariants do not distinguish ###########

print("\n2. two site irreps at one position the invariants do not distinguish")

# srsi_of takes p as a mapping over site keys -- the paper's own basis -- and
# returns the Z-valued half and the Z_n-valued half.
b2 = tqc.srsi_of(t211, {("B2", "d"): 1})
b3 = tqc.srsi_of(t211, {("B3", "d"): 1})
print(f"   (B2) at d        : theta_Z = ({', '.join(str(v) for v in b2[0])})   "
      f"theta_Zn = {b2[1]}")
print(f"   (B3) at d        : theta_Z = ({', '.join(str(v) for v in b3[0])})   "
      f"theta_Zn = {b3[1]}")

pairs211 = tqc.stably_equivalent_pairs(t211)
for pair in pairs211:
    print(f"   the pair         : {' / '.join(pair.irreps)} at Wyckoff "
          f"{pair.letter}, the EBRs {' and '.join(pair.names)}")
if pairs211:
    print("   both members induce the same symmetry-data vector and carry identical")
    print("   SRSIs, so neither a momentum-space nor a real-space measurement")
    print("   separates them.")

########### 3. printed indicator rows that are not indicators ###########

print("\n3. SG 147 without SOC: printed indicator rows that are not indicators")
t147 = tqc.srsi_tables(147, spinor=False)
print(f"   the tables       : #{t147.number_str}, tabulated indicators "
      + ", ".join(f"{f.name} (mod {f.mod})" for f in t147.indicators))

# (Au)_a is an EBR column of this very group -- an atomic limit -- so every
# indicator must read zero on it.  Here is what each printed row gives.
au = {("Au", "a"): 1}
au_key = [k for k in t147.site_keys if (k.irrep, k.letter) == ("Au", "a")][0]
raw = t147.theta_unreduced(au)
printed = []
for form in t147.indicators:
    value = sum((c * raw[i - 1] for i, c in form.theta.items()), Fraction(0))
    printed.append(f"{form.name} = {int(value) % form.mod} mod {form.mod}")
print(f"   the test case    : (Au) at {au_key.letter}, EBR column {au_key.column} of "
      f"#{t147.number_str} -- an atomic limit, so")
print("                      every indicator must read 0 on it")
print(f"   theta            : ({', '.join(str(v) for v in t147.theta(au)[0])})")
print(f"   the printed rows : {', '.join(printed)}")
print(f"   what is returned : {t147.indicators_of(au)}, the derived rows")

# ... and how many rows in all the tables behave this way, swept here.
# Thirteen space groups have a single site irrep in the cell, whose one invariant
# is the band count; the tables omit them, and srsi_tables refuses them.
relations = {}
different = []
for sg in range(1, 231):
    for spinor in (False, True):
        table = tqc.srsi_tables(sg, spinor=spinor)
        if isinstance(table, tqc.Refusal):
            continue
        for form in table.indicators:
            # a row whose name has no counterpart to derive it from has no
            # derived row to compare with
            relation = form.relation or "with no derived row"
            relations[relation] = relations.get(relation, 0) + 1
            if form.relation == "different":
                different.append(f"#{table.number_str} {form.name}")
print(f"   over all systems : {sum(relations.values())} tabulated indicator rows -- "
      + ", ".join(f"{n} {name}" for name, n
                  in sorted(relations.items(), key=lambda kv: -kv[1])))
if different:
    print("   the different    : " + ", ".join(different) + " -- and nothing else")
    print("   the rows returned are the derived ones, which vanish on every elementary")
    print("   band representation; the printed rows are kept beside them.")

########### 4. the supplement's split EBRs, re-asked ###########

print("\n4. the symmetry-indicated split EBRs, re-asked with a full enumeration")

# The supplement lists every way an EBR can split into EBRs that the symmetry
# indicators allow, with a mark saying whether the SRSIs of the two sides
# mismatch.  Both halves are measured below: that mark, recomputed from the theta
# rows it is printed beside; then the criterion, which asks the question the mark
# only sampled.
counts = {"single": 0, "double": 0}
flagged = 0
reproduced = 0
tally = {}
unflagged = []
exceptions = []
for sg in range(1, 231):
    for spinor in (False, True):
        table = tqc.srsi_tables(sg, spinor=spinor)
        if isinstance(table, tqc.Refusal) or not table.split_ebrs:
            continue
        kind = "double" if spinor else "single"
        for split in table.split_ebrs:
            counts[kind] += 1
            flagged += int(split.srsi_mismatch)
            parent = [0] * len(table.ebrs)
            parent[int(split.parent["column"])] = 1
            theta_parent = table.theta_unreduced_columns(parent)
            theta_children = [Fraction(0)] * len(theta_parent)
            for child in split.children:
                vector = [0] * len(table.ebrs)
                vector[int(child["column"])] = 1
                for i, value in enumerate(table.theta_unreduced_columns(vector)):
                    theta_children[i] += value
            mismatch = any((int(theta_parent[i]) - int(theta_children[i])) % n
                           for n, rows in table.zn_rows.items() for i in rows)
            if mismatch == bool(split.srsi_mismatch):
                reproduced += 1
            # ... and the criterion, with H = the parent EBR (every row of it
            # known exactly), V = the first child and C = the rest
            first = [0] * len(table.ebrs)
            first[int(split.children[0]["column"])] = 1
            theta_total = table.theta_columns(parent)
            theta_v = table.theta_columns(first)[0]
            theta_c = tuple(a - b for a, b in zip(theta_total[0], theta_v))
            got = nsit_criterion(table, theta_total, theta_v, theta_c)
            key = (bool(split.srsi_mismatch), kind, got.verdict)
            tally[key] = tally.get(key, 0) + 1
            entry = (table.number_str, split.parent["irrep"], split.parent["letter"],
                     tuple(c["irrep"] + "@" + c["letter"] for c in split.children),
                     got.verdict, len(got.consistent_pairs))
            if not split.srsi_mismatch:
                unflagged.append(entry)
            if split.srsi_mismatch and not got.nsit:
                exceptions.append(entry)

total = counts["single"] + counts["double"]
print(f"   the list         : {total} splittings, {counts['single']} without SOC and "
      f"{counts['double']} with, {flagged} of them marked SRSI-mismatching")
print(f"   the supplement's mismatch column, recomputed here from the theta rows "
      f"beside it, agrees on {reproduced} of {total}.")
print("   the criterion, with H the parent EBR, V the first child and C the rest:")
print("      the mark    valuedness  verdict                     count")
for key in sorted(tally, key=lambda k: (not k[0], k[1], k[2])):
    print(f"      {'flagged' if key[0] else 'unflagged':<11} {key[1]:<11} "
          f"{key[2]:<27} {tally[key]}")
diagnosed = sum(v for k, v in tally.items()
                if k[0] and k[2] == "diagnosed topological")
print(f"   the headline     : {diagnosed} of the {flagged} flagged splittings stay "
      "diagnosed topological")
for number, irrep, wyckoff, children, verdict, npairs in exceptions:
    print(f"   the exception    : #{number}  ({irrep})_{wyckoff} -> "
          f"{' + '.join(children)}   '{verdict}', {npairs} consistent pairs")
if exceptions:
    print("   a consistent pair is two atomic realizations, one per side, whose Z_n-SRSIs")
    print("   add up to the total's; where one exists, the splitting's topology is")
    print("   invisible to the invariants.")
if unflagged:
    print(f"   the {len(unflagged)} unflagged splittings, one by one:")
    for number, irrep, wyckoff, children, verdict, npairs in unflagged:
        parent_label = f"({irrep})_{wyckoff}"
        print(f"      #{number:<8} {parent_label:<12} -> "
              f"{' + '.join(children):<22} '{verdict}' ({npairs} consistent "
              f"pair{'' if npairs == 1 else 's'})")
n_no_zn = sum(1 for entry in unflagged if entry[4] == "no Z_n SRSIs")
if n_no_zn:
    print(f"   the {n_no_zn} 'no Z_n SRSIs' entries are groups whose SRSI group has no "
          "Z_n row, so there is no additivity to test.")

########### 5. the pairs the SRSIs do not distinguish, everywhere ###########

print("\n5. the pairs the SRSIs do not distinguish, over every tabulated system")
found = []
for sg in range(1, 231):
    for spinor in (False, True):
        table = tqc.srsi_tables(sg, spinor=spinor)
        if isinstance(table, tqc.Refusal):
            continue
        for pair in tqc.stably_equivalent_pairs(table):
            # the site-symmetry group of the position, as the tables themselves
            # carry it: the set of site irreps keyed at that Wyckoff letter
            signature = tuple(sorted(k.irrep for k in table.site_keys
                                     if k.letter == pair.letter))
            found.append((sg, spinor, pair.letter, pair.irreps, signature))
groups = sorted({sg for sg, *_ in found})
print(f"   the sweep        : {len(found)} pairs over {len(groups)} space groups, "
      f"{sum(1 for p in found if p[1])} of them with SOC")

by_signature = {}
for _sg, _sp, _w, _irreps, signature in found:
    by_signature[signature] = by_signature.get(signature, 0) + 1
print("   by the site irreps the tables carry at that letter: " + ";  ".join(
    f"{by_signature[s]} at ({', '.join(s)})"
    for s in sorted(by_signature, key=lambda s: -by_signature[s])))

by_members = {}
for _sg, _sp, _w, irreps, _s in found:
    by_members[irreps] = by_members.get(irreps, 0) + 1
print("   by the two members: " + ", ".join(
    "/".join(k) + f" ({v})" for k, v
    in sorted(by_members.items(), key=lambda kv: -kv[1])))
print(f"   the space groups : {groups}")
if found:
    print("   the tables give both members of each pair identical SRSIs, Z and Z_n,")
    print("   so whichever member a decomposition picks, no number moves.")

########### 6. the figure ###########

print("\n6. the figure: the split-EBR verdicts and the same-content tally")
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

labels = []
values = []
colours = []
for key in sorted(tally, key=lambda k: (not k[0], k[1], k[2])):
    labels.append(f"{'flagged' if key[0] else 'unflagged'}, {key[1]}\n{key[2]}")
    values.append(tally[key])
    colours.append("tab:red" if key[2] == "diagnosed topological"
                   else ("tab:blue" if key[2] == "invisible to the SRSIs" else "0.6"))
axes[0].barh(range(len(values)), values, color=colours)
axes[0].set_yticks(range(len(values)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xscale("log")
for i, v in enumerate(values):
    axes[0].annotate(str(v), xy=(v, i), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9)
axes[0].set_xlim(0.7, max(values) * 2.4)
axes[0].set_xlabel("splittings (log scale)")
axes[0].set_title(f"the supplement's {total} split EBRs, re-asked over every atomic\n"
                  f"realization: {diagnosed} of the {flagged} flagged stay "
                  f"diagnosed topological", fontsize=10)

sig_order = sorted(by_signature, key=lambda s: -by_signature[s])
sig_values = [by_signature[s] for s in sig_order]
axes[1].barh(range(len(sig_values)), sig_values, color="tab:purple")
axes[1].set_yticks(range(len(sig_values)))
axes[1].set_yticklabels([", ".join(s) for s in sig_order], fontsize=8)
axes[1].invert_yaxis()
for i, v in enumerate(sig_values):
    axes[1].annotate(str(v), xy=(v, i), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9)
axes[1].set_xlim(0, max(sig_values) * 1.12)
axes[1].set_xlabel("pairs")
axes[1].set_title(f"the {len(found)} pairs of site irreps the SRSIs do not\n"
                  "distinguish, by the site irreps at that letter", fontsize=10)

fig.suptitle("The shipped SRSI tables, with no model anywhere: the supplement's own "
             "split-EBR flags re-asked with a complete enumeration, and the "
             "same-content sweep.", fontsize=10)
fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.93))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_tables.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
