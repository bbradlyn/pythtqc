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
# 1. srsi_tables(211, spinor=False): one group's invariants, its whole basis of
#    site irreps, which are at maximal positions and which induce an elementary
#    band representation, and its band-count form
# 2. two site irreps at one position that the invariants do not distinguish
# 3. #147.14, single-valued: two printed indicator forms that are not
#    indicators on the group's own elementary band representation, and how
#    many forms in all the tables behave that way
# 4. the splittings of an elementary band representation that the symmetry
#    indicators allow, re-asked with the full enumeration of atomic
#    realizations, and the unflagged ones one by one
# 5. the pairs the SRSIs do not distinguish, over every tabulated system
# 6. the figure: the verdicts on the splittings, and the pairs by site symmetry

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
print("The tables of stable real-space invariants (SRSIs) of")
print(f"{t211.provenance['doi']}, read with no model.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def written(theta):
    """Invariants written out, ``(1, 0, -1)``."""
    return "(" + ", ".join(str(v) for v in theta) + ")"


def moduli(theta_zn):
    """The Z_n-valued invariants by modulus, ``mod 2 (0, 1)``."""
    return ", ".join(f"mod {n} {written(theta)}" for n, theta in sorted(theta_zn.items()))


def wrapped(items, sep=", ", width=64):
    """Items joined by sep, broken between items into lines of at most width characters.

    A list keeps its comma at the end of a line; a sum carries its plus sign to
    the start of the next.
    """
    lines = []
    for item in items:
        if lines and len(lines[-1]) + len(sep) + len(item) <= width:
            lines[-1] += sep + item
        elif lines and sep == " + ":
            lines.append("+ " + item)
        elif lines:
            lines[-1] += sep.rstrip()
            lines.append(item)
        else:
            lines.append(item)
    return lines


########### 1. one group's tables, read end to end ###########

print("\n1. srsi_tables(211, spinor=False): one group's tables, end to end")
row("the tables", f"#{t211.number_str}, {len(t211.rows)} SRSIs on {len(t211.site_keys)} site "
    f"irreps and {len(t211.ebrs)} elementary band", f"representations, {len(t211.indicators)} "
    f"symmetry indicators, and {len(t211.split_ebrs)} splittings of an",
    "elementary band representation that the indicators allow")
row("the SRSI group", t211.structure)

# The basis is the whole site-symmetry vector: p runs over the site-symmetry
# irreps of every Wyckoff position, maximal or not, because the adiabatic moves
# that define the SRSIs are the moves between them.  Only some keys induce an
# elementary band representation, and the basis holds the others too.
keys = t211.site_keys
row("the basis", f"{len(keys)} site irreps at Wyckoff positions, "
    f"{len([k for k in keys if k.maximal])} of them at maximal ones,",
    f"{len([k for k in keys if k.ebr])} inducing an elementary band representation")
row("inducing one", *wrapped([f"({k.irrep})_{k.letter}" for k in keys if k.ebr]))
row("inducing none", *wrapped([f"({k.irrep})_{k.letter}" for k in keys if not k.ebr]))
band_terms = [f"{v} theta_{i + 1}" for i, v in enumerate(t211.band_count_form) if v]
row("the band count", *wrapped(["nu = " + band_terms[0]] + band_terms[1:], " + "))
print("   the band count is itself an SRSI, and bounds the enumeration of atomic realizations")

########### 2. two site irreps the invariants do not distinguish ###########

print("\n2. two site irreps at one position the invariants do not distinguish")

# srsi_of takes p as a mapping over site keys -- the paper's own basis -- and
# returns the Z-valued half and the Z_n-valued half.
b2 = tqc.srsi_of(t211, {("B2", "d"): 1})
b3 = tqc.srsi_of(t211, {("B3", "d"): 1})
row("1 (B2)_d", f"Z-valued {written(b2[0])}, {moduli(b2[1])}")
row("1 (B3)_d", f"Z-valued {written(b3[0])}, {moduli(b3[1])}")

pairs211 = tqc.stably_equivalent_pairs(t211)
for pair in pairs211:
    row("the pair", f"{' and '.join(pair.irreps)} at Wyckoff position {pair.letter}, their "
        "elementary band", f"representations {' and '.join(pair.names)}")
if pairs211:
    print("   both induce the same content at every high-symmetry point and carry the same")
    print("   SRSIs, so no measurement in momentum space or in real space tells them apart")

########### 3. printed indicator forms that are not indicators ###########

print("\n3. #147.14, single-valued: printed indicator forms that are not indicators")
t147 = tqc.srsi_tables(147, spinor=False)
row("the tables", f"#{t147.number_str}, tabulated indicators "
    + ", ".join(f"{f.name} (mod {f.mod})" for f in t147.indicators))

# (Au)_a induces an elementary band representation of this very group -- an
# atomic limit -- so every indicator must read zero on it.  Here is what each
# printed form gives.
au = {("Au", "a"): 1}
au_key = [k for k in t147.site_keys if (k.irrep, k.letter) == ("Au", "a")][0]
raw = t147.theta_unreduced(au)
printed = []
for form in t147.indicators:
    value = sum((c * raw[i - 1] for i, c in form.theta.items()), Fraction(0))
    printed.append(f"{form.name} = {int(value) % form.mod} mod {form.mod}")
row(f"1 ({au_key.irrep})_{au_key.letter}", "an elementary band representation of "
    f"#{t147.number_str}, an atomic limit,", "so every indicator must read 0 on it")
row("its SRSIs", f"Z-valued {written(t147.theta(au)[0])}")
row("printed forms", ", ".join(printed))
row("derived forms", ", ".join(f"{name} = {value}"
                               for name, value in t147.indicators_of(au).items()))

# ... and how many forms in all the tables behave this way, swept here.
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
            # a form whose name has no counterpart to derive it from has no
            # derived form to compare with
            relation = form.relation or "with no derived form"
            relations[relation] = relations.get(relation, 0) + 1
            if form.relation == "different":
                different.append(f"#{table.number_str} {form.name}")
row("over all systems", f"{sum(relations.values())} tabulated indicator forms against the "
    "derived ones:", ", ".join(f"{n} {name}" for name, n
                              in sorted(relations.items(), key=lambda kv: -kv[1])))
if different:
    row("the different", ", ".join(different))
    print("   where they differ, the indicators returned are the derived forms, which")
    print("   vanish on every elementary band representation; the printed forms are kept")
    print("   beside them")

########### 4. the supplement's splittings, re-asked ###########

print("\n4. the splittings the symmetry indicators allow, re-asked with a full enumeration")

# The supplement lists every way an elementary band representation can split
# into others that the symmetry indicators allow, with a flag saying whether
# the SRSIs of the parts mismatch.  Both halves are measured below: that flag,
# recomputed from the invariants it is printed beside; then the criterion,
# which asks the question the flag only sampled.
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
            # ... and the criterion: the parent as all the bands (every invariant
            # of it known exactly), its first child as the valence bands and the
            # rest as the conduction bands
            first = [0] * len(table.ebrs)
            first[int(split.children[0]["column"])] = 1
            theta_total = table.theta_columns(parent)
            theta_v = table.theta_columns(first)[0]
            theta_c = tuple(a - b for a, b in zip(theta_total[0], theta_v))
            got = nsit_criterion(table, theta_total, theta_v, theta_c)
            key = (bool(split.srsi_mismatch), kind, got.verdict)
            tally[key] = tally.get(key, 0) + 1
            entry = (table.number_str, split.parent["irrep"], split.parent["letter"],
                     tuple(f"({c['irrep']})_{c['letter']}" for c in split.children),
                     got.verdict, len(got.consistent_pairs))
            if not split.srsi_mismatch:
                unflagged.append(entry)
            if split.srsi_mismatch and not got.nsit:
                exceptions.append(entry)

total = counts["single"] + counts["double"]
row("the list", f"{total} splittings of an elementary band representation into",
    f"others, {counts['single']} single-valued and {counts['double']} double-valued")
row("the paper's flag", f"{flagged} of them, where the Z_n-valued invariants of the parts do not",
    "add up to the whole's; recomputed from the invariants printed",
    f"beside it, the flag agrees on {reproduced} of {total}")
row("the criterion", "the parent as all the bands, its first child as the valence bands,",
    "the rest as the conduction bands")
print("      flag        valuedness  verdict                     count")
for key in sorted(tally, key=lambda k: (not k[0], k[1], k[2])):
    print(f"      {'flagged' if key[0] else 'unflagged':<11} {key[1]:<11} "
          f"{key[2]:<27} {tally[key]}")
diagnosed = sum(v for k, v in tally.items()
                if k[0] and k[2] == "diagnosed topological")
row("the headline", f"{diagnosed} of the {flagged} flagged splittings stay diagnosed "
    "topological")
for number, irrep, wyckoff, children, verdict, npairs in exceptions:
    row("the exception", f"#{number}, parent ({irrep})_{wyckoff}, first child {children[0]}, "
        f"rest {' + '.join(children[1:])},",
        f"{npairs} consistent pair{'' if npairs == 1 else 's'}")
n_no_zn = sum(1 for entry in unflagged if entry[4] == "no Z_n SRSIs")
if unflagged:
    row("the unflagged", f"{len(unflagged)} splittings, {n_no_zn} of them in groups with no "
        "Z_n-valued invariant")
    print(f"      {'group':<10}{'parent':<11}{'first child':<13}{'rest':<13}"
          f"{'verdict':<24}consistent pairs")
    for number, irrep, wyckoff, children, verdict, npairs in unflagged:
        parent_label = f"({irrep})_{wyckoff}"
        print(f"      #{number:<9}{parent_label:<11}{children[0]:<13}"
              f"{' + '.join(children[1:]):<13}{verdict:<24}{npairs}")
if exceptions or unflagged:
    print("   where a consistent pair exists, any topology of the splitting is invisible")
    print("   to the SRSIs")

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
row("the sweep", f"{len(found)} pairs in {len(groups)} space groups, "
    f"{sum(1 for p in found if p[1])} of them double-valued")

by_signature = {}
for _sg, _sp, _w, _irreps, signature in found:
    by_signature[signature] = by_signature.get(signature, 0) + 1
row("by site symmetry", *(f"{by_signature[s]} where the site irreps are {', '.join(s)}"
                          for s in sorted(by_signature, key=lambda s: -by_signature[s])))

by_members = {}
for _sg, _sp, _w, irreps, _s in found:
    by_members[irreps] = by_members.get(irreps, 0) + 1
row("by the members", *wrapped([f"{v} of {' and '.join(k)}" for k, v
                                in sorted(by_members.items(), key=lambda kv: -kv[1])]))
row("space groups", *wrapped([str(sg) for sg in groups]))
if found:
    print("   both members of each pair carry the same SRSIs, Z-valued and Z_n-valued, so")
    print("   whichever one a decomposition picks, no invariant changes")

########### 6. the figure ###########

print("\n6. the figure: the verdicts on the splittings, and the pairs by site symmetry")
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
