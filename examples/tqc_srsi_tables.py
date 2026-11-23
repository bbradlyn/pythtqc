# Example of the tables of stable real-space invariants (SRSIs), read with no
# model, using pythTQC.
# The tables are those of Hwang et al., "Stable real-space invariants and
# topology beyond symmetry indicators", Nat. Commun. 17, 8564 (2026),
# 10.1038/s41467-026-74844-w, converted from the paper's supplement and shipped
# with pythTQC.
# The site-symmetry vector p holds the multiplicity of each site irrep at each
# Wyckoff position. An SRSI is a linear combination of the entries of p that no
# smooth deformation preserving the symmetry changes. Some SRSIs are
# integer-valued, printed Z-valued, and some are valued modulo n. Those valued
# modulo n are not fixed by the symmetry data vector, and so can tell apart
# atomic limits with the same symmetry data vector. The paper writes each
# symmetry indicator as a form, a linear combination of the SRSIs.
# An atomic realization of a set of bands is a set of site irreps at maximal
# Wyckoff positions, with nonnegative multiplicities, that has its Z-valued
# SRSIs. The no-go criterion calls a gapped splitting of bands
# non-symmetry-indicated topological (nsit) where both sides have atomic
# realizations and no pair of them, one of each side, has SRSIs valued modulo n
# adding up to those of all the bands. A pair that does is a consistent pair.
# Run with python3, PythTB version 2.0 and pythTQC.

# The calculations in this python file include
# 1. srsi_tables(211, spinor=False): the SRSIs of one group, its site irreps,
#    which of them are at maximal Wyckoff positions, and which induce an
#    elementary band representation
# 2. two site irreps at one Wyckoff position that the SRSIs do not distinguish
# 3. space group 147, single-valued: the two indicator forms the paper prints,
#    which do not vanish on an elementary band representation of the group, and
#    how the forms of all the tables compare with a second form pythTQC derives
#    where it can
# 4. the splittings of an elementary band representation the paper lists: its
#    flag on each recomputed, the no-go criterion applied to each over every
#    atomic realization, and the unflagged ones one by one
# 5. the pairs of site irreps the SRSIs do not distinguish, over all the tables
# 6. the figure: the criterion on the splittings, and the pairs by site symmetry

# Author: the pythTQC examples

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

# The output is written as UTF-8, whatever the console's own encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

########### constants ###########

# TQC_EXAMPLE_QUICK changes nothing here. The sweeps below run over all the
# tables.
# TQC_EXAMPLE_FIGDIR is the directory the figure is written to when it is set.
OUTDIR = os.environ.get("TQC_EXAMPLE_FIGDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "tqc_srsi_tables")

os.makedirs(OUTDIR, exist_ok=True)

# srsi_tables takes the group of what realized_group returns, r.group, or a
# space-group number with spinor=. A number needs no model.
t211 = tqc.srsi_tables(211, spinor=False)
print("The tables of stable real-space invariants (SRSIs) of")
print(f"{t211.provenance.doi}, read with no model.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def written(theta):
    """Invariants written out, ``(1, 0, -1)``."""
    return "(" + ", ".join(str(v) for v in theta) + ")"


def moduli(theta_zn):
    """The invariants valued modulo n, by modulus, ``mod 2 (0, 1)``."""
    by_modulus = {}
    for entry in theta_zn.values():
        by_modulus.setdefault(entry.modulus, []).append(entry.value)
    return ", ".join(f"mod {n} {written(values)}" for n, values in sorted(by_modulus.items()))


def letter(wyckoff):
    """The Wyckoff letter of a position written with its multiplicity, ``d`` of ``12d``."""
    return wyckoff.lstrip("0123456789")


def wrapped(items, sep=", ", width=64):
    """Items joined by sep, broken between items into lines of at most width characters.

    A list keeps its comma at the end of a line.
    """
    lines = []
    for item in items:
        if lines and len(lines[-1]) + len(sep) + len(item) <= width:
            lines[-1] += sep + item
        elif lines:
            lines[-1] += sep.rstrip()
            lines.append(item)
        else:
            lines.append(item)
    return lines


########### 1. one group's tables, read end to end ###########

print("\n1. srsi_tables(211, spinor=False): one group's tables, end to end")
keys = t211.site_irreps
row("the tables", f"#{t211.group.number_str}, {len(t211.invariants)} SRSIs of a site-symmetry "
    f"vector of {len(keys)} site irreps,",
    f"{len([k for k in keys if k.ebr])} elementary band representations, "
    f"{len(t211.indicators)} symmetry indicators,",
    f"{len(t211.split_ebrs)} splittings of an elementary band representation into others")
row("the SRSI group", " x ".join("Z" if t211.moduli[name] is None else f"Z{t211.moduli[name]}"
                                 for name in t211.invariants))

# The entries of p run over the site irreps of every Wyckoff position, maximal
# or not, since a smooth deformation moves orbitals between them. Only some of
# these site irreps induce an elementary band representation.
row("the entries of p", f"{len(keys)} site irreps at Wyckoff positions, "
    f"{len([k for k in keys if k.maximal])} of them at maximal ones,",
    f"{len([k for k in keys if k.ebr])} inducing an elementary band representation "
    f"and {len([k for k in keys if not k.ebr])} a composite one")
row("elementary", *wrapped([f"({k.irrep})_{letter(k.wyckoff)}" for k in keys if k.ebr]))
row("composite", *wrapped([f"({k.irrep})_{letter(k.wyckoff)}" for k in keys if not k.ebr]))
print("   the band count is an integer combination of the Z-valued SRSIs, and every site irrep\n"
      "   adds bands, so a set of bands has finitely many atomic realizations")

########### 2. two site irreps the invariants do not distinguish ###########

print("\n2. two site irreps at one Wyckoff position that the SRSIs do not distinguish")

# theta takes p as a mapping {(site irrep, Wyckoff position): multiplicity}, and
# returns the integer-valued SRSIs and those valued modulo n, by name.
b2 = t211.theta({("B2", "12d"): 1})
b3 = t211.theta({("B3", "12d"): 1})
row("1 (B2)_d", f"Z-valued {written(b2.theta_Z.values())}, {moduli(b2.theta_Zn)}")
row("1 (B3)_d", f"Z-valued {written(b3.theta_Z.values())}, {moduli(b3.theta_Zn)}")

pairs211 = t211.stably_equivalent
for pair in pairs211:
    row("the pair", f"{' and '.join(pair.irreps)} at Wyckoff position {pair.wyckoff}, each "
        "inducing an", "elementary band representation")
if pairs211:
    print("   their elementary band representations have the same symmetry data vector, and")
    print("   the two carry the same SRSIs, so neither the symmetry data vector nor the SRSIs\n"
          "   tells them apart")

########### 3. the paper's two indicator forms, nonzero on an atomic limit ###########

print("\n3. #147.14, single-valued: the paper's two indicator forms, nonzero on an atomic limit")
t147 = tqc.srsi_tables(147, spinor=False)
row("the tables", f"#{t147.group.number_str}, tabulated indicators "
    + ", ".join(f"{name} (mod {t147.moduli[name]})" for name in t147.indicators))

# (Au)_a induces an elementary band representation of this group, an atomic
# limit, on which every indicator vanishes. The check of each form the paper
# prints lists the elementary band representations it does not vanish on, with
# its value on each. pythTQC derives, where it can, a second form of each
# indicator in the SRSIs. On every set of bands it takes, modulo the indicator's modulus, the
# value of the indicator of the same name that the irreptables package ships,
# or, where that package ships none and the symmetry-indicator group is cyclic,
# the symmetry-indicator class.
au = {("Au", "1a"): 1}
au_key = [k for k in t147.site_irreps if (k.irrep, k.wyckoff) == ("Au", "1a")][0]
printed = []
for name in t147.indicators:
    check = t147.numerical.indicators[name]
    value = next((Fraction(v) for irrep, wyckoff, v in check.ebr_failures
                  if (irrep, wyckoff) == ("Au", "1a")), Fraction(0))
    printed.append(f"{name} = {int(value) % t147.moduli[name]} mod {t147.moduli[name]}")
row(f"1 ({au_key.irrep})_{letter(au_key.wyckoff)}", "an elementary band representation of "
    f"#{t147.group.number_str}, an atomic limit,", "so every indicator must read 0 on it")
row("its SRSIs", f"Z-valued {written(t147.theta(au).theta_Z.values())}")
row("paper's forms", ", ".join(printed))
row("derived forms", ", ".join(f"{name} = {value}"
                               for name, value in t147.indicators_of(au).items()))

# The sweep below compares the forms of all the tables with the derived ones.
# Two forms are equal where their coefficients are, equivalent where their
# coefficients differ and their values agree modulo the indicator's modulus on
# every set of bands, and different where their values differ on some set of
# bands. Where they differ, indicators_of evaluates the derived form, and
# tables.indicators keeps the form the paper prints.
# The 13 space groups whose only Wyckoff position is the general position have
# one SRSI, the band count, and no table. srsi_tables refuses them.
relations = {}
different = []
for sg in range(1, 231):
    for spinor in (False, True):
        table = tqc.srsi_tables(sg, spinor=spinor)
        if isinstance(table, tqc.Refusal):
            continue
        for name in table.indicators:
            # a form with no derived form to compare with is counted apart
            check = table.numerical.indicators[name]
            relation = check.relation or "with no derived form"
            relations[relation] = relations.get(relation, 0) + 1
            if check.relation == "different":
                different.append(f"#{table.group.number_str} {name}")
row("over all tables", f"{sum(relations.values())} tabulated indicator forms against the "
    "derived ones:", ", ".join(f"{n} {name}" for name, n
                              in sorted(relations.items(), key=lambda kv: -kv[1])))
if different:
    row("the different", ", ".join(different))
    print("   where the two differ, the indicator is evaluated with the derived form, which")
    print("   vanishes on every elementary band representation. The form the paper prints is")
    print("   kept beside it")

########### 4. the splittings the paper lists, and the criterion ###########

print("\n4. the splittings the paper lists, and the no-go criterion on each")

# The paper's supplement lists splittings of an elementary band representation,
# the parent, into others, its children, whose symmetry data vectors add up to
# the parent's. Each is flagged where the SRSIs valued modulo n of the children
# do not add up to the parent's. The lines below recompute that flag from the
# tables, and then apply the no-go criterion to each splitting over every atomic
# realization of each side.
def reading(split):
    """The criterion's verdict on a splitting, with the reason for a ``not nsit``."""
    if any(v.denominator != 1 for v in split.valence.theta_Z.values()):
        return "not nsit, fractional"
    if split.nsit:
        return "nsit"
    if not split.theta_Zn:
        return "not nsit, no SRSI mod n"
    if split.consistent_pairs:
        return "not nsit, pairs add up"
    return "not nsit, a side not atomic"


# the order the readings are listed in
READINGS = ["nsit", "not nsit, pairs add up", "not nsit, no SRSI mod n",
            "not nsit, a side not atomic", "not nsit, fractional"]

counts ={"single": 0, "double": 0}
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
            parent = {(split.parent.irrep, split.parent.wyckoff): 1}
            theta_parent = table.theta_unreduced(parent)
            theta_children = {name: Fraction(0) for name in theta_parent}
            for child in split.children:
                for name, value in table.theta_unreduced({(child.irrep, child.wyckoff): 1}).items():
                    theta_children[name] += value
            mismatch = any((int(theta_parent[name]) - int(theta_children[name]))
                           % table.moduli[name]
                           for name in table.invariants if table.moduli[name] is not None)
            if mismatch == bool(split.srsi_mismatch):
                reproduced += 1
            # Then the no-go criterion: the parent is all the bands, its first
            # child the valence bands, and the other children the conduction
            # bands. The parent's SRSIs of both kinds are those of its site irrep.
            first = split.children[0]
            theta_total = table.theta(parent)
            theta_v = table.theta({(first.irrep, first.wyckoff): 1}).theta_Z
            theta_c = {name: theta_total.theta_Z[name] - theta_v[name] for name in theta_v}
            got = nsit_criterion(table, theta_total, theta_v, theta_c)
            key = (bool(split.srsi_mismatch), kind, reading(got))
            tally[key] = tally.get(key, 0) + 1
            entry = (table.group.number_str, split.parent.irrep, letter(split.parent.wyckoff),
                     tuple(f"({c.irrep})_{letter(c.wyckoff)}" for c in split.children),
                     reading(got), len(got.consistent_pairs))
            if not split.srsi_mismatch:
                unflagged.append(entry)
            if split.srsi_mismatch and not got.nsit:
                exceptions.append(entry)

total = counts["single"] + counts["double"]
row("the list", f"{total} splittings of an elementary band representation into",
    f"others, {counts['single']} single-valued and {counts['double']} double-valued")
row("the paper's flag", f"{flagged} of them flagged, where the SRSIs valued modulo n of the",
    "children do not add up to the parent's",
    f"recomputed from the tables, the flag agrees on {reproduced} of {total}")
row("the criterion", "the parent as all the bands, its first child as the valence bands,",
    "the other children as the conduction bands")
print("      flag        valued      nsit                        count")
for key in sorted(tally, key=lambda k: (not k[0], k[1], READINGS.index(k[2]))):
    print(f"      {'flagged' if key[0] else 'unflagged':<11} {key[1]:<11} "
          f"{key[2]:<27} {tally[key]}")
diagnosed = sum(v for k, v in tally.items() if k[0] and k[2] == "nsit")
row("the verdict", f"{diagnosed} of the {flagged} flagged splittings are nsit")
for number, irrep, wyckoff, children, verdict, npairs in exceptions:
    row("the exception", f"#{number}, parent ({irrep})_{wyckoff}, first child {children[0]}, "
        f"rest {' + '.join(children[1:])},",
        f"{npairs} consistent pair{'' if npairs == 1 else 's'}")
n_no_zn = sum(1 for entry in unflagged if entry[4] == "not nsit, no SRSI mod n")
if unflagged:
    row("the unflagged", f"{len(unflagged)} splittings, {n_no_zn} of them in groups with no "
        "invariant valued modulo n")
    print(f"      {'group':<10}{'parent':<11}{'first child':<13}{'rest':<13}"
          f"{'nsit':<28}consistent pairs")
    for number, irrep, wyckoff, children, verdict, npairs in unflagged:
        parent_label = f"({irrep})_{wyckoff}"
        print(f"      #{number:<9}{parent_label:<11}{children[0]:<13}"
              f"{' + '.join(children[1:]):<13}{verdict:<28}{npairs}")
if exceptions or unflagged:
    print("   where a consistent pair exists, the SRSIs are consistent with both sides being")
    print("   atomic limits")

########### 5. the pairs the SRSIs do not distinguish, everywhere ###########

print("\n5. the pairs of site irreps the SRSIs do not distinguish, over all the tables")
found = []
for sg in range(1, 231):
    for spinor in (False, True):
        table = tqc.srsi_tables(sg, spinor=spinor)
        if isinstance(table, tqc.Refusal):
            continue
        for pair in table.stably_equivalent:
            # the site irreps at the pair's Wyckoff position, which stand for its
            # site-symmetry group
            signature = tuple(sorted(k.irrep for k in table.site_irreps
                                     if k.wyckoff == pair.wyckoff))
            found.append((sg, spinor, pair.wyckoff, pair.irreps, signature))
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
    print("   both members of each pair carry the same SRSIs, Z-valued and valued modulo n, so")
    print("   whichever one a decomposition picks, no invariant changes")

########### 6. the figure ###########

print("\n6. the figure: the criterion on the splittings, and the pairs by site symmetry")
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

labels = []
values = []
colors = []
for key in sorted(tally, key=lambda k: (not k[0], k[1], READINGS.index(k[2]))):
    labels.append(f"{'flagged' if key[0] else 'unflagged'}, {key[1]}\n{key[2]}")
    values.append(tally[key])
    colors.append("tab:red" if key[2] == "nsit"
                   else ("tab:blue" if key[2] == "not nsit, pairs add up" else "0.6"))
axes[0].barh(range(len(values)), values, color=colors)
axes[0].set_yticks(range(len(values)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xscale("log")
for i, v in enumerate(values):
    axes[0].annotate(str(v), xy=(v, i), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9)
axes[0].set_xlim(0.7, max(values) * 2.4)
axes[0].set_xlabel("splittings (log scale)")
axes[0].set_title(f"the no-go criterion on the supplement's {total} splittings, over every\n"
                  f"atomic realization: {diagnosed} of the {flagged} flagged are nsit",
                  fontsize=10)

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

fig.suptitle("The SRSI tables, read with no model: the no-go criterion on the "
             "supplement's splittings of an elementary band representation, and the "
             "pairs of site irreps no SRSI distinguishes.", fontsize=10)
fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.93))
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "srsi_tables.pdf"))
plt.close(fig)

print("\nDone. The figure is written to", OUTDIR)
