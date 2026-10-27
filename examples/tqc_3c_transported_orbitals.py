# Example of the orbitals at the sites of one Wyckoff orbit as one another's
# images, using pythtqc.
# The model is a p pair at each of the three 3c sites of a simple cubic cell,
# each pair perpendicular to its own site's fourfold axis: (p_y, p_z) at
# (0, 1/2, 1/2), (p_x, p_z) at (1/2, 0, 1/2) and (p_x, p_y) at (1/2, 1/2, 0),
# with random hoppings averaged over the operations of T_d.  It is
# pythtqc.models' cubic_3c_perp_doublet().  An operation carrying one site onto
# another carries its pair onto a pair of different functions, and the
# question is what the atomic-limit table of such an orbit says, and what band
# representation it induces.
# Run with python3, PythTB version 2.0 and pythtqc.

# The calculations in this python file include
# 1. the model: the three 3c sites, each with its own axis and its p pair
# 2. the group the hoppings obey, and the operations they break
# 3. the atomic-limit table: each site's E of its own -42m, its characters
#    named in that site's own axes, the threefold that carries one site's
#    characters onto the next's, and the band representation the orbit induces
# 4. the same model with every pair rotated inside itself by one angle
# 5. the six bands as a sum of elementary band representations, and the figure

# Author: the pythtqc examples

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401  (in-repo runs: puts src/ on sys.path)
import pythtqc as tqc
from pythtqc import format_counts as fmt
from pythtqc.models import cubic_3c_perp_doublet
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
    "tqc_3c_transported_orbitals")
NK_PATH = 121 if QUICK else 601            # band-path density: the figure only

THETA = 0.37               # the rotation inside every pair in section 4, in radians

NODES = ("GM", "X", "M", "GM", "R", "X")  # the figure's path
TICKS = (r"$\Gamma$", "X", "M", r"$\Gamma$", "R", "X")

os.makedirs(OUTDIR, exist_ok=True)

print("Six p orbitals at the three 3c sites of a simple cubic cell, a pair at each")
print("site perpendicular to the site's own axis: the orbitals at the three sites")
print("are one another's images.")


def row(label, first, *rest):
    """One line of the ledger, a label of at most 16 characters, then its continuation lines."""
    print(f"   {label:<16} : {first}")
    for line in rest:
        print(f"{'':22}{line}")


def coords(site):
    """A site's position in the cell, its three coordinates written out."""
    return "(" + ", ".join(f"{x:g}" for x in site) + ")"


def atomic_limits(d):
    """The atomic limits of a band set, as a label and a value: the one sum of elementary
    band representations with this content, how many sums there are, or none."""
    if d.verdict == "none":
        return "atomic limits", "none"
    sums = d.solution_strings()
    if len(sums) == 1:
        return "atomic limit", sums[0]
    return "atomic limits", f"{len(sums)} sums of elementary band representations"


def differing(table, other):
    """Where another atomic-limit table differs: its rows, by orbital, and the lines under them."""
    first, second = table.summary().splitlines(), other.summary().splitlines()
    rows = [str(i) for i in range(len(table))
            if i + 1 >= len(second) or first[i + 1] != second[i + 1]]
    named = "the row of orbital " if len(rows) == 1 else "the rows of orbitals "
    parts = ([named + ", ".join(rows)] if rows else []) + (
        ["the lines under the rows"] if first[len(table) + 1:] != second[len(table) + 1:] else [])
    return " and ".join(parts) or "the line naming its columns"


########### 1. the model ###########

print("\n1. the model: a p pair at each 3c site, perpendicular to the site's own axis")
model = cubic_3c_perp_doublet()

# The three sites, read off the model in orbital order.  Each is the centre of a
# cube face, and its own axis is the cube axis normal to that face: the
# coordinate that vanishes.
positions = np.asarray(model.orb_vecs, dtype=float)
sites = list(dict.fromkeys(tuple(float(x) for x in p) for p in positions))
axes = ["xyz"[int(np.argmin(site))] for site in sites]
print("   site              axis   orbitals")
for site, axis in zip(sites, axes):
    pair = ", ".join(f"p{c}" for c in "xyz" if c != axis)
    print(f"   {coords(site):<17} {axis:<6} {pair}")
row("bands", f"{model.nstate}, from {model.norb} orbitals")
row("hoppings", f"{len(model.hoppings)}, random, to every orbital within one lattice constant,",
    "averaged over the operations of T_d")

########### 2. the group the hoppings obey ###########

print("\n2. the group: what the point set allows, and what the hoppings obey")
r = tqc.realized_group(model)
group = r.group
wyckoff = tqc.wyckoff_sites_tables(group, r.assignment.site_positions)   # letter and multiplicity
allowed = r.descent["from"]
row("point set allows", f"#{allowed['number_str']} {allowed['name']}")
row("realized group", f"#{group.number_str} {group.name}, orbitals at Wyckoff "
    f"{' and '.join(dict.fromkeys(f'{s.multiplicity}{s.letter}' for s in wyckoff))}")

# The descent names every operation the hoppings break.  The unitary ones are
# counted here by kind, each kind shown by its first member: C4^-1(z) counts as
# a C4, and "6 like C2([110])" is the six twofolds about the face diagonals.
lost = [b["name"] for b in r.descent["dropped"] if not b["name"].endswith("'")]
kinds = {}
for name in lost:
    kind = name.split("(")[0].split("^")[0]
    kinds.setdefault(kind, []).append(name.replace("^-1", ""))
census = [names[0] if len(names) == 1 else f"{len(names)} like {names[0]}"
          for names in sorted(kinds.values(), key=len)]
kept = [op.name for op in r.operations.values() if not op.time_reversal]
row("broken", ", ".join(census) + ",",
    f"each also with time reversal: {len(r.descent['dropped'])} of the {allowed['size']} "
    "coset representatives",
    "of the lattice translations")

# about each site's own axis: the fourfold rotation and the -4 rotoinversion
c4_broken = [f"C4({a})" for a in axes if f"C4({a})" in lost]
s4_kept = [f"S4({a})" for a in axes if f"S4({a})" in kept]
row("each site's axis", f"{', '.join(c4_broken)} broken; {', '.join(s4_kept)} kept")

# The kept operations are a subgroup of index two, so with inversion among the
# lost ones, the lost ones are inversion times the kept ones.
print("   the operations broken are inversion times those kept: each site loses the fourfold")
print("   rotation about its own axis and keeps the -4 rotoinversion about it")

########### 3. the atomic-limit table ###########

print("\n3. r.rep.characters: the atomic-limit table, what each orbital carries")
rec = r.rep.characters
print(textwrap.indent(rec.summary(), "   "))

# Each site's characters, one operation per class of its own -42m, each named in
# the frame of the site it fixes, which is a different frame at each site: the
# twofold about the site's own axis is the one the threefold below carries from
# site to site.
chars_of = {}
# r.assignment.site_orbitals lists, site by site, the orbitals that sit there
for pair in r.assignment.site_orbitals:
    site = tuple(float(x) for x in positions[pair[0]])
    axis = axes[sites.index(site)]
    chi = {name: complex(v).real for name, v in rec.orbitals[pair[0]].generator_character.items()}
    chars_of[axis] = chi

# The threefold along [111] carries each site onto the next, and each site's
# own axis onto the next site's: its matrix sends each site's pair into the
# pair at one other site, and its rotation part sends one cube axis to another.
c3 = tqc.operation_index(r, "{3+_111|0,0,0}")
M3 = np.asarray(r.rep.matrices[c3])
# an operation's rotation_cartesian is its rotation, a 3 x 3 matrix on Cartesian vectors
R3 = np.asarray(r.operations[c3].rotation_cartesian, dtype=float)
unit = {"x": np.eye(3)[0], "y": np.eye(3)[1], "z": np.eye(3)[2]}
row(r.operations[c3].name, "carries each site onto the next, and its axis onto the next site's")
for pair in r.assignment.site_orbitals:
    site = tuple(float(x) for x in positions[pair[0]])
    axis = axes[sites.index(site)]
    target = int(np.argmax(np.abs(M3[:, pair[0]])))     # an orbital of the image site
    there = tuple(float(x) for x in positions[target])
    turned = next(b for b, v in unit.items() if np.allclose(np.abs(R3 @ unit[axis]), v))
    row(coords(site), f"onto {coords(there)}, axis {axis} onto {turned}, C2({axis}) onto "
        f"C2({turned}), character {chars_of[axis][f'C2({axis})']:+.0f} at both")
print("   each site's characters are the first site's, carried by the operations of the orbit,")
print(f"   and each row is the {rec.irreps[0]} of its own site's {rec.orbitals[0].site_symmetry}")

########### 4. the same model in a rotated basis ###########

print(f"\n4. the same model, every pair rotated inside itself by {THETA} rad")
rotated = cubic_3c_perp_doublet(angle=THETA)
r_rot = tqc.realized_group(rotated)
rec_rot = r_rot.rep.characters
row("realized group", f"#{r_rot.group.number_str} {r_rot.group.name}")
# the rotated model's atomic-limit table, compared line for line with section 3's
if rec_rot.summary() == rec.summary():
    row("rep.characters", "section 3's atomic-limit table, line for line")
else:
    row("rep.characters", "section 3's atomic-limit table but for " + differing(rec, rec_rot))

table = tqc.hsp_table(r.rep, bands=6)          # the six bands, at every point
table_rot = tqc.hsp_table(r_rot.rep, bands=6)
row("all 6 bands", fmt(table.counts))
row("all 6, rotated", fmt(table_rot.counts))
print("   the rotation is a change of orbital basis inside each pair, a choice of Wannier gauge,")
print("   and a character is a trace, which no change of basis moves")

########### 5. the six bands as elementary band representations ###########

print("\n5. the six bands as a sum of elementary band representations")
d6 = tqc.decompose(table)
row(*atomic_limits(d6))

# the figure: the bands of both bases along one path, the content at each point
hsp = tqc.high_symmetry_points(group)
path = [hsp[point] for point in NODES]
kv, kd, kn = model.k_path(path, NK_PATH, report=False)
bands = np.asarray(model.solve_ham(kv))
bands_rot = np.asarray(rotated.solve_ham(kv))
row("both bases", f"largest energy difference {np.abs(bands - bands_rot).max():.6f} along "
    f"{'-'.join(NODES)}, the figure's path")
print(f"   {rec.irreps[0]} at {rec.band_representations[0].wyckoff} of {group.name} is one of the known exceptions to "
      "an irrep at a maximal")
print("   Wyckoff position inducing an elementary band representation")

fig, ax = plt.subplots(1, 1, figsize=(9.2, 6.4))
for i in range(model.nstate):
    ax.plot(kd, bands[:, i], lw=1.8, color="tab:blue",
            label="the pairs as built" if i == 0 else None)
    ax.plot(kd[::4], bands_rot[::4, i], ls="none", marker="o", ms=3.2,
            color="tab:orange",
            label=f"every pair rotated by {THETA} rad" if i == 0 else None)
for x in kn:
    ax.axvline(x, color="0.82", lw=0.7)
low = bands.min() - 0.06 * (bands.max() - bands.min())
for point, x in zip(NODES, kn):
    ax.annotate(fmt(table.content[point]), xy=(x, low), ha="center", va="top",
                fontsize=8.5, color="tab:blue",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_xticks(kn)
ax.set_xticklabels(TICKS, fontsize=13)
ax.set_xlim(kd[0] - 0.07 * kd[-1], kd[-1] * 1.07)
ax.set_ylim(bands.min() - 0.20 * (bands.max() - bands.min()),
            bands.max() + 0.14 * (bands.max() - bands.min()))
ax.set_ylabel("$E$", fontsize=15)
ax.legend(loc="upper center", fontsize=9)
ax.set_title(f"#{group.number_str} {group.name}: {rec.irreps[0]} at "
             f"{rec.orbitals[0].wyckoff} of {rec.orbitals[0].site_symmetry}, the six "
             f"bands\n{d6.solution_strings()[0]}", fontsize=10)
fig.tight_layout(pad=0.6)
# save_figure writes the PDF without its creation date, so a rerun gives the same file
save_figure(fig, os.path.join(OUTDIR, "3c_transported_orbitals.pdf"))
plt.close(fig)

print("\nDone — figure written to", OUTDIR)
