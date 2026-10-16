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
    where = "(" + ", ".join(f"{x:g}" for x in site) + ")"
    pair = ", ".join(f"p{c}" for c in "xyz" if c != axis)
    print(f"   {where:<17} {axis:<6} {pair}")
print(f"   {model.norb} orbitals, {model.nstate} bands; {len(model.hoppings)} hoppings, "
      "random, to every orbital within")
print("   one lattice constant, averaged over the operations of T_d")

########### 2. the group the hoppings obey ###########

print("\n2. the group: what the point set allows, and what the hoppings obey")
r = tqc.realized_group(model)
group = r.group
wyckoff = tqc.wyckoff_sites_tables(group, r.assignment.site_positions)   # letter and multiplicity
allowed = r.descent["from"]
print(f"   the point set allows : #{allowed['number_str']} {allowed['name']}")
print(f"   the hoppings obey    : #{group.number_str} {group.name}, orbitals at Wyckoff "
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
print("   broken               : " + ", ".join(census) + ",")
print(f"                          each also with time reversal: {len(r.descent['dropped'])} "
      f"of the {allowed['size']} coset")
print("                          representatives of the lattice translations")

# about each site's own axis: the fourfold rotation and the -4 rotoinversion
c4_broken = [f"C4({a})" for a in axes if f"C4({a})" in lost]
s4_kept = [f"S4({a})" for a in axes if f"S4({a})" in kept]
print(f"   about the site axes  : {', '.join(c4_broken)} broken; "
      f"{', '.join(s4_kept)} kept")

# The kept operations are a subgroup of index two, so with inversion among the
# lost ones, the lost ones are inversion times the kept ones.
print("   -> the hoppings break inversion, and the operations lost are inversion")
print("      times those kept: each site loses the fourfold rotation about its own")
print("      axis and keeps the -4 rotoinversion about it")

########### 3. the atomic-limit table ###########

print("\n3. the atomic-limit table: what each orbital carries, read off the hoppings")
rec = r.rep.characters
print(rec.summary())
print(f"   -> nothing is declared: the hoppings fix each row's {rec.irreps[0]} themselves, "
      "the two-dimensional")
print(f"      irrep of the site's own {rec.orbitals[0].site_symmetry}, and no row names a "
      "function: which real functions")
print("      a pair of Wannier functions is, inside its site, is a choice of basis, and")
print("      the orbital characters are what the hoppings fix.")

# Each site's characters, one operation per class of its own -42m.  The twofold
# and the -4 about the site's own axis, a twofold about a perpendicular cube
# axis, and a mirror through the axis along a face diagonal: each named in the
# frame of the site it fixes, which is a different frame at each site.
print("   each site's characters, named in its own axes:")
chars_of = {}
# r.assignment.site_orbitals lists, site by site, the orbitals that sit there
for pair in r.assignment.site_orbitals:
    site = tuple(float(x) for x in positions[pair[0]])
    axis = axes[sites.index(site)]
    chi = {name: complex(v).real for name, v in rec.orbitals[pair[0]].generator_character.items()}
    perp = next(name for name in chi if name.startswith("C2(") and name != f"C2({axis})")
    mirror = next(name for name in chi if name.startswith("m("))
    chars_of[axis] = chi
    where = "(" + ", ".join(f"{x:g}" for x in site) + ")"
    shown = [("E", "E"), (f"C2({axis})", f"C2({axis})"), (perp, perp),
             (f"S4({axis})", f"S4({axis})"), (mirror, mirror)]
    print(f"      {where:<15} axis {axis}:  "
          + ",  ".join(f"{label} {chi[name]:+.0f}" if abs(chi[name]) > 1e-9
                       else f"{label} 0" for label, name in shown))
print(f"   -> at every site the character is that of {rec.irreps[0]}: +2 on the identity, "
      "-2 on the twofold")
print("      about the site's own axis, and 0 on the -4 about it, on a perpendicular twofold")
print("      and on a diagonal mirror.  The numbers are the same at the three sites, and the")
print("      operations they belong to are not: each is named in its own site's axes.")

# The threefold along [111] carries each site onto the next, and each site's
# own axis onto the next site's: its matrix sends each site's pair into the
# pair at one other site, and its rotation part sends one cube axis to another.
c3 = tqc.operation_index(r, "{3+_111|0,0,0}")
M3 = np.asarray(r.rep.matrices[c3])
# an operation's rotation_cartesian is its rotation, a 3 x 3 matrix on Cartesian vectors
R3 = np.asarray(r.operations[c3].rotation_cartesian, dtype=float)
unit = {"x": np.eye(3)[0], "y": np.eye(3)[1], "z": np.eye(3)[2]}
print(f"   -> {r.operations[c3].name} carries each site onto the next, and its axis onto "
      "the next site's:")
for pair in r.assignment.site_orbitals:
    site = tuple(float(x) for x in positions[pair[0]])
    axis = axes[sites.index(site)]
    target = int(np.argmax(np.abs(M3[:, pair[0]])))     # an orbital of the image site
    there = tuple(float(x) for x in positions[target])
    turned = next(b for b, v in unit.items() if np.allclose(np.abs(R3 @ unit[axis]), v))
    here = "(" + ", ".join(f"{x:g}" for x in site) + ")"
    far = "(" + ", ".join(f"{x:g}" for x in there) + ")"
    print(f"         {here:<15} -> {far:<15} :  axis {axis} -> {turned},  "
          f"C2({axis}) -> C2({turned}), character "
          f"{chars_of[axis][f'C2({axis})']:+.0f} at both")
print("      so each site's characters are the image of the first site's, carried by the")
print(f"      operations of the orbit, and each row is the {rec.irreps[0]} of its own site's "
      f"{rec.orbitals[0].site_symmetry}.")

# The line under the table is the band representation the orbit induces,
# written in elementary ones: a sum of several, none of them at the orbit's own
# position.
print(f"   -> {rec.irreps[0]} is named by the point group: at {rec.orbitals[0].wyckoff} of "
      f"{group.name} it induces a composite band")
print("      representation, one of the known exceptions to an irrep at a maximal Wyckoff")
print("      position inducing an elementary one, so no elementary band representation")
print("      of the tables carries its name; the line under the table writes out the")
print("      elementary ones it is the sum of.")

########### 4. the same model in a rotated basis ###########

print(f"\n4. the same model, every pair rotated inside itself by {THETA} rad")
print("   a change of orbital basis at every site, the same at each: a choice of")
print("   Wannier gauge")
rotated = cubic_3c_perp_doublet(angle=THETA)
r_rot = tqc.realized_group(rotated)
rec_rot = r_rot.rep.characters
print(f"   the hoppings obey    : #{r_rot.group.number_str} {r_rot.group.name}")
# the rotated model's atomic-limit table, compared line for line with section 3's
print(f"   its atomic-limit table is section 3's, line for line: "
      f"{rec_rot.summary() == rec.summary()}")
print("   -> the group, the irreps and the characters at every row are those of section 3:")
print("      a character is a trace, and no change of basis inside a pair moves a trace.")

table = tqc.hsp_table(r.rep, bands=6)          # the six bands, at every point
table_rot = tqc.hsp_table(r_rot.rep, bands=6)
print(f"   the six bands        : {fmt(table.counts)}")
print(f"   {f'rotated by {THETA} rad':<21}: {fmt(table_rot.counts)}")

########### 5. the six bands as elementary band representations ###########

print("\n5. the six bands as a sum of elementary band representations")
d6 = tqc.decompose(table, group, rep=r.rep)
where = " and ".join(dict.fromkeys(d6.positions[0]))
print(f"   decompose            : {d6.verdict}, {len(d6.solutions)} solution"
      f"{'s' * (len(d6.solutions) != 1)}")
print(f"                          = {d6.solution_strings()[0]}")

# the bands' sum against the one the atomic-limit table wrote off the orbitals.
# solution_strings() writes each term under the label the group's list of
# elementary band representations gives it: the site irrep, induced (↑G) in its
# number of bands from its Wyckoff position, whose site group follows with time
# reversal and without.
print(f"   -> the band representation {rec.irreps[0]} at {rec.orbitals[0].wyckoff} "
      f"induces is this sum of elementary")
print(f"      ones at {where}: that is what composite means.  The line under the")
print("      atomic-limit table of section 3 is the same sum, each term written there")
print("      as its site irrep at its Wyckoff position.")

# the figure: the bands of both bases along one path, the content at each point
hsp = tqc.high_symmetry_points(group)
path = [hsp[point] for point in NODES]
kv, kd, kn = model.k_path(path, NK_PATH, report=False)
bands = np.asarray(model.solve_ham(kv))
bands_rot = np.asarray(rotated.solve_ham(kv))
print(f"   the figure           : the six bands along {'-'.join(NODES)}, one spectrum in "
      "both bases,")
print(f"                          max |E - E_rotated| = "
      f"{np.abs(bands - bands_rot).max():.6f} along the path")

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
