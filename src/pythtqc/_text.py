"""The words the records' printouts share: a coordinate, a k-point, a character, a
count of operations, a set of orbitals, and a sum of elementary band
representations.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import _sitechars
from ._tolerance import CHARACTER, _LATTICE_TOL


def _fmt_char(z) -> str:
    # a character that vanishes prints as the 0 a table would carry -- unsigned,
    # and not the rounding it was measured to; the value itself stays complex
    z = complex(z)
    re_, im_ = (0.0 if abs(z.real) < 1e-9 else z.real), (0.0 if abs(z.imag) < 1e-9 else z.imag)
    if im_ == 0.0:
        return "0" if re_ == 0.0 else f"{re_:+.3g}"
    return f"{re_:+.3g}{im_:+.3g}j"


#: What a printout that counts operations counts: one per coset of the lattice
#: translations, and on states carrying a spin the single-valued group's, the
#: double group holding each one's negative besides.  Said once in a printout,
#: wherever the count is the information; where the group's symbol fixes the
#: count, no count is printed.
_COSETS = "one per coset of the lattice translations"

_SINGLE_VALUED = ("the single-valued group's, each standing for itself and its negative "
                  "in the double group")


def _counted_as(spinful) -> str:
    """The unit a count of operations is in, as a printout says it once."""
    return _COSETS + (f", {_SINGLE_VALUED}" if spinful else "")


def _index_ranges(indices) -> str:
    """Sorted indices as runs, ``0-3, 5, 7-8``."""
    runs, start, prev = [], None, None
    for i in sorted(int(i) for i in indices):
        if start is not None and i == prev + 1:
            prev = i
            continue
        if start is not None:
            runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    if start is not None:
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(runs)


def _orbitals_named(orbitals) -> str:
    """A set of orbitals in words: ``"orbital 3"``, ``"orbitals 0-2"``."""
    orbitals = tuple(int(o) for o in orbitals)
    return f"orbital{'s' if len(orbitals) > 1 else ''} {_index_ranges(orbitals)}"


#: How many sums of elementary band representations a summary prints before it
#: says how many more there are.
_SHOWN = 6


def _term_words(multiplicity, name, position) -> str:
    """One term of a sum as the site irrep at its Wyckoff position: ``2×-E2 at 1b (4mm)``."""
    irrep = str(name).split("↑")[0]
    position = str(position)
    letter = position.split("(")[0]
    if "(" in position:
        site = position[position.index("(") + 1:position.rindex(")")].split(",")[-1]
        letter = f"{letter} ({site})"
    count = f"{int(multiplicity)}×" if int(multiplicity) != 1 else ""
    return f"{count}{irrep} at {letter}"


def _sum_words(terms) -> str:
    """A sum of elementary band representations in words: ``-E1g at 3a (-3m) + -E1u at 3a (-3m)``."""
    return " + ".join(_term_words(*term) for term in terms)


def _sum_lines(sums) -> list:
    """Each sum of elementary band representations on its own line, the first few."""
    lines = ["    " + _sum_words(terms) for terms in sums[:_SHOWN]]
    if len(sums) > _SHOWN:
        lines.append(f"    and {len(sums) - _SHOWN} more")
    return lines


def _ebrs_of(terms) -> str:
    """``"the elementary band representations of -E1g at 3a (-3m) + -E1u at 3a (-3m)"``."""
    one = sum(int(m) for m, _name, _position in terms) == 1
    return f"the elementary band representation{'' if one else 's'} of {_sum_words(terms)}"


def _content_terms(irrep) -> list:
    """A site content as a row spells it, ``"2x-E1 + -1E2E"``, as ``[(2, "-E1"), (1, "-1E2E")]``."""
    out = []
    for term in str(irrep).split(" + "):
        count, times, name = term.partition("x")
        out.append((int(count), name) if times and count.isdigit() else (1, term))
    return out


def _label_terms(label):
    """The irreps one label spells, each with its multiplicity, in the order written.

    ``"2x-E1 + -1E2E"`` is ``[("-E1", 2), ("-1E2E", 1)]``, and a single irrep is
    one term of multiplicity one.
    """
    out = []
    for part in str(label).split(" + "):
        part = part.strip()
        if not part:
            continue
        count, mark, name = part.partition("x")
        if mark and count.isdigit() and name:
            out.append((name, int(count)))
        else:
            out.append((part, 1))
    return out


# ==========================================================================
# the words a summary prints
# ==========================================================================

def _runs(indices) -> str:
    """Sorted indices as runs, ``0-3, 5, 7-8``."""
    runs, start, prev = [], None, None
    for i in sorted(int(i) for i in indices):
        if start is not None and i == prev + 1:
            prev = i
            continue
        if start is not None:
            runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    if start is not None:
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(runs)


def _listed(items) -> str:
    """Names as a sentence lists them: ``A``, ``A and B``, ``A, B and C``."""
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


#: The largest denominator a reduced coordinate is written as a fraction with:
#: a tabulated point's coordinates are halves, thirds, quarters, sixths and
#: eighths.
_DENOMINATOR = 24


def _k_text(k) -> str:
    """A k-point in reduced coordinates, ``(1/3, 2/3)``, a decimal where it is no fraction."""
    out = []
    for x in np.asarray(k, dtype=float).reshape(-1) + 0.0:
        value = Fraction(float(x)).limit_denominator(_DENOMINATOR)
        out.append(f"{float(x):.4g}" if abs(float(value) - x) > _LATTICE_TOL
                   else str(value.numerator) if value.denominator == 1
                   else f"{value.numerator}/{value.denominator}")
    return "(" + ", ".join(out) + ")"


def _character_text(value) -> str:
    """A character as a table prints it: ``2``, ``-1.414``, ``0.5+0.866i``, ``-i``."""
    z = complex(value)

    def part(x):
        return str(int(round(x))) if abs(x - round(x)) <= CHARACTER else f"{x:.4g}"

    if abs(z.imag) <= CHARACTER:
        return part(z.real + 0.0)
    imaginary = {"1": "i", "-1": "-i"}.get(part(z.imag), part(z.imag) + "i")
    if abs(z.real) <= CHARACTER:
        return imaginary
    return part(z.real) + ("" if imaginary.startswith("-") else "+") + imaginary


def _operation_name(rep, ind) -> str:
    """One operation's name in the model's frame, as the operation table writes it."""
    op = rep._operation(int(ind))
    return (_sitechars.operation_name(op, np.asarray(rep.group.lattice, dtype=float))
            + ("'" if op.time_reversal else ""))


# --------------------------------------------------------------------------
# the words a summary prints
# --------------------------------------------------------------------------

def _moduli(divisors) -> str:
    """A class's coordinates in words: ``an integer mod 6 and one mod 12``."""
    words = [f"an integer mod {divisors[0]}"] + [f"one mod {d}" for d in divisors[1:]]
    return _listed(words)


def _times(n) -> str:
    """How many times, in words: ``once``, ``twice``, ``3 times``."""
    return {1: "once", 2: "twice"}.get(int(n), f"{int(n)} times")


# --------------------------------------------------------------------------
# what the records' summaries print
# --------------------------------------------------------------------------

#: The unit a count of operations is in, named wherever the count is what is
#: read: the operations listed are one per coset of the lattice translations.
_PER_COSET = "coset representatives of the lattice translations"

#: What a spinful group's count adds, once in a printout: the operations listed
#: are the single-valued group's, and the double group holds each one's negative.
_DOUBLE_GROUP = ("the single-valued group's, each standing for itself and its negative "
                 "in the double group")

def _coordinate_text(x) -> str:
    """One reduced coordinate as the fraction it is, or as a decimal where it is none."""
    x = float(x)
    value = Fraction(x).limit_denominator(_SEITZ_DENOMINATOR)
    if abs(float(value) - x) > _LATTICE_TOL:
        return f"{x:.4g}"
    return (f"{value.numerator}" if value.denominator == 1
            else f"{value.numerator}/{value.denominator}")


def _point_text(v) -> str:
    """A point in reduced coordinates, ``(1/3, 2/3)``."""
    return "(" + ", ".join(_coordinate_text(x)
                           for x in np.asarray(v, dtype=float).reshape(-1)) + ")"


def _affine_text(jacobian, offset) -> str:
    """A Wyckoff position's map ``A·(x, y, z) + b`` as its three coordinates, ``(x, -x, 1/2)``."""
    out = []
    for row, b in zip(np.asarray(jacobian, dtype=float), np.asarray(offset, dtype=float)):
        terms = []
        for coefficient, symbol in zip(row, "xyz"):
            c = Fraction(float(coefficient)).limit_denominator(_SEITZ_DENOMINATOR)
            if c == 0:
                continue
            size = abs(c)
            body = (symbol if size == 1 else f"{size.numerator}{symbol}"
                    if size.denominator == 1 else
                    f"{'' if size.numerator == 1 else size.numerator}{symbol}/{size.denominator}")
            terms.append(("-" if c < 0 else "+", body))
        constant = Fraction(float(b)).limit_denominator(_SEITZ_DENOMINATOR)
        if constant != 0 or not terms:
            terms.append(("-" if constant < 0 else "+", _coordinate_text(abs(constant))))
        text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sign, body in terms[1:]:
            text += f" {sign} {body}"
        out.append(text)
    return "(" + ", ".join(out) + ")"


def _lattice_vectors(dirs) -> str:
    """Lattice-vector indices in words: ``the first two``, ``the third``, ``the first and third``."""
    words = ("first", "second", "third")
    dirs = sorted(int(d) for d in dirs)
    if dirs == [0, 1]:
        return "the first two"
    if len(dirs) == 3:
        return "all three"
    return "the " + " and ".join(words[d] for d in dirs)


def _names_text(names, shown=6) -> str:
    """Names in a sentence, the first ``shown`` of them and how many more."""
    names = [str(n) for n in names]
    return (", ".join(names[:shown])
            + (f" and {len(names) - shown} more" if len(names) > shown else ""))


#: Largest denominator a conventional-setting translation is written with.  The
#: tables carry halves, thirds, quarters, sixths, eighths and twelfths, and a
#: translation that snaps to nothing at this denominator within
#: :data:`_LATTICE_TOL` is arithmetic failing rather than a fraction.
_SEITZ_DENOMINATOR = 24


def _joined(parts) -> str:
    """Clauses joined as a sentence lists them: ``a``, ``a and b``, ``a, b and c``."""
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
