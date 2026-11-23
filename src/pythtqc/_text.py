"""The words the records' printouts share.

A reduced coordinate, a point, a k-point and a Wyckoff position's
coordinates; a character; a count of operations; a set of orbitals; a sum
of elementary band representations; names listed in a sentence.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ._tolerance import CHARACTER, _LATTICE_TOL


def _fmt_char(z) -> str:
    # A character that vanishes prints as the unsigned 0 a table carries, and
    # not as the rounding it was measured to. Only the printed text is rounded.
    z = complex(z)
    re_, im_ = (0.0 if abs(z.real) < 1e-9 else z.real), (0.0 if abs(z.imag) < 1e-9 else z.imag)
    if im_ == 0.0:
        return "0" if re_ == 0.0 else f"{re_:+.3g}"
    return f"{re_:+.3g}{im_:+.3g}j"


#: The unit a count of operations is in: one operation per coset of the lattice
#: translations.  With spin, the operations are the single-valued group's, each
#: standing for itself and its negative in the double group, as
#: :data:`_SINGLE_VALUED` says.  The unit is said once in a printout, wherever
#: the count is the information; where the group's symbol fixes the count, no
#: count is printed.
_COSETS = "one per coset of the lattice translations"

_SINGLE_VALUED = ("the single-valued group's, each standing for itself and its negative "
                  "in the double group")


def _counted_as(spinful) -> str:
    """The unit a count of operations is in, as a printout says it once.

    ``spinful`` is true for a double group, and the words then add that each
    operation listed stands for itself and its negative in it.
    """
    return _COSETS + (f", {_SINGLE_VALUED}" if spinful else "")


def _index_ranges(indices) -> str:
    """The integers ``indices``, sorted and written as runs, ``0-3, 5, 7-8``.

    ``indices`` may come in any order; none gives ``""``.
    """
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
    """A set of orbitals in words, ``"orbital 3"`` or ``"orbitals 0-2"``.

    ``orbitals`` holds the orbitals' indices, in any order, written as runs
    by :func:`_index_ranges`.
    """
    orbitals = tuple(int(o) for o in orbitals)
    return f"orbital{'s' if len(orbitals) > 1 else ''} {_index_ranges(orbitals)}"


#: How many sums of elementary band representations a summary prints before it
#: says how many more there are.
_SHOWN = 6


def _term_words(multiplicity, name, position) -> str:
    """One term of a sum of elementary band representations in words.

    The term is ``multiplicity`` copies of the band representation ``name``
    induced from ``position``, both written as the tables list them,
    ``"-E1g↑G(2)"`` and ``"3a(-3m1',-3m)"``.  The words give the
    multiplicity, the site irrep (the part of ``name`` before ``↑``), the
    Wyckoff position and its site-symmetry group without time reversal (the
    last group ``position`` lists): ``2x-E2 at 1b (4mm)``, or
    ``-E1g at 3a (-3m)`` where the multiplicity is one.
    """
    irrep = str(name).split("↑")[0]
    position = str(position)
    letter = position.split("(")[0]
    if "(" in position:
        site = position[position.index("(") + 1:position.rindex(")")].split(",")[-1]
        letter = f"{letter} ({site})"
    count = f"{int(multiplicity)}x" if int(multiplicity) != 1 else ""
    return f"{count}{irrep} at {letter}"


def _sum_words(terms) -> str:
    """A sum of elementary band representations in words.

    ``terms`` holds ``(multiplicity, name, position)`` for each term, as
    :func:`_term_words` reads one.  Returns
    ``-E1g at 3a (-3m) + -E1u at 3a (-3m)``.
    """
    return " + ".join(_term_words(*term) for term in terms)


def _sum_lines(sums) -> list:
    """The first few sums of elementary band representations, a line each.

    ``sums`` holds sums written as :func:`_sum_words` reads them.  Returns a
    list of lines indented by four spaces: the first :data:`_SHOWN` sums, then
    a line saying how many more there are, where there are more.
    """
    lines = ["    " + _sum_words(terms) for terms in sums[:_SHOWN]]
    if len(sums) > _SHOWN:
        lines.append(f"    and {len(sums) - _SHOWN} more")
    return lines


def _ebrs_of(terms) -> str:
    """A sum named by its elementary band representations, in words.

    ``terms`` is the sum, written as :func:`_sum_words` reads it.  Returns
    ``"the elementary band representation of -E1g at 3a (-3m)"``, in the plural
    where the multiplicities add to more than one.
    """
    one = sum(int(m) for m, _name, _position in terms) == 1
    return f"the elementary band representation{'' if one else 's'} of {_sum_words(terms)}"


def _content_terms(irrep) -> list:
    """The site irreps a sum names, each with its multiplicity.

    ``irrep`` is a site irrep, or a sum of them, as a printout spells it,
    ``"2x-E1 + -1E2E"``.  Returns ``[(2, "-E1"), (1, "-1E2E")]``, in the
    order written.  For a sum of named terms, :func:`_label_terms` reads the
    same text, each pair the other way round.  ``""`` gives ``[(1, "")]``
    here and no term there.
    """
    out = []
    for term in str(irrep).split(" + "):
        count, times, name = term.partition("x")
        out.append((int(count), name) if times and count.isdigit() else (1, term))
    return out


def _label_terms(label):
    """The site irreps one label spells, each with its multiplicity.

    ``label`` is a site irrep, or a sum of them, ``"2x-E1 + -1E2E"``.  Returns
    ``[("-E1", 2), ("-1E2E", 1)]``, in the order written.  A single site irrep
    is one term of multiplicity one, and ``""`` gives no term.
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
    """The integers ``indices``, sorted and written as runs, ``0-3, 5, 7-8``.

    The same text as :func:`_index_ranges` writes.
    """
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
    """Names as a sentence lists them: ``A``, ``A and B``, ``A, B and C``.

    ``items`` are written with ``str``; none gives ``""``.
    """
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


#: The largest denominator a k-point's reduced coordinate is written with as a
#: fraction.  The tables' k-points are halves and thirds in their conventional
#: cell, and a centered lattice's primitive cell brings quarters; 24 holds these
#: with room to spare.
_DENOMINATOR = 24


def _k_text(k) -> str:
    """A k-point in reduced coordinates, ``(1/3, 2/3)``.

    ``k`` holds its components.  Each is written as a fraction of denominator
    at most :data:`_DENOMINATOR` where it is one within :data:`_LATTICE_TOL`,
    and to four significant figures otherwise.
    """
    out = []
    for x in np.asarray(k, dtype=float).reshape(-1) + 0.0:
        value = Fraction(float(x)).limit_denominator(_DENOMINATOR)
        out.append(f"{float(x):.4g}" if abs(float(value) - x) > _LATTICE_TOL
                   else str(value.numerator) if value.denominator == 1
                   else f"{value.numerator}/{value.denominator}")
    return "(" + ", ".join(out) + ")"


def _character_text(value) -> str:
    """A character as a character table prints it.

    ``value`` is a complex number.  It is written ``2``, ``-1.414``,
    ``0.5+0.866i`` or ``-i``.  A real or imaginary part within
    :data:`.._tolerance.CHARACTER` of an integer is written as that integer,
    and any other to four significant figures.  A part within ``CHARACTER``
    of zero is left out.  A value whose real and imaginary parts are both
    within ``CHARACTER`` of zero prints ``0``.
    """
    z = complex(value)

    def part(x):
        return str(int(round(x))) if abs(x - round(x)) <= CHARACTER else f"{x:.4g}"

    if abs(z.imag) <= CHARACTER:
        return part(z.real + 0.0)
    imaginary = {"1": "i", "-1": "-i"}.get(part(z.imag), part(z.imag) + "i")
    if abs(z.real) <= CHARACTER:
        return imaginary
    return part(z.real) + ("" if imaginary.startswith("-") else "+") + imaginary


# --------------------------------------------------------------------------
# the words a summary prints
# --------------------------------------------------------------------------

def _moduli(divisors) -> str:
    """What each coordinate of a symmetry-indicator class is, in words.

    ``divisors`` holds the orders of the indicator group's cyclic factors,
    at least one.  Returns ``an integer mod 6 and one mod 12``.
    """
    words = [f"an integer mod {divisors[0]}"] + [f"one mod {d}" for d in divisors[1:]]
    return _listed(words)


def _times(n) -> str:
    """The whole number ``n`` as a number of times: ``once``, ``twice``, ``3 times``."""
    return {1: "once", 2: "twice"}.get(int(n), f"{int(n)} times")


# --------------------------------------------------------------------------
# what the records' summaries print
# --------------------------------------------------------------------------

#: The unit of :data:`_COSETS`, in the words the records' summaries print.
_PER_COSET = "coset representatives of the lattice translations"

#: The words of :data:`_SINGLE_VALUED`, as the records' summaries print them.
_DOUBLE_GROUP = ("the single-valued group's, each standing for itself and its negative "
                 "in the double group")

def _coordinate_text(x) -> str:
    """One reduced coordinate ``x``, as the fraction it is or as a decimal.

    The fraction has denominator at most :data:`_SEITZ_DENOMINATOR` and is
    within :data:`_LATTICE_TOL` of ``x``; the decimal has four significant
    figures.
    """
    x = float(x)
    value = Fraction(x).limit_denominator(_SEITZ_DENOMINATOR)
    if abs(float(value) - x) > _LATTICE_TOL:
        return f"{x:.4g}"
    return (f"{value.numerator}" if value.denominator == 1
            else f"{value.numerator}/{value.denominator}")


def _point_text(v) -> str:
    """A point in reduced coordinates, ``(1/3, 2/3)``.

    ``v`` holds its components, each written by :func:`_coordinate_text`.
    """
    return "(" + ", ".join(_coordinate_text(x)
                           for x in np.asarray(v, dtype=float).reshape(-1)) + ")"


def _affine_text(jacobian, offset) -> str:
    """A Wyckoff position's coordinates as its map writes them, ``(x, -x, 1/2)``.

    The map is ``A·(x, y, z) + b`` in reduced coordinates, ``jacobian`` its
    matrix ``A`` and ``offset`` its vector ``b``.  Each row of ``A`` with its
    entry of ``b`` gives one coordinate, its coefficients and constant read as
    fractions of denominator at most :data:`_SEITZ_DENOMINATOR`.
    """
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
    """Lattice vectors named in words: ``the first two``, ``the third``, ``all three``.

    ``dirs`` holds the vectors' indices, 0, 1 or 2, in any order.
    """
    words = ("first", "second", "third")
    dirs = sorted(int(d) for d in dirs)
    if dirs == [0, 1]:
        return "the first two"
    if len(dirs) == 3:
        return "all three"
    return "the " + " and ".join(words[d] for d in dirs)


def _names_text(names, shown=6) -> str:
    """Names in a sentence, the first ``shown`` of them and how many more.

    ``names`` are written with ``str`` and joined by commas,
    ``a, b, c, d, e, f and 2 more``.
    """
    names = [str(n) for n in names]
    return (", ".join(names[:shown])
            + (f" and {len(names) - shown} more" if len(names) > shown else ""))


#: Largest denominator a conventional-setting translation is written with, and a
#: point's reduced coordinate as :func:`_coordinate_text` writes it.  The
#: translations in the irreptables package's tables are halves, thirds, quarters
#: and sixths, and a translation that snaps to nothing at this denominator within
#: :data:`_LATTICE_TOL` is arithmetic failing rather than a fraction.  A point's
#: coordinate that snaps to nothing is written as a decimal.
_SEITZ_DENOMINATOR = 24


def _joined(parts) -> str:
    """Clauses joined as a sentence lists them: ``a``, ``a and b``, ``a, b and c``.

    ``parts`` holds at least one string.  :func:`_listed` writes the same for
    any items, none included.
    """
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
