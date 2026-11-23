"""Refusals, the errors the package raises, and the warning it issues.

* :class:`Refusal`: what a call returns in place of its record where it
  cannot complete;
* :class:`RefusalError`: a refusal raised as an exception, by
  :meth:`Refusal.raise_`;
* :class:`InternalError`: the error raised where the package disagrees with
  itself;
* :class:`MagneticScopeWarning`: the warning issued where time reversal is
  measured broken.

A malformed argument raises ``ValueError`` before anything is measured.
``docs/guide/refusals.md`` shows how to read a refusal and what to do about
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Refusal", "RefusalError", "InternalError", "MagneticScopeWarning"]


class RefusalError(RuntimeError):
    """A :class:`Refusal` raised as an exception, by :meth:`Refusal.raise_`.

    A ``RuntimeError`` whose message is the printed refusal.

    Attributes
    ----------
    refusal : Refusal
        The refusal raised.
    """

    def __init__(self, refusal: "Refusal"):
        super().__init__(str(refusal))
        self.refusal = refusal


class InternalError(RuntimeError):
    """The error raised where the package disagrees with itself.

    Two measurements of one thing disagree, the tables the package reads
    disagree with each other or with a measurement, or a step finds broken
    what an earlier step guarantees.  The message says what disagreed.  It is
    a defect in the package, not in the model.  Report it on the package's
    repository, with the model and the call that raised it.
    """


class MagneticScopeWarning(UserWarning):
    """The warning issued where time reversal is measured broken.

    :func:`..groups.realized_group` issues it.  The group returned is then the
    unitary subgroup of the Hamiltonian's magnetic group.  The labels, the
    elementary band representations and the symmetry indicators that later
    calls return are that subgroup's.  The operations of the group of the
    cell combined with time reversal that the Hamiltonian obeys are listed on
    :attr:`..groups.RealizedGroup.beyond_grey`.  No operation combined with
    time reversal is searched for outside the group of the cell.
    ``warnings.simplefilter("ignore", pythtqc.MagneticScopeWarning)`` silences
    it.
    """


@dataclass(frozen=True)
class Refusal:
    """What a call returns in place of its record where it cannot complete.

    A refusal names the check that stopped the call and what the check
    measured there.  It is returned, never raised.  Printed, a refusal shows
    the check, the reason, the number measured and its threshold, and the
    keys of ``context``.  A refusal is falsy, and so is the empty symmetry
    data vector that :func:`..ebr.subtract` returns for two equal vectors.
    ``isinstance(out, Refusal)`` tells a refusal from a returned value.

    Attributes
    ----------
    gate : str
        The name of the check that stopped the call, such as
        ``"multiplet_split"``.  The name does not change between versions.
        Each call lists the checks it can stop at under *Refusals*, and the
        documentation's page Refusals has an entry under each name.
    reason : str
        What the check found, in words, naming the orbital, operation, point
        or group concerned.
    value : float or None
        If not ``None``, the number the check measured.
    threshold : float or None
        If not ``None``, the number the check compared ``value`` against.
    context : dict
        If nonempty, the data needed to act on the refusal without running
        the call again, such as indices, labels and multiplicities.  A
        refusal that carries the group holds it as ``context["group"]``.  One
        that carries only its number holds it as ``context["number_str"]``,
        some with its symbol as ``context["name"]``.

    Examples
    --------
    >>> r = Refusal("wyckoff_orbit", "orbital 3 lies 0.031 off every orbit",
    ...             value=0.031, threshold=1e-5, context={"orbital": 3})
    >>> isinstance(r, Refusal), bool(r)
    (True, False)
    >>> r.gate
    'wyckoff_orbit'
    """

    gate: str
    reason: str
    value: float | None = None
    threshold: float | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        head = f"Refusal[{self.gate}]: {self.reason}"
        if self.value is not None and self.threshold is not None:
            head += f"  (measured {self.value:.6g}, threshold {self.threshold:.6g})"
        elif self.value is not None:
            head += f"  (measured {self.value:.6g})"
        if self.context:
            keys = ", ".join(sorted(self.context))
            head += f"\n    context: {keys}"
        return head

    def __repr__(self) -> str:  # pragma: no cover: convenience only
        return (f"Refusal(gate={self.gate!r}, value={self.value!r}, "
                f"threshold={self.threshold!r}, reason={self.reason!r})")

    def raise_(self):
        """This refusal, raised as a :class:`RefusalError`."""
        raise RefusalError(self)
