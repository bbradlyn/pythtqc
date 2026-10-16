"""What a call returns in place of a value it cannot measure, and what the package raises.

A :class:`Refusal` is a verdict about the physics: the check that stopped a
call, and what it measured there.  It is returned, never raised, and it is
falsy; some values are falsy too, an empty set of pairs or an empty content,
so ``isinstance(out, Refusal)`` is the test::

    r = tqc.realized_group(model)
    if isinstance(r, tqc.Refusal):
        print(r)          # the check, the diagnosis, the number measured

:meth:`Refusal.raise_` raises one as a :class:`RefusalError` where a script
wants an exception.  The package raises ``ValueError`` for a malformed
argument, before anything is measured, and :class:`InternalError` where its
own measurements disagree, a defect to report; it warns
:class:`MagneticScopeWarning` where time reversal is measured broken.
``docs/guide/refusals.md`` reads a refusal end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Refusal", "RefusalError", "InternalError", "MagneticScopeWarning"]


class RefusalError(RuntimeError):
    """A :class:`Refusal` raised as an exception, by :meth:`Refusal.raise_`; the refusal is on ``.refusal``."""

    def __init__(self, refusal: "Refusal"):
        super().__init__(str(refusal))
        self.refusal = refusal


class InternalError(RuntimeError):
    """Raised where the package's own measurements, or its reading of the tables, disagree: a defect to report.

    The message says what disagreed.  Unusual physics is refused rather than
    raised; this has nothing in the model to change.
    """


class MagneticScopeWarning(UserWarning):
    """Time reversal is measured broken: the group returned is the unitary subgroup of the Hamiltonian's magnetic group.

    Issued by :func:`..groups.realized_group`.  The labels, rows of elementary
    band representations and indicators that follow are that group's; the
    operations of the cell combined with time reversal that hold are on
    :attr:`..groups.RealizedGroup.beyond_grey`, and none beyond the cell's own are
    looked for.  ``warnings.simplefilter("ignore", pythtqc.MagneticScopeWarning)``
    silences it.
    """


@dataclass(frozen=True)
class Refusal:
    """What a call returns where it cannot complete: the check that stopped it, and what it measured there.

    ``print(r)`` reads it: the check, the diagnosis, the number measured against
    the one demanded, and the keys of its context.  It is falsy, as some values
    are, so ``isinstance(out, Refusal)`` is the test; it is frozen.

    Attributes
    ----------
    gate : str
        The name of the check, ``"wyckoff_orbit"``: stable, and listed under
        *Refusals* by every call that can return it.
    reason : str
        The diagnosis, naming the orbital, operation, point or group concerned.
    value, threshold : float or None
        The number measured and the number the check demanded; ``None`` for a
        check with no number.
    context : dict
        What acting on it needs without measuring again: indices, labels,
        multiplicities, and, once the group is identified, the group.  Each call
        names the keys it puts here.

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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"Refusal(gate={self.gate!r}, value={self.value!r}, "
                f"threshold={self.threshold!r}, reason={self.reason!r})")

    def raise_(self):
        """This refusal, raised as a :class:`RefusalError`."""
        raise RefusalError(self)
