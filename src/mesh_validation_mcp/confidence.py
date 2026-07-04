"""Confidence tiers and numeric error bounds attached to every reported quantity.

The point of this module is trust: a "PASS" is only useful if the agent knows *how*
the underlying number was obtained and how much slack sits around it. Every check or
distance carries a :class:`Confidence` describing

- ``tier`` — how the value was derived, from most to least trustworthy:
    * ``exact``        — combinatorial/analytic: euler number, watertight flag, a
                         per-vertex difference on matching topology, the divergence
                         volume of a watertight+consistent mesh. Error is float noise.
    * ``topological``  — an integer invariant (genus, body count, boundary-loop count).
                         Exact when its preconditions hold, but a *discrete* quantity.
    * ``sampled``      — Monte-Carlo surface sampling (chamfer, Hausdorff, curvature,
                         wall thickness). The true value is bracketed; ``error_abs`` is
                         the sampling/ spacing bound, never zero.
    * ``estimated``    — an optimizer's residual output (procrustes/ICP/primitive fit).
                         ``error_abs`` is the fit residual; the value is a best guess.

- ``error_abs`` / ``error_rel`` — absolute and relative uncertainty (relative is against
  a caller-supplied reference magnitude when meaningful).
- ``basis`` — a short human string naming the mechanism ("exact vertex correspondence",
  "ICP registration residual", "surface sampling n=2000", ...).

Nothing here computes geometry; it only labels quantities produced elsewhere so the
labels can ride along in the JSON the server emits.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["exact", "topological", "sampled", "estimated"]

# Float32 export round-trip (STL/PLY) contributes ~1e-7 relative noise; treat anything
# at or below this (relative to the reference magnitude) as exact-to-machine.
EXACT_REL_NOISE = 1e-6


class Confidence(BaseModel):
    """How a reported quantity was derived and how much uncertainty rides on it."""

    tier: Tier
    error_abs: float | None = Field(
        default=None, description="Absolute uncertainty in native units (None = unquantified)."
    )
    error_rel: float | None = Field(
        default=None, description="Relative uncertainty vs a reference magnitude, if meaningful."
    )
    basis: str = Field(description="Short name of the mechanism that produced the value.")

    def with_reference(self, magnitude: float) -> "Confidence":
        """Fill ``error_rel`` from ``error_abs`` against a reference magnitude."""
        if self.error_abs is not None and magnitude not in (0.0, None):
            self.error_rel = self.error_abs / abs(magnitude)
        return self


def exact(basis: str, error_abs: float = 0.0) -> Confidence:
    """A combinatorial or analytic quantity; error defaults to float noise (~0)."""
    return Confidence(tier="exact", error_abs=error_abs, basis=basis)


def topological(basis: str) -> Confidence:
    """A discrete integer invariant; exact when its preconditions hold."""
    return Confidence(tier="topological", error_abs=0.0, basis=basis)


def sampled(basis: str, error_abs: float) -> Confidence:
    """A sampled quantity bracketed by ``error_abs`` (the spacing/sampling bound)."""
    return Confidence(tier="sampled", error_abs=float(error_abs), basis=basis)


def estimated(basis: str, residual: float) -> Confidence:
    """An optimizer output whose ``error_abs`` is its fit residual."""
    return Confidence(tier="estimated", error_abs=float(residual), basis=basis)


def sampling_error(n: int, diagonal: float) -> float:
    """Characteristic nearest-neighbour spacing for ``n`` samples spread over a shape of
    bbox diagonal ``diagonal``: ``diagonal / sqrt(n)``.

    This is the resolution floor of any sampled max/Hausdorff estimate — the true extremum
    may sit up to about one sample spacing away from the nearest sample we happened to draw.
    """
    if n <= 0:
        return float("inf")
    return float(diagonal) / math.sqrt(float(n))


def spacing_bound(distances) -> float:
    """Upper bound on the gap left by a finite point set: the largest nearest-neighbour
    spacing among the samples (how far an unsampled point could hide from all of them).

    ``distances`` is a 1-D array of per-sample nearest-neighbour distances.
    """
    import numpy as np

    arr = np.asarray(distances, dtype=float)
    if arr.size == 0:
        return float("inf")
    return float(arr.max())
