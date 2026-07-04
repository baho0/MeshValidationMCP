"""Units / scale sanity and sampling self-consistency.

Two cheap guards that catch gross mistakes and quantify sampling trust:
- **units / scale sanity**: does the file declare the expected units, and does the overall
  size fall in a plausible range? (A part that should be ~100mm but is 0.1 units wide is
  almost certainly a unit mix-up.)
- **self-consistency**: recompute a *sampled* quantity at a second, independent seed and
  confirm the two agree within the sampling bound — evidence the sampled metrics are stable,
  and (at the same seed) that the pipeline is deterministic.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, exact, sampled
from .config import SEED
from .loading import LoadedMesh


class UnitsReport(BaseModel):
    declared_units: str | None
    expected_units: str | None
    units_match: bool | None
    bbox_diagonal: float
    plausible: bool | None  # within the caller's expected size range, if given
    # sampling self-consistency: a sampled probe recomputed at two seeds
    probe_seed_a: float
    probe_seed_b: float
    probe_agreement: bool
    deterministic: bool  # same seed -> identical sample (the SEED guarantee)
    confidence: Confidence
    caveats: list[str]


def _sampled_probe(mesh: trimesh.Trimesh, n: int, seed: int) -> float:
    """A genuinely sampled statistic used only to test stability: the mean distance from the
    mesh centroid to n surface samples. Two independent seeds must agree within the sampling
    noise; the same seed must reproduce exactly (determinism)."""
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return float(np.linalg.norm(pts - np.asarray(mesh.centroid), axis=1).mean())


def units_report(
    loaded: LoadedMesh,
    expected_units: str | None = None,
    plausible_diagonal_range: tuple[float, float] | None = None,
    samples: int = 2000,
) -> UnitsReport:
    mesh = loaded.combined
    declared = str(mesh.units) if mesh.units else None
    diagonal = float(np.linalg.norm(mesh.extents))
    caveats: list[str] = []

    units_match: bool | None = None
    if expected_units is not None:
        units_match = declared == expected_units
        if declared is None:
            caveats.append("the file declares no units; cannot confirm the expected units")

    plausible: bool | None = None
    if plausible_diagonal_range is not None:
        lo, hi = plausible_diagonal_range
        plausible = lo <= diagonal <= hi
        if not plausible:
            caveats.append(
                f"bbox diagonal {diagonal:.4g} is outside the plausible range "
                f"[{lo:.4g}, {hi:.4g}] — possible unit/scale mistake"
            )

    probe_a = _sampled_probe(mesh, samples, SEED)
    probe_b = _sampled_probe(mesh, samples, SEED + 1)
    ref = max(abs(probe_a), abs(probe_b), 1e-9)
    agreement = abs(probe_a - probe_b) <= 0.05 * ref
    deterministic = _sampled_probe(mesh, samples, SEED) == probe_a

    return UnitsReport(
        declared_units=declared,
        expected_units=expected_units,
        units_match=units_match,
        bbox_diagonal=diagonal,
        plausible=plausible,
        probe_seed_a=probe_a,
        probe_seed_b=probe_b,
        probe_agreement=agreement,
        deterministic=deterministic,
        confidence=exact("declared units + bbox diagonal") if expected_units else sampled(
            "sampling self-consistency", abs(probe_a - probe_b)
        ),
        caveats=caveats,
    )
