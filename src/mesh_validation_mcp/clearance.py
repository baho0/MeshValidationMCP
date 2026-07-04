"""Clearance / interference between two solids (assembly fit).

Do two parts collide, and if not, how close do they come? We answer this without a physics
collision library: a point of A that lies strictly inside watertight B is an interference
(signed distance > 0 inside), and the closest approach between the two surfaces is the minimum
sampled surface-to-surface distance. Both are sampled, so the result carries a spacing bound.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, sampled
from .config import SEED
from .loading import LoadedMesh


class ClearanceInfo(BaseModel):
    interfering: bool
    max_penetration: float  # deepest overlap (0 when clear)
    min_clearance: float  # closest surface-to-surface approach (0 when interfering)
    min_clearance_required: float | None
    meets_requirement: bool | None
    confidence: Confidence
    caveats: list[str]


def _samples(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=SEED)
    return pts


def check_clearance(
    loaded_a: LoadedMesh,
    loaded_b: LoadedMesh,
    min_clearance_required: float | None = None,
    samples: int = 2000,
) -> ClearanceInfo:
    a, b = loaded_a.combined, loaded_b.combined
    diagonal = float(np.linalg.norm(np.concatenate([a.extents, b.extents]))) or 1.0
    spacing = diagonal / np.sqrt(samples)
    caveats: list[str] = []

    pa, pb = _samples(a, samples), _samples(b, samples)
    # Surface-to-surface distance (closest approach), both directions.
    d_ab = trimesh.proximity.closest_point(b, pa)[1]
    d_ba = trimesh.proximity.closest_point(a, pb)[1]
    min_clearance = float(min(d_ab.min(), d_ba.min()))

    penetration = 0.0
    if a.is_watertight and b.is_watertight:
        # A point of A inside B (or vice versa) is an interference; depth = signed distance in.
        sdb = np.asarray(trimesh.proximity.signed_distance(b, pa))  # + inside B
        sda = np.asarray(trimesh.proximity.signed_distance(a, pb))
        penetration = float(max(sdb.max(), sda.max(), 0.0))
        interfering = penetration > 1e-6 * diagonal
    else:
        interfering = min_clearance <= 1e-6 * diagonal
        caveats.append(
            "an operand is not watertight: interference is inferred from surface contact only, "
            "penetration depth is unavailable"
        )

    if interfering:
        min_clearance = 0.0

    meets = None
    if min_clearance_required is not None:
        meets = (not interfering) and min_clearance >= min_clearance_required

    return ClearanceInfo(
        interfering=interfering,
        max_penetration=penetration,
        min_clearance=min_clearance,
        min_clearance_required=min_clearance_required,
        meets_requirement=meets,
        confidence=sampled(f"surface sampling n={samples}", spacing),
        caveats=caveats,
    )
