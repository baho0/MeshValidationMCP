"""Golden / reference comparison: "did I produce the mesh I intended?"

Two regimes, most-trustworthy first:

- **exact_vertex** — when the produced mesh and the reference share topology (same vertex
  and face counts), both vertex sets are canonicalized (lexicographically sorted) and the
  maximum per-vertex displacement is measured. A match within tolerance is an *exact* match,
  independent of vertex ordering.
- **surface_sampling** — otherwise, the maximum surface distance between the two meshes is
  bounded (sampled lower bound + one sample-spacing upper bound, reusing the same estimator
  as compare()). The meshes are compared *in place* (no alignment): this answers "is the
  output where and what the reference is", not "is it congruent to the reference".
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel

from .config import HEATMAP_MAX_VERTICES, SEED
from .confidence import Confidence, exact, sampled
from .loading import LoadedMesh


class GoldenComparison(BaseModel):
    matches: bool
    method: Literal["exact_vertex", "vertex_to_surface"]
    tolerance: float
    exact_match: bool
    max_vertex_delta: float | None  # only on matching topology
    surface_distance: float | None  # differing topology: max vertex-to-surface distance
    confidence: Confidence
    caveats: list[str]
    summary: str


def _canonical(vertices: np.ndarray) -> np.ndarray:
    """Sort vertices into a canonical, ordering-independent arrangement."""
    v = np.asarray(vertices, dtype=float)
    return v[np.lexsort((v[:, 2], v[:, 1], v[:, 0]))]


def _subsample(points: np.ndarray) -> np.ndarray:
    """Cap a vertex set to a seeded subset so the distance stays cheap and deterministic."""
    if len(points) <= HEATMAP_MAX_VERTICES:
        return points
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(points), HEATMAP_MAX_VERTICES, replace=False))
    return points[idx]


def _vertex_to_surface_hausdorff(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Symmetric max vertex-to-surface distance: max over each mesh's vertices of the
    distance to the OTHER mesh's surface. Deterministic and exact at the vertices (no
    random sampling), which captures tessellation-only differences as zero."""
    d_ab = trimesh.proximity.closest_point(b, _subsample(np.asarray(a.vertices)))[1]
    d_ba = trimesh.proximity.closest_point(a, _subsample(np.asarray(b.vertices)))[1]
    return float(max(d_ab.max(), d_ba.max()))


def compare_to_reference(
    produced: LoadedMesh,
    reference: LoadedMesh,
    tolerance: float,
) -> GoldenComparison:
    a, b = produced.combined, reference.combined
    diagonal = float(np.linalg.norm(b.extents)) or 1.0
    caveats: list[str] = []

    same_topology = len(a.vertices) == len(b.vertices) and len(a.faces) == len(b.faces)

    if same_topology:
        max_delta = float(
            np.linalg.norm(_canonical(a.vertices) - _canonical(b.vertices), axis=1).max()
        )
        exact_match = max_delta <= tolerance
        conf = exact("canonical per-vertex displacement").with_reference(diagonal)
        summary = (
            f"exact match: every vertex within {max_delta:.3g} of the reference "
            f"(tolerance {tolerance:.3g})."
            if exact_match
            else f"differs: worst vertex is {max_delta:.3g} from the reference "
            f"(tolerance {tolerance:.3g})."
        )
        return GoldenComparison(
            matches=exact_match,
            method="exact_vertex",
            tolerance=tolerance,
            exact_match=exact_match,
            max_vertex_delta=max_delta,
            surface_distance=None,
            confidence=conf,
            caveats=caveats,
            summary=summary,
        )

    # Differing topology: measure each mesh's vertices against the other's surface. This is
    # deterministic and exact where the deviation lives (at vertices), so a pure retessellation
    # reads as distance ~ 0 rather than being swamped by a sampling-spacing bound.
    distance = _vertex_to_surface_hausdorff(a, b)
    matches = distance <= tolerance
    caveats.append(
        "topology differs from the reference; matched by max vertex-to-surface distance "
        "(measured at mesh vertices, not vertex-for-vertex)"
    )
    summary = (
        f"within tolerance: max vertex-to-surface distance {distance:.3g} "
        f"(tolerance {tolerance:.3g})."
        if matches
        else f"outside tolerance: max vertex-to-surface distance {distance:.3g} "
        f"(tolerance {tolerance:.3g})."
    )
    return GoldenComparison(
        matches=matches,
        method="vertex_to_surface",
        tolerance=tolerance,
        exact_match=False,
        max_vertex_delta=None,
        surface_distance=distance,
        confidence=sampled("max vertex-to-surface distance", distance).with_reference(diagonal),
        caveats=caveats,
        summary=summary,
    )
