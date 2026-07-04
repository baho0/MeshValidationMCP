"""Silhouette / outline comparison: do two meshes look the same from a canonical direction?

Some edits must NOT change the outline (an emboss keeps the plate's silhouette; a texture pass
keeps the profile). We rasterize each mesh's orthographic silhouette onto a shared grid and
report the intersection-over-union and one-way containment. It is a coarse, view-dependent
check (a sampled/estimated tier), complementary to the exact geometric measures.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, sampled
from .loading import LoadedMesh

Axis = Literal["x", "y", "z"]
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class SilhouetteComparison(BaseModel):
    view_axis: Axis
    resolution: int
    iou: float
    a_covered_fraction: float  # fraction of A's silhouette also covered by B
    b_covered_fraction: float
    confidence: Confidence
    caveats: list[str]


def _raster_triangle(mask: np.ndarray, tri_px: np.ndarray) -> None:
    res = mask.shape[0]
    minx = max(int(np.floor(tri_px[:, 0].min())), 0)
    maxx = min(int(np.ceil(tri_px[:, 0].max())), res - 1)
    miny = max(int(np.floor(tri_px[:, 1].min())), 0)
    maxy = min(int(np.ceil(tri_px[:, 1].max())), res - 1)
    if maxx < minx or maxy < miny:
        return
    xs = np.arange(minx, maxx + 1) + 0.5
    ys = np.arange(miny, maxy + 1) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    (x1, y1), (x2, y2), (x3, y3) = tri_px
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-12:
        return
    a = ((y2 - y3) * (gx - x3) + (x3 - x2) * (gy - y3)) / denom
    b = ((y3 - y1) * (gx - x3) + (x1 - x3) * (gy - y3)) / denom
    c = 1.0 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    mask[miny : maxy + 1, minx : maxx + 1] |= inside


def _rasterize(
    mesh: trimesh.Trimesh, drop: int, lo: np.ndarray, span: float, resolution: int
) -> np.ndarray:
    keep = [i for i in range(3) if i != drop]
    verts2d = np.asarray(mesh.vertices)[:, keep]
    px = (verts2d - lo) / span * resolution
    mask = np.zeros((resolution, resolution), dtype=bool)
    for face in mesh.faces:
        _raster_triangle(mask, px[face])
    return mask


def silhouette_compare(
    loaded_a: LoadedMesh, loaded_b: LoadedMesh, view_axis: Axis = "z", resolution: int = 256
) -> SilhouetteComparison:
    a, b = loaded_a.combined, loaded_b.combined
    drop = _AXIS_INDEX[view_axis]
    keep = [i for i in range(3) if i != drop]

    # Shared square grid covering both silhouettes (equal scale so IoU is meaningful).
    allv = np.vstack([np.asarray(a.vertices)[:, keep], np.asarray(b.vertices)[:, keep]])
    lo = allv.min(axis=0)
    hi = allv.max(axis=0)
    span = float((hi - lo).max()) or 1.0
    lo = lo - 0.02 * span  # small margin
    span = span * 1.04

    mask_a = _rasterize(a, drop, lo, span, resolution)
    mask_b = _rasterize(b, drop, lo, span, resolution)
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum()) or 1.0
    area_a = float(mask_a.sum()) or 1.0
    area_b = float(mask_b.sum()) or 1.0

    return SilhouetteComparison(
        view_axis=view_axis,
        resolution=resolution,
        iou=inter / union,
        a_covered_fraction=inter / area_a,
        b_covered_fraction=inter / area_b,
        confidence=sampled(f"rasterized silhouette IoU at {resolution}px", 1.0 / resolution),
        caveats=[],
    )
