"""Planar cross-sections: slice a mesh with a plane and measure the resulting profile.

A cross-section turns a 3D solid into an exact 2D profile whose enclosed area and perimeter
can be checked against an analytic expectation (a prism's constant rectangle, a cylinder's
circle, an extrusion's profile). We compute area and perimeter WITHOUT shapely: each section
loop is projected onto an orthonormal basis of the cut plane and measured with the shoelace
formula; holes are subtracted by an even-odd point-in-polygon containment test.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, exact
from .errors import ErrorCode, MeshToolError
from .loading import LoadedMesh


class LoopInfo(BaseModel):
    perimeter: float
    area: float  # unsigned area enclosed by this loop alone
    is_hole: bool  # nested an odd number of loops deep -> subtracted from the net area
    is_closed: bool
    point_count: int
    centroid: list[float]  # 3D centroid of the loop vertices


class SectionInfo(BaseModel):
    plane_origin: list[float]
    plane_normal: list[float]
    intersects: bool
    loop_count: int
    net_area: float  # solid area minus holes
    gross_area: float  # sum of every loop's area (no hole subtraction)
    total_perimeter: float
    loops: list[LoopInfo]
    confidence: Confidence
    caveats: list[str]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A right-handed (u, v) basis spanning the plane with u x v == normal."""
    n = normal / (np.linalg.norm(normal) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(n, u)
    return u, v


def _shoelace(pts2d: np.ndarray) -> float:
    """Signed area of a 2D polyline via the shoelace formula (sign = winding)."""
    x, y = pts2d[:, 0], pts2d[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (poly is an (N,2) closed-ish ring)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def section_cut_faces(
    mesh: trimesh.Trimesh, plane_origin: list[float], plane_normal: list[float]
) -> np.ndarray:
    """Face ids the plane passes through (their vertices straddle the plane) — for an overlay
    that shows WHERE the section was taken."""
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    signed = (np.asarray(mesh.vertices) - np.asarray(plane_origin, dtype=float)) @ n
    face_signs = signed[mesh.faces]
    straddles = (face_signs.min(axis=1) < 0) & (face_signs.max(axis=1) > 0)
    return np.nonzero(straddles)[0]


class SectionProfile(BaseModel):
    axis: list[float]
    stations: list[float]  # signed positions along the axis (relative to the first station)
    areas: list[float]  # net cross-section area at each station
    perimeters: list[float]
    area_constant: bool  # every area within 2% of the mean (an extrusion/prism signature)
    area_monotonic: bool  # areas strictly increase or decrease (a taper signature)
    confidence: Confidence
    caveats: list[str]


def section_area_profile(
    loaded: LoadedMesh, axis: list[float], stations: int = 9
) -> SectionProfile:
    """Cross-section area sampled at several stations along an axis. A constant profile is the
    signature of an extrusion/prism; a monotonic profile is a taper; a wobble is a twist or a
    local feature. Stations span the mesh's extent along the axis (endpoints inset slightly)."""
    mesh = loaded.combined
    n = np.asarray(axis, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    proj = np.asarray(mesh.vertices) @ n
    lo, hi = float(proj.min()), float(proj.max())
    span = hi - lo
    ts = np.linspace(lo + 0.05 * span, hi - 0.05 * span, stations)

    areas: list[float] = []
    perims: list[float] = []
    for t in ts:
        info = inspect_section(loaded, [float(x) for x in n * t], [float(x) for x in n])
        areas.append(info.net_area)
        perims.append(info.total_perimeter)

    arr = np.asarray(areas)
    mean = float(arr.mean()) or 1.0
    area_constant = bool(np.all(np.abs(arr - mean) <= 0.02 * abs(mean)))
    diffs = np.diff(arr)
    area_monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))

    return SectionProfile(
        axis=[float(x) for x in n],
        stations=[float(t - ts[0]) for t in ts],
        areas=areas,
        perimeters=perims,
        area_constant=area_constant,
        area_monotonic=area_monotonic,
        confidence=exact("net section area at stations along the axis"),
        caveats=[],
    )


def inspect_section(
    loaded: LoadedMesh, plane_origin: list[float], plane_normal: list[float]
) -> SectionInfo:
    """Slice the mesh with a plane and report the profile's loops, area and perimeter."""
    mesh = loaded.combined
    origin = np.asarray(plane_origin, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    if np.linalg.norm(normal) == 0:
        raise MeshToolError(
            ErrorCode.INVALID_REGION, "plane_normal is the zero vector", "Provide a nonzero normal."
        )

    section = mesh.section(plane_origin=origin, plane_normal=normal)
    caveats: list[str] = []
    if section is None:
        return SectionInfo(
            plane_origin=[float(x) for x in origin],
            plane_normal=[float(x) for x in normal],
            intersects=False,
            loop_count=0,
            net_area=0.0,
            gross_area=0.0,
            total_perimeter=0.0,
            loops=[],
            confidence=exact("no plane/mesh intersection"),
            caveats=["the plane does not intersect the mesh"],
        )

    u, v = _plane_basis(normal)
    loops3d = [np.asarray(d, dtype=float) for d in section.discrete]
    loops2d = [np.column_stack([(p - origin) @ u, (p - origin) @ v]) for p in loops3d]

    areas = [abs(_shoelace(p)) for p in loops2d]
    perimeters = [
        float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in loops3d
    ]
    # A loop is a hole when it sits inside an odd number of other loops (even-odd rule).
    # The test point is a vertex ON this loop's boundary, not its centroid: concentric loops
    # (e.g. an annulus) share the same centroid, which would misclassify both as holes.
    is_hole = []
    for i, poly in enumerate(loops2d):
        probe = poly[0]
        depth = sum(
            _point_in_polygon(probe, loops2d[j]) for j in range(len(loops2d)) if j != i
        )
        is_hole.append(depth % 2 == 1)

    net_area = sum(a if not h else -a for a, h in zip(areas, is_hole))
    gross_area = sum(areas)

    loop_infos = []
    for i, pts3 in enumerate(loops3d):
        closed = bool(np.allclose(pts3[0], pts3[-1], atol=1e-9)) or len(pts3) > 2
        loop_infos.append(
            LoopInfo(
                perimeter=perimeters[i],
                area=areas[i],
                is_hole=is_hole[i],
                is_closed=closed,
                point_count=len(pts3),
                centroid=[float(x) for x in pts3.mean(axis=0)],
            )
        )
    if len(loops3d) > 1:
        caveats.append(
            "multiple section loops: net_area subtracts holes by an even-odd containment test"
        )

    return SectionInfo(
        plane_origin=[float(x) for x in origin],
        plane_normal=[float(x) for x in normal],
        intersects=True,
        loop_count=len(loops3d),
        net_area=float(net_area),
        gross_area=float(gross_area),
        total_perimeter=float(sum(perimeters)),
        loops=loop_infos,
        confidence=exact("planar section: shoelace area, polyline perimeter"),
        caveats=caveats,
    )
