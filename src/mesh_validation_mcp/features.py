"""CAD-feature measurement: wall thickness, draft/undercut, and region primitive fits.

These are the quantitative checks that turn "I filleted that edge" or "I shelled the part"
into a number the agent can assert: the minimum wall thickness of a shell, the draft angle
and undercut area for a pull direction, the radius of a fillet or bore (by fitting a cylinder
or sphere to the selected feature faces).
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, exact, sampled
from .config import SEED
from .errors import ErrorCode, MeshToolError
from .loading import LoadedMesh
from .primitives import CylinderFit, PlaneFit, SphereFit, fit_cylinder, fit_plane, fit_sphere
from .region import RegionBase


class ThicknessInfo(BaseModel):
    min_thickness: float
    p5_thickness: float  # 5th percentile: a robust minimum that ignores edge artefacts
    median_thickness: float
    mean_thickness: float
    max_thickness: float
    sample_count: int
    thinnest_point: list[float]
    confidence: Confidence
    caveats: list[str]


class DraftInfo(BaseModel):
    pull_direction: list[float]
    min_draft_angle_deg: float  # smallest draft among pullable faces (0 = vertical wall)
    undercut_area: float
    undercut_fraction: float  # fraction of total area that is undercut
    undercut_face_count: int
    threshold_deg: float | None
    area_below_threshold: float | None  # pullable-but-insufficient-draft area
    confidence: Confidence
    caveats: list[str]


def wall_thickness(loaded: LoadedMesh, sample_count: int = 2000) -> ThicknessInfo:
    """Inscribed-sphere wall thickness sampled over the surface. Requires a watertight solid
    (inside/outside must be well-defined). The min is reported but is biased low near sharp
    convex edges; p5_thickness is the robust thin-wall indicator."""
    mesh = loaded.combined
    if not mesh.is_watertight:
        raise MeshToolError(
            ErrorCode.INVALID_EXPECTATION,
            "wall thickness needs a watertight mesh (inside/outside must be defined)",
            "Repair the mesh to be watertight before measuring wall thickness.",
        )
    points, _fid = trimesh.sample.sample_surface_even(mesh, sample_count, seed=SEED)
    thickness = np.asarray(
        trimesh.proximity.thickness(mesh, points, exterior=False, method="max_sphere"),
        dtype=float,
    )
    finite = np.isfinite(thickness)
    thickness = thickness[finite]
    points = points[finite]
    if thickness.size == 0:
        raise MeshToolError(
            ErrorCode.INVALID_EXPECTATION, "thickness could not be measured at any sample point"
        )
    imin = int(np.argmin(thickness))
    diagonal = float(np.linalg.norm(mesh.extents)) or 1.0
    spacing = diagonal / float(np.sqrt(max(thickness.size, 1)))
    return ThicknessInfo(
        min_thickness=float(thickness.min()),
        p5_thickness=float(np.percentile(thickness, 5)),
        median_thickness=float(np.median(thickness)),
        mean_thickness=float(thickness.mean()),
        max_thickness=float(thickness.max()),
        sample_count=int(thickness.size),
        thinnest_point=[float(x) for x in points[imin]],
        confidence=sampled(
            f"inscribed-sphere thickness, {thickness.size} even surface samples", spacing
        ),
        caveats=[
            "the bare min is biased low near sharp convex edges; use p5_thickness as the "
            "robust thin-wall indicator"
        ],
    )


def draft_analysis(
    loaded: LoadedMesh, pull_direction: list[float], min_draft_deg: float | None = None
) -> DraftInfo:
    """Area-weighted draft/undercut analysis for a pull (de-mold) direction. A face's draft
    angle is 90deg minus the angle between its normal and the pull direction; a face whose
    normal points against the pull (draft < 0) is an undercut."""
    mesh = loaded.combined
    pull = np.asarray(pull_direction, dtype=float)
    norm = np.linalg.norm(pull)
    if norm == 0:
        raise MeshToolError(
            ErrorCode.INVALID_REGION, "pull_direction is the zero vector", "Provide a nonzero vector."
        )
    pull = pull / norm

    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    cos_theta = np.clip(normals @ pull, -1.0, 1.0)
    draft_deg = 90.0 - np.degrees(np.arccos(cos_theta))  # +90 faces pull, 0 vertical, <0 undercut

    undercut = draft_deg < -1e-6
    undercut_area = float(areas[undercut].sum())
    total_area = float(areas.sum()) or 1.0
    pullable = ~undercut
    min_draft = float(draft_deg[pullable].min()) if pullable.any() else float(draft_deg.min())

    area_below = None
    if min_draft_deg is not None:
        insufficient = pullable & (draft_deg < min_draft_deg)
        area_below = float(areas[insufficient].sum())

    return DraftInfo(
        pull_direction=[float(x) for x in pull],
        min_draft_angle_deg=min_draft,
        undercut_area=undercut_area,
        undercut_fraction=undercut_area / total_area,
        undercut_face_count=int(undercut.sum()),
        threshold_deg=min_draft_deg,
        area_below_threshold=area_below,
        confidence=exact("area-weighted face-normal vs pull direction"),
        caveats=[],
    )


FeatureFit = Union[PlaneFit, SphereFit, CylinderFit]


def fit_region(
    loaded: LoadedMesh, region: RegionBase, kind: Literal["plane", "sphere", "cylinder"]
) -> FeatureFit:
    """Fit a primitive to the vertices a region selects: a cylinder for a fillet band or a
    bore wall (radius), a plane for a chamfer face, a sphere for a spherical blend/dome. The
    returned residual_rms is how well the feature actually matches that primitive."""
    mesh = loaded.combined
    mask = region.vertex_mask(mesh)
    pts = np.asarray(mesh.vertices)[mask]
    if len(pts) < 4:
        raise MeshToolError(
            ErrorCode.INVALID_REGION,
            f"region selects only {len(pts)} vertices; need at least 4 to fit a {kind}",
            "Widen the region to cover the whole feature.",
        )
    if kind == "plane":
        return fit_plane(pts)
    if kind == "sphere":
        return fit_sphere(pts)
    return fit_cylinder(pts)
