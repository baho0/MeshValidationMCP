"""Least-squares primitive fitting: is this surface (patch) a plane, sphere or cylinder,
and with what parameters?

Fitting a primitive to a set of points and reporting the residual is how we measure a
feature quantitatively: a drilled bore's radius (fit a cylinder to its wall), a fillet's
radius (fit a sphere/circle to the blend), a face's flatness (fit a plane). Every fit
returns its RMS residual so the caller can tell a clean feature from a poor one.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, estimated


class PlaneFit(BaseModel):
    kind: Literal["plane"] = "plane"
    point: list[float]
    normal: list[float]
    residual_rms: float
    confidence: Confidence


class SphereFit(BaseModel):
    kind: Literal["sphere"] = "sphere"
    center: list[float]
    radius: float
    residual_rms: float
    confidence: Confidence


class CylinderFit(BaseModel):
    kind: Literal["cylinder"] = "cylinder"
    point: list[float]  # a point on the axis
    axis: list[float]  # unit axis direction
    radius: float
    residual_rms: float
    confidence: Confidence


def fit_plane(points: np.ndarray) -> PlaneFit:
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centroid)
    normal = vt[-1]  # least-variance direction
    residual = float(np.sqrt(np.mean(((pts - centroid) @ normal) ** 2)))
    return PlaneFit(
        point=[float(x) for x in centroid],
        normal=[float(x) for x in normal],
        residual_rms=residual,
        confidence=estimated("SVD plane fit", residual),
    )


def fit_sphere(points: np.ndarray) -> SphereFit:
    pts = np.asarray(points, dtype=float)
    center, radius, _err = trimesh.nsphere.fit_nsphere(pts)
    center = np.asarray(center, dtype=float)
    residual = float(np.sqrt(np.mean((np.linalg.norm(pts - center, axis=1) - radius) ** 2)))
    return SphereFit(
        center=[float(x) for x in center],
        radius=float(radius),
        residual_rms=residual,
        confidence=estimated("algebraic sphere fit (fit_nsphere)", residual),
    )


def fit_cylinder(points: np.ndarray, axis_hint: np.ndarray | None = None) -> CylinderFit:
    """Fit a cylinder: estimate the axis (the least-variance direction of the surface
    normals, or a hint), project onto the perpendicular plane and fit a circle there."""
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)

    if axis_hint is not None:
        axis = np.asarray(axis_hint, dtype=float)
    else:
        # For points on a cylinder wall the axis is the direction of LARGEST spread when the
        # cylinder is longer than it is wide; use the top principal component as the estimate.
        _u, _s, vt = np.linalg.svd(pts - centroid)
        axis = vt[0]
    axis = axis / (np.linalg.norm(axis) or 1.0)

    # Project points onto the plane perpendicular to the axis, fit a circle there.
    rel = pts - centroid
    planar = rel - np.outer(rel @ axis, axis)
    e1 = planar[np.argmax(np.linalg.norm(planar, axis=1))]
    e1 = e1 / (np.linalg.norm(e1) or 1.0)
    e2 = np.cross(axis, e1)
    coords2d = np.column_stack([planar @ e1, planar @ e2])
    center2d, radius, _err = trimesh.nsphere.fit_nsphere(coords2d)
    axis_point = centroid + center2d[0] * e1 + center2d[1] * e2

    radial = np.linalg.norm(planar - (center2d[0] * e1 + center2d[1] * e2), axis=1)
    residual = float(np.sqrt(np.mean((radial - radius) ** 2)))
    return CylinderFit(
        point=[float(x) for x in axis_point],
        axis=[float(x) for x in axis],
        radius=float(radius),
        residual_rms=residual,
        confidence=estimated("axis PCA + circle fit", residual),
    )
