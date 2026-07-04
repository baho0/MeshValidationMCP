"""Mirror and rotational self-symmetry detection.

Symmetry is both a thing to verify (a part that should be symmetric actually is) and a strong
prior for other checks (a symmetric part must survive its own mirror). Candidate mirror planes
come from the principal inertia axes; each is confirmed by reflecting the mesh about the plane
and measuring how far the reflected surface sits from the original. Rotational symmetry tries
successive folds about each principal axis. Everything is a sampled surface distance, so the
confidence tier is ``estimated``.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, estimated
from .config import SEED
from .loading import LoadedMesh


class MirrorPlane(BaseModel):
    normal: list[float]
    origin: list[float]
    symmetry_error: float  # max surface distance between the mesh and its reflection


class SymmetryInfo(BaseModel):
    tolerance: float
    mirror_planes: list[MirrorPlane]
    rotational_axis: list[float] | None
    rotational_fold: int | None  # largest N with N-fold symmetry about that axis
    confidence: Confidence
    caveats: list[str]


def _self_distance(mesh: trimesh.Trimesh, transformed: trimesh.Trimesh, samples: int) -> float:
    """Max of the two-sided sampled surface distance between a mesh and a transformed copy."""
    pa, _ = trimesh.sample.sample_surface(mesh, samples, seed=SEED)
    pb, _ = trimesh.sample.sample_surface(transformed, samples, seed=SEED)
    d1 = trimesh.proximity.closest_point(transformed, pa)[1]
    d2 = trimesh.proximity.closest_point(mesh, pb)[1]
    return float(max(d1.max(), d2.max()))


def _householder(normal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """4x4 reflection about the plane through `origin` with unit `normal`."""
    n = normal / (np.linalg.norm(normal) or 1.0)
    h = np.eye(4)
    h[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    h[:3, 3] = 2.0 * float(n @ origin) * n
    return h


def detect_symmetry(
    loaded: LoadedMesh, rel_tolerance: float = 1e-3, samples: int = 800
) -> SymmetryInfo:
    mesh = loaded.combined
    diagonal = float(np.linalg.norm(mesh.extents)) or 1.0
    tol = rel_tolerance * diagonal
    centroid = np.asarray(mesh.centroid, dtype=float)

    try:
        axes = np.asarray(mesh.principal_inertia_vectors, dtype=float)
    except Exception:
        axes = np.eye(3)

    mirror_planes: list[MirrorPlane] = []
    for axis in axes:
        reflected = mesh.copy()
        reflected.apply_transform(_householder(axis, centroid))
        err = _self_distance(mesh, reflected, samples)
        if err <= tol:
            mirror_planes.append(
                MirrorPlane(
                    normal=[float(x) for x in axis / (np.linalg.norm(axis) or 1.0)],
                    origin=[float(x) for x in centroid],
                    symmetry_error=err,
                )
            )

    # Rotational symmetry: the largest fold N (2..12) about a principal axis that maps the
    # mesh onto itself. Test each axis; keep the axis with the highest confirmed fold.
    best_axis: list[float] | None = None
    best_fold: int | None = None
    for axis in axes:
        axis_u = axis / (np.linalg.norm(axis) or 1.0)
        for n in range(12, 1, -1):
            angle = 2.0 * np.pi / n
            rot = trimesh.transformations.rotation_matrix(angle, axis_u, centroid)
            rotated = mesh.copy()
            rotated.apply_transform(rot)
            if _self_distance(mesh, rotated, samples) <= tol:
                if best_fold is None or n > best_fold:
                    best_fold = n
                    best_axis = [float(x) for x in axis_u]
                break

    return SymmetryInfo(
        tolerance=tol,
        mirror_planes=mirror_planes,
        rotational_axis=best_axis,
        rotational_fold=best_fold,
        confidence=estimated("reflected/rotated surface distance", tol),
        caveats=[]
        if (mirror_planes or best_fold)
        else ["no mirror or rotational symmetry found among the principal axes"],
    )
