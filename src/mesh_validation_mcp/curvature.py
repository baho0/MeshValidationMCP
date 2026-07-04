"""Curvature and sharp-edge measures.

These quantify how a surface bends, which is how CAD features leave a signature: a fillet
replaces a sharp edge with a band of roughly constant curvature; a chamfer replaces it with
a new flat face at a fixed dihedral angle; a boss/rib adds curvature where a face used to be
flat. We report the Gauss-Bonnet total (a topological cross-check), per-vertex Gaussian
angle-defect statistics, and a dihedral-angle histogram whose "sharp" tail counts hard edges.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, exact, sampled
from .loading import LoadedMesh

# Faces meeting at more than this dihedral angle count as a sharp edge (fillets/chamfers
# push edges below it).
DEFAULT_SHARP_ANGLE_DEG = 30.0


class CurvatureInfo(BaseModel):
    total_gaussian: float  # sum of vertex angle defects
    gaussian_over_2pi: float  # ~ Euler characteristic for a closed mesh
    max_abs_vertex_defect: float  # sharpest corner (radians)
    sharp_angle_threshold_deg: float
    sharp_edge_count: int
    max_dihedral_deg: float
    mean_dihedral_deg: float
    confidence: Confidence
    caveats: list[str]


def curvature_field(
    loaded: LoadedMesh, sharp_angle_deg: float = DEFAULT_SHARP_ANGLE_DEG
) -> CurvatureInfo:
    mesh = loaded.combined
    caveats: list[str] = []

    try:
        defects = np.asarray(mesh.vertex_defects, dtype=float)
        total_gaussian = float(defects.sum())
        max_abs_defect = float(np.abs(defects).max()) if defects.size else 0.0
    except Exception:
        total_gaussian, max_abs_defect = 0.0, 0.0
        caveats.append("vertex angle defects could not be computed")

    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    if angles.size:
        angles_deg = np.degrees(angles)
        sharp_edge_count = int((angles_deg >= sharp_angle_deg).sum())
        max_dihedral = float(angles_deg.max())
        mean_dihedral = float(angles_deg.mean())
    else:
        sharp_edge_count, max_dihedral, mean_dihedral = 0, 0.0, 0.0
        caveats.append("mesh has no interior edges to measure dihedral angles")

    if not mesh.is_watertight:
        caveats.append("mesh is not closed: Gaussian total omits the boundary geodesic term")

    return CurvatureInfo(
        total_gaussian=total_gaussian,
        gaussian_over_2pi=total_gaussian / (2.0 * np.pi),
        max_abs_vertex_defect=max_abs_defect,
        sharp_angle_threshold_deg=sharp_angle_deg,
        sharp_edge_count=sharp_edge_count,
        max_dihedral_deg=max_dihedral,
        mean_dihedral_deg=mean_dihedral,
        confidence=(
            exact("Gauss-Bonnet angle defects + dihedral angles")
            if mesh.is_watertight
            else sampled("dihedral angles (open mesh)", 0.0)
        ),
        caveats=caveats,
    )
