"""Region primitive: name a sub-region of a mesh for localized validation.

A Region is how an agent expresses "the area I manipulated" (the selected region
of an emboss/pocket/fillet). Every variant exposes ``vertex_mask(mesh)`` returning
a boolean array over the mesh vertices; ``face_mask`` derives from it. Spatial
variants (box/sphere/plane) also expose ``point_mask(points)`` so membership can be
tested on arbitrary points (used for footprint membership when topology differs).
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from .errors import ErrorCode, MeshToolError


class RegionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Spatial regions test a geometric predicate on coordinates and can be evaluated on
    # arbitrary points; index-based regions (vertex_ids/face_ids) cannot.
    spatial: ClassVar[bool] = False

    def point_mask(self, points: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("this region type cannot be evaluated on arbitrary points")

    def vertex_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def face_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """A face is inside the region when all three of its vertices are (conservative)."""
        vmask = self.vertex_mask(mesh)
        return vmask[mesh.faces].all(axis=1)


class BoxRegion(RegionBase):
    """Axis-aligned bounding box; a point is inside when min <= p <= max componentwise."""

    spatial: ClassVar[bool] = True
    kind: Literal["box"] = "box"
    min: list[float] = Field(min_length=3, max_length=3)
    max: list[float] = Field(min_length=3, max_length=3)

    def point_mask(self, points: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.min, dtype=float)
        hi = np.asarray(self.max, dtype=float)
        if np.any(hi < lo):
            raise MeshToolError(
                ErrorCode.INVALID_REGION,
                f"box region has max < min on some axis (min={self.min}, max={self.max})",
                "Ensure every component of max is >= the matching component of min.",
            )
        p = np.asarray(points, dtype=float)
        return np.all((p >= lo) & (p <= hi), axis=1)

    def vertex_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return self.point_mask(mesh.vertices)


class SphereRegion(RegionBase):
    """Ball of the given radius around center."""

    spatial: ClassVar[bool] = True
    kind: Literal["sphere"] = "sphere"
    center: list[float] = Field(min_length=3, max_length=3)
    radius: float = Field(gt=0)

    def point_mask(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        c = np.asarray(self.center, dtype=float)
        return np.linalg.norm(p - c, axis=1) <= self.radius

    def vertex_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return self.point_mask(mesh.vertices)


class PlaneRegion(RegionBase):
    """Half-space on the +normal side of the plane through origin (p inside when
    (p - origin) . normal >= 0)."""

    spatial: ClassVar[bool] = True
    kind: Literal["plane"] = "plane"
    origin: list[float] = Field(min_length=3, max_length=3)
    normal: list[float] = Field(min_length=3, max_length=3)

    def point_mask(self, points: np.ndarray) -> np.ndarray:
        n = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise MeshToolError(
                ErrorCode.INVALID_REGION,
                "plane region normal is the zero vector",
                "Provide a nonzero normal vector.",
            )
        p = np.asarray(points, dtype=float)
        o = np.asarray(self.origin, dtype=float)
        return (p - o) @ (n / norm) >= 0.0

    def vertex_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return self.point_mask(mesh.vertices)


class VertexIdsRegion(RegionBase):
    """Explicit set of vertex indices."""

    kind: Literal["vertex_ids"] = "vertex_ids"
    vertex_ids: list[int] = Field(min_length=1)

    def vertex_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:
        ids = np.asarray(self.vertex_ids, dtype=np.int64)
        n = len(mesh.vertices)
        if ids.min() < 0 or ids.max() >= n:
            raise MeshToolError(
                ErrorCode.INVALID_REGION,
                f"vertex_ids out of range for a mesh with {n} vertices "
                f"(got min={int(ids.min())}, max={int(ids.max())})",
                "Vertex ids must be in [0, vertex_count).",
            )
        mask = np.zeros(n, dtype=bool)
        mask[ids] = True
        return mask


class FaceIdsRegion(RegionBase):
    """Explicit set of face indices; their vertices define the region."""

    kind: Literal["face_ids"] = "face_ids"
    face_ids: list[int] = Field(min_length=1)

    def vertex_mask(self, mesh: trimesh.Trimesh) -> np.ndarray:
        ids = np.asarray(self.face_ids, dtype=np.int64)
        n = len(mesh.faces)
        if ids.min() < 0 or ids.max() >= n:
            raise MeshToolError(
                ErrorCode.INVALID_REGION,
                f"face_ids out of range for a mesh with {n} faces "
                f"(got min={int(ids.min())}, max={int(ids.max())})",
                "Face ids must be in [0, face_count).",
            )
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        mask[mesh.faces[ids].ravel()] = True
        return mask


Region = Annotated[
    Union[BoxRegion, SphereRegion, PlaneRegion, VertexIdsRegion, FaceIdsRegion],
    Field(discriminator="kind"),
]
