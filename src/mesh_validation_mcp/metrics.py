"""Geometric metric computation. Pure functions: LoadedMesh in, pydantic out."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from pydantic import BaseModel

from .integrity import IntegrityMetrics, compute_integrity
from .loading import LoadedMesh


class Bounds(BaseModel):
    min: list[float]
    max: list[float]


class BodyMetrics(BaseModel):
    vertex_count: int
    face_count: int
    volume: float | None
    is_watertight: bool
    bounds: Bounds


class MeshMetrics(BaseModel):
    path: str
    format: str
    file_size_bytes: int
    vertex_count: int
    face_count: int
    edge_count: int
    body_count: int
    is_watertight: bool
    is_winding_consistent: bool
    euler_number: int
    volume: float | None
    volume_reliable: bool
    surface_area: float
    bounds: Bounds
    extents: list[float]
    centroid: list[float]
    center_mass: list[float] | None
    bbox_diagonal: float
    units: str | None
    integrity: IntegrityMetrics
    bodies: list[BodyMetrics]
    caveats: list[str]


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _vec(value: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(value).ravel()]


def _body_metrics(body: trimesh.Trimesh) -> BodyMetrics:
    return BodyMetrics(
        vertex_count=len(body.vertices),
        face_count=len(body.faces),
        volume=_finite(body.volume),
        is_watertight=bool(body.is_watertight),
        bounds=Bounds(min=_vec(body.bounds[0]), max=_vec(body.bounds[1])),
    )


def compute_metrics(loaded: LoadedMesh) -> MeshMetrics:
    mesh = loaded.combined
    caveats: list[str] = []

    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    reliable = watertight and winding
    volume = _finite(mesh.volume)

    if not watertight:
        caveats.append(
            "not_watertight: the mesh has open boundaries; volume and center_mass are "
            "computed via the divergence theorem and may be unreliable"
        )
    elif not winding:
        caveats.append(
            "winding_inconsistent: face windings disagree; volume and center_mass may be unreliable"
        )
    if reliable and volume is not None and volume < 0:
        caveats.append("negative_volume: face normals point inward (inverted winding)")

    integrity = compute_integrity(mesh)
    if integrity.self_intersecting_face_count > 0:
        caveats.append(
            f"self_intersecting: {integrity.self_intersecting_face_count} faces intersect; "
            "volume and watertight status may be misleading"
        )
    elif not integrity.self_intersection_checked:
        caveats.append("self_intersection_not_checked: mesh exceeds the face cap for this test")
    if integrity.non_manifold_edge_count > 0:
        caveats.append(
            f"non_manifold: {integrity.non_manifold_edge_count} edges are shared by 3+ faces"
        )

    extents = np.asarray(mesh.extents, dtype=float)
    return MeshMetrics(
        path=loaded.path,
        format=loaded.format,
        file_size_bytes=loaded.file_size_bytes,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        edge_count=len(mesh.edges_unique),
        body_count=len(loaded.bodies),
        is_watertight=watertight,
        is_winding_consistent=winding,
        euler_number=int(mesh.euler_number),
        volume=volume,
        volume_reliable=reliable,
        surface_area=float(mesh.area),
        bounds=Bounds(min=_vec(mesh.bounds[0]), max=_vec(mesh.bounds[1])),
        extents=_vec(extents),
        centroid=_vec(mesh.centroid),
        center_mass=_vec(mesh.center_mass) if reliable else None,
        bbox_diagonal=float(np.linalg.norm(extents)),
        units=str(mesh.units) if mesh.units else None,
        integrity=integrity,
        bodies=[_body_metrics(b) for b in loaded.bodies],
        caveats=caveats,
    )
