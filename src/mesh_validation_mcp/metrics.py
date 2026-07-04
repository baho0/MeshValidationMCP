"""Geometric metric computation. Pure functions: LoadedMesh in, pydantic out."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from pydantic import BaseModel

from .curvature import CurvatureInfo, curvature_field
from .integrity import IntegrityMetrics, compute_integrity
from .loading import LoadedMesh
from .topology import TopologyInfo, analyze_topology


class Bounds(BaseModel):
    min: list[float]
    max: list[float]


class BodyMetrics(BaseModel):
    vertex_count: int
    face_count: int
    volume: float | None
    is_watertight: bool
    bounds: Bounds


class InertiaInfo(BaseModel):
    """Mass-distribution invariants (unit density). Defined only for a reliable solid."""

    principal_moments: list[float]  # eigenvalues of the inertia tensor, ascending
    principal_axes: list[list[float]]  # 3x3, rows are the principal directions


class CrossChecks(BaseModel):
    """Redundant, independent derivations of the same quantities. When two paths to the
    'same' number disagree, the mesh is pathological and the primary metric should not be
    trusted — `consistent` is the single flag an oracle can gate on."""

    # Volume by the divergence theorem vs by summing per-body signed volumes.
    volume_divergence: float | None
    volume_body_sum: float | None
    volume_agreement: bool | None  # None when volume is unreliable (open/inconsistent)
    # Euler characteristic by V-E+F vs by the Gauss-Bonnet angle-defect sum / 2*pi.
    euler_number: int
    euler_gauss_bonnet: float | None
    euler_agreement: bool | None
    # The is_watertight flag vs the independent "no boundary edges" test.
    watertight_flag: bool
    boundary_free: bool
    watertight_agreement: bool
    consistent: bool


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
    cross_checks: CrossChecks
    topology: TopologyInfo
    curvature: CurvatureInfo
    inertia: InertiaInfo | None
    bodies: list[BodyMetrics]
    caveats: list[str]


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _vec(value: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(value).ravel()]


def _cross_checks(
    mesh: trimesh.Trimesh,
    bodies: list[trimesh.Trimesh],
    integrity: IntegrityMetrics,
    volume: float | None,
    reliable: bool,
) -> CrossChecks:
    watertight = bool(mesh.is_watertight)
    boundary_free = integrity.boundary_edge_count == 0

    # Volume by an independent path: sum of per-body signed volumes (catches a body with
    # inverted winding whose negative volume the divergence sum would otherwise hide).
    body_sum: float | None = None
    volume_agreement: bool | None = None
    if reliable and volume is not None:
        try:
            body_sum = float(sum(float(b.volume) for b in bodies))
            volume_agreement = abs(body_sum - volume) <= 1e-4 * max(abs(volume), 1.0)
        except Exception:
            body_sum, volume_agreement = None, None

    # Euler characteristic by Gauss-Bonnet: on a CLOSED surface the angle-defect sum equals
    # 2*pi*chi exactly. An open surface carries a boundary term, so we only cross-check when
    # watertight (the value is still reported for information).
    euler_gb: float | None = None
    euler_agreement: bool | None = None
    try:
        defects = trimesh.curvature.vertex_defects(mesh)
        euler_gb = float(np.asarray(defects).sum() / (2.0 * math.pi))
        if watertight:
            euler_agreement = abs(euler_gb - mesh.euler_number) <= 0.5
    except Exception:
        euler_gb, euler_agreement = None, None

    watertight_agreement = watertight == boundary_free
    consistent = (
        (volume_agreement is not False)
        and (euler_agreement is not False)
        and watertight_agreement
    )
    return CrossChecks(
        volume_divergence=volume,
        volume_body_sum=body_sum,
        volume_agreement=volume_agreement,
        euler_number=int(mesh.euler_number),
        euler_gauss_bonnet=euler_gb,
        euler_agreement=euler_agreement,
        watertight_flag=watertight,
        boundary_free=boundary_free,
        watertight_agreement=watertight_agreement,
        consistent=consistent,
    )


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
    cross = _cross_checks(mesh, loaded.bodies, integrity, volume, reliable)
    topology = analyze_topology(loaded)
    curvature = curvature_field(loaded)

    inertia: InertiaInfo | None = None
    if reliable:
        try:
            inertia = InertiaInfo(
                principal_moments=_vec(mesh.principal_inertia_components),
                principal_axes=[
                    [float(x) for x in row] for row in np.asarray(mesh.principal_inertia_vectors)
                ],
            )
        except Exception:
            inertia = None
    if not cross.consistent:
        reasons = []
        if cross.euler_agreement is False:
            reasons.append(
                f"euler mismatch (V-E+F={cross.euler_number} vs Gauss-Bonnet "
                f"{cross.euler_gauss_bonnet:.3g})"
            )
        if cross.volume_agreement is False:
            reasons.append(
                f"volume paths disagree (divergence={cross.volume_divergence:.6g} vs "
                f"body-sum={cross.volume_body_sum:.6g})"
            )
        if not cross.watertight_agreement:
            reasons.append(
                f"is_watertight={cross.watertight_flag} but boundary_free={cross.boundary_free}"
            )
        caveats.append("cross_check_inconsistent: " + "; ".join(reasons))
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
        cross_checks=cross,
        topology=topology,
        curvature=curvature,
        inertia=inertia,
        bodies=[_body_metrics(b) for b in loaded.bodies],
        caveats=caveats,
    )
