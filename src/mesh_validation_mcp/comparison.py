"""Before/after mesh comparison: transform detection, distances, heatmap."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import (
    HEATMAP_MAX_VERTICES,
    LOCALIZED_CHANGE_REL,
    MAX_COMPARE_SAMPLES,
    SEED,
)
from .loading import LoadedMesh
from .metrics import Bounds, compute_metrics
from .region import RegionBase

Classification = Literal[
    "identical", "translation", "rotation", "rigid", "similarity", "mirrored", "deformed"
]

# Thresholds are relative to A's bbox diagonal. The exact-correspondence path
# only carries float32 export noise (~1e-7 relative); the ICP path additionally
# carries registration error (~1e-4 relative even for perfect transforms), so
# its thresholds are an order of magnitude looser.
_THRESHOLDS = {
    "procrustes_exact": {"residual": 1e-4, "translation": 1e-6, "angle_deg": 1e-3, "scale": 1e-6},
    "icp": {"residual": 1e-3, "translation": 1e-3, "angle_deg": 0.05, "scale": 1e-3},
}


class TransformInfo(BaseModel):
    matrix_4x4: list[list[float]]
    translation: list[float]
    rotation_axis: list[float]
    rotation_angle_deg: float
    uniform_scale: float
    residual_rms: float
    includes_reflection: bool = False


class DistanceInfo(BaseModel):
    chamfer_mean: float
    hausdorff_approx: float
    aligned_residual_rms: float
    sample_count: int
    seed: int


class ComparisonReport(BaseModel):
    classification: Classification
    method: Literal["procrustes_exact", "icp"]
    transform: TransformInfo
    distances: DistanceInfo
    metric_deltas: dict[str, Any]
    caveats: list[str]
    summary: str


class SignedDisplacement(BaseModel):
    """Signed motion of the changed vertices along the before-surface normal.
    Positive = material added outward (emboss/boss); negative = removed (pocket).
    `peak` is None when the signed depth could not be determined (differing topology
    where the changed vertices did not map into the region)."""

    peak: float | None  # signed value at the 95th-percentile magnitude (robust height/depth)
    mean: float
    max_outward: float
    max_inward: float
    outward_fraction: float


class LocalizedChange(BaseModel):
    reference_mesh: Literal["A", "B"]  # which mesh the region was evaluated on
    same_topology: bool
    change_threshold: float
    region_vertex_count: int
    changed_vertex_count: int
    inside_max_displacement: float
    outside_max_displacement: float  # the "is the rest untouched?" number
    changed_region_bounds: Bounds | None
    changed_region_centroid: list[float] | None
    signed_displacement: SignedDisplacement
    caveats: list[str]


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _fit_residual(
    a: trimesh.Trimesh, b: trimesh.Trimesh, matrix: np.ndarray, samples: int = 500
) -> float:
    probe, _ = trimesh.sample.sample_surface(a, samples, seed=SEED)
    return float(np.sqrt(np.mean(trimesh.proximity.closest_point(b, _apply(matrix, probe))[1] ** 2)))


def _mirror_about_centroid(centroid: np.ndarray, axis: int) -> np.ndarray:
    to_origin = np.eye(4)
    to_origin[:3, 3] = -centroid
    back = np.eye(4)
    back[:3, 3] = centroid
    flip = np.eye(4)
    flip[axis, axis] = -1.0
    return back @ flip @ to_origin


def _try_dereflect(
    a: trimesh.Trimesh, b: trimesh.Trimesh, matrix: np.ndarray, reflected_residual: float
) -> tuple[np.ndarray, float] | None:
    """ICP can lock onto a mirrored fit for mirror-symmetric parts. Composing the
    matrix with an axis mirror through A's centroid yields a proper (det>0)
    candidate; among candidates that fit about as well as the reflected fit,
    prefer the one with the smallest rotation angle (cleanest description)."""
    centroid = np.asarray(a.centroid, dtype=float)
    accept = max(reflected_residual * 1.5, 1e-9)
    candidates: list[tuple[np.ndarray, float, float]] = []
    for axis in range(3):
        candidate = matrix @ _mirror_about_centroid(centroid, axis)
        residual = _fit_residual(a, b, candidate)
        if residual <= accept:
            _scale, _t, angle_deg, _axis = _decompose(candidate)
            candidates.append((candidate, residual, angle_deg))
    if not candidates:
        return None
    best = min(candidates, key=lambda c: c[2])
    return best[0], best[1]


def _fmt_vec(v: np.ndarray | list[float]) -> str:
    return "[" + ", ".join(f"{float(x):.4g}" for x in v) + "]"


def _decompose(matrix: np.ndarray) -> tuple[float, np.ndarray, float, list[float]]:
    linear = matrix[:3, :3]
    det = float(np.linalg.det(linear))
    scale = float(np.cbrt(abs(det))) or 1.0
    # Re-orthogonalize before extracting the rotation (numeric noise safety).
    u, _s, vt = np.linalg.svd(linear / scale)
    if np.linalg.det(u @ vt) < 0:
        vt[-1] *= -1
    rotvec = Rotation.from_matrix(u @ vt).as_rotvec()
    angle_rad = float(np.linalg.norm(rotvec))
    angle_deg = math.degrees(angle_rad)
    axis = (rotvec / angle_rad).tolist() if angle_rad > 1e-12 else [0.0, 0.0, 1.0]
    return scale, matrix[:3, 3].copy(), angle_deg, axis


def _classify(
    residual: float,
    scale: float,
    translation: np.ndarray,
    angle_deg: float,
    diagonal: float,
    method: str,
) -> Classification:
    limits = _THRESHOLDS[method]
    if residual > limits["residual"] * diagonal:
        return "deformed"
    if abs(scale - 1.0) > limits["scale"]:
        return "similarity"
    moved = float(np.linalg.norm(translation)) > limits["translation"] * diagonal
    rotated = angle_deg > limits["angle_deg"]
    if moved and rotated:
        return "rigid"
    if rotated:
        return "rotation"
    if moved:
        return "translation"
    return "identical"


def _summary(
    classification: Classification,
    method: str,
    scale: float,
    translation: np.ndarray,
    angle_deg: float,
    axis: list[float],
    residual: float,
    diagonal: float,
) -> str:
    basis = "exact vertex correspondence" if method == "procrustes_exact" else "ICP estimate"
    if classification == "identical":
        return f"B is geometrically identical to A ({basis}, residual RMS {residual:.3g})."
    if classification == "translation":
        return f"B is A translated by {_fmt_vec(translation)} ({basis}, residual RMS {residual:.3g})."
    if classification == "rotation":
        return (
            f"B is A rotated {angle_deg:.4g} deg about axis {_fmt_vec(axis)} "
            f"({basis}, residual RMS {residual:.3g})."
        )
    if classification == "rigid":
        return (
            f"B is A rotated {angle_deg:.4g} deg about axis {_fmt_vec(axis)} and translated "
            f"by {_fmt_vec(translation)} ({basis}, residual RMS {residual:.3g})."
        )
    if classification == "mirrored":
        return (
            f"B is a MIRRORED copy of A: the fit requires a reflection ({basis}, "
            f"residual RMS {residual:.3g}). The 4x4 matrix includes the reflection; "
            "rotation axis/angle values are approximate."
        )
    if classification == "similarity":
        limits = _THRESHOLDS[method]
        parts = [f"scaled x{scale:.6g}"]
        if angle_deg > limits["angle_deg"]:
            parts.append(f"rotated {angle_deg:.4g} deg about {_fmt_vec(axis)}")
        if float(np.linalg.norm(translation)) > limits["translation"] * diagonal:
            parts.append(f"translated by {_fmt_vec(translation)}")
        return f"B is A {', '.join(parts)} ({basis}, residual RMS {residual:.3g})."
    return (
        f"B is NOT a rigid/similarity transform of A: best-fit residual RMS {residual:.3g} "
        f"(~{residual / diagonal:.2%} of A's bbox diagonal). "
        "Inspect the distances and the displacement heatmap."
    )


def _signed_displacement(outward: np.ndarray) -> SignedDisplacement:
    if outward.size == 0:
        return SignedDisplacement(
            peak=0.0, mean=0.0, max_outward=0.0, max_inward=0.0, outward_fraction=0.0
        )
    order = np.argsort(np.abs(outward))
    peak = float(outward[order[int(round(0.95 * (len(outward) - 1)))]])
    return SignedDisplacement(
        peak=peak,
        mean=float(outward.mean()),
        max_outward=float(max(outward.max(), 0.0)),
        max_inward=float(min(outward.min(), 0.0)),
        outward_fraction=float((outward > 0).mean()),
    )


def localized_change(
    loaded_a: LoadedMesh,
    loaded_b: LoadedMesh,
    region: RegionBase,
    change_threshold: float | None = None,
) -> LocalizedChange:
    """Restrict the before/after comparison to a region: how much moved inside it,
    whether everything outside stayed put, and the signed height/depth of the change."""
    a, b = loaded_a.combined, loaded_b.combined
    caveats: list[str] = []
    diagonal = float(np.linalg.norm(a.extents)) or 1.0
    threshold = (
        change_threshold if change_threshold is not None else LOCALIZED_CHANGE_REL * diagonal
    )

    same_topology = len(a.vertices) == len(b.vertices) and np.array_equal(a.faces, b.faces)
    if same_topology:
        reference: trimesh.Trimesh = a
        reference_mesh = "A"
        delta = np.asarray(b.vertices) - np.asarray(a.vertices)
        displacement = np.linalg.norm(delta, axis=1)
        outward_all = np.einsum("ij,ij->i", delta, np.asarray(a.vertex_normals))
        vmask = region.vertex_mask(reference)
    else:
        # No vertex correspondence: measure each B vertex's distance to A's surface
        # (0 => that point is unchanged). Region membership is decided by the vertex's
        # FOOTPRINT — its closest point on A — so a vertex that moved out of a surface-
        # aligned region (e.g. a pocket floor) still belongs to that region.
        reference = b
        reference_mesh = "B"
        bverts = np.asarray(b.vertices)
        closest, displacement, _tri = trimesh.proximity.closest_point(a, bverts)
        if a.is_watertight:
            # signed_distance is +inside A; outward (material added) is the negative.
            outward_all = -np.asarray(trimesh.proximity.signed_distance(a, bverts))
        else:
            outward_all = displacement.copy()  # sign unknown
            caveats.append(
                "before-mesh is not watertight: signed emboss/pocket direction is "
                "unavailable, reporting unsigned displacement magnitudes"
            )
        if region.spatial:
            vmask = region.point_mask(closest)
        else:
            vmask = region.vertex_mask(reference)
            caveats.append(
                "region is index-based but topology differs: membership uses the after-mesh "
                "vertex indices, which do not correspond to the before-mesh; results are approximate"
            )
        caveats.append(
            "topology differs between A and B: displacement measured to A's surface and "
            "region membership by footprint; values are approximate"
        )

    changed = displacement > threshold
    feature_mask = vmask & changed
    inside = displacement[vmask]
    outside = displacement[~vmask]
    # 'changed_region_*' describes the change INSIDE the region (feature_mask), staying
    # consistent with signed_displacement, which is also computed over feature_mask.
    changed_pts = np.asarray(reference.vertices)[feature_mask]

    if changed_pts.size:
        bounds = Bounds(
            min=[float(x) for x in changed_pts.min(axis=0)],
            max=[float(x) for x in changed_pts.max(axis=0)],
        )
        centroid = [float(x) for x in changed_pts.mean(axis=0)]
    else:
        bounds, centroid = None, None

    signed = _signed_displacement(outward_all[feature_mask])
    if not same_topology and feature_mask.sum() == 0 and int(changed.sum()) > 0:
        # A change exists but none of it mapped into the region on differing topology
        # (e.g. a deep pocket whose floor is nearer A's opposite face). Report the signed
        # depth as indeterminate rather than a misleading 0.0.
        signed.peak = None
        caveats.append(
            "a change was detected but did not map into the region on differing topology; "
            "the signed height/depth is indeterminate — re-run with matching topology "
            "(edit vertices in place) to measure it"
        )

    return LocalizedChange(
        reference_mesh=reference_mesh,
        same_topology=same_topology,
        change_threshold=threshold,
        region_vertex_count=int(vmask.sum()),
        changed_vertex_count=int(feature_mask.sum()),  # changed AND inside the region
        inside_max_displacement=float(inside.max()) if inside.size else 0.0,
        outside_max_displacement=float(outside.max()) if outside.size else 0.0,
        changed_region_bounds=bounds,
        changed_region_centroid=centroid,
        signed_displacement=signed,
        caveats=caveats,
    )


def _heatmap_scalars(a: trimesh.Trimesh, b: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex distance from B's vertices to A's surface."""
    vertices = np.asarray(b.vertices)
    if len(vertices) <= HEATMAP_MAX_VERTICES:
        return trimesh.proximity.closest_point(a, vertices)[1]
    rng = np.random.default_rng(SEED)
    subset = np.sort(rng.choice(len(vertices), HEATMAP_MAX_VERTICES, replace=False))
    distances = trimesh.proximity.closest_point(a, vertices[subset])[1]
    _dist, nearest = cKDTree(vertices[subset]).query(vertices)
    return distances[nearest]


def compare(
    loaded_a: LoadedMesh, loaded_b: LoadedMesh, sample_count: int = 2000
) -> tuple[ComparisonReport, np.ndarray]:
    """Returns the report plus per-vertex displacement scalars for B (heatmap)."""
    a, b = loaded_a.combined, loaded_b.combined
    caveats: list[str] = []
    diagonal = float(np.linalg.norm(a.extents)) or 1.0

    same_topology = len(a.vertices) == len(b.vertices) and np.array_equal(a.faces, b.faces)
    if same_topology:
        method = "procrustes_exact"
        matrix, transformed, _cost = trimesh.registration.procrustes(
            a.vertices, b.vertices, reflection=False, translation=True,
            scale=True, return_cost=True,
        )
        residual = float(np.sqrt(np.mean(np.sum((transformed - b.vertices) ** 2, axis=1))))
    else:
        method = "icp"
        caveats.append(
            "topology differs between A and B; correspondence was estimated with ICP, "
            "so the detected transform is approximate"
        )
        np.random.seed(SEED)  # mesh_other samples internally; keep runs repeatable
        matrix, _cost = trimesh.registration.mesh_other(
            a, b, samples=500, scale=True, icp_first=10, icp_final=50
        )
        matrix = np.asarray(matrix, dtype=float)
        residual = _fit_residual(a, b, matrix, samples=min(sample_count, 2000))

    matrix = np.asarray(matrix, dtype=float)
    includes_reflection = float(np.linalg.det(matrix[:3, :3])) < 0
    if includes_reflection:
        dereflected = _try_dereflect(a, b, matrix, residual)
        if dereflected is not None:
            matrix, residual = dereflected
            includes_reflection = False
        else:
            caveats.append(
                "the registration includes a reflection (negative determinant): B may be "
                "a mirrored copy of A; rotation axis/angle refer to the closest proper "
                "rotation and are approximate"
            )
    scale, translation, angle_deg, axis = _decompose(matrix)
    classification = _classify(residual, scale, translation, angle_deg, diagonal, method)

    # Poor proper fit: test the mirror hypothesis explicitly. Mirroring is a
    # common CAD manipulation and ICP rarely finds reflected optima on its own.
    if method == "icp" and classification == "deformed" and not includes_reflection:
        mirror = _mirror_about_centroid(np.asarray(a.centroid, dtype=float), 0)
        a_mirrored = a.copy()
        a_mirrored.apply_transform(mirror)
        np.random.seed(SEED)
        retry, _cost = trimesh.registration.mesh_other(
            a_mirrored, b, samples=500, scale=True, icp_first=10, icp_final=50
        )
        candidate = np.asarray(retry, dtype=float) @ mirror
        candidate_residual = _fit_residual(a, b, candidate)
        if candidate_residual <= _THRESHOLDS["icp"]["residual"] * diagonal:
            matrix, residual, includes_reflection = candidate, candidate_residual, True
            scale, translation, angle_deg, axis = _decompose(matrix)
            caveats.append(
                "B only registers to A after a reflection: B appears to be a mirrored "
                "copy of A; rotation axis/angle refer to the closest proper rotation "
                "and are approximate"
            )

    if includes_reflection and residual <= _THRESHOLDS[method]["residual"] * diagonal:
        classification = "mirrored"

    samples = int(min(max(sample_count, 100), MAX_COMPARE_SAMPLES))
    points_a, _ = trimesh.sample.sample_surface(a, samples, seed=SEED)
    points_b, _ = trimesh.sample.sample_surface(b, samples, seed=SEED)
    dist_a_to_b = trimesh.proximity.closest_point(b, points_a)[1]
    dist_b_to_a = trimesh.proximity.closest_point(a, points_b)[1]
    aligned = trimesh.proximity.closest_point(b, _apply(matrix, points_a))[1]

    metrics_a = compute_metrics(loaded_a)
    metrics_b = compute_metrics(loaded_b)
    for prefix, metrics in (("A", metrics_a), ("B", metrics_b)):
        caveats.extend(f"{prefix}: {c}" for c in metrics.caveats)

    report = ComparisonReport(
        classification=classification,
        method=method,
        transform=TransformInfo(
            matrix_4x4=[[float(v) for v in row] for row in matrix],
            translation=[float(v) for v in translation],
            rotation_axis=[float(v) for v in axis],
            rotation_angle_deg=angle_deg,
            uniform_scale=scale,
            residual_rms=residual,
            includes_reflection=includes_reflection,
        ),
        distances=DistanceInfo(
            chamfer_mean=float((dist_a_to_b.mean() + dist_b_to_a.mean()) / 2.0),
            hausdorff_approx=float(max(dist_a_to_b.max(), dist_b_to_a.max())),
            aligned_residual_rms=float(np.sqrt(np.mean(aligned**2))),
            sample_count=samples,
            seed=SEED,
        ),
        metric_deltas={
            "volume": [metrics_a.volume, metrics_b.volume],
            "surface_area": [metrics_a.surface_area, metrics_b.surface_area],
            "extents": [metrics_a.extents, metrics_b.extents],
            "vertex_count": [metrics_a.vertex_count, metrics_b.vertex_count],
            "face_count": [metrics_a.face_count, metrics_b.face_count],
            "body_count": [metrics_a.body_count, metrics_b.body_count],
            "is_watertight": [metrics_a.is_watertight, metrics_b.is_watertight],
        },
        caveats=caveats,
        summary=(
            "NOTE: the best fit includes a reflection. "
            if includes_reflection and classification != "mirrored"
            else ""
        )
        + _summary(
            classification, method, scale, translation, angle_deg, axis, residual, diagonal
        ),
    )
    return report, _heatmap_scalars(a, b)
