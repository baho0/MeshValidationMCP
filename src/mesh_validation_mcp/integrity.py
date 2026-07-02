"""Mesh integrity metrics beyond the single is_watertight bool.

Boolean-based edits (emboss/pocket/fillet via CSG) routinely leave self-intersections,
non-manifold edges, slivers and duplicate faces even when is_watertight reports True —
and the divergence-theorem volume then silently passes a broken mesh. These functions
localize and count those defects; they also return the offending face ids so the render
can highlight them.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel

from .config import SELF_INTERSECTION_MAX_FACES, SLIVER_QUALITY

# Faces flagged by these defects are painted red in the render overlay.
FLAG_KINDS = ("boundary", "non_manifold", "degenerate", "sliver", "flipped", "self_intersection")


class IntegrityMetrics(BaseModel):
    boundary_edge_count: int
    non_manifold_edge_count: int
    degenerate_face_count: int
    sliver_face_count: int
    min_triangle_quality: float
    duplicate_face_count: int
    unmerged_vertex_count: int
    unreferenced_vertex_count: int
    flipped_face_count: int
    self_intersecting_face_count: int
    self_intersection_checked: bool


def _edge_info(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique-edge use counts, per-row unique-edge index, face id per edge row).

    Each face contributes 3 rows to edges_sorted; `counts` is per UNIQUE undirected edge
    (so a boundary edge has count 1, a manifold edge 2, a non-manifold edge >2)."""
    edges = mesh.edges_sorted
    _uniq, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    face_of_edge = np.repeat(np.arange(len(mesh.faces)), 3)
    return counts, inverse, face_of_edge


def _triangle_quality(mesh: trimesh.Trimesh) -> np.ndarray:
    """Q = 4*sqrt(3)*area / sum(edge_len^2); equilateral -> 1, sliver/degenerate -> 0."""
    tri = np.asarray(mesh.triangles, dtype=float)
    edges = tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]]
    edge_sq = (edges**2).sum(axis=(1, 2))
    area = np.asarray(mesh.area_faces, dtype=float)
    quality = np.zeros(len(mesh.faces))
    nonzero = edge_sq > 0
    quality[nonzero] = 4.0 * np.sqrt(3.0) * area[nonzero] / edge_sq[nonzero]
    return quality


def _flipped_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Face ids whose winding disagrees with the consistent majority. A whole-mesh
    inversion (still consistent) yields none — it is caught by the negative-volume caveat."""
    fixed = mesh.copy()
    trimesh.repair.fix_winding(fixed)
    differs = ~(np.asarray(fixed.faces) == np.asarray(mesh.faces)).all(axis=1)
    ids = np.nonzero(differs)[0]
    # fix_winding flips the minority; if it flipped more than half, the "minority" is
    # actually the rest — report the smaller set.
    if len(ids) > len(mesh.faces) / 2:
        ids = np.nonzero(~differs)[0]
    return ids


def _self_intersecting_faces(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Face ids involved in a self-intersection, or None if skipped (too large).

    Broad-phase reuses the mesh's existing rtree AABB index (mesh.triangles_tree);
    adjacent (vertex-sharing) pairs are dropped; survivors get a Moller triangle-
    triangle test. Coplanar overlaps are not detected (documented limitation)."""
    faces = np.asarray(mesh.faces)
    if len(faces) > SELF_INTERSECTION_MAX_FACES:
        return None
    tri = np.asarray(mesh.triangles, dtype=float)
    eps = 1e-8 * (float(np.linalg.norm(mesh.extents)) or 1.0)
    tree = mesh.triangles_tree
    vertex_sets = [frozenset(f) for f in faces]
    flagged: set[int] = set()

    for i in range(len(faces)):
        lo = tri[i].min(axis=0)
        hi = tri[i].max(axis=0)
        for j in tree.intersection((lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])):
            if j <= i:
                continue
            if vertex_sets[i] & vertex_sets[j]:  # adjacent: shared edge/vertex is not a defect
                continue
            if _tri_tri_intersect(tri[i], tri[j], eps):
                flagged.add(i)
                flagged.add(j)
    return np.array(sorted(flagged), dtype=np.int64)


def _plane_interval(proj: np.ndarray, dist: np.ndarray) -> tuple[float, float] | None:
    """Parameter interval where a triangle (projected onto an axis) crosses a plane."""
    ts: list[float] = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = dist[i], dist[j]
        if di == 0.0:
            ts.append(float(proj[i]))
        if di * dj < 0.0:
            ts.append(float(proj[i] + (proj[j] - proj[i]) * di / (di - dj)))
    if len(ts) < 2:
        return None
    return min(ts), max(ts)


def _tri_tri_intersect(t1: np.ndarray, t2: np.ndarray, eps: float) -> bool:
    """Moller (1997) interval-overlap test. Coplanar pairs return False (approximate).

    Normals are unit-normalized so `du`/`dv` are true point-to-plane distances (length
    units, comparable to `eps`) and the parallel-plane test uses sin(theta) (dimensionless
    against a fixed tolerance) — otherwise the thresholds would be scale-dependent."""
    n1 = np.cross(t1[1] - t1[0], t1[2] - t1[0])
    n2 = np.cross(t2[1] - t2[0], t2[2] - t2[0])
    len1 = np.linalg.norm(n1)
    len2 = np.linalg.norm(n2)
    if len1 == 0.0 or len2 == 0.0:
        return False  # a degenerate triangle has no well-defined plane
    n1 /= len1
    n2 /= len2

    du = t2 @ n1 - n1.dot(t1[0])
    if np.all(du > eps) or np.all(du < -eps):
        return False
    dv = t1 @ n2 - n2.dot(t2[0])
    if np.all(dv > eps) or np.all(dv < -eps):
        return False

    direction = np.cross(n1, n2)  # magnitude == sin(angle between planes)
    if np.linalg.norm(direction) < 1e-8:
        return False  # coplanar/parallel: not handled

    axis = int(np.argmax(np.abs(direction)))
    i1 = _plane_interval(t1[:, axis], dv)
    i2 = _plane_interval(t2[:, axis], du)
    if i1 is None or i2 is None:
        return False
    return not (i1[1] < i2[0] - eps or i2[1] < i1[0] - eps)


def integrity_flags(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    """Map each defect kind to the face ids exhibiting it (for the render overlay)."""
    counts, inverse, face_of_edge = _edge_info(mesh)
    row_mult = counts[inverse]
    quality = _triangle_quality(mesh)
    degenerate = quality <= 0.0
    self_int = _self_intersecting_faces(mesh)
    return {
        "boundary": np.unique(face_of_edge[row_mult == 1]),
        "non_manifold": np.unique(face_of_edge[row_mult > 2]),
        "degenerate": np.nonzero(degenerate)[0],
        "sliver": np.nonzero((quality < SLIVER_QUALITY) & ~degenerate)[0],
        "flipped": _flipped_faces(mesh),
        "self_intersection": self_int if self_int is not None else np.array([], dtype=np.int64),
    }


def compute_integrity(mesh: trimesh.Trimesh) -> IntegrityMetrics:
    counts, _inverse, _face_of_edge = _edge_info(mesh)
    quality = _triangle_quality(mesh)
    degenerate = quality <= 0.0
    non_degenerate_quality = quality[~degenerate]

    # Detect geometric (coincident) duplicates: merge coincident vertices first so a
    # doubled surface with distinct vertex indices is recognized as duplicate faces.
    merged = mesh.copy()
    merged.merge_vertices()

    self_int = _self_intersecting_faces(mesh)

    return IntegrityMetrics(
        boundary_edge_count=int((counts == 1).sum()),
        non_manifold_edge_count=int((counts > 2).sum()),
        degenerate_face_count=int(degenerate.sum()),
        sliver_face_count=int(((quality < SLIVER_QUALITY) & ~degenerate).sum()),
        min_triangle_quality=(
            float(non_degenerate_quality.min()) if non_degenerate_quality.size else 0.0
        ),
        duplicate_face_count=int(len(merged.faces) - int(merged.unique_faces().sum())),
        unmerged_vertex_count=int(len(mesh.vertices) - len(merged.vertices)),
        unreferenced_vertex_count=int(len(mesh.vertices) - int(mesh.referenced_vertices.sum())),
        flipped_face_count=int(len(_flipped_faces(mesh))),
        self_intersecting_face_count=0 if self_int is None else int(len(self_int)),
        self_intersection_checked=self_int is not None,
    )
