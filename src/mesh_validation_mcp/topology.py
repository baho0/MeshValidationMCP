"""Topology invariants beyond the raw Euler number: per-body genus and boundary-loop count.

Genus (the number of handles/through-holes) and the number of open boundary loops are the
coarse topological fingerprints an operation must usually preserve: a fillet or a chamfer
must not punch a new hole; a repair must close boundary loops without changing the genus.
These are integer invariants — exact when their preconditions (a closed, orientable body)
hold — so they carry the ``topological`` confidence tier.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh
from pydantic import BaseModel

from .confidence import Confidence, topological
from .loading import LoadedMesh


class TopologyInfo(BaseModel):
    body_count: int
    euler_number: int
    genus_total: float | None  # sum over closed bodies (None if any body is open)
    per_body_genus: list[float | None]
    boundary_loop_count: int
    all_bodies_watertight: bool
    confidence: Confidence
    caveats: list[str]


def _body_genus(body: trimesh.Trimesh) -> float:
    """Genus of a closed body: g = (2 - chi) / 2 for a closed orientable surface."""
    return (2 - int(body.euler_number)) / 2.0


def _boundary_loop_count(mesh: trimesh.Trimesh) -> int:
    """Number of distinct open boundary loops (connected components of the boundary edges)."""
    edges = np.asarray(mesh.edges_sorted)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return 0
    graph = nx.Graph()
    graph.add_edges_from(boundary.tolist())
    return nx.number_connected_components(graph)


def analyze_topology(loaded: LoadedMesh) -> TopologyInfo:
    mesh = loaded.combined
    bodies = loaded.bodies
    # Gate genus on the COMBINED mesh's watertightness: split(only_watertight=False) silently
    # repairs holes, so a per-body test would call an open mesh closed. Genus is only defined
    # when the whole mesh is a closed orientable surface.
    all_closed = bool(mesh.is_watertight)
    if all_closed:
        per_body: list[float | None] = [_body_genus(b) for b in bodies]
        genus_total: float | None = float(sum(g for g in per_body if g is not None))
    else:
        per_body = [None] * len(bodies)
        genus_total = None

    caveats: list[str] = []
    if not all_closed:
        caveats.append("mesh is not watertight: genus is undefined")

    return TopologyInfo(
        body_count=len(bodies),
        euler_number=int(mesh.euler_number),
        genus_total=genus_total,
        per_body_genus=per_body,
        boundary_loop_count=_boundary_loop_count(mesh),
        all_bodies_watertight=all_closed,
        confidence=topological("genus from Euler characteristic; boundary loops from edge graph"),
        caveats=caveats,
    )
