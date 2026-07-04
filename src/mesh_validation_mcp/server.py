"""MCP server wiring — the only module that imports the mcp SDK, so a future
SDK major-version migration stays a one-file change."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field

from .comparison import _bounded_distances, compare, localized_change
from .config import DEFAULT_RESOLUTION, VALIDATE_RESOLUTION
from .golden import compare_to_reference
from .integrity import integrity_flags
from .loading import load_mesh
from .metrics import MeshMetrics, compute_metrics
from .oracles import PropertySpec, run_oracles
from .rendering import (
    DEFAULT_VIEWS,
    highlight_face_colors,
    render_views,
    scalars_to_face_colors,
)
from .validation import ChangeExpectations, Expectations, evaluate, evaluate_change

# Failed integrity checks whose offending faces we can highlight in the render.
_INTEGRITY_FLAG_KEYS = {
    "boundary_edge_count": "boundary",
    "non_manifold_edge_count": "non_manifold",
    "degenerate_face_count": "degenerate",
    "sliver_face_count": "sliver",
    "flipped_face_count": "flipped",
    "self_intersecting_face_count": "self_intersection",
}

INSTRUCTIONS = """Tools for validating 3D mesh files produced by mesh-manipulation code.
Typical agent loop: write code -> export the mesh -> validate_mesh with expectations ->
inspect the pass/fail report and the rendered views -> fix the code -> repeat.

- For LOCALIZED edits (emboss/pocket/fillet a selected region), export the mesh before AND
  after, then call compare_meshes with `localized` + a region: it verifies that only that
  region changed, that the rest is untouched, and the signed height/depth of the change.
- validate_mesh also checks mesh INTEGRITY (self-intersections, non-manifold/boundary edges,
  slivers, duplicate/flipped faces) — boolean-based edits often break these while still
  reporting watertight=true, which makes the volume check silently wrong.

Rules of thumb:
- All file paths must be ABSOLUTE; the server's working directory is not yours.
- All numeric values (volume, distances, ...) are in the file's native units.
- Renders are Z-up: 'front' looks along +Y, 'iso' views from the (+X,-Y,+Z) octant.
- Tool errors return a JSON envelope {"code", "message", "hint"} you can parse and act on.
"""

ViewName = Literal["iso", "iso_back", "front", "back", "left", "right", "top", "bottom"]

mcp = FastMCP("mesh-validator", instructions=INSTRUCTIONS)


def _round(value: Any) -> Any:
    """Recursively round floats to 6 significant digits and strip non-finite
    values so the JSON stays compact and always parseable."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(_round(payload), separators=(",", ":"), allow_nan=False)


def _mesh_summary(metrics: MeshMetrics) -> dict[str, Any]:
    return {
        "path": metrics.path,
        "format": metrics.format,
        "body_count": metrics.body_count,
        "vertex_count": metrics.vertex_count,
        "face_count": metrics.face_count,
        "is_watertight": metrics.is_watertight,
        "volume": metrics.volume,
        "surface_area": metrics.surface_area,
        "integrity": metrics.integrity.model_dump(),
        "cross_checks": metrics.cross_checks.model_dump(),
        "caveats": metrics.caveats,
    }


@mcp.tool()
def inspect_mesh(
    file_path: Annotated[
        str, Field(description="Absolute path to the mesh file (stl/obj/ply/glb/gltf/off/3mf)")
    ],
) -> MeshMetrics:
    """Load a 3D mesh file and return a full geometric report: vertex/face/edge/body
    counts, watertightness, winding consistency, euler number, volume (with a
    volume_reliable flag), surface area, bounds, extents, centroid, center of mass, a
    per-body breakdown, and an `integrity` block (boundary/non-manifold edges, degenerate/
    sliver/duplicate/flipped faces, self-intersections, min triangle quality). Use it to
    ground truth a mesh BEFORE writing validate_mesh expectations, or to diagnose a failed
    validation. Values are in the file's native units; `caveats` lists anything that makes
    a metric unreliable."""
    return compute_metrics(load_mesh(file_path))


@mcp.tool(structured_output=False)
def validate_mesh(
    file_path: Annotated[str, Field(description="Absolute path to the mesh file to validate")],
    expectations: Annotated[
        Expectations,
        Field(
            description="Expected properties. Only the keys you set are checked. Geometry: "
            "volume, surface_area, bbox_min/max/extents, centroid, vertex_count, face_count, "
            "watertight, winding_consistent, body_count, euler_number. Integrity (usually "
            "expected 0): non_manifold_edge_count, boundary_edge_count, self_intersecting_"
            "face_count, degenerate_face_count, sliver_face_count, duplicate_face_count, "
            "flipped_face_count, plus min_triangle_quality (a floor). Scalars/vectors accept a "
            'bare value (global tolerance, default 1% relative) or {"expected", "rel_tol", '
            '"abs_tol"}; counts accept an int or {"min", "max"}. Example: {"volume": 500, '
            '"watertight": true, "self_intersecting_face_count": 0, "bbox_extents": [10,10,5]}'
        ),
    ],
    include_render: Annotated[
        bool,
        Field(
            description="Append a 4-view render sheet (iso/front/top/right) so you can "
            "visually confirm the result. Set false in tight fix-iterate loops to save "
            "tokens; keep true for final verification."
        ),
    ] = True,
) -> list[str | Image]:
    """THE core validation tool. Checks a manipulated mesh file against your structured
    expectations (geometry AND integrity) and returns a deterministic pass/fail report per
    check (expected vs actual, deviation, tolerance) plus, by default, a multi-view render so
    you can visually confirm the manipulation looked right. When an integrity check fails, the
    offending faces are highlighted red in the render. Call inspect_mesh first if you are
    unsure what values to expect."""
    loaded = load_mesh(file_path)
    metrics = compute_metrics(loaded)
    report = evaluate(metrics, expectations)

    payload: dict[str, Any] = report.model_dump()
    payload["mesh"] = _mesh_summary(metrics)

    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]

    # If integrity checks failed, highlight the offending faces so the defect is visible.
    failed_flag_kinds = [
        _INTEGRITY_FLAG_KEYS[c.name]
        for c in report.checks
        if not c.passed and c.name in _INTEGRITY_FLAG_KEYS
    ]
    face_colors = None
    if failed_flag_kinds:
        flags = integrity_flags(loaded.combined)
        flagged = np.unique(np.concatenate([flags[k] for k in failed_flag_kinds]))
        face_colors = highlight_face_colors(loaded.combined, flagged)

    images, meta = render_views(
        loaded.combined,
        list(DEFAULT_VIEWS),
        resolution=VALIDATE_RESOLUTION,
        face_colors=face_colors,
    )
    payload["render"] = {"included": True, "defects_highlighted": bool(failed_flag_kinds), **meta}
    return [_json(payload), *(Image(data=img, format="png") for img in images)]


@mcp.tool(structured_output=False)
def render_mesh(
    file_path: Annotated[str, Field(description="Absolute path to the mesh file to render")],
    views: Annotated[
        list[ViewName] | None,
        Field(description="Views to render (default: iso, front, top, right). Max 6 when combined."),
    ] = None,
    style: Annotated[
        Literal["shaded", "shaded_edges", "wireframe"],
        Field(description="shaded_edges shows topology best; wireframe reveals internal structure"),
    ] = "shaded_edges",
    resolution: Annotated[
        int, Field(description="Long edge of the output image in pixels (256-1280)")
    ] = DEFAULT_RESOLUTION,
    combine: Annotated[
        bool,
        Field(description="true: one contact-sheet image; false: one image per view"),
    ] = True,
) -> list[str | Image]:
    """Render a mesh file from canonical camera views and return the image(s) for visual
    inspection, without running any checks. Each tile is labeled and carries an RGB axis
    gizmo (X=red, Y=green, Z=blue). Multi-body meshes get one tint per body so a wrong
    body count is visually obvious. Use validate_mesh instead when you also want
    numeric checks."""
    loaded = load_mesh(file_path)
    images, meta = render_views(
        loaded.combined,
        list(views) if views else list(DEFAULT_VIEWS),
        style=style,
        resolution=resolution,
        combine=combine,
    )
    return [_json(meta), *(Image(data=img, format="png") for img in images)]


@mcp.tool(structured_output=False)
def compare_meshes(
    file_a: Annotated[str, Field(description="Absolute path to the BEFORE mesh")],
    file_b: Annotated[str, Field(description="Absolute path to the AFTER mesh")],
    localized: Annotated[
        ChangeExpectations | None,
        Field(
            description="Optional: scope the comparison to a region for LOCALIZED edits "
            "(emboss/pocket/fillet a selected area). Provide a `region` (box/sphere/plane/"
            "vertex_ids/face_ids) to get inside-vs-outside displacement, the signed "
            "height/depth, and the changed-region bounds. Add assertions to verify it: "
            '{"region": {"kind": "box", "min": [-10,-10,0], "max": [10,10,6]}, '
            '"emboss_height": 3, "max_unchanged_deviation": 0.01}. Use max_unchanged_deviation '
            "to assert the rest of the mesh is untouched, emboss_height / pocket_depth for the "
            "signed feature size."
        ),
    ] = None,
    sample_count: Annotated[
        int, Field(description="Surface samples for distance metrics (100-20000)")
    ] = 2000,
    include_render: Annotated[
        bool, Field(description="Append a displacement heatmap render of B (viridis + colorbar)")
    ] = True,
) -> list[str | Image]:
    """Compare two mesh files (before vs after a manipulation). Detects the transform
    between them — translation vector, rotation axis+angle, uniform scale — via exact
    vertex correspondence when topology matches (or ICP otherwise), classifies the change
    (identical | translation | rotation | rigid | similarity | mirrored | deformed), and
    reports chamfer/Hausdorff distances plus metric deltas (volume, area, counts).

    For LOCALIZED edits (embossing/pocketing a selected region), pass `localized` with a
    region: the tool then reports how much moved inside the region, whether everything
    outside stayed put (untouched_region_max_deviation), and the signed height/depth
    (+ = material added, − = removed), and can assert all of these. The optional heatmap
    paints B by distance to A's surface."""
    loaded_a = load_mesh(file_a)
    loaded_b = load_mesh(file_b)
    report, heatmap = compare(loaded_a, loaded_b, sample_count)

    payload: dict[str, Any] = report.model_dump()
    payload["heatmap"] = {
        "min": float(heatmap.min()),
        "mean": float(heatmap.mean()),
        "max": float(heatmap.max()),
    }

    if localized is not None:
        change = localized_change(loaded_a, loaded_b, localized.region, localized.change_threshold)
        change_report = evaluate_change(change, localized)
        payload["localized"] = {
            "passed": change_report.passed,
            "summary": change_report.summary,
            "checks": [c.model_dump() for c in change_report.checks],
            "stats": change.model_dump(),
        }

    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]

    colors, colorbar = scalars_to_face_colors(
        loaded_b.combined, heatmap, label="distance from B to A's surface (file units)"
    )
    images, meta = render_views(
        loaded_b.combined,
        ["iso", "front"],
        resolution=DEFAULT_RESOLUTION,
        face_colors=colors,
        colorbar=colorbar,
    )
    payload["render"] = {"included": True, **meta}
    return [_json(payload), *(Image(data=img, format="png") for img in images)]


@mcp.tool(structured_output=False)
def assert_properties(
    file_path: Annotated[str, Field(description="Absolute path to the mesh to check")],
    properties: Annotated[
        list[PropertySpec],
        Field(
            description="Named invariants to assert. Unary (after-mesh only): "
            "preserves_watertight, non_self_intersecting. Binary (need reference_path): "
            "conserves_volume, preserves_genus, preserves_euler, no_new_defects, "
            "centroid_fixed, monotonic_offset (set direction: outward|inward), "
            "bounded_hausdorff (set max_distance). Each may carry a tolerance "
            '{relative, absolute}. Example: [{"name": "conserves_volume", "tolerance": '
            '{"relative": 0.001}}, {"name": "no_new_defects"}].'
        ),
    ],
    reference_path: Annotated[
        str | None,
        Field(description="Absolute path to the BEFORE mesh (required by binary oracles)"),
    ] = None,
    include_render: Annotated[
        bool, Field(description="Append a 4-view render sheet of the checked mesh")
    ] = True,
) -> list[str | Image]:
    """Assert named, operation-agnostic invariants about a manipulation: volume conserved,
    watertight preserved, no new defects introduced, centroid fixed, an offset moved the
    surface the right way, or the result stayed within a Hausdorff bound of a reference.
    Every result carries a confidence tier and fails closed on unreliable input. Provide
    reference_path for any before/after (binary) invariant."""
    loaded = load_mesh(file_path)
    metrics = compute_metrics(loaded)
    before_metrics = None
    hausdorff_upper: float | None = None
    if reference_path is not None:
        reference = load_mesh(reference_path)
        before_metrics = compute_metrics(reference)
        if any(p.name == "bounded_hausdorff" for p in properties):
            diagonal = float(np.linalg.norm(reference.combined.extents)) or 1.0
            hausdorff_upper = _bounded_distances(
                loaded.combined, reference.combined, 2000, diagonal
            )["hausdorff_upper"]

    report = run_oracles(metrics, properties, before_metrics, hausdorff_upper)
    payload: dict[str, Any] = report.model_dump()
    payload["mesh"] = _mesh_summary(metrics)

    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]

    # Highlight self-intersections when an integrity-related invariant failed.
    highlight_self_int = any(
        not c.passed and c.name in ("non_self_intersecting", "no_new_defects")
        for c in report.checks
    )
    face_colors = None
    if highlight_self_int:
        flagged = integrity_flags(loaded.combined)["self_intersection"]
        if len(flagged):
            face_colors = highlight_face_colors(loaded.combined, flagged)
    images, meta = render_views(
        loaded.combined, list(DEFAULT_VIEWS), resolution=VALIDATE_RESOLUTION, face_colors=face_colors
    )
    payload["render"] = {"included": True, **meta}
    return [_json(payload), *(Image(data=img, format="png") for img in images)]


@mcp.tool(structured_output=False)
def compare_to_golden(
    file_path: Annotated[str, Field(description="Absolute path to the mesh you produced")],
    reference_path: Annotated[
        str, Field(description="Absolute path to the golden / reference mesh")
    ],
    tolerance: Annotated[
        float,
        Field(
            description="Max allowed deviation in file units. On matching topology this is the "
            "per-vertex tolerance for an EXACT match; otherwise it bounds the surface distance."
        ),
    ] = 1e-4,
    include_render: Annotated[
        bool, Field(description="Append a 4-view render sheet of the produced mesh")
    ] = True,
) -> list[str | Image]:
    """Check whether a produced mesh matches a golden/reference mesh: "did I make the thing
    I intended?" On matching topology it reports an EXACT per-vertex match (ordering-
    independent); otherwise it bounds the surface distance to the reference. Returns
    matches/exact_match plus the vertex delta or the Hausdorff bracket."""
    produced = load_mesh(file_path)
    reference = load_mesh(reference_path)
    result = compare_to_reference(produced, reference, tolerance)

    payload: dict[str, Any] = result.model_dump()
    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]
    images, meta = render_views(
        produced.combined, list(DEFAULT_VIEWS), resolution=VALIDATE_RESOLUTION
    )
    payload["render"] = {"included": True, **meta}
    return [_json(payload), *(Image(data=img, format="png") for img in images)]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
