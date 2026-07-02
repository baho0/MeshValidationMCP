"""MCP server wiring — the only module that imports the mcp SDK, so a future
SDK major-version migration stays a one-file change."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field

from .comparison import compare
from .config import DEFAULT_RESOLUTION, VALIDATE_RESOLUTION
from .loading import load_mesh
from .metrics import MeshMetrics, compute_metrics
from .rendering import DEFAULT_VIEWS, render_views, scalars_to_face_colors
from .validation import Expectations, evaluate

INSTRUCTIONS = """Tools for validating 3D mesh files produced by mesh-manipulation code.
Typical agent loop: write code -> export the mesh -> validate_mesh with expectations ->
inspect the pass/fail report and the rendered views -> fix the code -> repeat.

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
    volume_reliable flag), surface area, bounds, extents, centroid, center of mass and a
    per-body breakdown. Use it to ground truth a mesh BEFORE writing validate_mesh
    expectations, or to diagnose a failed validation. Values are in the file's native
    units; `caveats` lists anything that makes a metric unreliable."""
    return compute_metrics(load_mesh(file_path))


@mcp.tool(structured_output=False)
def validate_mesh(
    file_path: Annotated[str, Field(description="Absolute path to the mesh file to validate")],
    expectations: Annotated[
        Expectations,
        Field(
            description="Expected properties. Only the keys you set are checked. Scalars/"
            "vectors accept a bare value (global tolerance, default 1% relative) or "
            '{"expected": ..., "rel_tol": ..., "abs_tol": ...}. Counts accept an int or '
            '{"min": ..., "max": ...}. Example: {"volume": 500, "watertight": true, '
            '"bbox_extents": [10, 10, 5], "tolerance": {"relative": 0.02}}'
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
    expectations and returns a deterministic pass/fail report per check (expected vs
    actual, deviation, tolerance) plus, by default, a multi-view render so you can
    visually confirm the manipulation looked right. Call inspect_mesh first if you are
    unsure what values to expect."""
    loaded = load_mesh(file_path)
    metrics = compute_metrics(loaded)
    report = evaluate(metrics, expectations)

    payload: dict[str, Any] = report.model_dump()
    payload["mesh"] = _mesh_summary(metrics)

    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]

    images, meta = render_views(
        loaded.combined, list(DEFAULT_VIEWS), resolution=VALIDATE_RESOLUTION
    )
    payload["render"] = {"included": True, **meta}
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
    (identical | translation | rotation | rigid | similarity | deformed), and reports
    chamfer/Hausdorff distances plus metric deltas (volume, area, counts). Perfect for
    validating transform code: apply your transform, then check the detected one matches
    what you intended. The optional heatmap paints B by distance to A's surface."""
    loaded_a = load_mesh(file_a)
    loaded_b = load_mesh(file_b)
    report, heatmap = compare(loaded_a, loaded_b, sample_count)

    payload: dict[str, Any] = report.model_dump()
    payload["heatmap"] = {
        "min": float(heatmap.min()),
        "mean": float(heatmap.mean()),
        "max": float(heatmap.max()),
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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
