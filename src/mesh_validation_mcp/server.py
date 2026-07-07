"""MCP server wiring — the only module that imports the mcp SDK, so a future
SDK major-version migration stays a one-file change."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

import numpy as np
import trimesh
from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field

from .array_validate import ArrayPattern, validate_array as _validate_array
from .boolean_validate import BooleanExpectations, validate_boolean as _validate_boolean
from .clearance import check_clearance as _check_clearance
from .dfm import DfmExpectations, dfm_report as _dfm_report
from .silhouette import SilhouetteComparison, silhouette_compare as _silhouette_compare
from .units import UnitsReport, units_report as _units_report
from .comparison import (
    _bounded_distances,
    compare,
    localized_change,
    signed_field_and_scalars,
)
from .generative_validate import (
    ExtrudeExpectations,
    RevolveExpectations,
    validate_extrude as _validate_extrude,
    validate_revolve as _validate_revolve,
)
from .remesh_validate import RemeshExpectations, validate_remesh as _validate_remesh
from .symmetry import detect_symmetry as _detect_symmetry
from .config import DEFAULT_RESOLUTION, VALIDATE_RESOLUTION
from .errors import ErrorCode, MeshToolError
from .features import (
    draft_analysis,
    draft_face_scalars,
    fit_region,
    thickness_vertex_scalars,
    wall_thickness,
)
from .golden import compare_to_reference
from .integrity import integrity_flags
from .loading import load_mesh
from .metrics import MeshMetrics, compute_metrics
from .oracles import PropertySpec, run_oracles
from .region import Region
from .section import inspect_section as _inspect_section, section_cut_faces
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

Beyond the core loop, tools cover specific manipulation families. Each result carries a
`confidence` tier (exact | topological | sampled | estimated) and an error bound, and every
check fails CLOSED on unreliable input (a non-watertight/self-intersecting volume never
silently PASSes):
- assert_properties: named before/after invariants (conserves_volume, preserves_watertight/
  genus, no_new_defects, centroid_fixed, monotonic_offset, bounded_hausdorff).
- compare_to_golden: does the output match a reference mesh (exact per-vertex or bounded surface)?
- inspect_section: cross-section area/perimeter/loops (exact); measure_thickness (min wall);
  analyze_draft (undercuts); fit_feature (fillet/bore radius, chamfer plane via primitive fit).
- validate_boolean: union/difference/intersection volume bounds + containment + seam integrity.
- detect_symmetry; validate_array (linear/polar); validate_generative (extrude/revolve by
  volume signature); measure_displacement (whole-mesh deformation field); validate_remesh.
- check_clearance (assembly fit); compare_silhouette (outline preserved); units_sanity; validate_dfm.

Most validators also return an inspectable render (a "photo"), and the key ones paint the
checked quantity onto it — measure_displacement is a SIGNED heatmap (red = material added,
blue = removed), measure_thickness a thin-wall heatmap, analyze_draft an undercut heatmap,
inspect_section highlights the cut band, fit_feature the fitted region. Set include_render
false in tight loops to save tokens.

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


def _with_render(
    payload: dict[str, Any],
    mesh: Any,
    include_render: bool,
    *,
    face_colors: Any = None,
    colorbar: Any = None,
    views: tuple[str, ...] = DEFAULT_VIEWS,
) -> list[str | Image]:
    """Attach a render (optionally with a scalar/highlight overlay) to a JSON payload so every
    validation can return a 'photo' the agent can inspect visually."""
    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]
    images, meta = render_views(
        mesh, list(views), resolution=VALIDATE_RESOLUTION, face_colors=face_colors, colorbar=colorbar
    )
    payload["render"] = {"included": True, **meta}
    return [_json(payload), *(Image(data=img, format="png") for img in images)]


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
        "topology": {
            "genus_total": metrics.topology.genus_total,
            "boundary_loop_count": metrics.topology.boundary_loop_count,
        },
        "curvature": {
            "sharp_edge_count": metrics.curvature.sharp_edge_count,
            "max_dihedral_deg": metrics.curvature.max_dihedral_deg,
        },
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
def inspect_section(
    file_path: Annotated[str, Field(description="Absolute path to the mesh file")],
    plane_origin: Annotated[
        list[float], Field(description="A point on the cutting plane [x,y,z]", min_length=3, max_length=3)
    ],
    plane_normal: Annotated[
        list[float],
        Field(description="The cutting plane normal [x,y,z] (need not be unit)", min_length=3, max_length=3),
    ],
    include_render: Annotated[
        bool, Field(description="Append a render with the cut band highlighted")
    ] = True,
) -> list[str | Image]:
    """Slice the mesh with a plane and measure the resulting 2D profile: the number of loops,
    each loop's perimeter and area, and the net cross-section area (holes subtracted). Use it
    to check an extrusion's constant cross-section, a prism/cylinder's analytic area, or that
    a bore/pocket produced the intended profile. Area/perimeter are computed exactly from the
    section polylines (no sampling); the render highlights where the plane cuts."""
    loaded = load_mesh(file_path)
    info = _inspect_section(loaded, plane_origin, plane_normal)
    payload: dict[str, Any] = info.model_dump()
    face_colors = None
    if include_render and info.intersects:
        cut = section_cut_faces(loaded.combined, plane_origin, plane_normal)
        if len(cut):
            face_colors = highlight_face_colors(loaded.combined, cut)
    return _with_render(payload, loaded.combined, include_render, face_colors=face_colors)


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


@mcp.tool(structured_output=False)
def measure_thickness(
    file_path: Annotated[str, Field(description="Absolute path to a watertight mesh")],
    sample_count: Annotated[int, Field(description="Surface samples (500-8000)")] = 2000,
    include_render: Annotated[
        bool, Field(description="Append a thickness heatmap (thin walls in red)")
    ] = True,
) -> list[str | Image]:
    """Measure wall/feature thickness by inscribing the largest interior sphere at surface
    samples. Reports min/p5/median/mean/max and the thinnest point; the render paints a
    thickness heatmap so thin walls are visible (red = thin). Use p5_thickness as the robust
    thin-wall indicator (the bare min is biased low near sharp convex edges). Watertight only."""
    loaded = load_mesh(file_path)
    info = wall_thickness(loaded, sample_count)
    payload: dict[str, Any] = info.model_dump()
    face_colors = colorbar = None
    if include_render:
        scalars = thickness_vertex_scalars(loaded)
        face_colors, colorbar = scalars_to_face_colors(
            loaded.combined, scalars, label="wall thickness (red = thin)", cmap="RdYlGn"
        )
    return _with_render(payload, loaded.combined, include_render, face_colors=face_colors, colorbar=colorbar)


@mcp.tool(structured_output=False)
def analyze_draft(
    file_path: Annotated[str, Field(description="Absolute path to the mesh")],
    pull_direction: Annotated[
        list[float],
        Field(description="Mold pull / de-mold direction [x,y,z]", min_length=3, max_length=3),
    ],
    min_draft_deg: Annotated[
        float | None,
        Field(description="If set, also report the pullable area below this draft angle"),
    ] = None,
    include_render: Annotated[
        bool, Field(description="Append a draft heatmap (undercuts in red)")
    ] = True,
) -> list[str | Image]:
    """Area-weighted draft/undercut analysis for a pull direction: the minimum draft angle
    over pullable faces, and the total area (and face count) of undercuts — faces whose normal
    points against the pull and so cannot be released. Draft is 90deg minus the angle between a
    face normal and the pull direction (0 = vertical wall). The render paints draft per face so
    undercuts (red) are visible."""
    loaded = load_mesh(file_path)
    info = draft_analysis(loaded, pull_direction, min_draft_deg)
    payload: dict[str, Any] = info.model_dump()
    face_colors = colorbar = None
    if include_render:
        scalars = draft_face_scalars(loaded, pull_direction)
        face_colors, colorbar = scalars_to_face_colors(
            loaded.combined, scalars, label="draft angle deg (red = undercut/low)",
            cmap="RdYlGn", per_face=True,
        )
    return _with_render(payload, loaded.combined, include_render, face_colors=face_colors, colorbar=colorbar)


@mcp.tool(structured_output=False)
def fit_feature(
    file_path: Annotated[str, Field(description="Absolute path to the mesh")],
    region: Annotated[
        Region,
        Field(
            description="The feature faces/vertices to fit (box/sphere/plane/vertex_ids/"
            "face_ids region). e.g. a fillet band, a bore wall, or a chamfer face."
        ),
    ],
    kind: Annotated[
        Literal["plane", "sphere", "cylinder"],
        Field(description="cylinder for a fillet/bore radius, plane for a chamfer, sphere for a dome"),
    ],
    include_render: Annotated[
        bool, Field(description="Append a render with the fitted feature region highlighted")
    ] = True,
) -> list[str | Image]:
    """Fit a primitive (plane/sphere/cylinder) to the vertices a region selects and report
    its parameters plus the RMS residual (the fit quality). This is how you measure a fillet
    or bore radius, or confirm a chamfer is planar: select the feature with a region, fit a
    cylinder/plane, and read radius/normal + residual. The render highlights the region so you
    can confirm you selected the right feature."""
    loaded = load_mesh(file_path)
    fit = fit_region(loaded, region, kind)
    payload: dict[str, Any] = fit.model_dump()
    face_colors = None
    if include_render:
        face_ids = np.nonzero(region.face_mask(loaded.combined))[0]
        face_colors = highlight_face_colors(loaded.combined, face_ids)
    return _with_render(payload, loaded.combined, include_render, face_colors=face_colors)


@mcp.tool(structured_output=False)
def validate_boolean(
    file_a: Annotated[str, Field(description="Absolute path to operand A")],
    file_b: Annotated[str, Field(description="Absolute path to operand B")],
    file_result: Annotated[str, Field(description="Absolute path to the boolean result mesh")],
    operation: Annotated[
        Literal["union", "difference", "intersection"],
        Field(description="The boolean operation that produced the result (difference = A - B)"),
    ],
    tolerance: Annotated[
        float, Field(description="Relative slack on the volume bounds (default 0.02)")
    ] = 0.02,
    include_render: Annotated[bool, Field(description="Append a render of the result")] = True,
) -> list[str | Image]:
    """Validate a boolean/CSG result against its operands: the result's integrity (watertight,
    non-self-intersecting), the volume bounds the operation must satisfy (union:
    max(Va,Vb)<=Vr<=Va+Vb; difference: Va-Vb<=Vr<=Va; intersection: Vr<=min(Va,Vb)), and
    signed-distance containment (union contains both operands; a difference stays inside A and
    clear of B). Catches the common failure where a seam self-intersects but reads watertight."""
    loaded_result = load_mesh(file_result)
    report = _validate_boolean(
        load_mesh(file_a), load_mesh(file_b), loaded_result,
        BooleanExpectations(operation=operation, tolerance=tolerance),
    )
    payload: dict[str, Any] = report.model_dump()
    payload["result_mesh"] = _mesh_summary(compute_metrics(loaded_result))
    if not include_render:
        payload["render"] = {"included": False}
        return [_json(payload)]
    images, meta = render_views(
        loaded_result.combined, list(DEFAULT_VIEWS), resolution=VALIDATE_RESOLUTION
    )
    payload["render"] = {"included": True, **meta}
    return [_json(payload), *(Image(data=img, format="png") for img in images)]


@mcp.tool(structured_output=False)
def detect_symmetry(
    file_path: Annotated[str, Field(description="Absolute path to the mesh")],
    rel_tolerance: Annotated[
        float, Field(description="Symmetry tolerance as a fraction of the bbox diagonal")
    ] = 1e-3,
    include_render: Annotated[bool, Field(description="Append a render of the part")] = True,
) -> list[str | Image]:
    """Detect mirror and rotational self-symmetry. Reports the mirror planes (from the
    principal inertia axes, each confirmed by reflecting and measuring the surface distance)
    and the largest rotational fold about a principal axis. Use it to verify a part that
    should be symmetric actually is."""
    loaded = load_mesh(file_path)
    info = _detect_symmetry(loaded, rel_tolerance)
    return _with_render(info.model_dump(), loaded.combined, include_render)


@mcp.tool(structured_output=False)
def measure_displacement(
    file_a: Annotated[str, Field(description="Absolute path to the BEFORE mesh")],
    file_b: Annotated[str, Field(description="Absolute path to the AFTER mesh")],
    include_render: Annotated[
        bool, Field(description="Append a signed displacement heatmap of the after-mesh")
    ] = True,
) -> list[str | Image]:
    """Whole-mesh signed displacement field between before and after (the region-free
    deformation measure). Reports max/mean displacement, the signed peak along the surface
    normal (+ outward), the outward fraction, a direction-consistency score (1 = a coherent
    offset, ~0 = tangential/twist), and the volume/area deltas. The render is a SIGNED heatmap
    of the after-mesh — red = material added (emboss/outward), blue = removed (pocket/inward) —
    so you can see exactly where and which way the surface moved."""
    la, lb = load_mesh(file_a), load_mesh(file_b)
    field, outward = signed_field_and_scalars(la, lb)
    payload: dict[str, Any] = field.model_dump()
    face_colors = colorbar = None
    if include_render:
        face_colors, colorbar = scalars_to_face_colors(
            lb.combined, outward,
            label="signed displacement (red = added/outward, blue = removed/inward)",
            cmap="RdBu_r", symmetric=True,
        )
    return _with_render(payload, lb.combined, include_render, face_colors=face_colors, colorbar=colorbar)


@mcp.tool(structured_output=False)
def validate_array(
    file_result: Annotated[str, Field(description="Absolute path to the arrayed result mesh")],
    file_base: Annotated[str, Field(description="Absolute path to the single base instance")],
    pattern: Annotated[
        ArrayPattern,
        Field(
            description="The expected pattern. Linear: {kind:'linear', count, step:[x,y,z]}. "
            "Polar: {kind:'polar', count, axis:[x,y,z], center:[x,y,z], angle_deg}."
        ),
    ],
    include_render: Annotated[bool, Field(description="Append a render of the arrayed result")] = True,
) -> list[str | Image]:
    """Validate a linear or polar array: the instance count, that every instance is congruent
    to the base (checked with the rigid invariants volume + principal inertia, no registration
    needed), and that the instances sit at the predicted grid/fan positions."""
    loaded_result = load_mesh(file_result)
    report = _validate_array(loaded_result, load_mesh(file_base), pattern)
    return _with_render(report.model_dump(), loaded_result.combined, include_render)


@mcp.tool(structured_output=False)
def validate_generative(
    file_path: Annotated[str, Field(description="Absolute path to the generated solid")],
    operation: Annotated[
        Literal["extrude", "revolve"], Field(description="The generative operation")
    ],
    profile_area: Annotated[float, Field(description="Area of the 2D profile", gt=0)],
    height: Annotated[
        float | None, Field(description="Extrude only: extrusion height")
    ] = None,
    profile_centroid_radius: Annotated[
        float | None,
        Field(description="Revolve only: distance of the profile centroid from the axis"),
    ] = None,
    axis: Annotated[
        list[float], Field(description="Extrude axis [x,y,z]", min_length=3, max_length=3)
    ] = [0.0, 0.0, 1.0],
    include_render: Annotated[bool, Field(description="Append a render of the result")] = True,
) -> list[str | Image]:
    """Validate an extrude or revolve by its exact volumetric signature. Extrude: volume =
    profile_area x height and a constant cross-section along the axis. Revolve: volume =
    2*pi*profile_centroid_radius*profile_area (Pappus). Volume checks fail closed on a
    non-watertight result."""
    loaded = load_mesh(file_path)
    if operation == "extrude":
        if height is None:
            raise MeshToolError(
                ErrorCode.INVALID_EXPECTATION, "extrude needs 'height'", "Provide the extrusion height."
            )
        report = _validate_extrude(
            loaded, ExtrudeExpectations(profile_area=profile_area, height=height, axis=axis)
        )
    else:
        if profile_centroid_radius is None:
            raise MeshToolError(
                ErrorCode.INVALID_EXPECTATION,
                "revolve needs 'profile_centroid_radius'",
                "Provide the profile centroid's distance from the revolve axis.",
            )
        report = _validate_revolve(
            loaded,
            RevolveExpectations(
                profile_area=profile_area, profile_centroid_radius=profile_centroid_radius
            ),
        )
    return _with_render(report.model_dump(), loaded.combined, include_render)


@mcp.tool(structured_output=False)
def validate_remesh(
    file_before: Annotated[str, Field(description="Absolute path to the original mesh")],
    file_after: Annotated[str, Field(description="Absolute path to the remeshed mesh")],
    max_deviation: Annotated[
        float, Field(description="Max allowed surface distance to the original", gt=0)
    ],
    preserve_topology: Annotated[
        bool, Field(description="Require watertightness and genus to be unchanged")
    ] = True,
    min_triangle_quality: Annotated[
        float | None, Field(description="Optional floor on triangle quality (0=sliver, 1=equilateral)")
    ] = None,
    include_render: Annotated[bool, Field(description="Append a render of the remeshed result")] = True,
) -> list[str | Image]:
    """Validate a remesh/simplify/subdivide: the surface stayed within max_deviation of the
    original (bounded vertex-to-surface distance), the topology (watertight + genus) is
    unchanged, and triangle quality holds above an optional floor. One verdict for
    'retessellation didn't change the object'."""
    loaded_after = load_mesh(file_after)
    report = _validate_remesh(
        load_mesh(file_before),
        loaded_after,
        RemeshExpectations(
            max_deviation=max_deviation,
            preserve_topology=preserve_topology,
            min_triangle_quality=min_triangle_quality,
        ),
    )
    return _with_render(report.model_dump(), loaded_after.combined, include_render)


@mcp.tool(structured_output=False)
def check_clearance(
    file_a: Annotated[str, Field(description="Absolute path to part A")],
    file_b: Annotated[str, Field(description="Absolute path to part B")],
    min_clearance: Annotated[
        float | None, Field(description="If set, require at least this gap between the parts")
    ] = None,
    include_render: Annotated[
        bool, Field(description="Append a render of both parts together (one tint each)")
    ] = True,
) -> list[str | Image]:
    """Check whether two parts interfere and, if not, how close they come. Reports interference
    (with penetration depth for watertight parts) and the minimum surface-to-surface clearance.
    Use it to verify an assembly fit or a required gap. Interference and clearance are sampled;
    the render shows both parts together (one tint each) so the fit is visible."""
    la, lb = load_mesh(file_a), load_mesh(file_b)
    info = _check_clearance(la, lb, min_clearance)
    payload: dict[str, Any] = info.model_dump()
    both = trimesh.util.concatenate([la.combined, lb.combined]) if include_render else None
    return _with_render(payload, both, include_render)


@mcp.tool()
def compare_silhouette(
    file_a: Annotated[str, Field(description="Absolute path to the BEFORE mesh")],
    file_b: Annotated[str, Field(description="Absolute path to the AFTER mesh")],
    view_axis: Annotated[
        Literal["x", "y", "z"], Field(description="Orthographic view direction")
    ] = "z",
    resolution: Annotated[int, Field(description="Raster resolution in pixels")] = 256,
) -> SilhouetteComparison:
    """Compare the orthographic silhouettes (outlines) of two meshes from a canonical direction:
    intersection-over-union plus one-way coverage. Use it to confirm an edit did NOT change the
    profile (an emboss/texture pass should keep IoU ~1). Coarse and view-dependent."""
    return _silhouette_compare(load_mesh(file_a), load_mesh(file_b), view_axis, resolution)


@mcp.tool()
def units_sanity(
    file_path: Annotated[str, Field(description="Absolute path to the mesh")],
    expected_units: Annotated[
        str | None, Field(description="Units the file should declare (e.g. 'mm')")
    ] = None,
    plausible_min_diagonal: Annotated[
        float | None, Field(description="Smallest plausible bbox diagonal")
    ] = None,
    plausible_max_diagonal: Annotated[
        float | None, Field(description="Largest plausible bbox diagonal")
    ] = None,
) -> UnitsReport:
    """Sanity-check units and scale, and probe sampling self-consistency. Confirms the declared
    units and that the overall size is plausible (catching unit mix-ups), and recomputes a
    sampled probe at a second seed to confirm the sampled metrics are stable and deterministic."""
    rng = None
    if plausible_min_diagonal is not None and plausible_max_diagonal is not None:
        rng = (plausible_min_diagonal, plausible_max_diagonal)
    return _units_report(load_mesh(file_path), expected_units, rng)


@mcp.tool(structured_output=False)
def validate_dfm(
    file_path: Annotated[str, Field(description="Absolute path to the part")],
    min_wall_thickness: Annotated[
        float | None, Field(description="Minimum allowed wall thickness")
    ] = None,
    pull_direction: Annotated[
        list[float] | None,
        Field(description="Mold pull direction [x,y,z] for the draft/undercut check", min_length=3, max_length=3),
    ] = None,
    min_draft_deg: Annotated[float, Field(description="Required draft angle for pullable faces")] = 0.0,
    max_undercut_area: Annotated[float, Field(description="Allowed undercut area")] = 0.0,
    check_trapped_voids: Annotated[
        bool, Field(description="Flag fully enclosed internal cavities")
    ] = True,
    include_render: Annotated[bool, Field(description="Append a render of the part")] = True,
) -> list[str | Image]:
    """Pure-geometry design-for-manufacturing check: composes minimum wall thickness, undercut
    area / draft angle for a pull direction, and a trapped-void (enclosed cavity) test into one
    verdict. This is geometric only (no materials/process model). Wall thickness uses the robust
    p5 indicator, which is conservative near sharp edges."""
    exp = DfmExpectations(
        min_wall_thickness=min_wall_thickness,
        pull_direction=pull_direction,
        min_draft_deg=min_draft_deg,
        max_undercut_area=max_undercut_area,
        check_trapped_voids=check_trapped_voids,
    )
    loaded = load_mesh(file_path)
    report = _dfm_report(loaded, exp)
    return _with_render(report.model_dump(), loaded.combined, include_render)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
