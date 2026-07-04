# MeshValidationMCP

An [MCP](https://modelcontextprotocol.io) server for validating 3D mesh manipulations.
It combines deterministic geometric checks with multi-view rendered feedback, giving
coding agents a reliable way to verify their own mesh-processing code.

## Overview

Testing geometry code is hard: unit tests may pass while the result is visibly wrong.
This server closes that gap. An agent writes mesh-manipulation code, exports the result,
and calls `validate_mesh` with the file path and its expected properties. The server
returns a per-check pass/fail report together with rendered views, so the agent can
confirm the result both numerically and visually — and fix its own mistakes before
reporting back.

```
write code → export mesh → validate_mesh → inspect report + renders → fix → repeat
```

## Features

- **Deterministic validation** — volume, surface area, bounding box, centroid,
  vertex/face counts, watertightness, body count and Euler number, with global and
  per-check tolerances
- **Localized-change validation** — for edits confined to a selected region (emboss,
  pocket, fillet), verify that *only* that region changed, that the rest of the mesh is
  untouched, and the *signed* height/depth of the change (+ material added, − removed)
- **Mesh integrity checks** — self-intersections, non-manifold and boundary edges,
  degenerate/sliver/duplicate/flipped faces, and triangle quality; defects that
  `is_watertight` alone would hide (and that silently corrupt the volume)
- **Visual feedback** — labeled multi-view contact sheets with axis gizmos, returned as
  images; failed integrity checks highlight the offending faces in red
- **Transform detection** — compares before/after meshes and classifies the change
  (`identical | translation | rotation | rigid | similarity | mirrored | deformed`),
  reporting translation vector, rotation axis/angle and uniform scale
- **Distance metrics** — sampled chamfer and Hausdorff distances, plus a displacement
  heatmap render
- **Confidence tiers & fail-closed checks** — every check and distance is tagged with how
  it was derived (`exact` | `topological` | `sampled` | `estimated`) and a numeric error
  bound; a volume that cannot be trusted (non-watertight, winding-inconsistent, or
  self-intersecting mesh) *fails* rather than silently passing
- **Property oracles** — named before/after invariants any operation can assert
  (`conserves_volume`, `preserves_watertight`/`genus`, `no_new_defects`, `centroid_fixed`,
  `monotonic_offset`, `bounded_hausdorff`) and golden/reference matching
- **Feature & operation validators** — cross-sections, wall thickness, draft/undercut,
  fillet/bore fitting, boolean/CSG volume-bound + containment checks, symmetry, linear/polar
  arrays, extrude/revolve (by volumetric signature), remesh, clearance, silhouette and a
  pure-geometry DfM composite
- **Agent-friendly errors** — failures return a structured JSON envelope
  (`{"code", "message", "hint"}`) the caller can parse and act on
- **Deterministic output** — fixed seeds, cameras and lighting; identical inputs produce
  identical reports

Supported formats: STL, OBJ, PLY, GLB/glTF, OFF, 3MF.

## Requirements

- [uv](https://docs.astral.sh/uv/) — provisions the pinned Python (3.13) and all
  dependencies automatically

## Installation

```bash
git clone https://github.com/baho0/MeshValidationMCP.git
cd MeshValidationMCP
uv sync
uv run pytest   # optional: verify the installation
```

### Registering with Claude Code

The repository ships a project-scoped [`.mcp.json`](.mcp.json); opening Claude Code
inside the repo picks the server up automatically. To register it globally:

```bash
claude mcp add mesh-validator --scope user -- \
  uv run --directory /path/to/MeshValidationMCP mesh-validation-mcp
```

## Tools

| Tool | Purpose |
|------|---------|
| `validate_mesh(file_path, expectations, include_render?)` | Check a mesh against structured geometry + integrity expectations; returns a pass/fail report and a 4-view render (defects highlighted) |
| `inspect_mesh(file_path)` | Full geometric + integrity report — use it to ground truth a mesh before writing expectations |
| `render_mesh(file_path, views?, style?, resolution?, combine?)` | Render canonical views (`iso`, `front`, `top`, ...) without running checks |
| `compare_meshes(file_a, file_b, localized?, sample_count?, include_render?)` | Detect and classify the transform (now incl. `affine` shear/anisotropic scale) between two meshes; distances, metric deltas, transform-invariant assertions, displacement heatmap, and (with `localized`) region-scoped change verification |

### Advanced validation tools

Each returns results tagged with a `confidence` tier and error bound, and fails closed on
unreliable input.

| Tool | Purpose |
|------|---------|
| `assert_properties(file_path, properties, reference_path?)` | Assert named before/after invariants (volume conserved, watertight/genus preserved, no new defects, centroid fixed, monotonic offset, bounded Hausdorff) |
| `compare_to_golden(file_path, reference_path, tolerance?)` | Does the output match a reference — exact per-vertex (matching topology) or bounded surface distance |
| `inspect_section(file_path, plane_origin, plane_normal)` | Planar cross-section: per-loop perimeter and net/gross area (exact) |
| `measure_thickness(file_path, sample_count?)` | Inscribed-sphere wall/feature thickness (min/p5/median) |
| `analyze_draft(file_path, pull_direction, min_draft_deg?)` | Area-weighted draft angle and undercut area for a pull direction |
| `fit_feature(file_path, region, kind)` | Fit a plane/sphere/cylinder to selected feature faces (fillet/bore radius, chamfer) with residual |
| `validate_boolean(file_a, file_b, file_result, operation, ...)` | Boolean/CSG result: volume bounds + signed-distance containment + seam integrity |
| `detect_symmetry(file_path, rel_tolerance?)` | Mirror planes and rotational fold |
| `validate_array(file_result, file_base, pattern)` | Linear/polar array: count, congruence, positions |
| `validate_generative(file_path, operation, profile_area, ...)` | Extrude (area × height) / revolve (Pappus) by volume signature |
| `measure_displacement(file_a, file_b)` | Whole-mesh signed displacement field (deformation) |
| `validate_remesh(file_before, file_after, max_deviation, ...)` | Retessellation kept shape + topology + quality |
| `check_clearance(file_a, file_b, min_clearance?)` | Assembly interference / minimum clearance |
| `compare_silhouette(file_a, file_b, view_axis?, resolution?)` | Orthographic outline IoU (profile preserved) |
| `units_sanity(file_path, expected_units?, ...)` | Units/scale plausibility + sampling self-consistency |
| `validate_dfm(file_path, min_wall_thickness?, pull_direction?, ...)` | Pure-geometry DfM: min wall + draft/undercut + trapped voids |

All file paths must be absolute. All values are interpreted in the file's native units.
Renders use a Z-up convention: `front` looks along +Y, `iso` views from the (+X, −Y, +Z)
octant.

### Expectations example

```json
{
  "volume": 500,
  "surface_area": {"expected": 400, "rel_tol": 0.02},
  "bbox_extents": [10, 10, 5],
  "vertex_count": {"min": 6, "max": 9},
  "watertight": true,
  "self_intersecting_face_count": 0,
  "non_manifold_edge_count": 0,
  "tolerance": {"relative": 0.01}
}
```

Only the keys you set are checked. Geometry keys: `volume`, `surface_area`, `bbox_min`,
`bbox_max`, `bbox_extents`, `centroid`, `vertex_count`, `face_count`, `watertight`,
`winding_consistent`, `body_count`, `euler_number`. Integrity keys (usually `0`):
`self_intersecting_face_count`, `non_manifold_edge_count`, `boundary_edge_count`,
`degenerate_face_count`, `sliver_face_count`, `duplicate_face_count`, `flipped_face_count`,
plus `min_triangle_quality` (a floor). Scalars and vectors accept a bare value (global
tolerance, default 1 % relative) or an object with `rel_tol`/`abs_tol` overrides; counts
accept an exact integer or a `min`/`max` range. The pass rule is
`|actual − expected| ≤ max(abs_tol, rel_tol · |expected|)`.

### Localized-change validation

For an edit confined to a selected region — the common CAD case ("emboss *this area* by
3mm", "pocket *here* 2mm deep") — export the mesh before and after, then pass `localized`
to `compare_meshes`:

```json
{
  "region": {"kind": "box", "min": [-12, -12, 1.5], "max": [12, 12, 8]},
  "emboss_height": 3.0,
  "max_unchanged_deviation": 0.01
}
```

The region can be a `box` (min/max), `sphere` (center/radius), `plane` (origin/normal
half-space), or explicit `vertex_ids` / `face_ids`. The tool reports how much moved inside
the region, whether everything outside stayed put (`max_unchanged_deviation`), and the
signed feature displacement (`emboss_height` for material added, `pocket_depth` for
material removed) — so a change that leaks outside the intended area, or moves the wrong
distance, fails the check.

## Rendering backends

| Backend | Selection | Notes |
|---------|-----------|-------|
| matplotlib (default) | automatic | Pure CPU, works everywhere; orthographic views with edge overlay — well suited to flat-faced CAD parts |
| pyrender + EGL | `MESH_MCP_RENDERER=pyrender` | GPU offscreen rendering; install with `uv sync --extra gl`. Falls back to matplotlib on any GL failure |

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MESH_MCP_RENDERER` | `matplotlib` | Render backend (`matplotlib` or `pyrender`) |
| `MESH_MCP_MAX_FILE_MB` | `500` | Maximum input file size |
| `MESH_MCP_MAX_FACES` | `5000000` | Maximum face count at load |
| `MESH_MCP_RENDER_MAX_FACES` | `120000` | Faces above this are subsampled for display only |
| `MESH_MCP_SELFINT_MAX_FACES` | `200000` | Above this the self-intersection test is skipped |

## Development

```
src/mesh_validation_mcp/
├── server.py            # MCP wiring (the only module importing the mcp SDK)
├── loading.py           # file loading and normalization
├── confidence.py        # confidence tiers + numeric error bounds
├── metrics.py           # geometric metrics (+ cross-checks, topology, curvature, inertia)
├── integrity.py         # mesh-integrity metrics (self-intersection, manifoldness, quality)
├── region.py            # Region primitive (box/sphere/plane/vertex-ids/face-ids)
├── validation.py        # expectations schema and assertion engine (fail-closed)
├── comparison.py        # transform/affine detection, distances, signed field, localized change
├── oracles.py           # named property oracles + golden.py reference matching
├── section.py           # planar cross-sections and section-area profiles
├── curvature.py primitives.py topology.py   # curvature, primitive fits, genus/loops
├── features.py          # wall thickness, draft/undercut, region primitive fits
├── boolean_validate.py array_validate.py generative_validate.py remesh_validate.py
├── symmetry.py clearance.py silhouette.py units.py dfm.py
└── rendering/           # camera math, backends, contact-sheet composition
tests/                   # unit tests + in-memory MCP integration tests
```

```bash
uv run pytest
```

## License

[MIT](LICENSE)
