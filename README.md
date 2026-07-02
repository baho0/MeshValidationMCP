# MeshValidationMCP

An MCP (Model Context Protocol) server that lets coding agents **validate 3D mesh
manipulations** — with deterministic geometric checks *and* multi-view visual renders.

The intended loop: an agent writes mesh-manipulation code, exports the result, then calls
`validate_mesh` with the file path and its structured expectations. The server returns a
per-check pass/fail report **plus** a rendered contact sheet, so a vision-capable agent
verifies the result both numerically and visually before moving on.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (`pacman -S uv` on Arch, or
`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv provisions the pinned Python
(3.13) and the virtualenv automatically:

```bash
uv sync          # install deps
uv run pytest    # 51 tests
```

### Register with Claude Code

The repo ships a project-scoped [.mcp.json](.mcp.json) — opening Claude Code inside this
repo picks the server up automatically (adjust the `--directory` path if you cloned
elsewhere). To register it globally instead:

```bash
claude mcp add mesh-validator -- uv run --directory /path/to/MeshValidationMCP mesh-validation-mcp
```

## Tools

All file paths must be **absolute**. All values are in the **file's native units**.
Supported formats: STL, OBJ, PLY, GLB/glTF, OFF, 3MF.

### `validate_mesh(file_path, expectations, include_render=true)`

The core hybrid tool: checks the mesh against structured expectations, returns a
deterministic report and (by default) a 4-view render sheet.

```jsonc
// expectations example
{
  "volume": 500,                                  // bare value -> global tolerance
  "surface_area": {"expected": 400, "rel_tol": 0.02},
  "bbox_extents": [10, 10, 5],
  "vertex_count": {"min": 6, "max": 9},           // counts: exact int or min/max range
  "watertight": true,
  "body_count": 1,
  "tolerance": {"relative": 0.01, "absolute": null}  // global default: 1% relative
}
```

Supported keys: `volume`, `surface_area`, `bbox_min`, `bbox_max`, `bbox_extents`,
`centroid` (scalar/vector with tolerances) · `vertex_count`, `face_count` (exact or
range) · `watertight`, `winding_consistent`, `body_count`, `euler_number` (exact) ·
`tolerance` (global). Pass rule: `|actual - expected| <= max(abs_tol, rel_tol*|expected|)`.

Response: `{"passed", "summary", "checks": [{name, passed, expected, actual, deviation,
deviation_pct, tolerance, caveats}], "mesh", "render"}` + a PNG image block.

### `inspect_mesh(file_path)`

Full geometric report (counts, watertightness, volume + `volume_reliable` flag, bounds,
per-body breakdown, caveats). Use it to ground truth a mesh before writing expectations.

### `render_mesh(file_path, views?, style?, resolution?, combine?)`

Visual-only. Views: `iso, iso_back, front, back, left, right, top, bottom` (≤6 per
sheet). Styles: `shaded`, `shaded_edges`, `wireframe`. Tiles carry the view label and an
RGB axis gizmo (X=red, Y=green, Z=blue); multi-body meshes get one tint per body.

### `compare_meshes(file_a, file_b, sample_count?, include_render?)`

Before/after comparison: detects the transform (translation vector, rotation axis+angle,
uniform scale) via exact vertex correspondence when topology matches (Umeyama/procrustes)
or ICP otherwise, classifies the change
(`identical | translation | rotation | rigid | similarity | mirrored | deformed`), and
reports chamfer/Hausdorff distances plus metric deltas. Mirroring is detected explicitly
(reflected fits are tried when no proper transform explains B), so `mirror()`-style
manipulations are verifiable too. The optional render paints B as a displacement heatmap
(viridis + colorbar).

### Errors

Tool failures return a parseable JSON envelope:
`{"code": "FILE_NOT_FOUND", "message": "...", "hint": "..."}`. Codes:
`FILE_NOT_FOUND, NOT_ABSOLUTE_PATH, UNSUPPORTED_FORMAT, LOAD_FAILED, EMPTY_MESH,
MESH_TOO_LARGE, INVALID_EXPECTATION, INVALID_VIEW, RENDER_FAILED`.

## Rendering backends

| Backend | Selection | Notes |
|---|---|---|
| **matplotlib** (default) | automatic | Pure CPU, works everywhere. Orthographic, painter's algorithm, Lambert shading, edge overlay — the most informative output for flat-faced CAD parts. |
| **pyrender + EGL** (opt-in) | `MESH_MCP_RENDERER=pyrender` | GPU offscreen, perspective + true depth; better occlusion on very large organic meshes. Install with `uv sync --extra gl`. Verified working on Linux/EGL (mesa & nvidia). No edge overlay. |

pyrender is a 2021-era package; the `[gl]` extra pins `pyglet<2` and overrides its stale
`PyOpenGL==3.1.0` pin (see `[tool.uv] override-dependencies`). If EGL picks the wrong GPU,
set `EGL_DEVICE_ID`. Any GL failure falls back to matplotlib automatically.

Camera convention: **Z-up**; `front` looks along +Y, `right` looks along −X, `iso` from
the (+X,−Y,+Z) octant.

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MESH_MCP_RENDERER` | *(auto → matplotlib)* | `matplotlib` or `pyrender` |
| `MESH_MCP_MAX_FILE_MB` | 500 | File size cap |
| `MESH_MCP_MAX_FACES` | 5,000,000 | Face-count cap at load |
| `MESH_MCP_RENDER_MAX_FACES` | 120,000 | Above this, renders use a seeded face subset (`render_decimated: true`) |

Determinism: every sampling operation uses seed 0; cameras, lights and colors are fixed —
identical inputs produce identical reports and near-identical images.

## Development

```
src/mesh_validation_mcp/
  server.py        # MCP wiring (the only module importing the mcp SDK)
  loading.py       # file -> LoadedMesh normalization + structured errors
  metrics.py       # geometric report (pydantic)
  validation.py    # expectations schema + assertion engine
  comparison.py    # transform detection, distances, heatmap scalars
  rendering/       # backends (matplotlib / pyrender), camera math, contact sheets
tests/             # pytest: unit + in-memory MCP integration tests
```

Run a quick manual check without an agent:

```bash
uv run python -c "import trimesh; trimesh.creation.box((10,10,5)).export('/tmp/box.stl')"
uv run mesh-validation-mcp   # then speak MCP over stdio, or just use Claude Code
```
