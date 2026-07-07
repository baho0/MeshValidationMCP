"""Renderer selection and the view-rendering orchestration used by tools."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import trimesh
from matplotlib import colormaps

from ..config import (
    MAX_COMBINED_VIEWS,
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    RENDER_MAX_FACES,
    RENDERER_ENV,
    SEED,
)
from ..errors import ErrorCode, MeshToolError
from .base import CAMERA_CONVENTION, VIEWS, Renderer, RenderRequest, RenderStyle, ViewSpec
from .compose import ColorbarSpec, annotate_tile, compose_sheet, encode_png, grid_shape
from .matplotlib_backend import MatplotlibRenderer

DEFAULT_VIEWS = ("iso", "front", "top", "right")

_active_renderer: Renderer | None = None


def get_renderer() -> Renderer:
    global _active_renderer
    if _active_renderer is not None:
        return _active_renderer

    forced = os.environ.get(RENDERER_ENV, "").strip().lower() or None
    if forced == "matplotlib":
        _active_renderer = MatplotlibRenderer()
    elif forced == "pyrender":
        from .pyrender_backend import PyrenderRenderer

        if not PyrenderRenderer.available():
            raise MeshToolError(
                ErrorCode.RENDER_FAILED,
                f"{RENDERER_ENV}=pyrender but the pyrender/EGL stack failed to initialize",
                "Install the gl extra (uv sync --extra gl), check EGL drivers, "
                f"or unset {RENDERER_ENV} to fall back to matplotlib.",
            )
        _active_renderer = PyrenderRenderer()
    elif forced is not None:
        raise MeshToolError(
            ErrorCode.RENDER_FAILED,
            f"Unknown {RENDERER_ENV} value: {forced!r}",
            "Valid values: matplotlib, pyrender.",
        )
    else:
        # matplotlib is the default even when the GL stack is present: its
        # shaded_edges output carries more geometric information for flat-faced
        # CAD parts, and it works everywhere. Set MESH_MCP_RENDERER=pyrender to
        # opt into the GPU backend (better occlusion on huge organic meshes).
        _active_renderer = MatplotlibRenderer()
    return _active_renderer


def resolve_views(names: list[str]) -> list[ViewSpec]:
    if not names:
        raise MeshToolError(
            ErrorCode.INVALID_VIEW, "views list is empty", f"Valid views: {', '.join(VIEWS)}"
        )
    specs = []
    for name in names:
        key = str(name).strip().lower()
        if key not in VIEWS:
            raise MeshToolError(
                ErrorCode.INVALID_VIEW,
                f"Unknown view {name!r}",
                f"Valid views: {', '.join(VIEWS)}",
            )
        specs.append(VIEWS[key])
    return specs


def body_face_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Tint each connected component differently so wrong body counts are
    visually obvious. Returns None for single-body meshes."""
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces)
    )
    if labels.max() == 0:
        return None
    palette = colormaps["tab10"](np.linspace(0.0, 1.0, 10))[:, :3]
    return palette[labels % 10]


_HIGHLIGHT_BASE = np.array((174, 184, 196)) / 255.0
_HIGHLIGHT_ALERT = np.array((220, 38, 38)) / 255.0


def highlight_face_colors(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> np.ndarray:
    """Paint the given faces a solid alert red over a neutral base, so defect
    locations (broken/sliver/self-intersecting/flipped faces) pop in the render."""
    colors = np.tile(_HIGHLIGHT_BASE, (len(mesh.faces), 1))
    ids = np.asarray(face_ids, dtype=np.int64)
    if ids.size:
        colors[ids] = _HIGHLIGHT_ALERT
    return colors


def scalars_to_face_colors(
    mesh: trimesh.Trimesh,
    scalars: np.ndarray,
    label: str,
    cmap: str = "viridis",
    per_face: bool = False,
    symmetric: bool = False,
) -> tuple[np.ndarray, ColorbarSpec]:
    """Map scalars to per-face colors + a colorbar spec.

    ``per_face``: the scalars are already one-per-face (else one-per-vertex, averaged to faces).
    ``symmetric``: normalize about zero (vmin=-M, vmax=+M) so a diverging colormap centers its
    neutral colour at 0 — used for signed fields (e.g. outward vs inward displacement)."""
    values = np.asarray(scalars, dtype=float)
    finite = values[np.isfinite(values)]
    if symmetric:
        m = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = -m, m
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    span = (vmax - vmin) or 1.0
    normalized = np.nan_to_num((values - vmin) / span, nan=0.5)
    face_values = normalized if per_face else normalized[mesh.faces].mean(axis=1)
    lut = colormaps[cmap](np.linspace(0.0, 1.0, 256))[:, :3]
    colors = lut[np.clip((face_values * 255).astype(int), 0, 255)]
    colorbar: ColorbarSpec = {
        "lut": (lut * 255).astype(np.uint8),
        "vmin": vmin,
        "vmax": vmax,
        "label": label,
    }
    return colors, colorbar


def render_views(
    mesh: trimesh.Trimesh,
    view_names: list[str],
    style: str = "shaded_edges",
    resolution: int = 1024,
    combine: bool = True,
    face_colors: np.ndarray | None = None,
    colorbar: ColorbarSpec | None = None,
) -> tuple[list[bytes], dict[str, Any]]:
    """Render the mesh from the named views; returns (PNG blobs, meta dict)."""
    views = resolve_views(view_names)
    if combine and len(views) > MAX_COMBINED_VIEWS:
        raise MeshToolError(
            ErrorCode.INVALID_VIEW,
            f"At most {MAX_COMBINED_VIEWS} views fit on one sheet, got {len(views)}",
            "Reduce the view list or pass combine=false for one image per view.",
        )
    resolution = int(min(max(resolution, MIN_RESOLUTION), MAX_RESOLUTION))

    if face_colors is None:
        face_colors = body_face_colors(mesh)

    decimated = False
    if len(mesh.faces) > RENDER_MAX_FACES:
        rng = np.random.default_rng(SEED)
        keep = np.sort(rng.choice(len(mesh.faces), RENDER_MAX_FACES, replace=False))
        display = mesh.copy()
        display.update_faces(keep)
        if face_colors is not None:
            face_colors = face_colors[keep]
        mesh = display
        decimated = True

    cols = grid_shape(len(views))[1] if combine else 1
    tile_px = max(220, resolution // cols)

    renderer = get_renderer()
    request = RenderRequest(
        mesh=mesh,
        views=views,
        tile_px=tile_px,
        style=RenderStyle(style),
        face_colors=face_colors,
    )
    try:
        tiles = renderer.render(request)
    except Exception:
        if renderer.name == "matplotlib":
            raise MeshToolError(
                ErrorCode.RENDER_FAILED, "matplotlib renderer failed unexpectedly"
            )
        # GPU backend died mid-flight: degrade gracefully for this process.
        global _active_renderer
        _active_renderer = renderer = MatplotlibRenderer()
        tiles = renderer.render(request)

    if combine:
        images = [encode_png(compose_sheet(tiles, colorbar=colorbar), max_edge=MAX_RESOLUTION)]
    else:
        images = [encode_png(annotate_tile(tile), max_edge=MAX_RESOLUTION) for tile in tiles]

    meta = {
        "backend": renderer.name,
        "views": [v.name for v in views],
        "style": style,
        "combined": combine,
        "resolution": resolution,
        "render_decimated": decimated,
        "camera": CAMERA_CONVENTION,
    }
    return images, meta
