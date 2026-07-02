"""Pure-CPU renderer: orthographic projection + painter's algorithm + Lambert
shading drawn through matplotlib's Agg canvas. Works everywhere, no GL."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from PIL import Image as PILImage

from ..config import EDGE_MAX_FACES
from .base import RenderRequest, RenderStyle, RenderedTile, camera_frame

_LIGHT = np.array((0.4, -0.7, 0.6))
_LIGHT /= np.linalg.norm(_LIGHT)
_BASE_RGB = np.array((174, 184, 196)) / 255.0
_EDGE_COLOR = "#30343A"
_BACKFACE_DIM = 0.55


class MatplotlibRenderer:
    name = "matplotlib"

    def render(self, request: RenderRequest) -> list[RenderedTile]:
        mesh = request.mesh
        face_count = len(mesh.faces)
        base = (
            request.face_colors
            if request.face_colors is not None
            else np.tile(_BASE_RGB, (face_count, 1))
        )
        normals = np.asarray(mesh.face_normals, dtype=float)
        lambert = 0.35 + 0.65 * np.clip(normals @ _LIGHT, 0.0, 1.0)

        tiles: list[RenderedTile] = []
        for view in request.views:
            cam = camera_frame(mesh.bounds, view)
            verts_cam = (mesh.vertices - cam.eye) @ cam.rotation.T
            tris = verts_cam[mesh.faces]  # (F, 3, 3)
            # Painter's algorithm: camera z points backward, so more negative z
            # is farther away; ascending sort draws far faces first.
            order = np.argsort(tris[:, :, 2].mean(axis=1))

            shade = lambert.copy()
            shade[normals @ np.asarray(view.direction) < 0] *= _BACKFACE_DIM
            rgba = np.empty((face_count, 4))
            rgba[:, :3] = np.clip(base * shade[:, None], 0.0, 1.0)
            rgba[:, 3] = 1.0

            fig = Figure(figsize=(request.tile_px / 100.0, request.tile_px / 100.0), dpi=100)
            canvas = FigureCanvasAgg(fig)
            fig.set_facecolor("white")
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
            ax.set_facecolor("white")

            polys = tris[order][:, :, :2]
            if request.style is RenderStyle.WIREFRAME:
                collection = PolyCollection(
                    polys, facecolors="none", edgecolors=_EDGE_COLOR, linewidths=0.4
                )
            else:
                draw_edges = (
                    request.style is RenderStyle.SHADED_EDGES and face_count <= EDGE_MAX_FACES
                )
                collection = PolyCollection(
                    polys,
                    facecolors=rgba[order],
                    # edgecolors="face" fills antialiasing seams between triangles
                    edgecolors=_EDGE_COLOR if draw_edges else "face",
                    linewidths=0.3 if draw_edges else 0.1,
                )
            ax.add_collection(collection)
            ax.set_xlim(-cam.half_extent, cam.half_extent)
            ax.set_ylim(-cam.half_extent, cam.half_extent)
            ax.set_aspect("equal")
            ax.set_axis_off()

            canvas.draw()
            image = PILImage.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
            tiles.append(RenderedTile(image=image, view=view, rotation=cam.rotation))
        return tiles
