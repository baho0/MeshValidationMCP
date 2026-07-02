"""Optional GPU renderer via pyrender + EGL offscreen (the [gl] extra).

pyrender is unmaintained (2021); everything here is guarded so any failure —
missing import, broken pyglet, EGL init error — just marks the backend
unavailable and the caller falls back to matplotlib.
"""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from .base import RenderRequest, RenderStyle, RenderedTile, camera_frame

_BASE_RGBA = np.array((174, 184, 196, 255), dtype=np.uint8)
# Perspective camera needs more distance than the orthographic backends:
# yfov 40deg fits the bounding sphere at ~2.9 radii.
_DISTANCE_FACTOR = 2.9


class PyrenderRenderer:
    name = "pyrender"
    _available: bool | None = None

    @classmethod
    def available(cls) -> bool:
        if cls._available is None:
            try:
                os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
                import pyrender

                scene = pyrender.Scene()
                scene.add(pyrender.PerspectiveCamera(yfov=0.7), pose=np.eye(4))
                probe = pyrender.OffscreenRenderer(64, 64)
                probe.render(scene)
                probe.delete()
                cls._available = True
            except Exception:
                cls._available = False
        return cls._available

    def render(self, request: RenderRequest) -> list[RenderedTile]:
        import pyrender

        mesh = request.mesh.copy()
        if request.face_colors is not None:
            rgba = np.hstack(
                [request.face_colors, np.ones((len(request.face_colors), 1))]
            )
            mesh.visual.face_colors = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            mesh.visual.face_colors = np.tile(_BASE_RGBA, (len(mesh.faces), 1))
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        flags = pyrender.RenderFlags.SKIP_CULL_FACES
        if request.style is RenderStyle.WIREFRAME:
            flags |= pyrender.RenderFlags.ALL_WIREFRAME

        renderer = pyrender.OffscreenRenderer(request.tile_px, request.tile_px)
        tiles: list[RenderedTile] = []
        try:
            for view in request.views:
                cam = camera_frame(mesh.bounds, view, distance_factor=_DISTANCE_FACTOR)
                scene = pyrender.Scene(
                    ambient_light=(0.25, 0.25, 0.25), bg_color=(1.0, 1.0, 1.0, 1.0)
                )
                scene.add(pmesh)
                pose = np.eye(4)
                pose[:3, 0] = cam.rotation[0]
                pose[:3, 1] = cam.rotation[1]
                pose[:3, 2] = cam.rotation[2]
                pose[:3, 3] = cam.eye
                camera = pyrender.PerspectiveCamera(
                    yfov=math.radians(40.0),
                    znear=cam.distance * 0.02,
                    zfar=cam.distance * 10.0,
                )
                scene.add(camera, pose=pose)
                # Key light offset from the camera so camera-facing faces are not
                # uniformly max-lit; weak fill from the opposite side.
                offset = Rotation.from_euler("xy", (-35.0, 30.0), degrees=True).as_matrix()
                key_pose = pose.copy()
                key_pose[:3, :3] = pose[:3, :3] @ offset
                scene.add(pyrender.DirectionalLight(intensity=2.5), pose=key_pose)
                fill_pose = pose.copy()
                fill_pose[:3, :3] = pose[:3, :3] @ offset.T
                scene.add(pyrender.DirectionalLight(intensity=0.8), pose=fill_pose)
                color, _depth = renderer.render(scene, flags=flags)
                image = PILImage.fromarray(color).convert("RGB")
                tiles.append(RenderedTile(image=image, view=view, rotation=cam.rotation))
        finally:
            renderer.delete()
        return tiles
