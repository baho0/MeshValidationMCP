"""Shared render types and camera math used by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import PIL.Image
    import trimesh


class RenderStyle(str, Enum):
    SHADED = "shaded"
    SHADED_EDGES = "shaded_edges"
    WIREFRAME = "wireframe"


@dataclass(frozen=True)
class ViewSpec:
    name: str
    direction: tuple[float, float, float]  # unit vector, center -> eye


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=float)
    a /= np.linalg.norm(a)
    return (float(a[0]), float(a[1]), float(a[2]))


VIEWS: dict[str, ViewSpec] = {
    "iso": ViewSpec("iso", _unit((1.0, -1.0, 1.0))),
    "iso_back": ViewSpec("iso_back", _unit((-1.0, 1.0, 1.0))),
    "front": ViewSpec("front", (0.0, -1.0, 0.0)),
    "back": ViewSpec("back", (0.0, 1.0, 0.0)),
    "right": ViewSpec("right", (1.0, 0.0, 0.0)),
    "left": ViewSpec("left", (-1.0, 0.0, 0.0)),
    "top": ViewSpec("top", (0.0, 0.0, 1.0)),
    "bottom": ViewSpec("bottom", (0.0, 0.0, -1.0)),
}

CAMERA_CONVENTION = (
    "Z-up. 'front' looks along +Y (camera on the -Y side), 'right' looks along -X "
    "(camera on the +X side), 'top' looks down -Z. 'iso' views from the (+X,-Y,+Z) octant."
)


@dataclass
class CameraFrame:
    rotation: np.ndarray  # (3,3) world->camera; rows are camera x/y/z in world coords
    eye: np.ndarray
    center: np.ndarray
    half_extent: float  # orthographic half-size including margin
    distance: float


def camera_frame(
    bounds: np.ndarray,
    view: ViewSpec,
    distance_factor: float = 2.2,
    margin: float = 1.08,
) -> CameraFrame:
    bounds = np.asarray(bounds, dtype=float)
    center = bounds.mean(axis=0)
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) / 2.0) or 1.0

    z_axis = np.asarray(view.direction, dtype=float)  # camera z points backward
    up_hint = np.array((0.0, 0.0, 1.0))
    if abs(float(np.dot(z_axis, up_hint))) > 0.99:
        up_hint = np.array((0.0, 1.0, 0.0))
    x_axis = np.cross(up_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    return CameraFrame(
        rotation=np.stack([x_axis, y_axis, z_axis]),
        eye=center + z_axis * radius * distance_factor,
        center=center,
        half_extent=radius * margin,
        distance=radius * distance_factor,
    )


@dataclass
class RenderRequest:
    mesh: "trimesh.Trimesh"
    views: list[ViewSpec]
    tile_px: int
    style: RenderStyle
    face_colors: np.ndarray | None = None  # (F,3) float RGB in [0,1]


@dataclass
class RenderedTile:
    image: "PIL.Image.Image"
    view: ViewSpec
    rotation: np.ndarray  # world->camera, for the axis gizmo overlay


class Renderer(Protocol):
    name: str

    def render(self, request: RenderRequest) -> list[RenderedTile]: ...
