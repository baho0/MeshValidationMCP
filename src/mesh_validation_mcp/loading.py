"""File loading and normalization to a uniform LoadedMesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

from .config import MAX_FACES_LOAD, MAX_FILE_MB
from .errors import ErrorCode, MeshToolError

SUPPORTED_SUFFIXES = ("stl", "obj", "ply", "glb", "gltf", "off", "3mf")

# STL stores unshared per-face vertices, so it MUST be merged on load or every vertex
# reads as unmerged and the solid reads as open. Formats that store explicit shared
# vertices are loaded WITHOUT processing so the author's exact vertex order survives a
# re-export: that is what keeps the before/after correspondence in compare() /
# localized_change on the exact path instead of silently demoting to ICP.
_MERGE_ON_LOAD = frozenset({"stl"})


@dataclass
class LoadedMesh:
    path: str
    format: str
    file_size_bytes: int
    combined: trimesh.Trimesh
    bodies: list[trimesh.Trimesh]


def _supported_hint() -> str:
    return f"Supported formats: {', '.join(SUPPORTED_SUFFIXES)}"


def load_mesh(file_path: str, process: bool | None = None) -> LoadedMesh:
    """Load and normalize a mesh file.

    ``process`` controls trimesh's on-load vertex merging. ``None`` (default) auto-selects
    by format: merge STL (which has no shared-vertex concept on disk) and leave shared-vertex
    formats untouched to preserve exact vertex correspondence. Pass an explicit bool to force.
    """
    path = Path(file_path)
    if not path.is_absolute():
        raise MeshToolError(
            ErrorCode.NOT_ABSOLUTE_PATH,
            f"Path is not absolute: {file_path!r}",
            "Pass an absolute path; the MCP server's working directory differs from yours.",
        )
    if not path.exists():
        raise MeshToolError(
            ErrorCode.FILE_NOT_FOUND,
            f"File does not exist: {file_path}",
            "Check the path and make sure your manipulation code actually exported the file.",
        )
    if not path.is_file():
        raise MeshToolError(ErrorCode.FILE_NOT_FOUND, f"Path is not a regular file: {file_path}")

    size = path.stat().st_size
    if size > MAX_FILE_MB * 1024 * 1024:
        raise MeshToolError(
            ErrorCode.MESH_TOO_LARGE,
            f"File is {size / 1e6:.0f} MB; the cap is {MAX_FILE_MB:.0f} MB",
            "Decimate or split the mesh before validating.",
        )

    suffix = path.suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_SUFFIXES:
        raise MeshToolError(
            ErrorCode.UNSUPPORTED_FORMAT, f"Unsupported format: .{suffix}", _supported_hint()
        )

    if process is None:
        process = suffix in _MERGE_ON_LOAD
    try:
        loaded = trimesh.load(str(path), process=process)
    except MeshToolError:
        raise
    except Exception as exc:
        raise MeshToolError(
            ErrorCode.LOAD_FAILED,
            f"trimesh failed to load {path.name}: {exc}",
            "The file may be corrupt, truncated or not really the format its extension claims.",
        ) from exc

    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not geoms:
            raise MeshToolError(
                ErrorCode.EMPTY_MESH, f"{path.name} contains no triangle geometry"
            )
        combined = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    elif isinstance(loaded, trimesh.Trimesh):
        combined = loaded
    else:
        raise MeshToolError(
            ErrorCode.EMPTY_MESH,
            f"{path.name} contains no triangle mesh (got {type(loaded).__name__})",
            "Point clouds and curve/path files are not supported.",
        )

    if len(combined.faces) == 0 or len(combined.vertices) == 0:
        raise MeshToolError(ErrorCode.EMPTY_MESH, f"{path.name} contains no triangle geometry")
    if len(combined.faces) > MAX_FACES_LOAD:
        raise MeshToolError(
            ErrorCode.MESH_TOO_LARGE,
            f"Mesh has {len(combined.faces)} faces; the cap is {MAX_FACES_LOAD}",
            "Decimate the mesh before validating.",
        )

    bodies = [b for b in combined.split(only_watertight=False) if len(b.faces) > 0]
    if not bodies:
        bodies = [combined]

    return LoadedMesh(
        path=str(path),
        format=suffix,
        file_size_bytes=size,
        combined=combined,
        bodies=bodies,
    )
