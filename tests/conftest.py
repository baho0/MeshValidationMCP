import numpy as np
import pytest
import trimesh


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def box():
    return trimesh.creation.box(extents=(10.0, 10.0, 5.0))


@pytest.fixture
def sphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=2.0)


def _export(mesh: trimesh.Trimesh, path) -> str:
    mesh.export(path)
    return str(path)


@pytest.fixture
def box_path(tmp_path, box):
    return _export(box, tmp_path / "box.stl")


@pytest.fixture
def sphere_path(tmp_path, sphere):
    return _export(sphere, tmp_path / "sphere.stl")


@pytest.fixture
def broken_box(box):
    mesh = box.copy()
    mask = np.ones(len(mesh.faces), dtype=bool)
    mask[0] = False
    mesh.update_faces(mask)
    return mesh


@pytest.fixture
def broken_path(tmp_path, broken_box):
    return _export(broken_box, tmp_path / "broken.stl")


@pytest.fixture
def two_bodies_path(tmp_path, box):
    merged = trimesh.util.concatenate([box, box.copy().apply_translation((20.0, 0.0, 0.0))])
    return _export(merged, tmp_path / "two_bodies.stl")


@pytest.fixture
def translated_path(tmp_path, box):
    return _export(box.copy().apply_translation((5.0, 0.0, 0.0)), tmp_path / "translated.stl")


@pytest.fixture
def rotated_path(tmp_path, box):
    mesh = box.copy()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30.0), (0, 0, 1)))
    return _export(mesh, tmp_path / "rotated.stl")


@pytest.fixture
def scaled_path(tmp_path, box):
    return _export(box.copy().apply_scale(2.0), tmp_path / "scaled.stl")


@pytest.fixture
def subdivided_path(tmp_path, box):
    return _export(box.copy().subdivide(), tmp_path / "subdivided.stl")


# --- Localized-change (emboss) fixtures: a subdivided plate, spanning z in [0, 2] ---


def _make_plate():
    plate = trimesh.creation.box(extents=(40.0, 40.0, 2.0))
    plate.apply_translation((0, 0, 1.0))
    for _ in range(5):
        plate = plate.subdivide()
    return plate


def _emboss(plate, top_only, center=(0.0, 0.0), radius=10.0, height=3.0):
    mesh = plate.copy()
    v = mesh.vertices.copy()
    in_xy = np.linalg.norm(v[:, :2] - np.asarray(center), axis=1) < radius
    if top_only:
        in_xy &= v[:, 2] > v[:, 2].max() - 1e-6
    v[in_xy, 2] += height
    mesh.vertices = v
    return mesh


@pytest.fixture
def plate_path(tmp_path):
    return _export(_make_plate(), tmp_path / "plate.stl")


@pytest.fixture
def emboss_good_path(tmp_path):
    """Correct emboss: only the top surface of the region is raised 3mm."""
    return _export(_emboss(_make_plate(), top_only=True), tmp_path / "emboss_good.stl")


@pytest.fixture
def emboss_bad_path(tmp_path):
    """Buggy emboss: the whole XY column (incl. the bottom) is moved 3mm."""
    return _export(_emboss(_make_plate(), top_only=False), tmp_path / "emboss_bad.stl")


# --- Integrity defect fixtures ---


@pytest.fixture
def self_intersecting_path(tmp_path):
    a = trimesh.creation.box((10, 10, 10))
    b = trimesh.creation.box((10, 10, 10))
    b.apply_translation((5, 5, 5))
    return _export(trimesh.util.concatenate([a, b]), tmp_path / "selfint.stl")


@pytest.fixture
def non_manifold_path(tmp_path):
    boxes = [
        trimesh.creation.box((10, 10, 10)),
        trimesh.creation.box((10, 10, 10)).apply_translation((10, 0, 0)),
        trimesh.creation.box((10, 10, 10)).apply_translation((0, 0, 10)),
    ]
    merged = trimesh.util.concatenate(boxes)
    merged.merge_vertices()
    return _export(merged, tmp_path / "nonmanifold.stl")


@pytest.fixture
def duplicated_path(tmp_path, box):
    return _export(
        trimesh.util.concatenate([box, box.copy()]), tmp_path / "duplicated.stl"
    )


@pytest.fixture
def flipped_path(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    faces = mesh.faces.copy()
    faces[:5] = faces[:5][:, ::-1]
    mesh.faces = faces
    return _export(mesh, tmp_path / "flipped.stl")


# --- Boolean / shell fixtures (need the manifold3d backend, a dev dependency) ---


@pytest.fixture
def boolean_operands(tmp_path):
    """Two overlapping unit-1000 cubes plus their union/difference/intersection results."""
    a = trimesh.creation.box((10, 10, 10))
    b = trimesh.creation.box((10, 10, 10))
    b.apply_translation((5, 0, 0))
    paths = {
        "a": _export(a, tmp_path / "bool_a.stl"),
        "b": _export(b, tmp_path / "bool_b.stl"),
        "union": _export(a.union(b), tmp_path / "bool_union.stl"),
        "difference": _export(a.difference(b), tmp_path / "bool_diff.stl"),
        "intersection": _export(a.intersection(b), tmp_path / "bool_inter.stl"),
    }
    return paths


@pytest.fixture
def shell_path(tmp_path):
    """A 20mm cube hollowed by a 16mm cube => 2mm walls."""
    shell = trimesh.creation.box((20, 20, 20)).difference(trimesh.creation.box((16, 16, 16)))
    return _export(shell, tmp_path / "shell.stl")
