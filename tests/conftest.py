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
