import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from mesh_validation_mcp.errors import ErrorCode, MeshToolError
from mesh_validation_mcp.region import Region

_adapter = TypeAdapter(Region)


def _region(data):
    return _adapter.validate_python(data)


def test_box_region_mask(box):
    # box vertices at +-5, +-5, +-2.5; select the +x half
    region = _region({"kind": "box", "min": [0, -5, -2.5], "max": [5, 5, 2.5]})
    vmask = region.vertex_mask(box)
    assert vmask.sum() == 4
    assert region.face_mask(box).sum() == 2  # two faces fully in the +x half


def test_sphere_region_mask(box):
    region = _region({"kind": "sphere", "center": [5, 5, 2.5], "radius": 1.0})
    assert region.vertex_mask(box).sum() == 1


def test_plane_region_mask(box):
    region = _region({"kind": "plane", "origin": [0, 0, 0], "normal": [0, 0, 1]})
    assert region.vertex_mask(box).sum() == 4  # z >= 0 corners


def test_vertex_ids_region(box):
    region = _region({"kind": "vertex_ids", "vertex_ids": [0, 1, 2]})
    assert region.vertex_mask(box).sum() == 3


def test_face_ids_region(box):
    region = _region({"kind": "face_ids", "face_ids": [0]})
    assert region.vertex_mask(box).sum() == 3  # the 3 vertices of face 0


def test_bad_radius_rejected_by_schema():
    with pytest.raises(ValidationError):
        _region({"kind": "sphere", "center": [0, 0, 0], "radius": -1})


def test_zero_normal_rejected(box):
    region = _region({"kind": "plane", "origin": [0, 0, 0], "normal": [0, 0, 0]})
    with pytest.raises(MeshToolError) as exc:
        region.vertex_mask(box)
    assert exc.value.code is ErrorCode.INVALID_REGION


def test_box_max_below_min_rejected(box):
    region = _region({"kind": "box", "min": [5, 0, 0], "max": [0, 0, 0]})
    with pytest.raises(MeshToolError) as exc:
        region.vertex_mask(box)
    assert exc.value.code is ErrorCode.INVALID_REGION


def test_vertex_ids_out_of_range(box):
    region = _region({"kind": "vertex_ids", "vertex_ids": [9999]})
    with pytest.raises(MeshToolError) as exc:
        region.vertex_mask(box)
    assert exc.value.code is ErrorCode.INVALID_REGION


def test_face_ids_out_of_range(box):
    region = _region({"kind": "face_ids", "face_ids": [9999]})
    with pytest.raises(MeshToolError) as exc:
        region.vertex_mask(box)
    assert exc.value.code is ErrorCode.INVALID_REGION
