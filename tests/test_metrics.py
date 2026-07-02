import math

import pytest

from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.metrics import compute_metrics


def test_box_metrics(box_path):
    m = compute_metrics(load_mesh(box_path))
    assert m.volume == pytest.approx(500.0, rel=1e-6)
    assert m.surface_area == pytest.approx(400.0, rel=1e-6)
    assert m.vertex_count == 8
    assert m.face_count == 12
    assert m.edge_count == 18
    assert m.is_watertight and m.is_winding_consistent and m.volume_reliable
    assert m.euler_number == 2
    assert m.body_count == 1
    assert m.extents == pytest.approx([10.0, 10.0, 5.0])
    assert m.bounds.min == pytest.approx([-5.0, -5.0, -2.5])
    assert m.bounds.max == pytest.approx([5.0, 5.0, 2.5])
    assert m.centroid == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert m.center_mass == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert m.bbox_diagonal == pytest.approx(15.0, rel=1e-6)
    assert m.caveats == []


def test_sphere_metrics(sphere_path):
    m = compute_metrics(load_mesh(sphere_path))
    assert m.vertex_count == 642
    assert m.face_count == 1280
    assert m.is_watertight
    assert m.volume == pytest.approx(4.0 / 3.0 * math.pi * 8.0, rel=0.02)
    assert m.surface_area == pytest.approx(4.0 * math.pi * 4.0, rel=0.02)


def test_broken_box_metrics(broken_path):
    m = compute_metrics(load_mesh(broken_path))
    assert not m.is_watertight
    assert not m.volume_reliable
    assert m.center_mass is None
    assert m.volume is not None  # still reported, with a caveat
    assert any("not_watertight" in c for c in m.caveats)


def test_two_bodies_metrics(two_bodies_path):
    m = compute_metrics(load_mesh(two_bodies_path))
    assert m.body_count == 2
    assert len(m.bodies) == 2
    assert m.volume == pytest.approx(1000.0, rel=1e-6)
    assert m.euler_number == 4
    assert all(b.is_watertight for b in m.bodies)
    assert all(b.volume == pytest.approx(500.0, rel=1e-6) for b in m.bodies)
