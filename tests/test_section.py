"""Phase 3.1 — planar cross-sections against analytic ground truth."""

import math

import pytest
import trimesh

from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.section import inspect_section


def _section(path, origin, normal):
    return inspect_section(load_mesh(path), origin, normal)


def test_box_midsection_is_a_rectangle(box_path):
    s = _section(box_path, [0, 0, 0], [0, 0, 1])
    assert s.intersects
    assert s.loop_count == 1
    assert s.net_area == pytest.approx(100.0, abs=1e-6)
    assert s.total_perimeter == pytest.approx(40.0, abs=1e-6)
    assert s.confidence.tier == "exact"


def test_cylinder_section_is_a_circle(tmp_path):
    cyl = trimesh.creation.cylinder(radius=3.0, height=10.0)
    path = str(tmp_path / "cyl.stl")
    cyl.export(path)
    s = _section(path, [0, 0, 0], [0, 0, 1])
    assert s.loop_count == 1
    # Faceted, so slightly under the true circle — within 1%.
    assert s.net_area == pytest.approx(math.pi * 9.0, rel=0.01)


def test_annulus_section_subtracts_the_hole(tmp_path):
    tube = trimesh.creation.annulus(r_min=2.0, r_max=5.0, height=10.0)
    path = str(tmp_path / "tube.stl")
    tube.export(path)
    s = _section(path, [0, 0, 0], [0, 0, 1])
    assert s.loop_count == 2
    assert sum(loop.is_hole for loop in s.loops) == 1
    assert s.net_area == pytest.approx(math.pi * (25.0 - 4.0), rel=0.02)
    assert s.gross_area > s.net_area  # gross does not subtract the hole


def test_plane_missing_the_mesh_reports_no_intersection(box_path):
    s = _section(box_path, [0, 0, 100], [0, 0, 1])
    assert not s.intersects
    assert s.net_area == 0.0
    assert any("does not intersect" in c for c in s.caveats)


def test_zero_normal_rejected(box_path):
    from mesh_validation_mcp.errors import MeshToolError

    with pytest.raises(MeshToolError):
        _section(box_path, [0, 0, 0], [0, 0, 0])
