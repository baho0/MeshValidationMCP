"""Phase 4.1-4.4 — wall thickness, draft/undercut, region primitive fits."""

import pytest

from mesh_validation_mcp.errors import MeshToolError
from mesh_validation_mcp.features import draft_analysis, fit_region, wall_thickness
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.region import BoxRegion


def test_shell_wall_thickness(shell_path):
    info = wall_thickness(load_mesh(shell_path), 1500)
    # 20mm cube hollowed by 16mm cube -> 2mm walls; the median sample sits on a wall.
    assert info.median_thickness == pytest.approx(2.0, abs=0.1)
    assert info.confidence.tier == "sampled"


def test_wall_thickness_requires_watertight(broken_path):
    with pytest.raises(MeshToolError):
        wall_thickness(load_mesh(broken_path))


def test_draft_box_pulled_up_has_undercut_bottom(box_path):
    info = draft_analysis(load_mesh(box_path), [0, 0, 1], min_draft_deg=3.0)
    # Side walls are vertical (0 draft); the bottom face is a full undercut.
    assert info.min_draft_angle_deg == pytest.approx(0.0, abs=1e-6)
    # box is 10x10x5: bottom face 100 of total area 400 -> 0.25.
    assert info.undercut_fraction == pytest.approx(0.25, abs=0.01)
    assert info.undercut_face_count == 2  # two triangles of the bottom face


def test_draft_zero_direction_rejected(box_path):
    with pytest.raises(MeshToolError):
        draft_analysis(load_mesh(box_path), [0, 0, 0])


def test_fit_region_sphere(sphere_path):
    loaded = load_mesh(sphere_path)
    region = BoxRegion(min=[-10, -10, -10], max=[10, 10, 10])  # whole sphere
    fit = fit_region(loaded, region, "sphere")
    assert fit.kind == "sphere"
    assert fit.radius == pytest.approx(2.0, rel=1e-3)
    assert fit.residual_rms < 1e-3


def test_fit_region_too_few_points_rejected(box_path):
    loaded = load_mesh(box_path)
    region = BoxRegion(min=[100, 100, 100], max=[200, 200, 200])  # selects nothing
    with pytest.raises(MeshToolError):
        fit_region(loaded, region, "plane")
