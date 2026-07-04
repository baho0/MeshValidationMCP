"""Phase 6 — clearance, silhouette, units/self-consistency, DfM (optional geometric checks)."""

import pytest
import trimesh

from mesh_validation_mcp.clearance import check_clearance
from mesh_validation_mcp.dfm import DfmExpectations, dfm_report
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.silhouette import silhouette_compare
from mesh_validation_mcp.units import units_report


# --- 6.1 clearance ---


def _box_path(tmp_path, name, translate=(0, 0, 0)):
    b = trimesh.creation.box((10, 10, 10))
    b.apply_translation(translate)
    p = str(tmp_path / f"{name}.stl")
    b.export(p)
    return p


def test_interference_detected(tmp_path):
    a = _box_path(tmp_path, "a")
    overlap = _box_path(tmp_path, "ov", (5, 0, 0))
    info = check_clearance(load_mesh(a), load_mesh(overlap))
    assert info.interfering
    assert info.max_penetration == pytest.approx(5.0, abs=0.3)


def test_clearance_gap_measured(tmp_path):
    a = _box_path(tmp_path, "a")
    clear = _box_path(tmp_path, "cl", (13, 0, 0))  # 3mm gap
    info = check_clearance(load_mesh(a), load_mesh(clear), min_clearance_required=2.0)
    assert not info.interfering
    assert info.min_clearance == pytest.approx(3.0, abs=0.1)
    assert info.meets_requirement is True


def test_clearance_requirement_not_met(tmp_path):
    a = _box_path(tmp_path, "a")
    clear = _box_path(tmp_path, "cl", (11, 0, 0))  # 1mm gap
    info = check_clearance(load_mesh(a), load_mesh(clear), min_clearance_required=2.0)
    assert info.meets_requirement is False


# --- 6.3 silhouette ---


def test_identical_silhouette_iou_one(box_path, tmp_path, box):
    copy = str(tmp_path / "copy.stl")
    box.export(copy)
    s = silhouette_compare(load_mesh(box_path), load_mesh(copy), "z")
    assert s.iou == pytest.approx(1.0, abs=1e-6)


def test_different_shape_lower_iou(box_path, sphere_path):
    s = silhouette_compare(load_mesh(box_path), load_mesh(sphere_path), "z")
    assert s.iou < 0.95


def test_emboss_preserves_silhouette(plate_path, emboss_good_path):
    # A top-surface emboss must not change the plate's top-down outline.
    s = silhouette_compare(load_mesh(plate_path), load_mesh(emboss_good_path), "z")
    assert s.iou > 0.99


# --- 6.4 units / self-consistency ---


def test_units_scale_plausibility(box_path):
    report = units_report(load_mesh(box_path), plausible_diagonal_range=(1.0, 100.0))
    assert report.plausible is True
    assert report.deterministic  # same seed reproduces exactly
    assert report.probe_agreement  # two seeds agree within the sampling bound


def test_units_implausible_scale_flagged(box_path):
    report = units_report(load_mesh(box_path), plausible_diagonal_range=(1000.0, 2000.0))
    assert report.plausible is False
    assert any("outside the plausible range" in c for c in report.caveats)


# --- 6.2 DfM (composite) ---


def test_dfm_flags_thin_wall_and_trapped_void(shell_path):
    # The hollow-cube shell has a fully enclosed cavity and ~2mm walls.
    report = dfm_report(
        shell_path_loaded := load_mesh(shell_path),
        DfmExpectations(min_wall_thickness=5.0, check_trapped_voids=True, sample_count=1200),
    )
    assert not report.passed
    assert any(c.name == "no_trapped_voids" and not c.passed for c in report.checks)


def test_dfm_draft_passes_for_pullable_part(tmp_path):
    # A wedge with generous draft and no undercut in +z, single solid (no voids).
    wedge = trimesh.creation.box((10, 10, 10))
    wedge.apply_transform(trimesh.transformations.rotation_matrix(0.2, [1, 0, 0]))
    path = str(tmp_path / "wedge.stl")
    wedge.export(path)
    report = dfm_report(
        load_mesh(path),
        DfmExpectations(pull_direction=[0, 0, 1], max_undercut_area=1e9, check_trapped_voids=True),
    )
    # max_undercut_area huge -> undercut check passes; single solid -> no trapped voids.
    assert report.passed
