"""Phase 5 — symmetry, array, generative, deformation, remesh."""

import math

import numpy as np
import pytest
import trimesh

from mesh_validation_mcp.array_validate import ArrayPattern, validate_array
from mesh_validation_mcp.comparison import signed_field
from mesh_validation_mcp.generative_validate import (
    ExtrudeExpectations,
    RevolveExpectations,
    validate_extrude,
    validate_revolve,
)
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.remesh_validate import RemeshExpectations, validate_remesh
from mesh_validation_mcp.symmetry import detect_symmetry


# --- 5.1 symmetry ---


def test_box_has_three_mirror_planes(box_path):
    sym = detect_symmetry(load_mesh(box_path))
    assert len(sym.mirror_planes) == 3
    assert sym.rotational_fold == 4  # square cross-section about the tall axis


def test_asymmetric_part_has_fewer_planes(tmp_path):
    box = trimesh.creation.box((10, 10, 5))
    bump = trimesh.creation.box((3, 3, 3))
    bump.apply_translation((4, 0, 3))  # breaks x-symmetry, keeps the y-mirror
    part = trimesh.util.concatenate([box, bump])
    path = str(tmp_path / "part.stl")
    part.export(path)
    sym = detect_symmetry(load_mesh(path))
    assert len(sym.mirror_planes) < 3


# --- 5.2 array ---


def _linear_array(tmp_path, count, step):
    base = trimesh.creation.box((4, 4, 4))
    base_path = str(tmp_path / "base.stl")
    base.export(base_path)
    insts = [base.copy().apply_translation(np.array(step) * k) for k in range(count)]
    arr_path = str(tmp_path / "arr.stl")
    trimesh.util.concatenate(insts).export(arr_path)
    return arr_path, base_path


def test_linear_array_passes(tmp_path):
    arr_path, base_path = _linear_array(tmp_path, 4, [20, 0, 0])
    report = validate_array(
        load_mesh(arr_path), load_mesh(base_path), ArrayPattern(kind="linear", count=4, step=[20, 0, 0])
    )
    assert report.passed


def test_array_wrong_count_fails(tmp_path):
    arr_path, base_path = _linear_array(tmp_path, 3, [20, 0, 0])
    report = validate_array(
        load_mesh(arr_path), load_mesh(base_path), ArrayPattern(kind="linear", count=5, step=[20, 0, 0])
    )
    assert not report.passed
    assert any(c.name == "instance_count" and not c.passed for c in report.checks)


def test_array_wrong_spacing_fails(tmp_path):
    arr_path, base_path = _linear_array(tmp_path, 3, [20, 0, 0])
    report = validate_array(
        load_mesh(arr_path), load_mesh(base_path), ArrayPattern(kind="linear", count=3, step=[25, 0, 0])
    )
    assert not report.passed
    assert any(c.name == "instance_positions" and not c.passed for c in report.checks)


def test_polar_array_passes(tmp_path):
    base = trimesh.creation.box((2, 2, 2))
    base.apply_translation((10, 0, 0))
    base_path = str(tmp_path / "pbase.stl")
    base.export(base_path)
    insts = []
    for k in range(6):
        rot = trimesh.transformations.rotation_matrix(k * math.pi / 3, [0, 0, 1], [0, 0, 0])
        insts.append(base.copy().apply_transform(rot))
    arr_path = str(tmp_path / "polar.stl")
    trimesh.util.concatenate(insts).export(arr_path)
    report = validate_array(
        load_mesh(arr_path),
        load_mesh(base_path),
        ArrayPattern(kind="polar", count=6, axis=[0, 0, 1], center=[0, 0, 0], angle_deg=60),
    )
    assert report.passed


# --- 5.3 generative ---


def test_extrude_volume_and_constant_section(box_path):
    report = validate_extrude(
        load_mesh(box_path), ExtrudeExpectations(profile_area=100.0, height=5.0, axis=[0, 0, 1])
    )
    assert report.passed


def test_extrude_wrong_area_fails(box_path):
    report = validate_extrude(
        load_mesh(box_path), ExtrudeExpectations(profile_area=80.0, height=5.0, axis=[0, 0, 1])
    )
    assert not report.passed


def test_revolve_pappus(tmp_path):
    cyl = trimesh.creation.cylinder(radius=3.0, height=10.0)
    path = str(tmp_path / "cyl.stl")
    cyl.export(path)
    # rect profile 3 wide x 10 tall = area 30, centroid at radius 1.5 -> V = 2*pi*1.5*30 = 90pi
    report = validate_revolve(
        load_mesh(path), RevolveExpectations(profile_area=30.0, profile_centroid_radius=1.5)
    )
    assert report.passed


# --- 5.4 signed field / deformation ---


def test_signed_field_inflation_is_coherent_outward(tmp_path, box):
    box.export(tmp_path / "a.ply")
    inflated = box.copy()
    inflated.vertices = box.vertices * 1.1
    inflated.faces = box.faces.copy()
    inflated.export(tmp_path / "b.ply")
    field = signed_field(load_mesh(str(tmp_path / "a.ply")), load_mesh(str(tmp_path / "b.ply")))
    assert field.same_topology
    assert field.direction_consistency == pytest.approx(1.0, abs=1e-6)
    assert field.outward_fraction == pytest.approx(1.0)
    assert field.volume_delta > 0
    assert field.signed_peak > 0


def test_signed_field_reports_volume_loss_on_pocket(tmp_path, box):
    box.export(tmp_path / "a.ply")
    pocket = box.copy()
    v = box.vertices.copy()
    top = v[:, 2] > v[:, 2].max() - 1e-6
    v[top, 2] -= 1.0  # push the top face down
    pocket.vertices = v
    pocket.faces = box.faces.copy()
    pocket.export(tmp_path / "b.ply")
    field = signed_field(load_mesh(str(tmp_path / "a.ply")), load_mesh(str(tmp_path / "b.ply")))
    assert field.volume_delta < 0
    assert field.signed_peak < 0  # inward


# --- 5.5 remesh ---


def test_subdivide_preserves_shape_and_topology(box_path, tmp_path, box):
    sub = box.copy().subdivide()
    sub_path = str(tmp_path / "sub.stl")
    sub.export(sub_path)
    report = validate_remesh(
        load_mesh(box_path),
        load_mesh(sub_path),
        RemeshExpectations(max_deviation=0.01, min_triangle_quality=0.3),
    )
    assert report.passed


def test_remesh_detects_shape_change(box_path, sphere_path):
    report = validate_remesh(
        load_mesh(box_path), load_mesh(sphere_path), RemeshExpectations(max_deviation=0.01)
    )
    assert not report.passed
    assert any(c.name == "shape_preserved" and not c.passed for c in report.checks)
