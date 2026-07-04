"""Phase 3.2-3.5 — curvature, primitive fits, mass/inertia, topology."""

import math

import numpy as np
import pytest
import trimesh

from mesh_validation_mcp.curvature import curvature_field
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.metrics import compute_metrics
from mesh_validation_mcp.primitives import fit_cylinder, fit_plane, fit_sphere
from mesh_validation_mcp.topology import analyze_topology


# --- 3.2 curvature ---


def test_box_has_twelve_sharp_edges(box_path):
    c = curvature_field(load_mesh(box_path))
    assert c.sharp_edge_count == 12
    assert c.max_dihedral_deg == pytest.approx(90.0, abs=1e-3)
    assert c.gaussian_over_2pi == pytest.approx(2.0, abs=1e-6)


def test_sphere_has_no_sharp_edges(sphere_path):
    c = curvature_field(load_mesh(sphere_path))
    assert c.sharp_edge_count == 0
    assert c.max_dihedral_deg < 30.0


# --- 3.3 primitives ---


def test_fit_sphere_recovers_radius(sphere_path):
    sphere = load_mesh(sphere_path).combined
    fit = fit_sphere(sphere.vertices)
    assert fit.radius == pytest.approx(2.0, rel=1e-3)
    assert fit.residual_rms < 1e-3
    assert fit.center == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_fit_plane_recovers_normal():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200), np.zeros(200)])
    fit = fit_plane(pts)
    assert abs(abs(fit.normal[2]) - 1.0) < 1e-6
    assert fit.residual_rms < 1e-9


def test_fit_cylinder_recovers_radius_on_wall_points():
    theta = np.linspace(0, 2 * math.pi, 60, endpoint=False)
    z = np.linspace(-5, 5, 20)
    tt, zz = np.meshgrid(theta, z)
    pts = np.column_stack(
        [3.0 * np.cos(tt).ravel(), 3.0 * np.sin(tt).ravel(), zz.ravel()]
    )
    fit = fit_cylinder(pts)
    assert fit.radius == pytest.approx(3.0, rel=1e-3)
    assert abs(abs(fit.axis[2]) - 1.0) < 1e-3
    assert fit.residual_rms < 1e-3


# --- 3.4 mass / inertia ---


def test_box_inertia(box_path):
    m = compute_metrics(load_mesh(box_path))
    assert m.inertia is not None
    # Solid box 10x10x5, mass=volume=500: I_z = m/12 (a^2+b^2) = 500/12*(100+100)=8333.3
    assert max(m.inertia.principal_moments) == pytest.approx(8333.33, rel=1e-3)


def test_open_mesh_has_no_inertia(broken_path):
    m = compute_metrics(load_mesh(broken_path))
    assert m.inertia is None  # gated on a reliable solid


# --- 3.5 topology ---


def test_box_is_genus_zero(box_path):
    t = analyze_topology(load_mesh(box_path))
    assert t.genus_total == 0.0
    assert t.boundary_loop_count == 0
    assert t.all_bodies_watertight


def test_torus_is_genus_one(tmp_path):
    torus = trimesh.creation.torus(3.0, 1.0)
    path = str(tmp_path / "torus.stl")
    torus.export(path)
    t = analyze_topology(load_mesh(path))
    assert t.genus_total == pytest.approx(1.0)


def test_open_mesh_reports_boundary_loop(broken_path):
    t = analyze_topology(load_mesh(broken_path))
    assert not t.all_bodies_watertight
    assert t.genus_total is None
    assert t.boundary_loop_count == 1  # one removed face -> one boundary loop


def test_two_bodies_topology(two_bodies_path):
    t = analyze_topology(load_mesh(two_bodies_path))
    assert t.body_count == 2
    assert t.genus_total == 0.0
    assert t.per_body_genus == [0.0, 0.0]
