"""Phase 1 — trust foundation: confidence tiers, fail-closed volume, cross-checks,
format-aware loading, and same-topology reflection detection."""

import math

import numpy as np
import pytest
import trimesh

from mesh_validation_mcp import confidence
from mesh_validation_mcp.comparison import compare
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.metrics import compute_metrics
from mesh_validation_mcp.validation import Expectations, evaluate


# --- confidence.py primitives ---


def test_sampling_error_shrinks_with_n():
    assert confidence.sampling_error(100, 10.0) == pytest.approx(1.0)
    assert confidence.sampling_error(10000, 10.0) == pytest.approx(0.1)
    assert confidence.sampling_error(0, 10.0) == math.inf


def test_tier_helpers():
    assert confidence.exact("x").tier == "exact"
    assert confidence.topological("x").tier == "topological"
    assert confidence.sampled("x", 0.5).tier == "sampled"
    assert confidence.estimated("x", 0.3).error_abs == 0.3


def test_with_reference_fills_relative():
    c = confidence.sampled("x", 0.2).with_reference(4.0)
    assert c.error_rel == pytest.approx(0.05)


# --- 1.1 confidence rides on every check ---


def test_checks_carry_confidence(box_path):
    metrics = compute_metrics(load_mesh(box_path))
    report = evaluate(metrics, Expectations(volume=500.0, vertex_count=8, body_count=1))
    tiers = {c.name: c.confidence.tier for c in report.checks}
    assert tiers["volume"] == "exact"
    assert tiers["vertex_count"] == "exact"
    assert tiers["body_count"] == "topological"


# --- 1.2 fail-closed volume ---


def test_volume_fails_closed_on_open_mesh(broken_path):
    metrics = compute_metrics(load_mesh(broken_path))
    report = evaluate(metrics, Expectations(volume=500.0))
    assert not report.passed  # no silent PASS on an unreliable volume
    assert report.checks[0].confidence.tier == "estimated"
    assert any("refusing to PASS" in c for c in report.checks[0].caveats)


def test_volume_fails_closed_on_self_intersection(self_intersecting_path):
    # Two overlapping boxes read as watertight but the divergence volume is wrong.
    metrics = compute_metrics(load_mesh(self_intersecting_path))
    assert metrics.is_watertight
    report = evaluate(metrics, Expectations(volume=metrics.volume))
    assert not report.passed
    assert any("self-intersect" in c for c in report.checks[0].caveats)


def test_healthy_volume_still_passes(box_path):
    metrics = compute_metrics(load_mesh(box_path))
    assert evaluate(metrics, Expectations(volume=500.0)).passed


# --- 1.5 cross checks ---


def test_cross_checks_consistent_on_clean_mesh(box_path):
    cc = compute_metrics(load_mesh(box_path)).cross_checks
    assert cc.consistent
    assert cc.euler_agreement
    assert cc.euler_gauss_bonnet == pytest.approx(2.0, abs=1e-6)
    assert cc.watertight_agreement


def test_cross_checks_flag_open_mesh(broken_path):
    cc = compute_metrics(load_mesh(broken_path)).cross_checks
    # An open mesh: is_watertight is False and there are boundary edges, so the two agree,
    # but the Euler number no longer matches a closed surface's Gauss-Bonnet total.
    assert cc.watertight_flag is False
    assert cc.boundary_free is False


# --- 1.4 format-aware loading + same-topology reflection ---


def test_stl_is_merged_on_load(box_path):
    # STL stores unshared vertices; load must merge them back to the 8 box corners.
    assert len(load_mesh(box_path).combined.vertices) == 8


def test_ply_preserves_vertex_order(tmp_path, box):
    path = tmp_path / "box.ply"
    box.export(path)
    loaded = load_mesh(str(path))
    assert len(loaded.combined.vertices) == len(box.vertices)


def _mirror_same_topology(mesh, path):
    """Reflect X but keep the face array => same topology, opposite handedness."""
    m = mesh.copy()
    m.vertices = mesh.vertices * np.array([-1.0, 1.0, 1.0])
    m.faces = mesh.faces.copy()
    m.export(path)
    return str(path)


def test_same_topology_mirror_is_detected(tmp_path):
    box = trimesh.creation.box((10, 10, 5))
    bump = trimesh.creation.box((4, 4, 4))
    bump.apply_translation((5, 3, 4))
    part = trimesh.util.concatenate([box, bump])  # asymmetric: a genuine mirror
    a_path = str(tmp_path / "part.ply")
    part.export(a_path)
    b_path = _mirror_same_topology(part, tmp_path / "mirror.ply")

    report, _ = compare(load_mesh(a_path), load_mesh(b_path))
    assert report.method == "procrustes_exact"  # matching topology
    assert report.classification == "mirrored"
    assert report.transform.includes_reflection is True
