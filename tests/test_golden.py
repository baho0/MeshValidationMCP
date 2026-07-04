"""Phase 2.2 — golden / reference comparison."""

import numpy as np
import pytest

from mesh_validation_mcp.golden import compare_to_reference
from mesh_validation_mcp.loading import load_mesh


def test_identical_is_exact_match(box_path, tmp_path, box):
    copy_path = str(tmp_path / "copy.stl")
    box.export(copy_path)
    result = compare_to_reference(load_mesh(copy_path), load_mesh(box_path), tolerance=1e-4)
    assert result.matches
    assert result.exact_match
    assert result.method == "exact_vertex"
    assert result.max_vertex_delta == pytest.approx(0.0, abs=1e-6)
    assert result.confidence.tier == "exact"


def test_reordered_vertices_still_exact(tmp_path, box):
    # A vertex permutation must still count as an exact match (canonicalized comparison).
    ref_path = str(tmp_path / "ref.ply")
    box.export(ref_path)
    shuffled = box.copy()
    perm = np.array([7, 0, 3, 1, 6, 2, 5, 4])
    remap = np.argsort(perm)
    shuffled.vertices = box.vertices[perm]
    shuffled.faces = remap[box.faces]
    prod_path = str(tmp_path / "prod.ply")
    shuffled.export(prod_path)
    result = compare_to_reference(load_mesh(prod_path), load_mesh(ref_path), tolerance=1e-4)
    assert result.exact_match


def test_small_move_exceeds_tolerance(tmp_path, box):
    ref_path = str(tmp_path / "ref.stl")
    box.export(ref_path)
    moved = box.copy()
    moved.apply_translation((0.5, 0, 0))
    prod_path = str(tmp_path / "prod.stl")
    moved.export(prod_path)
    result = compare_to_reference(load_mesh(prod_path), load_mesh(ref_path), tolerance=1e-3)
    assert not result.matches
    assert result.max_vertex_delta == pytest.approx(0.5, abs=1e-4)


def test_differing_topology_uses_surface_distance(tmp_path, box):
    ref_path = str(tmp_path / "ref.stl")
    box.export(ref_path)
    subdivided = box.copy().subdivide()  # same shape, different topology
    prod_path = str(tmp_path / "prod.stl")
    subdivided.export(prod_path)
    result = compare_to_reference(load_mesh(prod_path), load_mesh(ref_path), tolerance=1e-2)
    assert result.method == "vertex_to_surface"
    assert result.matches  # same surface, distance ~ 0
    assert result.surface_distance == pytest.approx(0.0, abs=1e-6)
    assert result.confidence.tier == "sampled"
    assert any("topology differs" in c for c in result.caveats)
