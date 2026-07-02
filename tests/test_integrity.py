import pytest

from mesh_validation_mcp.integrity import compute_integrity, integrity_flags
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.metrics import compute_metrics
from mesh_validation_mcp.validation import Expectations, evaluate


def _integrity(path):
    return compute_integrity(load_mesh(path).combined)


def test_clean_box_has_no_defects(box_path):
    m = _integrity(box_path)
    assert m.boundary_edge_count == 0
    assert m.non_manifold_edge_count == 0
    assert m.degenerate_face_count == 0
    assert m.sliver_face_count == 0
    assert m.duplicate_face_count == 0
    assert m.unmerged_vertex_count == 0
    assert m.unreferenced_vertex_count == 0
    assert m.flipped_face_count == 0
    assert m.self_intersecting_face_count == 0
    assert m.self_intersection_checked
    assert m.min_triangle_quality > 0.5


def test_clean_sphere_has_no_defects(sphere_path):
    m = _integrity(sphere_path)
    assert m.non_manifold_edge_count == 0
    assert m.self_intersecting_face_count == 0
    assert m.flipped_face_count == 0


def test_boundary_edges_on_open_mesh(broken_path):
    m = _integrity(broken_path)
    assert m.boundary_edge_count == 3


def test_non_manifold_edges(non_manifold_path):
    m = _integrity(non_manifold_path)
    assert m.non_manifold_edge_count > 0


def test_self_intersection(self_intersecting_path):
    m = _integrity(self_intersecting_path)
    assert m.self_intersecting_face_count == 12
    flags = integrity_flags(load_mesh(self_intersecting_path).combined)
    assert len(flags["self_intersection"]) == 12


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_self_intersection_is_scale_invariant(scale):
    # Regression: the Moller test must not depend on absolute mesh scale (the coplanar
    # threshold once compared a length^1 eps against a length^4 cross product).
    import trimesh

    from mesh_validation_mcp.integrity import _self_intersecting_faces

    a = trimesh.creation.box((10, 10, 10))
    b = trimesh.creation.box((10, 10, 10))
    b.apply_translation((5, 5, 5))
    mesh = trimesh.util.concatenate([a, b]).subdivide()  # small triangles
    mesh.apply_scale(scale)
    assert len(_self_intersecting_faces(mesh)) > 0


def test_duplicate_faces(duplicated_path):
    # STL merges coincident vertices on load, so the doubled surface shows up as
    # duplicate faces (and, once merged, non-manifold edges) rather than unmerged vertices.
    m = _integrity(duplicated_path)
    assert m.duplicate_face_count == 12
    assert m.non_manifold_edge_count > 0


def test_unmerged_vertices_in_memory(box):
    import trimesh

    from mesh_validation_mcp.integrity import compute_integrity

    # concatenate keeps the two boxes' vertices separate (no file round trip to merge them)
    doubled = trimesh.util.concatenate([box, box.copy()])
    m = compute_integrity(doubled)
    assert m.unmerged_vertex_count == 8


def test_flipped_faces(flipped_path):
    m = _integrity(flipped_path)
    assert m.flipped_face_count == 5


def test_metrics_include_integrity_and_caveat(self_intersecting_path):
    metrics = compute_metrics(load_mesh(self_intersecting_path))
    assert metrics.integrity.self_intersecting_face_count == 12
    assert any("self_intersecting" in c for c in metrics.caveats)


def test_validation_integrity_keys(self_intersecting_path):
    metrics = compute_metrics(load_mesh(self_intersecting_path))
    report = evaluate(metrics, Expectations.model_validate({"self_intersecting_face_count": 0}))
    assert not report.passed
    assert report.checks[0].actual == 12


def test_min_triangle_quality_lower_bound(box_path):
    metrics = compute_metrics(load_mesh(box_path))
    assert evaluate(
        metrics, Expectations.model_validate({"min_triangle_quality": 0.5})
    ).passed
    assert not evaluate(
        metrics, Expectations.model_validate({"min_triangle_quality": 0.99})
    ).passed


def test_manifold_box_passes_integrity_suite(box_path):
    metrics = compute_metrics(load_mesh(box_path))
    report = evaluate(
        metrics,
        Expectations.model_validate(
            {
                "non_manifold_edge_count": 0,
                "boundary_edge_count": 0,
                "self_intersecting_face_count": 0,
                "degenerate_face_count": 0,
                "duplicate_face_count": 0,
            }
        ),
    )
    assert report.passed
