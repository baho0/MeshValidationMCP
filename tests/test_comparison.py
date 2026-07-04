import numpy as np
import pytest
import trimesh

from mesh_validation_mcp.comparison import compare
from mesh_validation_mcp.loading import load_mesh


def _compare(path_a, path_b, **kwargs):
    return compare(load_mesh(path_a), load_mesh(path_b), **kwargs)


def test_identical(box_path):
    report, heatmap = _compare(box_path, box_path)
    assert report.classification == "identical"
    assert report.method == "procrustes_exact"
    assert report.distances.chamfer_mean < 1e-6
    assert report.transform.residual_rms < 1e-6
    assert len(heatmap) == 8
    assert heatmap.max() < 1e-6


def test_translation_detected(box_path, translated_path):
    report, _ = _compare(box_path, translated_path)
    assert report.classification == "translation"
    assert report.transform.translation == pytest.approx([5.0, 0.0, 0.0], abs=1e-3)
    assert report.transform.uniform_scale == pytest.approx(1.0, abs=1e-5)
    assert report.distances.aligned_residual_rms < 1e-3
    assert report.distances.chamfer_mean > 1.0  # raw distance shows the move
    assert "translated by [5" in report.summary


def test_rotation_detected(box_path, rotated_path):
    report, _ = _compare(box_path, rotated_path)
    assert report.classification == "rotation"
    assert report.transform.rotation_angle_deg == pytest.approx(30.0, abs=0.01)
    assert report.transform.rotation_axis[2] > 0.99
    assert np.linalg.norm(report.transform.translation) < 1e-3


def test_scale_detected(box_path, scaled_path):
    report, _ = _compare(box_path, scaled_path)
    assert report.classification == "similarity"
    assert report.transform.uniform_scale == pytest.approx(2.0, rel=1e-4)
    assert report.metric_deltas["volume"][0] == pytest.approx(500.0, rel=1e-5)
    assert report.metric_deltas["volume"][1] == pytest.approx(4000.0, rel=1e-5)


def test_icp_fallback_on_different_topology(box_path, subdivided_path):
    report, _ = _compare(box_path, subdivided_path)
    assert report.method == "icp"
    assert any("ICP" in c for c in report.caveats)
    assert report.classification != "deformed"
    assert report.transform.uniform_scale == pytest.approx(1.0, rel=1e-2)
    assert report.distances.chamfer_mean < 0.2


def test_icp_detects_scale_despite_topology_change(tmp_path, box_path, box):
    scaled = box.copy()
    scaled.apply_scale(2.0)
    scaled = scaled.subdivide()  # different topology forces the ICP path
    path = tmp_path / "scaled_subdivided.stl"
    scaled.export(path)
    report, _ = _compare(box_path, str(path))
    assert report.method == "icp"
    assert report.classification == "similarity"
    assert report.transform.uniform_scale == pytest.approx(2.0, rel=1e-2)


def test_mirrored_symmetric_part_dereflects(tmp_path, box_path, box):
    # A mirrored box is indistinguishable from an unmirrored one; the reflected
    # ICP fit must be converted back to a proper transform.
    mirrored = box.copy()
    mirrored.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
    mirrored = mirrored.subdivide()  # force the ICP path
    path = tmp_path / "mirrored.stl"
    mirrored.export(path)
    report, _ = _compare(box_path, str(path))
    assert report.method == "icp"
    assert report.transform.includes_reflection is False
    assert report.classification != "deformed"


def test_mirrored_asymmetric_part_reports_reflection(tmp_path, box):
    # An asymmetric part only fits its mirror when reflected; the report must say so.
    bump = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    bump.apply_translation((5.0, 3.0, 4.0))
    part = trimesh.util.concatenate([box, bump])
    part_path = tmp_path / "part.stl"
    part.export(part_path)

    mirrored = part.copy()
    mirrored.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
    mirrored_path = tmp_path / "part_mirrored.stl"
    mirrored.export(mirrored_path)

    report, _ = _compare(str(part_path), str(mirrored_path))
    assert report.classification == "mirrored"
    assert report.transform.includes_reflection is True
    assert any("reflection" in c or "mirrored" in c for c in report.caveats)
    assert "MIRRORED" in report.summary


def test_deformed(box_path, sphere_path):
    report, heatmap = _compare(box_path, sphere_path)
    assert report.classification == "deformed"
    assert report.distances.chamfer_mean > 0.5
    assert len(heatmap) == 642


def _same_topology(mesh, linear, path):
    m = mesh.copy()
    m.vertices = mesh.vertices @ np.asarray(linear, dtype=float).T
    m.faces = mesh.faces.copy()
    m.export(path)
    return str(path)


def test_anisotropic_scale_is_affine(tmp_path, box):
    a_path = str(tmp_path / "a.ply")
    box.export(a_path)
    b_path = _same_topology(box, np.diag([2.0, 1.0, 1.0]), tmp_path / "aniso.ply")
    report, _ = _compare(a_path, b_path)
    assert report.classification == "affine"
    assert report.transform.affine is not None
    assert report.transform.affine.anisotropic
    assert not report.transform.affine.has_shear
    assert sorted(report.transform.affine.singular_values, reverse=True)[0] == pytest.approx(2.0, rel=1e-3)


def test_shear_is_affine_with_shear_flag(tmp_path, box):
    a_path = str(tmp_path / "a.ply")
    box.export(a_path)
    b_path = _same_topology(box, [[1, 0.4, 0], [0, 1, 0], [0, 0, 1]], tmp_path / "shear.ply")
    report, _ = _compare(a_path, b_path)
    assert report.classification == "affine"
    assert report.transform.affine.has_shear
    assert report.transform.affine.determinant == pytest.approx(1.0, rel=1e-3)  # shear conserves volume


def test_rigid_transform_reports_invariants(box_path, rotated_path):
    report, _ = _compare(box_path, rotated_path)
    names = {c.name for c in report.transform_invariants}
    assert "volume_preserved" in names and "area_preserved" in names
    assert all(c.passed for c in report.transform_invariants)


def test_similarity_reports_scaling_invariants(box_path, scaled_path):
    report, _ = _compare(box_path, scaled_path)
    names = {c.name for c in report.transform_invariants}
    assert "volume_scales_cubically" in names
    assert all(c.passed for c in report.transform_invariants)


def test_metric_deltas_and_seed(box_path, translated_path):
    report, _ = _compare(box_path, translated_path, sample_count=500)
    # Sampling is adaptive: the requested count is a floor, not an exact target.
    assert report.distances.sample_count >= 500
    assert report.distances.seed == 0
    assert report.metric_deltas["face_count"] == [12, 12]
    assert report.metric_deltas["is_watertight"] == [True, True]


def test_hausdorff_is_bounded(box_path, translated_path):
    # The true Hausdorff must sit inside the reported [lower, upper] bracket, and the
    # bracket must carry a sampled-tier confidence with a positive spacing error bound.
    report, _ = _compare(box_path, translated_path)
    d = report.distances
    assert d.hausdorff_lower <= d.hausdorff_approx <= d.hausdorff_upper
    assert d.hausdorff_upper > d.hausdorff_lower  # a real (non-degenerate) bracket
    assert d.confidence is not None
    assert d.confidence.tier == "sampled"
    assert d.confidence.error_abs is not None and d.confidence.error_abs > 0


def test_identical_hausdorff_bracket_is_tight(box_path):
    report, _ = _compare(box_path, box_path)
    assert report.distances.hausdorff_lower < 1e-6
