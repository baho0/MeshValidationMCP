"""Phase 2.1 — universal property oracles."""

import numpy as np
import pytest
import trimesh

from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.metrics import compute_metrics
from mesh_validation_mcp.oracles import PropertySpec, run_oracles


def _metrics(path):
    return compute_metrics(load_mesh(path))


def _spec(name, **kw):
    return PropertySpec.model_validate({"name": name, **kw})


def test_conserves_volume_holds_for_rigid(box_path, translated_path):
    before, after = _metrics(box_path), _metrics(translated_path)
    report = run_oracles(after, [_spec("conserves_volume")], before=before)
    assert report.passed
    assert report.checks[0].confidence.tier == "exact"


def test_conserves_volume_fails_for_scale(box_path, scaled_path):
    before, after = _metrics(box_path), _metrics(scaled_path)
    report = run_oracles(after, [_spec("conserves_volume")], before=before)
    assert not report.passed


def test_binary_oracle_without_reference_fails(box_path):
    report = run_oracles(_metrics(box_path), [_spec("conserves_volume")])
    assert not report.passed
    assert "needs a reference" in report.checks[0].caveats[0]


def test_non_self_intersecting_fails_closed(self_intersecting_path):
    report = run_oracles(_metrics(self_intersecting_path), [_spec("non_self_intersecting")])
    assert not report.passed
    assert report.checks[0].actual == 12


def test_no_new_defects_detects_regression(box_path, self_intersecting_path):
    before, after = _metrics(box_path), _metrics(self_intersecting_path)
    report = run_oracles(after, [_spec("no_new_defects")], before=before)
    assert not report.passed


def test_preserves_watertight_and_euler(box_path, translated_path):
    before, after = _metrics(box_path), _metrics(translated_path)
    report = run_oracles(
        after,
        [_spec("preserves_watertight"), _spec("preserves_euler"), _spec("preserves_genus")],
        before=before,
    )
    assert report.passed
    genus = next(c for c in report.checks if c.name == "preserves_genus")
    assert genus.expected == 0.0  # a box is genus 0


def test_monotonic_offset_direction(tmp_path, box):
    box.export(tmp_path / "a.ply")
    bigger = box.copy()
    bigger.vertices = box.vertices * 1.1  # inflate -> volume grows (outward)
    bigger.faces = box.faces.copy()
    bigger.export(tmp_path / "b.ply")
    before = _metrics(str(tmp_path / "a.ply"))
    after = _metrics(str(tmp_path / "b.ply"))
    assert run_oracles(after, [_spec("monotonic_offset", direction="outward")], before=before).passed
    assert not run_oracles(
        after, [_spec("monotonic_offset", direction="inward")], before=before
    ).passed


def test_centroid_fixed(box_path, rotated_path, translated_path):
    before = _metrics(box_path)
    # rotation about the centroid keeps it fixed; a translation moves it
    assert run_oracles(_metrics(rotated_path), [_spec("centroid_fixed")], before=before).passed
    assert not run_oracles(
        _metrics(translated_path), [_spec("centroid_fixed")], before=before
    ).passed


def test_bounded_hausdorff_requires_max_distance(box_path):
    from mesh_validation_mcp.errors import MeshToolError

    with pytest.raises(MeshToolError):
        run_oracles(_metrics(box_path), [_spec("bounded_hausdorff")], before=_metrics(box_path))


def test_bounded_hausdorff_check(box_path):
    report = run_oracles(
        _metrics(box_path),
        [_spec("bounded_hausdorff", max_distance=1.0)],
        before=_metrics(box_path),
        hausdorff_upper=0.3,
    )
    assert report.passed
    report2 = run_oracles(
        _metrics(box_path),
        [_spec("bounded_hausdorff", max_distance=0.1)],
        before=_metrics(box_path),
        hausdorff_upper=0.3,
    )
    assert not report2.passed
