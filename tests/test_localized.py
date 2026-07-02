import pytest

from mesh_validation_mcp.comparison import localized_change
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.validation import ChangeExpectations, evaluate_change

# Region covering the top surface of the plate (z above the slab) and the disc footprint.
TOP_REGION = {"kind": "box", "min": [-12, -12, 1.5], "max": [12, 12, 8]}


def _change(path_a, path_b, region=TOP_REGION, threshold=None):
    a, b = load_mesh(path_a), load_mesh(path_b)
    exp = ChangeExpectations.model_validate({"region": region, "change_threshold": threshold})
    return localized_change(a, b, exp.region, exp.change_threshold)


def test_good_emboss_stats(plate_path, emboss_good_path):
    change = _change(plate_path, emboss_good_path)
    assert change.same_topology
    assert change.outside_max_displacement == pytest.approx(0.0, abs=1e-6)
    assert change.inside_max_displacement == pytest.approx(3.0, abs=1e-4)
    assert change.signed_displacement.peak == pytest.approx(3.0, abs=1e-4)
    assert change.signed_displacement.outward_fraction == pytest.approx(1.0)
    assert change.changed_region_bounds is not None


def test_bad_emboss_leaks_outside_region(plate_path, emboss_bad_path):
    change = _change(plate_path, emboss_bad_path)
    # the bottom of the column moved too -> displacement outside the top region
    assert change.outside_max_displacement == pytest.approx(3.0, abs=1e-4)


def test_good_emboss_passes_roi_checks(plate_path, emboss_good_path):
    a, b = load_mesh(plate_path), load_mesh(emboss_good_path)
    exp = ChangeExpectations.model_validate(
        {"region": TOP_REGION, "emboss_height": 3.0, "max_unchanged_deviation": 0.01}
    )
    report = evaluate_change(localized_change(a, b, exp.region, exp.change_threshold), exp)
    assert report.passed
    assert "2/2" in report.summary


def test_bad_emboss_fails_untouched_check(plate_path, emboss_bad_path):
    a, b = load_mesh(plate_path), load_mesh(emboss_bad_path)
    exp = ChangeExpectations.model_validate(
        {"region": TOP_REGION, "emboss_height": 3.0, "max_unchanged_deviation": 0.01}
    )
    report = evaluate_change(localized_change(a, b, exp.region, exp.change_threshold), exp)
    assert not report.passed
    assert "untouched_region_max_deviation" in report.summary


def test_pocket_depth_is_signed(plate_path, tmp_path):
    # A downward (inward) move of the top surface is a pocket: signed peak is negative.
    import trimesh

    plate = load_mesh(plate_path).combined.copy()
    import numpy as np

    v = plate.vertices.copy()
    disc = (np.linalg.norm(v[:, :2], axis=1) < 10.0) & (v[:, 2] > v[:, 2].max() - 1e-6)
    v[disc, 2] -= 1.5
    plate.vertices = v
    pocket_path = str(tmp_path / "pocket.stl")
    plate.export(pocket_path)

    a, b = load_mesh(plate_path), load_mesh(pocket_path)
    exp = ChangeExpectations.model_validate({"region": TOP_REGION, "pocket_depth": 1.5})
    change = localized_change(a, b, exp.region, exp.change_threshold)
    assert change.signed_displacement.peak == pytest.approx(-1.5, abs=1e-4)
    report = evaluate_change(change, exp)
    assert report.passed


def test_changed_region_is_scoped_to_region(plate_path, emboss_bad_path):
    # Regression: changed_region_* must describe the change INSIDE the region, not globally.
    # The bad emboss also moves the bottom, but the top region's changed extent stays on top.
    change = _change(plate_path, emboss_bad_path)
    assert change.outside_max_displacement == pytest.approx(3.0, abs=1e-4)  # leak detected
    assert change.changed_region_bounds is not None
    assert change.changed_region_bounds.min[2] >= 1.5  # in-region change only (top surface)


def test_differing_topology_emboss_uses_footprint(plate_path, tmp_path):
    # Regression: on differing topology, a raised region must still be measured via footprint.
    import numpy as np

    plate = load_mesh(plate_path).combined.copy().subdivide()  # break vertex correspondence
    v = plate.vertices.copy()
    sel = (np.linalg.norm(v[:, :2], axis=1) < 10) & (v[:, 2] > v[:, 2].max() - 1e-6)
    v[sel, 2] += 3.0
    plate.vertices = v
    emb = str(tmp_path / "emb_diff.stl")
    plate.export(emb)

    a, b = load_mesh(plate_path), load_mesh(emb)
    exp = ChangeExpectations.model_validate({"region": TOP_REGION, "emboss_height": 3.0})
    change = localized_change(a, b, exp.region, exp.change_threshold)
    assert not change.same_topology
    assert change.signed_displacement.peak == pytest.approx(3.0, abs=0.1)
    assert evaluate_change(change, exp).passed


def test_differing_topology_pocket_is_indeterminate_not_zero(plate_path, tmp_path):
    # Regression: a deep pocket on differing topology whose floor maps outside the region
    # must report peak=None (indeterminate) rather than a misleading 0.0 -> spurious FAIL.
    import numpy as np

    plate = load_mesh(plate_path).combined.copy().subdivide()
    v = plate.vertices.copy()
    sel = (np.linalg.norm(v[:, :2], axis=1) < 10) & (v[:, 2] > v[:, 2].max() - 1e-6)
    v[sel, 2] -= 1.5
    plate.vertices = v
    pocket = str(tmp_path / "pocket_diff.stl")
    plate.export(pocket)

    a, b = load_mesh(plate_path), load_mesh(pocket)
    change = localized_change(a, b, ChangeExpectations.model_validate({"region": TOP_REGION}).region)
    assert not change.same_topology
    assert change.signed_displacement.peak is None
    assert any("indeterminate" in c for c in change.caveats)


def test_region_only_reports_stats(plate_path, emboss_good_path):
    a, b = load_mesh(plate_path), load_mesh(emboss_good_path)
    exp = ChangeExpectations.model_validate({"region": TOP_REGION})
    report = evaluate_change(localized_change(a, b, exp.region, exp.change_threshold), exp)
    assert report.passed  # no assertions -> vacuously passed
    assert report.checks == []
