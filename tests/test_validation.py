import pytest

from mesh_validation_mcp.errors import ErrorCode, MeshToolError
from mesh_validation_mcp.loading import load_mesh
from mesh_validation_mcp.metrics import compute_metrics
from mesh_validation_mcp.validation import Expectations, evaluate


@pytest.fixture
def box_metrics(box_path):
    return compute_metrics(load_mesh(box_path))


def test_all_checks_pass(box_metrics):
    report = evaluate(
        box_metrics,
        Expectations(
            volume=500.0,
            surface_area=400.0,
            bbox_extents=[10.0, 10.0, 5.0],
            centroid={"expected": [0.0, 0.0, 0.0], "abs_tol": 1e-6},
            vertex_count=8,
            face_count=12,
            watertight=True,
            winding_consistent=True,
            body_count=1,
            euler_number=2,
        ),
    )
    assert report.passed
    assert len(report.checks) == 10
    assert report.summary.startswith("10/10")


def test_volume_fail_reports_deviation(box_metrics):
    report = evaluate(box_metrics, Expectations(volume=900.0))
    assert not report.passed
    check = report.checks[0]
    assert check.name == "volume"
    assert check.deviation_pct == pytest.approx(-44.44, abs=0.05)
    assert "FAIL volume" in report.summary


def test_per_check_tolerance_override(box_metrics):
    report = evaluate(box_metrics, Expectations(volume={"expected": 505.0, "rel_tol": 0.02}))
    assert report.passed
    report = evaluate(box_metrics, Expectations(volume={"expected": 505.0, "rel_tol": 0.001}))
    assert not report.passed


def test_global_tolerance(box_metrics):
    exp = Expectations.model_validate({"tolerance": {"relative": 0.5}, "volume": 600.0})
    assert evaluate(box_metrics, exp).passed


def test_count_range(box_metrics):
    assert evaluate(
        box_metrics, Expectations(vertex_count={"min": 6, "max": 9})
    ).passed
    assert not evaluate(box_metrics, Expectations(vertex_count={"min": 9})).passed


def test_bool_check_fail(broken_path):
    metrics = compute_metrics(load_mesh(broken_path))
    report = evaluate(metrics, Expectations(watertight=True))
    assert not report.passed
    assert "FAIL watertight" in report.summary


def test_volume_caveat_on_open_mesh(broken_path):
    metrics = compute_metrics(load_mesh(broken_path))
    report = evaluate(metrics, Expectations(volume=500.0))
    assert any("not_watertight" in c for c in report.checks[0].caveats)


def test_unknown_key_rejected(box_metrics):
    exp = Expectations.model_validate({"volume": 500.0, "bounding_box": [1, 2, 3]})
    with pytest.raises(MeshToolError) as exc:
        evaluate(box_metrics, exp)
    assert exc.value.code is ErrorCode.INVALID_EXPECTATION
    assert "bounding_box" in exc.value.message
    assert "bbox_extents" in (exc.value.hint or "")


def test_empty_expectations_rejected(box_metrics):
    with pytest.raises(MeshToolError) as exc:
        evaluate(box_metrics, Expectations())
    assert exc.value.code is ErrorCode.INVALID_EXPECTATION


def test_vector_wrong_length_rejected(box_metrics):
    with pytest.raises(MeshToolError) as exc:
        evaluate(box_metrics, Expectations(bbox_extents=[10.0, 10.0]))
    assert exc.value.code is ErrorCode.INVALID_EXPECTATION


def test_count_check_without_bounds_rejected(box_metrics):
    with pytest.raises(MeshToolError) as exc:
        evaluate(box_metrics, Expectations(vertex_count={}))
    assert exc.value.code is ErrorCode.INVALID_EXPECTATION
