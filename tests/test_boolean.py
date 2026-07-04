"""Phase 4.5 — boolean / CSG validation."""

from mesh_validation_mcp.boolean_validate import BooleanExpectations, validate_boolean
from mesh_validation_mcp.loading import load_mesh


def _run(paths, operation, result_key=None, tolerance=0.02):
    result_key = result_key or operation
    return validate_boolean(
        load_mesh(paths["a"]),
        load_mesh(paths["b"]),
        load_mesh(paths[result_key]),
        BooleanExpectations(operation=operation, tolerance=tolerance),
    )


def test_union_validates(boolean_operands):
    report = _run(boolean_operands, "union")
    assert report.passed
    assert {"result_watertight", "union_volume_bounds", "union_contains_operands"} <= {
        c.name for c in report.checks
    }


def test_difference_validates(boolean_operands):
    report = _run(boolean_operands, "difference")
    assert report.passed


def test_intersection_validates(boolean_operands):
    report = _run(boolean_operands, "intersection")
    assert report.passed


def test_wrong_operation_fails(boolean_operands):
    # The union result (volume 1500) cannot be a difference of A (1000): volume bound breaks.
    report = _run(boolean_operands, "difference", result_key="union")
    assert not report.passed
    assert any(c.name == "difference_volume_bounds" and not c.passed for c in report.checks)


def test_open_operand_fails_closed(boolean_operands, broken_path):
    report = validate_boolean(
        load_mesh(broken_path),
        load_mesh(boolean_operands["b"]),
        load_mesh(boolean_operands["union"]),
        BooleanExpectations(operation="union"),
    )
    volume_check = next(c for c in report.checks if "volume" in c.name)
    assert not volume_check.passed  # unreliable operand volume -> no PASS
