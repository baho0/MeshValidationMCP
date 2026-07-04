"""Expectations schema and the deterministic assertion engine."""

from __future__ import annotations

from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence, exact, sampled, topological
from .errors import ErrorCode, MeshToolError
from .metrics import MeshMetrics
from .region import Region

if TYPE_CHECKING:
    from .comparison import LocalizedChange

# Integrity keys map directly onto IntegrityMetrics fields (typically expected 0).
_INTEGRITY_COUNT_KEYS = (
    "boundary_edge_count",
    "non_manifold_edge_count",
    "degenerate_face_count",
    "sliver_face_count",
    "duplicate_face_count",
    "unmerged_vertex_count",
    "unreferenced_vertex_count",
    "flipped_face_count",
    "self_intersecting_face_count",
)

# Matched against MeshMetrics fields; order here is the evaluation/report order.
SUPPORTED_KEYS = (
    "volume",
    "surface_area",
    "bbox_min",
    "bbox_max",
    "bbox_extents",
    "centroid",
    "vertex_count",
    "face_count",
    "watertight",
    "winding_consistent",
    "body_count",
    "euler_number",
    *_INTEGRITY_COUNT_KEYS,
    "min_triangle_quality",
)

_TINY = 1e-12

# Confidence basis per check key. Everything an assertion touches here is an exact
# combinatorial or analytic quantity except body_count (a topological invariant),
# volume on an unreliable mesh, and an unverified self-intersection count.
_EXACT_BASIS = {
    "volume": "divergence-theorem volume (watertight, consistent winding)",
    "surface_area": "sum of triangle areas",
    "bbox_min": "coordinate extrema",
    "bbox_max": "coordinate extrema",
    "bbox_extents": "coordinate extrema",
    "centroid": "area-weighted triangle centroid",
    "vertex_count": "vertex array length",
    "face_count": "face array length",
    "watertight": "edge-manifold test",
    "winding_consistent": "winding-consistency test",
    "euler_number": "V - E + F",
    "min_triangle_quality": "analytic triangle quality",
}
_TOPO_BASIS = {"body_count": "connected-component count"}


def _confidence_for(name: str, metrics: MeshMetrics) -> Confidence:
    """Label a check's `actual` value with how it was derived (see confidence.py)."""
    if name == "volume":
        if metrics.volume_reliable:
            return exact(_EXACT_BASIS["volume"])
        return Confidence(
            tier="estimated",
            error_abs=None,
            basis="divergence-theorem volume on a non-watertight / winding-inconsistent mesh "
            "(unreliable)",
        )
    if name == "self_intersecting_face_count":
        if metrics.integrity.self_intersection_checked:
            return exact("Moller triangle-triangle intersection test")
        return Confidence(
            tier="estimated",
            error_abs=None,
            basis="self-intersection not verified (mesh exceeds the face cap)",
        )
    if name in _EXACT_BASIS:
        return exact(_EXACT_BASIS[name])
    if name in _TOPO_BASIS:
        return topological(_TOPO_BASIS[name])
    if name in _INTEGRITY_COUNT_KEYS:
        return exact("combinatorial integrity count")
    return exact("exact combinatorial/analytic quantity")


class Tolerance(BaseModel):
    """Global tolerance applied to every scalar/vector check unless overridden."""

    model_config = ConfigDict(extra="forbid")

    relative: float | None = Field(default=0.01, ge=0)
    absolute: float | None = Field(default=None, ge=0)


class ScalarCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: float
    rel_tol: float | None = Field(default=None, ge=0)
    abs_tol: float | None = Field(default=None, ge=0)


class VectorCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: list[float] = Field(min_length=3, max_length=3)
    rel_tol: float | None = Field(default=None, ge=0)
    abs_tol: float | None = Field(default=None, ge=0)


class CountCheck(BaseModel):
    """Exact when `expected` is given, otherwise a [min, max] range."""

    model_config = ConfigDict(extra="forbid")

    expected: int | None = None
    min: int | None = None
    max: int | None = None


class Expectations(BaseModel):
    """Expected properties of the mesh. Every field is optional; only the
    checks you specify are evaluated. Scalars/vectors accept either a bare
    value (uses the global tolerance) or an object with per-check tolerances.
    All values are in the file's native units."""

    # extra="allow" so unknown keys reach evaluate(), which rejects them with
    # a helpful message instead of a generic pydantic error.
    model_config = ConfigDict(extra="allow")

    tolerance: Tolerance = Field(default_factory=Tolerance)
    volume: float | ScalarCheck | None = None
    surface_area: float | ScalarCheck | None = None
    bbox_min: list[float] | VectorCheck | None = None
    bbox_max: list[float] | VectorCheck | None = None
    bbox_extents: list[float] | VectorCheck | None = None
    centroid: list[float] | VectorCheck | None = None
    vertex_count: int | CountCheck | None = None
    face_count: int | CountCheck | None = None
    watertight: bool | None = None
    winding_consistent: bool | None = None
    body_count: int | None = None
    euler_number: int | None = None
    # Integrity checks (usually expected 0); see IntegrityMetrics for meaning.
    boundary_edge_count: int | CountCheck | None = None
    non_manifold_edge_count: int | CountCheck | None = None
    degenerate_face_count: int | CountCheck | None = None
    sliver_face_count: int | CountCheck | None = None
    duplicate_face_count: int | CountCheck | None = None
    unmerged_vertex_count: int | CountCheck | None = None
    unreferenced_vertex_count: int | CountCheck | None = None
    flipped_face_count: int | CountCheck | None = None
    self_intersecting_face_count: int | CountCheck | None = None
    min_triangle_quality: float | ScalarCheck | None = None  # lower bound (actual >= expected)


class CheckResult(BaseModel):
    name: str
    passed: bool
    expected: Any
    actual: Any
    deviation: float | None = None
    deviation_pct: float | None = None
    tolerance: dict[str, float | None] | None = None
    caveats: list[str] = Field(default_factory=list)
    # How the `actual` value was derived and how much numeric slack rides on it.
    confidence: Confidence | None = None


class ValidationReport(BaseModel):
    passed: bool
    summary: str
    checks: list[CheckResult]


def _reject_unknown_keys(exp: Expectations) -> None:
    extra = sorted(exp.model_extra or {})
    if not extra:
        return
    suggestions = []
    for key in extra:
        match = get_close_matches(key, SUPPORTED_KEYS, n=1, cutoff=0.5)
        if match:
            suggestions.append(f"did you mean '{match[0]}' instead of '{key}'?")
    hint = f"Supported keys: {', '.join(SUPPORTED_KEYS)}, tolerance."
    if suggestions:
        hint += " " + " ".join(suggestions)
    raise MeshToolError(
        ErrorCode.INVALID_EXPECTATION, f"Unknown expectation key(s): {extra}", hint
    )


def _resolve_tols(
    rel_tol: float | None, abs_tol: float | None, global_tol: Tolerance
) -> tuple[float, float]:
    rel = rel_tol if rel_tol is not None else (global_tol.relative or 0.0)
    abs_ = abs_tol if abs_tol is not None else (global_tol.absolute or 0.0)
    return rel, abs_


def _allowed(expected: float, rel: float, abs_: float) -> float:
    return max(abs_, rel * abs(expected), _TINY)


def _scalar_check(
    name: str, spec: float | ScalarCheck, actual: float | None, global_tol: Tolerance
) -> CheckResult:
    check = spec if isinstance(spec, ScalarCheck) else ScalarCheck(expected=float(spec))
    rel, abs_ = _resolve_tols(check.rel_tol, check.abs_tol, global_tol)
    tolerance = {"relative": rel, "absolute": abs_}
    if actual is None:
        return CheckResult(
            name=name, passed=False, expected=check.expected, actual=None,
            tolerance=tolerance, caveats=["actual value could not be computed"],
        )
    deviation = actual - check.expected
    deviation_pct = (deviation / check.expected * 100.0) if check.expected != 0 else None
    return CheckResult(
        name=name,
        passed=abs(deviation) <= _allowed(check.expected, rel, abs_),
        expected=check.expected,
        actual=actual,
        deviation=deviation,
        deviation_pct=deviation_pct,
        tolerance=tolerance,
    )


def _vector_check(
    name: str, spec: list[float] | VectorCheck, actual: list[float], global_tol: Tolerance
) -> CheckResult:
    if isinstance(spec, VectorCheck):
        check = spec
    else:
        if len(spec) != 3:
            raise MeshToolError(
                ErrorCode.INVALID_EXPECTATION,
                f"'{name}' expects a 3-component vector, got {len(spec)} components",
            )
        check = VectorCheck(expected=[float(v) for v in spec])
    rel, abs_ = _resolve_tols(check.rel_tol, check.abs_tol, global_tol)
    diffs = [a - e for a, e in zip(actual, check.expected)]
    passed = all(
        abs(d) <= _allowed(e, rel, abs_) for d, e in zip(diffs, check.expected)
    )
    worst_i = max(range(3), key=lambda i: abs(diffs[i]))
    worst_expected = check.expected[worst_i]
    return CheckResult(
        name=name,
        passed=passed,
        expected=check.expected,
        actual=actual,
        deviation=diffs[worst_i],
        deviation_pct=(diffs[worst_i] / worst_expected * 100.0) if worst_expected != 0 else None,
        tolerance={"relative": rel, "absolute": abs_},
    )


def _count_check(name: str, spec: int | CountCheck, actual: int) -> CheckResult:
    check = spec if isinstance(spec, CountCheck) else CountCheck(expected=int(spec))
    if check.expected is not None:
        return CheckResult(
            name=name,
            passed=actual == check.expected,
            expected=check.expected,
            actual=actual,
            deviation=float(actual - check.expected),
        )
    if check.min is None and check.max is None:
        raise MeshToolError(
            ErrorCode.INVALID_EXPECTATION,
            f"'{name}' needs either 'expected' or a 'min'/'max' range",
        )
    lo = check.min if check.min is not None else actual
    hi = check.max if check.max is not None else actual
    return CheckResult(
        name=name,
        passed=lo <= actual <= hi,
        expected={"min": check.min, "max": check.max},
        actual=actual,
    )


def _upper_bound_check(
    name: str, spec: float | ScalarCheck, actual: float, global_tol: Tolerance
) -> CheckResult:
    """Pass when actual <= expected (a limit), allowing the usual tolerance slack above."""
    check = spec if isinstance(spec, ScalarCheck) else ScalarCheck(expected=float(spec))
    rel, abs_ = _resolve_tols(check.rel_tol, check.abs_tol, global_tol)
    limit = check.expected + _allowed(check.expected, rel, abs_)
    return CheckResult(
        name=name,
        passed=actual <= limit,
        expected=check.expected,
        actual=actual,
        deviation=actual - check.expected,
        tolerance={"relative": rel, "absolute": abs_},
    )


def _lower_bound_check(
    name: str, spec: float | ScalarCheck, actual: float, global_tol: Tolerance
) -> CheckResult:
    """Pass when actual >= expected (a floor), allowing the usual tolerance slack below."""
    check = spec if isinstance(spec, ScalarCheck) else ScalarCheck(expected=float(spec))
    rel, abs_ = _resolve_tols(check.rel_tol, check.abs_tol, global_tol)
    floor = check.expected - _allowed(check.expected, rel, abs_)
    return CheckResult(
        name=name,
        passed=actual >= floor,
        expected=check.expected,
        actual=actual,
        deviation=actual - check.expected,
        tolerance={"relative": rel, "absolute": abs_},
    )


def _exact_check(name: str, expected: Any, actual: Any) -> CheckResult:
    return CheckResult(name=name, passed=actual == expected, expected=expected, actual=actual)


def _fail_detail(check: CheckResult) -> str:
    if isinstance(check.expected, dict):  # count range
        rng = f"[{check.expected.get('min')}, {check.expected.get('max')}]"
        return f"FAIL {check.name}: expected in {rng}, actual {check.actual}"
    if check.deviation_pct is not None:
        return (
            f"FAIL {check.name}: expected {check.expected}, actual {check.actual} "
            f"({check.deviation_pct:+.2f}%)"
        )
    return f"FAIL {check.name}: expected {check.expected}, actual {check.actual}"


def _volume_unreliable_reason(metrics: MeshMetrics) -> str | None:
    """Why a volume/mass check must fail-closed, or None if the volume is trustworthy.

    A watertight+consistent mesh can still self-intersect (common after boolean edits),
    in which case trimesh's divergence volume is silently wrong — so a positive OR an
    unverified self-intersection count also blocks a PASS. No silent pass on bad input.
    """
    if not metrics.volume_reliable:
        return (
            "volume is unreliable: the mesh is not watertight or its winding is inconsistent "
            "(divergence-theorem volume cannot be trusted) — refusing to PASS the volume check"
        )
    integrity = metrics.integrity
    if integrity.self_intersecting_face_count > 0:
        return (
            f"volume may be wrong: the mesh has {integrity.self_intersecting_face_count} "
            "self-intersecting faces (watertight can read true while the volume is off) — "
            "refusing to PASS the volume check"
        )
    if not integrity.self_intersection_checked:
        return (
            "volume reliability is unverified: the self-intersection test was skipped "
            "(mesh exceeds the face cap) — refusing to PASS the volume check"
        )
    return None


def evaluate(metrics: MeshMetrics, expectations: Expectations) -> ValidationReport:
    _reject_unknown_keys(expectations)
    tol = expectations.tolerance
    volume_caveats = [c for c in metrics.caveats if "volume" in c]

    checks: list[CheckResult] = []
    if expectations.volume is not None:
        check = _scalar_check("volume", expectations.volume, metrics.volume, tol)
        check.caveats.extend(volume_caveats)
        reason = _volume_unreliable_reason(metrics)
        if reason is not None:
            check.passed = False
            check.caveats.append(reason)
        checks.append(check)
    if expectations.surface_area is not None:
        checks.append(
            _scalar_check("surface_area", expectations.surface_area, metrics.surface_area, tol)
        )
    if expectations.bbox_min is not None:
        checks.append(_vector_check("bbox_min", expectations.bbox_min, metrics.bounds.min, tol))
    if expectations.bbox_max is not None:
        checks.append(_vector_check("bbox_max", expectations.bbox_max, metrics.bounds.max, tol))
    if expectations.bbox_extents is not None:
        checks.append(
            _vector_check("bbox_extents", expectations.bbox_extents, metrics.extents, tol)
        )
    if expectations.centroid is not None:
        checks.append(_vector_check("centroid", expectations.centroid, metrics.centroid, tol))
    if expectations.vertex_count is not None:
        checks.append(_count_check("vertex_count", expectations.vertex_count, metrics.vertex_count))
    if expectations.face_count is not None:
        checks.append(_count_check("face_count", expectations.face_count, metrics.face_count))
    if expectations.watertight is not None:
        checks.append(_exact_check("watertight", expectations.watertight, metrics.is_watertight))
    if expectations.winding_consistent is not None:
        checks.append(
            _exact_check(
                "winding_consistent", expectations.winding_consistent,
                metrics.is_winding_consistent,
            )
        )
    if expectations.body_count is not None:
        checks.append(_exact_check("body_count", expectations.body_count, metrics.body_count))
    if expectations.euler_number is not None:
        checks.append(
            _exact_check("euler_number", expectations.euler_number, metrics.euler_number)
        )

    for key in _INTEGRITY_COUNT_KEYS:
        spec = getattr(expectations, key)
        if spec is None:
            continue
        check = _count_check(key, spec, getattr(metrics.integrity, key))
        if key == "self_intersecting_face_count" and not metrics.integrity.self_intersection_checked:
            check.passed = False
            check.caveats.append("self-intersection was not verified: mesh exceeds the face cap")
        checks.append(check)
    if expectations.min_triangle_quality is not None:
        checks.append(
            _lower_bound_check(
                "min_triangle_quality",
                expectations.min_triangle_quality,
                metrics.integrity.min_triangle_quality,
                tol,
            )
        )

    if not checks:
        raise MeshToolError(
            ErrorCode.INVALID_EXPECTATION,
            "expectations contained no checks",
            f"Provide at least one of: {', '.join(SUPPORTED_KEYS)}. "
            "Use render_mesh if you only want images.",
        )

    for check in checks:
        if check.confidence is None:
            check.confidence = _confidence_for(check.name, metrics)

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} checks passed."
    if failed:
        details = [_fail_detail(c) for c in failed[:5]]
        if len(failed) > 5:
            details.append(f"... and {len(failed) - 5} more failures")
        summary += " " + ". ".join(details) + "."

    return ValidationReport(passed=not failed, summary=summary, checks=checks)


class ChangeExpectations(BaseModel):
    """Assertions about a LOCALIZED change between a before- and after-mesh, scoped to a
    region. `region` is required; every assertion field is optional (provide the region
    alone to just report the localized stats). All values are in the file's native units."""

    model_config = ConfigDict(extra="forbid")

    region: Region
    change_threshold: float | None = Field(
        default=None,
        description="Displacement above which a vertex counts as 'changed' (default: "
        "a small fraction of the before-mesh bbox diagonal).",
    )
    max_unchanged_deviation: float | ScalarCheck | None = Field(
        default=None,
        description="Upper bound on how far any vertex OUTSIDE the region may move "
        "(assert the rest of the mesh is untouched).",
    )
    emboss_height: float | ScalarCheck | None = Field(
        default=None, description="Expected outward height of the change (material added)."
    )
    pocket_depth: float | ScalarCheck | None = Field(
        default=None, description="Expected inward depth of the change (material removed)."
    )
    tolerance: Tolerance = Field(default_factory=Tolerance)


_SIDEWALL_CAVEAT = (
    "signed height/depth is the 95th-percentile-magnitude of the changed vertices' motion "
    "along the before-surface normal; tessellated side walls can bias it"
)


def evaluate_change(change: LocalizedChange, exp: ChangeExpectations) -> ValidationReport:
    tol = exp.tolerance
    approx = [] if change.same_topology else ["measured on differing topology; approximate"]
    # On matching topology the displacement is an exact per-vertex difference; otherwise
    # it is a closest-point distance to A's surface, resolved to the change threshold.
    conf = (
        exact("per-vertex displacement (matching topology)")
        if change.same_topology
        else sampled(
            "closest-point displacement to A's surface (differing topology)",
            change.change_threshold,
        )
    )
    checks: list[CheckResult] = []

    if exp.max_unchanged_deviation is not None:
        check = _upper_bound_check(
            "untouched_region_max_deviation",
            exp.max_unchanged_deviation,
            change.outside_max_displacement,
            tol,
        )
        check.caveats.extend(approx)
        checks.append(check)
    peak = change.signed_displacement.peak
    if exp.emboss_height is not None:
        check = _scalar_check("emboss_height", exp.emboss_height, peak, tol)
        check.caveats.append(_SIDEWALL_CAVEAT)
        check.caveats.extend(approx)
        checks.append(check)
    if exp.pocket_depth is not None:
        check = _scalar_check(
            "pocket_depth", exp.pocket_depth, (-peak if peak is not None else None), tol
        )
        check.caveats.append(_SIDEWALL_CAVEAT)
        check.caveats.extend(approx)
        checks.append(check)

    for check in checks:
        check.confidence = conf.model_copy()

    failed = [c for c in checks if not c.passed]
    if not checks:
        summary = "no localized-change assertions; reporting stats only."
    else:
        summary = f"{len(checks) - len(failed)}/{len(checks)} localized checks passed."
        if failed:
            summary += " " + ". ".join(_fail_detail(c) for c in failed) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
