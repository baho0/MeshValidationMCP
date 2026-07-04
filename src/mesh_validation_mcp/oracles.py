"""Universal property oracles: named, reusable invariants an agent can assert about a
manipulation regardless of what the manipulation was.

Where ``validation.evaluate`` asserts *absolute* properties of one mesh (volume==500),
oracles assert *relational* invariants — usually before-vs-after — that recur across every
operation family: "this edit conserved volume", "it introduced no new defects", "it kept
the part watertight", "the result stayed within a Hausdorff bound of a reference". Each
oracle returns a :class:`CheckResult` carrying a :class:`Confidence`, so oracle output drops
straight into the same report machinery as everything else.

Oracles are fail-closed: if the input needed to judge an invariant is unreliable (an
unreliable volume, an unverified self-intersection check), the oracle refuses to PASS.
"""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .confidence import Confidence, estimated, exact, sampled, topological
from .errors import ErrorCode, MeshToolError
from .metrics import MeshMetrics
from .validation import (
    _INTEGRITY_COUNT_KEYS,
    CheckResult,
    Tolerance,
    ValidationReport,
    _allowed,
    _fail_detail,
    _volume_unreliable_reason,
)

OracleName = Literal[
    "conserves_volume",
    "preserves_watertight",
    "preserves_genus",
    "preserves_euler",
    "no_new_defects",
    "non_self_intersecting",
    "centroid_fixed",
    "monotonic_offset",
    "bounded_hausdorff",
]

# Oracles that compare a before- and after-mesh; the rest judge the after-mesh alone.
_BINARY_ORACLES = frozenset(
    {
        "conserves_volume",
        "preserves_genus",
        "preserves_euler",
        "no_new_defects",
        "centroid_fixed",
        "monotonic_offset",
        "bounded_hausdorff",
    }
)


class PropertySpec(BaseModel):
    """One invariant to assert. Only the fields an oracle needs are read."""

    model_config = ConfigDict(extra="forbid")

    name: OracleName
    tolerance: Tolerance = Field(default_factory=Tolerance)
    # bounded_hausdorff: the maximum allowed surface distance to the reference.
    max_distance: float | None = Field(default=None, ge=0)
    # monotonic_offset: which way the surface was pushed (volume must move that way).
    direction: Literal["outward", "inward"] | None = None


def _genus_total(m: MeshMetrics) -> float | None:
    """Total genus of a closed mesh: for B orientable closed bodies, chi = 2B - 2g."""
    if not m.is_watertight:
        return None
    return m.body_count - m.euler_number / 2.0


def _reference_required(name: str) -> CheckResult:
    return CheckResult(
        name=name,
        passed=False,
        expected="a reference (before) mesh",
        actual=None,
        caveats=[f"the '{name}' oracle needs a reference mesh but none was provided"],
        confidence=Confidence(tier="exact", error_abs=None, basis="missing reference"),
    )


def _conserves_volume(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    reason = _volume_unreliable_reason(before) or _volume_unreliable_reason(after)
    rel = spec.tolerance.relative or 0.0
    abs_ = spec.tolerance.absolute or 0.0
    expected, actual = before.volume, after.volume
    caveats: list[str] = []
    if reason is not None:
        passed = False
        caveats.append(reason)
        conf = estimated("volume delta on an unreliable mesh", float("nan"))
    else:
        deviation = abs(actual - expected)
        passed = deviation <= _allowed(expected, rel, abs_)
        conf = exact("divergence-theorem volume delta")
    return CheckResult(
        name="conserves_volume",
        passed=passed,
        expected=expected,
        actual=actual,
        deviation=(actual - expected) if (actual is not None and expected is not None) else None,
        tolerance={"relative": rel, "absolute": abs_},
        caveats=caveats,
        confidence=conf,
    )


def _preserves_watertight(
    before: MeshMetrics | None, after: MeshMetrics, spec: PropertySpec
) -> CheckResult:
    caveats: list[str] = []
    if before is not None and not before.is_watertight:
        caveats.append("the reference mesh was not watertight either")
    return CheckResult(
        name="preserves_watertight",
        passed=bool(after.is_watertight),
        expected=True,
        actual=after.is_watertight,
        caveats=caveats,
        confidence=exact("edge-manifold watertight test"),
    )


def _preserves_genus(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    gb, ga = _genus_total(before), _genus_total(after)
    if gb is None or ga is None:
        return CheckResult(
            name="preserves_genus",
            passed=False,
            expected=gb,
            actual=ga,
            caveats=["genus is defined only for watertight meshes; refusing to PASS"],
            confidence=topological("genus from Euler characteristic"),
        )
    return CheckResult(
        name="preserves_genus",
        passed=gb == ga,
        expected=gb,
        actual=ga,
        deviation=ga - gb,
        confidence=topological("genus = body_count - euler/2"),
    )


def _preserves_euler(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    return CheckResult(
        name="preserves_euler",
        passed=before.euler_number == after.euler_number,
        expected=before.euler_number,
        actual=after.euler_number,
        deviation=float(after.euler_number - before.euler_number),
        confidence=exact("Euler characteristic V-E+F"),
    )


def _no_new_defects(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    worsened: list[str] = []
    for key in _INTEGRITY_COUNT_KEYS:
        b = getattr(before.integrity, key)
        a = getattr(after.integrity, key)
        if a > b:
            worsened.append(f"{key}: {b} -> {a}")
    checked = after.integrity.self_intersection_checked and before.integrity.self_intersection_checked
    conf = (
        exact("per-defect integrity diff")
        if checked
        else estimated("integrity diff with unverified self-intersection", float("nan"))
    )
    caveats = [] if checked else ["self-intersection was not verified on one input"]
    return CheckResult(
        name="no_new_defects",
        passed=(not worsened) and checked,
        expected="no increase in any defect count",
        actual=("; ".join(worsened) if worsened else "no new defects"),
        caveats=caveats,
        confidence=conf,
    )


def _non_self_intersecting(
    before: MeshMetrics | None, after: MeshMetrics, spec: PropertySpec
) -> CheckResult:
    integrity = after.integrity
    if not integrity.self_intersection_checked:
        return CheckResult(
            name="non_self_intersecting",
            passed=False,
            expected=0,
            actual=None,
            caveats=["self-intersection was not verified (face cap exceeded); refusing to PASS"],
            confidence=estimated("self-intersection unverified", float("nan")),
        )
    return CheckResult(
        name="non_self_intersecting",
        passed=integrity.self_intersecting_face_count == 0,
        expected=0,
        actual=integrity.self_intersecting_face_count,
        confidence=exact("Moller triangle-triangle intersection test"),
    )


def _centroid_fixed(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    rel = spec.tolerance.relative or 0.0
    abs_ = spec.tolerance.absolute or 0.0
    diffs = [a - b for a, b in zip(after.centroid, before.centroid)]
    worst = max(range(3), key=lambda i: abs(diffs[i]))
    # Tolerance is scaled by the bbox diagonal so it is meaningful for a centroid near origin.
    limit = max(abs_, rel * before.bbox_diagonal, 1e-9)
    return CheckResult(
        name="centroid_fixed",
        passed=all(abs(d) <= limit for d in diffs),
        expected=before.centroid,
        actual=after.centroid,
        deviation=diffs[worst],
        tolerance={"relative": rel, "absolute": abs_},
        confidence=exact("area-weighted centroid delta"),
    )


def _monotonic_offset(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    reason = _volume_unreliable_reason(before) or _volume_unreliable_reason(after)
    direction = spec.direction or "outward"
    delta = (after.volume - before.volume) if reason is None else None
    if reason is not None:
        passed = False
        caveats = [reason]
    else:
        passed = (delta > 0) if direction == "outward" else (delta < 0)
        caveats = []
    return CheckResult(
        name="monotonic_offset",
        passed=passed,
        expected=f"volume {'increases' if direction == 'outward' else 'decreases'}",
        actual=delta,
        deviation=delta,
        caveats=caveats,
        confidence=exact("signed volume delta") if reason is None else estimated(
            "signed volume delta on an unreliable mesh", float("nan")
        ),
    )


def _bounded_hausdorff(before: MeshMetrics, after: MeshMetrics, spec: PropertySpec) -> CheckResult:
    # The distance itself is supplied by the caller (compare's bounded upper bound) via
    # run_oracles(distances=...); without it the oracle cannot judge and fails closed.
    raise NotImplementedError  # handled specially in run_oracles


_REGISTRY: dict[str, Callable[..., CheckResult]] = {
    "conserves_volume": _conserves_volume,
    "preserves_watertight": _preserves_watertight,
    "preserves_genus": _preserves_genus,
    "preserves_euler": _preserves_euler,
    "no_new_defects": _no_new_defects,
    "non_self_intersecting": _non_self_intersecting,
    "centroid_fixed": _centroid_fixed,
    "monotonic_offset": _monotonic_offset,
}


def _hausdorff_check(spec: PropertySpec, hausdorff_upper: float | None) -> CheckResult:
    if spec.max_distance is None:
        raise MeshToolError(
            ErrorCode.INVALID_EXPECTATION,
            "bounded_hausdorff requires 'max_distance'",
            "Set the maximum allowed surface distance to the reference, e.g. max_distance: 0.1.",
        )
    if hausdorff_upper is None:
        return CheckResult(
            name="bounded_hausdorff",
            passed=False,
            expected=spec.max_distance,
            actual=None,
            caveats=["no reference distance available; provide a reference mesh"],
            confidence=sampled("Hausdorff upper bound unavailable", float("nan")),
        )
    return CheckResult(
        name="bounded_hausdorff",
        passed=hausdorff_upper <= spec.max_distance,
        expected=spec.max_distance,
        actual=hausdorff_upper,
        deviation=hausdorff_upper - spec.max_distance,
        confidence=sampled("Hausdorff upper bound (sampled max + spacing)", hausdorff_upper),
    )


def run_oracles(
    after: MeshMetrics,
    specs: list[PropertySpec],
    before: MeshMetrics | None = None,
    hausdorff_upper: float | None = None,
) -> ValidationReport:
    """Evaluate a list of property oracles into one pass/fail report."""
    if not specs:
        raise MeshToolError(
            ErrorCode.INVALID_EXPECTATION,
            "no properties to assert",
            f"Provide at least one of: {', '.join(sorted(set(_REGISTRY) | {'bounded_hausdorff'}))}.",
        )
    checks: list[CheckResult] = []
    for spec in specs:
        if spec.name == "bounded_hausdorff":
            checks.append(_hausdorff_check(spec, hausdorff_upper))
            continue
        if spec.name in _BINARY_ORACLES and before is None:
            checks.append(_reference_required(spec.name))
            continue
        checks.append(_REGISTRY[spec.name](before, after, spec))

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} properties held."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed[:5]) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
