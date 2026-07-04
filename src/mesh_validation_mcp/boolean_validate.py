"""Boolean / CSG validation: check that A op B actually produced the claimed result.

Boolean edits are the most common way to break a mesh while still reporting watertight=True
(the seam self-intersects or leaves slivers), which then makes the volume silently wrong. We
check three things: (1) the result's integrity (watertight, non-self-intersecting, no new
defects); (2) volume bounds the operation must satisfy (union: max(Va,Vb) <= Vr <= Va+Vb;
difference: Va-Vb <= Vr <= Va; intersection: Vr <= min(Va,Vb)); (3) containment via signed
distance (union contains both operands; a difference stays inside A and clear of B's interior).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict

from .confidence import exact, sampled
from .config import SEED
from .loading import LoadedMesh
from .metrics import compute_metrics
from .validation import CheckResult, ValidationReport, _fail_detail

Operation = Literal["union", "difference", "intersection"]


class BooleanExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Operation
    tolerance: float = 0.02  # relative slack on the volume bounds


def _bound_check(name: str, lo: float | None, hi: float | None, actual: float, slack: float) -> CheckResult:
    lo_ok = lo is None or actual >= lo - slack
    hi_ok = hi is None or actual <= hi + slack
    return CheckResult(
        name=name,
        passed=bool(lo_ok and hi_ok),
        expected={"min": lo, "max": hi},
        actual=actual,
        confidence=exact("volume bound from set algebra"),
    )


def _contains(container: trimesh.Trimesh, points: np.ndarray, tol: float) -> float:
    """Fraction of points that lie inside-or-on the container (signed_distance >= -tol)."""
    if not container.is_watertight:
        return float("nan")
    sd = np.asarray(trimesh.proximity.signed_distance(container, points))
    return float((sd >= -tol).mean())


def validate_boolean(
    loaded_a: LoadedMesh,
    loaded_b: LoadedMesh,
    loaded_result: LoadedMesh,
    exp: BooleanExpectations,
) -> ValidationReport:
    a, b, r = loaded_a.combined, loaded_b.combined, loaded_result.combined
    ma = compute_metrics(loaded_a)
    mb = compute_metrics(loaded_b)
    mr = compute_metrics(loaded_result)
    diagonal = float(np.linalg.norm(r.extents)) or 1.0
    tol = 1e-3 * diagonal
    checks: list[CheckResult] = []

    # 1) result integrity
    checks.append(
        CheckResult(
            name="result_watertight",
            passed=mr.is_watertight,
            expected=True,
            actual=mr.is_watertight,
            confidence=exact("edge-manifold watertight test"),
        )
    )
    si = mr.integrity
    checks.append(
        CheckResult(
            name="result_non_self_intersecting",
            passed=si.self_intersection_checked and si.self_intersecting_face_count == 0,
            expected=0,
            actual=si.self_intersecting_face_count if si.self_intersection_checked else None,
            caveats=[] if si.self_intersection_checked else ["self-intersection not verified"],
            confidence=(
                exact("Moller triangle-triangle test")
                if si.self_intersection_checked
                else sampled("self-intersection unverified", float("nan"))
            ),
        )
    )

    # 2) volume bounds — only trustworthy when operands and result are watertight
    reliable = ma.volume_reliable and mb.volume_reliable and mr.volume_reliable
    va, vb, vr = ma.volume, mb.volume, mr.volume
    if reliable and None not in (va, vb, vr):
        slack = exp.tolerance * max(va, vb, vr)
        if exp.operation == "union":
            checks.append(_bound_check("union_volume_bounds", max(va, vb), va + vb, vr, slack))
        elif exp.operation == "difference":
            checks.append(_bound_check("difference_volume_bounds", va - vb, va, vr, slack))
        else:  # intersection
            checks.append(_bound_check("intersection_volume_bounds", 0.0, min(va, vb), vr, slack))
    else:
        checks.append(
            CheckResult(
                name="volume_bounds",
                passed=False,
                expected="watertight operands and result",
                actual=None,
                caveats=["a volume is unreliable (not watertight); refusing to check volume bounds"],
                confidence=sampled("volume bounds unavailable", float("nan")),
            )
        )

    # 3) containment via signed distance (sampled)
    pa, _ = trimesh.sample.sample_surface(a, 1000, seed=SEED)
    pb, _ = trimesh.sample.sample_surface(b, 1000, seed=SEED)
    pr, _ = trimesh.sample.sample_surface(r, 1000, seed=SEED)
    spacing = diagonal / np.sqrt(1000)
    if exp.operation == "union":
        frac = min(_contains(r, pa, tol), _contains(r, pb, tol))
        checks.append(
            CheckResult(
                name="union_contains_operands",
                passed=frac >= 0.99,
                expected=">=0.99 of A and B inside result",
                actual=frac,
                confidence=sampled("signed-distance containment (1000 samples)", spacing),
            )
        )
    elif exp.operation == "difference":
        inside_a = _contains(a, pr, tol)
        sd_b = np.asarray(trimesh.proximity.signed_distance(b, pr)) if b.is_watertight else None
        clear_of_b = float((sd_b <= tol).mean()) if sd_b is not None else float("nan")
        checks.append(
            CheckResult(
                name="difference_inside_a",
                passed=inside_a >= 0.99,
                expected=">=0.99 of result inside A",
                actual=inside_a,
                confidence=sampled("signed-distance containment (1000 samples)", spacing),
            )
        )
        checks.append(
            CheckResult(
                name="difference_clear_of_b",
                passed=(not np.isnan(clear_of_b)) and clear_of_b >= 0.99,
                expected=">=0.99 of result outside B's interior",
                actual=clear_of_b,
                confidence=sampled("signed-distance containment (1000 samples)", spacing),
            )
        )
    else:  # intersection
        frac = min(_contains(a, pr, tol), _contains(b, pr, tol))
        checks.append(
            CheckResult(
                name="intersection_inside_both",
                passed=frac >= 0.99,
                expected=">=0.99 of result inside both A and B",
                actual=frac,
                confidence=sampled("signed-distance containment (1000 samples)", spacing),
            )
        )

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} boolean checks passed ({exp.operation})."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed[:5]) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
