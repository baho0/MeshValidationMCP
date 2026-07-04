"""Generative-operation validation: extrude, revolve (and their invariants).

Generative features have exact volumetric signatures we can assert:
- an **extrude** of a profile of area A through height h has volume A*h and a constant cross
  section along the extrusion axis;
- a **revolve** of a profile (area A, centroid at radius R from the axis) sweeps a volume of
  2*pi*R*A (Pappus's theorem).

We check the volume relation and, for an extrude, that the section area really is constant.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .confidence import exact
from .loading import LoadedMesh
from .metrics import compute_metrics
from .section import section_area_profile
from .validation import CheckResult, ValidationReport, _fail_detail


class ExtrudeExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_area: float = Field(gt=0)
    height: float = Field(gt=0)
    axis: list[float] = Field(default=[0.0, 0.0, 1.0], min_length=3, max_length=3)
    tolerance: float = 0.02


class RevolveExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_area: float = Field(gt=0)
    profile_centroid_radius: float = Field(gt=0)  # distance of the profile centroid from the axis
    tolerance: float = 0.05


def _volume_check(name: str, expected: float, actual: float | None, rel: float) -> CheckResult:
    if actual is None:
        return CheckResult(
            name=name, passed=False, expected=expected, actual=None,
            caveats=["volume is unavailable/unreliable"],
            confidence=exact("analytic generative volume"),
        )
    passed = abs(actual - expected) <= rel * abs(expected)
    return CheckResult(
        name=name,
        passed=passed,
        expected=expected,
        actual=actual,
        deviation=actual - expected,
        deviation_pct=(actual - expected) / expected * 100.0 if expected else None,
        tolerance={"relative": rel, "absolute": None},
        confidence=exact("analytic generative volume"),
    )


def validate_extrude(loaded: LoadedMesh, exp: ExtrudeExpectations) -> ValidationReport:
    metrics = compute_metrics(loaded)
    checks: list[CheckResult] = []
    if not metrics.volume_reliable:
        checks.append(
            CheckResult(
                name="extrude_volume", passed=False, expected=exp.profile_area * exp.height,
                actual=metrics.volume, caveats=["volume unreliable (not watertight); refusing to PASS"],
                confidence=exact("analytic generative volume"),
            )
        )
    else:
        checks.append(
            _volume_check(
                "extrude_volume", exp.profile_area * exp.height, metrics.volume, exp.tolerance
            )
        )

    profile = section_area_profile(loaded, exp.axis, stations=9)
    checks.append(
        CheckResult(
            name="constant_cross_section",
            passed=profile.area_constant,
            expected="constant section area along the axis",
            actual={"areas": [round(a, 4) for a in profile.areas]},
            confidence=exact("section area at stations along the axis"),
        )
    )
    mean_area = float(np.mean(profile.areas)) if profile.areas else 0.0
    checks.append(_volume_check("section_area", exp.profile_area, mean_area, exp.tolerance))

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} extrude checks passed."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)


def validate_revolve(loaded: LoadedMesh, exp: RevolveExpectations) -> ValidationReport:
    metrics = compute_metrics(loaded)
    pappus_volume = 2.0 * math.pi * exp.profile_centroid_radius * exp.profile_area
    checks: list[CheckResult] = []
    if not metrics.volume_reliable:
        checks.append(
            CheckResult(
                name="pappus_volume", passed=False, expected=pappus_volume, actual=metrics.volume,
                caveats=["volume unreliable (not watertight); refusing to PASS"],
                confidence=exact("Pappus volume 2*pi*R*A"),
            )
        )
    else:
        check = _volume_check("pappus_volume", pappus_volume, metrics.volume, exp.tolerance)
        check.confidence = exact("Pappus volume 2*pi*R*A")
        checks.append(check)

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} revolve checks passed."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
