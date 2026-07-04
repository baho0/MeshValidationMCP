"""Design-for-manufacturing oracle — a pure-geometry composite (optional).

The user scoped this project to geometry, so this is deliberately NOT a materials/process
model: it just composes the geometric checks that already exist into one manufacturability
verdict — minimum wall thickness, undercut area for a pull direction, and (when watertight)
a trapped-void check from the internal-cavity count. Each sub-check keeps its own confidence.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .confidence import exact, topological
from .features import draft_analysis, wall_thickness
from .loading import LoadedMesh
from .validation import CheckResult, ValidationReport, _fail_detail


class DfmExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_wall_thickness: float | None = Field(default=None, gt=0)
    pull_direction: list[float] | None = Field(default=None, min_length=3, max_length=3)
    min_draft_deg: float = 0.0
    max_undercut_area: float = 0.0
    check_trapped_voids: bool = True
    sample_count: int = 2000


def _internal_void_count(loaded: LoadedMesh) -> int:
    """Number of enclosed internal cavities: a watertight body that contains another shell
    shows up as extra connected components beyond the outer surface."""
    bodies = loaded.bodies
    if len(bodies) <= 1:
        return 0
    # A void is a body fully inside another (its centroid lies inside a larger body).
    voids = 0
    for i, inner in enumerate(bodies):
        ci = np.asarray(inner.centroid)
        for j, outer in enumerate(bodies):
            if i == j or not outer.is_watertight:
                continue
            if float(outer.volume) > float(inner.volume) and bool(outer.contains([ci])[0]):
                voids += 1
                break
    return voids


def dfm_report(loaded: LoadedMesh, exp: DfmExpectations) -> ValidationReport:
    checks: list[CheckResult] = []

    if exp.min_wall_thickness is not None:
        info = wall_thickness(loaded, exp.sample_count)
        checks.append(
            CheckResult(
                name="min_wall_thickness",
                passed=info.p5_thickness >= exp.min_wall_thickness,
                expected=exp.min_wall_thickness,
                actual=info.p5_thickness,
                deviation=info.p5_thickness - exp.min_wall_thickness,
                caveats=["uses p5 thickness (robust to edge artefacts)"],
                confidence=info.confidence,
            )
        )

    if exp.pull_direction is not None:
        draft = draft_analysis(loaded, exp.pull_direction, exp.min_draft_deg)
        checks.append(
            CheckResult(
                name="undercut_area",
                passed=draft.undercut_area <= exp.max_undercut_area,
                expected=exp.max_undercut_area,
                actual=draft.undercut_area,
                confidence=draft.confidence,
            )
        )
        if exp.min_draft_deg > 0:
            checks.append(
                CheckResult(
                    name="min_draft_angle",
                    passed=draft.min_draft_angle_deg >= exp.min_draft_deg,
                    expected=exp.min_draft_deg,
                    actual=draft.min_draft_angle_deg,
                    confidence=draft.confidence,
                )
            )

    if exp.check_trapped_voids:
        voids = _internal_void_count(loaded)
        checks.append(
            CheckResult(
                name="no_trapped_voids",
                passed=voids == 0,
                expected=0,
                actual=voids,
                confidence=topological("enclosed-cavity count"),
            )
        )

    if not checks:
        checks.append(
            CheckResult(
                name="dfm", passed=True, expected="at least one DfM criterion", actual="none given",
                caveats=["no DfM criteria were specified"], confidence=exact("no-op"),
            )
        )

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} DfM checks passed."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
