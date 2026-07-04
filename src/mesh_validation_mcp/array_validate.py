"""Array / pattern validation: is the result N congruent copies of a base at the right places?

A linear or polar array must satisfy two independent facts: every instance is *congruent* to
the base (same shape, allowing a rigid motion), and the instances sit at the *expected*
positions (a grid for a linear array, an angular fan for a polar one). Congruence is tested
with rigid-motion invariants — volume and the principal inertia moments — so it needs no
registration; positions are checked by matching instance centroids to the predicted grid.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from .confidence import exact, topological
from .loading import LoadedMesh
from .validation import CheckResult, ValidationReport, _fail_detail


class ArrayPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["linear", "polar"]
    count: int = Field(ge=1)
    # linear
    step: list[float] | None = Field(default=None, min_length=3, max_length=3)
    # polar
    axis: list[float] | None = Field(default=None, min_length=3, max_length=3)
    center: list[float] | None = Field(default=None, min_length=3, max_length=3)
    angle_deg: float | None = None
    tolerance: float = 0.02  # relative slack on congruence / position


def _inertia_signature(mesh: trimesh.Trimesh) -> np.ndarray | None:
    try:
        return np.sort(np.asarray(mesh.principal_inertia_components, dtype=float))
    except Exception:
        return None


def _expected_centroids(base_centroid: np.ndarray, pattern: ArrayPattern) -> np.ndarray:
    if pattern.kind == "linear":
        step = np.asarray(pattern.step, dtype=float)
        return np.array([base_centroid + k * step for k in range(pattern.count)])
    axis = np.asarray(pattern.axis, dtype=float)
    center = np.asarray(pattern.center, dtype=float)
    angle = np.radians(pattern.angle_deg or 0.0)
    out = []
    for k in range(pattern.count):
        rot = trimesh.transformations.rotation_matrix(k * angle, axis, center)
        out.append((rot @ np.append(base_centroid, 1.0))[:3])
    return np.array(out)


def validate_array(
    loaded_result: LoadedMesh, loaded_base: LoadedMesh, pattern: ArrayPattern
) -> ValidationReport:
    base = loaded_base.combined
    bodies = loaded_result.bodies
    diagonal = float(np.linalg.norm(loaded_result.combined.extents)) or 1.0
    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            name="instance_count",
            passed=len(bodies) == pattern.count,
            expected=pattern.count,
            actual=len(bodies),
            confidence=topological("connected-component count"),
        )
    )

    # Congruence: each instance shares the base's volume and principal inertia moments
    # (both invariant under any rigid motion), so no registration is needed.
    base_vol = float(base.volume)
    base_sig = _inertia_signature(base)
    vol_tol = pattern.tolerance * max(abs(base_vol), 1.0)
    worst_vol = 0.0
    worst_sig = 0.0
    for body in bodies:
        worst_vol = max(worst_vol, abs(float(body.volume) - base_vol))
        sig = _inertia_signature(body)
        if sig is not None and base_sig is not None:
            denom = np.maximum(np.abs(base_sig), 1.0)
            worst_sig = max(worst_sig, float(np.max(np.abs(sig - base_sig) / denom)))
    checks.append(
        CheckResult(
            name="instances_congruent",
            passed=worst_vol <= vol_tol and worst_sig <= pattern.tolerance,
            expected="every instance congruent to the base",
            actual={"max_volume_delta": worst_vol, "max_inertia_rel_delta": worst_sig},
            confidence=exact("volume + principal inertia (rigid invariants)"),
        )
    )

    # Positions: match instance centroids to the predicted grid (greedy nearest assignment).
    expected = _expected_centroids(np.asarray(base.centroid, dtype=float), pattern)
    actual = np.array([np.asarray(b.centroid, dtype=float) for b in bodies])
    pos_tol = pattern.tolerance * diagonal
    max_pos_err = 0.0
    if len(actual) == len(expected):
        remaining = list(range(len(actual)))
        for exp_c in expected:
            j = min(remaining, key=lambda i: np.linalg.norm(actual[i] - exp_c))
            max_pos_err = max(max_pos_err, float(np.linalg.norm(actual[j] - exp_c)))
            remaining.remove(j)
        pos_passed = max_pos_err <= pos_tol
    else:
        pos_passed = False
    checks.append(
        CheckResult(
            name="instance_positions",
            passed=pos_passed,
            expected="instances at the predicted grid",
            actual={"max_position_error": max_pos_err},
            confidence=exact("centroid grid vs predicted pattern"),
        )
    )

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} array checks passed ({pattern.kind})."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
