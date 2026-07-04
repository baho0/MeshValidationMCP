"""Remesh / simplify / subdivide validation: did retessellation keep the shape and topology?

A remesh must change the tessellation without changing the *object*: the surface should stay
within a small deviation of the original (bounded vertex-to-surface Hausdorff), the topology
(genus, watertightness) should be unchanged, and triangle quality should not collapse. This
rolls those into one verdict so an agent can assert "my decimation/subdivision was safe".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .confidence import exact, sampled, topological
from .golden import _vertex_to_surface_hausdorff
from .loading import LoadedMesh
from .metrics import compute_metrics
from .validation import CheckResult, ValidationReport, _fail_detail


class RemeshExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_deviation: float = Field(gt=0, description="max allowed surface distance to the original")
    preserve_topology: bool = True
    min_triangle_quality: float | None = None


def validate_remesh(
    loaded_before: LoadedMesh, loaded_after: LoadedMesh, exp: RemeshExpectations
) -> ValidationReport:
    before = compute_metrics(loaded_before)
    after = compute_metrics(loaded_after)
    checks: list[CheckResult] = []

    deviation = _vertex_to_surface_hausdorff(loaded_before.combined, loaded_after.combined)
    checks.append(
        CheckResult(
            name="shape_preserved",
            passed=deviation <= exp.max_deviation,
            expected=exp.max_deviation,
            actual=deviation,
            deviation=deviation - exp.max_deviation,
            confidence=sampled("symmetric max vertex-to-surface distance", deviation),
        )
    )

    if exp.preserve_topology:
        checks.append(
            CheckResult(
                name="watertight_preserved",
                passed=after.is_watertight == before.is_watertight,
                expected=before.is_watertight,
                actual=after.is_watertight,
                confidence=exact("edge-manifold watertight test"),
            )
        )
        gb, ga = before.topology.genus_total, after.topology.genus_total
        checks.append(
            CheckResult(
                name="genus_preserved",
                passed=(gb == ga),
                expected=gb,
                actual=ga,
                caveats=[] if gb is not None else ["genus undefined (open mesh)"],
                confidence=topological("genus from Euler characteristic"),
            )
        )

    if exp.min_triangle_quality is not None:
        q = after.integrity.min_triangle_quality
        checks.append(
            CheckResult(
                name="min_triangle_quality",
                passed=q >= exp.min_triangle_quality,
                expected=exp.min_triangle_quality,
                actual=q,
                deviation=q - exp.min_triangle_quality,
                confidence=exact("analytic triangle quality"),
            )
        )

    failed = [c for c in checks if not c.passed]
    summary = f"{len(checks) - len(failed)}/{len(checks)} remesh checks passed."
    if failed:
        summary += " " + ". ".join(_fail_detail(c) for c in failed) + "."
    return ValidationReport(passed=not failed, summary=summary, checks=checks)
