"""Artifact and numerical validation for MLIP evaluation."""

from __future__ import annotations

import math
from pathlib import Path

from mlip_research_agent.artifacts.registry import Artifact, ArtifactRegistry
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError


def validation_error(message: str) -> SkillError:
    return SkillError(
        message,
        failure_class=FailureClass.VALIDATION_ERROR,
        severity=Severity.CRITICAL,
        retryable=False,
    )


def registered_artifact(
    registry: ArtifactRegistry,
    artifact_id: str,
    *,
    expected_kind: str,
) -> tuple[Artifact, Path]:
    artifact = registry.get(artifact_id)
    if artifact is None:
        raise validation_error(f"required artifact is not registered: {artifact_id}")
    if artifact.kind != expected_kind:
        raise validation_error(
            f"artifact {artifact_id} has kind {artifact.kind!r}; expected {expected_kind!r}"
        )
    if not registry.verify(artifact_id):
        raise validation_error(f"artifact integrity verification failed: {artifact_id}")
    path = (registry.run_dir / artifact.relative_path).resolve()
    try:
        path.relative_to(registry.run_dir.resolve())
    except ValueError as exc:
        raise validation_error(f"artifact escapes the run directory: {artifact_id}") from exc
    return artifact, path


def require_finite(values: list[float], description: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise validation_error(f"non-finite values in {description}")
