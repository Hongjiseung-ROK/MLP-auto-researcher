"""Boundary validation for oracle input artifacts."""

from __future__ import annotations

from pathlib import Path

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError


def run_relative_file(run_dir: Path, relative_path: str, description: str) -> Path:
    candidate = (run_dir / relative_path).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise SkillError(
            f"{description} escapes the run directory: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        ) from exc
    if not candidate.is_file():
        raise SkillError(
            f"{description} not found: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return candidate
