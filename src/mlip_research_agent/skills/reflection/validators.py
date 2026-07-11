"""Validation for the tea-time skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError
from mlip_research_agent.skills.reflection.poems import POEM_CORPUS


def validate_poem_key(poem_key: str | None) -> None:
    if poem_key is not None and poem_key not in POEM_CORPUS:
        raise SkillError(
            f"unknown poem {poem_key!r}; corpus: {sorted(POEM_CORPUS)}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.MEDIUM,
            retryable=False,
        )


def load_data(run_dir: Path, data_path: str | None) -> dict[str, Any] | None:
    if data_path is None:
        return None
    path = run_dir / data_path
    if not path.is_file():
        raise SkillError(
            f"data artifact not found: {data_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.MEDIUM,
            retryable=False,
        )
    try:
        loaded: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SkillError(
            f"data artifact is not valid JSON: {data_path} ({exc})",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.MEDIUM,
            retryable=False,
        ) from exc
    if not isinstance(loaded, dict):
        raise SkillError(
            f"data artifact must be a JSON object: {data_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.MEDIUM,
            retryable=False,
        )
    return loaded


def assert_no_claims_registered(n_claims: int) -> None:
    """Tea-time output is agent-text class: it must never register claims."""
    if n_claims != 0:
        raise SkillError(
            f"tea_time_with_reading_poem registered {n_claims} claim(s); its output "
            "is reflection, never evidence",
            failure_class=FailureClass.UNSUPPORTED_CLAIM,
            severity=Severity.HIGH,
            retryable=False,
        )
