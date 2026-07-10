"""Validation for the mock literature skill."""

from __future__ import annotations

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError
from mlip_research_agent.skills.literature.schema import Reference


def validate_references(references: list[Reference]) -> None:
    keys = [r.key for r in references]
    if len(keys) != len(set(keys)):
        raise SkillError(
            "duplicate reference keys in retrieval result",
            failure_class=FailureClass.TOOL_ERROR,
            severity=Severity.MEDIUM,
            retryable=True,
        )
