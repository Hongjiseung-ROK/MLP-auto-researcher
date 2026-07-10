"""Bounded recovery policy: RETRY / REFINE / PIVOT / ESCALATE / ABORT.

No skill may retry indefinitely, silently change method, or mark a run
successful because a command merely exited. All decisions are recorded.
"""

from __future__ import annotations

import uuid

from mlip_research_agent.schemas.failure import (
    FailureClass,
    FailureRecord,
    RecoveryDecision,
    Severity,
)
from mlip_research_agent.skills.base import SkillError


def build_failure_record(
    error: SkillError,
    *,
    skill_name: str,
    remaining_budget: int,
    attempted_repairs: list[str],
    artifact_references: list[str],
) -> FailureRecord:
    return FailureRecord(
        failure_id=f"failure-{uuid.uuid4().hex[:12]}",
        skill_name=skill_name,
        failure_class=error.failure_class,
        severity=error.severity,
        retryable=error.retryable,
        observed_evidence=str(error),
        likely_causes=error.likely_causes,
        attempted_repairs=attempted_repairs,
        remaining_budget=remaining_budget,
        recommended_action=decide(error, remaining_budget),
        artifact_references=artifact_references,
    )


def decide(error: SkillError, remaining_budget: int) -> RecoveryDecision:
    """Map a typed failure to a bounded recovery decision."""
    if error.severity is Severity.CRITICAL:
        return RecoveryDecision.ABORT
    if not error.retryable:
        # e.g. unsupported claims or validation errors: a human must decide.
        if error.failure_class in (FailureClass.UNSUPPORTED_CLAIM, FailureClass.VALIDATION_ERROR):
            return RecoveryDecision.ESCALATE
        return (
            error.recommended_action
            if error.recommended_action in (RecoveryDecision.PIVOT, RecoveryDecision.ESCALATE)
            else RecoveryDecision.ESCALATE
        )
    if remaining_budget <= 0:
        return RecoveryDecision.ESCALATE
    if error.repair_params:
        return RecoveryDecision.REFINE
    return RecoveryDecision.RETRY
