"""Validation for the tea-time research-pause skill."""

from __future__ import annotations

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


FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "api_key",
        "energy_ev",
        "forces_ev_per_a",
        "hidden_labels",
        "per_record_errors",
        "secret",
        "stress_ev_per_a3",
        "test_metric_details",
    }
)


def assert_no_sensitive_payload(value: Any) -> None:
    """Fail if a future edit tries to serialize restricted research data."""
    if isinstance(value, dict):
        forbidden = FORBIDDEN_PAYLOAD_KEYS.intersection(value)
        if forbidden:
            raise SkillError(
                f"Tea Time payload contains forbidden keys: {sorted(forbidden)}",
                failure_class=FailureClass.VALIDATION_ERROR,
                severity=Severity.HIGH,
                retryable=False,
            )
        for nested in value.values():
            assert_no_sensitive_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_no_sensitive_payload(nested)


def assert_claim_count_unchanged(before: int, after: int) -> None:
    """Tea-time output is agent-text class: it must never add claims."""
    if after != before:
        raise SkillError(
            "tea_time_with_reading_poem changed the claim count; its output is "
            "reflection, never evidence",
            failure_class=FailureClass.UNSUPPORTED_CLAIM,
            severity=Severity.HIGH,
            retryable=False,
        )
