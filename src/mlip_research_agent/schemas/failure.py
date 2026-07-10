"""Failure as a first-class object (plan.md failure-recovery section)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FailureClass(StrEnum):
    VALIDATION_ERROR = "validation_error"
    CONVERGENCE_FAILURE = "convergence_failure"  # SCF-style; mock analogue in v0
    SIMULATION_INSTABILITY = "simulation_instability"
    OOD_BEHAVIOR = "ood_behavior"
    OVERFITTING = "overfitting"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    TOOL_ERROR = "tool_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryDecision(StrEnum):
    RETRY = "retry"
    REFINE = "refine"
    PIVOT = "pivot"
    ESCALATE = "escalate"
    ABORT = "abort"


class FailureRecord(BaseModel):
    """Common failure object emitted for every failed skill attempt."""

    model_config = ConfigDict(extra="forbid")

    failure_id: str
    skill_name: str
    failure_class: FailureClass
    severity: Severity
    retryable: bool
    observed_evidence: str
    likely_causes: list[str] = Field(default_factory=list)
    attempted_repairs: list[str] = Field(default_factory=list)
    remaining_budget: int = Field(ge=0)
    recommended_action: RecoveryDecision
    artifact_references: list[str] = Field(default_factory=list)
