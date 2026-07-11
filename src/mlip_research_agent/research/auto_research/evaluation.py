"""EvaluationOutcome: produced only by an evaluator boundary that is not the
controller. The controller receives aggregates — never protected per-record
test errors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.validators import (
    SHA256_HEX_LENGTH,
    ContentAddressedModel,
)


class ConstraintResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    passed: bool
    observed: float
    limit: float
    description: str = Field(min_length=3, max_length=500)


class BaselineReference(BaseModel):
    """Where the comparison numbers come from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: str = Field(min_length=1)
    metrics: dict[str, float]
    source_artifact: str | None = None


class EvaluationOutcome(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    evaluation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    execution_id: str = Field(min_length=3)
    evaluator_name: str = Field(min_length=1)
    evaluator_source_sha256: str = Field(
        min_length=SHA256_HEX_LENGTH,
        max_length=SHA256_HEX_LENGTH,
        description="SHA-256 of the evaluator's source file: pins the exact judge",
    )
    input_prediction_artifacts: list[str] = Field(min_length=1)
    metric_artifacts: list[str] = Field(min_length=1)
    legality_result_sha256: str = Field(
        min_length=SHA256_HEX_LENGTH, max_length=SHA256_HEX_LENGTH
    )
    aggregate_metrics: dict[str, float] = Field(
        min_length=1, description="Aggregates only; per-record errors never leave the evaluator"
    )
    constraint_results: list[ConstraintResult]
    baseline_reference: BaselineReference
    resource_metrics: dict[str, float] = Field(default_factory=dict)
    uncertainty_notes: str = Field(min_length=5, max_length=2000)
    scientific_status: str = Field(pattern=r"^(non_scientific|staging_only|pilot_only)$")

    @model_validator(mode="after")
    def _no_per_record_leakage(self) -> EvaluationOutcome:
        # Aggregate keys must not smuggle per-record payloads: a bounded,
        # structural guard (the trace grader re-checks the artifacts).
        for key in self.aggregate_metrics:
            if key.startswith("record_") or "::" in key:
                raise ValueError(f"aggregate metric key {key!r} looks per-record")
        return self
