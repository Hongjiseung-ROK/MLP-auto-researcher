"""ExperimentExecution: the evidence record of what actually ran."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.validators import (
    SHA256_HEX_LENGTH,
    ContentAddressedModel,
)


class ResourceUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    wall_seconds: float = Field(ge=0)
    peak_memory_mb: float = Field(ge=0)
    device: str = Field(pattern=r"^(cpu|gpu)$")


class EventLogRange(BaseModel):
    """Half-open [first, last] sequence span in the run's events.jsonl."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first_sequence: int = Field(ge=0)
    last_sequence: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> EventLogRange:
        if self.last_sequence < self.first_sequence:
            raise ValueError("last_sequence must be >= first_sequence")
        return self


class ExperimentExecution(ContentAddressedModel):
    """References everything an executed experiment depended on and produced."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    proposal_artifact_sha256: str = Field(
        min_length=SHA256_HEX_LENGTH, max_length=SHA256_HEX_LENGTH
    )
    git_commit: str = Field(min_length=7, max_length=40)
    environment_artifact: str = Field(min_length=1)
    config_artifact: str = Field(min_length=1)
    dataset_artifact: str = Field(
        min_length=1, description="For synthetic fixtures: the fixture-definition artifact"
    )
    split_artifact: str | None = Field(
        default=None, description="None when the workload has no data split (fixture demos)"
    )
    model_artifact: str | None = Field(
        default=None, description="None when the workload trains no persisted model"
    )
    compute_attestation: str = Field(
        min_length=1,
        description="Attestation artifact id, or the literal 'local-cpu' for local runs",
    )
    event_log_range: EventLogRange
    produced_artifacts: list[str] = Field(min_length=1)
    failure_artifact: str | None = None
    succeeded: bool
    resource_usage: ResourceUsage

    @model_validator(mode="after")
    def _failure_consistency(self) -> ExperimentExecution:
        if not self.succeeded and self.failure_artifact is None:
            raise ValueError("a failed execution must reference its failure artifact")
        if self.succeeded and self.failure_artifact is not None:
            raise ValueError("a successful execution cannot carry a failure artifact")
        return self
