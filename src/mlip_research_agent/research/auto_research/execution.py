"""Execution record tracking."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentExecution(BaseModel):
    """The runtime record of a proposed experiment execution. Immutable."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1)
    exact_commit: str = Field(min_length=1)
    configuration_hash: str = Field(min_length=1)
    environment_hash: str = Field(min_length=1)
    data_split_model_artifacts: dict[str, str]
    compute_attestation: str = Field(min_length=1)
    start_time_utc: str
    end_time_utc: str | None = None
    status: ExecutionStatus
    failure: dict[str, Any] | None = None
    resource_use: dict[str, Any] | None = None
    output_artifacts: dict[str, str] | None = None

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
