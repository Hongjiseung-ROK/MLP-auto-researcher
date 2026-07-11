"""Experiment proposal schema."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentProposal(BaseModel):
    """A proposed experiment before execution or mutation. Immutable."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1)
    parent_objective_id: str = Field(min_length=1)
    parent_experiment_id: str | None = None
    explicit_hypothesis: str = Field(min_length=1)
    mutation: dict[str, Any]
    expected_mechanism: str = Field(min_length=1)
    predicted_observable: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    estimated_compute: str = Field(min_length=1)
    affected_config_paths: list[str]
    risk_class: str = Field(min_length=1)
    required_approval: str = Field(min_length=1)
    seed: int = Field(ge=0)
    proposal_source: str = Field(min_length=1)

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
