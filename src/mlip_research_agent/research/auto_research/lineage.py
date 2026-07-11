"""Experiment lineage schema."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class ExperimentLineage(BaseModel):
    """Sequence of content-addressed artifacts making up a full experiment cycle."""

    model_config = ConfigDict(extra="forbid")

    objective_hash: str = Field(min_length=1)
    proposal_hash: str = Field(min_length=1)
    tea_time_hash: str = Field(min_length=1)
    legality_hash: str = Field(min_length=1)
    mutation_hash: str = Field(min_length=1)
    execution_hash: str = Field(min_length=1)
    evaluation_hash: str = Field(min_length=1)
    decision_hash: str = Field(min_length=1)
    lesson_hash: str = Field(min_length=1)
    next_proposal_hash: str | None = None

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
