"""Evaluation outcome schema."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationOutcome(BaseModel):
    """The outcome of an independent evaluation. Immutable."""

    model_config = ConfigDict(extra="forbid")

    evaluator_identity: str = Field(min_length=1)
    evaluator_source_hash: str = Field(min_length=1)
    legality_result: bool
    aggregate_validation_metrics: dict[str, float]
    resource_metrics: dict[str, float]
    constraints: dict[str, Any]
    comparison_baseline: dict[str, float]
    uncertainty: dict[str, float]
    artifact_references: dict[str, str]

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
