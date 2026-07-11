"""Experiment decision schema."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DecisionType(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    REFINE = "refine"
    PIVOT_REQUEST = "pivot_request"
    ESCALATE = "escalate"
    ABORT = "abort"


class ExperimentDecision(BaseModel):
    """A decision based on an evaluation outcome. Immutable."""

    model_config = ConfigDict(extra="forbid")

    decision: DecisionType
    legality: bool
    engineering_success: bool
    validation_improvement: bool
    cost: float
    reproducibility: bool
    scientific_uncertainty: float

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
