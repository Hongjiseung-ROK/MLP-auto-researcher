"""Research objective representing the core mission of the campaign."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResearchObjective(BaseModel):
    """The overarching research goal and constraints. Immutable."""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(min_length=1)
    research_goal: str = Field(min_length=1)
    scientific_domain: str = Field(min_length=1)
    benchmark_adapter: str = Field(min_length=1)
    success_criteria: str = Field(min_length=1)
    budgets: dict[str, Any]
    maximum_iterations: int = Field(gt=0)
    protected_constants: list[str]
    allowed_mutation_classes: list[str]
    evaluator_identity: str = Field(min_length=1)
    stopping_rules: dict[str, Any]
    approval_requirements: list[str]
    scientific_status_ceiling: str = Field(min_length=1)

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
