"""Research lesson schema."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class ResearchLesson(BaseModel):
    """An immutable recorded lesson from an experiment."""

    model_config = ConfigDict(extra="forbid")

    attempted_change: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    evidence: list[str]
    plausible_explanation: str = Field(min_length=1)
    explanation_limitations: str = Field(min_length=1)
    applicability_conditions: str = Field(min_length=1)
    anti_patterns: list[str]
    next_proposal_class: str = Field(min_length=1)

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
