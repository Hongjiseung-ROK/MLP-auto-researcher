"""ResearchLesson: what the loop learned. Agent prose is not evidence — every
observed result must reference verifiable artifacts."""

from __future__ import annotations

from pydantic import ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.mutation import MutationClass
from mlip_research_agent.research.auto_research.validators import ContentAddressedModel


class ResearchLesson(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid")

    lesson_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    iteration_id: str = Field(min_length=3)
    attempt_summary: str = Field(min_length=10, max_length=2000)
    observed_result: str = Field(min_length=10, max_length=2000)
    evidence_references: list[str] = Field(
        min_length=1, description="Artifact ids backing the observed result; prose alone is void"
    )
    likely_explanation: str = Field(min_length=10, max_length=2000)
    alternative_explanations: list[str] = Field(min_length=1, max_length=10)
    limits: str = Field(min_length=10, max_length=2000)
    applicability_conditions: str = Field(min_length=10, max_length=2000)
    anti_pattern: str = Field(
        min_length=5, max_length=1000, description="What the next iteration must not repeat"
    )
    recommended_next_mutation_class: MutationClass

    @model_validator(mode="after")
    def _bounded_recommendation(self) -> ResearchLesson:
        if self.recommended_next_mutation_class in {
            MutationClass.IMMUTABLE,
            MutationClass.OWNER_GATED,
        }:
            raise ValueError(
                "a lesson cannot recommend mutating immutable or owner-gated parameters"
            )
        return self
