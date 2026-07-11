"""ExperimentDecision: every accept/reject weighs each dimension separately."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.validators import ContentAddressedModel


class DecisionValue(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    REFINE = "refine"
    PIVOT_REQUEST = "pivot_request"
    ESCALATE = "escalate"
    ABORT = "abort"


class DimensionAssessment(BaseModel):
    """One independently evaluated decision dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    detail: str = Field(min_length=3, max_length=1000)


class ExperimentDecision(ContentAddressedModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    proposal_id: str = Field(min_length=3)
    evaluation_id: str | None = Field(
        default=None, description="None only when the decision precedes evaluation (illegal "
        "mutation or unrecoverable execution failure)"
    )
    decision: DecisionValue
    acceptance_policy_version: str = Field(min_length=1)
    # Every decision evaluates each dimension separately:
    legality: DimensionAssessment
    scientific_objective: DimensionAssessment
    engineering_validity: DimensionAssessment
    metric_result: DimensionAssessment
    tail_risk: DimensionAssessment
    compute_budget: DimensionAssessment
    reproducibility: DimensionAssessment
    uncertainty: DimensionAssessment
    rationale: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def _consistent(self) -> ExperimentDecision:
        if self.decision is DecisionValue.ACCEPT:
            failed = [
                name
                for name, dim in self.dimensions().items()
                if not dim.passed
            ]
            if failed:
                raise ValueError(f"cannot accept with failing dimensions: {failed}")
        if self.decision is not DecisionValue.ACCEPT and all(
            dim.passed for dim in self.dimensions().values()
        ):
            raise ValueError("a non-accept decision must name at least one failing dimension")
        return self

    def dimensions(self) -> dict[str, DimensionAssessment]:
        return {
            "legality": self.legality,
            "scientific_objective": self.scientific_objective,
            "engineering_validity": self.engineering_validity,
            "metric_result": self.metric_result,
            "tail_risk": self.tail_risk,
            "compute_budget": self.compute_budget,
            "reproducibility": self.reproducibility,
            "uncertainty": self.uncertainty,
        }
