"""ExperimentProposal: one falsifiable, bounded, typed experiment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.mutation import Mutation
from mlip_research_agent.research.auto_research.validators import ContentAddressedModel


class EstimatedCompute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device: str = Field(pattern=r"^(cpu|gpu)$")
    estimated_seconds: float = Field(gt=0)
    remote: bool = False


class ExpectedObservable(BaseModel):
    """What the proposal expects to see if its mechanism is right."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    expected_direction: str = Field(pattern=r"^(decrease|increase|unchanged)$")
    rationale: str = Field(min_length=5, max_length=1000)


class ExperimentProposal(ContentAddressedModel):
    """A bounded next experiment. Must state what would prove it wrong."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    objective_id: str = Field(min_length=3)
    parent_iteration_id: str | None = Field(
        default=None,
        description="None only for the first proposal; later proposals name the "
        "iteration whose outcome generated them",
    )
    hypothesis: str = Field(min_length=10, max_length=2000)
    proposed_mutations: list[Mutation] = Field(min_length=1, max_length=10)
    expected_mechanism: str = Field(min_length=10, max_length=2000)
    expected_observables: list[ExpectedObservable] = Field(min_length=1, max_length=10)
    falsification_condition: str = Field(
        min_length=10,
        max_length=2000,
        description="The observation that would prove the hypothesis wrong",
    )
    estimated_compute: EstimatedCompute
    required_skills: list[str] = Field(default_factory=list, max_length=20)
    risk_class: str = Field(pattern=r"^(low|medium|high)$")
    required_approval: str | None = Field(
        default=None, description="Human gate name when the proposal needs one; None otherwise"
    )
    seed: int = Field(ge=0)
    proposal_policy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _sealed_mutations(self) -> ExperimentProposal:
        for m in self.proposed_mutations:
            if not m.verify_seal():
                raise ValueError("every proposed mutation must be sealed and verified")
        return self
