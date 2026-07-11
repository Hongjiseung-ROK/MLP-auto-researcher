"""Typed contract for the acquisition decision gate.

Pipeline: invalid filtering → uncertainty-window construction → diversity
selection → exact budget enforcement → machine-readable reason artifact.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.skills.active_learning.diversity_select.schema import DescriptorSet
from mlip_research_agent.skills.active_learning.selection_types import CandidateMeta

POLICY_VERSION = "decision-gate/1.0.0"


class UncertaintySignal(BaseModel):
    """One candidate's acquisition signal, produced by ensemble_uq."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    force_disagreement: float = Field(
        description="Ensemble disagreement (never called true uncertainty)"
    )
    invalid: bool = False


class DecisionGateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_candidate_ids: list[str] = Field(min_length=1, max_length=100_000)
    metadata: dict[str, CandidateMeta]
    signals: list[UncertaintySignal] = Field(min_length=1)
    descriptors: DescriptorSet
    budget: int = Field(gt=0, le=10_000)
    uncertainty_window_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Top fraction of valid candidates (by disagreement) that stay "
        "eligible for the diversity stage",
    )
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_signals(self) -> DecisionGateInput:
        ids = [s.candidate_id for s in self.signals]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate uncertainty signals for one candidate")
        return self


class DecisionGateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_artifact: str
    selection_path: str
    n_selected: int = Field(gt=0)
    n_filtered_invalid: int = Field(ge=0)
    n_in_window: int = Field(gt=0)
    policy_version: str
