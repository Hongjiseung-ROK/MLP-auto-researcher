"""Typed contract for the label-free random-selection arm."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.skills.active_learning.selection_types import CandidateMeta

POLICY_VERSION = "random-select/1.0.0"


class RandomSelectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_candidate_ids: list[str] = Field(min_length=1, max_length=100_000)
    metadata: dict[str, CandidateMeta] = Field(
        description="Label-free metadata keyed by candidate id (every pool id required)"
    )
    budget: int = Field(gt=0, le=10_000)
    seed: int = Field(ge=0)
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)


class RandomSelectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_artifact: str
    selection_path: str
    n_selected: int = Field(gt=0)
    policy_version: str
