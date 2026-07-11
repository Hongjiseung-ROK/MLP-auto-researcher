"""Typed contract for descriptor-space diversity selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.skills.active_learning.selection_types import CandidateMeta

POLICY_VERSION = "diversity-select/1.0.0"


class DescriptorSet(BaseModel):
    """Explicit descriptor artifact: candidate_id -> fixed-length vector.

    ``source`` declares descriptor provenance. ``fixture`` is the
    deterministic CI path; ``mace_descriptor_adapter`` is the declared future
    boundary for real MACE node features and fails closed until that adapter
    is implemented and reviewed.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["fixture", "mace_descriptor_adapter"]
    dimension: int = Field(gt=0, le=4096)
    vectors: dict[str, list[float]] = Field(min_length=1)


class DiversitySelectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_candidate_ids: list[str] = Field(min_length=1, max_length=100_000)
    metadata: dict[str, CandidateMeta]
    descriptors: DescriptorSet
    budget: int = Field(gt=0, le=10_000)
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)


class DiversitySelectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_artifact: str
    selection_path: str
    n_selected: int = Field(gt=0)
    mean_pairwise_distance: float = Field(ge=0)
    policy_version: str
