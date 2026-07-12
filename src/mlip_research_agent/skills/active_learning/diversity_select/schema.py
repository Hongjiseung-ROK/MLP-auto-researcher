"""Typed contract for descriptor-space diversity selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.skills.active_learning.selection_types import CandidateMeta

POLICY_VERSION = "diversity-select/1.0.0"


class DescriptorSet(BaseModel):
    """Explicit descriptor artifact: candidate_id -> fixed-length vector.

    ``source`` declares descriptor provenance. ``fixture`` is the
    deterministic CI path; ``mace_descriptor_adapter`` is the reviewed
    label-free boundary for real MACE invariant node features.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["fixture", "mace_descriptor_adapter"]
    dimension: int = Field(gt=0, le=4096)
    vectors: dict[str, list[float]] = Field(min_length=1)


class DiversitySelectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_candidate_ids: list[str] = Field(min_length=1, max_length=100_000)
    metadata: dict[str, CandidateMeta]
    descriptors: DescriptorSet | None = None
    descriptor_artifact: str | None = None
    budget: int = Field(gt=0, le=10_000)
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _descriptor_source_boundary(self) -> DiversitySelectInput:
        if (self.descriptors is None) == (self.descriptor_artifact is None):
            raise ValueError("provide exactly one of descriptors or descriptor_artifact")
        if self.descriptors is not None and self.descriptors.source != "fixture":
            raise ValueError(
                "mace_descriptor_adapter vectors must come from a registered artifact"
            )
        return self


class DiversitySelectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_artifact: str
    selection_path: str
    n_selected: int = Field(gt=0)
    mean_pairwise_distance: float = Field(ge=0)
    policy_version: str
