"""Typed contract for the force-tail acquisition proxy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.skills.active_learning.ensemble_uq.schema import EnsembleMember
from mlip_research_agent.skills.active_learning.selection_types import CandidateMeta

POLICY_VERSION = "force-tail-proxy/1.0.0"


class TailRiskScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pool_candidate_ids: list[str] = Field(min_length=1, max_length=100_000)
    metadata: dict[str, CandidateMeta]
    members: list[EnsembleMember] = Field(min_length=3, max_length=16)
    tail_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    magnitude_weight: float = Field(default=1.0, ge=0.0)
    disagreement_weight: float = Field(default=1.0, ge=0.0)
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_and_weights(self) -> TailRiskScoreInput:
        member_ids = [member.member_id for member in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("ensemble member ids must be unique")
        provenance = [member.provenance_sha256 for member in self.members]
        if len(set(provenance)) != len(provenance):
            raise ValueError("ensemble members must have distinct provenance hashes")
        if self.magnitude_weight == 0.0 and self.disagreement_weight == 0.0:
            raise ValueError("at least one force-tail proxy weight must be positive")
        return self


class TailRiskScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_artifact: str
    score_path: str
    n_candidates_ranked: int = Field(ge=0)
    n_candidates_invalid: int = Field(ge=0)
    invalid_member_flags: dict[str, list[str]] = Field(default_factory=dict)
    signal_kind: str = Field(default="force_tail_proxy", pattern=r"^force_tail_proxy$")
    policy_version: str
