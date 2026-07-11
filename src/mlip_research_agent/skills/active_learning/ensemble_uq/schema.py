"""Typed contract for ensemble-disagreement acquisition signals.

Terminology is deliberate: the skill computes *disagreement* between
explicitly identified ensemble members. Disagreement is an acquisition
ranking signal, not calibrated "true uncertainty", and the output says so.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.skills.active_learning.selection_types import CandidateMeta

POLICY_VERSION = "ensemble-uq/1.0.0"

MIN_ENSEMBLE_MEMBERS = 3


class EnsembleMember(BaseModel):
    """One identified ensemble member and its per-candidate predictions."""

    model_config = ConfigDict(extra="forbid")

    member_id: str = Field(min_length=1)
    provenance_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Content hash of the member's model manifest / checkpoint",
    )
    # candidate_id -> (n_atoms x 3) force components, model units.
    predicted_forces: dict[str, list[list[float]]] = Field(min_length=1)
    # candidate_id -> predicted energy per atom (optional signal).
    predicted_energy_per_atom: dict[str, float] = Field(default_factory=dict)


class EnsembleUQInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_candidate_ids: list[str] = Field(min_length=1, max_length=100_000)
    metadata: dict[str, CandidateMeta]
    members: list[EnsembleMember] = Field(min_length=MIN_ENSEMBLE_MEMBERS, max_length=16)
    include_energy_disagreement: bool = False
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_members(self) -> EnsembleUQInput:
        ids = [m.member_id for m in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError("ensemble member ids must be unique")
        hashes = [m.provenance_sha256 for m in self.members]
        if len(set(hashes)) != len(hashes):
            raise ValueError("ensemble members must have distinct provenance hashes")
        return self


class EnsembleUQOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uq_artifact: str
    uq_path: str
    n_candidates_ranked: int = Field(ge=0)
    n_candidates_invalid: int = Field(ge=0)
    invalid_member_flags: dict[str, list[str]] = Field(
        default_factory=dict,
        description="candidate_id -> member ids that produced non-finite predictions",
    )
    signal_kind: str = Field(
        default="ensemble_disagreement",
        pattern=r"^ensemble_disagreement$",
        description="This is member disagreement, not calibrated uncertainty",
    )
    policy_version: str
