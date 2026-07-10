"""Scientific claim registry schema.

Every numerical claim must reference concrete artifacts; agent text is never
evidence (plan.md data-provenance policy).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ClaimStatus(StrEnum):
    REGISTERED = "registered"
    VERIFIED = "verified"
    REJECTED = "rejected"


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")
    statement: str = Field(min_length=1)
    value: float | int | str
    units: str = ""
    created_by_step: str
    artifact_references: list[str] = Field(
        min_length=1, description="Artifact ids in the run manifest that back this claim"
    )
    status: ClaimStatus = ClaimStatus.REGISTERED
    rejection_reason: str | None = None
