"""Typed inputs/outputs for the claim verification skill."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims_path: str = Field(default="claims.json", description="Run-dir-relative claims file")
    strict: bool = Field(
        default=True, description="Fail the step if any claim lacks verified backing artifacts"
    )


class VerificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_artifact: str
    report_path: str
    verified_claims_artifact: str
    verified_claims_path: str
    n_claims: int = Field(ge=0)
    n_verified: int = Field(ge=0)
    n_rejected: int = Field(ge=0)
