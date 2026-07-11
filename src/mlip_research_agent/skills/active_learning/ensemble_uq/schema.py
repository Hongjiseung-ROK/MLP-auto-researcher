"""Typed contract for ensemble uncertainty quantification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EnsembleUqInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(description="Run-dir-relative path to the unlabelled pool dataset")
    pool_ids: list[str] = Field(min_length=1, description="List of IDs to rank")
    member_identities: list[str] = Field(min_length=3, description="Declared independent member identities")
    member_prediction_paths: list[str] = Field(min_length=3, description="Paths to predicted datasets for each member")
    aggregation_formula: str = Field(description="Explicit aggregation formula, e.g., 'mean_std_dev'")
    include_energy_disagreement: bool = Field(default=False, description="Optional energy disagreement calculation")


class EnsembleUqOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_ids: list[str] = Field(description="Candidate IDs ranked by uncertainty (descending)")
    uncertainty_scores: dict[str, float] = Field(description="Ranking scores (never call this true error)")
    uq_artifact: str = Field(description="Artifact ID of the UQ record")
    uq_path: str = Field(description="Relative path of the UQ record")
