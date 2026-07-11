"""Typed contract for decision gate."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DecisionGateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(description="Run-dir-relative path to the pool dataset")
    pool_ids: list[str] = Field(min_length=1, description="List of IDs to consider")
    budget: int = Field(gt=0, description="Exact number of candidates to select")
    seed: int = Field(description="Deterministic random seed")
    policy_identity: str = Field(description="Name of the active learning policy")
    uncertainty_scores_path: str = Field(description="Path to JSON mapping config_id to uncertainty score")
    descriptors_path: str = Field(description="Path to JSON mapping config_id to descriptor float array")


class DecisionGateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_ids: list[str] = Field(description="Final selected candidate IDs")
    rejected_ids: list[str] = Field(description="Final rejected candidate IDs")
    reasons: dict[str, str] = Field(description="Machine-readable reasons for selection/rejection")
    source_hashes: dict[str, str] = Field(description="Source hashes for provenance")
    decision_artifact: str = Field(description="Artifact ID of the decision record")
    decision_path: str = Field(description="Relative path of the decision record")
