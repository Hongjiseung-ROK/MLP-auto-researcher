"""Typed contract for diversity selection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DiversitySelectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(description="Run-dir-relative path to the pool dataset")
    pool_ids: list[str] = Field(min_length=1, description="List of IDs to select from")
    budget: int = Field(gt=0, description="Exact number of candidates to select")
    seed: int = Field(description="Deterministic random seed for tie-breaking")
    descriptor_path: str = Field(description="Run-dir-relative path to the candidate descriptors (JSON map config_id -> float array)")


class DiversitySelectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_ids: list[str] = Field(description="Selected candidate IDs via farthest point sampling")
    reasons: dict[str, str] = Field(description="Machine-readable reasons for selection")
    selection_artifact: str = Field(description="Artifact ID of the selection record")
    selection_path: str = Field(description="Relative path of the selection record")
