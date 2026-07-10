"""Typed inputs/outputs for the mock acquisition skill."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AcquisitionStrategy(StrEnum):
    MOCK_UNCERTAINTY = "mock_uncertainty"
    RANDOM = "random"


class AcquisitionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative structure set path")
    n_select: int = Field(gt=0, le=10_000)
    strategy: AcquisitionStrategy = AcquisitionStrategy.MOCK_UNCERTAINTY


class AcquisitionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_artifact: str
    selection_path: str
    n_selected: int = Field(gt=0)
    strategy: AcquisitionStrategy
