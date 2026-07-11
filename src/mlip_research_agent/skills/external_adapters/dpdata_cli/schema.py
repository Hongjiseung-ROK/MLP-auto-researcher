"""Typed I/O for the dpdata dataset-conversion adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DpdataConvertInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(
        description="Run-dir-relative NormalizedDataset JSON (first-party format)"
    )
    to_format: Literal["deepmd/npy"] = "deepmd/npy"


class DpdataConvertOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_manifest_artifact: str
    output_manifest_path: str
    n_configurations: int = Field(gt=0)
    n_systems: int = Field(gt=0)
    n_files: int = Field(gt=0)
