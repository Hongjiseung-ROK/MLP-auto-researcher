"""Typed I/O for the DeePMD inference adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DeepMDInferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    model_path: str = Field(description="Local DeePMD model file (never downloaded)")
    model_sha256: str = Field(
        min_length=64, max_length=64, description="Pinned model hash; mismatch aborts"
    )
    device: Literal["cpu"] = "cpu"


class DeepMDInferenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions_artifact: str
    predictions_path: str
    n_structures: int = Field(gt=0)
    model_sha256: str = Field(min_length=64, max_length=64)
