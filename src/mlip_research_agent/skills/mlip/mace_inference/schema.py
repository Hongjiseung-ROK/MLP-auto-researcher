"""Typed I/O for pinned MACE inference."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MACEInferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    checkpoint_manifest_path: str = Field(description="Repository/local manifest JSON")
    checkpoint_path: str = Field(description="Local gitignored checkpoint path")
    device: Literal["cpu", "cuda"] = "cpu"
    default_dtype: Literal["float32", "float64"] = "float64"


class MACEInferenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions_artifact: str
    predictions_path: str
    model_manifest_artifact: str
    model_manifest_path: str
    model_id: str
    checkpoint_sha256: str
    n_structures: int = Field(gt=0)
    device: str
    default_dtype: str
