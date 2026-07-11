"""Typed I/O for pinned MACE inference."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MACEInferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str = Field(description="Run-dir-relative StructureSet JSON")
    checkpoint_manifest_path: str = Field(description="Repository/local manifest JSON")
    checkpoint_path: str = Field(description="Local gitignored checkpoint path")
    dataset_id: str = Field(min_length=1)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    record_ids: list[str] = Field(
        min_length=1,
        description="Dataset record ids in the exact order of structures_path",
    )
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
