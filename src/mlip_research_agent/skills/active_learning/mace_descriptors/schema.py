"""Typed I/O for label-free MACE structure descriptors."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MACEDescriptorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    acquisition_view_artifact: str = Field(min_length=1)
    checkpoint_manifest_path: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)
    device: Literal["cpu", "cuda"] = "cpu"
    default_dtype: Literal["float32", "float64"] = "float64"
    num_layers: int = Field(default=-1, ge=-1)
    pooling: Literal["mean"] = "mean"
    min_feature_std: float = Field(default=1.0e-12, gt=0)
    l2_normalize: bool = True

    @field_validator("num_layers")
    @classmethod
    def _nonzero_layer_count(cls, value: int) -> int:
        if value == 0:
            raise ValueError("num_layers cannot be zero")
        return value


class MACEDescriptorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    descriptor_artifact: str
    descriptor_path: str
    n_candidates: int = Field(gt=0)
    raw_dimension: int = Field(gt=0)
    dimension: int = Field(gt=0)
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
