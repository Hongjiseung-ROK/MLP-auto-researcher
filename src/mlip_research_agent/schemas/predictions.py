"""Shared, evaluator-ready MLIP prediction artifact schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    record_id: str = Field(min_length=1)
    energy_ev: float
    forces_ev_per_a: list[list[float]] = Field(min_length=1)
    stress_ev_per_a3_voigt6: list[float] | None = None

    @model_validator(mode="after")
    def _force_shape(self) -> PredictionRecord:
        if any(len(row) != 3 for row in self.forces_ev_per_a):
            raise ValueError(f"{self.record_id}: predicted forces must have shape n_atoms x 3")
        return self


class PredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["2.0.0"] = "2.0.0"
    dataset_id: str = Field(min_length=1)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    model_id: str = Field(min_length=1)
    model_manifest_artifact: str = Field(min_length=1)
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    energy_unit: Literal["eV"]
    force_unit: Literal["eV/angstrom"]
    device: str | None = None
    default_dtype: str | None = None
    structure_set_sha256: str = Field(min_length=64, max_length=64)
    predictions: list[PredictionRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_record_ids(self) -> PredictionBatch:
        ids = [record.record_id for record in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("prediction record ids must be unique")
        return self
