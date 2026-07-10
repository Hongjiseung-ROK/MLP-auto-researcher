"""Typed inputs/outputs for the mock MLIP training and evaluation skills."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrainingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels_path: str
    model_name: str = Field(default="mock_mean_baseline")
    holdout_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)


class TrainingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_artifact: str
    model_path: str
    model_name: str
    n_train: int = Field(gt=0)
    n_holdout: int = Field(gt=0)


class EvaluationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str
    labels_path: str


class EvaluationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics_artifact: str
    metrics_path: str
    energy_mae: float = Field(ge=0.0)
    n_holdout: int = Field(gt=0)
