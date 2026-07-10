"""Typed inputs/outputs for the mock DFT labeling skill."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LabelingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structures_path: str
    selection_path: str | None = Field(
        default=None, description="Optional selection artifact; omit to label everything"
    )
    method: str = Field(default="mock_lj", description="Only 'mock_lj' is supported in v0")
    scf_damping: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mock convergence knob; set by REFINE repairs and recorded in labels",
    )
    inject_failure_times: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Demo/testing: fail the first N attempts with a recoverable mock SCF error",
    )


class LabelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    energy: float
    fmax: float = Field(ge=0.0)


class LabelingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels_artifact: str
    labels_path: str
    n_labeled: int = Field(gt=0)
    method: str
