"""Typed inputs/outputs for the tea-time-with-reading-poem skill."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TeaTimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_question: str = Field(
        min_length=5,
        max_length=500,
        description="The research question the agent has been staring at",
    )
    data_path: str | None = Field(
        default=None,
        description="Optional run-dir-relative JSON artifact (labels or metrics) to re-view",
    )
    n_provocations: int = Field(default=4, ge=1, le=7)
    poem_key: str | None = Field(
        default=None, description="Override the seeded poem choice (must exist in the corpus)"
    )


class TeaTimeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_artifact: str
    report_path: str
    reframings_artifact: str
    reframings_path: str
    poem_key: str
    techniques: list[str]
    n_provocations: int = Field(ge=1)
