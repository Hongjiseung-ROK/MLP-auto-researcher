"""Pilot run state: stages, per-round records, honest completion status.

A partially completed pilot is a first-class outcome (plan_phase_2.md §6.5,
§13.3): the status file must say exactly how far the run got, and a partial
run can never be marked complete.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

COMPLETION_STATUS_NAME = "completion_status.json"


class PilotStage(StrEnum):
    """Ordered scientific stages; checkpoints happen at every boundary."""

    PREREG_VERIFIED = "prereg_verified"
    DATA_VERIFIED = "data_verified"
    SPLIT_VERIFIED = "split_verified"
    MODEL_VERIFIED = "model_verified"
    ZERO_SHOT_EVAL = "zero_shot_eval"
    STATIC_FINETUNE = "static_finetune"
    ACQUISITION_ROUNDS = "acquisition_rounds"
    FINAL_EVAL = "final_eval"
    REPORT = "report"


STAGE_ORDER: tuple[PilotStage, ...] = tuple(PilotStage)


class RoundState(BaseModel):
    """State of one acquisition round for one arm."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    round_index: int = Field(ge=0)
    labels_revealed_this_round: int = Field(ge=0)
    labels_revealed_cumulative: int = Field(ge=0)
    selection_artifact_id: str | None = None
    model_checkpoint_artifact_id: str | None = None
    metrics_artifact_id: str | None = None
    completed: bool = False


class ContinuityBreak(BaseModel):
    """A session discontinuity (e.g. Colab disconnect) and where we resumed."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    resumed_from_stage: PilotStage
    detail: str = ""


class PilotStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    campaign_name: str
    scientific_status: str = "pilot_only"
    preregistration_sha256: str
    repo_commit: str = Field(min_length=7)
    stages_completed: list[PilotStage] = Field(default_factory=list)
    current_stage: PilotStage = PilotStage.PREREG_VERIFIED
    rounds: list[RoundState] = Field(default_factory=list)
    continuity_breaks: list[ContinuityBreak] = Field(default_factory=list)
    complete: bool = False
    partial_reason: str = ""

    @model_validator(mode="after")
    def _honest_completion(self) -> PilotStatus:
        if self.complete:
            missing = [s for s in STAGE_ORDER if s not in self.stages_completed]
            if missing:
                raise ValueError(
                    f"cannot mark complete with unfinished stages: {[s.value for s in missing]}"
                )
            unfinished_rounds = [r for r in self.rounds if not r.completed]
            if unfinished_rounds:
                raise ValueError("cannot mark complete with unfinished acquisition rounds")
            if self.partial_reason:
                raise ValueError("a complete run cannot carry a partial_reason")
        return self

    def mark_stage_complete(self, stage: PilotStage) -> None:
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)
        idx = STAGE_ORDER.index(stage)
        if idx + 1 < len(STAGE_ORDER):
            self.current_stage = STAGE_ORDER[idx + 1]

    def save(self, run_dir: Path) -> Path:
        path = run_dir / COMPLETION_STATUS_NAME
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        return path

    @classmethod
    def load(cls, run_dir: Path) -> PilotStatus:
        return cls.model_validate(
            json.loads((run_dir / COMPLETION_STATUS_NAME).read_text())
        )
