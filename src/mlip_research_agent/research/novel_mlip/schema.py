"""Typed, append-only state for the novel MLIP research campaign."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ArmName = Literal["random", "disagreement", "disagreement_fps", "tail_risk_fps"]


class ModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: str
    round_index: int = Field(ge=0, le=3)
    seed: int
    model_path: str
    model_sha256: str = Field(min_length=64, max_length=64)
    training_checkpoint_path: str
    training_checkpoint_sha256: str = Field(min_length=64, max_length=64)
    manifest_path: str
    manifest_sha256: str = Field(min_length=64, max_length=64)
    training_data_sha256: str = Field(min_length=64, max_length=64)
    optimizer_steps: int = Field(gt=0)
    completed_epochs: int = Field(gt=0)
    runtime_seconds: float = Field(gt=0)
    peak_gpu_memory_mb: float = Field(ge=0)


class SelectionRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_index: int = Field(ge=1, le=3)
    policy: ArmName
    selected_record_ids: list[str] = Field(min_length=12, max_length=12)
    selection_artifact_path: str
    selection_artifact_sha256: str = Field(min_length=64, max_length=64)
    prior_training_head_sha256: str = Field(min_length=64, max_length=64)
    new_training_head_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _unique_ids(self) -> SelectionRound:
        if self.selected_record_ids != sorted(set(self.selected_record_ids)):
            raise ValueError("selected record ids must be sorted and unique")
        return self


class ArmState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: ArmName
    cumulative_training_record_ids: list[str]
    selections: list[SelectionRound] = Field(default_factory=list)
    current_models: list[ModelIdentity] = Field(min_length=3, max_length=3)
    validation_history_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _canonical_state(self) -> ArmState:
        if self.cumulative_training_record_ids != sorted(
            set(self.cumulative_training_record_ids)
        ):
            raise ValueError("cumulative training ids must be sorted and unique")
        if sorted(model.seed for model in self.current_models) != [42, 43, 44]:
            raise ValueError("every arm state requires the frozen three model seeds")
        return self


class CampaignState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    campaign_id: Literal["cu-tail-risk-ralph-20260712"] = (
        "cu-tail-risk-ralph-20260712"
    )
    research_spec_sha256: str = Field(min_length=64, max_length=64)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    dataset_file_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    base_checkpoint_sha256: str = Field(min_length=64, max_length=64)
    provider: str
    completed_round: int = Field(ge=0, le=3)
    frozen_initial_record_ids: list[str] = Field(min_length=32, max_length=32)
    d0_models: list[ModelIdentity] = Field(min_length=3, max_length=3)
    arms: dict[ArmName, ArmState]
    zero_shot_validation_path: str
    static_validation_path: str
    raw_descriptor_path: str
    raw_descriptor_sha256: str = Field(min_length=64, max_length=64)
    numerical_calibration_path: str
    oracle_ledger_path: str
    experiment_ledger_path: str
    learning_rate_used: float = Field(gt=0)

    @model_validator(mode="after")
    def _arms_and_rounds(self) -> CampaignState:
        expected = {"random", "disagreement", "disagreement_fps", "tail_risk_fps"}
        if set(self.arms) != expected:
            raise ValueError("campaign state must contain every frozen active arm")
        if any(len(arm.selections) != self.completed_round for arm in self.arms.values()):
            raise ValueError("arm selection histories do not match completed round")
        expected_labels = 32 + 12 * self.completed_round
        if any(
            len(arm.cumulative_training_record_ids) != expected_labels
            for arm in self.arms.values()
        ):
            raise ValueError("arm label heads do not match the frozen round budget")
        return self
