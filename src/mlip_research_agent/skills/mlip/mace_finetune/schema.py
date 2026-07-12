"""Typed contracts for resumable, leakage-safe MACE fine-tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FineTuneSubsetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    role: Literal["train", "validation"]
    dataset_id: str = Field(min_length=1)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
    record_ids: list[str] = Field(min_length=1)
    training_lineage_artifact: str | None = None

    @model_validator(mode="after")
    def _canonical_ids(self) -> FineTuneSubsetManifest:
        if self.record_ids != sorted(set(self.record_ids)):
            raise ValueError("fine-tune subset record ids must be sorted and unique")
        if self.role == "train" and self.training_lineage_artifact is None:
            raise ValueError("training subset must reference its training-lineage artifact")
        if self.role == "validation" and self.training_lineage_artifact is not None:
            raise ValueError("validation subset cannot reference training lineage")
        return self

    def save(self, path: Path) -> str:
        text = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()


class MACEFineTuneInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    dataset_artifact: str
    dataset_manifest_artifact: str
    split_manifest_artifact: str
    train_subset_manifest_artifact: str
    validation_subset_manifest_artifact: str
    training_lineage_artifact: str
    checkpoint_manifest_path: str
    checkpoint_path: str
    execution_mode: Literal["boundary_test", "pilot", "authorized_campaign"] = (
        "boundary_test"
    )
    h2_preregistration_frozen: bool = False
    campaign_id: str | None = None
    research_spec_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    campaign_authorization_artifact: str | None = None
    run_name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    seed: int = Field(ge=0)
    max_epochs: int = Field(gt=0, le=500)
    max_optimizer_steps: int = Field(gt=0, le=100000)
    early_stopping_patience: int = Field(gt=0)
    gradient_clip: float = Field(gt=0)
    optimizer: Literal["adam", "adamw"] = "adam"
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    valid_batch_size: int = Field(gt=0)
    epoch_mode: Literal["single_batch", "full_epoch"] = "single_batch"
    e0_policy: Literal["foundation"] = "foundation"
    energy_loss_weight: float = Field(default=1.0, gt=0)
    force_loss_weight: float = Field(default=100.0, gt=0)
    trainable_layer_policy: Literal["all", "readout_only", "last_interaction_and_readout"] = "all"
    device: Literal["cpu", "cuda"] = "cpu"
    default_dtype: Literal["float32", "float64"] = "float32"
    max_wall_seconds: int = Field(default=3600, gt=0, le=7200)
    resume_state_artifact: str | None = None

    @model_validator(mode="after")
    def _approval_boundary(self) -> MACEFineTuneInput:
        if self.execution_mode == "pilot" and not self.h2_preregistration_frozen:
            raise ValueError("pilot fine-tuning requires a frozen H2 preregistration")
        if self.execution_mode == "authorized_campaign":
            missing = [
                name
                for name, value in (
                    ("campaign_id", self.campaign_id),
                    ("research_spec_sha256", self.research_spec_sha256),
                    ("campaign_authorization_artifact", self.campaign_authorization_artifact),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "authorized campaign fine-tuning requires " + ", ".join(missing)
                )
        if self.execution_mode == "boundary_test":
            if self.h2_preregistration_frozen:
                raise ValueError("boundary test cannot claim H2 preregistration approval")
            if self.max_epochs > 2 or self.max_optimizer_steps > 2:
                raise ValueError("boundary test is limited to two epochs/optimizer steps")
            if self.epoch_mode != "single_batch":
                raise ValueError("boundary test requires single_batch epoch mode")
            if any(
                value is not None
                for value in (
                    self.campaign_id,
                    self.research_spec_sha256,
                    self.campaign_authorization_artifact,
                )
            ):
                raise ValueError("boundary test cannot carry campaign authorization fields")
        return self


class MACEFineTuneOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_artifact: str
    model_path: str
    model_manifest_artifact: str
    model_manifest_path: str
    checkpoint_artifact: str
    checkpoint_path: str
    training_state_artifact: str
    training_state_path: str
    training_metrics_artifact: str
    training_metrics_path: str
    config_artifact: str
    config_path: str
    n_train: int = Field(gt=0)
    n_validation: int = Field(gt=0)
    completed_epochs: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    changed_parameter_tensors: int = Field(gt=0)
    resumed: bool
    trainable_parameter_names: list[str] = Field(default_factory=list)
    frozen_parameter_names: list[str] = Field(default_factory=list)
    changed_frozen_parameter_tensors: int = Field(default=0, ge=0)


class FineTuneResumeState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    run_name: str
    seed: int
    completed_epochs: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    best_validation_force_mae_ev_per_a: float = Field(ge=0)
    patience_count: int = Field(ge=0)
    checkpoint_artifact: str
    model_artifact: str
    resume_contract_sha256: str = Field(min_length=64, max_length=64)
    base_checkpoint_sha256: str = Field(min_length=64, max_length=64)
    dataset_content_sha256: str = Field(min_length=64, max_length=64)
    split_semantic_sha256: str = Field(min_length=64, max_length=64)
