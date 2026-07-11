"""Machine-readable preregistration and human-approval records.

Mirrors docs/research/phase2_preregistration.md. A preregistration is either
a draft (fields may be pending) or frozen; only a frozen preregistration with
a recorded H2 approval may precede a result-producing run. Frozen means: no
pending fields, and the content hash recorded at freeze time.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.experiment_matrix import ExperimentMatrix

PENDING = "TBD"
"""Sentinel for values the due-diligence work packages have not pinned yet."""


class HumanGate(StrEnum):
    H1_DATASET_PROMOTION = "H1"
    H2_PREREGISTRATION = "H2"
    H3_REMOTE_EXECUTION = "H3"
    H4_SCIENTIFIC_PIVOT = "H4"
    H5_CLAIM_RELEASE = "H5"


class ApprovalRecord(BaseModel):
    """A serialized human decision. Stored append-only in approvals.jsonl."""

    model_config = ConfigDict(extra="forbid")

    gate: HumanGate
    approved: bool
    approved_by: str = Field(min_length=1)
    timestamp_utc: str
    subject_sha256: str = Field(
        min_length=64, max_length=64, description="Hash of the exact artifact approved"
    )
    notes: str = ""

    @classmethod
    def create(
        cls,
        gate: HumanGate,
        *,
        approved: bool,
        approved_by: str,
        subject_sha256: str,
        notes: str = "",
    ) -> ApprovalRecord:
        return cls(
            gate=gate,
            approved=approved,
            approved_by=approved_by,
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            subject_sha256=subject_sha256,
            notes=notes,
        )


class ApprovalMissingError(Exception):
    """Raised when a gated action runs without its recorded human approval."""


def require_approval(
    gate: HumanGate, subject_sha256: str, approvals: list[ApprovalRecord]
) -> ApprovalRecord:
    """Return the approval for (gate, subject) or raise. Fail-closed: a
    rejection or a hash mismatch is as blocking as an absent record."""
    for record in approvals:
        if record.gate is gate and record.subject_sha256 == subject_sha256:
            if not record.approved:
                raise ApprovalMissingError(
                    f"gate {gate.value} was explicitly rejected by {record.approved_by}"
                )
            return record
    raise ApprovalMissingError(
        f"no {gate.value} approval recorded for subject {subject_sha256[:12]}…; "
        "gated actions must not run (plan_phase_2.md §12)"
    )


def load_approvals(path: Path) -> list[ApprovalRecord]:
    if not path.is_file():
        return []
    return [
        ApprovalRecord.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def append_approval(path: Path, record: ApprovalRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(record.model_dump_json() + "\n")


class DatasetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    candidate_source: str
    qualified_manifest_sha256: str = PENDING
    license_spdx: str = PENDING
    level_of_theory: str = PENDING
    overlap_risk_statement: str = PENDING


class SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grouping_rule: str = PENDING
    split_seed: int = Field(ge=0)
    initial_labeled_count: int = Field(gt=0)
    pool_min: int = Field(gt=0)
    pool_max: int = Field(gt=0)
    validation_fraction_min: float = Field(gt=0.0, lt=1.0)
    test_fraction_min: float = Field(gt=0.0, lt=1.0)
    split_manifest_sha256: str = PENDING

    @model_validator(mode="after")
    def _pool_bounds(self) -> SplitSection:
        if self.pool_max < self.pool_min:
            raise ValueError("pool_max must be >= pool_min")
        return self


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str = "mace"
    mace_torch_version: str = PENDING
    torch_version: str = PENDING
    checkpoint_id: str = PENDING
    checkpoint_sha256: str = PENDING
    checkpoint_license: str = PENDING
    eval_dtype: str = "float64"
    train_dtype: str = "float32"
    e0_policy: str = Field(
        min_length=1,
        description="Source of atomic reference energies; silent averaging forbidden",
    )


class FinetuneSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: str = "naive"
    max_epochs: int = Field(gt=0, le=500)
    early_stopping_patience: int = Field(gt=0)
    gradient_clip: float = Field(gt=0.0)
    optimizer: str = PENDING
    learning_rate: str = PENDING
    base_seed: int = Field(ge=0)


class ComputeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload_class: str = "bounded_research_pilot"
    requested_gpu: str = "nvidia-l4"
    oom_fallback_gpu: str = "nvidia-a100-40gb"
    max_runtime_minutes: int = Field(gt=0, le=60)
    max_gpus: int = Field(default=1, ge=1, le=1)


class StoppingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(gt=0, le=3)
    relative_improvement_floor: float = Field(gt=0.0, lt=1.0)
    stagnant_rounds_to_stop: int = Field(default=2, gt=0)


class Preregistration(BaseModel):
    """The frozen scientific protocol. Immutable after H2 without a PIVOT."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0.0"
    campaign_name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    scientific_status: str = "pilot_only"
    research_question: str = Field(min_length=10)
    primary_endpoint: str = "force_mae_ev_per_angstrom_on_frozen_test"
    secondary_endpoints: list[str]
    dataset: DatasetSection
    split: SplitSection
    model_section: ModelSection
    finetune: FinetuneSection
    matrix: ExperimentMatrix
    stopping: StoppingSection
    compute: ComputeSection

    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def pending_fields(self) -> list[str]:
        """Dotted paths of every field still carrying the PENDING sentinel."""

        def walk(value: Any, path: str) -> list[str]:
            if isinstance(value, str):
                return [path] if value == PENDING else []
            if isinstance(value, dict):
                return [p for k, v in value.items() for p in walk(v, f"{path}.{k}")]
            if isinstance(value, list):
                return [p for i, v in enumerate(value) for p in walk(v, f"{path}[{i}]")]
            return []

        return walk(self.model_dump(mode="json"), "preregistration")

    @property
    def is_frozen_candidate(self) -> bool:
        return not self.pending_fields()

    def assert_result_run_allowed(self, approvals: list[ApprovalRecord]) -> None:
        """Fail-closed gate for any result-producing run: fully pinned + H2."""
        pending = self.pending_fields()
        if pending:
            raise ApprovalMissingError(
                f"preregistration has pending fields, cannot run: {pending}"
            )
        require_approval(HumanGate.H2_PREREGISTRATION, self.content_hash(), approvals)


class PilotConfig(BaseModel):
    """configs/research/*.yaml: a preregistration plus runtime locations."""

    model_config = ConfigDict(extra="forbid")

    preregistration: Preregistration
    output_root: str = "artifacts/research"
    data_registry_root: str = "data_registry/datasets"
    approvals_file: str = "docs/research/approvals.jsonl"


def load_pilot_config(path: Path) -> PilotConfig:
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"pilot config {path} must contain a YAML mapping")
    return PilotConfig.model_validate(raw)
