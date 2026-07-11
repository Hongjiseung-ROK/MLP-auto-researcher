"""Typed contract for an auditable hidden-label reveal."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OracleRevealInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(description="Run-dir-relative qualified normalized dataset")
    qualified_manifest_path: str = Field(
        description="Run-dir-relative qualified normalized manifest"
    )
    split_manifest_path: str = Field(description="Run-dir-relative frozen split manifest")
    expected_split_manifest_sha256: str = Field(min_length=64, max_length=64)
    candidate_ids: list[str] = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)
    total_budget: int = Field(ge=0)
    round_budgets: dict[str, int] = Field(default_factory=dict)
    previous_state_path: str | None = None


class OracleRevealOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_artifact: str
    request_path: str
    decision_artifact: str
    decision_path: str
    label_batch_artifact: str
    label_batch_path: str
    budget_ledger_artifact: str
    budget_ledger_path: str
    training_lineage_artifact: str
    training_lineage_path: str
    oracle_state_artifact: str
    oracle_state_path: str
    n_selected: int = Field(gt=0)
    n_newly_charged: int = Field(ge=0)
    remaining_budget: int = Field(ge=0)
    idempotent_replay: bool
