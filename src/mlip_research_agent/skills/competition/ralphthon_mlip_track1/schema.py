"""Typed General Track 1 routing contracts for bounded MLIP research."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RalphthonMLIPTrack1Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_domain: Literal["mlip_cu_mace"] = "mlip_cu_mace"
    competition_path: Literal["general_track_1"] = "general_track_1"
    metrics: list[str] = Field(min_length=1)
    use_vessl_compute: bool = False
    wandb_mode: Literal["disabled", "offline"] = "disabled"
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False
    h2_open: Literal[True] = True
    h3_open: Literal[True] = True
    h5_open: Literal[True] = True

    @model_validator(mode="after")
    def _separate_benchmarks(self) -> RalphthonMLIPTrack1Input:
        lowered = {metric.casefold() for metric in self.metrics}
        if "val_bpb" in lowered:
            raise ValueError("Karpathy val_bpb cannot enter the MLIP General Track 1 route")
        return self


class RalphthonMLIPTrack1Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: list[str]
    required_official_skills: list[str]
    evidence_metrics: list[str]
    paper_page_range: tuple[int, int]
    self_review_required: bool
    track1_contract_artifact: str
    scientific_status: Literal["infrastructure_only"] = "infrastructure_only"
    claim_eligible: Literal[False] = False
    karpathy_training_used: Literal[False] = False
    wandb_online_enabled: Literal[False] = False
