"""ResearchObjective: the top lineage node every experiment descends from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.acceptance_policy import AcceptanceConstraints
from mlip_research_agent.research.auto_research.mutation import MutationClass
from mlip_research_agent.research.auto_research.validators import (
    ConfigValue,
    ContentAddressedModel,
)
from mlip_research_agent.research.preregistration import HumanGate
from mlip_research_agent.schemas.claims import ScientificEvidenceTier


class SuccessCriterion(BaseModel):
    """One machine-checkable success condition for the objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    metric: str = Field(min_length=1)
    direction: str = Field(pattern=r"^(minimize|maximize)$")
    description: str = Field(min_length=5, max_length=500)


class ObjectiveBudget(BaseModel):
    """Hard resource ceilings for the whole objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_compute_seconds: float = Field(gt=0)
    max_retries_per_iteration: int = Field(ge=0, le=10)
    max_labels: int = Field(ge=0, description="0 when the objective reveals no oracle labels")
    max_remote_jobs: int = Field(ge=0, description="0 keeps the objective strictly local")


class ProtectedConstant(BaseModel):
    """A value the Auto Research loop must never change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    value: ConfigValue
    reason: str = Field(min_length=5, max_length=500)


class StoppingRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    description: str = Field(min_length=5, max_length=500)


class ResearchObjective(ContentAddressedModel):
    """What the researcher is trying to learn, under which hard bounds."""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    title: str = Field(min_length=5, max_length=200)
    research_question: str = Field(min_length=10, max_length=2000)
    domain: str = Field(min_length=2, max_length=100)
    benchmark_adapter: str = Field(
        min_length=1, description="Registered adapter name; the controller has no Cu branches"
    )
    success_criteria: list[SuccessCriterion] = Field(min_length=1, max_length=10)
    budget: ObjectiveBudget
    maximum_iterations: int = Field(gt=0, le=100)
    protected_constants: list[ProtectedConstant] = Field(default_factory=list, max_length=50)
    allowed_mutation_classes: list[MutationClass] = Field(min_length=1)
    required_evaluator: str = Field(
        min_length=1, description="Evaluator identity the acceptance decision must come from"
    )
    stopping_rules: list[StoppingRule] = Field(min_length=1, max_length=10)
    approval_requirements: list[HumanGate] = Field(default_factory=list)
    scientific_status_ceiling: ScientificEvidenceTier
    created_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bounded(self) -> ResearchObjective:
        forbidden = {MutationClass.IMMUTABLE, MutationClass.OWNER_GATED}
        bad = [c for c in self.allowed_mutation_classes if c in forbidden]
        if bad:
            raise ValueError(
                "immutable/owner_gated can never be granted as allowed mutation classes"
            )
        if self.scientific_status_ceiling in {
            ScientificEvidenceTier.REPLICATED,
            ScientificEvidenceTier.PUBLICATION_ELIGIBLE,
        }:
            raise ValueError(
                "an Auto Research objective cannot self-authorize replicated or "
                "publication-grade evidence"
            )
        return self


class LocalDemoConfig(BaseModel):
    """configs/research/ralphthon_local_demo.yaml: objective + fixture wiring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    objective: ResearchObjective
    mutation_policy_path: str = Field(min_length=1)
    base_config: dict[str, ConfigValue]
    fixture_seed: int = Field(ge=0)
    output_root: str = "artifacts/auto_research"
    acceptance: AcceptanceConstraints

    @classmethod
    def load(cls, path: Path) -> LocalDemoConfig:
        raw: Any = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"local demo config {path} must be a YAML mapping")
        return cls.model_validate(raw)
