"""Machine-readable guardrail separating program goals from benchmark scaffolds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.preregistration import HumanGate
from mlip_research_agent.schemas.claims import ScientificEvidenceTier


class ResearchProgramGuardrail(BaseModel):
    """Bound the current benchmark without narrowing the long-term program."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    current_program_goal: str = Field(min_length=10, max_length=2000)
    current_benchmark_role: str = Field(min_length=10, max_length=1000)
    non_goals: list[str] = Field(min_length=1, max_length=20)
    allowed_claim_tier: ScientificEvidenceTier
    next_human_gate: HumanGate
    criteria_to_exit_benchmark_mode: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _no_evidence_inflation(self) -> ResearchProgramGuardrail:
        if self.allowed_claim_tier in {
            ScientificEvidenceTier.REPLICATED,
            ScientificEvidenceTier.PUBLICATION_ELIGIBLE,
        }:
            raise ValueError(
                "benchmark scaffolds cannot authorize replicated or publication evidence"
            )
        return self

    def content_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> ResearchProgramGuardrail:
        raw: Any = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"research program guardrail {path} must be a YAML mapping")
        return cls.model_validate(raw)

    def assert_claim_tier_allowed(self, tier: ScientificEvidenceTier) -> None:
        order = {
            ScientificEvidenceTier.NON_SCIENTIFIC: 0,
            ScientificEvidenceTier.PILOT_ONLY: 1,
            ScientificEvidenceTier.REPLICATED: 2,
            ScientificEvidenceTier.PUBLICATION_ELIGIBLE: 3,
        }
        if order[tier] > order[self.allowed_claim_tier]:
            raise ValueError(
                f"claim tier {tier.value} exceeds program limit "
                f"{self.allowed_claim_tier.value}"
            )
