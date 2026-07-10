"""Research campaign specification: the validated entry point of every run."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_BACKENDS = ("mock",)


class CampaignMode(StrEnum):
    """Research mode. plan.md: AL must be conditional, not dogmatic."""

    FINE_TUNE_ONLY = "fine_tune_only"
    ACTIVE_LEARNING = "active_learning"
    COMMITTEE_SCREENING = "committee_screening"


class TargetSystem(BaseModel):
    """Bounded description of the material system under study."""

    model_config = ConfigDict(extra="forbid")

    formula: str = Field(min_length=1, description="Element or compound, e.g. 'Cu'")
    crystal_structure: str = Field(description="ASE bulk lattice keyword, e.g. 'fcc'")
    lattice_constant: float = Field(gt=0.0, description="Lattice constant a in Angstrom")
    supercell: tuple[int, int, int] = Field(default=(2, 2, 2))

    @field_validator("supercell")
    @classmethod
    def _positive_supercell(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(n < 1 for n in v):
            raise ValueError("supercell repetitions must be >= 1")
        return v


class LabelBudget(BaseModel):
    """Hard resource bounds. No skill may exceed these silently."""

    model_config = ConfigDict(extra="forbid")

    max_labels: int = Field(gt=0, le=10_000)
    max_retries_per_step: int = Field(default=2, ge=0, le=5)


class StoppingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=1, gt=0, le=100)
    target_energy_mae: float | None = Field(default=None, gt=0.0)


class FailureInjection(BaseModel):
    """Deterministic recoverable-failure injection for demos and tests."""

    model_config = ConfigDict(extra="forbid")

    step: str = Field(description="Workflow step id in which to inject the failure")
    times: int = Field(default=1, ge=1, le=3, description="Consecutive attempts that fail")


class CampaignSpec(BaseModel):
    """A small YAML research campaign, validated before any execution."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    description: str = ""
    mode: CampaignMode
    backend: str = Field(default="mock")
    target_system: TargetSystem
    budget: LabelBudget
    stopping: StoppingRule = StoppingRule()
    seed: int = Field(default=0, ge=0)
    n_candidates: int = Field(default=16, gt=1, le=10_000)
    failure_injection: FailureInjection | None = None

    @field_validator("backend")
    @classmethod
    def _supported_backend(cls, v: str) -> str:
        if v not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend {v!r} not supported yet; supported: {SUPPORTED_BACKENDS}. "
                "Real backends are gated on approval (docs/OPEN_QUESTIONS.md)."
            )
        return v

    def content_hash(self) -> str:
        """Deterministic hash of the campaign for provenance and checkpoints."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_campaign(path: Path) -> CampaignSpec:
    """Load and validate a campaign YAML file."""
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"campaign file {path} must contain a YAML mapping")
    return CampaignSpec.model_validate(raw)
