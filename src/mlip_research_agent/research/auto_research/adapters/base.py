"""Benchmark-neutral execution boundary for Auto Research."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from mlip_research_agent.research.auto_research.validators import ConfigValue


class AdapterRunResult(BaseModel):
    """Paths and resource evidence produced by one adapter execution.

    Paths are local implementation details consumed immediately by the
    controller. Persisted trace nodes carry registry-relative artifact ids.
    The legacy ``fixture_definition_path`` name remains as the generic
    execution-definition path until the execution schema is versioned.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    succeeded: bool
    fixture_definition_path: str
    predictions_path: str | None
    predictions_rerun_path: str | None
    failure_path: str | None
    failure_category: str | None
    repair_available: bool
    simulated_wall_seconds: float = Field(ge=0)
    simulated_peak_memory_mb: float = Field(ge=0)
    split_path: str | None = None
    model_path: str | None = None
    compute_attestation: str = "local-cpu"
    device: str = Field(default="cpu", pattern=r"^(cpu|gpu)$")
    scientific_status: str = "non_scientific"
    claim_eligible: bool = False
    operation_receipt_path: str | None = None
    additional_artifact_paths: list[str] = Field(default_factory=list)


class BenchmarkAdapter(Protocol):
    """Typed execution boundary; the controller knows no benchmark details."""

    @property
    def name(self) -> str: ...

    @property
    def remote(self) -> bool: ...

    def execute(
        self, config: dict[str, ConfigValue], seed: int, workdir: Path
    ) -> AdapterRunResult: ...
