"""Fail-closed validation for the Ralphthon MLIP route."""

from __future__ import annotations

from mlip_research_agent.skills.competition.ralphthon_mlip_track1.schema import (
    RalphthonMLIPTrack1Input,
)

OFFICIAL_SKILL_DEPENDENCIES = ("auto-research", "vessl-cloud-onboarding")
FORBIDDEN_MLIP_METRICS = {"val_bpb"}


def validate_mlip_route(params: RalphthonMLIPTrack1Input) -> None:
    lowered = {metric.casefold() for metric in params.metrics}
    overlap = lowered & FORBIDDEN_MLIP_METRICS
    if overlap:
        raise ValueError(f"Karpathy metrics forbidden in MLIP route: {sorted(overlap)}")
    if not all(metric.strip() for metric in params.metrics):
        raise ValueError("MLIP metric names must be non-empty")
