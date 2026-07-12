"""Fail-closed validation for the Ralphthon MLIP route."""

from __future__ import annotations

from mlip_research_agent.skills.competition.ralphthon_mlip_track1.schema import (
    RalphthonMLIPTrack1Input,
)

OFFICIAL_SKILL_DEPENDENCIES = ("auto-research", "vessl-cloud-onboarding")
FORBIDDEN_MLIP_METRICS = {"val_bpb"}
MLIP_METRIC_MARKERS = ("energy", "force", "stress", "virial")


def validate_mlip_route(params: RalphthonMLIPTrack1Input) -> None:
    lowered = {metric.casefold() for metric in params.metrics}
    overlap = lowered & FORBIDDEN_MLIP_METRICS
    if overlap:
        raise ValueError(f"Karpathy metrics forbidden in MLIP route: {sorted(overlap)}")
    if not all(metric.strip() for metric in params.metrics):
        raise ValueError("MLIP metric names must be non-empty")


def validate_karpathy_training_metrics(metrics: list[str]) -> None:
    lowered = [metric.casefold() for metric in metrics]
    if lowered != ["val_bpb"]:
        raise ValueError("Karpathy Training evidence must contain only val_bpb")
    if any(marker in metric for marker in MLIP_METRIC_MARKERS for metric in lowered):
        raise ValueError("MLIP metrics cannot enter the Karpathy Training route")
