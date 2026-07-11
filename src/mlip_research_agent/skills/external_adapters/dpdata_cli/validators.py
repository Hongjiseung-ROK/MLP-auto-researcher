"""Validation for the dpdata dataset-conversion adapter."""

from __future__ import annotations

import math
from typing import Any

from mlip_research_agent.skills.external_adapters.common import adapter_error


def finite_labeled_configurations(configurations: list[Any]) -> None:
    for configuration in configurations:
        values = [
            configuration.energy_ev,
            *(value for force in configuration.forces_ev_per_a for value in force),
        ]
        if not all(math.isfinite(value) for value in values):
            raise adapter_error(
                f"configuration {configuration.config_id} contains non-finite labels"
            )


__all__ = ["finite_labeled_configurations"]
