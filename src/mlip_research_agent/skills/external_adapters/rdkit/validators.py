"""Validation for the RDKit conformer adapter."""

from __future__ import annotations

import math

from mlip_research_agent.skills.external_adapters.common import adapter_error


def finite_conformers(conformers: list[list[list[float]]]) -> None:
    for index, coordinates in enumerate(conformers):
        values = [value for row in coordinates for value in row]
        if not all(math.isfinite(value) for value in values):
            raise adapter_error(f"conformer {index} contains non-finite coordinates")


__all__ = ["finite_conformers"]
