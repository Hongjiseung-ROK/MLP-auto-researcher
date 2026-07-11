"""Validation helpers shared by external adapters."""

from __future__ import annotations

import math
from typing import Any

from mlip_research_agent.skills.external_adapters.common import adapter_error


def finite_structure_records(records: list[Any]) -> None:
    for record in records:
        values = [
            value
            for row in (*record.positions, *record.cell)
            for value in row
        ]
        if not all(math.isfinite(value) for value in values):
            raise adapter_error(f"structure record {record.index} contains non-finite values")


def structure_index(records: list[Any], index: int) -> Any:
    if index < 0 or index >= len(records):
        raise adapter_error(
            f"structure_index {index} out of range for {len(records)} structures"
        )
    return records[index]


def require_species_coverage(symbols: set[str], provided: set[str], what: str) -> None:
    missing = sorted(symbols - provided)
    if missing:
        raise adapter_error(f"{what} missing for species: {missing}")
