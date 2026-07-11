"""Validation helpers for real MACE inference."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.atomistics.structures_io import StructureSet
from mlip_research_agent.skills.base import SkillError


def run_relative_file(run_dir: Path, relative_path: str, description: str) -> Path:
    candidate = (run_dir / relative_path).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise SkillError(
            f"{description} escapes the run directory: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        ) from exc
    if not candidate.is_file():
        raise SkillError(
            f"{description} not found: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return candidate


def local_file(path_text: str, description: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise SkillError(
            f"{description} not found: {path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return path


def require_mace_installation() -> None:
    if importlib.util.find_spec("mace") is None:
        raise SkillError(
            "mace-torch is not installed; install the pinned 'mace' optional extra",
            failure_class=FailureClass.TOOL_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )


def validate_structures(structures: StructureSet, supported_species: set[str]) -> None:
    if not structures.systems:
        raise SkillError(
            "MACE inference requires at least one structure",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    unsupported = sorted(
        {symbol for record in structures.systems for symbol in record.symbols}
        - supported_species
    )
    if unsupported:
        raise SkillError(
            f"checkpoint manifest does not declare support for species: {unsupported}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )


def require_finite(values: list[float], description: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise SkillError(
            f"MACE produced non-finite {description}",
            failure_class=FailureClass.SIMULATION_INSTABILITY,
            severity=Severity.CRITICAL,
            retryable=False,
        )
