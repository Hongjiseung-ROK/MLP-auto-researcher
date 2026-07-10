"""Validation for the mock acquisition skill."""

from __future__ import annotations

from pathlib import Path

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.atomistics.structures_io import StructureSet
from mlip_research_agent.skills.base import SkillError


def load_structures(run_dir: Path, relative_path: str) -> StructureSet:
    path = run_dir / relative_path
    if not path.is_file():
        raise SkillError(
            f"structure set not found: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return StructureSet.load(path)


def validate_selection_size(n_select: int, n_structures: int) -> None:
    if n_select > n_structures:
        raise SkillError(
            f"cannot select {n_select} from {n_structures} candidates",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
