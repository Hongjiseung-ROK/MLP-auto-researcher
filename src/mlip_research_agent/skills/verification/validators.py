"""Validation for the claim verification skill."""

from __future__ import annotations

import json
from pathlib import Path

from mlip_research_agent.schemas.claims import Claim
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError


def load_claims(run_dir: Path, relative_path: str) -> list[Claim]:
    path = run_dir / relative_path
    if not path.is_file():
        raise SkillError(
            f"claims file not found: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise SkillError(
            f"claims file is not a JSON list: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return [Claim.model_validate(item) for item in raw]
