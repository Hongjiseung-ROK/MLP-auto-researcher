"""Validation for the mock MLIP skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError

SUPPORTED_MODELS = ("mock_mean_baseline",)
MIN_LABELS_FOR_SPLIT = 4


def load_json_artifact(run_dir: Path, relative_path: str, description: str) -> dict[str, Any]:
    path = run_dir / relative_path
    if not path.is_file():
        raise SkillError(
            f"{description} not found: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    loaded: Any = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SkillError(
            f"{description} is not a JSON object: {relative_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return loaded


def validate_model_name(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise SkillError(
            f"unsupported model {model_name!r}; supported: {SUPPORTED_MODELS}. "
            "Real MLIP backends are gated on approval (docs/OPEN_QUESTIONS.md).",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )


def validate_label_count(n_labels: int) -> None:
    if n_labels < MIN_LABELS_FOR_SPLIT:
        raise SkillError(
            f"need at least {MIN_LABELS_FOR_SPLIT} labels for a train/holdout split, "
            f"got {n_labels}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
