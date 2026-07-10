"""Validation for the mock DFT labeling skill."""

from __future__ import annotations

import math

from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError
from mlip_research_agent.skills.dft.schema import LabelingInput, LabelRecord

SUPPORTED_METHODS = ("mock_lj",)


def validate_inputs(inputs: LabelingInput) -> None:
    if inputs.method not in SUPPORTED_METHODS:
        raise SkillError(
            f"unsupported labeling method {inputs.method!r}; supported: {SUPPORTED_METHODS}. "
            "Real DFT backends are gated on approval (docs/OPEN_QUESTIONS.md).",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )


def validate_labels(labels: list[LabelRecord]) -> None:
    bad = [
        rec.index
        for rec in labels
        if not (math.isfinite(rec.energy) and math.isfinite(rec.fmax))
    ]
    if bad:
        raise SkillError(
            f"non-finite energies/forces for structures {bad}",
            failure_class=FailureClass.CONVERGENCE_FAILURE,
            severity=Severity.HIGH,
            retryable=True,
            likely_causes=["overlapping atoms", "diverging mock potential"],
        )
