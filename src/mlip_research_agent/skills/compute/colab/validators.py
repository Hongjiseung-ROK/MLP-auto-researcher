"""Validation for the colab_preflight skill."""

from __future__ import annotations

from pathlib import Path

from mlip_research_agent.compute.schemas import AcceleratorId
from mlip_research_agent.schemas.failure import FailureClass, Severity
from mlip_research_agent.skills.base import SkillError
from mlip_research_agent.skills.compute.colab.schema import (
    VALIDATION_WORKLOADS,
    ColabPreflightInput,
)


def validate_inputs(inputs: ColabPreflightInput) -> None:
    if inputs.workload_class not in VALIDATION_WORKLOADS:
        allowed = [w.value for w in VALIDATION_WORKLOADS]
        raise SkillError(
            f"colab_preflight only runs validation workloads {allowed}; "
            f"got {inputs.workload_class.value!r}. Full campaigns belong on VESSL.",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    if inputs.requested_accelerator is AcceleratorId.CPU:
        raise SkillError(
            "colab_preflight validates GPU runtimes; request nvidia-l4 or an A100 id "
            "(local CPU work belongs to the local provider)",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )


def validate_policy_file(policy_path: str) -> Path:
    path = Path(policy_path)
    if not path.is_file():
        raise SkillError(
            f"compute policy file not found: {policy_path}",
            failure_class=FailureClass.VALIDATION_ERROR,
            severity=Severity.HIGH,
            retryable=False,
        )
    return path
